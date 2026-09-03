"""The mandatory catalog seed names its failure reason and stops an authoritative run.

On the 2026-08-30 attempt the seed's one-line "could not form" message
conflated four bail points (identifying the real one, the surgery's zero-error
post-check rejecting a parse-broken input, took a manual replay), and the run
then ran the whole downstream phase sequence before the finalizer refused
publication for the missing Phase 8 recommendation. Under
PROTOTYPING_AUTHORITATIVE=1 it now stops at the seed.
"""
from __future__ import annotations

import pytest

from src.agents.orchestrator import Orchestrator
from src.dse import variation_introducer as vi
from src.dse.variation_introducer import VariantSpec, introduce_variation
from src.sysml.lite_model import build_lite_model

_QUANTIFIED_REQS = [
    "REQ-PERF-002: The system shall sustain flight for a minimum of 25 "
    "minutes when carrying the maximum rated payload at nominal cruise speed.",
]

_HOSTLESS_MODEL = """package M {
    part def Controller {
        attribute speed : Real = 1.0;
    }
    part controller : Controller;
}"""

_VARIANTS = [
    VariantSpec(name="a", type_name="AImpl", attrs="attribute x : Real = 1.0;"),
    VariantSpec(name="b", type_name="BImpl", attrs="attribute x : Real = 2.0;"),
]


class _NoCallLLM:
    def complete(self, *_args, **_kwargs):  # pragma: no cover
        raise AssertionError("fail-fast must trigger before any LLM call")


def test_variation_names_bail_points():
    ok_model = "package M {\n    part def P;\n    part host : P;\n}\n"

    _, ok = introduce_variation(ok_model, "host", "P", _VARIANTS[:1], "r", [])
    assert not ok and "at least 2" in vi.LAST_FAILURE_REASON

    _, ok = introduce_variation(ok_model, "missing", "P", _VARIANTS, "r", [])
    assert not ok and "not found" in vi.LAST_FAILURE_REASON

    broken = ok_model + "\nstray }"
    _, ok = introduce_variation(broken, "host", "P", _VARIANTS, "r", [])
    assert not ok
    assert "post-surgery text fails to parse" in vi.LAST_FAILURE_REASON
    assert "error-free" in vi.LAST_FAILURE_REASON


def test_variation_clears_reason():
    model = "package M {\n    part def P;\n    part host : P;\n}\n"
    _, ok = introduce_variation(model, "host", "P", _VARIANTS, "r", ["REQ-1"])
    assert ok
    assert vi.LAST_FAILURE_REASON == ""


def _orchestrator() -> Orchestrator:
    return Orchestrator(_NoCallLLM(), max_iterations=1, quality_threshold=0.5)


def test_authoritative_fails_fast_no_seed(monkeypatch):
    monkeypatch.setenv("PROTOTYPING_AUTHORITATIVE", "1")
    orch = _orchestrator()
    model = build_lite_model(_HOSTLESS_MODEL, model_name="M")

    with pytest.raises(RuntimeError) as err:
        orch._introduce_variations(model, list(_QUANTIFIED_REQS))

    message = str(err.value)
    assert "authoritative fail-fast" in message
    assert "Phase 8" in message
    assert orch.last_variation_proposal_source == "catalog-seed-unavailable"


def test_non_authoritative_limps_along(monkeypatch):
    monkeypatch.delenv("PROTOTYPING_AUTHORITATIVE", raising=False)
    orch = _orchestrator()
    monkeypatch.setattr(
        Orchestrator, "_propose_variants",
        lambda self, usage, type_name, requirements: None,
    )
    model = build_lite_model(_HOSTLESS_MODEL, model_name="M")

    returned = orch._introduce_variations(model, list(_QUANTIFIED_REQS))

    assert returned is model
    assert orch.last_variation_proposal_source == "catalog-seed-unavailable"


def test_unquantified_no_fail_fast(monkeypatch):
    monkeypatch.setenv("PROTOTYPING_AUTHORITATIVE", "1")
    orch = _orchestrator()
    monkeypatch.setattr(
        Orchestrator, "_propose_variants",
        lambda self, usage, type_name, requirements: None,
    )
    model = build_lite_model(_HOSTLESS_MODEL, model_name="M")

    returned = orch._introduce_variations(
        model, ["REQ-CONS-004: The system shall comply with EASA rules."]
    )

    assert returned is model
    assert orch.last_variation_proposal_source == "catalog-seed-not-required"


def test_rank_key_prefers_repaired():
    """The 2026-08-30 shape: four iterations at 0.700, only the first still carrying
    the parser error.

    Strict score-only `>` kept the broken first model; the rank key does not.
    """
    from types import SimpleNamespace

    from src.agents.refinement import RefinementClosure
    from src.agents.refinement_intelligence import ScriptedRefinementIntelligence
    from src.simulation.validator import SimulationResult

    orch = _orchestrator()
    closure = RefinementClosure(
        orch,
        intelligence=ScriptedRefinementIntelligence(),
        simulation_runner=lambda _t, name: SimulationResult(model_name=name),
    )
    engine = closure._RefinementClosure__implementation
    key = engine._model_rank_key

    broken = SimpleNamespace(total_errors=lambda: 1)
    clean = SimpleNamespace(total_errors=lambda: 0)

    first_broken = key(0.700, broken, 0)
    later_clean = key(0.700, clean, 1)
    assert later_clean > first_broken
    assert key(0.700, clean, 3) > key(0.700, clean, 1)
    assert key(0.800, broken, 0) > later_clean
    assert key(0.700, None, 2) == (0.700, 0, 2)
