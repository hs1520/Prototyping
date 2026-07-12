"""T13 guard: deterministic ontology fallback when the LLM proposes no variants.

Two consecutive real runs (2026-07-08) had the proposal LLM judge every component
"not relevant" → zero variation points → the whole objective-DSE + realization
path silently fell back to catalog bilevel. These tests pin the fallback:
quantified emergent targets + a concern-matching connected component must always
yield an admissible variation space, with the source recorded (never hidden).
"""
from __future__ import annotations

from types import SimpleNamespace

from src.agents.orchestrator import Orchestrator
from src.dse.variation_dse import run_variation_dse
from src.dse.variation_parser import admitted, parse_variation_points

_MODEL_TEXT = """package Drone {
    port def Sig;
    part def GroundStation { out port cmd : Sig; }
    part def PowerSystem { in port cmd : Sig; }
    part def Airframe { in port cmd : Sig; }
    part gcs : GroundStation;
    part powerSystem : PowerSystem;
    part airframe : Airframe;
    connect gcs.cmd to powerSystem.cmd;
    connect gcs.cmd to airframe.cmd;
}"""

_REQS = ["REQ-PERF-002: flight endurance of at least 20 minutes."]


def _model(text: str = _MODEL_TEXT) -> SimpleNamespace:
    return SimpleNamespace(metadata={"last_sysml_text": text})


def _decline_all(self, usage, type_name, requirements):  # noqa: ARG001
    return None


def test_fallback_injects_admissible_variation_when_llm_declines(monkeypatch):
    monkeypatch.setattr(Orchestrator, "_propose_variants", _decline_all)
    orch = Orchestrator(llm=object(), use_variation_dse=True)
    model = _model()

    out = orch._introduce_variations(model, _REQS)

    text = out.metadata["last_sysml_text"]
    ok, bad = admitted(parse_variation_points(text))
    assert [p.point_id for p in ok] == ["airframe"]  # concern preference over powerSystem
    assert bad == []
    assert orch.last_variation_proposal_source == "fallback"
    # the fallback declares real design inputs spanning a rotor/mass trade
    assert "rotorCount" in text and "rotorRadiusM" in text and "batteryCells" in text
    assert "deterministic ontology fallback" in text  # rationale visible in the model
    # and the injected space actually drives the variation DSE end to end
    res = run_variation_dse(out, requirements=_REQS, iterations=15, random_seed=0)
    assert res is not None
    assert res.recommended_design is not None
    assert set(res.recommended_choices) == {"airframe"}


def test_fallback_skipped_without_quantified_emergent_targets(monkeypatch):
    monkeypatch.setattr(Orchestrator, "_propose_variants", _decline_all)
    orch = Orchestrator(llm=object(), use_variation_dse=True)
    model = _model()

    out = orch._introduce_variations(model, ["REQ-FUNC-001: navigate autonomously."])

    assert parse_variation_points(out.metadata["last_sysml_text"]) == []
    assert orch.last_variation_proposal_source == "fallback-not-required"


def test_fallback_skipped_when_no_component_matches_a_concern(monkeypatch):
    monkeypatch.setattr(Orchestrator, "_propose_variants", _decline_all)
    orch = Orchestrator(llm=object(), use_variation_dse=True)
    text = """package P {
        port def Sig;
        part def Radio { in port cmd : Sig; }
        part def Camera { out port cmd : Sig; }
        part radio : Radio;
        part camera : Camera;
        connect camera.cmd to radio.cmd;
    }"""
    out = orch._introduce_variations(_model(text), _REQS)

    assert parse_variation_points(out.metadata["last_sysml_text"]) == []
    assert orch.last_variation_proposal_source == "fallback-unavailable"


def test_llm_proposals_take_precedence_over_fallback(monkeypatch):
    from src.dse.variation_introducer import VariantSpec

    def propose(self, usage, type_name, requirements):  # noqa: ARG001
        if usage != "airframe":
            return None
        return (
            "llm rationale",
            ["REQ-PERF-002"],
            [
                VariantSpec("alpha", "AlphaAirframeImpl",
                            "attribute rotorCount : Real = 4.0;"),
                VariantSpec("beta", "BetaAirframeImpl",
                            "attribute rotorCount : Real = 6.0;"),
            ],
        )

    monkeypatch.setattr(Orchestrator, "_propose_variants", propose)
    orch = Orchestrator(llm=object(), use_variation_dse=True)

    out = orch._introduce_variations(_model(), _REQS)

    text = out.metadata["last_sysml_text"]
    assert orch.last_variation_proposal_source == "llm"
    assert "AlphaAirframeImpl" in text
    assert "Quad_fallbackAirframeImpl" not in text


def test_existing_variation_resets_stale_source_and_records_model_origin(monkeypatch):
    monkeypatch.setattr(Orchestrator, "_propose_variants", _decline_all)
    orch = Orchestrator(llm=object(), use_variation_dse=True)
    model = orch._introduce_variations(_model(), _REQS)
    assert orch.last_variation_proposal_source == "fallback"

    orch._introduce_variations(model, _REQS)

    assert orch.last_variation_proposal_source == "model-existing"


def test_bounds_that_leave_only_one_fallback_variant_are_reported_not_silenced(monkeypatch):
    import src.dse.domain_objective as objective

    monkeypatch.setattr(Orchestrator, "_propose_variants", _decline_all)
    monkeypatch.setattr(
        objective,
        "within_requirement_bounds",
        lambda design, req_ids, requirements: design["rotor_count"] == 4,
    )
    orch = Orchestrator(llm=object(), use_variation_dse=True)

    out = orch._introduce_variations(_model(), _REQS)

    assert parse_variation_points(out.metadata["last_sysml_text"]) == []
    assert orch.last_variation_proposal_source == "fallback-unavailable"
