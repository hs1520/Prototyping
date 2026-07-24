"""LLM-authored A/G R2 intervention (design §15, line 1089).

A second R2 generation mode: the LLM authors the bounded A/G decomposition instead
of the deterministic emitter. Because the prediction is then the LLM's own work
scored against frozen gold, agreement below 1.0 is a *real generation accuracy*
signal, not the deterministic round-trip fidelity. It is a separate frozen
configuration/version and must never pool with the deterministic run.

These tests use a stub LLM (no provider call): a correct authored package scores
perfect agreement, an imperfect one scores below 1.0 (the specific error is
caught), invalid output fails the run closed, and the deterministic mode is
unchanged and remains the default.
"""
from __future__ import annotations

import pytest

from src.agents.orchestrator import Orchestrator
from src.prototyping.ag_chains import REQ_SAFE_005_CHAIN
from src.prototyping.ag_contracts import check_ag_graph
from src.prototyping.ag_emitter import emit_ag_package
from src.prototyping.ag_evaluation import evaluate_ag_against_gold
from src.prototyping.ag_extractor import extract_ag_graph
from src.prototyping.experiment_arms import R2_DETERMINISTIC_GENERATION_MODE
from tests.test_option2_ag_evaluation import REQ_SAFE_005_GOLD

_BASE = (
    "package Src { requirement def REQ_SAFE_005 { doc /* deploy the ballistic "
    "recovery parachute within 0.5 s of a critical propulsion failure. */ } }"
)
_REQS = ["REQ-SAFE-005: deploy the parachute within 0.5 s"]


class _StubLLM:
    """Returns a fixed A/G package as the 'LLM-authored' output; never asserts."""

    def __init__(self, package_text: str):
        self._package = package_text

    def chat(self, prompt, system_prompt=None):  # noqa: D401 - stub
        return self._package


def _llm_orch(package_text: str) -> Orchestrator:
    return Orchestrator(
        _StubLLM(package_text),
        revised_experiment_arm="R2-BBAG",
        r2_generation_mode="LLM_AUTHORED_AG",
    )


def _prediction(package_text: str) -> dict:
    merged = _llm_orch(package_text)._apply_ag_contract_layer(_BASE, _REQS)
    return check_ag_graph(extract_ag_graph(merged)).to_dict()


def test_llm_authored_correct_package_scores_perfect_agreement():
    # the LLM authored exactly the reviewed decomposition
    out = evaluate_ag_against_gold(
        _prediction(emit_ag_package(REQ_SAFE_005_CHAIN)), REQ_SAFE_005_GOLD
    )
    assert out["guarantee_allocation"]["f1"] == 1.0
    assert out["assumption_discharge"]["f1"] == 1.0


def test_llm_authored_imperfect_package_yields_real_accuracy_below_one():
    # the LLM got the components/guarantees right but omitted the discharge edges
    correct = emit_ag_package(REQ_SAFE_005_CHAIN)
    imperfect = "\n".join(
        line for line in correct.splitlines()
        if "dependency discharge" not in line.lower()
    )
    out = evaluate_ag_against_gold(_prediction(imperfect), REQ_SAFE_005_GOLD)
    # this is the whole point: a wrong LLM decomposition scores a genuine
    # accuracy below 1.0, not round-trip fidelity
    assert out["assumption_discharge"]["f1"] < 1.0
    assert out["assumption_discharge"]["recall"] < 1.0


def test_llm_authored_invalid_output_fails_the_run_closed():
    with pytest.raises(RuntimeError, match="failed closed"):
        _llm_orch("this is not valid SysML at all")._apply_ag_contract_layer(
            _BASE, _REQS
        )


def test_deterministic_mode_is_unchanged_and_is_the_default():
    # default mode ignores the stub entirely and renders the reviewed spec
    orch = Orchestrator(_StubLLM("unused"), revised_experiment_arm="R2-BBAG")
    assert orch.r2_generation_mode == R2_DETERMINISTIC_GENERATION_MODE
    merged = orch._apply_ag_contract_layer(_BASE, _REQS)
    assert "SystemParachuteContract" in merged
    assert "unused" not in merged


def test_unknown_generation_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown r2_generation_mode"):
        Orchestrator(
            _StubLLM("x"),
            revised_experiment_arm="R2-BBAG",
            r2_generation_mode="BOGUS_MODE",
        )


def test_setup_c_prompt_gives_complete_interfaces_but_not_the_discharge_answer():
    """(C): the LLM is handed the complete architecture (all guarantees + the
    consumes/produces interfaces) but must still DERIVE the discharge wiring."""
    captured = {}

    class _Capture:
        def chat(self, prompt, system_prompt=None):
            captured["prompt"] = prompt
            return emit_ag_package(REQ_SAFE_005_CHAIN)

    orch = Orchestrator(_Capture(), revised_experiment_arm="R2-BBAG",
                        r2_generation_mode="LLM_AUTHORED_AG")
    orch._generate_llm_authored_ag_package(REQ_SAFE_005_CHAIN, _BASE)
    prompt = captured["prompt"]
    # complete architecture: interfaces + the secondary guarantee are now given
    assert "consumes" in prompt and "produces" in prompt
    assert "parachuteResponseSelected" in prompt
    assert "airborne" in prompt
    # but the discharge relationships are the LLM's to derive — not handed over
    assert "dischargeCommand" not in prompt
    assert "discharge from" not in prompt.lower()
