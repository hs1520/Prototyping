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
            captured["system"] = system_prompt or ""
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


def test_prompt_states_the_notation_conventions_but_not_the_reviewed_answers():
    """The checker's conventions are properties of the notation, so stating them
    is fair (the deterministic emitter has them by construction). The *safety
    facts* — which pattern applies, the timing origin, the deadline, the winning
    priority member — stay the LLM's to derive, or the accuracy signal is void."""
    captured = {}

    class _Capture:
        def chat(self, prompt, system_prompt=None):
            captured["both"] = f"{system_prompt or ''}\n{prompt}"
            return emit_ag_package(REQ_SAFE_005_CHAIN)

    orch = Orchestrator(_Capture(), revised_experiment_arm="R2-BBAG",
                        r2_generation_mode="LLM_AUTHORED_AG")
    orch._generate_llm_authored_ag_package(REQ_SAFE_005_CHAIN, _BASE)
    text = captured["both"]

    # conventions the checker enforces must be stated, or a defect is unfixable
    # the imports that make DurationValue/Boolean resolve — stating the types
    # without their imports cost every v2 seed its first iteration to a
    # "No Type named 'DurationValue' found" syntax rejection
    for statement in ("private import ScalarValues::*;",
                      "private import ISQ::*;", "private import SI::*;"):
        assert statement in text
    assert "verification def" in text          # OBSERVATION_MISSING
    assert "realize" in text                   # REALIZATION_MISSING
    assert "safety_pattern=" in text           # PATTERN_DECLARATION_INCONSISTENT
    assert "timing_origin=" in text

    # the full pattern vocabulary is offered, so classifying is a real derivation
    for pattern in ("TRIGGERED_TIMED_FAILSAFE_RESPONSE", "STARTUP_INHIBIT",
                    "LOCKED_UNTIL_AUTHORISED_RELEASE"):
        assert pattern in text
    # ...and the answer is never asserted for this requirement
    assert "safety_pattern=TRIGGERED_TIMED_FAILSAFE_RESPONSE" not in text
    assert f"timing_origin={REQ_SAFE_005_CHAIN.timing_origin}" not in text

    # Reviewed safety facts stay withheld. The requirement text itself is the
    # legitimate input (it is what the deadline must be *read from*), so the leak
    # check is on everything the prompt adds around it.
    assert "0.5" in text, "the requirement text must still carry its own deadline"
    requirement_text = _BASE.split("doc /*")[1].split("*/")[0]
    added = text.replace(requirement_text, "")
    assert requirement_text not in added, "requirement text must be stripped once"
    # the deadline VALUE is never stated outside the requirement — only the
    # attribute name that must carry it, and an instruction to derive it
    assert str(REQ_SAFE_005_CHAIN.deadline) not in added
    assert "maxLatency" in text                # the convention name is fair game
    assert REQ_SAFE_005_CHAIN.priority.response_set_id not in added
    for member in REQ_SAFE_005_CHAIN.priority.members:
        assert member not in added


class _SequenceLLM:
    """Returns a fixed sequence of packages across successive chat() calls."""

    def __init__(self, packages):
        self._packages = list(packages)
        self._calls = 0

    def chat(self, prompt, system_prompt=None):
        self.last_prompt = prompt
        pkg = self._packages[min(self._calls, len(self._packages) - 1)]
        self._calls += 1
        return pkg


def test_feedback_loop_converges_to_pass_under_the_ag_check():
    """The A/G check gates each round; a broken first attempt is regenerated on
    the checker's diagnostics until the merged model is verified PASS."""
    correct = emit_ag_package(REQ_SAFE_005_CHAIN)
    broken = "\n".join(
        line for line in correct.splitlines()
        if "dependency discharge" not in line.lower()
    )
    llm = _SequenceLLM([broken, correct])
    orch = Orchestrator(llm, revised_experiment_arm="R2-BBAG",
                        r2_generation_mode="LLM_AUTHORED_AG")
    result = orch._author_llm_ag_with_feedback(
        REQ_SAFE_005_CHAIN, _BASE, max_iterations=4)

    history = result["history"]
    assert history[0]["verdict"] != "PASS"      # undischarged -> not yet robust
    assert history[-1]["verdict"] == "PASS"      # converged to a verified model
    assert history[-1]["error_count"] == 0
    assert len(history) == 2                      # fixed on the first feedback round
    # the second call carried the checker's diagnostics + the previous attempt
    assert "PREVIOUS attempt" in llm.last_prompt
    assert "DISCHARGE" in llm.last_prompt.upper()  # the checker's actual defect code


def test_feedback_loop_reports_the_verdict_of_the_package_it_returns():
    """A later round can regress — even to invalid syntax. The reported verdict
    must describe the package actually delivered, not the last attempt made."""
    correct = emit_ag_package(REQ_SAFE_005_CHAIN)
    broken = "\n".join(
        line for line in correct.splitlines()
        if "dependency discharge" not in line.lower()
    )
    # scores 2 errors, then regresses to unparseable output on every later round
    orch = Orchestrator(_SequenceLLM([broken, "not sysml at all"]),
                        revised_experiment_arm="R2-BBAG",
                        r2_generation_mode="LLM_AUTHORED_AG")
    result = orch._author_llm_ag_with_feedback(
        REQ_SAFE_005_CHAIN, _BASE, max_iterations=3)

    # the last attempt never even parsed, so its history verdict is absent...
    assert result["history"][-1]["syntax_ok"] is False
    assert result["history"][-1]["verdict"] is None
    # ...but the delivered package is the scoreable first one, reported as such
    assert result["final_verdict"] == "FAIL"
    assert result["final_error_count"] == result["history"][0]["error_count"]
    assert result["final_error_codes"] == result["history"][0]["error_codes"]
    assert "SystemParachuteContract" in result["final_package"]


def test_feedback_loop_returns_the_lowest_error_attempt_within_budget():
    correct = emit_ag_package(REQ_SAFE_005_CHAIN)
    broken = "\n".join(
        line for line in correct.splitlines()
        if "dependency discharge" not in line.lower()
    )
    # never converges (always broken): the loop must still return a usable best
    orch = Orchestrator(_SequenceLLM([broken]), revised_experiment_arm="R2-BBAG",
                        r2_generation_mode="LLM_AUTHORED_AG")
    result = orch._author_llm_ag_with_feedback(
        REQ_SAFE_005_CHAIN, _BASE, max_iterations=3)
    assert len(result["history"]) == 3           # exhausted the budget
    assert all(h["verdict"] != "PASS" for h in result["history"])
    assert "SystemParachuteContract" in result["final_package"]
