"""Second board-mediated handoff: DesignAgent -> VerificationAgent (design §15).

The VerificationAgent knowledge source consumes the committed model (the relevant
requirement defs) and the authoritative requirements from the Blackboard and
publishes a typed per-requirement verification plan. Its purpose is to give the
§13 coordination metrics a handoff/role denominator greater than one — two
migrated handoffs instead of the illustrative single one — while every coordination
invariant (revision-pinned, gold-free, one role per session) still holds.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.agents.orchestrator import Orchestrator
from src.prototyping.run_metrics import compute_coordination_metrics
from src.prototyping.verification_planning import (
    plan_verification,
    requirement_def_slice,
)
from src.sysml.lite_model import build_lite_model
from src.utils.sysml_text_utils import get_sysml_text


class _NoCallLLM:
    def complete(self, *_a, **_k):  # pragma: no cover
        raise AssertionError("verification planning must not call the LLM")


_MODEL = (
    "package DeliveryUAV { "
    "requirement def REQ_SAFE_004 { doc /* do not arm if a sensor fails the "
    "power-on self-test. */ } "
    "requirement def REQ_SAFE_005 { doc /* deploy the parachute within 0.5 s of a "
    "critical propulsion failure. */ } "
    "part def SafetyMonitor {} }"
)
_REQS = [
    "REQ-SAFE-004: do not arm if a sensor fails the power-on self-test.",
    "REQ-SAFE-005: deploy the parachute within 0.5 s of a critical propulsion "
    "failure.",
]


# ── the planning knowledge source ───────────────────────────────────────────

def test_plan_verification_classifies_each_requirement_by_its_text():
    plan = plan_verification(_MODEL)
    by_req = {e["requirement"]: e for e in plan["entries"]}
    assert plan["planned"] == 2
    # a Boolean invariant vs a quantified threshold get different planned tiers
    assert by_req["REQ_SAFE_004"]["planned_tier"] == "behavioral_or_inspection"
    assert by_req["REQ_SAFE_005"]["planned_tier"] == "analysis_or_sitl"
    assert all(e["traceable_in_model"] for e in plan["entries"])


def test_plan_verification_skips_ag_contract_defs():
    # an A/G system contract def is not a stakeholder requirement and is not planned
    model = _MODEL + (
        " package AG { requirement def SystemParachuteContract { doc /* bounded "
        "A/G system contract for REQ_SAFE_005; safety_pattern=X */ } }"
    )
    reqs = {e["requirement"] for e in plan_verification(model)["entries"]}
    assert reqs == {"REQ_SAFE_004", "REQ_SAFE_005"}


def test_requirement_def_slice_keeps_every_requirement_def():
    sliced = requirement_def_slice(_MODEL)
    assert "REQ_SAFE_004" in sliced and "REQ_SAFE_005" in sliced
    assert "part def SafetyMonitor" not in sliced  # only requirement defs kept


def test_planning_module_never_imports_gold_or_the_runtime_checker():
    src = Path("src/prototyping/verification_planning.py").read_text(encoding="utf-8")
    imports = "\n".join(
        l for l in src.splitlines() if l.strip().startswith(("import ", "from "))
    )
    assert "gold" not in imports.lower()
    assert "ag_contracts" not in imports and "ag_extractor" not in imports


# ── the board-mediated handoff ──────────────────────────────────────────────

def _run(arm: str):
    orch = Orchestrator(_NoCallLLM(), revised_experiment_arm=arm)
    orch.last_requirement_input = {"mode": "frozen", "requirement_set_digest": "d"}
    orch._prepare_design_handoff("DeliveryUAV", _REQS)
    model = build_lite_model(_MODEL, model_name="DeliveryUAV")
    orch._finalize_design_handoff(
        SimpleNamespace(success=True, reasoning="generated", metadata={}), model
    )
    final = orch._apply_ag_contract_layer(get_sysml_text(model), _REQS)
    orch._commit_terminal_model(final, producer="test")
    artifacts = orch._build_collaboration_artifacts(final)
    return orch, artifacts


def test_second_handoff_makes_the_metric_denominator_two_in_r2():
    orch, artifacts = _run("R2-BBAG")
    assert artifacts["verification_plan"]["planned"] == 2

    # a typed VerificationAgent result is on the board, in its own session/task
    results = orch.blackboard.records(topic="agent.verification.result")
    assert len(results) == 1 and results[0].payload["success"] is True
    sessions = artifacts["collaboration"]["task_sessions"]["sessions"]
    roles = sorted(s["agent_role"] for s in sessions)
    assert roles == ["DesignAgent", "VerificationAgent"]

    m = compute_coordination_metrics(artifacts["collaboration"])
    assert m["counts"]["migrated_handoffs"] == 2
    assert m["cross_agent_handoff_completeness"]["value"] == 1.0
    assert m["cross_agent_handoff_completeness"]["illustrative_single_handoff"] is False
    # both roles still obey the invariants
    assert m["cross_role_contamination"]["count"] == 0
    assert m["stale_revision_use"]["count"] == 0
    assert m["context_revision_consistency"]["value"] == 1.0
    sources = artifacts["control_agenda"]["registered_knowledge_sources"]
    assert [item["name"] for item in sources] == [
        "verification_planning",
        "ag_semantic_assurance",
    ]
    assert sources[1]["precondition_topics"] == [
        "model.terminal.ready",
        "agent.verification.result",
    ]
    assert [item["status"] for item in artifacts["control_agenda"]["activations"]] == [
        "COMPLETED",
        "COMPLETED",
    ]


def test_second_handoff_also_runs_in_r1_without_the_ag_layer():
    orch, artifacts = _run("R1-BBCTX")
    assert "ag_contract_graph" not in artifacts  # R1 has no A/G trace
    assert artifacts["verification_plan"]["planned"] == 2
    m = compute_coordination_metrics(artifacts["collaboration"])
    assert m["counts"]["migrated_handoffs"] == 2
    assert m["cross_role_contamination"]["count"] == 0


def test_verification_envelope_is_revision_pinned_and_gold_free():
    orch, artifacts = _run("R2-BBAG")
    envelopes = artifacts["collaboration"]["contexts"]["envelopes"]
    verif = [e for e in envelopes if e["agent_role"] == "VerificationAgent"]
    assert len(verif) == 1
    committed = {
        (r["revision"], r["model_digest"])
        for r in artifacts["collaboration"]["blackboard"]["model_revisions"]
    }
    # the envelope pins a real committed revision (the terminal one)
    assert (verif[0]["model_revision"], verif[0]["model_digest"]) in committed
