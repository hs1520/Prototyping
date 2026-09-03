"""Regression suite for the plan-frozen repair deadlock (2026-08-31 run 2).

The run terminated NOT_QUALIFIED after 8 idle refinement iterations against
``structural_repair_blocked``. Root causes, pinned against the run's archived
plan and final model (zero LLM):
- D1 - ``requirement_realizations[].behavior_kind`` contradicted the same plan's
  ``behaviors[]`` and failed a structurally correct model (17/19). behaviors[]
  is the sole writer, so the kind is derived and reconciled at parse time.
- D2 - the inhibition gate was scoped per-behaviour and fired zero times on the
  plan whose vehicle separated its payload during an active abort; it is now
  anchored on the whole plan (held state + departure vocabulary + fail-loud).
- The futility loop - ``structural_repair_blocked`` had no consumer, so the loop
  paid for iterations that could not progress.
- The missing edge - no path back to revise a wrong frozen plan;
  ``PlanRevision`` is that path, issue-authorized and diff-gated.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping

from tests._dep_stubs import install_missing_dep_stubs

install_missing_dep_stubs()

from src.agents.plan_revision import (
    PlanRevision,
    PlanRevisionRequest,
    _issue_authorized_diff,
)
from src.agents.refinement import _structural_block_signature
from src.prototyping.generation_plan import ModelGenerationPlan
from src.prototyping.planned_behavior import (
    PlannedBehavior,
    PlannedState,
    PlannedTransition,
    _plan_inhibition_issues,
)
from src.prototyping.structural_obligations import (
    compile_source_anchored_structural_obligations,
    validate_structural_obligations,
)
from src.utils.req_id import source_requirements_by_id

_FIXTURES = Path(__file__).parent / "fixtures" / "plan_deadlock_20260831"


def _run2_plan_payload() -> Dict[str, Any]:
    return json.loads(
        (_FIXTURES / "whole_model_generation_plan.json").read_text()
    )


def _run2_requirements() -> List[str]:
    return json.loads((_FIXTURES / "requirements.json").read_text())


def _run2_model_text() -> str:
    return (_FIXTURES / "final_model.sysml").read_text()


class TestBehaviorKindReconciliation:
    def test_run2_plan_kinds_reconciled(self):
        """The archived plan against the final model: 17/19 with the contradictory
        ACTION_DEF kinds, 19/19 once the kind is reconciled toward behaviors[]; that
        is run 2's NOT_QUALIFIED.
        """
        plan = ModelGenerationPlan.from_dict(_run2_plan_payload())

        kinds = {
            obligation.requirement_id: obligation.behavior_kind
            for obligation in plan.structural_obligations
            if obligation.requirement_id in ("REQ_FUNC_006", "REQ_FUNC_007")
        }
        assert kinds == {
            "REQ_FUNC_006": "STATE_DEF",
            "REQ_FUNC_007": "STATE_DEF",
        }

        report = validate_structural_obligations(
            _run2_model_text(),
            plan.structural_obligations,
            model_name="drone_v2",
        )
        assert report["status"] == "PASS"
        assert all(
            item["status"] == "PASS" for item in report["results"]
        )

    def test_reconciliation_audited_idempotent(self):
        plan = ModelGenerationPlan.from_dict(_run2_plan_payload())
        audit = [
            item for item in plan.behavior_identity_reconciliations
            if "behavior_kind" in item
        ]
        assert len(audit) == 2
        assert any("REQ_FUNC_006" in item for item in audit)
        assert any("sole writer" in item for item in audit)

        second = ModelGenerationPlan.from_dict(plan.to_dict())
        assert list(second.behavior_identity_reconciliations).count(
            audit[0]
        ) == 1

    def test_compiler_rejects_contradiction(self):
        plan = ModelGenerationPlan.from_dict(_run2_plan_payload())
        contradictory = [
            item.__class__.from_dict({
                **item.to_dict(),
                "behavior_kind": "ACTION_DEF",
            })
            if item.requirement_id == "REQ_FUNC_006" else item
            for item in plan.requirement_realizations
        ]

        _obligations, issues = (
            compile_source_anchored_structural_obligations(
                contradictory,
                plan.components,
                plan.connections,
                requirements=_run2_requirements(),
                planned_behaviors=plan.planned_behaviors,
            )
        )
        assert any(
            "contradicts behaviors[]" in issue for issue in issues
        )


_SAFE_006 = (
    "REQ-SAFE-006: The system shall maintain the payload in the mechanically "
    "locked state whenever a delivery-abort condition is active, regardless "
    "of geographic proximity to the delivery waypoint."
)
_FUNC_005 = (
    "REQ-FUNC-005: The system shall release the payload within 1.0 m of the "
    "designated delivery waypoint."
)


def _release_behavior(guard: str = "") -> PlannedBehavior:
    return PlannedBehavior(
        owner="PayloadMechanism",
        behavior_id="PayloadReleaseBehavior",
        initial_state="Initial",
        states=(
            PlannedState("Initial", role="INITIAL"),
            PlannedState("ReleasingPayload", role="RESPONSE",
                         entry_action="releasePayload"),
        ),
        transitions=(PlannedTransition(
            transition_id="executeRelease",
            source="Initial", target="ReleasingPayload",
            trigger_kind="ACCEPT", trigger="DeliveryWaypointReached",
            guard=guard,
        ),),
        source_requirement_id="REQ_FUNC_005",
    )


def _abort_lock_behavior() -> PlannedBehavior:
    return PlannedBehavior(
        owner="PayloadMechanism",
        behavior_id="DeliveryAbortLockManagement",
        initial_state="LockNominal",
        states=(
            PlannedState("LockNominal", role="INITIAL"),
            PlannedState("LockActive", role="RESPONSE",
                         entry_action="holdLock"),
        ),
        transitions=(PlannedTransition(
            transition_id="activateLock",
            source="LockNominal", target="LockActive",
            trigger_kind="GUARD", trigger="deliveryAbortConditionActive",
        ),),
        source_requirement_id="REQ_SAFE_006",
    )


class TestPlanWideInhibitionGate:
    def test_run2_plan_fires_one_issue(self):
        plan = ModelGenerationPlan.from_dict(_run2_plan_payload())
        issues = _plan_inhibition_issues(
            plan.planned_behaviors,
            source_requirements_by_id(_run2_requirements()),
        )
        assert len(issues) == 1
        assert "PayloadReleaseBehavior" in issues[0]
        assert "REQ_SAFE_006" in issues[0]
        assert "abort" in issues[0]
        assert "guard" in issues[0]

    def test_two_machines_caught_by_gate(self):
        issues = _plan_inhibition_issues(
            [_release_behavior(), _abort_lock_behavior()],
            source_requirements_by_id([_SAFE_006, _FUNC_005]),
        )
        assert any(
            "executeRelease" in issue and "REQ_SAFE_006" in issue
            for issue in issues
        )

    def test_condition_guard_satisfies_gate(self):
        issues = _plan_inhibition_issues(
            [
                _release_behavior("not deliveryAbortConditionActive"),
                _abort_lock_behavior(),
            ],
            source_requirements_by_id([_SAFE_006, _FUNC_005]),
        )
        assert issues == []

    def test_plan_without_anchor_fails_loud(self):
        unrelated = PlannedBehavior(
            owner="FlightController",
            behavior_id="WaypointNavigationBehavior",
            initial_state="Initial",
            states=(
                PlannedState("Initial", role="INITIAL"),
                PlannedState("Navigating", role="RESPONSE",
                             entry_action="navigateWaypoints"),
            ),
            transitions=(PlannedTransition(
                transition_id="startNavigation",
                source="Initial", target="Navigating",
                trigger_kind="ACCEPT", trigger="WaypointsUploaded",
            ),),
            source_requirement_id="REQ_FUNC_001",
        )
        issues = _plan_inhibition_issues(
            [unrelated],
            source_requirements_by_id([_SAFE_006]),
        )
        assert len(issues) == 1
        assert "cannot express this inhibition" in issues[0]

    def test_forbidden_state_needs_guard(self):
        requirement = (
            "REQ-SAFE-004: The system shall not transition to the armed "
            "state if any onboard sensor reports a failure."
        )
        arming = PlannedBehavior(
            owner="FlightController",
            behavior_id="ArmingBehavior",
            initial_state="Idle",
            states=(
                PlannedState("Idle", role="INITIAL"),
                PlannedState("Armed", role="RESPONSE",
                             entry_action="enableMotors"),
            ),
            transitions=(PlannedTransition(
                transition_id="armVehicle",
                source="Idle", target="Armed",
                trigger_kind="ACCEPT", trigger="ArmCommand",
            ),),
            source_requirement_id="REQ_SAFE_004",
        )
        issues = _plan_inhibition_issues(
            [arming], source_requirements_by_id([requirement])
        )
        assert len(issues) == 1
        assert "armVehicle" in issues[0]
        assert "forbids" in issues[0]

        guarded = PlannedBehavior(
            owner=arming.owner,
            behavior_id=arming.behavior_id,
            initial_state=arming.initial_state,
            states=arming.states,
            transitions=(PlannedTransition(
                transition_id="armVehicle",
                source="Idle", target="Armed",
                trigger_kind="ACCEPT", trigger="ArmCommand",
                guard="not sensorFailureDetected",
            ),),
            source_requirement_id="REQ_SAFE_004",
        )
        assert _plan_inhibition_issues(
            [guarded], source_requirements_by_id([requirement])
        ) == []


class _MetadataModel:
    def __init__(self, metadata: Dict[str, Any]) -> None:
        self.metadata = metadata


class TestStructuralBlockSignature:
    _BLOCKED = {
        "structural_repair_blocked": {
            "structural_obligation_report": {
                "results": [
                    {"obligation_id": "STRUCT_REQ_FUNC_006_001",
                     "status": "FAIL"},
                    {"obligation_id": "STRUCT_REQ_FUNC_001_001",
                     "status": "PASS"},
                ],
            },
        },
        "last_sysml_text": "package P {}",
    }

    def test_unblocked_model_has_no_signature(self):
        assert _structural_block_signature(
            _MetadataModel({"last_sysml_text": "package P {}"})
        ) is None

    def test_identical_block_same_signature(self):
        first = _structural_block_signature(
            _MetadataModel(dict(self._BLOCKED))
        )
        second = _structural_block_signature(
            _MetadataModel(dict(self._BLOCKED))
        )
        assert first == second
        assert first[0] == frozenset({"STRUCT_REQ_FUNC_006_001"})

    def test_changed_text_breaks_signature(self):
        base = _structural_block_signature(
            _MetadataModel(dict(self._BLOCKED))
        )
        retexted = dict(self._BLOCKED)
        retexted["last_sysml_text"] = "package P { part def X; }"
        progressed = json.loads(json.dumps(self._BLOCKED))
        progressed["structural_repair_blocked"][
            "structural_obligation_report"
        ]["results"][0]["status"] = "PASS"

        assert _structural_block_signature(
            _MetadataModel(retexted)
        ) != base
        assert _structural_block_signature(
            _MetadataModel(progressed)
        ) != base


@dataclass
class _FakePlanResponse:
    extracted_json: Any = None
    final_answer: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class _FakePrompter:
    def __init__(self, payloads: List[Any]) -> None:
        self._payloads = list(payloads)
        self.call_count = 0
        self.contexts: List[str] = []

    def decompose_architecture(
        self, *, system_name, requirements, context, temperature
    ) -> _FakePlanResponse:
        self.call_count += 1
        self.contexts.append(context)
        payload = (
            self._payloads[self.call_count - 1]
            if self.call_count <= len(self._payloads)
            else self._payloads[-1]
        )
        return _FakePlanResponse(extracted_json=payload)


def _blocked_record(requirement_id: str = "REQ_FUNC_006") -> Dict[str, Any]:
    return {
        "structural_obligation_report": {
            "results": [{
                "obligation_id": f"STRUCT_{requirement_id}_001",
                "status": "FAIL",
                "issues": [
                    "missing required local behavior ACTION_DEF "
                    f"FlightController.SomeBehavior [{requirement_id}]"
                ],
            }],
        },
        "generation_plan_conformance": {"issues": []},
    }


def _guarded_revision(frozen_payload: Mapping[str, Any]) -> Dict[str, Any]:
    """A minimal valid revision of the run-2 payload.

    Re-validated under the current gates the frozen plan carries one issue - the
    plan-wide inhibition gate on the unguarded release boundary - so an acceptable
    revision guards it, and the authorizing issue names the behaviour.
    """
    revised = json.loads(json.dumps(dict(frozen_payload)))
    for behavior in revised["behaviors"]:
        if behavior.get("behavior_id") == "PayloadReleaseBehavior":
            for transition in behavior["transitions"]:
                transition["guard"] = "not deliveryAbortConditionActive"
    return revised


class TestPlanRevision:
    def test_authorized_revision_accepted(self):
        frozen_payload = _run2_plan_payload()
        requirements = _run2_requirements()
        revised_payload = _guarded_revision(frozen_payload)
        for item in revised_payload["requirement_realizations"]:
            if item["requirement_id"] == "REQ_FUNC_006":
                item["behavior_name"] = "WaypointModificationBehavior"
        prompter = _FakePrompter([revised_payload])

        outcome = PlanRevision(prompter).revise(PlanRevisionRequest(
            system_name="drone_v2",
            requirements=requirements,
            frozen_plan=frozen_payload,
            blocked=_blocked_record("REQ_FUNC_006"),
        ))

        assert outcome.record["status"] == "ACCEPTED"
        assert outcome.plan is not None
        assert prompter.call_count == 1
        assert "REQ_FUNC_006" in prompter.contexts[0]
        assert "FROZEN PLAN" in prompter.contexts[0]

    def test_unauthorized_component_rejected(self):
        frozen_payload = _run2_plan_payload()
        requirements = _run2_requirements()
        tampered = _guarded_revision(frozen_payload)
        # Semantically valid on its own - a passive bracket passes every
        # validator - but no issue authorizes touching the component surface.
        tampered["components"].append({
            "name": "AuxiliaryBracket",
            "responsibility": "Purely structural mounting bracket.",
            "requirements": [],
            "ports": [],
            "passive": True,
            "passive_rationale": "exchanges nothing",
        })
        prompter = _FakePrompter([tampered, tampered])

        outcome = PlanRevision(prompter).revise(PlanRevisionRequest(
            system_name="drone_v2",
            requirements=requirements,
            frozen_plan=frozen_payload,
            blocked=_blocked_record("REQ_FUNC_006"),
        ))

        assert outcome.plan is None
        assert outcome.record["status"] == "REJECTED"
        assert any(
            "components changed" in issue
            for attempt in outcome.record["attempts"]
            for issue in attempt["issues"]
        )

    def test_unnamed_behavior_change_rejected(self):
        frozen_payload = _run2_plan_payload()
        requirements = _run2_requirements()
        tampered = _guarded_revision(frozen_payload)
        # A transition-id rename is semantically inert and passes every
        # validator; only the diff gate can know it was not asked for.
        for behavior in tampered["behaviors"]:
            if behavior.get("behavior_id") == "WaypointNavigationBehavior":
                behavior["transitions"][0]["transition_id"] = "navStart"
        prompter = _FakePrompter([tampered, tampered])

        outcome = PlanRevision(prompter).revise(PlanRevisionRequest(
            system_name="drone_v2",
            requirements=requirements,
            frozen_plan=frozen_payload,
            blocked=_blocked_record("REQ_FUNC_006"),
        ))

        assert outcome.plan is None
        assert any(
            "WaypointNavigationBehavior" in issue
            and "no issue names it" in issue
            for attempt in outcome.record["attempts"]
            for issue in attempt["issues"]
        )

    def test_revision_fails_closed_no_json(self):
        frozen_payload = _run2_plan_payload()
        prompter = _FakePrompter([None, None])

        outcome = PlanRevision(prompter).revise(PlanRevisionRequest(
            system_name="drone_v2",
            requirements=_run2_requirements(),
            frozen_plan=frozen_payload,
            blocked=_blocked_record("REQ_FUNC_006"),
        ))

        assert outcome.plan is None
        assert outcome.record["status"] == "REJECTED"
        assert prompter.call_count == 2

    def test_diff_gate_names_every_violation(self):
        frozen = ModelGenerationPlan.from_dict(_run2_plan_payload())
        tampered_payload = json.loads(json.dumps(_run2_plan_payload()))
        tampered_payload["connections"] = (
            tampered_payload["connections"][:-1]
        )
        tampered = ModelGenerationPlan.from_payload(
            tampered_payload,
            requirements=_run2_requirements(),
            source="LLM_PLAN_REVISION",
        )
        violations = _issue_authorized_diff(
            frozen, tampered, ["STRUCT_REQ_FUNC_006_001: some issue"]
        )
        assert any("connections changed" in item for item in violations)


class TestDiagnosticFalsePositives:
    def test_portless_parts_not_reported(self):
        from src.dse.diagnostics import diagnose
        from src.sysml.lite_model import build_lite_model

        text = """package P {
            port def SignalPort;
            part def PropulsionSystem {
                in port cmd : SignalPort;
                out port fault : SignalPort;
            }
            part def Catalog_r4PropulsionsystemImpl :> PropulsionSystem { }
            part def DseDesignAnalysis { }
            part def MountingBracket {
                // PLAN-PASSIVE MountingBracket: exchanges nothing
            }
            part def GenuinelyUnwired { }
        }"""
        model = build_lite_model(text, model_name="P")
        issues, _recs = diagnose(model, None)

        port_issues = [i for i in issues if "no ports" in i]
        assert len(port_issues) == 1
        assert "GenuinelyUnwired" in port_issues[0]
        assert "Catalog_r4PropulsionsystemImpl" not in port_issues[0]
        assert "DseDesignAnalysis" not in port_issues[0]
        assert "MountingBracket" not in port_issues[0]

    def test_feedback_header_matches_payload(self):
        """D4: the header prescribed `connect` with `::` endpoints above
        missing-behaviour issues, while the same prompt states `::` breaks
        the parser."""
        from types import SimpleNamespace

        from src.agents.refinement_transaction import (
            _build_refinement_feedback,
        )

        feedback = _build_refinement_feedback(SimpleNamespace(
            mcts_constraints="",
            evaluation=SimpleNamespace(issues=[], recommendations=[]),
            simulation_issues=[
                "FROZEN CAUSAL PATH STRUCT_REQ_FUNC_006_001 failed: missing "
                "required local behavior ACTION_DEF FlightController."
                "WaypointModificationBehavior",
            ],
            persistent_issues=[],
            cot_feedback="",
        ))

        assert "missing local behavior" in feedback
        assert "state def / action def" in feedback
        assert "`::` breaks the parser" in feedback
        assert "<source_part>.<port>" in feedback
        assert "<source_part>::<port>" not in feedback


def _wired_context_and_orch(prompter):
    from types import SimpleNamespace

    from src.agents.orchestrator import Orchestrator, PrototypingState
    from src.agents.pipeline_records import GenerationContext
    from src.llm.interface import MockLLM
    from src.sysml.lite_model import build_lite_model

    orch = Orchestrator(llm=MockLLM(), use_surgical_refinement=False)
    orch.state = PrototypingState(system_name="drone_v2",
                                  system_description="")
    orch.cot = prompter
    requirements = _run2_requirements()
    model = build_lite_model(_run2_model_text(), model_name="drone_v2")
    plan_payload = _run2_plan_payload()
    model.metadata["whole_model_generation_plan"] = plan_payload
    model.metadata["structural_repair_blocked"] = {
        "structural_obligation_report": {
            "results": [
                {"obligation_id": "STRUCT_REQ_FUNC_006_001",
                 "status": "FAIL",
                 "issues": ["missing required local behavior ACTION_DEF "
                            "FlightController.WaypointModificationBehavior "
                            "[REQ_FUNC_006]"]},
                {"obligation_id": "STRUCT_REQ_FUNC_007_001",
                 "status": "FAIL",
                 "issues": ["missing required local behavior ACTION_DEF "
                            "FlightController.ContingencyReturnBehavior "
                            "[REQ_FUNC_007]"]},
            ],
        },
        "generation_plan_conformance": {"issues": []},
    }
    orch._active_model_generation_plan = dict(plan_payload)
    context = GenerationContext(
        system_name="drone_v2",
        system_description="",
        additional_requirements=[],
        parse_strict=None,
        platform_profile=None,
        frozen_requirements=None,
    )
    context.requirements = requirements
    context.final_model = model
    context.final_score = 0.9
    context.final_sim = None
    # The accepted path re-runs refinement; these tests pin the wiring, not
    # the engine, so the closure is a stub that echoes the revised model.
    orch.refinement_closure = SimpleNamespace(
        refine=lambda request: SimpleNamespace(
            materialize=lambda: (context.final_model, 0.95, None),
        ),
    )
    return orch, context


class TestPlanRevisionWiring:
    def test_rejected_revision_keeps_plan(self):
        prompter = _FakePrompter([None, None])
        orch, context = _wired_context_and_orch(prompter)
        frozen = dict(context.final_model.metadata[
            "whole_model_generation_plan"
        ])

        orch._attempt_frozen_plan_revision(context)

        record = context.final_model.metadata["plan_revision"]
        assert record["status"] == "REJECTED"
        assert record["applied"] is False
        assert context.final_model.metadata[
            "whole_model_generation_plan"
        ] == frozen
        assert "structural_repair_blocked" in context.final_model.metadata
        assert prompter.call_count == 2

    def test_accepted_revision_applied(self):
        revised = _guarded_revision(_run2_plan_payload())
        prompter = _FakePrompter([revised])
        orch, context = _wired_context_and_orch(prompter)

        orch._attempt_frozen_plan_revision(context)

        record = context.final_model.metadata["plan_revision"]
        assert record["status"] == "ACCEPTED"
        assert record["applied"] is True
        assert record["monotonic"] is True
        assert record["unsatisfied_after"] == []
        assert "structural_repair_blocked" not in (
            context.final_model.metadata
        )
        committed = context.final_model.metadata[
            "whole_model_generation_plan"
        ]
        release = next(
            item for item in committed["behaviors"]
            if item["behavior_id"] == "PayloadReleaseBehavior"
        )
        assert release["transitions"][0]["guard"]
        assert orch._active_model_generation_plan == committed
        assert context.final_score == 0.95

    def test_revision_can_be_disabled(self):
        prompter = _FakePrompter([None])
        orch, context = _wired_context_and_orch(prompter)
        orch.enable_plan_revision = False

        orch._attempt_frozen_plan_revision(context)

        assert prompter.call_count == 0
        assert "plan_revision" not in context.final_model.metadata
