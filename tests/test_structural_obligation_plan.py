from types import SimpleNamespace

from src.prototyping.ag_behavior_plan import (
    compile_behavior_obligation_plan,
)
from src.prototyping.ag_chains import REQ_SAFE_008_CHAIN
from src.prototyping.generation_plan import ModelGenerationPlan
from src.prototyping.planned_behavior import (
    PlannedBehavior,
    PlannedState,
    PlannedTransition,
)
from src.prototyping.structural_obligations import (
    RequirementRealizationPlan,
    _concept_terms,
    _represents,
    compile_source_anchored_structural_obligations,
)


def _plan_payload():
    return {
        "components": [
            {
                "name": "Sensor",
                "responsibility": "Produces observations.",
                "requirements": ["REQ_FUNC_001"],
                "ports": [{
                    "name": "data",
                    "direction": "out",
                    "type": "DataPort",
                    "external": False,
                }],
            },
            {
                "name": "Controller",
                "responsibility": "Processes observations.",
                "requirements": ["REQ_FUNC_001"],
                "ports": [
                    {
                        "name": "data",
                        "direction": "in",
                        "type": "DataPort",
                        "external": False,
                    },
                    {
                        "name": "command",
                        "direction": "out",
                        "type": "CommandPort",
                        "external": False,
                    },
                ],
            },
            {
                "name": "Actuator",
                "responsibility": "Applies commands.",
                "requirements": ["REQ_FUNC_001"],
                "ports": [{
                    "name": "command",
                    "direction": "in",
                    "type": "CommandPort",
                    "external": False,
                }],
            },
        ],
        "connections": [
            {
                "source": {"component": "Sensor", "port": "data"},
                "target": {"component": "Controller", "port": "data"},
                "item_type": "DataPort",
                "requirements": ["REQ_FUNC_001"],
            },
            {
                "source": {"component": "Controller", "port": "command"},
                "target": {"component": "Actuator", "port": "command"},
                "item_type": "CommandPort",
                "requirements": ["REQ_FUNC_001"],
            },
        ],
    }


def test_compiles_path_from_connections():
    plan = ModelGenerationPlan.from_payload(
        _plan_payload(),
        requirements=["REQ_FUNC_001: sense and actuate"],
    )

    assert plan.status == "PASS"
    assert plan.schema_version == "3.0"
    assert len(plan.structural_obligations) == 1
    obligation = plan.structural_obligations[0]
    assert obligation.obligation_id == "STRUCT_REQ_FUNC_001_001"
    assert obligation.required_components == (
        "Sensor",
        "Controller",
        "Actuator",
    )
    assert len(obligation.required_connections) == 2


def test_parallel_paths_stay_separate():
    payload = _plan_payload()
    payload["components"].append({
        "name": "BackupController",
        "responsibility": "Processes an independent safety channel.",
        "requirements": ["REQ_FUNC_001"],
        "ports": [
            {
                "name": "data",
                "direction": "in",
                "type": "DataPort",
                "external": False,
            },
            {
                "name": "command",
                "direction": "out",
                "type": "CommandPort",
                "external": False,
            },
        ],
    })
    payload["connections"].extend([
        {
            "source": {"component": "Sensor", "port": "data"},
            "target": {"component": "BackupController", "port": "data"},
            "item_type": "DataPort",
            "requirements": ["REQ_FUNC_001"],
        },
        {
            "source": {"component": "BackupController", "port": "command"},
            "target": {"component": "Actuator", "port": "command"},
            "item_type": "CommandPort",
            "requirements": ["REQ_FUNC_001"],
        },
    ])
    # A shared pure input violates the plan's single-driver rule; inout keeps the
    # test on path compilation rather than fan-in.
    payload["components"][2]["ports"][0]["direction"] = "inout"

    plan = ModelGenerationPlan.from_payload(payload)

    assert len(plan.structural_obligations) == 2
    assert {
        item.required_components for item in plan.structural_obligations
    } == {
        ("Sensor", "Controller", "Actuator"),
        ("Sensor", "BackupController", "Actuator"),
    }


def test_rejects_untraced_requirement():
    payload = _plan_payload()
    payload["components"][0]["requirements"].append("REQ_SAFE_002")

    plan = ModelGenerationPlan.from_payload(
        payload,
        requirements=[
            "REQ_FUNC_001: sense and actuate",
            "REQ_SAFE_002: stop safely",
        ],
    )

    assert plan.status == "INVALID"
    assert (
        "REQ_SAFE_002 has no requirement-traceable structural path"
        in plan.issues
    )


def _source_anchored_safety_payload():
    return {
        "components": [
            {
                "name": "PerceptionSystem",
                "responsibility": "Reports sensor health.",
                "requirements": ["REQ_SAFE_005"],
                "ports": [{
                    "name": "sensorStatus",
                    "direction": "out",
                    "type": "SensorStatusPort",
                }],
            },
            {
                "name": "PropulsionSystem",
                "responsibility": "Reports critical propulsion failures.",
                "requirements": ["REQ_SAFE_005"],
                "ports": [{
                    "name": "propulsionStatus",
                    "direction": "out",
                    "type": "PropulsionStatusPort",
                }],
            },
            {
                "name": "SafetyResponseArbiter",
                "responsibility": "Commands parachute recovery after failure.",
                "requirements": ["REQ_SAFE_005"],
                "ports": [
                    {
                        "name": "sensorStatus",
                        "direction": "in",
                        "type": "SensorStatusPort",
                    },
                    {
                        "name": "propulsionStatus",
                        "direction": "in",
                        "type": "PropulsionStatusPort",
                    },
                    {
                        "name": "parachuteCmd",
                        "direction": "out",
                        "type": "ParachuteCommandPort",
                    },
                ],
            },
            {
                "name": "RecoverySystem",
                "responsibility": "Performs commanded parachute recovery.",
                "requirements": ["REQ_SAFE_005"],
                "ports": [{
                    "name": "parachuteCmd",
                    "direction": "in",
                    "type": "ParachuteCommandPort",
                }],
            },
        ],
        "connections": [
            {
                "source": {
                    "component": "PerceptionSystem",
                    "port": "sensorStatus",
                },
                "target": {
                    "component": "SafetyResponseArbiter",
                    "port": "sensorStatus",
                },
                "item_type": "SensorStatusPort",
                "requirements": ["REQ_SAFE_005"],
            },
            {
                "source": {
                    "component": "PropulsionSystem",
                    "port": "propulsionStatus",
                },
                "target": {
                    "component": "SafetyResponseArbiter",
                    "port": "propulsionStatus",
                },
                "item_type": "PropulsionStatusPort",
                "requirements": ["REQ_SAFE_005"],
            },
            {
                "source": {
                    "component": "SafetyResponseArbiter",
                    "port": "parachuteCmd",
                },
                "target": {
                    "component": "RecoverySystem",
                    "port": "parachuteCmd",
                },
                "item_type": "ParachuteCommandPort",
                "requirements": ["REQ_SAFE_005"],
            },
        ],
    }


def _realization(first_component, first_port, first_type):
    return {
        "requirement_id": "REQ_SAFE_005",
        "realization_kind": "CAUSAL_PATH",
        "trigger_concept": "critical propulsion subsystem failure",
        "effect_concept": "command parachute recovery",
        "connection_path": [
            {
                "source_component": first_component,
                "source_port": first_port,
                "target_component": "SafetyResponseArbiter",
                "target_port": first_port,
                "item_type": first_type,
            },
            {
                "source_component": "SafetyResponseArbiter",
                "source_port": "parachuteCmd",
                "target_component": "RecoverySystem",
                "target_port": "parachuteCmd",
                "item_type": "ParachuteCommandPort",
            },
        ],
    }


def test_anchor_rejects_wrong_trigger():
    payload = _source_anchored_safety_payload()
    payload["requirement_realizations"] = [
        _realization(
            "PerceptionSystem", "sensorStatus", "SensorStatusPort"
        )
    ]

    plan = ModelGenerationPlan.from_payload(
        payload,
        requirements=[
            "REQ_SAFE_005: Upon critical propulsion subsystem failure, "
            "the system shall command parachute recovery."
        ],
        require_source_anchored_paths=True,
    )

    assert plan.status == "INVALID"
    assert any(
        "does not represent trigger phrase" in issue
        for issue in plan.issues
    )


def test_anchor_freezes_causal_path():
    payload = _source_anchored_safety_payload()
    payload["requirement_realizations"] = [
        _realization(
            "PropulsionSystem",
            "propulsionStatus",
            "PropulsionStatusPort",
        )
    ]

    plan = ModelGenerationPlan.from_payload(
        payload,
        requirements=[
            "REQ_SAFE_005: Upon critical propulsion subsystem failure, "
            "the system shall command parachute recovery."
        ],
        require_source_anchored_paths=True,
    )

    assert plan.status == "PASS"
    assert plan.schema_version == "6.0"
    obligation = plan.structural_obligations[0]
    assert obligation.source_component == "PropulsionSystem"
    assert obligation.target_component == "RecoverySystem"
    assert obligation.provenance == "FROZEN_REQUIREMENT_REALIZATION"
    assert obligation.trigger_concept == (
        "critical propulsion subsystem failure"
    )


def test_power_on_uses_initial_state():
    requirement = (
        "REQ-SAFE-008: The payload-release actuator shall default to the "
        "mechanically locked state upon power-on, before any arming or flight "
        "authorisation."
    )
    realization = RequirementRealizationPlan(
        requirement_id="REQ_SAFE_008",
        realization_kind="LOCAL_BEHAVIOR",
        trigger_concept="upon power-on",
        effect_concept="default to the mechanically locked state",
        connection_path=(),
        owner_component="PayloadLockMechanism",
        behavior_kind="STATE_DEF",
        behavior_name="LockStateBehavior",
    )
    component = SimpleNamespace(
        name="PayloadLockMechanism",
        requirements=("REQ_SAFE_008",),
        responsibility="Keeps the payload mechanically locked by default.",
    )
    ag_plan = compile_behavior_obligation_plan((REQ_SAFE_008_CHAIN,))

    obligations, issues = compile_source_anchored_structural_obligations(
        (realization,),
        (component,),
        (),
        requirements=(requirement,),
        require_complete=True,
        behavior_obligations=ag_plan.obligations,
    )

    assert issues == ()
    assert len(obligations) == 1
    assert obligations[0].entry_kind == (
        "SOURCE_ANCHORED_LOCAL_BEHAVIOR"
    )


def test_power_on_needs_typed_state():
    requirement = (
        "REQ-SAFE-008: The payload-release actuator shall default to the "
        "mechanically locked state upon power-on."
    )
    realization = RequirementRealizationPlan(
        requirement_id="REQ_SAFE_008",
        realization_kind="LOCAL_BEHAVIOR",
        trigger_concept="upon power-on",
        effect_concept="default to the mechanically locked state",
        connection_path=(),
        owner_component="PayloadLockMechanism",
        behavior_kind="STATE_DEF",
        behavior_name="LockStateBehavior",
    )
    component = SimpleNamespace(
        name="PayloadLockMechanism",
        requirements=("REQ_SAFE_008",),
        responsibility="Keeps the payload mechanically locked by default.",
    )

    obligations, issues = compile_source_anchored_structural_obligations(
        (realization,),
        (component,),
        (),
        requirements=(requirement,),
        require_complete=True,
    )

    assert obligations == ()
    assert any(
        "requires a typed state behavior with an initial state" in issue
        for issue in issues
    )


def test_event_behavior_typed_evidence():
    requirement = (
        "REQ-SAFE-009: On an emergency command, the system shall enter "
        "failsafe mode."
    )
    realization = RequirementRealizationPlan(
        requirement_id="REQ_SAFE_009",
        realization_kind="LOCAL_BEHAVIOR",
        trigger_concept="emergency command",
        effect_concept="enter failsafe mode",
        connection_path=(),
        owner_component="SafetyController",
        behavior_kind="STATE_DEF",
        behavior_name="SafetyBehavior",
    )
    component = SimpleNamespace(
        name="SafetyController",
        requirements=("REQ_SAFE_009",),
        responsibility="Controls system safety.",
    )
    behavior = PlannedBehavior(
        owner="SafetyController",
        behavior_id="SafetyBehavior",
        initial_state="nominal",
        states=(
            PlannedState("nominal", "INITIAL"),
            PlannedState(
                "failsafeMode",
                "RESPONSE",
                entry_action="enterFailsafeMode",
            ),
        ),
        transitions=(
            PlannedTransition(
                "activateFailsafe",
                "nominal",
                "failsafeMode",
                "ACCEPT",
                "EmergencyCommandSignal",
            ),
        ),
        provenance="FROZEN_REQUIREMENT",
        source_requirement_id="REQ_SAFE_009",
    )

    obligations, issues = compile_source_anchored_structural_obligations(
        (realization,),
        (component,),
        (),
        requirements=(requirement,),
        require_complete=True,
        planned_behaviors=(behavior,),
    )

    assert issues == ()
    assert len(obligations) == 1


def test_strict_needs_anchored_path():
    plan = ModelGenerationPlan.from_payload(
        _plan_payload(),
        requirements=["REQ_FUNC_001: sense and actuate"],
        require_source_anchored_paths=True,
    )

    assert plan.status == "INVALID"
    assert (
        "REQ_FUNC_001 has no source-anchored requirement realization"
        in plan.issues
    )


def test_local_behavior_no_connection():
    payload = _plan_payload()
    for component in payload["components"]:
        component["requirements"] = []
    payload["components"][1]["requirements"] = ["REQ_OPER_001"]
    payload["components"][1]["responsibility"] = (
        "Transitions operating mode from standby to active."
    )
    for connection in payload["connections"]:
        connection["requirements"] = []
    payload["requirement_realizations"] = [{
        "requirement_id": "REQ_OPER_001",
        "realization_kind": "LOCAL_BEHAVIOR",
        "trigger_concept": "standby mode",
        "effect_concept": "active mode",
        "owner_component": "Controller",
        "behavior_kind": "STATE_DEF",
        "behavior_name": "OperatingModes",
        "connection_path": [],
    }]

    plan = ModelGenerationPlan.from_payload(
        payload,
        requirements=[
            "REQ_OPER_001: The controller shall transition from standby "
            "mode to active mode."
        ],
        require_source_anchored_paths=True,
    )

    assert plan.status == "PASS"
    obligation = plan.structural_obligations[0]
    assert obligation.realization_kind == "LOCAL_BEHAVIOR"
    assert obligation.required_connections == ()
    assert obligation.behavior_name == "OperatingModes"


def _latency_requirement():
    return [
        "REQ_SAFE_005: Upon critical propulsion subsystem failure, the "
        "system shall command parachute recovery with a deployment latency "
        "of less than 0.5 seconds."
    ]


def test_adjunct_needs_no_endpoint():
    """A tolerance clause qualifies a behaviour instead of naming one.

    The planner copies the effect phrase verbatim, and where the requirement offers
    only a bound there is nothing a port name could represent (2026-08-01 attempt).
    """
    payload = _source_anchored_safety_payload()
    realization = _realization(
        "PropulsionSystem", "propulsionStatus", "PropulsionStatusPort"
    )
    realization["effect_concept"] = (
        "with a deployment latency of less than 0.5 seconds"
    )
    payload["requirement_realizations"] = [realization]

    plan = ModelGenerationPlan.from_payload(
        payload,
        requirements=_latency_requirement(),
        require_source_anchored_paths=True,
    )

    assert plan.status == "PASS"
    assert not any(
        "does not represent effect phrase" in issue for issue in plan.issues
    )


def test_adjunct_rejects_unrelated():
    payload = _source_anchored_safety_payload()
    realization = _realization(
        "PerceptionSystem", "sensorStatus", "SensorStatusPort"
    )
    realization["trigger_concept"] = (
        "when the propulsion unit has failed"
    )
    payload["requirement_realizations"] = [realization]

    plan = ModelGenerationPlan.from_payload(
        payload,
        requirements=[
            "REQ_SAFE_005: When the propulsion unit has failed, the system "
            "shall command parachute recovery."
        ],
        require_source_anchored_paths=True,
    )

    assert plan.status == "INVALID"
    assert any(
        "does not represent trigger phrase" in issue for issue in plan.issues
    )


def test_abbreviation_represents_phrase():
    assert _represents(
        _concept_terms("autonomously navigate to designated GPS waypoints"),
        _concept_terms("PerceptionSystem navState NavigationStatePort"),
    )
    assert _represents(
        _concept_terms("receive and apply differential GNSS corrections"),
        _concept_terms("GNSSReceiver correctionIn"),
    )


def test_synonym_bridges_identifier():
    assert _represents(
        _concept_terms("approaching a stationary collision threat"),
        _concept_terms("PerceptionSystem obstacleData"),
    )
    assert _represents(
        _concept_terms("transmit a post-flight system health report"),
        _concept_terms("CommunicationSystem telemetry"),
    )


def test_unrelated_endpoint_fails():
    assert not _represents(
        _concept_terms("completing an automated landing"),
        _concept_terms("FlightController telemetry"),
    )
    assert not _represents(
        _concept_terms("not transition to the armed or airborne state"),
        _concept_terms("SafetyMonitor sensorStatus"),
    )


def test_negation_not_abbreviation():
    assert not _represents(
        _concept_terms("shall not arm"),
        _concept_terms("NotificationService noticeOut"),
    )
