from src.prototyping.generation_plan import ModelGenerationPlan


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


def test_compiles_stable_requirement_path_from_typed_connections():
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


def test_parallel_requirement_paths_remain_separate_obligations():
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
    # The shared pure input would violate the plan's single-driver rule.  Use
    # inout here so the test isolates path compilation rather than fan-in.
    payload["components"][2]["ports"][0]["direction"] = "inout"

    plan = ModelGenerationPlan.from_payload(payload)

    assert len(plan.structural_obligations) == 2
    assert {
        item.required_components for item in plan.structural_obligations
    } == {
        ("Sensor", "Controller", "Actuator"),
        ("Sensor", "BackupController", "Actuator"),
    }


def test_rejects_structural_requirement_without_traced_connection():
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
