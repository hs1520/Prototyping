from __future__ import annotations

from src.prototyping.repair_packet import build_scoped_repair_packet
from src.prototyping.requirement_contracts import build_contract_bundle
from src.prototyping.safety_patterns import select_patterns


_TARGET = (
    "REQ-SAFE-005: Critical propulsion failure shall deploy the parachute "
    "within 0.5 seconds."
)
_UNRELATED = (
    "REQ-SAFE-006: When battery state of charge falls to 25 percent, the "
    "system shall return to base."
)


def _report():
    return {
        "traces": [
            {
                "req_id": "REQ_SAFE_005",
                "contract_status": "READY",
                "pattern_ids": ["TriggeredFailsafeResponse"],
                "links": [{
                    "req_id": "REQ_SAFE_005",
                    "obligation_id": "REQ_SAFE_005.O1",
                    "link_kind": "action_to_command",
                    "expected_concept": "deploy_parachute",
                    "observed_element": "CMD_LAND",
                    "status": "FAIL",
                    "evidence": ["CMD_LAND"],
                }],
                "conformance": "FAIL",
                "findings": [{
                    "diagnostic_id": (
                        "REQ_SAFE_005:REQ_SAFE_005.O1:"
                        "ACTION_PLATFORM_BINDING_MISMATCH"
                    ),
                    "req_id": "REQ_SAFE_005",
                    "finding_code": "ACTION_PLATFORM_BINDING_MISMATCH",
                    "expected": {"command": "MAV_CMD_DO_PARACHUTE"},
                    "observed": {"command": "CMD_LAND"},
                    "affected_elements": ["SafetyMonitor", "deployParachute"],
                }],
            },
            {
                "req_id": "REQ_SAFE_006",
                "contract_status": "READY",
                "pattern_ids": ["TriggeredFailsafeResponse"],
                "links": [],
                "conformance": "PASS",
                "findings": [],
            },
        ]
    }


def test_repair_packet_contains_only_failed_requirement_context():
    bundle = build_contract_bundle([_TARGET, _UNRELATED])
    packet = build_scoped_repair_packet(
        bundle,
        _report(),
        select_patterns(bundle),
        issues=[
            "[SEMANTIC-TRACE] REQ_SAFE_005 "
            "ACTION_PLATFORM_BINDING_MISMATCH"
        ],
    )

    assert packet["scope"]["req_ids"] == ["REQ_SAFE_005"]
    assert packet["scope"]["affected_elements"] == [
        "SafetyMonitor", "deployParachute"
    ]
    assert [item["req_id"] for item in packet["contracts"]] == [
        "REQ_SAFE_005"
    ]
    assert packet["contracts"][0]["source_text"] == _TARGET
    assert [item["req_id"] for item in packet["traces"]] == [
        "REQ_SAFE_005"
    ]
    assert packet["pattern_constraints"]
    assert packet["platform_bindings"][0]["response_concept"] == "deploy_parachute"
    assert packet["platform_bindings"][0]["canonical_commands"] == (
        "MAV_CMD_DO_PARACHUTE",
    )
    assert len(packet["packet_digest"]) == 64
    assert "REQ_SAFE_006" not in str(packet)


def test_repair_packet_is_empty_without_a_matching_semantic_failure():
    bundle = build_contract_bundle([_TARGET])
    packet = build_scoped_repair_packet(
        bundle,
        _report(),
        select_patterns(bundle),
        issues=["ordinary evaluator recommendation"],
    )

    # With no explicit semantic issue, the builder scopes to actual failures.
    assert packet["scope"]["req_ids"] == ["REQ_SAFE_005"]

    no_failure_report = {"traces": [{
        "req_id": "REQ_SAFE_005", "findings": [], "links": []
    }]}
    assert build_scoped_repair_packet(
        bundle, no_failure_report, select_patterns(bundle), issues=[]
    ) == {}
