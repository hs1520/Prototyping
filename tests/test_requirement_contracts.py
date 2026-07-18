from __future__ import annotations

from src.prototyping.requirement_contracts import (
    analyse_requirements,
    obstacle_avoidance_contract,
)
from examples.run_realization_report import require_authoritative_requirement_contracts


def test_obstacle_contract_preserves_initiation_vs_clearance_semantics():
    original = (
        "REQ-FUNC-002: The system shall detect potential collision threats within a "
        "15-metre sensor range and initiate an avoidance manoeuvre before the "
        "separation distance falls below 5 metres."
    )

    contract = obstacle_avoidance_contract(original)

    assert contract is not None
    assert contract.detection_range_m == 15.0
    assert contract.response_threshold_m == 5.0
    assert contract.minimum_separation_m is None
    assert contract.response_semantics == "initiation"
    assert not contract.contract_ready
    assert any("closing" in gap for gap in contract.semantic_gaps)
    assert any("field-of-view" in gap for gap in contract.semantic_gaps)
    assert any("does not require" in note for note in contract.semantic_notes)


def test_complete_obstacle_contract_carries_reproducible_envelope_and_oracle():
    requirement = (
        "REQ-FUNC-002: When approaching a stationary collision threat directly ahead "
        "within the forward sensor field of view at a closing speed no greater than "
        "1.5 m/s, the system shall execute an avoidance manoeuvre following threat "
        "detection no later than 15 metres, maintaining an airframe-to-obstacle "
        "separation of at least 5 metres."
    )

    contract = obstacle_avoidance_contract(requirement)

    assert contract is not None and contract.contract_ready
    assert contract.detection_range_m == 15.0
    assert contract.max_closing_speed_mps == 1.5
    assert contract.minimum_separation_m == 5.0
    assert contract.geometry == "forward_sensor_axis"
    assert contract.response_semantics == "maintain_clearance"
    assert analyse_requirements([requirement])["verification_ready"]
    assert require_authoritative_requirement_contracts([requirement])["verification_ready"]


def test_authoritative_gate_rejects_incomplete_high_fidelity_contract():
    import pytest

    with pytest.raises(RuntimeError, match="incomplete verification contracts"):
        require_authoritative_requirement_contracts([
            "REQ-FUNC-002: The system shall detect obstacles and avoid collisions."
        ])
