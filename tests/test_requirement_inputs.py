from __future__ import annotations

import pytest

from src.prototyping.requirement_inputs import (
    build_frozen_requirement_set,
    requirement_change_impact,
    resolve_frozen_requirement_set,
)


REQ = (
    "REQ-SAFE-005: Critical propulsion failure shall deploy the parachute "
    "within 0.5 seconds."
)


def test_frozen_artifact_round_trips():
    artifact = build_frozen_requirement_set([REQ], name="option2")

    requirements, canonical = resolve_frozen_requirement_set(artifact)

    assert requirements == [REQ]
    assert canonical["requirement_set_digest"] == artifact["requirement_set_digest"]


def test_graph_normalises_ids():
    prerequisite = "REQ-INTF-001: The system shall receive navigation data."
    artifact = build_frozen_requirement_set(
        [REQ, prerequisite],
        dependencies=[{"from": "REQ-SAFE-005", "to": "REQ-INTF-001"}],
    )

    _, canonical = resolve_frozen_requirement_set(artifact)

    assert canonical["dependency_graph"]["edges"] == [
        {"from": "REQ_SAFE_005", "to": "REQ_INTF_001"}
    ]


def test_graph_rejects_bad_endpoints():
    with pytest.raises(ValueError, match="unknown endpoint"):
        build_frozen_requirement_set(
            [REQ], dependencies=[{"from": "REQ-SAFE-005", "to": "REQ-X-999"}]
        )
    with pytest.raises(ValueError, match="self-referential"):
        build_frozen_requirement_set(
            [REQ], dependencies=[{"from": "REQ-SAFE-005", "to": "REQ-SAFE-005"}]
        )


def test_change_invalidates_dependents():
    req_a = "REQ-FUNC-001: The system shall perform A."
    req_b = "REQ-FUNC-002: The system shall perform B."
    req_c = "REQ-FUNC-003: The system shall perform C."
    previous = build_frozen_requirement_set(
        [req_a, req_b, req_c],
        dependencies=[{"from": "REQ-FUNC-002", "to": "REQ-FUNC-001"}],
    )
    current = build_frozen_requirement_set(
        [req_a + " Changed.", req_b, req_c],
        dependencies=[{"from": "REQ-FUNC-002", "to": "REQ-FUNC-001"}],
    )

    impact = requirement_change_impact(previous, current)

    assert impact["directly_changed_requirement_ids"] == ["REQ_FUNC_001"]
    assert impact["invalidated_requirement_ids"] == [
        "REQ_FUNC_001", "REQ_FUNC_002",
    ]
    assert impact["evidence_invalidations"] == [
        {"req_id": "REQ_FUNC_001", "status": "STALE"},
        {"req_id": "REQ_FUNC_002", "status": "STALE"},
    ]


def test_edge_change_propagates():
    reqs = [
        "REQ-FUNC-001: The system shall perform A.",
        "REQ-FUNC-002: The system shall perform B.",
        "REQ-FUNC-003: The system shall perform C.",
    ]
    previous = build_frozen_requirement_set(
        reqs,
        dependencies=[{"from": "REQ-FUNC-003", "to": "REQ-FUNC-002"}],
    )
    current = build_frozen_requirement_set(
        reqs,
        dependencies=[
            {"from": "REQ-FUNC-002", "to": "REQ-FUNC-001"},
            {"from": "REQ-FUNC-003", "to": "REQ-FUNC-002"},
        ],
    )

    impact = requirement_change_impact(previous, current)

    assert impact["dependency_edges_changed"] is True
    assert impact["directly_changed_requirement_ids"] == ["REQ_FUNC_002"]
    assert impact["invalidated_requirement_ids"] == [
        "REQ_FUNC_002", "REQ_FUNC_003",
    ]


def test_drone_input_carries_edges():
    from examples.drone_system_v2 import DRONE_FROZEN_REQUIREMENTS

    graph = DRONE_FROZEN_REQUIREMENTS["dependency_graph"]

    assert graph["artifact_type"] == "REQUIREMENT_DEPENDENCY_GRAPH"
    assert len(graph["edges"]) == 13
    assert {"from": "REQ_FUNC_001", "to": "REQ_INTF_002"} in graph["edges"]
