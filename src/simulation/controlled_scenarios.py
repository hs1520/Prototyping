"""Fixed role-level structural scenarios for controlled model comparison.

Thesis baseline/evidence code: exercised by its own tests and invoked on
demand rather than wired into the runtime pipeline, so it is not dead code.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .exec_graph import build_exec_graph, shortest_path
from .extractor import extract_behavioral_graph
from .scenarios import classify_parts_by_role


CONTROLLED_SCENARIO_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class ControlledScenarioSpec:
    scenario_id: str
    source_role: str
    target_role: str
    requirement_ids: tuple[str, ...]
    tags: tuple[str, ...]


# The set is requirement-derived once and fixed across every experimental arm.
# Role resolution adapts names to a model; a missing role produces FAIL
# rather than dropping the scenario from the denominator.
CONTROLLED_SCENARIOS: tuple[ControlledScenarioSpec, ...] = (
    ControlledScenarioSpec(
        "S01_SENSOR_TO_CONTROLLER", "sensor", "controller",
        ("REQ_FUNC_001", "REQ_FUNC_002", "REQ_FUNC_003"),
        ("nominal", "sensing"),
    ),
    ControlledScenarioSpec(
        "S02_SAFETY_TO_CONTROLLER", "safety", "controller",
        ("REQ_SAFE_001", "REQ_SAFE_002", "REQ_SAFE_003", "REQ_SAFE_005"),
        ("safety", "emergency"),
    ),
    ControlledScenarioSpec(
        "S03_POWER_TO_SAFETY", "power", "safety",
        ("REQ_SAFE_001", "REQ_SAFE_002"), ("safety", "power"),
    ),
    ControlledScenarioSpec(
        "S04_POWER_TO_CONTROLLER", "power", "controller",
        ("REQ_PERF_002", "REQ_SAFE_001"), ("nominal", "power"),
    ),
    ControlledScenarioSpec(
        "S05_COMMS_TO_CONTROLLER", "comms", "controller",
        ("REQ_FUNC_004", "REQ_FUNC_006", "REQ_INTF_001"),
        ("nominal", "comms"),
    ),
    ControlledScenarioSpec(
        "S06_CONTROLLER_TO_COMMS", "controller", "comms",
        ("REQ_FUNC_008", "REQ_INTF_001"), ("nominal", "comms"),
    ),
    ControlledScenarioSpec(
        "S07_CONTROLLER_TO_ACTUATOR", "controller", "actuator",
        ("REQ_FUNC_005", "REQ_SAFE_006", "REQ_SAFE_008"),
        ("nominal", "actuation"),
    ),
)


def evaluate_controlled_scenarios(
    model_text: str, *, model_name: str = "model"
) -> dict[str, Any]:
    """Evaluate the same seven existential role paths for every model."""
    graph = extract_behavioral_graph(model_text)
    execution = build_exec_graph(graph)
    roles = classify_parts_by_role(graph)
    results = []
    for spec in CONTROLLED_SCENARIOS:
        sources = list(roles.get(spec.source_role, ()))
        targets = list(roles.get(spec.target_role, ()))
        best: list[str] | None = None
        for source in sources:
            for target in targets:
                path = shortest_path(execution, source, target)
                if path and (best is None or len(path) < len(best)):
                    best = path
        if not sources:
            failure_code = "MISSING_SOURCE_ROLE"
        elif not targets:
            failure_code = "MISSING_TARGET_ROLE"
        elif best is None:
            failure_code = "NO_DIRECTED_ROLE_PATH"
        else:
            failure_code = None
        results.append({
            **asdict(spec),
            "status": "PASS" if best is not None else "FAIL",
            "source_candidates": sources,
            "target_candidates": targets,
            "observed_path": best or [],
            "failure_code": failure_code,
        })
    passed = sum(item["status"] == "PASS" for item in results)
    return {
        "schema_version": CONTROLLED_SCENARIO_SCHEMA_VERSION,
        "artifact_type": "CONTROLLED_ROLE_SCENARIO_EVALUATION",
        "model_name": model_name,
        "scenario_set_fixed": True,
        "scenario_count": len(CONTROLLED_SCENARIOS),
        "role_assignments": roles,
        "results": results,
        "counts": {"PASS": passed, "FAIL": len(results) - passed},
        "pass_rate": passed / len(results) if results else None,
        "claim_boundary": (
            "Role paths measure structural reachability only; they do not "
            "replace requirement-level semantic traces or execution oracles."
        ),
    }
