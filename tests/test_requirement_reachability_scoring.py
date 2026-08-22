"""The evaluator's structural term is trace-first by design: it prefers the
plan-obligation pass rate over the untraced role heuristic, whose scenario
count scales with component richness and is not comparable across
configurations. The plan payload lives on the orchestrator runtime, and the
refinement engine read it from itself instead, so the requirement-traced
score was unset on every archived run and the fallback silently governed
every score. These tests pin the wiring."""
from __future__ import annotations

from types import SimpleNamespace

from src.agents.refinement import _RefinementEngine
from src.simulation.validator import SimulationResult

_MODEL = """package DeliveryUAV {
    item def StatusData;
    port def StatusPort { item payload : StatusData; }
    part def PerceptionSystem { out port sensorStatus : StatusPort; }
    part def SafetyMonitor { in port sensorStatus : StatusPort; }
    part perceptionSystem : PerceptionSystem;
    part safetyMonitor : SafetyMonitor;
    connect perceptionSystem.sensorStatus to safetyMonitor.sensorStatus;
}"""

_REQ = ("REQ-SAFE-004: The system shall not arm when any onboard sensor "
        "reports a failure.")

_PLAN = {
    "components": [
        {"name": "PerceptionSystem", "responsibility": "sensing",
         "requirements": ["REQ-SAFE-004"],
         "ports": [{"name": "sensorStatus", "direction": "out",
                    "type": "StatusPort"}],
         "attributes": []},
        {"name": "SafetyMonitor", "responsibility": "monitoring",
         "requirements": ["REQ-SAFE-004"],
         "ports": [{"name": "sensorStatus", "direction": "in",
                    "type": "StatusPort"}],
         "attributes": []},
    ],
    "connections": [
        {"source": {"component": "PerceptionSystem", "port": "sensorStatus"},
         "target": {"component": "SafetyMonitor", "port": "sensorStatus"},
         "item_type": "StatusData"},
    ],
    "requirement_realizations": [
        {"requirement_id": "REQ_SAFE_004",
         "realization_kind": "CAUSAL_PATH",
         "trigger_concept": "any onboard sensor reports a failure",
         "effect_concept": "shall not arm",
         "connection_path": [
             {"source_component": "PerceptionSystem",
              "source_port": "sensorStatus",
              "target_component": "SafetyMonitor",
              "target_port": "sensorStatus",
              "item_type": "StatusData"},
         ]},
    ],
}


def _engine(runtime):
    engine = _RefinementEngine.__new__(_RefinementEngine)
    object.__setattr__(engine, "_runtime", runtime)
    object.__setattr__(
        engine, "_simulation_runner",
        lambda text, name: SimulationResult(model_name=name),
    )
    return engine


def test_plan_payload_is_found_on_the_runtime():
    runtime = SimpleNamespace(_active_model_generation_plan=_PLAN)
    assert _engine(runtime)._active_plan_payload() is _PLAN


def test_engine_local_payload_takes_precedence():
    runtime = SimpleNamespace(_active_model_generation_plan={"components": []})
    engine = _engine(runtime)
    object.__setattr__(engine, "_active_model_generation_plan", _PLAN)
    assert engine._active_plan_payload() is _PLAN


def test_requirement_reachability_is_attached_from_the_runtime_plan():
    """Replay an archived pilot cell: with the plan read from the runtime,
    the requirement-traced score must be attached (it was None on every
    archived run while the engine read the plan from itself)."""
    import json
    import os
    import pytest

    cell = ("examples/output/pilot_n6_0c26731_20260821_2204/"
            "seed-0/R1-BBCTX")
    if not os.path.exists(f"{cell}/run_report.json"):
        pytest.skip("archived pilot cell not present")
    payload = json.load(open(f"{cell}/run_report.json"))[
        "whole_model_generation_plan"
    ]
    model = open(f"{cell}/shared_model_final.sysml").read()
    runtime = SimpleNamespace(_active_model_generation_plan=payload)
    result = _engine(runtime)._run_simulation(model, "DeliveryUAV")
    assert result.requirement_scenarios_total >= 1
    assert result.requirement_reachability_score is not None
    assert 0.0 <= result.requirement_reachability_score <= 1.0
