"""The parametric sweep supplies measured inputs instead of demanding literals.

The sweep required a numeric initial value while the typed binding convention
initialises those runtime attributes from a reference chain into the measurement
port, so every binding-bound ALWAYS constraint failed behaviour execution
("Cannot sweep 'currentX': no initial value found", six times in run 00e4d333).
Reference-chain initialisers now count as measured inputs and the start value is
synthesised inside the valid region and recorded; a missing initialiser on a
non-measured attribute stays a hard failure.
"""
from __future__ import annotations

from pathlib import Path

from src.simulation.behavioral_sim import run_behavioral_simulation

_REPO = Path(__file__).resolve().parents[1]


def _constraint_results(text: str):
    result = run_behavioral_simulation(text, model_name="AutonomousDrone")
    return [
        item for item in result.scenario_results
        if "constraint" in item.name
    ]


def _model(initializer_line: str) -> str:
    return f"""package P {{
    item def AirspeedData {{
        attribute cruiseAirspeed : Real;
    }}
    port def AirspeedPort {{
        in item payload : AirspeedData;
    }}
    part def Controller {{
        in port airspeed : AirspeedPort;
        {initializer_line}
        attribute minCruiseAirspeed : Real = 18.0;
        // PLAN-CONSTRAINT airspeedConstraint provenance=FROZEN_REQUIREMENT activation=ALWAYS verification=PARAMETRIC_SWEEP
        assert constraint airspeedConstraint {{
            currentCruiseAirspeed >= minCruiseAirspeed
        }}
    }}
}}"""


def test_reference_chain_swept():
    scenarios = _constraint_results(_model(
        "attribute currentCruiseAirspeed : Real = airspeed.payload.cruiseAirspeed;"
    ))
    scenario = next(
        item for item in scenarios if "airspeedConstraint" in item.name
    )
    assert scenario.passed, scenario.violations
    assert "measured_input_start_synthesized" in scenario.tags
    assert any("measured input" in line for line in scenario.timeline)


def test_missing_initialiser_fails():
    scenarios = _constraint_results(_model(
        "attribute currentCruiseAirspeed : Real;"
    ))
    scenario = next(
        item for item in scenarios if "airspeedConstraint" in item.name
    )
    assert not scenario.passed
    assert any("no initial value" in v for v in scenario.violations)


def test_literal_initialiser_kept():
    scenarios = _constraint_results(_model(
        "attribute currentCruiseAirspeed : Real = 20.0;"
    ))
    scenario = next(
        item for item in scenarios if "airspeedConstraint" in item.name
    )
    assert scenario.passed, scenario.violations
    assert "measured_input_start_synthesized" not in scenario.tags


def test_archived_models_all_execute():
    fixtures = {
        "00e4d333": (
            _REPO / "examples/output/runs"
            / "00e4d333-d87a-4f33-bc5c-b7768fdfeb79/final_model.sysml",
            6,
        ),
        "pilot4": (
            _REPO / "experiments/ablation/results"
            / "20260829_174841_pilot4/runs/FULL_seed0.final.sysml",
            0,
        ),
    }
    for label, (path, expected_synthesized) in fixtures.items():
        scenarios = _constraint_results(path.read_text())
        failed = [item.name for item in scenarios if not item.passed]
        assert failed == [], f"{label}: {failed}"
        synthesized = sum(
            1 for item in scenarios
            if "measured_input_start_synthesized" in (item.tags or [])
        )
        assert synthesized == expected_synthesized, label
