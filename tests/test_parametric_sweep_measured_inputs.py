"""The parametric sweep supplies measured inputs instead of demanding literals.

Measured conflict this pins (authoritative run 00e4d333): the sweep required
a numeric initial value for the constrained attribute, while the typed
semantic binding convention requires exactly those runtime attributes to be
initialised from a reference chain into the measurement port — never a local
literal.  Every binding-bound ALWAYS constraint therefore failed behaviour
execution mechanically ("Cannot sweep 'currentX': no initial value found",
six identical failures).  The sweep now recognises reference-chain
initialisers as measured inputs and synthesises the start value inside the
valid region — recorded, never silent — while a genuinely missing
initialiser on a non-measured attribute remains the hard failure it was.
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


def test_reference_chain_initialiser_is_swept_as_a_measured_input():
    scenarios = _constraint_results(_model(
        "attribute currentCruiseAirspeed : Real = airspeed.payload.cruiseAirspeed;"
    ))
    scenario = next(
        item for item in scenarios if "airspeedConstraint" in item.name
    )
    assert scenario.passed, scenario.violations
    assert "measured_input_start_synthesized" in scenario.tags
    assert any("measured input" in line for line in scenario.timeline)


def test_a_missing_initialiser_on_a_plain_attribute_still_fails():
    scenarios = _constraint_results(_model(
        "attribute currentCruiseAirspeed : Real;"
    ))
    scenario = next(
        item for item in scenarios if "airspeedConstraint" in item.name
    )
    assert not scenario.passed
    assert any("no initial value" in v for v in scenario.violations)


def test_a_literal_initialiser_keeps_the_original_semantics():
    scenarios = _constraint_results(_model(
        "attribute currentCruiseAirspeed : Real = 20.0;"
    ))
    scenario = next(
        item for item in scenarios if "airspeedConstraint" in item.name
    )
    assert scenario.passed, scenario.violations
    assert "measured_input_start_synthesized" not in scenario.tags


def test_archived_models_execute_every_constraint_scenario():
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
