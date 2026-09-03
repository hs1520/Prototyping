"""Layer 4: DSE design inputs -> ArduPilot SITL configuration + predicted baseline.

Settable inputs (battery/rotor/cruise) map to SITL params; emergent mass is not
set (fixed frame model), a caveat the calibration layer surfaces.
"""
from __future__ import annotations

from src.sitl.parameter_projection import (
    design_inputs_from_model,
    design_parm_lines,
    merge_base_parameters,
    project_model_parameters,
)

_MODEL = """package Drone {
    part def Prop { attribute rotorCount : Real = 6.0; attribute rotorRadiusM : Real = 0.18; attribute cruiseSpeedMps : Real = 18.0; }
    part def Pow  { attribute batteryCapacityMah : Real = 12000.0; attribute batteryCells : Real = 6.0; }
    part def Pay  { attribute massKg : Real = 0.4; }
}"""


def test_design_inputs_merge():
    d = design_inputs_from_model(_MODEL)
    assert d.battery_capacity_mah == 12000.0 and d.battery_cells == 6
    assert d.rotor_count == 6 and d.cruise_speed_mps == 18.0
    assert d.payload_mass_kg == 0.4


def test_battery_maps_to_capacity():
    parm = design_parm_lines(design_inputs_from_model(_MODEL))
    assert any(line.startswith("BATT_CAPACITY") and "12000" in line for line in parm)


def test_rotor_count_maps_frame_class():
    parm = design_parm_lines(design_inputs_from_model(_MODEL))
    assert any(line.startswith("FRAME_CLASS") and line.strip().endswith("2") for line in parm)


def test_cruise_maps_wpnav_speed():
    parm = design_parm_lines(design_inputs_from_model(_MODEL))
    assert any(line.startswith("WPNAV_SPEED") and "1800" in line for line in parm)


def test_mass_not_a_param():
    parm = " ".join(design_parm_lines(design_inputs_from_model(_MODEL)))
    assert "MASS" not in parm.upper()


def test_predicted_baseline_present():
    p = project_model_parameters(_MODEL).predicted
    assert p["endurance_min"] > 0 and p["total_mass_kg"] > 0


def test_plan_bundles_parm_and_caveat():
    plan = project_model_parameters(_MODEL)
    assert plan.parm_lines and plan.predicted
    assert "mass is emergent" in plan.caveat


def test_defaults_merge_without_override():
    design = design_inputs_from_model(_MODEL)
    lines = design_parm_lines(
        design,
        base_params={"FRAME_CLASS": 1, "ARMING_CHECK": 0},
    )
    assert sum(line.startswith("FRAME_CLASS") for line in lines) == 1
    assert next(line for line in lines if line.startswith("FRAME_CLASS")).endswith("2")
    assert any(line.startswith("ARMING_CHECK") for line in lines)

    merged = merge_base_parameters(
        "FRAME_CLASS 2\n# ARMING_CHECK mentioned only in prose",
        {"FRAME_CLASS": 1, "ARMING_CHECK": 0},
    )
    assert merged.count("FRAME_CLASS") == 1
    assert "ARMING_CHECK                   0  # base SITL param" in merged
