"""Layer 4: DSE design inputs → ArduPilot SITL configuration + predicted baseline.

Settable inputs (battery/rotor/cruise) map to SITL params; emergent mass is NOT set
(fixed frame model) — the caveat the calibration layer will surface.
"""
from __future__ import annotations

from src.sitl.dse_sitl_params import (
    design_to_sitl_parm,
    model_design_inputs,
    predicted_emergent,
    sitl_plan,
)

_MODEL = """package Drone {
    part def Prop { attribute rotorCount : Real = 6.0; attribute rotorRadiusM : Real = 0.18; attribute cruiseSpeedMps : Real = 18.0; }
    part def Pow  { attribute batteryCapacityMah : Real = 12000.0; attribute batteryCells : Real = 6.0; }
    part def Pay  { attribute massKg : Real = 0.4; }
}"""


def test_model_design_inputs_merge_across_components():
    d = model_design_inputs(_MODEL)
    assert d.battery_capacity_mah == 12000.0 and d.battery_cells == 6
    assert d.rotor_count == 6 and d.cruise_speed_mps == 18.0
    assert d.payload_mass_kg == 0.4


def test_battery_maps_to_batt_capacity():
    parm = design_to_sitl_parm(model_design_inputs(_MODEL))
    assert any(line.startswith("BATT_CAPACITY") and "12000" in line for line in parm)


def test_rotor_count_maps_to_frame_class():
    parm = design_to_sitl_parm(model_design_inputs(_MODEL))
    assert any(line.startswith("FRAME_CLASS") and line.strip().endswith("2") for line in parm)  # hexa


def test_cruise_maps_to_wpnav_speed_cm_s():
    parm = design_to_sitl_parm(model_design_inputs(_MODEL))
    assert any(line.startswith("WPNAV_SPEED") and "1800" in line for line in parm)  # 18 m/s × 100


def test_mass_is_not_a_sitl_param():
    parm = " ".join(design_to_sitl_parm(model_design_inputs(_MODEL)))
    assert "MASS" not in parm.upper()  # emergent mass can't be set in SITL


def test_predicted_emergent_baseline_present():
    p = predicted_emergent(model_design_inputs(_MODEL))
    assert p["endurance_min"] > 0 and p["total_mass_kg"] > 0


def test_sitl_plan_bundles_parm_prediction_and_caveat():
    plan = sitl_plan(_MODEL)
    assert plan["sitl_parm"] and plan["predicted"]
    assert "mass is emergent" in plan["caveat"]
