"""Tests for the requirement-target-driven domain objective (design-input edition).

Variants now declare SITL-settable DESIGN INPUTS (battery / mass / rotor / cruise
speed); the objective derives emergent metrics via the physics estimator and scores
them against requirement targets. This mirrors what SITL measures, enabling calibration.
"""
from __future__ import annotations

from src.dse.domain_objective import (
    architecture_design,
    architecture_objectives,
    endurance_target,
    objective_families,
    objective_names,
    requirement_targets,
    variant_design_inputs,
)


def test_endurance_target_ignores_latency_seconds():
    # a response-time req in SECONDS must not be mistaken for flight endurance (minutes):
    # the inner BO would otherwise size for ~1 unit. Regression for the >=1.0 constraint bug.
    reqs = [
        "REQ-FUNC-006: incorporate a waypoint within 1.0 second of command.",
        "REQ-PERF-002: sustain flight for a minimum of 25 minutes.",
    ]
    assert endurance_target(reqs) == 25.0


def test_endurance_target_zero_without_minutes_or_hours():
    assert endurance_target(["REQ-FUNC-006: respond within 1.0 second."]) == 0.0
from src.dse.variation_parser import admitted, parse_variation_points

_MODEL = """package Drone {
    part def QuadRotor { attribute batteryCapacityMah : Real = 12000.0; attribute massKg : Real = 0.3; attribute rotorRadiusM : Real = 0.16; attribute cruiseSpeedMps : Real = 16.0; }
    part def VtolWing  { attribute batteryCapacityMah : Real = 5000.0; attribute massKg : Real = 0.3; attribute rotorRadiusM : Real = 0.13; attribute cruiseSpeedMps : Real = 26.0; }
    part def Airframe {
        variation part liftArch {
            doc /* rationale: speed vs endurance; satisfies REQ-PERF-001, REQ-PERF-002 */
            variant part quad : QuadRotor;
            variant part vtol : VtolWing;
        }
    }
}"""
_REQS = ["REQ-PERF-001: cruise speed at least 20 m/s", "REQ-PERF-002: endurance at least 30 minutes"]


def test_requirement_targets_strips_id_and_uses_unit():
    t = requirement_targets(_REQS)
    assert ("speed", 20.0) in t["REQ-PERF-001"]
    assert all(v != 1.0 for _, v in t["REQ-PERF-001"])  # the "001" not parsed as a target
    assert ("time", 30.0) in t["REQ-PERF-002"]


def test_variant_design_inputs_extracted_by_field():
    d = variant_design_inputs(_MODEL, "QuadRotor")
    assert d["battery_capacity_mah"] == 12000.0
    assert d["payload_mass_kg"] == 0.3
    assert d["rotor_radius_m"] == 0.16
    assert d["cruise_speed_mps"] == 16.0


def test_architecture_design_merges_with_defaults():
    vps, _ = admitted(parse_variation_points(_MODEL))
    di = architecture_design(vps, {"liftArch": "quad"}, _MODEL)
    assert di.payload_mass_kg == 0.3 and di.battery_capacity_mah == 12000.0
    assert di.battery_cells == 4  # not declared → DESIGN_DEFAULTS


def test_objective_names_are_emergent_family_plus_cost():
    # speed is settable (L1), not a variant objective; only emergent endurance remains
    assert objective_families(_REQS) == ["time"]
    assert objective_names(_REQS) == ["time_sat", "cost_efficiency"]


def test_objectives_make_quad_and_vtol_nondominated():
    vps, _ = admitted(parse_variation_points(_MODEL))
    quad = architecture_objectives(vps, {"liftArch": "quad"}, _MODEL, _REQS)
    vtol = architecture_objectives(vps, {"liftArch": "vtol"}, _MODEL, _REQS)
    # estimator-derived trade-off: quad (big battery, heavy) wins endurance;
    # vtol (light) wins cost → neither dominates.
    assert quad["time_sat"] > vtol["time_sat"]
    assert vtol["cost_efficiency"] > quad["cost_efficiency"]


def test_within_requirement_bounds_rejects_over_spec_payload():
    from src.dse.domain_objective import within_requirement_bounds
    reqs = ["REQ-FUNC-003: transport payloads with a gross mass of up to 2.5 kg"]
    assert within_requirement_bounds({"payload_mass_kg": 2.5}, ["REQ-FUNC-003"], reqs)
    assert not within_requirement_bounds({"payload_mass_kg": 10.0}, ["REQ-FUNC-003"], reqs)
    # unrelated requirement (not in satisfies) → no bound applied
    assert within_requirement_bounds({"payload_mass_kg": 10.0}, [], reqs)
