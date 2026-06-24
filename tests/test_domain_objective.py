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
    normalize_variation_ownership,
    objective_families,
    objective_names,
    requirement_targets,
    strip_inner_loop_attrs,
    variant_design_inputs,
)


_INCONSISTENT_POWER = """package D {
    port def Sig;
    part def PowerIface { out port p : Sig; }
    part def Power_4s :> PowerIface { attribute batteryCells : Real = 4.0; attribute batteryCapacityMah : Real = 3000.0; }
    part def Power_8s :> PowerIface { attribute batteryCells : Real = 8.0; }
    part def Power_12s :> PowerIface { attribute batteryCells : Real = 12.0; attribute batteryCapacityMah : Real = 11170.0; }
    part def Sys {
        variation part powerSystem : PowerIface { doc /* satisfies REQ-PERF-002 */
            variant part p4 : Power_4s; variant part p8 : Power_8s; variant part p12 : Power_12s; }
    }
}"""


def test_strip_inner_loop_attrs_uniform_interface():
    from src.dse.variation_parser import admitted, parse_variation_points
    pts = admitted(parse_variation_points(_INCONSISTENT_POWER))[0]
    out, notes = strip_inner_loop_attrs(_INCONSISTENT_POWER, pts)
    # every power variant now exposes the SAME interface: batteryCells only, no capacity
    for t in ("Power_4s", "Power_8s", "Power_12s"):
        assert variant_design_inputs(out, t) == {"battery_cells": float(t.split("_")[1][:-1])}
    assert any("batteryCapacityMah" in n for n in notes)


def test_strip_inner_loop_attrs_noop_when_already_uniform():
    model = _INCONSISTENT_POWER.replace(" attribute batteryCapacityMah : Real = 3000.0;", "") \
                               .replace(" attribute batteryCapacityMah : Real = 11170.0;", "")
    from src.dse.variation_parser import admitted, parse_variation_points
    pts = admitted(parse_variation_points(model))[0]
    out, notes = strip_inner_loop_attrs(model, pts)
    assert out == model and notes == []
from src.dse.variation_parser import admitted as _admitted
from src.dse.variation_parser import parse_variation_points as _parse


_OVERLAP = """package Drone {
    port def Sig;
    part def LiftIface { in port cmd : Sig; out port thrust : Sig; }
    part def Hexa_Prop :> LiftIface { attribute rotorCount : Real = 6.0; attribute rotorRadiusM : Real = 0.2; }
    part def Octo_Prop :> LiftIface { attribute rotorCount : Real = 8.0; attribute rotorRadiusM : Real = 0.15; }
    part def Frame_Hexa :> LiftIface { attribute rotorCount : Real = 6.0; attribute rotorRadiusM : Real = 0.165; }
    part def Frame_Octo :> LiftIface { attribute rotorCount : Real = 8.0; attribute rotorRadiusM : Real = 0.22; }
    part def Sys {
        variation part propulsionSystem : LiftIface { doc /* satisfies REQ-PERF-002 */
            variant part p_hexa : Hexa_Prop; variant part p_octo : Octo_Prop; }
        variation part airframe : LiftIface { doc /* satisfies REQ-CONS-003 */
            variant part f_hexa : Frame_Hexa; variant part f_octo : Frame_Octo; }
    }
}"""


def _pts(model):
    return _admitted(_parse(model))[0]


def test_normalize_strips_duplicate_field_from_non_owner():
    out, notes = normalize_variation_ownership(_OVERLAP, _pts(_OVERLAP))
    # rotor stays on propulsion (concern match), stripped from airframe variants
    assert variant_design_inputs(out, "Hexa_Prop") == {"rotor_count": 6.0, "rotor_radius_m": 0.2}
    assert variant_design_inputs(out, "Frame_Octo") == {}
    assert any("kept in 'propulsionSystem'" in n for n in notes)
    assert any("airframe' is now physics-inert" in n for n in notes)


def test_normalize_noop_without_overlap():
    out, notes = normalize_variation_ownership(_MODEL, _pts(_MODEL))
    assert out == _MODEL and notes == []


def test_normalize_first_declarer_fallback_without_concern_match():
    # neutral names so NO concern keyword matches either point → keep the first declarer
    model = """package Drone {
    port def Sig;
    part def Iface { in port cmd : Sig; out port thrust : Sig; }
    part def OptA1 :> Iface { attribute rotorCount : Real = 6.0; }
    part def OptA2 :> Iface { attribute rotorCount : Real = 8.0; }
    part def OptB1 :> Iface { attribute rotorCount : Real = 6.0; }
    part def OptB2 :> Iface { attribute rotorCount : Real = 8.0; }
    part def Sys {
        variation part groupA : Iface { doc /* satisfies REQ-PERF-002 */
            variant part a1 : OptA1; variant part a2 : OptA2; }
        variation part groupB : Iface { doc /* satisfies REQ-CONS-003 */
            variant part b1 : OptB1; variant part b2 : OptB2; }
    }
}"""
    out, notes = normalize_variation_ownership(model, _pts(model))
    assert any("first declarer" in n for n in notes)
    assert variant_design_inputs(out, "OptA1") == {"rotor_count": 6.0}   # first declarer keeps it
    assert variant_design_inputs(out, "OptB2") == {}                     # second stripped


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
