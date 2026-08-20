"""Traceability-only verification usages for quantities the model calc set cannot evaluate.

Speed needs a drag/thrust model the estimator does not expose, range needs a
non-zero design cruise speed, altitude is a geofence CONFIG bound. Fabricating an
`assert` for these would invent physics; the emitter instead declares the
verification ROUTE in-model (requirement usage + verification usage whose doc names
the responsible tier), so the model states HOW every quantified requirement is
verified — evaluable or not.
"""
from __future__ import annotations

from src.dse.analysis_emitter import inject_endurance_analysis
from src.dse.physics_estimator import DesignInputs
from src.simulation.syntax_checker import check_syntax

_MODEL = """package Drone {
    part def PropulsionSystem { attribute rotorCount : Real; attribute rotorRadiusM : Real; }
    part def Hexa :> PropulsionSystem { attribute rotorCount : Real = 6.0; attribute rotorRadiusM : Real = 0.2032; }
    part propulsionSystem : Hexa;
}"""

_REQS = [
    "REQ-PERF-002: The system shall sustain continuous flight for a minimum of 25 minutes at maximum rated payload.",
    "REQ-CONS-003: The maximum take-off mass of the system shall not exceed 8.0 kg.",
    "REQ-PERF-003: The system shall achieve a forward airspeed of at least 18 m/s in still air.",
    "REQ-CONS-001: The system shall not exceed an operating altitude of 120 m above ground level.",
]

# cruise_speed_mps = 0 → range/speed are NOT evaluable in-model.
_DESIGN = DesignInputs(
    payload_mass_kg=1.5, battery_capacity_mah=16000.0, battery_cells=6,
    rotor_count=6, rotor_radius_m=0.2032, cruise_speed_mps=0.0,
)


def test_non_evaluable_quantities_get_traceability_verification_defs():
    out, ok = inject_endurance_analysis(_MODEL, _REQS, design=_DESIGN)
    assert ok and not check_syntax(out).has_errors
    # Evaluable closure unchanged: endurance/MTOW keep their asserts.
    assert "assert constraint enduranceMeetsReq" in out
    assert "assert constraint mtowWithinReq" in out
    # Non-evaluable quantities: verification usage + tier note, NO fabricated assert.
    assert "verification req_perf_003_check" in out
    assert "verification req_cons_001_check" in out
    assert "forward_flight tier" in out
    assert "geofence parameter consistency" in out
    assert "speedMeetsReq" not in out          # no invented speed assert
    assert "altitudeWithinReq" not in out      # no invented altitude assert


def test_evaluable_requirements_are_not_duplicated_as_traceability_defs():
    out, ok = inject_endurance_analysis(_MODEL, _REQS, design=_DESIGN)
    assert ok
    # Endurance already has an evaluable verification usage — exactly one.
    assert out.count("verification req_perf_002_check") == 1
