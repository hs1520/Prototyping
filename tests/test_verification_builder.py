"""SysML v2 verification-case generation for DSE-recommended models (layer 0).

Each quantified requirement becomes a standard verification case (requirement def +
usage + verification def whose objective `verify`s the usage) bound to the model's
system type. All emitted SysML must be Syside-valid.
"""
from __future__ import annotations

from src.dse.verification_builder import build_verification_cases, _find_system_type
from src.simulation.syntax_checker import check_syntax

_MODEL = """package Drone {
    port def Sig;
    part def Propulsion { out port thrust : Sig; }
    part def QuadProp :> Propulsion { attribute speedMps : Real = 20.0; attribute enduranceMinutes : Real = 35.0; }
    part def FC { in port t : Sig; }
    part def Airframe {
        part propulsionSystem : QuadProp;
        part flightController : FC;
        connect propulsionSystem.thrust to flightController.t;
    }
}"""

_REQS = [
    "REQ-PERF-001: cruise at least 18 m/s.",
    "REQ-PERF-002: endurance at least 30 min.",
    "REQ-CONS-001: takeoff mass at most 25 kg.",
    "REQ-FUNC-009: navigate autonomously.",  # no numeric target
]


def test_finds_assembly_as_system_type():
    assert _find_system_type(_MODEL) == "Airframe"  # owns the most nested part usages


def test_one_verification_case_per_quantified_family():
    _, names = build_verification_cases(_MODEL, _REQS)
    assert names == [
        "ReqPerf001SpeedVerification",
        "ReqPerf002TimeVerification",
        "ReqCons001MassVerification",
    ]


def test_generated_model_is_syside_valid():
    out, names = build_verification_cases(_MODEL, _REQS)
    assert names
    assert not check_syntax(out).has_errors


def test_verify_targets_requirement_usage_and_binds_system_subject():
    out, _ = build_verification_cases(_MODEL, _REQS)
    # verify references the usage (lowercase), not the def — Syside requires this
    assert "verify reqPerf001SpeedReq;" in out
    assert "subject s : Airframe;" in out
    assert "attribute target : Real = 18.0;" in out


def test_direction_is_perf_ge_cost_le():
    out, _ = build_verification_cases(_MODEL, _REQS)
    assert "speed >= 18.0" in out      # performance family: at-least
    assert "mass <= 25.0" in out       # cost family: at-most


def test_no_quantified_targets_is_noop():
    out, names = build_verification_cases(_MODEL, ["REQ-FUNC-009: navigate autonomously."])
    assert names == [] and out == _MODEL


_FLAT_MODEL = """package Drone {
    port def Sig;
    part def FlightController { in port s : Sig; out port m : Sig; }
    part def Propulsion { in port cmd : Sig; }
    part def Sensor { out port data : Sig; }
    part flightController : FlightController;
    part propulsion : Propulsion;
    part sensor : Sensor;
    connect sensor.data to flightController.s;
    connect flightController.m to propulsion.cmd;
}"""


def test_flat_package_assembly_still_finds_a_subject():
    # no wrapping assembly part def → fall back to the most-connected part's type
    assert _find_system_type(_FLAT_MODEL) == "FlightController"


def test_flat_package_assembly_generates_cases():
    out, names = build_verification_cases(_FLAT_MODEL, _REQS)
    assert names                                   # was 0 before the flat-assembly fix
    assert not check_syntax(out).has_errors
    assert "subject s : FlightController;" in out
