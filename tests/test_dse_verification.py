"""Layer 0+1 closed loop: DSE model → verification cases + .parm + L1.

The minimal deterministic no-flight loop of the DSE→SITL plan.
"""
from __future__ import annotations

from src.sitl.dse_verification import build_dse_verification, build_from_explore_result
from src.simulation.syntax_checker import check_syntax

_MODEL = """package Drone {
    port def Sig;
    part def Propulsion { out port thrust : Sig; }
    part def QuadProp :> Propulsion {
        attribute speedMps : Real = 18.0;
        attribute enduranceMinutes : Real = 32.0;
        attribute massKg : Real = 2.4;
    }
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
]


def test_loop_produces_cases_parm_and_l1():
    rep = build_dse_verification(_MODEL, _REQS)
    assert len(rep.verification_cases) == 3      # speed, time, mass
    assert rep.parm_lines == ["WPNAV_SPEED          1800.0"]   # 18 m/s × 100
    assert rep.l1_ok


def test_verification_model_is_syside_valid():
    rep = build_dse_verification(_MODEL, _REQS)
    assert not check_syntax(rep.verification_model).has_errors


def test_emergent_families_have_a_case_but_no_parm():
    rep = build_dse_verification(_MODEL, _REQS)
    # endurance & mass get a verification case (target recorded) ...
    assert "ReqPerf002TimeVerification" in rep.verification_cases
    assert "ReqCons001MassVerification" in rep.verification_cases
    # ... but never a settable L1 param (they're measured in L2)
    assert all("WPNAV_SPEED" in line or "speed" in line.lower() for line in rep.parm_lines)
    assert len(rep.parm_lines) == 1


def test_from_explore_result_dict():
    rep = build_from_explore_result({"model_sysml": _MODEL, "requirements": _REQS})
    assert rep.verification_cases and rep.parm_lines


def test_summary_is_readable():
    rep = build_dse_verification(_MODEL, _REQS)
    s = rep.summary()
    assert "verification case" in s and ".parm" in s


def test_orchestrator_artifact_hook():
    """The explore() Phase-7 hook returns a complete artifact dict."""
    from src.agents.orchestrator import Orchestrator
    art = Orchestrator._dse_verification_artifact(_MODEL, _REQS)
    assert art is not None
    assert art["verification_cases"] and art["parm_lines"] and art["l1_ok"]
    assert "verification_model_sysml" in art and "summary" in art


def test_orchestrator_artifact_none_when_nothing_to_verify():
    from src.agents.orchestrator import Orchestrator
    bare = "package P { part def X { attribute foo : Real = 1.0; } }"
    assert Orchestrator._dse_verification_artifact(bare, ["REQ-FUNC-001: do a thing."]) is None
