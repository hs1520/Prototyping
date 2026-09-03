from __future__ import annotations

from src.dse.variation_parser import (
    admitted,
    parse_variation_points,
    port_safe,
    resolve_model,
)
from src.simulation.syntax_checker import check_syntax

_MODEL = """package Drone {
    part def QuadRotor; part def HexaRotor;
    part def SingleBattery; part def DualBattery;
    part def Airframe {
        variation part propulsion {
            doc /* rationale: lift vs endurance; satisfies REQ-PERF-002, REQ-PERF-003 */
            variant part quad : QuadRotor;
            variant part hexa : HexaRotor;
        }
        variation part battery {
            doc /* rationale: endurance vs mass; satisfies REQ-SAFE-001 */
            variant part single : SingleBattery;
            variant part dual : DualBattery;
        }
        variation part unjustified {
            variant part a : QuadRotor;
            variant part b : HexaRotor;
        }
    }
}"""


def test_parses_points_and_variants():
    pts = {p.point_id: p for p in parse_variation_points(_MODEL)}
    assert set(pts) == {"propulsion", "battery", "unjustified"}
    assert pts["propulsion"].variant_names == ["quad", "hexa"]
    assert pts["propulsion"].type_of("hexa") == "HexaRotor"
    assert set(pts["propulsion"].requirements) == {"REQ-PERF-002", "REQ-PERF-003"}
    assert "lift vs endurance" in pts["propulsion"].rationale


def test_admission_rejects_unjustified():
    ok, bad = admitted(parse_variation_points(_MODEL))
    assert {p.point_id for p in ok} == {"propulsion", "battery"}
    assert {p.point_id for p in bad} == {"unjustified"}


def test_resolve_binds_variants():
    ok, _ = admitted(parse_variation_points(_MODEL))
    concrete = resolve_model(_MODEL, ok, {"propulsion": "hexa", "battery": "dual"})
    assert "part propulsion : HexaRotor;" in concrete
    assert "part battery : DualBattery;" in concrete
    assert "variation part propulsion" not in concrete
    assert "variation part unjustified" in concrete
    assert not check_syntax(concrete).has_errors


def test_is_objective_rule():
    pts = {p.point_id: p for p in parse_variation_points(_MODEL)}
    assert pts["propulsion"].is_objective()
    assert not pts["unjustified"].is_objective()


def test_port_safe_requires_shared_interface():
    safe = """package P {
        port def Sig;
        part def Iface { in port c : Sig; }
        part def A :> Iface;
        part def B :> Iface;
        part def Sub { variation part x { variant part a : A; variant part b : B; } }
    }"""
    p = parse_variation_points(safe)[0]
    assert port_safe(p, safe)
    unsafe = safe.replace(":> Iface", "")
    pu = parse_variation_points(unsafe)[0]
    assert not port_safe(pu, unsafe)
