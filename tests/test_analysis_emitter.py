"""Analysable SysML fragment emitter: mirror physics estimator into a calc def that
Syside Automator evaluates, closing the bare-attribute gap (constraint/analysis/satisfy).

Needs Syside for the Automator-evaluation tests; skipped if unavailable.
"""
from __future__ import annotations

import pytest

from src.dse.analysis_emitter import (
    emit_endurance_analysis, endurance_calc_def, inject_endurance_analysis,
)
from src.dse.physics_estimator import DesignInputs, endurance_min
from src.simulation.syntax_checker import check_syntax

try:
    import syside  # noqa: F401
    _HAS_SYSIDE = True
except Exception:
    _HAS_SYSIDE = False

_D = DesignInputs(0.5, 5000, 4, 4, 0.13)


def test_emitted_fragment_parses_and_has_closure_elements():
    sysml, ok = emit_endurance_analysis(_D, target_min=10.0)
    assert ok and not check_syntax(sysml).has_errors
    assert "calc def Endurance" in sysml            # analysis relation
    assert "assert constraint enduranceMeetsReq" in sysml  # constraint
    assert "satisfy req_perf_002" in sysml          # traceability
    assert "Endurance(5000.0, 4.0, 4.0, 0.13, 0.5)" in sysml  # variant params wired in


def test_calc_def_uses_estimator_constants():
    # single source of truth: constants come from physics_estimator
    from src.dse.physics_estimator import FOM, ENERGY_DENSITY_WH_KG
    cd = endurance_calc_def()
    assert str(FOM) in cd and str(ENERGY_DENSITY_WH_KG) in cd


def _eval(sysml, part, feature):
    import syside
    c = syside.Compiler()
    model, _ = syside.try_load_model(sysml_source=sysml)
    p = next(e for e in model.elements(syside.PartDefinition) if e.name == part)
    f = {x.name: x for x in p.features if getattr(x, "name", None)}[feature]
    val, _ = c.evaluate_feature(f, scope=p)
    return val


@pytest.mark.skipif(not _HAS_SYSIDE, reason="syside not installed")
def test_automator_eval_matches_python_estimator():
    sysml, _ = emit_endurance_analysis(_D, target_min=10.0)
    val = _eval(sysml, "AnalyzedDesign", "enduranceMin")
    assert abs(val - endurance_min(_D)) < 0.05   # SysML analysis == Python physics (no drift)


@pytest.mark.skipif(not _HAS_SYSIDE, reason="syside not installed")
def test_constraint_reflects_reachability():
    reachable, _ = emit_endurance_analysis(_D, target_min=10.0)   # ~12.9min ≥ 10 → met
    unreachable, _ = emit_endurance_analysis(_D, target_min=90.0)  # can't reach 90 → not met
    # add an evaluable boolean to each and check it
    import syside
    c = syside.Compiler()
    for sysml, target, expect in [(reachable, 10.0, True), (unreachable, 90.0, False)]:
        model, _ = syside.try_load_model(sysml_source=sysml)
        p = next(e for e in model.elements(syside.PartDefinition) if e.name == "AnalyzedDesign")
        f = {x.name: x for x in p.features if getattr(x, "name", None)}["enduranceMin"]
        val, _ = c.evaluate_feature(f, scope=p)
        assert (val >= target) is expect


# --- inject_endurance_analysis: wire the closure into the real DSE product model ---

_DEFS = """
    part def PropulsionSystem { attribute rotorCount : Real; attribute rotorRadiusM : Real; }
    part def PowerSystem { attribute batteryCells : Real; }
    part def PayloadSystem { attribute massKg : Real; }
    part def Hexa :> PropulsionSystem { attribute rotorCount : Real = 6.0; attribute rotorRadiusM : Real = 0.1524; }
    part def Power12 :> PowerSystem { attribute batteryCells : Real = 12.0; }
    part def PayLight :> PayloadSystem { attribute massKg : Real = 2.5; }"""
_FLAT = (f"package Drone {{{_DEFS}\n    part propulsionSystem : Hexa;\n"
         f"    part powerSystem : Power12;\n    part payloadSystem : PayLight;\n}}")
_NESTED = (f"package Drone {{{_DEFS}\n    part def DeliveryDrone {{\n"
           f"        part propulsionSystem : Hexa;\n        part powerSystem : Power12;\n"
           f"        part payloadSystem : PayLight;\n    }}\n}}")
_EREQ = ["REQ-PERF-002: flight endurance of at least 25 min."]


def test_inject_flat_wraps_and_refs_chosen_variants():
    out, ok = inject_endurance_analysis(_FLAT, _EREQ, capacity_mah=18000.0)
    assert ok and not check_syntax(out).has_errors
    assert "part def DseEnduranceAnalysis" in out               # flat → wrapper part def
    assert "powerSystem.batteryCapacityMah" in out              # constraint refs the variant attrs
    assert "propulsionSystem.rotorCount" in out
    assert "batteryCapacityMah : Real = 18000.0" in out         # capacity written into CHOSEN power
    assert "assert constraint enduranceMeetsReq" in out and "satisfy req_perf_002" in out


def test_inject_nested_goes_into_root_without_wrapper():
    out, ok = inject_endurance_analysis(_NESTED, _EREQ, capacity_mah=18000.0)
    assert ok and not check_syntax(out).has_errors
    assert "part def DseEnduranceAnalysis" not in out           # nested → straight into the root
    assert "propulsionSystem.rotorCount" in out


def test_inject_noop_without_endurance_target():
    out, ok = inject_endurance_analysis(_FLAT, ["REQ-FUNC-001: navigate autonomously."])
    assert not ok and out == _FLAT


@pytest.mark.skipif(not _HAS_SYSIDE, reason="syside not installed")
def test_inject_automator_eval_matches_python():
    for model, part in ((_FLAT, "DseEnduranceAnalysis"), (_NESTED, "DeliveryDrone")):
        out, _ = inject_endurance_analysis(model, _EREQ, capacity_mah=18000.0)
        val = _eval(out, part, "enduranceMin")
        assert abs(val - endurance_min(DesignInputs(2.5, 18000, 12, 6, 0.1524))) < 0.05
