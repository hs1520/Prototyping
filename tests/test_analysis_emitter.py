"""Analysable SysML fragment emitter: mirror physics estimator into a calc def that
Syside Automator evaluates, closing the bare-attribute gap (constraint/analysis/satisfy).

Needs Syside for the Automator-evaluation tests; skipped if unavailable.
"""
from __future__ import annotations

import pytest

from src.dse.analysis_emitter import emit_endurance_analysis, endurance_calc_def
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
