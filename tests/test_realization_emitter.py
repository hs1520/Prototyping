import pytest

from src.dse.physics_estimator import DesignInputs
from src.realization.closure import close_the_loop
from src.realization.realization_emitter import emit_realization_package, inject_realization_analysis
from src.simulation.syntax_checker import check_syntax

from .realization_fixtures import catalog, frame

try:
    import syside  # noqa: F401
    _HAS_SYSIDE = True
except ImportError:
    _HAS_SYSIDE = False


def _report(req="REQ-PERF-002: endurance at least 15 minutes."):
    d = DesignInputs(1.0, 12000, 6, 4, 18 * 0.0254 / 2, 0.0)
    return close_the_loop(d, [], [req], catalog())


def test_emit_realization_package_syntax_passes():
    sysml, ok = emit_realization_package(_report())
    assert ok and not check_syntax(sysml).has_errors
    assert "package RealizationPackage" in sysml
    assert "RealizedEndurance(" in sysml
    assert "assert constraint realizationCloses0" in sysml


def test_inject_realization_analysis_rolls_back_on_bad_model():
    bad = "package Drone { part def A { "
    out, ok = inject_realization_analysis(bad, _report())
    assert not ok and out == bad


def test_inject_realization_analysis_into_valid_package():
    base = "package Drone { part def A; }"
    out, ok = inject_realization_analysis(base, _report())
    assert ok and out != base
    assert "part def RealizedDesign" in out
    assert not check_syntax(out).has_errors


def test_infeasible_report_still_emits_false_assert():
    d = DesignInputs(1.0, 12000, 6, 4, 18 * 0.0254 / 2, 0.0)
    rep = close_the_loop(d, [], ["REQ-PERF-002: endurance at least 15 minutes."],
                         catalog(frames=[frame(arms=6)]))
    assert rep.verdict == "INFEASIBLE_REALIZATION"
    sysml, ok = emit_realization_package(rep)
    assert ok
    assert "assert constraint realizationCloses { false }" in sysml


@pytest.mark.skipif(not _HAS_SYSIDE, reason="syside not installed")
def test_automator_realized_endurance_matches_python():
    import syside

    rep = _report()
    sysml, ok = emit_realization_package(rep)
    assert ok
    model, _ = syside.try_load_model(sysml_source=sysml)
    compiler = syside.Compiler()
    part = next(e for e in model.elements(syside.PartDefinition) if e.name == "RealizedDesign")
    feature = {x.name: x for x in part.features if getattr(x, "name", None)}["realizedEnduranceMin"]
    value, _ = compiler.evaluate_feature(feature, scope=part)
    assert value == pytest.approx(rep.chosen.metrics.endurance_min, rel=1e-6)
