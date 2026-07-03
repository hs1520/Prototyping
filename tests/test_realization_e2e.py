from src.dse.physics_estimator import DesignInputs
from src.realization.catalog import DEFAULT_CATALOG
from src.realization.closure import close_the_loop

from .realization_fixtures import catalog


def test_synthetic_e2e_can_close():
    d = DesignInputs(1.0, 12000, 6, 4, 18 * 0.0254 / 2, 0.0)
    rep = close_the_loop(d, [(d, {})], ["REQ-PERF-002: endurance at least 15 minutes."], catalog())
    assert rep.verdict == "CLOSED"
    assert rep.chosen is not None
    assert rep.per_requirement and all(v.met for v in rep.per_requirement)


def test_honesty_regression_25min_at_2_5kg_is_not_false_green():
    d = DesignInputs(2.5, 12000, 6, 4, 18 * 0.0254 / 2, 0.0)
    reqs = [
        "REQ-PERF-002: flight endurance of at least 25 minutes at maximum rated payload.",
        "REQ-FUNC-003: transport payloads of up to 2.5 kg.",
    ]
    rep = close_the_loop(d, [(d, {})], reqs, catalog())
    assert rep.verdict == "INFEASIBLE_REALIZATION"
    assert rep.per_requirement
    assert any(v.family == "time" and not v.met for v in rep.per_requirement)


def test_default_real_catalog_reports_gap_until_packs_and_frames_are_collected():
    d = DesignInputs(2.5, 16000, 6, 4, 18 * 0.0254 / 2, 0.0)
    rep = close_the_loop(d, [], ["REQ-PERF-002: flight endurance of at least 25 minutes."],
                         DEFAULT_CATALOG)
    assert rep.verdict == "INFEASIBLE_REALIZATION"
    assert any(ch.name in {"catalog_packs", "catalog_frames"} for ch in rep.failed_checks)

