from src.dse.physics_estimator import DesignInputs
from src.realization.closure import close_the_loop

from .realization_fixtures import catalog, frame, pack


def _design(capacity=16000, payload=1.0):
    return DesignInputs(payload, capacity, 6, 4, 18 * 0.0254 / 2, 0.0)


def test_closed_verdict():
    rep = close_the_loop(_design(), [], ["REQ-PERF-002: endurance at least 15 minutes."], catalog())
    assert rep.verdict == "CLOSED"
    assert rep.chosen is not None
    assert all(v.met for v in rep.per_requirement)


def test_closed_after_resize_verdict():
    cat = catalog(packs=[
        pack("small", capacity=8000, mass=900),
        pack("big", capacity=20000, mass=2100),
    ])
    rep = close_the_loop(_design(capacity=8000), [], ["REQ-PERF-002: endurance at least 40 minutes."], cat)
    assert rep.verdict == "CLOSED_AFTER_RESIZE"
    assert "small" in rep.resize_note and "big" in rep.resize_note


def test_infeasible_verdict_has_failed_checks_or_gap():
    rep = close_the_loop(_design(), [], ["REQ-PERF-002: endurance at least 80 minutes."], catalog())
    assert rep.verdict == "INFEASIBLE_REALIZATION"
    assert rep.failed_checks


def test_no_feasible_candidate_reports_interface_failures():
    rep = close_the_loop(_design(), [], ["REQ-PERF-002: endurance at least 15 minutes."],
                         catalog(frames=[frame(arms=6)]))
    assert rep.verdict == "INFEASIBLE_REALIZATION"
    assert any(ch.name == "arms_match" for ch in rep.failed_checks)


def test_rank_skipped_when_fewer_than_three_candidates():
    rep = close_the_loop(_design(), [(_design(), {})], ["REQ-PERF-002: endurance at least 15 minutes."],
                         catalog())
    assert rep.rank_preservation["n"] == 1.0
    assert any("fewer than 3" in n for n in rep.notes)
