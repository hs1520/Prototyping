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


def test_default_real_catalog_closes_feasible_hexa_recommendation():
    # DSE-style hexa recommendation (18in rotors, 6S, modest payload) must realize on
    # the collected real catalog: Tarot X6 (hexa, 18in) + MN5008 (6S, 18in) + a 6S Tattu
    # pack — the meet-in-the-middle CLOSED demo on real manufacturer data.
    d = DesignInputs(1.5, 16000, 6, 6, 18 * 0.0254 / 2, 0.0)
    rep = close_the_loop(d, [], ["REQ-PERF-002: flight endurance of at least 25 minutes."],
                         DEFAULT_CATALOG)
    assert rep.verdict in {"CLOSED", "CLOSED_AFTER_RESIZE"}
    assert rep.chosen is not None
    assert rep.per_requirement and all(v.met for v in rep.per_requirement)


def test_default_real_catalog_quad_18in_is_honestly_infeasible():
    # A quad design with 18in rotors cannot realize: the only quad frame (Tarot 650
    # Sport) caps at 15in props while both combos carry 16/18in props. The nearest
    # combination is the hexa X6 frame (arms_match fails) or a quad frame with an
    # oversized prop (prop_fits fails) — either way the gap must be attributed to a
    # named structural check, not silently passed.
    d = DesignInputs(1.5, 16000, 6, 4, 18 * 0.0254 / 2, 0.0)
    rep = close_the_loop(d, [], ["REQ-PERF-002: flight endurance of at least 25 minutes."],
                         DEFAULT_CATALOG)
    assert rep.verdict == "INFEASIBLE_REALIZATION"
    assert rep.failed_checks
    assert all(ch.name in {"arms_match", "prop_fits"} for ch in rep.failed_checks)

