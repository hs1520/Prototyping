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


def test_default_real_catalog_closes_4s_quad_after_catalog_extension():
    # The 4S extension gives the matcher a structurally consistent quad path:
    # Tarot 650 Sport + MN3508/P15x5 4S + Tattu 4S pack.
    d = DesignInputs(0.2, 1300, 4, 4, 15 * 0.0254 / 2, 0.0)
    rep = close_the_loop(d, [], ["REQ-PERF-002: flight endurance of at least 5 minutes."],
                         DEFAULT_CATALOG)
    assert rep.verdict in {"CLOSED", "CLOSED_AFTER_RESIZE"}
    assert rep.chosen is not None
    assert rep.chosen.rd.combo.cells == 4
    assert rep.chosen.rd.pack.cells == 4
    assert rep.chosen.rd.frame.arms == 4


def test_default_real_catalog_closes_feasible_octo_after_x8_collection():
    # The traced X8 frame removes the former octo structural gap. A feasible 6S octo
    # should now match X8 + MN3508/P15x5 6S + Tattu 6S and close honestly.
    d = DesignInputs(1.5, 16000, 6, 8, 15 * 0.0254 / 2, 0.0)
    rep = close_the_loop(d, [], ["REQ-PERF-002: flight endurance of at least 25 minutes."],
                         DEFAULT_CATALOG)
    assert rep.verdict in {"CLOSED", "CLOSED_AFTER_RESIZE"}
    assert rep.chosen is not None
    assert rep.chosen.rd.frame.arms == 8
    assert rep.chosen.rd.frame.name.startswith("Tarot X8")
    assert rep.per_requirement and all(v.met for v in rep.per_requirement)
