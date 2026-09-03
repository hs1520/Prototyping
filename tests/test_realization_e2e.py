from src.dse.physics_estimator import DesignInputs
from src.realization.catalog import DEFAULT_CATALOG, U7_V2_KV490_17x58_4S
from src.realization.closure import close_the_loop
from src.realization.matcher import match

from .realization_fixtures import catalog


def test_synthetic_e2e_can_close():
    d = DesignInputs(1.0, 12000, 6, 4, 18 * 0.0254 / 2, 0.0)
    rep = close_the_loop(d, [(d, {})], ["REQ-PERF-002: endurance at least 15 minutes."], catalog())
    assert rep.verdict == "CLOSED"
    assert rep.chosen is not None
    assert rep.per_requirement and all(v.met for v in rep.per_requirement)


def test_25min_at_2_5kg_not_green():
    d = DesignInputs(2.5, 12000, 6, 4, 18 * 0.0254 / 2, 0.0)
    reqs = [
        "REQ-PERF-002: flight endurance of at least 25 minutes at maximum rated payload.",
        "REQ-FUNC-003: transport payloads of up to 2.5 kg.",
    ]
    rep = close_the_loop(d, [(d, {})], reqs, catalog())
    assert rep.verdict == "INFEASIBLE_REALIZATION"
    assert rep.per_requirement
    assert any(v.family == "time" and not v.met for v in rep.per_requirement)


def test_catalog_closes_hexa():
    # DSE-style hexa recommendation (18in rotors, 6S, modest payload) realizes on the
    # collected real catalog: Tarot X6 (hexa, 18in) + MN5008 + a 6S Tattu pack, the
    # meet-in-the-middle CLOSED case.
    d = DesignInputs(1.5, 16000, 6, 6, 18 * 0.0254 / 2, 0.0)
    rep = close_the_loop(d, [], ["REQ-PERF-002: flight endurance of at least 25 minutes."],
                         DEFAULT_CATALOG)
    assert rep.verdict in {"CLOSED", "CLOSED_AFTER_RESIZE"}
    assert rep.chosen is not None
    assert rep.per_requirement and all(v.met for v in rep.per_requirement)


def test_catalog_closes_4s_quad():
    d = DesignInputs(0.2, 1300, 4, 4, 15 * 0.0254 / 2, 0.0)
    rep = close_the_loop(d, [], ["REQ-PERF-002: flight endurance of at least 5 minutes."],
                         DEFAULT_CATALOG)
    assert rep.verdict in {"CLOSED", "CLOSED_AFTER_RESIZE"}
    assert rep.chosen is not None
    assert rep.chosen.rd.combo.cells == 4
    assert rep.chosen.rd.pack.cells == 4
    assert rep.chosen.rd.frame.arms == 4


def test_4s_quad_snaps_endurance_pack():
    # A1 regression: the 4S catalog overlaps the inner BO capacity axis, not only the
    # small R-Line racing packs, so an 18min request snaps to the traced 5200mAh 4S
    # pack.
    d = DesignInputs(0.2, 5200, 4, 4, 15 * 0.0254 / 2, 0.0)
    rep = close_the_loop(d, [], ["REQ-PERF-002: flight endurance of at least 18 minutes."],
                         DEFAULT_CATALOG)
    assert rep.verdict in {"CLOSED", "CLOSED_AFTER_RESIZE"}
    assert rep.chosen is not None
    assert rep.chosen.rd.combo.cells == 4
    assert rep.chosen.rd.pack.name == "Tattu G-Tech 5200mAh 4S 35C"
    assert rep.chosen.metrics.endurance_min >= 18.0


def test_4s_hexa_heavy_payload_gap():
    # A 10Ah recommendation does not jump to the new 24/30Ah packs: the 10%
    # identity boundary still binds after the catalog grew.
    d = DesignInputs(1.5, 10000, 4, 6, 17 * 0.0254 / 2, 0.0)
    reqs = [
        "REQ-PERF-002: minimum endurance of 25 minutes with maximum rated payload",
        "REQ-CONS-003: maximum take-off mass shall not exceed 8.0 kg",
    ]
    assert match(d, reqs, DEFAULT_CATALOG)
    rep = close_the_loop(d, [], reqs, DEFAULT_CATALOG)
    assert rep.verdict == "INFEASIBLE_REALIZATION"
    assert rep.chosen is not None
    assert rep.chosen.rd.combo is U7_V2_KV490_17x58_4S
    assert rep.chosen.rd.pack.capacity_mah == 10000.0
    assert rep.chosen.metrics.endurance_min < 25.0


def test_catalog_closes_4s_hexa_derated():
    d = DesignInputs(1.5, 30000, 4, 6, 17 * 0.0254 / 2, 0.0)
    reqs = [
        "REQ-PERF-002: minimum endurance of 25 minutes with maximum rated payload",
        "REQ-CONS-003: maximum take-off mass shall not exceed 8.0 kg",
    ]
    rep = close_the_loop(d, [], reqs, DEFAULT_CATALOG)
    assert rep.verdict == "CLOSED"
    assert rep.chosen is not None
    assert rep.chosen.rd.combo is U7_V2_KV490_17x58_4S
    assert "T960 TL960A" in rep.chosen.rd.frame.name
    assert rep.chosen.rd.pack.capacity_mah == 30000.0
    assert rep.chosen.rd.pack.nominal_voltage_v == 14.4
    assert rep.chosen.rd.integration_bundle.mass_g == 500.0
    assert rep.chosen.metrics.pack_voltage_v == 14.4
    assert rep.chosen.metrics.voltage_ratio < 1.0
    assert rep.chosen.metrics.total_mass_kg <= 8.0
    assert rep.chosen.metrics.endurance_min >= 25.0


def test_octo_counts_integration_mass():
    # The X8 still maps structurally, but adding its explicit 600g octo
    # integration allowance reveals that the old 25min closure was optimistic.
    d = DesignInputs(1.5, 16000, 6, 8, 15 * 0.0254 / 2, 0.0)
    rep = close_the_loop(d, [], ["REQ-PERF-002: flight endurance of at least 25 minutes."],
                         DEFAULT_CATALOG)
    assert rep.verdict == "INFEASIBLE_REALIZATION"
    assert rep.chosen is not None
    assert rep.chosen.rd.frame.arms == 8
    assert rep.chosen.rd.frame.name.startswith("Tarot X8")
    assert rep.chosen.rd.integration_bundle.mass_g == 600.0
    assert rep.chosen.metrics.endurance_min < 25.0
    assert rep.per_requirement and not all(v.met for v in rep.per_requirement)
