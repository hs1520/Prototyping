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


def test_default_real_catalog_4s_quad_snaps_to_endurance_pack_for_realistic_need():
    # A1 regression: the 4S catalog must overlap the inner BO capacity axis, not only
    # the small R-Line racing packs. This realistic 18min request now snaps to the
    # traced 5200mAh 4S pack and closes on datasheet hover metrics.
    d = DesignInputs(0.2, 5200, 4, 4, 15 * 0.0254 / 2, 0.0)
    rep = close_the_loop(d, [], ["REQ-PERF-002: flight endurance of at least 18 minutes."],
                         DEFAULT_CATALOG)
    assert rep.verdict in {"CLOSED", "CLOSED_AFTER_RESIZE"}
    assert rep.chosen is not None
    assert rep.chosen.rd.combo.cells == 4
    assert rep.chosen.rd.pack.name == "Tattu G-Tech 5200mAh 4S 35C"
    assert rep.chosen.metrics.endurance_min >= 18.0


def test_default_real_catalog_4s_hexa_has_heavy_payload_mapping_but_honest_gap():
    # A 10Ah recommendation must not silently jump to the new 24/30Ah packs:
    # the 10% identity boundary remains binding even though the catalog grew.
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


def test_default_real_catalog_closes_4s_hexa_with_integration_and_voltage_derating():
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


def test_default_real_catalog_octo_no_longer_ignores_integration_mass():
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
