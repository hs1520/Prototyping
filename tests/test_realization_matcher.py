from src.dse.physics_estimator import DesignInputs
from src.realization.catalog import DEFAULT_CATALOG
from src.realization.matcher import (
    catalog_capacity_options,
    catalog_design_domain,
    match,
    variant_design_is_catalog_admissible,
)

from .realization_fixtures import catalog, combo, frame, pack


def _design(**kw):
    base = dict(payload_mass_kg=1.0, battery_capacity_mah=12000, battery_cells=6,
                rotor_count=4, rotor_radius_m=18 * 0.0254 / 2, cruise_speed_mps=0.0)
    base.update(kw)
    return DesignInputs(**base)


def test_all_hard_filters_can_pass():
    assert match(_design(), [], catalog())


def test_cells_mismatch_filtered_out():
    assert match(_design(), [], catalog(packs=[pack(cells=4)])) == []


def test_catalog_voltage_must_also_match_recommended_design_voltage():
    # Pack and motor curve agree with each other at 4S, but the recommended
    # architecture is 6S.  This is a different design, not a valid mapping.
    assert match(
        _design(battery_cells=6), [],
        catalog(combos=[combo(cells=4)], packs=[pack(cells=4)]),
    ) == []


def test_arms_mismatch_filtered_out():
    assert match(_design(), [], catalog(frames=[frame(arms=6)])) == []


def test_prop_fit_filtered_out():
    assert match(_design(), [], catalog(frames=[frame(max_prop=17.0)])) == []


def test_can_hover_filtered_out():
    heavy = _design(payload_mass_kg=9.0)
    assert match(heavy, [], catalog()) == []


def test_hover_throttle_filtered_out():
    d = _design(payload_mass_kg=5.0)
    assert match(d, [], catalog()) == []


def test_twr_filtered_out():
    weak = combo(max_thrust=1200.0)
    assert match(_design(), [], catalog(combos=[weak])) == []


def test_c_rating_filtered_out():
    assert match(_design(), [], catalog(packs=[pack(c=1.0)])) == []


def test_distance_sorting_is_stable():
    near = pack("near", capacity=12000)
    far = pack("far", capacity=13000)
    got = match(_design(), [], catalog(packs=[far, near]))
    assert [c.rd.pack.name for c in got] == ["near", "far"]


def test_design_drift_is_reported_and_bounded():
    d = _design(battery_capacity_mah=11000, rotor_radius_m=0.22)
    got = match(d, [], catalog(packs=[pack(capacity=12000)]))
    assert got
    drift = {d.name: d for d in got[0].design_drift}
    assert drift["battery_capacity_mah"].expected == 11000
    assert drift["battery_capacity_mah"].realized == 12000
    assert drift["battery_capacity_mah"].delta == 1000
    assert drift["rotor_radius_m"].realized != drift["rotor_radius_m"].expected
    assert drift["battery_cells"].delta == 0
    assert all(item.within_limit for item in drift.values())


def test_rotor_radius_drift_over_ten_percent_is_not_the_same_design():
    assert match(_design(rotor_radius_m=0.20), [], catalog()) == []


def test_battery_capacity_drift_over_ten_percent_is_not_the_same_design():
    assert match(_design(battery_capacity_mah=10000), [], catalog()) == []


def test_ten_percent_capacity_boundary_is_admissible():
    got = match(
        _design(battery_capacity_mah=10000), [],
        catalog(packs=[pack(capacity=11000)]),
    )
    assert got
    drift = {x.name: x for x in got[0].design_drift}
    assert drift["battery_capacity_mah"].within_limit is True
    assert drift["battery_capacity_mah"].limit == 0.10


def test_default_catalog_domain_exposes_only_evidence_backed_voltage_families():
    domain = catalog_design_domain()
    assert domain["battery_cells"] == [4, 6]
    assert 8 not in domain["battery_cells"] and 12 not in domain["battery_cells"]
    assert domain["architectures"]
    assert 30000.0 in domain["battery_capacity_mah_by_cells"]["4"]


def test_partial_variant_is_checked_against_a_complete_catalog_architecture():
    assert variant_design_is_catalog_admissible({
        "rotor_count": 6, "battery_cells": 6, "rotor_radius_m": 0.2286,
    })
    assert not variant_design_is_catalog_admissible({"battery_cells": 8})
    assert not variant_design_is_catalog_admissible({
        "rotor_count": 4, "battery_cells": 4, "rotor_radius_m": 0.2286,
    })


def test_catalog_capacity_options_are_real_packs_that_map_exactly():
    domain = catalog_design_domain()
    arch = next(x for x in domain["architectures"] if x["battery_cells"] == 6)
    d = DesignInputs(
        payload_mass_kg=0.1,
        battery_capacity_mah=10000,
        battery_cells=arch["battery_cells"],
        rotor_count=arch["rotor_count"],
        rotor_radius_m=arch["rotor_radius_m"],
        cruise_speed_mps=0.0,
    )
    options = catalog_capacity_options(d, [])
    catalog_caps = {p.capacity_mah for p in DEFAULT_CATALOG.packs if p.cells == 6}
    assert options
    assert set(options) <= catalog_caps
    assert all(match(DesignInputs(**{**vars(d), "battery_capacity_mah": cap}), [])
               for cap in options)
