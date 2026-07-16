from dataclasses import FrozenInstanceError

import pytest

from examples.catalog_coverage_diagnostic import CAPACITIES_MAH, _cell_catalog
from gazebo_poc.component_data import MN5008_KV340_18x61 as OLD_MN5008
from gazebo_poc.component_data import MotorProp as OldMotorProp
from src.realization.catalog import (
    DEFAULT_CATALOG,
    MN5008_KV340_18x61,
    MotorPropCombo,
    U7_V2_KV490_17x58_4S,
    U7_V2_KV490_18x61_4S,
)


def test_catalog_dataclasses_are_frozen():
    with pytest.raises(FrozenInstanceError):
        MN5008_KV340_18x61.motor_mass_g = 1.0


def test_gazebo_poc_reexport_keeps_old_import_path():
    assert OLD_MN5008 is MN5008_KV340_18x61
    assert OldMotorProp is MotorPropCombo
    assert OLD_MN5008.interp_at_thrust(1390.0)[0] == pytest.approx(
        MN5008_KV340_18x61.interp_at_thrust(1390.0)[0]
    )


def test_mn5008_has_traceable_realization_fields():
    m = MN5008_KV340_18x61
    assert m.source_url.startswith("https://store.tmotor.com/")
    assert m.retrieved == "2026-07-03"
    assert m.motor_mass_g == 135.0
    assert m.prop_mass_g == 31.5
    assert m.prop_diameter_in == 18.0


def _has_source_url(url: str) -> bool:
    return url.startswith(("http://", "https://"))


def test_default_catalog_entries_keep_source_and_retrieval_traceability():
    for combo in DEFAULT_CATALOG.combos:
        assert _has_source_url(combo.source_url)
        assert combo.retrieved.startswith("2026-07-")
        assert combo.motor_mass_g > 0
        assert combo.prop_mass_g > 0
        assert combo.prop_diameter_in > 0
        assert combo.curve
        for point in combo.curve:
            assert point.thrust_g > 0
            assert point.current_a > 0
            assert point.power_w > 0
    for pack in DEFAULT_CATALOG.packs:
        assert _has_source_url(pack.source_url)
        assert pack.retrieved.startswith("2026-07-")
        assert pack.capacity_mah > 0
        assert pack.cells > 0
        assert pack.mass_g > 0
        assert pack.c_rating > 0
    for frame in DEFAULT_CATALOG.frames:
        assert _has_source_url(frame.source_url)
        assert frame.retrieved.startswith("2026-07-")
        assert frame.mass_g > 0
        assert frame.arms > 0
        assert frame.max_prop_in > 0
    for bundle in DEFAULT_CATALOG.integration_bundles:
        assert bundle.source_url
        assert bundle.retrieved
        assert bundle.mass_g > 0
        assert bundle.components


def test_u7_4s_gap_fill_uses_official_voltage_specific_bench_data():
    for combo, diameter, max_thrust in (
        (U7_V2_KV490_17x58_4S, 17.0, 3000.0),
        (U7_V2_KV490_18x61_4S, 18.0, 3240.0),
    ):
        assert combo.cells == 4
        assert combo.voltage_v == 14.8
        assert combo.prop_diameter_in == diameter
        assert combo.motor_mass_g == 299.0
        assert combo.max_thrust_g() == max_thrust
        assert combo.source_url.startswith("https://store.tmotor.com/")
        assert combo.prop_source_url.startswith("https://store.tmotor.com/")
        assert combo.retrieved == "2026-07-14"


def test_4s_uav_capacity_gap_now_contains_10000mah_pack():
    packs = [
        pack for pack in DEFAULT_CATALOG.packs
        if pack.cells == 4 and pack.capacity_mah == 10000.0
    ]
    assert len(packs) == 1
    assert packs[0].mass_g == 940.0
    assert packs[0].c_rating == 25.0
    assert packs[0].retrieved == "2026-07-14"


def test_heavy_payload_4s_gap_has_light_frame_and_voltage_traced_uav_packs():
    t960 = next(f for f in DEFAULT_CATALOG.frames if "T960 TL960A" in f.name)
    assert t960.mass_g == 1050.0
    assert t960.arms == 6
    assert t960.max_prop_in == 18.0
    assert t960.source_url.startswith("https://tarotrc.com/")

    packs = {
        p.capacity_mah: p for p in DEFAULT_CATALOG.packs
        if p.name.startswith("Enepaq") and p.cells == 4
    }
    assert set(packs) == {24000.0, 30000.0}
    assert packs[24000.0].mass_g == 1860.0
    assert packs[30000.0].mass_g == 2300.0
    assert all(p.nominal_voltage_v == 14.4 for p in packs.values())
    assert all(p.operating_voltage_v() == 14.4 for p in packs.values())
    assert all(p.capacity_mah / 1000.0 * p.c_rating >= 240.0 for p in packs.values())


def test_coverage_diagnostic_uses_current_capacity_and_integration_domains():
    assert 24000 in CAPACITIES_MAH
    assert 30000 in CAPACITIES_MAH
    exact_4s = _cell_catalog(4)
    assert exact_4s.integration_bundles == DEFAULT_CATALOG.integration_bundles
