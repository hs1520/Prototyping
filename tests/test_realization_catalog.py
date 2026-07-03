from dataclasses import FrozenInstanceError

import pytest

from gazebo_poc.component_data import MN5008_KV340_18x61 as OLD_MN5008
from gazebo_poc.component_data import MotorProp as OldMotorProp
from src.realization.catalog import DEFAULT_CATALOG, MN5008_KV340_18x61, MotorPropCombo


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


def test_default_catalog_entries_keep_source_and_retrieval_traceability():
    for combo in DEFAULT_CATALOG.combos:
        assert combo.source_url.startswith("https://")
        assert combo.retrieved == "2026-07-03"
        assert combo.motor_mass_g > 0
        assert combo.prop_mass_g > 0
        assert combo.prop_diameter_in > 0
        assert combo.curve
        for point in combo.curve:
            assert point.thrust_g > 0
            assert point.current_a > 0
            assert point.power_w > 0
    for pack in DEFAULT_CATALOG.packs:
        assert pack.source_url.startswith("https://")
        assert pack.retrieved == "2026-07-03"
        assert pack.capacity_mah > 0
        assert pack.cells > 0
        assert pack.mass_g > 0
        assert pack.c_rating > 0
    for frame in DEFAULT_CATALOG.frames:
        assert frame.source_url.startswith("https://")
        assert frame.retrieved == "2026-07-03"
        assert frame.mass_g > 0
        assert frame.arms > 0
        assert frame.max_prop_in > 0
