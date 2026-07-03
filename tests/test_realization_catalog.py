from dataclasses import FrozenInstanceError

import pytest

from gazebo_poc.component_data import MN5008_KV340_18x61 as OLD_MN5008
from gazebo_poc.component_data import MotorProp as OldMotorProp
from src.realization.catalog import MN5008_KV340_18x61, MotorPropCombo


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

