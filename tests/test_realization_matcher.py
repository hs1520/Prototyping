from src.dse.physics_estimator import DesignInputs
from src.realization.matcher import match

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
    far = pack("far", capacity=18000)
    got = match(_design(), [], catalog(packs=[far, near]))
    assert [c.rd.pack.name for c in got] == ["near", "far"]


def test_design_drift_is_reported_without_filtering_candidate():
    d = _design(battery_capacity_mah=10000, rotor_radius_m=0.20)
    got = match(d, [], catalog(packs=[pack(capacity=12000)]))
    assert got
    drift = {d.name: d for d in got[0].design_drift}
    assert drift["battery_capacity_mah"].expected == 10000
    assert drift["battery_capacity_mah"].realized == 12000
    assert drift["battery_capacity_mah"].delta == 2000
    assert drift["rotor_radius_m"].realized != drift["rotor_radius_m"].expected
    assert drift["battery_cells"].delta == 0
