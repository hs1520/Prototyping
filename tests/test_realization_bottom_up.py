import pytest

from src.dse.physics_estimator import DesignInputs
from src.realization.bottom_up import (
    RealizedDesign,
    payload_split,
    realized_metrics,
    realized_total_mass_kg,
)
from src.realization.catalog import (
    BatteryPack,
    Frame,
    IntegrationBundle,
    MotorPropCombo,
    MotorPropPoint,
)


def _combo(price=100.0):
    return MotorPropCombo(
        name="synthetic combo",
        source_url="synthetic",
        retrieved="synthetic",
        voltage_v=22.2,
        cells=6,
        motor_mass_g=100.0,
        prop_mass_g=20.0,
        prop_diameter_in=18.0,
        price_usd=price,
        curve=(
            MotorPropPoint(0.4, 500.0, 2.0, 44.4),
            MotorPropPoint(0.6, 1500.0, 6.0, 133.2),
            MotorPropPoint(1.0, 3000.0, 20.0, 444.0),
        ),
    )


def _pack(capacity, mass, price=200.0):
    return BatteryPack("synthetic pack", "synthetic", "synthetic", capacity, 6, mass, 20.0, price)


def _frame(price=50.0):
    return Frame("synthetic frame", "synthetic", "synthetic", 800.0, 4, 18.0, price)


def _rd(pack):
    return RealizedDesign(_combo(), pack, _frame(), 4, 1.5, 0.4)


def test_payload_split_uses_rated_delivery_and_equipment_delta():
    d = DesignInputs(2.1, 12000, 6, 4, 0.23)
    reqs = [
        "REQ-PERF-002: endurance at least 25 minutes at maximum rated payload.",
        "REQ-FUNC-003: transport payloads of up to 1.5 kg.",
    ]
    assert payload_split(d, reqs) == pytest.approx((1.5, 0.6))


def test_payload_split_clamps_negative_equipment_to_zero():
    d = DesignInputs(0.5, 12000, 6, 4, 0.23)
    reqs = ["REQ-FUNC-003: transport payloads of up to 1.5 kg."]
    assert payload_split(d, reqs) == pytest.approx((1.5, 0.0))


def test_realized_total_mass_is_sum_of_real_components():
    rd = _rd(_pack(12000, 1400.0))
    assert realized_total_mass_kg(rd) == pytest.approx(0.8 + 4 * 0.12 + 1.4 + 1.5 + 0.4)


def test_realized_total_mass_includes_explicit_integration_bundle():
    bundle = IntegrationBundle(
        "integration budget", "engineering assumption", "2026-07-14", 500.0,
        ("ESCs", "flight controller", "wiring"),
    )
    base = _rd(_pack(12000, 1400.0))
    rd = RealizedDesign(
        base.combo, base.pack, base.frame, base.rotor_count,
        base.delivery_payload_kg, base.equipment_mass_kg, bundle,
    )
    assert realized_total_mass_kg(rd) == pytest.approx(
        realized_total_mass_kg(base) + 0.5
    )


def test_lower_nominal_voltage_is_derated_instead_of_treated_as_equal_s_count():
    lipo = _pack(12000, 1400.0)
    li_ion = BatteryPack(
        "synthetic 6S Li-ion", "synthetic", "synthetic",
        12000, 6, 1400.0, 20.0, nominal_voltage_v=21.6,
    )
    baseline = realized_metrics(_rd(lipo))
    derated = realized_metrics(_rd(li_ion))
    assert derated.pack_voltage_v == 21.6
    assert derated.voltage_ratio == pytest.approx(21.6 / 22.2)
    assert derated.hover_current_per_motor_a > baseline.hover_current_per_motor_a
    assert derated.hover_throttle > baseline.hover_throttle
    assert derated.derated_max_thrust_per_motor_g < baseline.derated_max_thrust_per_motor_g
    assert derated.endurance_min < baseline.endurance_min
    assert "derated" in derated.notes[0]


def test_endurance_increases_with_larger_pack_but_with_diminishing_return():
    small = realized_metrics(_rd(_pack(8000, 900.0))).endurance_min
    medium = realized_metrics(_rd(_pack(12000, 1400.0))).endurance_min
    large = realized_metrics(_rd(_pack(16000, 2100.0))).endurance_min
    assert small < medium < large
    assert (large - medium) < (medium - small)


def test_cost_axis_price_and_missing_price_fallback():
    rd = _rd(_pack(12000, 1400.0, price=200.0))
    assert realized_metrics(rd, cost_axis="price").cost == pytest.approx(350.0)
    no_price = RealizedDesign(_combo(price=None), rd.pack, rd.frame, 4, 1.5, 0.4)
    m = realized_metrics(no_price, cost_axis="price")
    assert m.cost == pytest.approx(realized_total_mass_kg(no_price))
    assert "price requested" in m.notes[0]
