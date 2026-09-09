from src.realization.catalog import BatteryPack, ComponentCatalog, Frame, MotorPropCombo, MotorPropPoint


def combo(name="synthetic combo", cells=6, diameter=18.0, motor_mass=100.0,
          prop_mass=20.0, high_current=20.0, max_thrust=3000.0):
    mid_high = min(1600.0, max_thrust * 0.75)
    return MotorPropCombo(
        name=name,
        source_url="synthetic",
        retrieved="synthetic",
        voltage_v=22.2,
        cells=cells,
        motor_mass_g=motor_mass,
        prop_mass_g=prop_mass,
        prop_diameter_in=diameter,
        price_usd=100.0,
        curve=(
            MotorPropPoint(0.30, 400.0, 1.0, 22.2),
            MotorPropPoint(0.50, 1000.0, 4.0, 88.8),
            MotorPropPoint(0.65, mid_high, 8.0, 177.6),
            MotorPropPoint(1.00, max_thrust, high_current, 444.0),
        ),
    )


def pack(name="synthetic pack", capacity=12000.0, cells=6, mass=1200.0, c=20.0, price=200.0):
    return BatteryPack(name, "synthetic", "synthetic", capacity, cells, mass, c, price)


def frame(name="synthetic frame", arms=4, max_prop=18.0, mass=800.0, price=50.0):
    return Frame(name, "synthetic", "synthetic", mass, arms, max_prop, price)


def catalog(combos=None, packs=None, frames=None):
    return ComponentCatalog(tuple(combos or [combo()]), tuple(packs or [pack()]), tuple(frames or [frame()]))
