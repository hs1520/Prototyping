"""Inner-layer BO over battery capacity (the continuous half of the bilevel).

Endurance satisfaction saturates while mass grows with capacity, so the BO
settles on an internal optimum (cheapest pack that meets endurance) rather than
the boundary, even for an unreachable target.
"""
from __future__ import annotations

from src.dse.inner_sizing import (
    CAPACITY_BOUNDS,
    optimize_capacity,
    optimize_discrete_capacity,
)

_QUAD = {
    "payload_mass_kg": 0.5, "battery_cells": 4,
    "rotor_count": 4, "rotor_radius_m": 0.13, "cruise_speed_mps": 0.0,
}


def test_reachable_target_internal():
    r = optimize_capacity(_QUAD, target_endurance_min=15.0)
    assert r["capacity_mah"] < 12000
    assert 13.0 <= r["endurance_min"] <= 18.0


def test_higher_target_more_capacity():
    c12 = optimize_capacity(_QUAD, 12.0)["capacity_mah"]
    c15 = optimize_capacity(_QUAD, 15.0)["capacity_mah"]
    c18 = optimize_capacity(_QUAD, 18.0)["capacity_mah"]
    assert c12 < c15 < c18


def test_unreachable_not_pinned():
    r = optimize_capacity(_QUAD, target_endurance_min=90.0)
    assert r["capacity_mah"] < CAPACITY_BOUNDS[1]
    assert r["endurance_min"] < 90.0


def test_bo_searches():
    r = optimize_capacity(_QUAD, 15.0, n_init=4, n_iter=16)
    assert r["bo_evals"] == 20


def test_discrete_picks_catalog_capacity():
    capacities = [8000.0, 10000.0, 12000.0]
    r = optimize_discrete_capacity(_QUAD, 15.0, capacities)
    assert r["capacity_mah"] in capacities
    assert r["bo_evals"] == len(capacities)
    assert r["sizing_mode"] == "catalog_discrete"
