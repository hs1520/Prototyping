"""Parasitic-drag area is derived from the airframe that is actually drawn, and
speed measurements must reach steady state before they may be reported.

Both guards exist because of one concrete defect in the 2026-08-30 authoritative
run: the airframe carried no body drag, so a forward dash never reached terminal
velocity and its ground speed *rose* from 14.2 to 28.9 m/s after a 15 m/s
headwind was injected.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from gazebo_poc.airframe_drag import (
    CD_ARM,
    CD_HUB,
    CD_PAYLOAD,
    PAYLOAD_BOX_SIZE_M,
    drag_breakdown,
    flat_plate_area_m2,
    terminal_speed_mps,
)
from gazebo_poc.multirotor_sdf import (
    ARM_THICKNESS_M,
    HUB_HEIGHT_M,
    arm_length_m,
    arm_width_m,
    hub_radius_m,
)
from gazebo_poc.sdf_generator import multirotor_inertia
from gazebo_poc.steady_state import steady_state

_TEMPLATES = Path("gazebo_poc/templates")
_HAS_TEMPLATES = (_TEMPLATES / "all_models" / "iris_with_gimbal" / "model.sdf").exists()

# The 2026-08-30 authoritative run's recommended design.
_ROTORS, _RADIUS, _MASS = 6, 0.2032, 5.54


def test_flat_plate_area_matches_independent_hand_calculation():
    arm_len = arm_length_m(_RADIUS)
    arm_w = arm_width_m(_RADIUS)
    hub = 2.0 * hub_radius_m(arm_len) * HUB_HEIGHT_M
    # HEXA_X arms sit at 90, -90, -30, 150, 30, -150 degrees.
    angles = (90.0, -90.0, -30.0, 150.0, 30.0, -150.0)
    arms = sum(
        ARM_THICKNESS_M
        * (arm_len * abs(math.sin(math.radians(a))) + arm_w * abs(math.cos(math.radians(a))))
        for a in angles
    )
    expected = CD_HUB * hub + CD_ARM * arms

    assert flat_plate_area_m2(_ROTORS, _RADIUS) == pytest.approx(expected)
    # Sanity: a bare 0.9 m hexa hub-and-arms frame is a few hundredths of a m^2.
    assert 0.03 < expected < 0.09


def test_payload_adds_its_own_frontal_area():
    bare = drag_breakdown(_ROTORS, _RADIUS, payload_attached=False)
    laden = drag_breakdown(_ROTORS, _RADIUS, payload_attached=True)
    _, box_y, box_z = PAYLOAD_BOX_SIZE_M

    assert bare.payload_area_m2 == 0.0
    assert laden.payload_area_m2 == pytest.approx(box_y * box_z)
    assert laden.flat_plate_area_m2 == pytest.approx(
        bare.flat_plate_area_m2 + CD_PAYLOAD * box_y * box_z
    )


def test_drag_area_scales_with_the_design_rather_than_being_a_constant():
    """The old model hard-coded 0.05 m^2 for every design."""
    small = flat_plate_area_m2(4, 0.12)
    large = flat_plate_area_m2(8, 0.28)
    assert large > 2.0 * small


@pytest.mark.skipif(not _HAS_TEMPLATES, reason="iris templates absent (regenerate via docker cp)")
def test_drag_geometry_cannot_diverge_from_the_drawn_sdf(tmp_path):
    """The hub and arms the drag model sums are the ones the SDF draws."""
    from gazebo_poc.multirotor_sdf import generate_multirotor_sdf

    standoffs, _gimbal, _fc = generate_multirotor_sdf(
        _MASS, _ROTORS, _RADIUS, multirotor_inertia(_MASS, _ROTORS, _RADIUS),
        0.01, _TEMPLATES, tmp_path,
    )
    sdf = standoffs.read_text()

    hub = re.search(
        r"hub_visual.*?<cylinder><radius>([\d.]+)</radius><length>([\d.]+)</length>",
        sdf, re.S,
    )
    assert hub, "hub visual not found in generated SDF"
    assert float(hub.group(1)) == pytest.approx(hub_radius_m(arm_length_m(_RADIUS)), abs=1e-4)
    assert float(hub.group(2)) == pytest.approx(HUB_HEIGHT_M, abs=1e-4)

    arms = re.findall(r"arm_\d+_visual.*?<box><size>([\d.]+) ([\d.]+) ([\d.]+)</size>", sdf, re.S)
    assert len(arms) == _ROTORS
    for length, width, thickness in arms:
        assert float(length) == pytest.approx(arm_length_m(_RADIUS), abs=1e-4)
        assert float(width) == pytest.approx(arm_width_m(_RADIUS), abs=1e-4)
        assert float(thickness) == pytest.approx(ARM_THICKNESS_M, abs=1e-4)


def test_terminal_speed_is_the_drag_thrust_balance():
    f = flat_plate_area_m2(_ROTORS, _RADIUS, payload_attached=True)
    tilt = math.radians(19.1)                       # RC2=1330 against ANGLE_MAX=45 deg
    v = terminal_speed_mps(f, _MASS, tilt)
    drag_n = 0.5 * 1.2041 * f * v * v
    assert drag_n == pytest.approx(_MASS * 9.81 * math.tan(tilt))
    # A dash that accelerates forever has no terminal speed; this one does, and
    # it is well below the 28.9 m/s the drag-free airframe reached.
    assert 15.0 < v < 28.0


# --------------------------------------------------------------------------
# steady state
# --------------------------------------------------------------------------

def test_accelerating_dash_is_not_steady_state():
    """The shape the 2026-08-30 run actually recorded: still accelerating."""
    samples = [(t * 0.5, 21.2 + 0.36 * (t * 0.5)) for t in range(40)]
    verdict = steady_state(samples)
    assert not verdict.steady
    assert verdict.reason == "still trending"
    assert verdict.second_half_mean > verdict.first_half_mean
    assert "NOT steady state" in verdict.describe()


def test_plateaued_dash_with_noise_is_steady_state():
    noise = [0.0, 0.12, -0.09, 0.05, -0.14, 0.08, -0.03, 0.11, -0.07, 0.02]
    samples = [(t * 0.5, 23.8 + noise[t % len(noise)]) for t in range(40)]
    verdict = steady_state(samples)
    assert verdict.steady
    assert verdict.drift_fraction < 0.05
    assert "steady-state confirmed" in verdict.describe()


def test_short_or_sparse_windows_cannot_claim_steady_state():
    assert not steady_state([(0.0, 20.0), (0.5, 20.0)]).steady
    dense_but_brief = [(t * 0.05, 20.0) for t in range(20)]
    verdict = steady_state(dense_but_brief)
    assert not verdict.steady
    assert "shorter than" in verdict.reason


def test_monotone_ramp_is_caught_even_when_the_slope_fit_is_flattered():
    """A ramp that stalls at both ends still fails on the half-window means."""
    values = [20.0] * 10 + list(20.0 + 0.4 * i for i in range(1, 11)) + [24.0] * 10
    samples = [(t * 0.5, v) for t, v in enumerate(values)]
    verdict = steady_state(samples)
    assert not verdict.steady
