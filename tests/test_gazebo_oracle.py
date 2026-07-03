"""Architecture-axis Gazebo oracle: sweep, prediction, gating, rank calibration.

Live flights are RUN_GAZEBO-gated; everything here runs offline by injecting
``measure_fn``.  These tests pin the property the ladder was missing: the
estimator IS discriminating on the architecture axis, and the calibration
plumbing can both confirm a faithful oracle and detect a divergent one.
"""
from __future__ import annotations

import pytest

from src.dse.gazebo_oracle import (
    GazeboArchitectureOracle,
    architecture_sweep,
    calibrate_architecture_axis,
    predicted_hover_power_w,
)
from src.dse.physics_estimator import DesignInputs, total_mass_kg
from src.sitl.dse_calibration import calibrate_endurance, calibrate_ranking

_BASE = DesignInputs(
    payload_mass_kg=1.5,
    battery_capacity_mah=8000,
    battery_cells=6,
    rotor_count=4,
    rotor_radius_m=0.19,
)


class TestArchitectureSweep:
    def test_sweep_covers_supported_frames_only(self):
        designs = architecture_sweep(_BASE, rotor_counts=(4, 6, 8, 12))
        assert [d.rotor_count for d in designs] == [4, 6, 8]

    def test_mass_emerges_per_architecture(self):
        # propulsion mass scales with disk area → more rotors = heavier airframe;
        # this is exactly what SITL-default's fixed frame mass cannot express.
        masses = [total_mass_kg(d) for d in architecture_sweep(_BASE)]
        assert masses[0] < masses[1] < masses[2]

    def test_payload_and_battery_are_held_constant(self):
        for d in architecture_sweep(_BASE):
            assert d.payload_mass_kg == _BASE.payload_mass_kg
            assert d.battery_capacity_mah == _BASE.battery_capacity_mah


class TestPredictedPower:
    def test_estimator_discriminates_architectures(self):
        powers = [predicted_hover_power_w(d) for d in architecture_sweep(_BASE)]
        assert len(set(round(p) for p in powers)) == 3  # all distinct
        assert all(p > 0 for p in powers)


class TestLiveGate:
    def test_measure_raises_without_env_gate(self, monkeypatch):
        monkeypatch.delenv("RUN_GAZEBO", raising=False)
        oracle = GazeboArchitectureOracle()
        assert not oracle.is_available()
        with pytest.raises(RuntimeError, match="RUN_GAZEBO"):
            oracle.measure(_BASE)


class TestCalibrateArchitectureAxis:
    def test_faithful_oracle_yields_trustworthy_ranking(self):
        # Gazebo measuring ~15% above momentum theory but order-preserving:
        # ranking validated, absolute offset visible in the deltas.
        def faithful(design):
            return {"hover_power_w": predicted_hover_power_w(design) * 1.15,
                    "hover_stable": True}

        arch = calibrate_architecture_axis(_BASE, measure_fn=faithful)
        assert arch.result is not None
        assert arch.result.spearman == pytest.approx(1.0)
        assert arch.result.top1_match
        assert arch.result.rank_trustworthy
        assert all(p.hover_stable for p in arch.points)
        assert "trustworthy" in arch.summary()

    def test_divergent_oracle_is_detected(self):
        # An oracle that inverts the ordering must produce negative correlation
        # — the framework detects divergence rather than rubber-stamping.
        powers = {d.rotor_count: predicted_hover_power_w(d)
                  for d in architecture_sweep(_BASE)}
        inverted = dict(zip(sorted(powers), sorted(powers.values(), reverse=True)))

        def divergent(design):
            return {"hover_power_w": inverted[design.rotor_count],
                    "hover_stable": True}

        arch = calibrate_architecture_axis(_BASE, measure_fn=divergent)
        assert arch.result is not None
        assert arch.result.spearman < 0
        assert not arch.result.rank_trustworthy

    def test_failed_flight_is_recorded_not_fatal(self):
        def flaky(design):
            if design.rotor_count == 6:
                raise RuntimeError("SITL connection reset")
            return {"hover_power_w": predicted_hover_power_w(design) * 1.1,
                    "hover_stable": True}

        arch = calibrate_architecture_axis(_BASE, measure_fn=flaky)
        assert arch.result is not None            # 2 points still calibrate
        assert len(arch.result.labels) == 2
        failed = [p for p in arch.points if p.measured_power_w is None]
        assert len(failed) == 1 and "flight failed" in failed[0].note

    def test_too_few_measurements_skips_ranking_honestly(self):
        def mostly_dead(design):
            if design.rotor_count != 4:
                raise RuntimeError("no telemetry")
            return {"hover_power_w": 500.0, "hover_stable": True}

        arch = calibrate_architecture_axis(_BASE, measure_fn=mostly_dead)
        assert arch.result is None
        assert any("fewer than 2" in n for n in arch.notes)
        assert "not enough" in arch.summary()

    def test_missing_power_telemetry_becomes_unmeasured_point(self):
        def stable_but_mute(design):
            return {"hover_stable": True}  # flight ok, no power capture

        arch = calibrate_architecture_axis(_BASE, measure_fn=stable_but_mute)
        assert arch.result is None
        assert all(p.note == "no hover power telemetry" for p in arch.points)


def test_calibrate_endurance_alias_preserved():
    # capacity-axis callers keep working; both names are the same computation
    assert calibrate_endurance is calibrate_ranking
