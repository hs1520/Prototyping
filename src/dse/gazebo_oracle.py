"""Gazebo FDM oracle - the architecture-axis level of the multi-fidelity ladder.

Ladder:  physics estimator (ms)  ->  SITL-default (s)  ->  Gazebo FDM (min).

SITL-default has a fixed frame mass, so hover current is identical for quad and
octa (27.3 A measured - see docs/DSE_REDESIGN.md §9.3/9.4): the capacity axis has
a calibrated anchor (Spearman ρ=1.0) while the architecture axis (rotor count /
radius / emergent mass) had only a declared scope limitation. This module flies
the designed airframe in Gazebo (gazebo_poc puts mass, rotor count and radius in
the generated SDF) and rank-correlates the estimator's prediction against the
measurement.

Both compared quantities are mechanical hover power:
  predicted  = momentum-theory hover power for the design's emergent total
               mass and disk area (``physics_estimator.hover_power_w``);
  measured   = Gazebo rotor-telemetry hover power (Σ Cp*ρ*n³*D⁵, captured by
               ``gazebo_poc.run_flight`` as ``hover_power_w``).

Live flights take ~5 min each (Docker + ArduPilot binary) and are gated by
``RUN_GAZEBO=1``; the calibration plumbing is testable offline by injecting
``measure_fn``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .calibration import CalibrationResult, calibrate_ranking
from .physics_estimator import DesignInputs, disk_area_m2, hover_power_w, total_mass_kg

SUPPORTED_ROTOR_COUNTS: Tuple[int, ...] = (4, 6, 8)


def predicted_hover_power_w(design: DesignInputs) -> float:
    """Estimator-side mechanical hover power for the design's emergent mass."""
    mass = total_mass_kg(design)
    return hover_power_w(mass, disk_area_m2(design.rotor_count, design.rotor_radius_m))


def architecture_sweep(
    base: DesignInputs,
    rotor_counts: Sequence[int] = SUPPORTED_ROTOR_COUNTS,
) -> List[DesignInputs]:
    """Architecture-axis design points: same payload/battery, frame swept.

    Only the frame varies, but total mass still emerges per point (propulsion mass
    scales with disk area), so the points differ the way architectures do - the
    dimension SITL-default cannot discriminate.
    """
    return [
        replace(base, rotor_count=n)
        for n in rotor_counts
        if n in SUPPORTED_ROTOR_COUNTS
    ]


@dataclass
class ArchMeasurement:
    """One Gazebo flight of one architecture."""
    label: str
    design: DesignInputs
    predicted_power_w: float
    measured_power_w: Optional[float] = None
    hover_stable: Optional[bool] = None
    note: str = ""


@dataclass
class ArchCalibration:
    """Architecture-axis calibration outcome (points + ranking verdict)."""
    points: List[ArchMeasurement]
    result: Optional[CalibrationResult]
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        flown = [p for p in self.points if p.measured_power_w is not None]
        head = f"architecture axis: {len(flown)}/{len(self.points)} point(s) measured"
        if self.result is None:
            return f"{head} — not enough for rank calibration"
        return f"{head}; {self.result.summary()}"


class GazeboArchitectureOracle:
    """Live oracle: fly one design in Gazebo, return the measured hover power.

    Gated like the SITL oracle: flights run only with ``RUN_GAZEBO=1`` (~5 min each,
    needs Docker + the ArduPilot binary). ``measure`` raises when gated so callers
    fail fast rather than skip.
    """

    ENV_GATE = "RUN_GAZEBO"

    @classmethod
    def is_available(cls) -> bool:
        return os.environ.get(cls.ENV_GATE) == "1"

    def measure(self, design: DesignInputs) -> Dict[str, object]:  # pragma: no cover - live Gazebo
        if not self.is_available():
            raise RuntimeError(
                f"live Gazebo gated; set {self.ENV_GATE}=1 to fly architectures"
            )
        from gazebo_poc import run_flight

        run_flight.main(
            mass_kg=total_mass_kg(design),
            rotor_radius=design.rotor_radius_m,
            capacity_mah=design.battery_capacity_mah,
            rotor_count=design.rotor_count,
            calibrate=True,
        )
        return dict(run_flight.LAST_RESULT)


def calibrate_architecture_axis(
    base_design: DesignInputs,
    rotor_counts: Sequence[int] = SUPPORTED_ROTOR_COUNTS,
    measure_fn: Optional[Callable[[DesignInputs], Dict[str, object]]] = None,
) -> ArchCalibration:
    """Sweep architectures, measure each in Gazebo, rank-correlate vs estimator.

    ``measure_fn(design) -> {"hover_power_w": float, "hover_stable": bool}`` is
    injectable for offline tests; the default is the live RUN_GAZEBO-gated
    :class:`GazeboArchitectureOracle`. Failed flights become unmeasured points -
    recorded, excluded from the correlation - so one transient does not sink the run.
    """
    measure = measure_fn or GazeboArchitectureOracle().measure
    points: List[ArchMeasurement] = []
    notes: List[str] = []

    for design in architecture_sweep(base_design, rotor_counts):
        label = f"{design.rotor_count}rotor"
        point = ArchMeasurement(
            label=label,
            design=design,
            predicted_power_w=predicted_hover_power_w(design),
        )
        try:
            r = measure(design) or {}
            power = r.get("hover_power_w")
            point.measured_power_w = float(power) if power else None
            stable = r.get("hover_stable")
            point.hover_stable = None if stable is None else bool(stable)
            if point.measured_power_w is None:
                point.note = "no hover power telemetry"
        except Exception as e:
            point.note = f"flight failed: {e!r}"
            notes.append(f"{label}: {point.note}")
        points.append(point)

    flown = [p for p in points if p.measured_power_w is not None]
    result: Optional[CalibrationResult] = None
    if len(flown) >= 2:
        result = calibrate_ranking(
            labels=[p.label for p in flown],
            predicted=[p.predicted_power_w for p in flown],
            measured=[p.measured_power_w for p in flown],
        )
    else:
        notes.append("fewer than 2 measured points — rank calibration skipped")
    return ArchCalibration(points=points, result=result, notes=notes)
