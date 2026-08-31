"""Did the measurement actually reach steady state?

A mean taken over a window that is still accelerating is not a cruise speed —
it is a function of how long the window was. The 2026-08-30 authoritative run
recorded a "nil-wind cruise" of 21.2 m/s and then a *higher* 28.9 m/s after a
15 m/s headwind was injected; both numbers came from a dash that never stopped
accelerating, because the airframe carried no parasitic drag.

Every speed this harness reports now passes through :func:`steady_state`. A
series that has not plateaued yields ``steady=False``, and the caller must
report INCONCLUSIVE rather than a number. Reporting no value is honest;
reporting an accelerating vehicle's instantaneous speed as its cruise speed is
not.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

#: Total trend-driven change across the window, as a fraction of the window
#: mean, below which the series counts as plateaued.
DEFAULT_MAX_DRIFT_FRACTION = 0.05
#: A window shorter than this cannot distinguish plateau from noise.
DEFAULT_MIN_SAMPLES = 8
DEFAULT_MIN_DURATION_S = 3.0
DEFAULT_MAX_TREND_T_STAT = 2.0


@dataclass(frozen=True)
class SteadyState:
    steady: bool
    reason: str
    mean: Optional[float] = None
    slope_per_s: Optional[float] = None
    drift_fraction: Optional[float] = None
    first_half_mean: Optional[float] = None
    second_half_mean: Optional[float] = None
    samples: int = 0
    duration_s: Optional[float] = None
    trend_t_stat: Optional[float] = None

    def as_dict(self) -> dict:
        def r(value, digits=6):
            return None if value is None else round(value, digits)

        return {
            "steady": self.steady,
            "reason": self.reason,
            "mean": r(self.mean, 4),
            "slope_per_s": r(self.slope_per_s),
            "drift_fraction": r(self.drift_fraction),
            "first_half_mean": r(self.first_half_mean, 4),
            "second_half_mean": r(self.second_half_mean, 4),
            "samples": self.samples,
            "duration_s": r(self.duration_s, 3),
            "trend_t_stat": r(self.trend_t_stat, 3),
        }

    def describe(self) -> str:
        """One clause, suitable for pasting into an evidence string."""
        if self.steady:
            return (
                f"steady-state confirmed (drift {self.drift_fraction:.1%} of mean "
                f"over {self.duration_s:.1f} s, n={self.samples})"
            )
        if self.drift_fraction is None:
            return f"steady state not established: {self.reason}"
        return (
            f"NOT steady state: {self.reason} "
            f"(drift {self.drift_fraction:.1%} of mean over {self.duration_s:.1f} s, "
            f"{self.first_half_mean:.2f} -> {self.second_half_mean:.2f} m/s, "
            f"n={self.samples})"
        )


def _linear_slope(times: Sequence[float], values: Sequence[float]) -> float:
    n = len(values)
    mean_t = sum(times) / n
    mean_v = sum(values) / n
    denominator = sum((t - mean_t) ** 2 for t in times)
    if denominator <= 0:
        return 0.0
    numerator = sum((t - mean_t) * (v - mean_v) for t, v in zip(times, values))
    return numerator / denominator


def _trend_t_stat(
    times: Sequence[float], values: Sequence[float], slope: float,
) -> float:
    """Signal-to-noise ratio of the fitted slope under ordinary least squares."""
    n = len(values)
    mean_t = sum(times) / n
    mean_v = sum(values) / n
    sxx = sum((time_value - mean_t) ** 2 for time_value in times)
    intercept = mean_v - slope * mean_t
    residual_sum_squares = sum(
        (value - (intercept + slope * time_value)) ** 2
        for time_value, value in zip(times, values)
    )
    if residual_sum_squares <= 1e-18:
        return math.inf if abs(slope) > 1e-12 else 0.0
    slope_standard_error = math.sqrt(
        (residual_sum_squares / (n - 2)) / sxx
    )
    return abs(slope) / slope_standard_error


def steady_state(
    samples: Sequence[Tuple[float, float]],
    *,
    max_drift_fraction: float = DEFAULT_MAX_DRIFT_FRACTION,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    min_duration_s: float = DEFAULT_MIN_DURATION_S,
    max_trend_t_stat: float = DEFAULT_MAX_TREND_T_STAT,
) -> SteadyState:
    """Classify ``[(time_s, value)]`` as plateaued or still trending.

    Both tests must pass: the least-squares trend must move the value by less
    than ``max_drift_fraction`` of the mean across the window, and the two half
    means must agree to within the same fraction. The half-mean test catches a
    monotone ramp whose endpoints a single slope fit can flatter.
    """
    ordered = sorted(samples, key=lambda item: item[0])
    n = len(ordered)
    if n < min_samples:
        return SteadyState(False, f"only {n} samples (need {min_samples})", samples=n)

    times = [float(t) for t, _ in ordered]
    values = [float(v) for _, v in ordered]
    duration = times[-1] - times[0]
    if duration < min_duration_s:
        return SteadyState(
            False,
            f"window {duration:.1f} s is shorter than {min_duration_s:.1f} s",
            samples=n,
            duration_s=duration,
        )

    mean = sum(values) / n
    if abs(mean) < 1e-9:
        return SteadyState(False, "mean is zero", mean=mean, samples=n,
                           duration_s=duration)

    slope = _linear_slope(times, values)
    trend_t_stat = _trend_t_stat(times, values, slope)
    drift_fraction = abs(slope * duration) / abs(mean)

    half = n // 2
    first_half = sum(values[:half]) / half
    second_half = sum(values[half:]) / (n - half)
    half_gap = abs(second_half - first_half) / abs(mean)

    trending = drift_fraction > max_drift_fraction
    significant_trend = trend_t_stat > max_trend_t_stat
    halves_disagree = half_gap > max_drift_fraction
    if trending or significant_trend or halves_disagree:
        reason = (
            "still trending" if trending
            else "statistically significant trend" if significant_trend
            else "half-window means disagree"
        )
        return SteadyState(
            False, reason, mean, slope, drift_fraction, first_half, second_half,
            n, duration, trend_t_stat,
        )

    return SteadyState(
        True, "plateaued", mean, slope, drift_fraction, first_half, second_half,
        n, duration, trend_t_stat,
    )
