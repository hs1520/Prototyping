"""Check whether a measurement reached steady state.

A mean over a still-accelerating window is a function of the window length, not
a cruise speed: one run recorded a 21.2 m/s nil-wind cruise and then a higher
28.9 m/s against a 15 m/s headwind, both from a dash that never stopped
accelerating. Every speed the harness reports passes through
:func:`steady_state`; a series that has not plateaued yields ``steady=False``
and the caller reports inconclusive rather than a number.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

DEFAULT_MAX_DRIFT_FRACTION = 0.05
# A window shorter than this cannot distinguish plateau from noise.
DEFAULT_MIN_SAMPLES = 8
DEFAULT_MIN_DURATION_S = 3.0
DEFAULT_MAX_TREND_T_STAT = 2.0
# A trend must be both statistically significant and practically large. The
# t-statistic is |slope| / standard-error, so with many samples and little noise
# any nonzero slope is significant: a cruise point going 15.91 -> 15.91 m/s over
# 9.9 s (0.015% drift) was rejected as a trend. Below this fraction of the mean
# the window counts as a plateau.
DEFAULT_MIN_PRACTICAL_DRIFT_FRACTION = 0.01


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
    min_practical_drift_fraction: float = DEFAULT_MIN_PRACTICAL_DRIFT_FRACTION,
) -> SteadyState:
    """Classify ``[(time_s, value)]`` as plateaued or still trending.

    Both tests must pass: the least-squares trend moves the value by less than
    ``max_drift_fraction`` of the mean, and the two half means agree to within the
    same fraction. The half-mean test catches a monotone ramp a single slope fit
    can flatter.
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
    significant_trend = (
        trend_t_stat > max_trend_t_stat
        and drift_fraction > min_practical_drift_fraction
    )
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
