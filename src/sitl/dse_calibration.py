"""Layer 6 of the DSE→SITL plan: calibrate the estimator against SITL.

Correlates the estimator's PREDICTED emergent metric (e.g. endurance) with the value
MEASURED in SITL over several design points. The headline is the RANK correlation
(Spearman/Kendall): if the estimator orders designs the same way SITL does, the cheap
analytic proxy is trustworthy for DSE ranking — even if absolute values diverge. The
absolute deviation is reported too, and large divergence is a finding (it localises a
fidelity gap), not necessarily an estimator error.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..dse.calibration import kendall_tau, spearman_rho


@dataclass
class CalibrationResult:
    labels: List[str]
    predicted: List[float]
    measured: List[float]
    spearman: float
    kendall: float
    top1_match: bool
    deltas: List[float]          # predicted − measured, per design
    max_abs_deviation: float

    @property
    def rank_trustworthy(self) -> bool:
        """The estimator's DESIGN RANKING is validated by SITL."""
        return self.spearman >= 0.9

    def summary(self) -> str:
        mt = "match" if self.top1_match else "MISMATCH"
        return (f"Spearman {self.spearman:+.3f}, Kendall {self.kendall:+.3f}, "
                f"top-1 {mt}, max |Δ| {self.max_abs_deviation:.1f} "
                f"(ranking {'trustworthy' if self.rank_trustworthy else 'NOT trustworthy'})")


def calibrate_ranking(
    labels: List[str], predicted: List[float], measured: List[float]
) -> CalibrationResult:
    """Rank-calibrate estimator-predicted vs oracle-measured values over design
    points.  Metric-agnostic: endurance vs SITL (capacity axis) and hover power
    vs Gazebo (architecture axis) both go through here."""
    if not (len(labels) == len(predicted) == len(measured)) or len(labels) < 2:
        raise ValueError("need ≥2 aligned (label, predicted, measured) points")
    deltas = [p - m for p, m in zip(predicted, measured)]
    top_pred = labels[predicted.index(max(predicted))]
    top_meas = labels[measured.index(max(measured))]
    return CalibrationResult(
        labels=list(labels),
        predicted=list(predicted),
        measured=list(measured),
        spearman=spearman_rho(predicted, measured),
        kendall=kendall_tau(predicted, measured),
        top1_match=(top_pred == top_meas),
        deltas=deltas,
        max_abs_deviation=max((abs(d) for d in deltas), default=0.0),
    )


# Original (capacity-axis) name kept as an alias — same computation.
calibrate_endurance = calibrate_ranking
