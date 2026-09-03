"""Surrogate-vs-oracle calibration - the contribution-#3 artifact.

Quantifies how well the search surrogate (``grounded_eval`` model-level fault
injection) predicts a high-fidelity oracle (ArduPilot SITL closed-loop fault
injection) across a set of architectures. Rank-based (Spearman / Kendall) rather
than absolute, because the search only needs the oracle ordering preserved; scale
bias does not matter. Pure Python, no scipy; the oracle is pluggable (see
``sitl_oracle``): a live SITL run on the Pareto front, or a reference model offline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple


def _ranks(xs: Sequence[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    if n == 0:
        return 0.0
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va == 0 or vb == 0:
        return 0.0
    return cov / (va ** 0.5 * vb ** 0.5)


def spearman_rho(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman rank correlation in [-1, 1]."""
    return max(-1.0, min(1.0, _pearson(_ranks(xs), _ranks(ys))))


def kendall_tau(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Kendall's tau-a: (concordant - discordant) / (n choose 2)."""
    n = len(xs)
    if n < 2:
        return 0.0
    con = dis = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (xs[i] - xs[j]) * (ys[i] - ys[j])
            if s > 0:
                con += 1
            elif s < 0:
                dis += 1
    total = n * (n - 1) // 2
    return (con - dis) / total if total else 0.0


@dataclass
class CalibrationResult:
    labels: List[str]
    predicted: List[float]
    measured: List[float]
    spearman: float
    kendall: float
    top1_match: bool
    deltas: List[float]
    max_abs_deviation: float

    @property
    def rank_trustworthy(self) -> bool:
        """SITL validates the estimator's design ranking."""
        return self.spearman >= 0.9

    def summary(self) -> str:
        mt = "match" if self.top1_match else "MISMATCH"
        return (f"Spearman {self.spearman:+.3f}, Kendall {self.kendall:+.3f}, "
                f"top-1 {mt}, max |Δ| {self.max_abs_deviation:.1f} "
                f"(ranking {'trustworthy' if self.rank_trustworthy else 'NOT trustworthy'})")


def calibrate_ranking(
    labels: List[str], predicted: List[float], measured: List[float]
) -> CalibrationResult:
    """Rank-calibrate estimator-predicted vs oracle-measured values over design points."""
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


def top1_agreement(xs: Sequence[float], ys: Sequence[float]) -> bool:
    """Same best architecture under surrogate and oracle."""
    if not xs:
        return False
    return max(range(len(xs)), key=lambda i: xs[i]) == max(
        range(len(ys)), key=lambda i: ys[i]
    )


@dataclass
class CalibrationReport:
    labels: List[str]
    surrogate: List[float]
    oracle: List[float]
    spearman: float = 0.0
    kendall: float = 0.0
    top1_match: bool = False
    pairs: List[Tuple[str, float, float]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"calibration over {len(self.labels)} architectures:",
            f"  Spearman rho = {self.spearman:+.3f}   Kendall tau = {self.kendall:+.3f}"
            f"   top-1 match = {self.top1_match}",
        ]
        for lbl, s, o in self.pairs:
            lines.append(f"    {lbl:24s} surrogate={s:.4f}  oracle={o:.4f}")
        return "\n".join(lines)


def calibrate(
    labels: Sequence[str],
    surrogate: Sequence[float],
    oracle: Sequence[float],
) -> CalibrationReport:
    """Compute the surrogate↔oracle calibration report from paired scores."""
    if not (len(labels) == len(surrogate) == len(oracle)):
        raise ValueError("labels, surrogate, oracle must have equal length")
    s, o = list(surrogate), list(oracle)
    return CalibrationReport(
        labels=list(labels),
        surrogate=s,
        oracle=o,
        spearman=spearman_rho(s, o),
        kendall=kendall_tau(s, o),
        top1_match=top1_agreement(s, o),
        pairs=list(zip(labels, s, o)),
    )
