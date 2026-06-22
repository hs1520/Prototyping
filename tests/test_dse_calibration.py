"""Layer 6: estimator-vs-SITL endurance calibration.

Uses the real probe data (default quad, 4/8/16 Ah): estimator models battery
self-weight (diminishing returns) while SITL has fixed frame mass (linear), so absolute
endurance diverges at big batteries yet the RANKING is identical — the calibration must
report ρ=1 (ranking trustworthy) and surface the large absolute deviation.
"""
from __future__ import annotations

import pytest

from src.sitl.dse_calibration import calibrate_endurance

# real measured SITL endurance (sitl_probe.py) vs calibrated estimator predictions
_LABELS = ["4Ah", "8Ah", "16Ah"]
_PREDICTED = [11.1, 16.9, 22.1]   # estimator (battery + propulsion self-weight → diminishing)
_MEASURED = [8.8, 17.6, 35.2]     # SITL (fixed frame mass → linear)


def test_ranking_is_validated_by_sitl():
    r = calibrate_endurance(_LABELS, _PREDICTED, _MEASURED)
    assert r.spearman == pytest.approx(1.0)   # identical ordering
    assert r.kendall == pytest.approx(1.0)
    assert r.top1_match                        # both rank 16Ah highest
    assert r.rank_trustworthy


def test_absolute_divergence_is_surfaced():
    r = calibrate_endurance(_LABELS, _PREDICTED, _MEASURED)
    # the big-battery gap (estimator self-weight vs SITL fixed mass) is the finding
    assert r.max_abs_deviation > 10.0
    assert r.deltas[-1] < 0      # at 16Ah, estimator < SITL (estimator pays self-weight)


def test_summary_readable():
    s = calibrate_endurance(_LABELS, _PREDICTED, _MEASURED).summary()
    assert "Spearman" in s and "trustworthy" in s


def test_needs_at_least_two_points():
    with pytest.raises(ValueError):
        calibrate_endurance(["a"], [1.0], [1.0])
