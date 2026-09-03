"""Layer 6: estimator-vs-SITL endurance calibration.

Probe data (default quad, 4/8/16 Ah): the estimator models battery self-weight
(diminishing returns), SITL has fixed frame mass (linear), so absolute endurance
diverges at big batteries while the ranking holds. Calibration reports rho=1 and
the absolute deviation.
"""
from __future__ import annotations

import pytest

from src.sitl.dse_calibration import calibrate_endurance

_LABELS = ["4Ah", "8Ah", "16Ah"]
_PREDICTED = [11.1, 16.9, 22.1]
_MEASURED = [8.8, 17.6, 35.2]


def test_ranking_validated_by_sitl():
    r = calibrate_endurance(_LABELS, _PREDICTED, _MEASURED)
    assert r.spearman == pytest.approx(1.0)
    assert r.kendall == pytest.approx(1.0)
    assert r.top1_match
    assert r.rank_trustworthy


def test_absolute_divergence_surfaced():
    r = calibrate_endurance(_LABELS, _PREDICTED, _MEASURED)
    assert r.max_abs_deviation > 10.0
    assert r.deltas[-1] < 0


def test_summary_readable():
    s = calibrate_endurance(_LABELS, _PREDICTED, _MEASURED).summary()
    assert "Spearman" in s and "trustworthy" in s


def test_needs_at_least_two_points():
    with pytest.raises(ValueError):
        calibrate_endurance(["a"], [1.0], [1.0])
