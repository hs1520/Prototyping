"""Layer 6 of the DSE→SITL plan: calibrate the estimator against SITL.

Correlates the estimator's PREDICTED emergent metric (e.g. endurance) with the value
MEASURED in SITL over several design points. The headline is the RANK correlation
(Spearman/Kendall): if the estimator orders designs the same way SITL does, the cheap
analytic proxy is trustworthy for DSE ranking — even if absolute values diverge. The
absolute deviation is reported too, and large divergence is a finding (it localises a
fidelity gap), not necessarily an estimator error.
"""
from __future__ import annotations

from ..dse.calibration import CalibrationResult, calibrate_ranking


# Original (capacity-axis) name kept as an alias — same computation.
calibrate_endurance = calibrate_ranking
