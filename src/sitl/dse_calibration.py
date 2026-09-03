"""Layer 6 of the DSE->SITL plan: calibrate the estimator against SITL.

Correlates the estimator's predicted emergent metric (e.g. endurance) with
the SITL-measured value over several design points. The headline is the rank
correlation (Spearman/Kendall): matching order is enough for DSE ranking even
when absolute values diverge. Absolute deviation is reported too; a large
divergence localises a fidelity gap rather than an estimator error.
"""
from __future__ import annotations

from ..dse.calibration import calibrate_ranking


# Original (capacity-axis) name kept as an alias - same computation.
calibrate_endurance = calibrate_ranking
