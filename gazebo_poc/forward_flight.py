"""Backward-compatible import path for the realization forward-flight model."""
from src.realization.forward_flight import (  # noqa: F401
    DEFAULT_DRAG_AREA,
    FOM,
    G,
    RHO,
    ForwardPower,
    RangeResult,
    effective_drag_area_from_power,
    power_at_speed,
    range_estimate,
)

__all__ = [
    "DEFAULT_DRAG_AREA",
    "FOM",
    "G",
    "RHO",
    "ForwardPower",
    "RangeResult",
    "effective_drag_area_from_power",
    "power_at_speed",
    "range_estimate",
]
