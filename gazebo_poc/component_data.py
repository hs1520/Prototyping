"""Compatibility re-export for the formal realization catalog."""
from __future__ import annotations

from src.realization.catalog import (
    CATALOG,
    MN5008_KV340_18x61,
    MotorPropCombo as MotorProp,
    MotorPropPoint,
)

__all__ = ["MotorPropPoint", "MotorProp", "MN5008_KV340_18x61", "CATALOG"]
