"""Architecture DSE operators (bilevel outer-layer actions).

Each operator resolves a SysML v2 ``variation`` point (or is a gated generative
operator) into valid-by-construction model structure. See docs/DSE_OPERATORS.md.
"""
from .redundantize import CATALOG, RedundantizeComponent
from .decompose import DecomposeController
from .protocol import ReplaceInterfaceProtocol
from .sensing import AddRedundantSensor

__all__ = [
    "RedundantizeComponent",
    "DecomposeController",
    "ReplaceInterfaceProtocol",
    "AddRedundantSensor",
    "CATALOG",
]
