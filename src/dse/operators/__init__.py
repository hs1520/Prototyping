"""Architecture DSE operators (bilevel outer-layer actions)."""
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
