"""SysML v2 model representation package."""

from .model import SysMLModel
from .lite_model import SysMLLiteModel, build_lite_model

__all__ = [
    "SysMLModel",
    "SysMLLiteModel",
    "build_lite_model",
]
