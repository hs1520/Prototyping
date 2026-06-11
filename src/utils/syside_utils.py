"""Shared syside AST utilities used across the package."""
from __future__ import annotations

from typing import Dict

try:
    import syside
    SYSIDE_OK = True
except ImportError:
    syside = None  # type: ignore
    SYSIDE_OK = False


def extract_attr_values(text: str) -> Dict[str, float]:
    """Evaluate every AttributeUsage expression in *text* via the syside Compiler.

    Returns {attribute_name: float_value}.  Used to supplement IR model
    attribute values that may be unparsed expressions (e.g. ``mass * g``).
    Falls back to {} when syside is unavailable or parsing fails.
    """
    if not SYSIDE_OK or not text:
        return {}
    out: Dict[str, float] = {}
    try:
        model, _ = syside.try_load_model(sysml_source=text)
        compiler = syside.Compiler()
        for attr in model.nodes(syside.AttributeUsage):
            try:
                expr = attr.feature_value_expression
                if expr is None:
                    continue
                val, report = compiler.evaluate(expr)
                if not report.fatal and val is not None:
                    out[attr.name] = float(val)
            except Exception:
                pass
    except Exception:
        pass
    return out
