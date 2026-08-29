"""Shared syside AST utilities used across the package."""
from __future__ import annotations

from typing import Dict

from .suppressed import record_suppressed

try:
    import syside
    SYSIDE_OK = True
except ImportError:
    syside = None  # type: ignore
    SYSIDE_OK = False


def coerce_static_number(value) -> float | None:
    """A static scalar from a syside evaluation result, or None.

    ``Compiler.evaluate`` returns the referenced node itself (an
    ``AttributeUsage``) when an initializer is a feature-reference chain — the
    typed semantic bindings write exactly those (``attribute currentX : T =
    channel.payload.feature;``).  Such an initializer has no static scalar;
    that is data, not an error, so callers must not ``float()`` blindly (the
    ablation pilot recorded 369 suppressed TypeErrors from three sites doing
    just that).
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


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
                if not report.fatal:
                    number = coerce_static_number(val)
                    if number is not None:
                        out[attr.name] = number
            except Exception as exc:
                record_suppressed("utils.syside_utils.attr_eval", exc)
    except Exception as exc:
        record_suppressed("utils.syside_utils.attr_load", exc)
    return out
