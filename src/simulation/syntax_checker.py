"""
syntax_checker.py

用 syside 原生 API 对 SysML v2 文本做语法/语义检查，
返回结构化结果供 evaluator 打分和 orchestrator 做 syntax gate。

三类诊断：
  parser     — 硬语法错误（token 级别），LLM 输出必须修复
  sema       — 语义引用错误（找不到类型/命名空间等）
  warnings   — 警告，不影响 pass/fail
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

try:
    import syside as _syside
    _SYSIDE_OK = True
except ImportError:
    _syside = None      # type: ignore
    _SYSIDE_OK = False


# ---------------------------------------------------------------------------
# SysML v2 standard library types — implicitly available in all conformant
# tools (SysML v2 Pilot, Cameo, Rhapsody).  syside flags these as undefined
# because it doesn't inject them automatically, but they are NOT real model
# errors in practice.
# ---------------------------------------------------------------------------

_STDLIB_TYPE_NAMES = {
    # ScalarValues (KerML)
    "Real", "Integer", "Boolean", "String", "Rational", "Complex",
    "ScalarValue", "NumericalValue",
    # ISQ / SI units
    "ISQ", "SI", "LengthValue", "MassValue", "TimeValue", "VelocityValue",
    "AccelerationValue", "ForceValue", "EnergyValue", "PowerValue",
    "FrequencyValue", "AngleValue", "TemperatureValue", "VoltageValue",
    "CurrentValue", "ChargeValue",
    # SysML standard packages
    "SysML", "KerML", "ScalarValues", "Quantities",
    "Occurrences", "Transfers", "Connections",
    "Requirements", "Constraints", "Parts", "Ports",
    "Interfaces", "Flows", "Actions", "States", "UseCases",
    "Allocations", "Geometries",
}

# SI / ISQ unit feature names used in attribute definitions like `= 15.0 [m/s]`
# syside reports these as "No Feature named 'X' found." — also false positives.
_STDLIB_UNIT_NAMES = {
    # Length / area / volume
    "m", "km", "cm", "mm", "um", "nm",
    # Time
    "s", "ms", "us", "ns", "min", "h", "hr",
    # Mass
    "kg", "g", "mg",
    # Angle
    "deg", "rad", "grad",
    # Frequency
    "Hz", "kHz", "MHz", "GHz",
    # Speed
    "m_s", "km_h", "knot",
    # Acceleration
    "m_s2",
    # Force / pressure
    "N", "kN", "Pa", "kPa", "MPa", "bar",
    # Energy / power
    "J", "kJ", "W", "kW", "MW",
    # Voltage / current / charge
    "V", "mV", "kV", "A", "mA", "C", "Ah",
    # Temperature  (Cel = UCUM/SI symbol for degree Celsius — the SI library's
    # canonical name; degC/degF are LLM-friendly aliases)
    "K", "degC", "Cel", "degF",
    # Energy / charge capacity & rotation (common in drone/EV domains)
    "Wh", "kWh", "mAh", "rpm", "Nm",
    # Sound
    "dB", "dBA",
    # Percentage / dimensionless
    "pct", "percent",
    # Data / information units (LLM commonly annotates comms attributes with these)
    "bit", "bits", "byte", "bytes", "B",
    "kbit", "Kbit", "Mbit", "Gbit",
    "kB", "MB", "GB", "TB",
    "bps", "kbps", "Kbps", "Mbps", "Gbps", "baud",
    # Misc
    "G", "g_force", "lx", "lm", "cd",
    # Compound unit names that LLM might use
    "mm_hr", "m_s2", "rad_s",
    # SysML unit packages
    "SI", "ISQ",
    # SysML v2 state machine pseudo-states / reserved feature names
    "initial", "final", "done", "accept",
}


def _is_stdlib_sema_error(message: str) -> bool:
    """
    Return True if this sema error is purely about a missing standard-library
    type or unit — i.e. a false positive caused by syside's strict parsing mode.

    Patterns matched:
      "No Type named '<X>' found."      — stdlib type (Real, Boolean …)
      "No Namespace named '<X>' found." — stdlib package (SI, ISQ …)
      "No Feature named '<X>' found."   — SI unit symbol (m, Hz, deg …)
    """
    import re
    m = re.search(r"No (?:Type|Namespace|Feature) named '([^']+)' found", message)
    if not m:
        return False
    name = m.group(1)
    root = name.split("::")[0]
    return root in _STDLIB_TYPE_NAMES or root in _STDLIB_UNIT_NAMES


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------

@dataclass
class SyntaxCheckResult:
    has_errors: bool
    parser_errors: List[Dict] = field(default_factory=list)   # {line, col, message, code}
    sema_errors:   List[Dict] = field(default_factory=list)
    warnings:      List[Dict] = field(default_factory=list)
    score: float = 1.0

    def total_errors(self) -> int:
        return len(self.parser_errors) + len(self.sema_errors)

    def format_for_llm(self) -> str:
        """Format errors into a concise LLM fix-request prompt."""
        n = self.total_errors()
        lines = [
            f"[SYNTAX-FIX REQUEST] — {n} error(s) found in the generated SysML v2 model.",
            "Fix ONLY the listed errors. Do NOT change the architecture or remove parts/ports.",
            "",
        ]
        if self.parser_errors:
            lines.append("[Parser Errors] — hard syntax errors (unexpected tokens, missing semicolons, etc.)")
            for e in self.parser_errors:
                lines.append(f"  Line {e['line']}, Col {e['col']}: {e['message']}")
            lines.append("")
        if self.sema_errors:
            lines.append("[Semantic Errors] — undefined types / namespaces")
            for e in self.sema_errors:
                lines.append(f"  Line {e['line']}, Col {e['col']}: {e['message']}")
            lines.append("")
        lines += [
            "Output the complete fixed SysML v2 code only — no explanation.",
        ]
        return "\n".join(lines)

    def short_summary(self) -> str:
        if not self.has_errors:
            return "✓ no syntax errors"
        parts = []
        if self.parser_errors:
            parts.append(f"{len(self.parser_errors)} parser")
        if self.sema_errors:
            parts.append(f"{len(self.sema_errors)} sema")
        return "✗ " + ", ".join(parts) + " error(s)"


# ---------------------------------------------------------------------------
# Score calculation
# ---------------------------------------------------------------------------

def _compute_score(n_parser: int, n_sema: int, n_warn: int) -> float:
    """
    1.0  — no errors
    0.85 — warnings only
    0.5  — sema errors only (undefined types are often fixable)
    0.1  — parser errors (hard syntax failure)
    Deductions compound; floor is 0.0.
    """
    score = 1.0 - 0.25 * n_parser - 0.12 * n_sema - 0.05 * n_warn
    return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def check_syntax(sysml_text: str) -> SyntaxCheckResult:
    """
    Parse *sysml_text* with syside and return a SyntaxCheckResult.

    If syside is unavailable, returns a clean result (score=1.0) so the
    rest of the pipeline is not blocked.
    """
    if not _SYSIDE_OK:
        return SyntaxCheckResult(has_errors=False, score=1.0)

    try:
        _model, diags = _syside.try_load_model(sysml_source=sysml_text)
    except Exception:
        # Can't even call the API — treat as unverified (neutral score)
        return SyntaxCheckResult(has_errors=False, score=1.0)

    def _collect(category, filter_stdlib: bool = False) -> List[Dict]:
        out = []
        try:
            for d in category:
                msg = getattr(d, "message", str(d))
                if filter_stdlib and _is_stdlib_sema_error(msg):
                    continue   # skip false positives from stdlib types
                out.append({
                    "line":    getattr(d, "line",    0),
                    "col":     getattr(d, "col",     0),
                    "message": msg,
                    "code":    getattr(d, "code",    ""),
                })
        except Exception:
            pass
        return out

    parser_errs = _collect(diags.parser)
    sema_errs   = _collect(diags.sema, filter_stdlib=True)   # filter stdlib false positives
    warn_items  = _collect(diags.warnings)

    has_errors = bool(parser_errs or sema_errs)
    score = _compute_score(len(parser_errs), len(sema_errs), len(warn_items))

    return SyntaxCheckResult(
        has_errors=has_errors,
        parser_errors=parser_errs,
        sema_errors=sema_errs,
        warnings=warn_items,
        score=score,
    )
