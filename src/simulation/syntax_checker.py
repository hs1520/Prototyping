"""syntax_checker.py"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

from ..sysml.diagnostics import is_stdlib_sema_error
from ..utils.suppressed import record_suppressed

try:
    import syside as _syside
    _SYSIDE_OK = True
except ImportError:
    _syside = None      # type: ignore
    _SYSIDE_OK = False


_EXPECTED_SET_RE = re.compile(r"expected one of \[(.*)\]", re.S)
_QUOTED_TOKEN_RE = re.compile(r'"(?:[^"\\]|\\.)*"')

_KEPT_ALTERNATIVES = 6


def condense_diagnostic(message: str, *, kept: int = _KEPT_ALTERNATIVES) -> str:
    """Shorten a syside parser diagnostic for a repair prompt, not for logs.

    Syside reports a parse failure with the whole expected-terminal set of its
    current state - around 200 alternatives, ~2400 characters, including grammar
    rule names like ``Dependency_repeat1`` - longer than the code chunk it is
    attached to, and not discriminating: one measured run listed the rejected
    token inside its own expected set. The head (which token was rejected, and
    where) is kept and the tail cut to a handful of alternatives. Prompt-budget
    measure only: raw messages stay verbatim in the console, the run artifacts
    and the diagnostics record.
    """
    text = str(message or "")
    match = _EXPECTED_SET_RE.search(text)
    if not match:
        return text
    alternatives = _QUOTED_TOKEN_RE.findall(match.group(1))
    if len(alternatives) <= kept:
        return text
    remaining = len(alternatives) - kept
    condensed = (
        "expected one of ["
        + ", ".join(alternatives[:kept])
        + f", … +{remaining} more]"
    )
    return text[:match.start()] + condensed + text[match.end():]


@dataclass
class SyntaxCheckResult:
    has_errors: bool
    parser_errors: List[Dict] = field(default_factory=list)
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


def _compute_score(n_parser: int, n_sema: int, n_warn: int) -> float:
    """1.0  - no diagnostics
    >=0.5 - warnings only (0.05 each, floored at 0.5: the old linear formula
           sent 46 warnings with zero errors to 0.0 - the same score as a hard
           parse failure - which tripped the "fails compilation" veto and
           pinned refinement at the cap.)
    0.5  - sema errors only (undefined types are often fixable)
    0.1  - parser errors (hard syntax failure)
    With errors present, deductions compound; floor is 0.0.
    """
    if n_parser == 0 and n_sema == 0:
        return round(max(0.5, 1.0 - 0.05 * n_warn), 4)
    score = 1.0 - 0.25 * n_parser - 0.12 * n_sema - 0.05 * n_warn
    return round(max(0.0, min(1.0, score)), 4)


def check_syntax(
    sysml_text: str,
    *,
    fail_closed: bool = False,
    filter_stdlib_diagnostics: bool = True,
) -> SyntaxCheckResult:
    """Parse *sysml_text* with syside and return a SyntaxCheckResult.

    Legacy callers keep the best-effort behavior. Evidence-producing paths pass
    ``fail_closed=True`` and ``filter_stdlib_diagnostics=False`` so a missing
    tool, load failure or unresolved standard-library reference is an error
    rather than a synthetic PASS.
    """
    if not _SYSIDE_OK:
        if fail_closed:
            return SyntaxCheckResult(
                has_errors=True,
                sema_errors=[{
                    "line": 0,
                    "col": 0,
                    "message": "Syside Python API is unavailable",
                    "code": "SYSIDE_UNAVAILABLE",
                }],
                score=0.0,
            )
        return SyntaxCheckResult(has_errors=False, score=1.0)

    try:
        _model, diags = _syside.try_load_model(sysml_source=sysml_text)
    except Exception as exc:
        if fail_closed:
            return SyntaxCheckResult(
                has_errors=True,
                sema_errors=[{
                    "line": 0,
                    "col": 0,
                    "message": f"Syside model load failed: {exc}",
                    "code": "SYSIDE_LOAD_FAILED",
                }],
                score=0.0,
            )
        # Historical non-evidence callers treat tool failure as unverified/neutral.
        return SyntaxCheckResult(has_errors=False, score=1.0)

    def _collect(category, filter_stdlib: bool = False) -> List[Dict]:
        out = []
        try:
            for d in category:
                msg = getattr(d, "message", str(d))
                if filter_stdlib and is_stdlib_sema_error(msg):
                    continue
                out.append({
                    "line":    getattr(d, "line",    0),
                    "col":     getattr(d, "col",     0),
                    "message": msg,
                    "code":    getattr(d, "code",    ""),
                })
        except Exception as exc:
            record_suppressed("simulation.syntax_checker.diag_collect", exc)
        return out

    parser_errs = _collect(diags.parser)
    sema_errs = _collect(
        diags.sema,
        filter_stdlib=filter_stdlib_diagnostics,
    )
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
