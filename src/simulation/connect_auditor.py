"""
connect_auditor.py

Programmatically audits every existing ``connect`` statement in a SysML model
text using the same five rules as connectivity_fixer.  Invalid connects are
removed from the text so downstream simulation sees a clean model.

Public API
──────────
  ConnectViolation  — one invalid connect with a reason
  AuditResult       — summary of the audit (has_violations, n_removed,
                      violations, cleaned_text)
  audit_connects(sysml_text) -> AuditResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set, Tuple

from .connectivity_fixer import (
    ConnectStmt,
    PortDirectory,
    _CONNECT_RE,
    build_port_directory,
    parse_connects,
)


@dataclass
class ConnectViolation:
    stmt: ConnectStmt
    reason: str

    def summary(self) -> str:
        return f"{self.stmt.to_sysml()} — {self.reason}"


@dataclass
class AuditResult:
    has_violations: bool
    n_removed: int
    violations: List[ConnectViolation] = field(default_factory=list)
    cleaned_text: str = ""


def audit_connects(sysml_text: str) -> AuditResult:
    """
    Audit all ``connect`` statements in *sysml_text* and remove invalid ones.

    Applies the five rules from connectivity_fixer:
      1. Both instances + ports must exist in the port directory.
      2. Source port direction: out or inout.  Target: in or inout.
      3. Port types must match.
      4. A pure ``in`` target port may have only one source.
      5. No duplicate connects.

    Returns an AuditResult with ``cleaned_text`` where invalid connects are
    removed and ``violations`` listing what was removed and why.
    """
    directory = build_port_directory(sysml_text)
    connects = parse_connects(sysml_text)

    violations: List[ConnectViolation] = []
    accepted_keys: Set[Tuple[str, str, str, str]] = set()
    driven_in: Set[Tuple[str, str]] = set()
    invalid_stmts: List[ConnectStmt] = []

    for stmt in connects:
        reason = _check_stmt(stmt, directory, accepted_keys, driven_in)
        if reason:
            violations.append(ConnectViolation(stmt=stmt, reason=reason))
            invalid_stmts.append(stmt)
        else:
            accepted_keys.add(stmt.key())
            tgt_info = directory.port(stmt.tgt_inst, stmt.tgt_port)
            if tgt_info is not None and tgt_info.direction == "in":
                driven_in.add((stmt.tgt_inst, stmt.tgt_port))

    cleaned = _remove_connects(sysml_text, invalid_stmts)

    return AuditResult(
        has_violations=bool(violations),
        n_removed=len(invalid_stmts),
        violations=violations,
        cleaned_text=cleaned,
    )


def _check_stmt(
    stmt: ConnectStmt,
    directory: PortDirectory,
    accepted_keys: Set[Tuple[str, str, str, str]],
    driven_in: Set[Tuple[str, str]],
) -> str:
    """Return a non-empty reason string if stmt violates any rule, else ''."""
    # Rule 5: duplicates
    if stmt.key() in accepted_keys:
        return "duplicate connect"

    # Rule 1: instances and ports must exist
    if not directory.has_instance(stmt.src_inst):
        return f"unknown instance '{stmt.src_inst}'"
    if not directory.has_instance(stmt.tgt_inst):
        return f"unknown instance '{stmt.tgt_inst}'"

    src_info = directory.port(stmt.src_inst, stmt.src_port)
    tgt_info = directory.port(stmt.tgt_inst, stmt.tgt_port)

    if src_info is None:
        return f"'{stmt.src_inst}' has no port '{stmt.src_port}'"
    if tgt_info is None:
        return f"'{stmt.tgt_inst}' has no port '{stmt.tgt_port}'"

    # Rule 2: direction
    if src_info.direction not in ("out", "inout"):
        return (f"source port '{stmt.src_port}' direction is "
                f"{src_info.direction} (need out/inout)")
    if tgt_info.direction not in ("in", "inout"):
        return (f"target port '{stmt.tgt_port}' direction is "
                f"{tgt_info.direction} (need in/inout)")

    # Rule 3: port type match
    if src_info.port_type != tgt_info.port_type:
        return f"port type mismatch ({src_info.port_type} ≠ {tgt_info.port_type})"

    # Rule 4: no double-source on pure in ports
    if tgt_info.direction == "in" and (stmt.tgt_inst, stmt.tgt_port) in driven_in:
        return f"target in-port '{stmt.tgt_inst}.{stmt.tgt_port}' already driven"

    return ""


def _remove_connects(sysml_text: str, to_remove: List[ConnectStmt]) -> str:
    """Remove lines in sysml_text that match invalid connect statements."""
    if not to_remove:
        return sysml_text

    remove_keys = {s.key() for s in to_remove}
    out_lines = []
    for line in sysml_text.split('\n'):
        m = _CONNECT_RE.search(line)
        if m:
            key = (m.group(1), m.group(2), m.group(3), m.group(4))
            if key in remove_keys:
                continue
        out_lines.append(line)
    return '\n'.join(out_lines)
