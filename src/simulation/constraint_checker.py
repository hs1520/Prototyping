"""
constraint_checker.py

从 SysML 文本里提取 assert constraint 块并评估。

评估分三类：
  STATIC    — 两侧都是 initial_values 里的已知常量，直接计算
  GUARD     — LHS 是状态机 guard 变量（运行时变量），由 behavioral_sim 已验证
  UNCHECKED — LHS 是运行时变量但无对应 guard，需要外部仿真才能验证

属性值提取：优先使用 syside Compiler 精确求值（支持算术表达式和单位），
          降级到调用方传入的 initial_values 字典。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..utils.sysml_text_utils import find_block_end


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ParsedConstraint:
    name: str
    owner_part: str
    lhs: str          # 变量名
    operator: str     # <=, >=, <, >, ==
    rhs: str          # 变量名或字面量
    raw_expr: str
    plan_constraint_id: Optional[str] = None
    provenance: str = "UNSPECIFIED"
    activation: str = "ALWAYS"
    verification_tier: str = "PARAMETRIC_SWEEP"
    activation_ref: Optional[str] = None
    containing_behavior: Optional[str] = None
    containing_state: Optional[str] = None


@dataclass
class ConstraintCheckResult:
    constraint_name: str
    owner_part: str
    expression: str
    status: str        # "PASS" | "FAIL" | "UNCHECKED" | "GUARD_VERIFIED"
    detail: str        # 人可读的说明
    lhs_value: Optional[float] = None
    rhs_value: Optional[float] = None


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_ASSERT_RE = re.compile(
    r'assert\s+constraint\s+(\w+)\s*\{([^}]+)\}',
    re.DOTALL,
)
_EXPR_RE = re.compile(
    r'(\w+)\s*(<=|>=|<|>|==)\s*([\w.]+)'
)
_PART_RE = re.compile(
    r'\bpart\s+def\s+([A-Za-z_]\w*)\s*\{'
)
_PLAN_RE = re.compile(
    r'//\s*PLAN-CONSTRAINT\s+([A-Za-z_]\w*)\s+'
    r'provenance=([A-Z_]+)\s+activation=([A-Z_]+)\s+'
    r'verification=([A-Z_]+)'
    r'(?:\s+reference=([A-Za-z_]\w*::[A-Za-z_]\w*))?'
)

def _find_owner(text: str, match_start: int) -> str:
    """Resolve the innermost enclosing part definition."""
    owners: List[Tuple[str, int]] = []
    for match in _PART_RE.finditer(text):
        opening = text.find("{", match.start(), match.end())
        closing = find_block_end(text, opening)
        if closing != -1 and match.start() < match_start < closing:
            owners.append((match.group(1), match.start()))
    return max(owners, key=lambda item: item[1])[0] if owners else "unknown"


def _find_plan_metadata(
    text: str, match_start: int
) -> Tuple[Optional[str], str, str, str, Optional[str]]:
    line_start = text.rfind("\n", 0, match_start) + 1
    previous_line_start = text.rfind(
        "\n", 0, max(0, line_start - 1)
    ) + 1
    previous = text[previous_line_start:line_start]
    matches = list(_PLAN_RE.finditer(previous))
    if not matches:
        return None, "UNSPECIFIED", "ALWAYS", "PARAMETRIC_SWEEP", None
    match = matches[-1]
    return (
        match.group(1),
        match.group(2),
        match.group(3),
        match.group(4),
        match.group(5),
    )


def _find_state_context(
    text: str,
    match_start: int,
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve enclosing state definition and state usage lexically."""
    behavior: Optional[Tuple[str, int, int]] = None
    for match in re.finditer(
        r"\bstate\s+def\s+([A-Za-z_]\w*)\s*\{", text
    ):
        opening = text.find("{", match.start(), match.end())
        closing = find_block_end(text, opening)
        if opening < match_start < closing:
            candidate = (match.group(1), opening, closing)
            if behavior is None or opening > behavior[1]:
                behavior = candidate
    if behavior is None:
        return None, None
    state: Optional[Tuple[str, int]] = None
    for match in re.finditer(
        r"\bstate\s+(?!def\b)([A-Za-z_]\w*)\s*\{",
        text[behavior[1] + 1:behavior[2]],
    ):
        absolute_start = behavior[1] + 1 + match.start()
        opening = text.find(
            "{",
            absolute_start,
            behavior[1] + 1 + match.end(),
        )
        closing = find_block_end(text, opening)
        if opening < match_start < closing:
            if state is None or opening > state[1]:
                state = (match.group(1), opening)
    return behavior[0], state[0] if state else None


def extract_constraints(sysml_text: str) -> List[ParsedConstraint]:
    """
    从 SysML 文本里提取所有 assert constraint 块。
    返回 ParsedConstraint 列表。
    """
    results: List[ParsedConstraint] = []
    for m in _ASSERT_RE.finditer(sysml_text):
        name      = m.group(1)
        raw_expr  = m.group(2).strip()
        owner     = _find_owner(sysml_text, m.start())
        plan_id, provenance, activation, verification, activation_ref = (
            _find_plan_metadata(sysml_text, m.start())
        )
        containing_behavior, containing_state = _find_state_context(
            sysml_text, m.start()
        )

        expr_m = _EXPR_RE.search(raw_expr)
        if not expr_m:
            continue

        results.append(ParsedConstraint(
            name=name,
            owner_part=owner,
            lhs=expr_m.group(1),
            operator=expr_m.group(2),
            rhs=expr_m.group(3),
            raw_expr=raw_expr,
            plan_constraint_id=plan_id,
            provenance=provenance,
            activation=activation,
            verification_tier=verification,
            activation_ref=activation_ref,
            containing_behavior=containing_behavior,
            containing_state=containing_state,
        ))
    return results


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def _try_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def eval_op(lhs_val: float, op: str, rhs_val: float) -> bool:
    if op == "<=": return lhs_val <= rhs_val
    if op == ">=": return lhs_val >= rhs_val
    if op == "<":  return lhs_val <  rhs_val
    if op == ">":  return lhs_val >  rhs_val
    if op == "==": return lhs_val == rhs_val
    return False



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
