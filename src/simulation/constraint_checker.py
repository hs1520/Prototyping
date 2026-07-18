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
_OWNER_RE = re.compile(
    r'//\s*OWNER:\s*(\w+)'
)

# 当前上下文：解析时向前看最近的 // OWNER: 注释
def _find_owner(text: str, match_start: int) -> str:
    """找 match_start 之前最近的 // OWNER: <Name>，返回 Part 名或 'unknown'。"""
    prefix = text[:match_start]
    owners = _OWNER_RE.findall(prefix)
    return owners[-1] if owners else "unknown"


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

