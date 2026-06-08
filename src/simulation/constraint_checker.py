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

try:
    import syside as _syside
    _SYSIDE_OK = True
except ImportError:
    _syside = None      # type: ignore
    _SYSIDE_OK = False


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
# Syside-based attribute value extraction
# ---------------------------------------------------------------------------

def _extract_attribute_values_via_syside(sysml_text: str) -> Dict[str, float]:
    """
    Walk the syside AST and evaluate every AttributeUsage expression.

    Handles arithmetic expressions and unit-bearing literals that regex cannot
    (e.g. `mass * g`, `15.0 [m/s]`).  Returns {attribute_name: float_value}.
    Falls back to {} when syside is unavailable or parsing fails.
    """
    if not _SYSIDE_OK or not sysml_text:
        return {}
    out: Dict[str, float] = {}
    try:
        model, _ = _syside.try_load_model(sysml_source=sysml_text)
        compiler = _syside.Compiler()
        for attr in model.nodes(_syside.AttributeUsage):
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

# back-compat alias
_eval_op = eval_op


def check_constraints(
    constraints: List[ParsedConstraint],
    initial_values: Dict[str, Any],      # part → {attr → value}
    guard_variables: set,                # 已被状态机 guard 使用的变量名
) -> List[ConstraintCheckResult]:
    """
    评估 assert constraint 列表。

    initial_values: 所有 part 的 {attr_name: numeric_value} 合并字典
    guard_variables: behavioral_sim 已验证的变量名集合（这些的约束算 GUARD_VERIFIED）
    """
    results: List[ConstraintCheckResult] = []

    for c in constraints:
        expr_str = f"{c.lhs} {c.operator} {c.rhs}"

        # 解析两侧的值
        lhs_val = _try_float(c.lhs) or initial_values.get(c.lhs)
        rhs_val = _try_float(c.rhs) or initial_values.get(c.rhs)

        # 转 float
        if lhs_val is not None:
            try:
                lhs_val = float(lhs_val)
            except (TypeError, ValueError):
                lhs_val = None
        if rhs_val is not None:
            try:
                rhs_val = float(rhs_val)
            except (TypeError, ValueError):
                rhs_val = None

        # ── GUARD_VERIFIED：LHS 是状态机 guard 变量 ──────────────────────
        if c.lhs in guard_variables:
            results.append(ConstraintCheckResult(
                constraint_name=c.name,
                owner_part=c.owner_part,
                expression=expr_str,
                status="GUARD_VERIFIED",
                detail=f"'{c.lhs}' is a state machine guard variable — "
                       f"constraint verified by behavioral simulation.",
                lhs_value=lhs_val,
                rhs_value=rhs_val,
            ))
            continue

        # ── STATIC：两侧都有值，直接计算 ─────────────────────────────────
        if lhs_val is not None and rhs_val is not None:
            passed = _eval_op(lhs_val, c.operator, rhs_val)
            results.append(ConstraintCheckResult(
                constraint_name=c.name,
                owner_part=c.owner_part,
                expression=expr_str,
                status="PASS" if passed else "FAIL",
                detail=(
                    f"Static check: {lhs_val} {c.operator} {rhs_val} = "
                    f"{'true' if passed else 'FALSE'}"
                ),
                lhs_value=lhs_val,
                rhs_value=rhs_val,
            ))
            continue

        # ── UNCHECKED：运行时变量无法静态求值 ────────────────────────────
        missing = []
        if lhs_val is None:
            missing.append(f"'{c.lhs}' has no static value (runtime variable)")
        if rhs_val is None:
            missing.append(f"'{c.rhs}' not found in model attributes")

        results.append(ConstraintCheckResult(
            constraint_name=c.name,
            owner_part=c.owner_part,
            expression=expr_str,
            status="UNCHECKED",
            detail=f"Requires runtime data: {'; '.join(missing)}. "
                   f"Verify via SITL or physical simulation.",
            lhs_value=lhs_val,
            rhs_value=rhs_val,
        ))

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_constraint_checks(
    sysml_text: str,
    all_initial_values: Dict[str, float],
    guard_variables: Optional[set] = None,
) -> List[ConstraintCheckResult]:
    """
    完整流程：提取 → 评估。

    Parameters
    ----------
    sysml_text         : 完整 SysML 源文本
    all_initial_values : 所有 part 的属性初始值合并字典（{attr_name: float}）
    guard_variables    : behavioral_sim 用到的 guard 变量名（用于标注 GUARD_VERIFIED）
    """
    constraints = extract_constraints(sysml_text)
    if not constraints:
        return []

    # Syside-evaluated values supersede regex-extracted initial_values for
    # the same attribute name: the Compiler handles multi-operand arithmetic
    # and unit-bearing literals that the regex path cannot evaluate.
    syside_values = _extract_attribute_values_via_syside(sysml_text)
    merged_values = {**all_initial_values, **syside_values}

    return check_constraints(
        constraints,
        merged_values,
        guard_variables or set(),
    )


def format_constraint_report(results: List[ConstraintCheckResult]) -> str:
    """人可读的约束检查报告。"""
    if not results:
        return "No assert constraints found in model."

    counts = {"PASS": 0, "FAIL": 0, "GUARD_VERIFIED": 0, "UNCHECKED": 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1

    lines = [
        f"Parametric Constraint Check — {len(results)} constraints",
        f"  PASS={counts['PASS']}  FAIL={counts['FAIL']}  "
        f"GUARD_VERIFIED={counts['GUARD_VERIFIED']}  UNCHECKED={counts['UNCHECKED']}",
        "",
    ]
    for r in results:
        icon = {"PASS": "✓", "FAIL": "✗", "GUARD_VERIFIED": "✓", "UNCHECKED": "~"}.get(r.status, "?")
        lines.append(f"  {icon} [{r.status:<14}] {r.owner_part}.{r.constraint_name}")
        lines.append(f"       expr  : {r.expression}")
        lines.append(f"       detail: {r.detail}")
    return "\n".join(lines)
