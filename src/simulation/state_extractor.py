"""
state_extractor.py

用 syside 原生 API 从 SysML v2 文本中提取状态机定义。
不依赖我们自己的 Syside_AST_Parser，因为 model.py 目前不含 StateDefinition。

提取内容：
  - StateMachineDef  — 一个 state def 块
  - StateNode        — 一个状态（含 entry action 名）
  - TransitionDef    — 一条转移（含 guard 条件）
  - GuardCondition   — guard 的结构化表示（支持数值比较 / 布尔 / 复合 AND/OR）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import syside as _syside
    _SYSIDE_OK = True
except ImportError:
    _syside = None          # type: ignore
    _SYSIDE_OK = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GuardCondition:
    """
    结构化的 guard 条件。

    kind == 'comparison' : attribute operator threshold
        e.g. batteryCharge < 15.0
    kind == 'bool_true'  : attribute (boolean flag must be True)
        e.g. sensorSelfTestFailed
    kind == 'compound'   : compound_op over operands
        e.g. channelAFailed and channelBFailed
    """
    kind: str                                         # 'comparison' | 'bool_true' | 'compound'
    attribute: str = ""                               # for comparison / bool_true
    operator: str = ""                                # '<' '<=' '>' '>=' '==' (comparison only)
    threshold: float = 0.0                            # RHS value (comparison only)
    compound_op: str = ""                             # 'and' | 'or' (compound only)
    operands: List["GuardCondition"] = field(default_factory=list)

    def description(self) -> str:
        if self.kind == "comparison":
            return f"{self.attribute} {self.operator} {self.threshold}"
        if self.kind == "bool_true":
            return f"{self.attribute} == true"
        if self.kind == "compound":
            sep = f" {self.compound_op} "
            return sep.join(op.description() for op in self.operands)
        return "?"

    def involved_attributes(self) -> List[str]:
        """Return all attribute names referenced by this guard."""
        if self.kind in ("comparison", "bool_true"):
            return [self.attribute] if self.attribute else []
        attrs: List[str] = []
        for op in self.operands:
            attrs.extend(op.involved_attributes())
        return attrs


@dataclass
class StateNode:
    name: str
    entry_action: Optional[str] = None   # name of the entry action, or None


@dataclass
class TransitionDef:
    name: Optional[str]
    source: Optional[str]          # source state name; None → initial transition
    target: Optional[str]          # target state name
    guards: List[GuardCondition] = field(default_factory=list)
    is_initial: bool = False


@dataclass
class StateMachineDef:
    name: str
    owner_part: str
    states: List[StateNode] = field(default_factory=list)
    transitions: List[TransitionDef] = field(default_factory=list)
    initial_state: Optional[str] = None
    # Initial attribute values extracted from the owner part's attributes
    initial_values: Dict[str, Any] = field(default_factory=dict)

    def fault_transitions(self) -> List[TransitionDef]:
        """Non-initial transitions that have guard conditions."""
        return [t for t in self.transitions if not t.is_initial and t.guards]

    def entry_action_for_state(self, state_name: str) -> Optional[str]:
        for s in self.states:
            if s.name == state_name:
                return s.entry_action
        return None


# ---------------------------------------------------------------------------
# Guard extraction (recursive)
# ---------------------------------------------------------------------------

def _extract_guard(expr) -> Optional[GuardCondition]:
    """
    Recursively convert a syside expression node into a GuardCondition.
    Returns None if the expression type is unrecognised.
    """
    if expr is None:
        return None

    tname = type(expr).__name__

    # ── Boolean feature reference: e.g. `sensorSelfTestFailed` ───────────────
    if tname == "FeatureReferenceExpression":
        ref = expr.referent
        attr = ref.name if ref else None
        if attr:
            return GuardCondition(kind="bool_true", attribute=attr)
        return None

    # ── Operator expression: numeric comparison or compound boolean ───────────
    if tname == "OperatorExpression":
        op = str(expr.operator).strip()
        args = list(expr.arguments)

        # Compound AND / OR
        _AND_OPS = {"and", "&", "&&"}
        _OR_OPS  = {"or",  "|", "||"}
        if op in _AND_OPS or op in _OR_OPS:
            if len(args) == 2:
                left  = _extract_guard(args[0])
                right = _extract_guard(args[1])
                if left and right:
                    compound_op = "and" if op in _AND_OPS else "or"
                    return GuardCondition(
                        kind="compound",
                        compound_op=compound_op,
                        operands=[left, right],
                    )
            return None

        # Numeric comparison: <  <=  >  >=  ==
        if op in ("<", "<=", ">", ">=", "==", "!=") and len(args) == 2:
            lhs, rhs = args[0], args[1]

            # LHS: feature reference → attribute name
            attr_name: Optional[str] = None
            if type(lhs).__name__ == "FeatureReferenceExpression":
                ref = lhs.referent
                attr_name = ref.name if ref else None

            # RHS: literal value
            threshold: Optional[float] = None
            rhs_type = type(rhs).__name__
            if rhs_type in ("LiteralRational", "LiteralInteger", "LiteralReal"):
                try:
                    threshold = float(rhs.value)
                except (TypeError, ValueError):
                    pass
            elif rhs_type == "LiteralBoolean":
                # e.g.  attr == true / attr == false
                if attr_name:
                    val = bool(rhs.value)
                    if op == "==" and val:
                        return GuardCondition(kind="bool_true", attribute=attr_name)
                    # For == false or != true, skip (unusual in safety specs)
                return None

            if attr_name is not None and threshold is not None:
                return GuardCondition(
                    kind="comparison",
                    attribute=attr_name,
                    operator=op,
                    threshold=threshold,
                )

    return None


# ---------------------------------------------------------------------------
# Initial attribute value extraction
# ---------------------------------------------------------------------------

def _extract_part_attrs(part_def) -> Dict[str, Any]:
    """
    Extract initial attribute values from a syside PartDefinition node.
    Returns a dict of {attr_name: numeric_or_bool_value}.
    """
    result: Dict[str, Any] = {}
    try:
        for attr in part_def.owned_attributes:
            name = attr.name
            if not name:
                continue
            fve = attr.feature_value_expression
            if fve is None:
                continue
            fve_type = type(fve).__name__
            if fve_type in ("LiteralRational", "LiteralInteger", "LiteralReal"):
                try:
                    result[name] = float(fve.value)
                except (TypeError, ValueError):
                    pass
            elif fve_type == "LiteralBoolean":
                result[name] = bool(fve.value)
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------

def extract_state_machines(sysml_text: str) -> List[StateMachineDef]:
    """
    Parse *sysml_text* with the syside native API and return all
    StateMachineDef objects found in the model.

    Returns an empty list if syside is unavailable or parsing fails.
    """
    if not _SYSIDE_OK:
        return []

    try:
        model, _diags = _syside.try_load_model(sysml_source=sysml_text)
    except Exception:
        return []

    result: List[StateMachineDef] = []

    for sd in model.elements(_syside.StateDefinition):
        owner = sd.owner
        owner_name: str = owner.name if owner else "__unknown__"

        sm = StateMachineDef(
            name=sd.name,
            owner_part=owner_name,
            initial_values=_extract_part_attrs(owner) if owner else {},
        )

        # ── States ────────────────────────────────────────────────────────────
        for st in sd.owned_states:
            entry_name: Optional[str] = None
            ea = st.entry_action
            if ea:
                entry_name = ea.name
            sm.states.append(StateNode(name=st.name, entry_action=entry_name))

        # ── Transitions ───────────────────────────────────────────────────────
        for tr in sd.owned_transitions:
            src  = tr.source
            tgt  = tr.target
            src_name = src.name if src else None
            tgt_name = tgt.name if tgt else None
            is_initial = src_name is None

            guards: List[GuardCondition] = []
            guard_expr = tr.guard_expression
            if guard_expr is not None:
                g = _extract_guard(guard_expr)
                if g:
                    guards.append(g)

            td = TransitionDef(
                name=tr.name,
                source=src_name,
                target=tgt_name,
                guards=guards,
                is_initial=is_initial,
            )
            sm.transitions.append(td)

            if is_initial and tgt_name:
                sm.initial_state = tgt_name

        result.append(sm)

    return result
