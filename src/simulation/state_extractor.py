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
# Expression tree (Layer 1 — supports variable/arithmetic RHS in guards)
# ---------------------------------------------------------------------------
#
# A guard comparison's two sides are represented as expression trees so that
# guards like `batteryCharge <= returnEnergyRequired` (variable RHS) or
# `commLossTime > timeToHub + 300.0` (arithmetic RHS) are captured fully,
# instead of being dropped when the RHS is not a bare literal.


@dataclass
class Expr:
    """Base expression node."""
    def eval(self, env: Dict[str, Any]) -> Optional[float]:  # pragma: no cover
        raise NotImplementedError

    def vars(self) -> List[str]:  # pragma: no cover
        raise NotImplementedError

    def render(self) -> str:  # pragma: no cover
        raise NotImplementedError


@dataclass
class Const(Expr):
    value: float

    def eval(self, env: Dict[str, Any]) -> Optional[float]:
        return self.value

    def vars(self) -> List[str]:
        return []

    def render(self) -> str:
        return f"{self.value:g}"


@dataclass
class VarRef(Expr):
    name: str

    def eval(self, env: Dict[str, Any]) -> Optional[float]:
        v = env.get(self.name)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def vars(self) -> List[str]:
        return [self.name]

    def render(self) -> str:
        return self.name


@dataclass
class BinOp(Expr):
    op: str            # '+' '-' '*' '/'
    left: Expr
    right: Expr

    def eval(self, env: Dict[str, Any]) -> Optional[float]:
        a = self.left.eval(env)
        b = self.right.eval(env)
        if a is None or b is None:
            return None
        if self.op == "+": return a + b
        if self.op == "-": return a - b
        if self.op == "*": return a * b
        if self.op == "/": return a / b if b != 0 else None
        return None

    def vars(self) -> List[str]:
        return self.left.vars() + self.right.vars()

    def render(self) -> str:
        return f"({self.left.render()} {self.op} {self.right.render()})"


_ARITH_OPS = {"+", "-", "*", "/"}


def _to_expr(node) -> Optional[Expr]:
    """Convert a syside expression node into an Expr, or None if unsupported."""
    if node is None:
        return None
    tname = type(node).__name__

    if tname == "FeatureReferenceExpression":
        ref = node.referent
        name = ref.name if ref else None
        return VarRef(name) if name else None

    if tname in ("LiteralRational", "LiteralInteger", "LiteralReal"):
        try:
            return Const(float(node.value))
        except (TypeError, ValueError):
            return None

    if tname == "OperatorExpression":
        op = str(node.operator).strip()
        args = list(node.arguments)
        if op in _ARITH_OPS and len(args) == 2:
            left = _to_expr(args[0])
            right = _to_expr(args[1])
            if left is not None and right is not None:
                return BinOp(op, left, right)
        return None

    return None


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
    kind == 'enum_eq'    : attribute == EnumType::Value  (Layer 2)
        e.g. flightMode == DroneMode::SELF_TEST
    """
    kind: str                                         # 'comparison' | 'bool_true' | 'compound' | 'enum_eq'
    attribute: str = ""                               # LHS variable name (comparison / bool_true / enum_eq)
    operator: str = ""                                # '<' '<=' '>' '>=' (comparison only)
    threshold: float = 0.0                            # effective RHS constant (comparison only)
    compound_op: str = ""                             # 'and' | 'or' (compound only)
    operands: List["GuardCondition"] = field(default_factory=list)
    # Layer 1: full expression trees for a comparison's two sides.
    lhs: Optional[Expr] = None
    rhs: Optional[Expr] = None
    # Layer 2: enum equality fields
    enum_type: str = ""                               # e.g. "DroneMode"
    enum_value: str = ""                              # e.g. "SELF_TEST"

    def description(self) -> str:
        if self.kind == "comparison":
            if self.rhs is not None and self.lhs is not None:
                return f"{self.lhs.render()} {self.operator} {self.rhs.render()}"
            return f"{self.attribute} {self.operator} {self.threshold}"
        if self.kind == "bool_true":
            return f"{self.attribute} == true"
        if self.kind == "enum_eq":
            return f"{self.attribute} == {self.enum_type}::{self.enum_value}"
        if self.kind == "compound":
            sep = f" {self.compound_op} "
            return sep.join(op.description() for op in self.operands)
        return "?"

    def involved_attributes(self) -> List[str]:
        """Return all attribute names referenced by this guard (lhs + rhs)."""
        if self.kind == "bool_true":
            return [self.attribute] if self.attribute else []
        if self.kind == "enum_eq":
            return [self.attribute] if self.attribute else []
        if self.kind == "comparison":
            names: List[str] = []
            if self.lhs is not None:
                names += self.lhs.vars()
            if self.rhs is not None:
                names += self.rhs.vars()
            if not names and self.attribute:
                names = [self.attribute]
            # de-dupe preserving order
            seen: set = set()
            return [n for n in names if not (n in seen or seen.add(n))]
        attrs: List[str] = []
        for op in self.operands:
            attrs.extend(op.involved_attributes())
        return attrs

    def resolve_threshold(self, defaults: Dict[str, Any]) -> Optional[float]:
        """
        Partial-evaluate the RHS against *defaults* (the owner part's initial
        attribute values) to get an effective numeric threshold.

        Returns the constant, or None when the RHS cannot be reduced to a
        number (e.g. references a variable with no known default — genuinely
        dynamic, deferred to Layer 2/3).
        """
        if self.rhs is None:
            return None
        return self.rhs.eval(defaults)

    def eval(self, env: Dict[str, Any]) -> bool:
        """
        Evaluate the full guard against a complete variable environment.
        Used by the executor when lhs/rhs expression trees are available.
        """
        if self.kind == "comparison" and self.lhs is not None and self.rhs is not None:
            a = self.lhs.eval(env)
            b = self.rhs.eval(env)
            if a is None or b is None:
                return False
            op = self.operator
            if op == "<":  return a <  b
            if op == "<=": return a <= b
            if op == ">":  return a >  b
            if op == ">=": return a >= b
            if op == "==": return a == b
            if op == "!=": return a != b
            return False
        if self.kind == "bool_true":
            return bool(env.get(self.attribute, False))
        if self.kind == "enum_eq":
            # Compare current string value of the mode attribute against the target enum value.
            current = env.get(self.attribute)
            if current is None:
                return False
            return str(current) == self.enum_value
        if self.kind == "compound":
            if self.compound_op == "and":
                return all(op.eval(env) for op in self.operands)
            if self.compound_op == "or":
                return any(op.eval(env) for op in self.operands)
        return False


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

            # ── Boolean literal RHS: `attr == true` / `attr == false` ──────────
            if type(rhs).__name__ == "LiteralBoolean":
                attr_name = None
                if type(lhs).__name__ == "FeatureReferenceExpression":
                    ref = lhs.referent
                    attr_name = ref.name if ref else None
                if attr_name and op == "==" and bool(rhs.value):
                    return GuardCondition(kind="bool_true", attribute=attr_name)
                # `== false` / `!= true` — skip (handled as no-fault by Layer1)
                return None

            # ── Enum equality: `attr == EnumType::Value`  (Layer 2) ─────────────
            # syside represents EnumType::Value as a FeatureReferenceExpression
            # whose referent is an EnumerationUsage (not an AttributeUsage).
            if (op == "=="
                    and type(lhs).__name__ == "FeatureReferenceExpression"
                    and type(rhs).__name__ == "FeatureReferenceExpression"):
                lhs_ref = lhs.referent
                rhs_ref = rhs.referent
                if (rhs_ref is not None
                        and type(rhs_ref).__name__ == "EnumerationUsage"):
                    lhs_name = lhs_ref.name if lhs_ref else None
                    rhs_value = rhs_ref.name
                    rhs_owner = getattr(rhs_ref, "owner", None)
                    rhs_type = rhs_owner.name if rhs_owner else ""
                    if lhs_name and rhs_value:
                        return GuardCondition(
                            kind="enum_eq",
                            attribute=lhs_name,
                            enum_type=rhs_type,
                            enum_value=rhs_value,
                        )

            # ── General comparison: build expression trees for both sides ──────
            lhs_expr = _to_expr(lhs)
            rhs_expr = _to_expr(rhs)
            if lhs_expr is None or rhs_expr is None:
                return None

            # Back-compat: when LHS is a bare variable, expose it as `attribute`
            # so the existing single-variable driver keeps working.  `threshold`
            # is resolved later (after initial_values are known) — see
            # _resolve_guard_thresholds().
            lhs_var = lhs_expr.name if isinstance(lhs_expr, VarRef) else ""
            init_threshold = rhs_expr.eval({})  # resolves when RHS is constant
            return GuardCondition(
                kind="comparison",
                attribute=lhs_var,
                operator=op,
                threshold=init_threshold if init_threshold is not None else 0.0,
                lhs=lhs_expr,
                rhs=rhs_expr,
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
            elif fve_type == "FeatureReferenceExpression":
                # Enum-typed attribute: initial value is an EnumerationUsage
                # e.g. `attribute flightMode : DroneMode = DroneMode::POWER_ON`
                ref = getattr(fve, "referent", None)
                if ref is not None and type(ref).__name__ == "EnumerationUsage":
                    result[name] = ref.name  # store as string, e.g. "POWER_ON"
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

        # ── Resolve effective thresholds now that initial_values are known ──
        # A comparison whose RHS references an attribute (e.g.
        # `batteryCharge <= returnEnergyRequired`) or contains arithmetic
        # (`timeToHub + 300.0`) gets its `threshold` reduced to a constant
        # using the owner part's default attribute values.
        _resolve_guard_thresholds(sm)

        result.append(sm)

    return result


def _resolve_guard_thresholds(sm: StateMachineDef) -> None:
    """
    Walk every comparison guard and set `threshold` to the RHS partially
    evaluated against the owner part's initial attribute values.

    Leaves the original `threshold` (0.0 / literal) untouched when the RHS
    cannot be reduced to a number (genuinely dynamic — deferred to Layer 2/3).
    """
    def _walk(g: GuardCondition) -> None:
        if g.kind == "comparison":
            resolved = g.resolve_threshold(sm.initial_values)
            if resolved is not None:
                g.threshold = resolved
        elif g.kind == "compound":
            for op in g.operands:
                _walk(op)

    for tr in sm.transitions:
        for g in tr.guards:
            _walk(g)
