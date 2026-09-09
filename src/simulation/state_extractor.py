"""state_extractor.py"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..utils.suppressed import record_suppressed
from ..utils.sysml_text_utils import find_block_end, named_def_pattern

try:
    import syside as _syside
    _SYSIDE_OK = True
except ImportError:
    _syside = None          # type: ignore
    _SYSIDE_OK = False


# ---------------------------------------------------------------------------
# Expression tree (Layer 1 - supports variable/arithmetic RHS in guards)
# ---------------------------------------------------------------------------
#
# Both sides of a guard comparison are expression trees, so guards like
# `batteryCharge <= returnEnergyRequired` or `commLossTime > timeToHub + 300.0`
# are captured instead of dropped when the RHS is not a bare literal.


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
    op: str
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


@dataclass
class GuardCondition:
    """结构化的 guard 条件。"""
    kind: str
    attribute: str = ""
    operator: str = ""
    threshold: float = 0.0
    compound_op: str = ""
    operands: List["GuardCondition"] = field(default_factory=list)
    lhs: Optional[Expr] = None
    rhs: Optional[Expr] = None
    enum_type: str = ""
    enum_value: str = ""

    def description(self) -> str:
        if self.kind == "comparison":
            if self.rhs is not None and self.lhs is not None:
                return f"{self.lhs.render()} {self.operator} {self.rhs.render()}"
            return f"{self.attribute} {self.operator} {self.threshold}"
        if self.kind == "bool_true":
            return f"{self.attribute} == true"
        if self.kind == "bool_false":
            return f"{self.attribute} == false"
        if self.kind == "enum_eq":
            return f"{self.attribute} == {self.enum_type}::{self.enum_value}"
        if self.kind == "compound":
            sep = f" {self.compound_op} "
            return sep.join(op.description() for op in self.operands)
        return "?"

    def involved_attributes(self) -> List[str]:
        """Return all attribute names referenced by this guard (lhs + rhs)."""
        if self.kind in {"bool_true", "bool_false"}:
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
            return list(dict.fromkeys(names))
        attrs: List[str] = []
        for op in self.operands:
            attrs.extend(op.involved_attributes())
        return attrs

    def resolve_threshold(self, defaults: Dict[str, Any]) -> Optional[float]:
        """Partial-evaluate the RHS against *defaults* (the owner part's initial values)
        to get an effective numeric threshold.
        """
        if self.rhs is None:
            return None
        return self.rhs.eval(defaults)

    def eval(self, env: Dict[str, Any]) -> bool:
        """Evaluate the full guard against a complete variable environment."""
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
        if self.kind == "bool_false":
            return not bool(env.get(self.attribute, False))
        if self.kind == "enum_eq":
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
    entry_action: Optional[str] = None
    entry_action_def: Optional[str] = None
    do_action: Optional[str] = None
    do_action_def: Optional[str] = None
    sends: List[tuple] = field(default_factory=list)


@dataclass
class TransitionDef:
    name: Optional[str]
    source: Optional[str]
    target: Optional[str]
    guards: List[GuardCondition] = field(default_factory=list)
    is_initial: bool = False
    accept_trigger: Optional[str] = None


@dataclass
class StateMachineDef:
    name: str
    owner_part: str
    states: List[StateNode] = field(default_factory=list)
    transitions: List[TransitionDef] = field(default_factory=list)
    initial_state: Optional[str] = None
    initial_values: Dict[str, Any] = field(default_factory=dict)

    def fault_transitions(self) -> List[TransitionDef]:
        """Non-initial transitions that have guard conditions."""
        return [t for t in self.transitions if not t.is_initial and t.guards]

    def has_accept_transitions(self) -> bool:
        """True if any non-initial transition is driven by an accept trigger."""
        return any(
            t.accept_trigger for t in self.transitions if not t.is_initial
        )

    def entry_action_for_state(self, state_name: str) -> Optional[str]:
        for s in self.states:
            if s.name == state_name:
                return s.entry_action
        return None

    def do_action_for_state(self, state_name: str) -> Optional[str]:
        for state in self.states:
            if state.name == state_name:
                return state.do_action
        return None

    def response_action_for_state(self, state_name: str) -> Optional[str]:
        """Return the executable entry action, or otherwise the do action."""
        return (
            self.entry_action_for_state(state_name)
            or self.do_action_for_state(state_name)
        )

    def response_action_definition_for_state(
        self, state_name: str,
    ) -> Optional[str]:
        """Return the action definition invoked by a state's response usage."""
        for state in self.states:
            if state.name != state_name:
                continue
            return state.entry_action_def or state.do_action_def
        return None

    def all_sends(self) -> List[tuple]:
        """Return [(state_name, cmd_type, port_name)] for every fault-state send."""
        out = []
        for s in self.states:
            if s.entry_action or s.do_action:
                for cmd, port in s.sends:
                    out.append((s.name, cmd, port))
        return out


_STD_NAMESPACES = frozenset({
    "Actions::", "Occurrences::", "Base::",
    "Performances::", "Transfers::", "Links::",
})


def _extract_accept_trigger(tr) -> Optional[str]:
    actions = getattr(tr, "trigger_actions", None)
    if not actions:
        return None

    for action in actions:
        if "Accept" not in type(action).__name__:
            continue
        pp = getattr(action, "payload_parameter", None)
        if pp is None:
            continue
        try:
            for defn in pp.definitions:
                if type(defn).__name__ != "ActionDefinition":
                    continue
                qname = str(getattr(defn, "qualified_name", "") or "")
                if any(qname.startswith(ns) for ns in _STD_NAMESPACES):
                    continue
                name = getattr(defn, "name", None)
                if name:
                    return str(name)
        except Exception as exc:
            record_suppressed("simulation.state_extractor.action_def_lookup", exc)

    return None


def _extract_guard(expr) -> Optional[GuardCondition]:
    if expr is None:
        return None

    tname = type(expr).__name__

    if tname == "FeatureReferenceExpression":
        ref = expr.referent
        attr = ref.name if ref else None
        if attr:
            return GuardCondition(kind="bool_true", attribute=attr)
        return None

    if tname == "OperatorExpression":
        op = str(expr.operator).strip()
        args = list(expr.arguments)

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

        if op == "not" and len(args) == 1:
            operand = args[0]
            if type(operand).__name__ == "FeatureReferenceExpression":
                ref = operand.referent
                attr = ref.name if ref else None
                if attr:
                    return GuardCondition(kind="bool_false", attribute=attr)
            return None

        if op in ("<", "<=", ">", ">=", "==", "!=") and len(args) == 2:
            lhs, rhs = args[0], args[1]

            if type(rhs).__name__ == "LiteralBoolean":
                attr_name = None
                if type(lhs).__name__ == "FeatureReferenceExpression":
                    ref = lhs.referent
                    attr_name = ref.name if ref else None
                if attr_name and op == "==":
                    return GuardCondition(
                        kind="bool_true" if bool(rhs.value) else "bool_false",
                        attribute=attr_name,
                    )
                if attr_name and op == "!=":
                    return GuardCondition(
                        kind="bool_false" if bool(rhs.value) else "bool_true",
                        attribute=attr_name,
                    )
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

            lhs_expr = _to_expr(lhs)
            rhs_expr = _to_expr(rhs)
            if lhs_expr is None or rhs_expr is None:
                return None

            # Back-compat: when LHS is a bare variable, expose it as `attribute` so the
            # single-variable driver keeps working. `threshold` is resolved later, once
            # initial_values are known - see _resolve_guard_thresholds().
            lhs_var = lhs_expr.name if isinstance(lhs_expr, VarRef) else ""
            init_threshold = rhs_expr.eval({})
            return GuardCondition(
                kind="comparison",
                attribute=lhs_var,
                operator=op,
                threshold=init_threshold if init_threshold is not None else 0.0,
                lhs=lhs_expr,
                rhs=rhs_expr,
            )

    return None


def _numeric_default_value(expression) -> Optional[float]:
    if expression is None:
        return None
    expression_type = type(expression).__name__
    if expression_type in (
        "LiteralRational",
        "LiteralInteger",
        "LiteralReal",
    ):
        try:
            return float(expression.value)
        except (TypeError, ValueError):
            return None
    if expression_type == "OperatorExpression":
        operator = getattr(expression, "operator", None)
        operator_name = str(getattr(operator, "name", "") or "")
        operator_value = str(getattr(operator, "value", "") or "")
        if operator_name == "Quantity" or operator_value == "[":
            try:
                arguments = list(expression.arguments)
            except Exception:
                return None
            if arguments:
                return _numeric_default_value(arguments[0])
    return None


def _extract_part_attrs(part_def) -> Dict[str, Any]:
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
            numeric = _numeric_default_value(fve)
            if numeric is not None:
                result[name] = numeric
            elif fve_type == "LiteralBoolean":
                result[name] = bool(fve.value)
            elif fve_type == "FeatureReferenceExpression":
                ref = getattr(fve, "referent", None)
                if ref is not None and type(ref).__name__ == "EnumerationUsage":
                    result[name] = ref.name
    except Exception as exc:
        record_suppressed("simulation.state_extractor.initial_values", exc)
    return result


# Sentinel separating "candidate attribute absent in this syside build"
# (expected probing, silent) from "attribute present but access failed"
# (recorded via record_suppressed).
_CANDIDATE_ABSENT = object()


def _extract_send_payload_name(sa) -> Optional[str]:
    try:
        pa = sa.payload_argument
        if pa is None:
            return None
        # 按优先级尝试不同属性名。候选名在当前 syside 版本里缺席是预期探测结果，
        # 记录它会把 suppressed 通道刷出数千条噪声（ablation pilot: 324 条/run）。
        # 只有属性存在但访问失败才记录。
        for attr in ("action_definitions", "definitions", "types"):
            source = getattr(pa, attr, _CANDIDATE_ABSENT)
            if source is _CANDIDATE_ABSENT:
                continue
            try:
                for defn in source:
                    name = getattr(defn, "name", None)
                    if name:
                        return str(name)
            except Exception as exc:
                record_suppressed("simulation.state_extractor.performed_action_name", exc)
        ref = getattr(pa, "referent", None)
        if ref:
            name = getattr(ref, "name", None)
            if name:
                return str(name)
        return getattr(pa, "name", None)
    except Exception:
        return None


def _extract_send_receiver_name(sa) -> Optional[str]:
    try:
        ra = sa.receiver_argument
        if ra is None:
            return None
        ref = getattr(ra, "referent", None)
        if ref:
            name = getattr(ref, "name", None)
            if name:
                return str(name)
        return getattr(ra, "name", None)
    except Exception:
        return None


def _iter_action_body(action_usage) -> List:
    if action_usage is None:
        return []

    # 先拿到 ActionDefinition（typed by this usage）。候选属性名缺席（如当前
    # syside 没有 owned_actions）是预期探测结果，静默跳过；只有属性存在但迭代
    # 失败才进 suppressed（否则会刷出 5265 条/run 的噪声）。
    defs: List = []
    for attr in ("action_definitions", "definitions", "types"):
        source = getattr(action_usage, attr, _CANDIDATE_ABSENT)
        if source is _CANDIDATE_ABSENT:
            continue
        try:
            defs = [d for d in source]
            if defs:
                break
        except Exception as exc:
            record_suppressed("simulation.state_extractor.action_defs_fetch", exc)

    nodes: List = []
    seen_ids: set = set()

    for defn in defs:
        for attr in ("nested_actions", "owned_actions", "owned_members"):
            source = getattr(defn, attr, _CANDIDATE_ABSENT)
            if source is _CANDIDATE_ABSENT:
                continue
            try:
                for node in source:
                    nid = id(node)
                    if nid not in seen_ids:
                        seen_ids.add(nid)
                        nodes.append(node)
            except Exception as exc:
                record_suppressed("simulation.state_extractor.action_nodes_fetch", exc)

    return nodes


def _extract_action_definition_name(action_usage) -> Optional[str]:
    """Return the user action definition invoked by an action usage.

    SysML separates the usage label (``entry action updatePlan``) from its type
    (``: reviseWaypointSequence``); both are kept, the label for traces and the
    definition for response semantics.
    """
    if action_usage is None:
        return None
    for attr in ("action_definitions", "definitions", "types"):
        try:
            for defn in getattr(action_usage, attr):
                name = getattr(defn, "name", None)
                if name:
                    return str(name)
        except Exception as exc:
            record_suppressed("simulation.state_extractor.action_def_name", exc)
    return None


def _extract_send_usages(entry_action_usage) -> List[tuple]:
    result: List[tuple] = []
    if entry_action_usage is None:
        return result

    for node in _iter_action_body(entry_action_usage):
        if not (hasattr(node, "payload_argument") and hasattr(node, "receiver_argument")):
            continue
        # 也接受 type name 匹配（防止 hasattr 在 proxy 上误报）
        tname = type(node).__name__
        if tname not in ("SendActionUsage",) and not (
            hasattr(node, "payload_argument") and hasattr(node, "receiver_argument")
            and hasattr(node, "sender_argument")
        ):
            continue
        cmd = _extract_send_payload_name(node)
        port = _extract_send_receiver_name(node)
        if cmd and port:
            result.append((cmd, port))

    return result


def extract_state_machines(sysml_text: str) -> List[StateMachineDef]:
    """Parse *sysml_text* with the syside native API and return all StateMachineDefs."""
    if not _SYSIDE_OK:
        return []

    try:
        model, _diags = _syside.try_load_model(sysml_source=sysml_text)
    except Exception:
        return []

    result: List[StateMachineDef] = []

    for sd in model.elements(_syside.StateDefinition):
        state_source = sysml_text
        state_match = named_def_pattern("state", str(sd.name)).search(sysml_text)
        if state_match is not None:
            state_open = state_match.end() - 1
            state_close = find_block_end(sysml_text, state_open)
            if state_close != -1:
                state_source = sysml_text[
                    state_match.start():state_close + 1
                ]
        owner = sd.owner
        # A parse error upstream can reparent a state def under an unnamed
        # expression node, so an owner may exist with no name; fall back rather
        # than lose every scenario in the run.
        owner_name: str = (owner.name if owner else None) or "__unknown__"

        sm = StateMachineDef(
            name=sd.name,
            owner_part=owner_name,
            initial_values=_extract_part_attrs(owner) if owner else {},
        )

        for st in sd.owned_states:
            entry_name: Optional[str] = None
            entry_def_name: Optional[str] = None
            do_name: Optional[str] = None
            do_def_name: Optional[str] = None
            sends: List[tuple] = []
            ea = st.entry_action
            if ea:
                entry_name = ea.name
                entry_def_name = _extract_action_definition_name(ea)
                sends = _extract_send_usages(ea)
            da = st.do_action
            if da:
                do_name = da.name
                do_def_name = _extract_action_definition_name(da)
                sends.extend(_extract_send_usages(da))
            sm.states.append(StateNode(
                name=st.name,
                entry_action=entry_name,
                entry_action_def=entry_def_name,
                do_action=do_name,
                do_action_def=do_def_name,
                sends=sends,
            ))

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

            accept_trigger = _extract_accept_trigger(tr)
            # Text fallback: a standalone parse without the standard library cannot
            # resolve the accepted action def, so _extract_accept_trigger returns None
            # for every accept transition and an event-driven mode machine collapses
            # into a monitor. Guards survive the degraded parse, accepts do not, so
            # recover the trigger name from the source text keyed on the transition name.
            if accept_trigger is None:
                import re as _re
                if tr.name:
                    _pattern = (
                        r"\btransition\s+" + _re.escape(tr.name)
                        + r"\b[^;{}]*?\baccept\s+(\w+)"
                    )
                else:
                    # The transition name is optional in SysML v2, so an unnamed
                    # transition is keyed on its source and target states instead
                    # and does not lose its trigger.
                    _pattern = (
                        r"\btransition\b\s+first\s+"
                        + _re.escape(src_name or "")
                        + r"\b[^;{}]*?\baccept\s+(\w+)"
                        + r"[^;{}]*?\bthen\s+"
                        + _re.escape(tgt_name or "")
                        + r"\s*;"
                    )
                _m = _re.search(_pattern, state_source)
                if _m:
                    accept_trigger = _m.group(1)

            td = TransitionDef(
                name=tr.name,
                source=src_name,
                target=tgt_name,
                guards=guards,
                is_initial=is_initial,
                accept_trigger=accept_trigger,
            )
            sm.transitions.append(td)

            if is_initial and tgt_name:
                sm.initial_state = tgt_name

        # Syside exposes `transition initial then X;` as an owned transition,
        # but the bounded A/G spelling `entry; then X;` is an entry edge rather
        # than a TransitionUsage. Preserve the same initial-state semantics.
        if sm.initial_state is None:
            entry_initial = re.search(
                r"\bentry\s*;\s*then\s+([A-Za-z_]\w*)\s*;",
                state_source,
            )
            if entry_initial is not None:
                sm.initial_state = entry_initial.group(1)

        _resolve_guard_thresholds(sm)

        result.append(sm)

    return result


def _resolve_guard_thresholds(sm: StateMachineDef) -> None:
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
