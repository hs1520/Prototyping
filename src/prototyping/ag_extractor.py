"""Extract a bounded Assume-Guarantee graph from committed SysML v2 text.

This is the extractor half of Increment 2 (§6.2, §15 of
``docs/OPTION2_IMPLEMENTATION_DESIGN.md``). The committed SysML model is the sole
semantic authority: A/G facts are read *out of* the model using the validated
bounded convention and never supplied from JSON. The output :class:`AGGraph`
carries the source revision/digest and per-element spans so the derived
``ag_contract_graph.json`` view (§14) is fully attributable and regenerable.

Bounded convention (validated against the Syside gate — see
``tests/test_option2_ag_sysml_spike.py`` and ``test_option2_ag_checker.py``):

    requirement def <ContractName> {
        attribute <var> : <Type> [= <default>];
        attribute latencyBudget : Real = 0.10;   // component timing budget
        attribute maxLatency    : Real = 0.5;     // system deadline
        assume  constraint [env_]<name> { <boolean-id | var <cmp> rhs> }
        require constraint <name>        { <boolean-id | var <cmp> rhs> }
    }
    dependency decomposition from <System> to <Component>;

An ``assume constraint`` whose name begins with ``env`` is an explicit
environment assumption. The decomposition edge source is the system contract; the
targets are components.
"""
from __future__ import annotations

import re
from typing import Dict, List, Mapping, Optional, Tuple

from .ag_contracts import (
    AGDiagnostic,
    AGEdge,
    AGGraph,
    Assumption,
    BehaviorRealization,
    BehaviorTransition,
    Contract,
    Guarantee,
    InvariantRealization,
    Span,
)
from .blackboard import text_digest
from ..utils.sysml_text_utils import find_block_end

_REQ_DEF_RE = re.compile(r"\brequirement\s+def\s+(\w+)\s*\{")
_ATTR_RE = re.compile(
    r"\battribute\s+(\w+)\s*:\s*(\w+)\s*(?:=\s*(-?[\d.]+)"
    r"\s*(?:\[([^\]]+)\])?)?\s*;"
)
_BOOL_ATTR_RE = re.compile(
    r"\battribute\s+(\w+)\s*:\s*Boolean\s*=\s*(true|false)\s*;",
    re.I,
)
_EVENT_DEF_RE = re.compile(
    r"\b(?:action|attribute)\s+def\s+(\w+)\s*(?:\{\s*\}|;)"
)
_ASSUME_RE = re.compile(r"\bassume\s+constraint\s+(\w+)?\s*\{([^{}]*)\}")
_REQUIRE_RE = re.compile(r"\brequire\s+constraint\s+(\w+)?\s*\{([^{}]*)\}")
_DEP_RE = re.compile(
    r"\bdependency\s+(\w+)\s+from\s+(\w+)\s+to\s+(\w+)\s*;"
)
_SATISFY_RE = re.compile(
    r"\bsatisfy\s+requirement\s+\w+\s*:\s*(\w+)\s+by\s+(\w+)\s*;"
)
_STATE_DEF_RE = re.compile(r"\bstate\s+def\s+(\w+)\s*\{")
_ASSERT_CONSTRAINT_RE = re.compile(
    r"\bassert\s+constraint\s+(\w+)\s*\{([^{}]*)\}"
)
_TRANSITION_RE = re.compile(
    r"\btransition\s+\w+\s+first\s+(\w+)\s+accept\s+(\w+)"
    r"(?:\s+if\s+(.+?))?\s+then\s+(\w+)\s*;",
    re.DOTALL,
)
_VERIFICATION_DEF_RE = re.compile(r"\bverification\s+def\s+(\w+)\s*\{")
_VERIFY_REQ_RE = re.compile(
    r"\bverify\s+requirement\s+\w+\s*:\s*(\w+)\s*;"
)
_SOURCE_REQ_RE = re.compile(r"bounded\s+A/G\s+system\s+contract\s+for\s+(REQ[_-]\w+)", re.I)
_SAFETY_PATTERN_RE = re.compile(r"safety_pattern\s*=\s*(\w+)", re.I)
_TIMING_ORIGIN_RE = re.compile(r"timing_origin\s*=\s*(\w+)", re.I)
_PRIORITY_AUXILIARY_MARKER = "bounded A/G evaluator semantic auxiliary"
_CMP_RE = re.compile(r"^(\w+)\s*(<=|>=|==|<|>)\s*([A-Za-z_][\w]*|-?[\d.]+)$")
_IDENT_RE = re.compile(r"^(\w+)$")
_BOOL_TOKEN_RE = re.compile(r"\s*(\(|\)|not\b|and\b|or\b|[A-Za-z_]\w*)", re.I)

# Attribute names that carry a timing budget/deadline.
_COMPONENT_BUDGET_KEYS = ("latencybudget",)
_SYSTEM_BUDGET_KEYS = ("maxlatency", "deadline", "systemdeadline")
#: Deliberately NOT budget keys: the margin is not the deadline, and reading it as
#: one would let a reserved margin masquerade as the whole budget.
_SEGMENT_GROUP_KEY = "timingsegmentgroup"
_MARGIN_KEY = "timingmargin"


def _boolean_ast(expr: str) -> Optional[Dict[str, object]]:
    """Parse the fixed Identifier/Not/And/Or subset used by the bounded profile."""
    source = str(expr or "").strip()
    tokens: List[str] = []
    cursor = 0
    while cursor < len(source):
        match = _BOOL_TOKEN_RE.match(source, cursor)
        if not match:
            return None
        tokens.append(match.group(1))
        cursor = match.end()
    position = 0

    def primary():
        nonlocal position
        if position >= len(tokens):
            raise ValueError
        token = tokens[position]
        if token == "(":
            position += 1
            node = parse_or()
            if position >= len(tokens) or tokens[position] != ")":
                raise ValueError
            position += 1
            return node
        if token.lower() in {"and", "or", "not"} or token == ")":
            raise ValueError
        position += 1
        return {"node": "Identifier", "name": token}

    def parse_not():
        nonlocal position
        if position < len(tokens) and tokens[position].lower() == "not":
            position += 1
            return {"node": "Not", "expr": parse_not()}
        return primary()

    def parse_and():
        nonlocal position
        operands = [parse_not()]
        while position < len(tokens) and tokens[position].lower() == "and":
            position += 1
            operands.append(parse_not())
        return operands[0] if len(operands) == 1 else {
            "node": "And", "operands": operands,
        }

    def parse_or():
        nonlocal position
        operands = [parse_and()]
        while position < len(tokens) and tokens[position].lower() == "or":
            position += 1
            operands.append(parse_and())
        return operands[0] if len(operands) == 1 else {
            "node": "Or", "operands": operands,
        }

    try:
        result = parse_or()
    except (KeyError, ValueError):
        return None
    return result if position == len(tokens) else None


def _parse_expr(
    expr: str, attrs: Mapping[str, Optional[float]]
) -> Tuple[str, str, Dict[str, object]]:
    """Return (concept, kind, extra) for one constraint body.

    kind ∈ {"boolean", "numeric", "unsupported"}. Numeric right-hand sides that
    name a declared attribute are resolved to that attribute's default value.
    """
    body = " ".join((expr or "").split())
    m = _CMP_RE.match(body)
    if m:
        var, cmp_op, rhs = m.group(1), m.group(2), m.group(3)
        try:
            value: Optional[float] = float(rhs)
        except ValueError:
            value = attrs.get(rhs.lower())
        return var, "numeric", {"variable": var, "comparator": cmp_op, "value": value}
    m = _IDENT_RE.match(body)
    if m:
        ast = {"node": "Identifier", "name": m.group(1)}
        return m.group(1), "boolean", {"ast": ast}
    ast = _boolean_ast(body)
    if ast is not None:
        return body, "boolean", {"ast": ast}
    return body, "unsupported", {}


def _parse_contract(name: str, block: str, span: Span) -> Contract:
    attrs: Dict[str, Optional[float]] = {}
    attr_units: Dict[str, Optional[str]] = {}
    for m in _ATTR_RE.finditer(block):
        attr_name = m.group(1)
        default = m.group(3)
        attrs[attr_name.lower()] = float(default) if default is not None else None
        attr_units[attr_name.lower()] = m.group(4)
    bool_attrs = {
        m.group(1).lower(): m.group(2).lower() == "true"
        for m in _BOOL_ATTR_RE.finditer(block)
    }

    assumptions: List[Assumption] = []
    for m in _ASSUME_RE.finditer(block):
        cname, expr = m.group(1), m.group(2)
        concept, kind, extra = _parse_expr(expr, attrs)
        is_env = bool(cname and cname.lower().startswith("env"))
        assumptions.append(Assumption(
            concept=concept, expr=" ".join(expr.split()), kind=kind,
            is_environment=is_env, constraint_name=cname,
            variable=extra.get("variable"), comparator=extra.get("comparator"),
            value=extra.get("value"),
            ast=extra.get("ast"),
        ))

    guarantees: List[Guarantee] = []
    for m in _REQUIRE_RE.finditer(block):
        cname, expr = m.group(1), m.group(2)
        concept, kind, extra = _parse_expr(expr, attrs)
        guarantees.append(Guarantee(
            concept=concept, expr=" ".join(expr.split()), kind=kind,
            constraint_name=cname, variable=extra.get("variable"),
            comparator=extra.get("comparator"), value=extra.get("value"),
            ast=extra.get("ast"),
        ))

    timing_budget: Optional[float] = None
    timing_unit: Optional[str] = None
    timing_value_literal: Optional[str] = None
    for key in _COMPONENT_BUDGET_KEYS + _SYSTEM_BUDGET_KEYS:
        if attrs.get(key) is not None:
            timing_budget = attrs[key]
            timing_unit = attr_units.get(key)
            match = next(
                (
                    item for item in _ATTR_RE.finditer(block)
                    if item.group(1).lower() == key
                ),
                None,
            )
            timing_value_literal = match.group(3) if match else None
            break

    source = _SOURCE_REQ_RE.search(block)
    pattern = _SAFETY_PATTERN_RE.search(block)
    timing_origin = _TIMING_ORIGIN_RE.search(block)
    # Composition structure and reserved margin. Both are optional: absent, the
    # composition is the plain sum it always was.
    group = attrs.get(_SEGMENT_GROUP_KEY)
    margin = attrs.get(_MARGIN_KEY)

    return Contract(
        name=name,
        role="component",  # provisional; fixed once edges are known
        assumptions=tuple(assumptions),
        guarantees=tuple(guarantees),
        timing_budget=timing_budget,
        timing_unit=timing_unit,
        timing_value_literal=timing_value_literal,
        timing_segment_required=bool_attrs.get("timingsegmentrequired"),
        timing_segment_group=int(group) if group is not None else None,
        timing_margin=margin,
        timing_margin_unit=attr_units.get(_MARGIN_KEY),
        timing_origin=timing_origin.group(1) if timing_origin else None,
        observation=None,  # set for the system contract only
        element_id=name,
        span=span,
        source_requirement=source.group(1).upper().replace("-", "_") if source else None,
        declared_pattern=pattern.group(1).upper() if pattern else None,
    )


def _parse_behavior(name: str, block: str, span: Span) -> BehaviorRealization:
    initial = re.search(r"\bentry\s*;\s*then\s+(\w+)\s*;", block)
    transitions = tuple(
        BehaviorTransition(
            m.group(1), m.group(2), m.group(4),
            " ".join(m.group(3).split()) if m.group(3) else None,
        )
        for m in _TRANSITION_RE.finditer(block)
    )
    entry_actions: Dict[str, str] = {}
    for state in re.finditer(r"\bstate\s+(\w+)\s*\{", block):
        brace = block.find("{", state.start())
        end = find_block_end(block, brace)
        if end == -1:
            continue
        action = re.search(r"\bentry\s+action\s+(\w+)", block[brace + 1:end])
        if action:
            entry_actions[state.group(1)] = action.group(1)
    return BehaviorRealization(
        name=name,
        initial_state=initial.group(1) if initial else None,
        transitions=transitions,
        entry_actions=entry_actions,
        element_id=name,
        span=span,
    )


def _extract_priority(text: str) -> Dict[str, object]:
    """Extract the bounded priority facts from authoritative SysML constructs."""
    contract_match = re.search(
        r"\brequirement\s+def\s+SafetyResponsePriorityContract\s*\{", text
    )
    state_match = re.search(
        r"\bstate\s+def\s+SafetyResponseArbitration\s*\{", text
    )
    if not contract_match or not state_match:
        return {}
    contract_brace = text.index("{", contract_match.start())
    contract_end = find_block_end(text, contract_brace)
    state_brace = text.index("{", state_match.start())
    state_end = find_block_end(text, state_brace)
    if contract_end == -1 or state_end == -1:
        return {}
    contract = text[contract_brace + 1:contract_end]
    state = text[state_brace + 1:state_end]

    response_set = re.search(r"response_set_id\s*=\s*(\w+)", contract)
    trigger_match = re.search(
        r"assume\s+constraint\s+priorityTrigger\s*\{\s*(\w+)\s*\}",
        contract,
    )
    selected_match = re.search(
        r"require\s+constraint\s+selectHighestPriority\s*\{"
        r"\s*selectedResponse\s*==\s*(\w+)::(\w+)\s*\}",
        contract,
    )
    if not response_set or not trigger_match or not selected_match:
        return {}
    response_set_id = response_set.group(1)
    enum_match = re.search(
        rf"\benum\s+def\s+{re.escape(response_set_id)}\s*\{{", text
    )
    if not enum_match:
        return {}
    enum_brace = text.index("{", enum_match.start())
    enum_end = find_block_end(text, enum_brace)
    members = re.findall(r"\benum\s+(\w+)\s*;", text[enum_brace + 1:enum_end])
    selected = selected_match.group(2)
    lower_members = re.findall(
        r"require\s+constraint\s+precedence_\w+\s*\{"
        r"\s*not\s+\w+\s+or\s+selectedResponse\s*!=\s*\w+::(\w+)\s*\}",
        contract,
    )
    edges = [{"higher": selected, "lower": lower} for lower in lower_members]
    member_provenance = [
        {
            "response": match.group(1),
            "source_kind": match.group(2),
            "source_id": match.group(3),
        }
        for match in re.finditer(
            r"response_member\s*=\s*(\w+)\s*;\s*"
            r"source_kind\s*=\s*(\w+)\s*;\s*"
            r"source_id\s*=\s*(\w+)",
            contract,
        )
    ]
    # Wiring can be judged from the selection/precedence contract even when the
    # enum vocabulary itself is malformed or incomplete. Coupling these made a
    # blocked response-set defect manufacture repairable transition/action faults.
    known_responses = list(dict.fromkeys([
        *members,
        selected,
        *lower_members,
    ]))
    transitions = tuple(
        BehaviorTransition(
            match.group(1),
            match.group(2),
            match.group(4),
            " ".join(match.group(3).split()) if match.group(3) else None,
        )
        for match in _TRANSITION_RE.finditer(state)
    )

    def name_key(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", value.lower())

    def response_member(state_name: str) -> Optional[str]:
        """Map an authored state name to its declared enum member structurally."""
        state_key = name_key(state_name)
        matches = [
            member
            for member in known_responses
            if (
                name_key(member) == state_key
                or name_key(member) in state_key
                or state_key in name_key(member)
            )
        ]
        return matches[0] if len(matches) == 1 else None

    guards = [
        {
            "response": response_member(transition.target) or transition.target,
            "guard_ast": {
                "node": "Not",
                "expr": {
                    "node": "Identifier",
                    "name": trigger_match.group(1),
                },
            },
        }
        for transition in transitions
        if transition.guard == f"not {trigger_match.group(1)}"
    ]
    initial_match = re.search(r"\bentry\s*;\s*then\s+(\w+)\s*;", state)
    reachable_states = {initial_match.group(1)} if initial_match else set()
    changed = True
    while changed:
        changed = False
        for transition in transitions:
            if (
                transition.source in reachable_states
                and transition.target not in reachable_states
            ):
                reachable_states.add(transition.target)
                changed = True
    selection_transitions = [
        transition
        for transition in transitions
        if (
            transition.guard == trigger_match.group(1)
            and response_member(transition.target) == selected
        )
    ]
    selected_transition = any(
        transition.source in reachable_states
        for transition in selection_transitions
    )
    selected_targets = {
        transition.target for transition in selection_transitions
    }

    # The response-selection guarantee is derived from the contract that realizes
    # this arbitration behavior. It is not a reviewed state/signal/action name.
    realizing_contracts = re.findall(
        r"\bdependency\s+\w+\s+from\s+(\w+)\s+to\s+"
        r"SafetyResponseArbitration\s*;",
        text,
    )
    selection_guarantees: set[str] = set()
    for contract_name in realizing_contracts:
        realized_match = re.search(
            rf"\brequirement\s+def\s+{re.escape(contract_name)}\s*\{{",
            text,
        )
        if not realized_match:
            continue
        realized_brace = text.index("{", realized_match.start())
        realized_end = find_block_end(text, realized_brace)
        if realized_end == -1:
            continue
        realized_body = text[realized_brace + 1:realized_end]
        for _constraint_name, expression in _REQUIRE_RE.findall(realized_body):
            concept = expression.strip()
            if re.fullmatch(r"[A-Za-z_]\w*", concept) and "selected" in name_key(
                concept
            ):
                selection_guarantees.add(concept)

    target_actions: List[str] = []
    for state_match_item in re.finditer(r"\bstate\s+(\w+)\s*\{", state):
        if state_match_item.group(1) not in selected_targets:
            continue
        brace = state.find("{", state_match_item.start())
        end = find_block_end(state, brace)
        if end == -1:
            continue
        action = re.search(r"\bentry\s+action\s+(\w+)", state[brace + 1:end])
        if action:
            target_actions.append(action.group(1))
    selection_action_connected = bool(
        selection_guarantees
        and any(
            name_key(guarantee) in name_key(action)
            for guarantee in selection_guarantees
            for action in target_actions
        )
    )
    selected_elements = set(members)
    selected_elements.update({
        trigger_match.group(1),
        "selectedResponse",
    })
    return {
        "response_set_id": response_set_id,
        "members": members,
        "edges": edges,
        "trigger": trigger_match.group(1),
        "member_provenance": member_provenance,
        "arbitration_topology": {
            "response_set_id": response_set_id,
            "members": members,
            "edges": edges,
            "trigger": trigger_match.group(1),
            "member_provenance": member_provenance,
            "selection": {
                "when": trigger_match.group(1),
                "selected_response": selected,
            },
            "competing_transition_guards": guards,
            "selected_model_elements": sorted(selected_elements),
            "parachute_transition_reachable": selected_transition,
            "selection_action_connected": selection_action_connected,
            "deployment_action_connected": False,
            "observation_connected": False,
        },
    }


_INVARIANT_NAME_RE = re.compile(
    r"^inv__(.+?)__source__(.+?)__kind__(.+)$"
)


def _extract_invariants(
    text: str,
    *,
    system_contract: str | None = None,
) -> Tuple[Mapping[str, object], ...]:
    result: List[Mapping[str, object]] = []
    for requirement in _REQ_DEF_RE.finditer(text):
        if (
            system_contract is not None
            and requirement.group(1) != system_contract
        ):
            continue
        brace = text.index("{", requirement.start())
        end = find_block_end(text, brace)
        if end == -1:
            continue
        block = text[brace + 1:end]
        for constraint in _REQUIRE_RE.finditer(block):
            name = str(constraint.group(1) or "")
            metadata = _INVARIANT_NAME_RE.match(name)
            if not metadata:
                continue
            lowered = _boolean_ast(constraint.group(2))
            if not isinstance(lowered, Mapping):
                continue
            operands = (
                lowered.get("operands")
                if lowered.get("node") == "Or" else None
            )
            if (
                not isinstance(operands, list)
                or len(operands) != 2
                or not isinstance(operands[0], Mapping)
                or operands[0].get("node") != "Not"
            ):
                continue
            result.append({
                "invariant_id": metadata.group(1),
                "scope": requirement.group(1),
                "trigger_or_antecedent_ast": operands[0].get("expr"),
                "required_consequent_ast": operands[1],
                "source_kind": metadata.group(3),
                "source_id": metadata.group(2),
            })
    return tuple(result)


_PACKAGE_RE = re.compile(r"\bpackage\s+(\w+)\s*\{")
# The emitter stamps this exact marker on every A/G system contract (see
# ``ag_emitter.emit_ag_package``); it is the reliable signal that a top-level
# package carries a bounded A/G chain candidate rather than base model content.
_AG_PACKAGE_MARKER = "bounded A/G system contract for"


def _top_level_packages(text: str) -> List[Tuple[str, int, int]]:
    """Return (name, start, end_exclusive) for each top-level package block."""
    result: List[Tuple[str, int, int]] = []
    consumed_to = 0
    for m in _PACKAGE_RE.finditer(text):
        if m.start() < consumed_to:  # nested inside an already-consumed package
            continue
        brace = text.index("{", m.start())
        end = find_block_end(text, brace)
        if end == -1:
            continue
        result.append((m.group(1), m.start(), end + 1))
        consumed_to = end + 1
    return result


def extract_ag_graphs(
    sysml_text: str,
    *,
    revision: Optional[int] = None,
    model_digest: Optional[str] = None,
) -> List[AGGraph]:
    """Extract one bounded A/G graph per selected chain in the committed model.

    A model may carry several independent A/G chains (one selected decomposition
    per selected requirement — the drone system co-selects REQ_SAFE_004 and
    REQ_SAFE_005). Each chain is emitted as its own top-level package with a
    single system contract, so each is a self-contained A/G decomposition that
    must be checked independently: pooling two system contracts into one graph
    would make the decomposition root ambiguous (``system=None``).

    With zero or one A/G package this returns exactly ``[extract_ag_graph(...)]``
    — byte-identical to the single-chain path. With two or more, the base model
    (which carries the immutable source ``requirement def`` provenance) is paired
    with each A/G package in turn so every per-chain graph both resolves its
    system contract uniquely and keeps its source-requirement provenance. Every
    per-chain graph reports the committed model's revision/digest, not the slice's.
    """
    text = sysml_text or ""
    digest = model_digest if model_digest is not None else text_digest(text)
    ag_spans = [
        (start, end)
        for (_name, start, end) in _top_level_packages(text)
        if _AG_PACKAGE_MARKER in text[start:end]
    ]
    if len(ag_spans) <= 1:
        return [extract_ag_graph(text, revision=revision, model_digest=digest)]

    # Base model = everything that is not an A/G package (source requirement defs
    # live here); it is prepended to each A/G package so provenance resolves.
    others_parts: List[str] = []
    cursor = 0
    for start, end in sorted(ag_spans):
        others_parts.append(text[cursor:start])
        cursor = end
    others_parts.append(text[cursor:])
    base = "".join(others_parts).rstrip()

    graphs: List[AGGraph] = []
    for start, end in sorted(ag_spans):
        slice_text = base + "\n\n" + text[start:end] + "\n"
        graphs.append(
            extract_ag_graph(slice_text, revision=revision, model_digest=digest)
        )
    return graphs


def extract_ag_graph(
    sysml_text: str,
    *,
    revision: Optional[int] = None,
    model_digest: Optional[str] = None,
) -> AGGraph:
    """Parse committed SysML v2 text into a bounded A/G graph (§6.2).

    This resolves a single system contract. For a model that may carry more than
    one selected chain, use :func:`extract_ag_graphs`, which returns one graph
    per chain and degrades to ``[this]`` when only one chain is present.
    """
    text = sysml_text or ""
    digest = model_digest if model_digest is not None else text_digest(text)
    parse_diags: List[AGDiagnostic] = []

    raw: Dict[str, Contract] = {}
    all_requirement_ids: List[str] = []
    for m in _REQ_DEF_RE.finditer(text):
        name = m.group(1)
        all_requirement_ids.append(name.upper().replace("-", "_"))
        brace = text.index("{", m.start())
        end = find_block_end(text, brace)
        if end == -1:
            continue
        block = text[brace + 1:end]
        contract = _parse_contract(name, block, Span(brace + 1, end))
        # Only a requirement def that declares assume/require constraints is an A/G
        # contract (§6.2). Ordinary stakeholder requirement defs imported into the
        # model carry no A/G semantics and must not pollute the graph.
        if (
            (contract.assumptions or contract.guarantees)
            and _PRIORITY_AUXILIARY_MARKER not in block
        ):
            raw[name] = contract

    owners: Dict[str, List[str]] = {}
    for match in _SATISFY_RE.finditer(text):
        owners.setdefault(match.group(1), []).append(match.group(2))

    behaviors: List[BehaviorRealization] = []
    for match in _STATE_DEF_RE.finditer(text):
        brace = text.index("{", match.start())
        end = find_block_end(text, brace)
        if end != -1:
            behaviors.append(_parse_behavior(
                match.group(1), text[brace + 1:end], Span(brace + 1, end)
            ))
    invariant_realizations = tuple(
        InvariantRealization(
            name=match.group(1),
            expression=" ".join(match.group(2).split()),
            element_id=match.group(1),
            span=Span(match.start(2), match.end(2)),
        )
        for match in _ASSERT_CONSTRAINT_RE.finditer(text)
    )

    verification_targets: Dict[str, Tuple[str, ...]] = {}
    for match in _VERIFICATION_DEF_RE.finditer(text):
        brace = text.index("{", match.start())
        end = find_block_end(text, brace)
        if end != -1:
            verification_targets[match.group(1)] = tuple(
                item.group(1) for item in _VERIFY_REQ_RE.finditer(text[brace + 1:end])
            )

    edges: List[AGEdge] = []
    for m in _DEP_RE.finditer(text):
        raw_kind, src, dst = m.group(1), m.group(2), m.group(3)
        kind = raw_kind.lower()
        subject = None
        if kind.startswith("decompos"):
            edge_kind = "decomposes"
        elif kind.startswith("discharge"):
            edge_kind = "discharges"
            subject = raw_kind[len("discharge"):].split("__to__", 1)[0]
        elif kind.startswith("realize"):
            edge_kind = "realized_by"
        elif kind.startswith("observe"):
            edge_kind = "observed_by"
        else:
            edge_kind = kind
        edges.append(AGEdge(kind=edge_kind, src=src, dst=dst, subject=subject))

    sources = {e.src for e in edges if e.kind == "decomposes"}
    targets = {e.dst for e in edges if e.kind == "decomposes"}
    system_names = sources - targets

    system_name: Optional[str] = None
    if len(system_names) == 1:
        system_name = next(iter(system_names))
    elif len(raw) == 1:
        system_name = next(iter(raw))
    else:
        # Ambiguous: fall back to a name-based heuristic and record it.
        candidates = [n for n in raw if "system" in n.lower()]
        if len(candidates) == 1:
            system_name = candidates[0]
        parse_diags.append(AGDiagnostic(
            "CONTRACT_INCOMPLETE",
            ("system contract could not be uniquely identified from decomposition "
             f"edges (sources={sorted(system_names)})"),
            severity="warning",
        ))

    system: Optional[Contract] = None
    components: List[Contract] = []
    for name, contract in raw.items():
        if name == system_name:
            # The system observation is its first Boolean guarantee concept.
            obs = next(
                (g.concept for g in contract.guarantees if g.kind == "boolean"),
                None,
            )
            system = Contract(
                name=contract.name, role="system",
                assumptions=contract.assumptions, guarantees=contract.guarantees,
                timing_budget=contract.timing_budget, timing_unit=contract.timing_unit,
                timing_value_literal=contract.timing_value_literal,
                timing_segment_required=contract.timing_segment_required,
                timing_segment_group=contract.timing_segment_group,
                timing_margin=contract.timing_margin,
                timing_margin_unit=contract.timing_margin_unit,
                timing_origin=contract.timing_origin,
                observation=obs, element_id=contract.element_id, span=contract.span,
                owners=tuple(owners.get(name, ())),
                source_requirement=contract.source_requirement,
                declared_pattern=contract.declared_pattern,
            )
        else:
            components.append(Contract(
                name=contract.name, role=contract.role,
                assumptions=contract.assumptions, guarantees=contract.guarantees,
                timing_budget=contract.timing_budget, timing_unit=contract.timing_unit,
                timing_value_literal=contract.timing_value_literal,
                timing_segment_required=contract.timing_segment_required,
                timing_segment_group=contract.timing_segment_group,
                timing_margin=contract.timing_margin,
                timing_margin_unit=contract.timing_margin_unit,
                timing_origin=contract.timing_origin,
                observation=contract.observation, element_id=contract.element_id,
                span=contract.span, owners=tuple(owners.get(name, ())),
                source_requirement=contract.source_requirement,
            ))

    return AGGraph(
        system=system,
        components=tuple(components),
        edges=tuple(edges),
        revision=revision,
        model_digest=digest,
        parse_diagnostics=tuple(parse_diags),
        behaviors=tuple(behaviors),
        invariant_realizations=invariant_realizations,
        verification_targets=verification_targets,
        source_requirement_ids=tuple(dict.fromkeys(all_requirement_ids)),
        priority=_extract_priority(text),
        invariants=_extract_invariants(text, system_contract=system_name),
        selected_model_elements=tuple(sorted({
            match.group(1) for match in re.finditer(
                r"\battribute\s+(\w+)\s*:", text
            )
        })),
        declared_event_signals=tuple(dict.fromkeys(
            match.group(1) for match in _EVENT_DEF_RE.finditer(text)
        )),
    )
