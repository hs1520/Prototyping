"""LLM-decided A/G specs, rendered deterministically (R2 generation mode C).

Measurement drove this. Across nine seeds the LLM-authored-SysML mode got the
*engineering* right — guarantee allocation agreed with frozen gold 6/6 times and
assumption discharge 4/6 — while losing round after round to the *notation*:
unparseable output, invented keywords, guards on undeclared concepts, missing
realization links. No seed ever reached PASS.

So the notation is taken away from the model. It emits the decisions only — which
pattern the requirement instantiates, what starts the timing, how the deadline is
apportioned, which producer discharges each assumption, how responses are ordered —
and `ag_emitter` renders the SysML from them. Convention conformance is then true
by construction rather than by luck, and what the model is judged on is exactly
what it is good at.

This does not make the output *correct*: a decision can be wrong, and the evaluator
still scores it against frozen human gold. It makes the output *well formed*.

The component set, ownership and interfaces come from the frozen architecture
boundary, never from the model — the same split the LLM-authored mode uses.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .ag_chains import (
    AGAssumptionSpec,
    AGRealizationPathSpec,
    AGChainSpec,
    AGComponentSpec,
    AGInvariantSpec,
    AGPrioritySpec,
)

#: Invariant source kinds. Whether an invariant is stated by the stakeholder or
#: derived by the designer changes the evaluator's denominators, so it is a
#: decision the author must make, not something inferred here.
INVARIANT_SOURCE_KINDS = ("STAKEHOLDER", "STUDENT_DERIVED_DESIGN_CONSTRAINT")

#: Patterns an author may choose between; the choice itself is the model's.
KNOWN_PATTERNS = (
    "TRIGGERED_TIMED_FAILSAFE_RESPONSE",
    "STARTUP_INHIBIT",
    "LOCKED_UNTIL_AUTHORISED_RELEASE",
)


class DecisionError(ValueError):
    """The decisions are missing, malformed, or internally inconsistent."""


def extract_decisions(raw: str) -> Dict[str, Any]:
    """Pull the decision object out of a model response.

    Fails closed: a response that does not contain one well-formed JSON object is
    an error, never a partially-guessed decision set.
    """
    text = str(raw or "")
    for fence in ("```json", "```"):
        text = text.replace(fence, "")
    start = text.find("{")
    if start == -1:
        raise DecisionError("no JSON object in the response")
    depth = 0
    end = -1
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end == -1:
        raise DecisionError("unterminated JSON object in the response")
    try:
        decisions = json.loads(text[start:end])
    except json.JSONDecodeError as exc:
        raise DecisionError(f"decisions are not valid JSON: {exc}") from exc
    if not isinstance(decisions, Mapping):
        raise DecisionError("decisions must be a JSON object")
    return dict(decisions)


def _identifier(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not re.fullmatch(r"[A-Za-z_]\w*", text):
        raise DecisionError(f"{field} must be a bare identifier, got {value!r}")
    return text


def _capitalise(concept: str) -> str:
    return concept[:1].upper() + concept[1:] if concept else concept


def _bounded_expression(value: Any, field: str) -> str:
    """A bare concept, or a bounded Boolean expression over concepts.

    An invariant pattern's system guarantee is an expression rather than a single
    concept — REQ_SAFE_008 observes ``not powerOnInitialisation or payloadLocked``
    — so the observation accepts the same bounded subset the constraints use, and
    nothing richer.
    """
    text = str(value or "").strip()
    if not text:
        raise DecisionError(f"{field} must be set")
    tokens = text.replace("(", " ").replace(")", " ").split()
    if not tokens:
        raise DecisionError(f"{field} must be set")
    for token in tokens:
        if token in ("not", "and", "or"):
            continue
        if not re.fullmatch(r"[A-Za-z_]\w*", token):
            raise DecisionError(
                f"{field} must use only concepts and not/and/or, got {value!r}"
            )
    return text


def _expression_concepts(expression: str) -> Tuple[str, ...]:
    """The concepts a bounded Boolean expression names, in order, deduplicated.

    The emitter declares one `attribute <concept> : Boolean;` per observation
    concept, so an observation that is an *expression* must be handed over as its
    concepts, never as the expression itself: `attribute a and b : Boolean;` does
    not parse. The hand-encoded chains carry the split explicitly; decisions did
    not, and a measured seed died on the raw syntax gate for it.
    """
    return tuple(dict.fromkeys(
        token
        for token in expression.replace("(", " ").replace(")", " ").split()
        if token not in ("not", "and", "or")
    ))


def _ast_concepts(node: Any) -> List[str]:
    """Every concept name a bounded Boolean AST references."""
    if not isinstance(node, Mapping):
        return []
    if node.get("node") == "Identifier":
        return [str(node.get("name"))]
    if node.get("node") == "Not":
        return _ast_concepts(node.get("expr"))
    return [
        name for operand in (node.get("operands") or ())
        for name in _ast_concepts(operand)
    ]


def _conjunction_ast(terms: Any, field: str) -> Dict[str, Any]:
    """Build the bounded Boolean AST from a list of possibly-negated concepts.

    The profile's invariants are conjunctions of literals — every invariant across
    the three encoded chains has this shape — so the decision format is a list of
    ``{"concept": ..., "negated": ...}`` rather than a nested AST the model would
    have to assemble correctly.
    """
    if not isinstance(terms, Sequence) or isinstance(terms, str) or not terms:
        raise DecisionError(f"{field} must be a non-empty list of terms")
    nodes = []
    for term in terms:
        if not isinstance(term, Mapping):
            raise DecisionError(f"{field} terms must be objects")
        node: Dict[str, Any] = {
            "node": "Identifier",
            "name": _identifier(term.get("concept"), f"{field}.concept"),
        }
        if term.get("negated"):
            node = {"node": "Not", "expr": node}
        nodes.append(node)
    return nodes[0] if len(nodes) == 1 else {"node": "And", "operands": nodes}


def _locked_release_lifecycle(
    decisions: Mapping[str, Any],
    boundary_component: Mapping[str, Any],
    produces: Sequence[str],
    assumptions: Any,
) -> tuple:
    """The lock lifecycle, synthesised from the declared invariants.

    A locked-until-authorised-release mechanism needs more than the single
    trigger-response transition the other patterns use: it must start locked, admit
    exactly one authorised way out, and return on a *distinct* event. Those three
    facts are already in the decisions — the invariants name the locked concept and
    the authorisation concept, and the boundary names the events the component
    consumes — so the lifecycle is derived rather than asked for, and the model is
    never required to write a state machine.
    """
    if decisions.get("safety_pattern") != "LOCKED_UNTIL_AUTHORISED_RELEASE":
        return ()
    locked = authorisation = None
    for item in decisions.get("invariants") or ():
        if not isinstance(item, Mapping):
            continue
        antecedent = [
            str(term.get("concept")) for term in item.get("antecedent") or ()
            if isinstance(term, Mapping)
        ]
        consequent = [
            str(term.get("concept")) for term in item.get("consequent") or ()
            if isinstance(term, Mapping)
        ]
        negated = any(
            term.get("negated") for term in item.get("antecedent") or ()
            if isinstance(term, Mapping)
        )
        if len(consequent) != 1:
            continue
        if negated:
            locked = consequent[0]          # not <power> => <locked>
        elif len(antecedent) == 1 and "lock" in consequent[0].lower():
            locked = locked or consequent[0]
        elif len(antecedent) == 1:
            authorisation = consequent[0]   # <unlocked> => <authorisation>
    if not locked or not authorisation:
        return ()
    # only the component that actually guarantees the lock has a lock lifecycle;
    # applying it to every component gave the authorisation gateway a nonsensical
    # machine built from whatever inputs it happened to consume
    if locked not in [str(item) for item in produces]:
        return ()

    consumed = [
        str(concept)
        for concept in boundary_component.get("interfaces", {}).get("consumes", ())
    ]
    # the events that are not the authorisation itself power the component up and
    # down; their order in the interface is the boundary's, not ours to invent
    events = [concept for concept in consumed if concept != authorisation]
    if len(events) < 2:
        return ()
    power_on, power_lost = events[0], events[1]
    locked_state = f"{locked}Unpowered"
    powered_state = f"{locked}Powered"
    unlocked_state = f"{locked}Released"
    return (
        AGRealizationPathSpec(
            source=locked_state, trigger=f"{_capitalise(power_on)}Signal",
            target=powered_state, action=f"maintain{_capitalise(locked)}",
        ),
        AGRealizationPathSpec(
            source=powered_state,
            trigger=f"{_capitalise(authorisation)}Signal",
            target=unlocked_state,
            action=f"enforce{_capitalise(authorisation)}Only",
        ),
        AGRealizationPathSpec(
            source=unlocked_state, trigger=f"{_capitalise(power_lost)}Signal",
            target=locked_state, action=f"set{_capitalise(locked)}",
        ),
    )


def component_aliases(boundary: Mapping[str, Any]) -> Dict[str, str]:
    """Map every name a boundary component answers to onto its component_id.

    The boundary shows a component as ``Contract (part usage : Def)``, and a model
    asked to name one legitimately reaches for the part instead. Referring to the
    same component by its own part or definition is not an attempt to extend the
    architecture, so it is resolved rather than refused; a name belonging to no
    component still fails closed.
    """
    aliases: Dict[str, str] = {}
    for item in boundary.get("components", ()):
        component_id = str(item["component_id"])
        for alias in (
            component_id,
            str(item.get("owner_usage") or ""),
            str(item.get("owner_def") or ""),
        ):
            if alias:
                aliases[alias] = component_id
                aliases[alias.lower()] = component_id
    return aliases


def _resolve(name: Any, aliases: Mapping[str, str], field: str) -> str:
    text = str(name or "").strip()
    resolved = aliases.get(text) or aliases.get(text.lower())
    if not resolved:
        raise DecisionError(
            f"unknown component {text!r} in {field}; the architecture is frozen "
            "and may not be extended"
        )
    return resolved


def validate_decisions(
    decisions: Mapping[str, Any], boundary: Mapping[str, Any]
) -> None:
    """Reject decisions the emitter could not render into a coherent model.

    Checked against the *boundary*, never against gold: an author may allocate
    wrongly and still be well formed. Only incoherence is refused.
    """
    pattern = str(decisions.get("safety_pattern") or "")
    if pattern not in KNOWN_PATTERNS:
        raise DecisionError(
            f"safety_pattern must be one of {list(KNOWN_PATTERNS)}, got {pattern!r}"
        )
    known_components = {
        str(item["component_id"]) for item in boundary.get("components", ())
    }
    aliases = component_aliases(boundary)
    produced: Dict[str, str] = {}
    for item in boundary.get("components", ()):
        for concept in item.get("interfaces", {}).get("produces", ()):
            produced[str(concept)] = str(item["component_id"])

    decided = decisions.get("components")
    if not isinstance(decided, Sequence) or not decided:
        raise DecisionError("components must be a non-empty list")
    seen = set()
    for entry in decided:
        if not isinstance(entry, Mapping):
            raise DecisionError("each component decision must be an object")
        name = _resolve(entry.get("component_id"), aliases, "component_id")
        if name in seen:
            raise DecisionError(f"component {name!r} decided twice")
        seen.add(name)
        for assumption in entry.get("assumptions", ()):
            if not isinstance(assumption, Mapping):
                raise DecisionError("each assumption must be an object")
            concept = _identifier(assumption.get("concept"), "assumption.concept")
            discharged_by = assumption.get("discharged_by")
            if discharged_by in (None, "", "environment"):
                continue
            producer = _resolve(
                discharged_by, aliases, f"{name}.{concept}.discharged_by"
            )
            if producer == name:
                raise DecisionError(f"{name}.{concept} discharges itself")
            if produced.get(concept) is None:
                raise DecisionError(
                    f"{name}.{concept} is marked discharged but no component "
                    "produces it"
                )
    if seen != known_components:
        raise DecisionError(
            f"every boundary component must be decided; missing "
            f"{sorted(known_components - seen)}"
        )

    if pattern != "TRIGGERED_TIMED_FAILSAFE_RESPONSE":
        # an invariant pattern states its obligation as invariants and must carry
        # no timing budget — the two are mutually exclusive in the profile
        invariants = decisions.get("invariants")
        if not isinstance(invariants, Sequence) or not invariants:
            raise DecisionError(
                f"{pattern} must state at least one invariant; an invariant "
                "absent from the model does not exist"
            )
        if decisions.get("deadline_seconds") not in (None, ""):
            raise DecisionError(
                f"{pattern} is an invariant pattern and must not carry a deadline"
            )
        for index, item in enumerate(invariants):
            if not isinstance(item, Mapping):
                raise DecisionError(f"invariants[{index}] must be an object")
            _identifier(item.get("invariant_id"), f"invariants[{index}].invariant_id")
            _conjunction_ast(item.get("antecedent"), f"invariants[{index}].antecedent")
            _conjunction_ast(item.get("consequent"), f"invariants[{index}].consequent")
            kind = str(item.get("source_kind") or "")
            if kind not in INVARIANT_SOURCE_KINDS:
                raise DecisionError(
                    f"invariants[{index}].source_kind must be one of "
                    f"{list(INVARIANT_SOURCE_KINDS)}, got {kind!r}"
                )

    if pattern == "TRIGGERED_TIMED_FAILSAFE_RESPONSE":
        if decisions.get("deadline_seconds") in (None, ""):
            raise DecisionError("a timed pattern needs deadline_seconds")
        priority = decisions.get("priority")
        if not isinstance(priority, Mapping):
            raise DecisionError("a timed pattern needs a priority decision")
        members = [str(item) for item in priority.get("members", ())]
        selected = str(priority.get("selected_response") or "")
        if len(members) < 2:
            raise DecisionError("priority.members needs at least two responses")
        if selected not in members:
            raise DecisionError(
                f"priority.selected_response {selected!r} is not in members"
            )


def build_spec_from_decisions(
    decisions: Mapping[str, Any], boundary: Mapping[str, Any]
) -> AGChainSpec:
    """Assemble the emitter's spec from frozen boundary facts + model decisions.

    Everything structural (component identity, ownership, interfaces) comes from
    the boundary. Everything judged (pattern, timing, discharge wiring, ordering)
    comes from the decisions. Element naming is mechanical, so the model never has
    to guess a convention.
    """
    validate_decisions(decisions, boundary)

    requirement = str(boundary["source_requirement"])
    by_id = {
        str(item["component_id"]): item for item in boundary.get("components", ())
    }
    produced_by = {
        str(concept): str(item["component_id"])
        for item in boundary.get("components", ())
        for concept in item.get("interfaces", {}).get("produces", ())
    }
    observation = _bounded_expression(decisions.get("observation"), "observation")

    # The arbiter is the component feeding the one that produces the system
    # observation. Under a timed pattern the profile realizes it with the fixed
    # arbitration behaviour, so it is identified structurally here rather than
    # left to the model to name.
    observer = produced_by.get(observation)
    observer_consumes = set()
    if observer and observer in by_id:
        observer_consumes = {
            str(concept)
            for concept in by_id[observer].get("interfaces", {}).get("consumes", ())
        }
    arbiter = next(
        (
            str(item["component_id"]) for item in boundary.get("components", ())
            if str(item["component_id"]) != observer
            and observer_consumes & {
                str(concept)
                for concept in item.get("interfaces", {}).get("produces", ())
            }
            and len(item.get("interfaces", {}).get("produces", ())) > 1
        ),
        None,
    ) if isinstance(decisions.get("priority"), Mapping) else None

    aliases = component_aliases(boundary)
    components: List[AGComponentSpec] = []
    for entry in decisions["components"]:
        name = _resolve(entry["component_id"], aliases, "component_id")
        boundary_component = by_id[name]
        produces = [
            str(concept)
            for concept in boundary_component.get("interfaces", {}).get(
                "produces", ()
            )
        ]
        if not produces:
            raise DecisionError(f"{name} produces nothing in the boundary")
        # The emitter derives the discharge edge from the matching upstream
        # guarantee, so the decision only has to say whether the assumption is an
        # environment input or is discharged internally.
        # Concepts the author declared as typed lifecycle events are consumed but
        # not *assumed*: a mechanism that assumes its power-on event is not locked
        # by default, it is locked once that event happens to have occurred. The
        # boundary merges both into one `consumes` list, so which is which is the
        # author's judgement, and without it a default-safe component cannot be
        # assembled at all.
        events = {
            str(item) for item in entry.get("lifecycle_events", ())
            if isinstance(item, str)
        }
        assumptions = tuple(
            AGAssumptionSpec(
                concept=_identifier(item.get("concept"), "assumption.concept"),
                environment=(
                    item.get("discharged_by") in (None, "", "environment")
                    # a concept nothing in the boundary produces is environmental
                    # however the decision labelled it
                    or produced_by.get(str(item.get("concept"))) is None
                ),
            )
            for item in entry.get("assumptions", ())
            if str(item.get("concept")) not in events
        )
        primary = produces[0]
        stem = name[:-len("Contract")] if name.endswith("Contract") else name
        budget = entry.get("latency_budget_seconds")
        segment = entry.get("timing_segment_required")
        trigger_concept = next(
            (
                item.concept for item in assumptions
                if not item.environment and produced_by.get(item.concept)
            ),
            None,
        )
        lifecycle = _locked_release_lifecycle(
            decisions, boundary_component, produces, entry.get("assumptions", ())
        )
        origin = str(decisions.get("timing_origin") or "")
        is_arbiter = name == arbiter
        if is_arbiter and origin:
            # the arbitration is triggered by the timing origin even though that
            # concept is an environment input, not an upstream guarantee
            trigger_concept = origin
        behavior = "SafetyResponseArbitration" if is_arbiter else f"{stem}Behavior"
        # the responding action must establish every guarantee the state settles,
        # which for the arbiter is both the selection and the command it issues
        action_concepts = [primary, *produces[1:]] if is_arbiter else [primary]
        response_action = "set" + "And".join(
            _capitalise(concept) for concept in action_concepts
        )
        components.append(AGComponentSpec(
            name=name,
            owner_def=str(boundary_component["owner_def"]),
            owner_usage=str(boundary_component["owner_usage"]),
            guarantee=primary,
            behavior=behavior,
            trigger_signal=(
                lifecycle[0].trigger if lifecycle
                else (f"{_capitalise(trigger_concept)}Signal"
                      if trigger_concept else None)
            ),
            initial_state=(
                lifecycle[0].source if lifecycle
                else ("awaitingResponse" if is_arbiter else (
                    "idle" if trigger_concept else primary
                ))
            ),
            response_state=primary,
            response_action=response_action,
            realization_paths=lifecycle,
            assumptions=assumptions,
            interface_inputs=tuple(
                str(concept)
                for concept in boundary_component.get("interfaces", {}).get(
                    "consumes", ()
                )
            ),
            additional_guarantees=tuple(produces[1:]),
            latency_budget=float(budget) if budget not in (None, "") else None,
            timing_segment_required=(
                bool(segment) if segment is not None else None
            ),
        ))

    priority: Optional[AGPrioritySpec] = None
    decided_priority = decisions.get("priority")
    if isinstance(decided_priority, Mapping):
        members = tuple(str(item) for item in decided_priority.get("members", ()))
        selected = str(decided_priority.get("selected_response"))
        priority = AGPrioritySpec(
            response_set_id=_identifier(
                decided_priority.get("response_set_id") or "RESPONSE_SET_V1",
                "priority.response_set_id",
            ),
            members=members,
            edges=tuple(
                (selected, item) for item in members if item != selected
            ),
            trigger=_identifier(
                decisions.get("timing_origin"), "timing_origin"
            ),
            selected_response=selected,
            source_kind="STUDENT_DERIVED_DESIGN_CONSTRAINT",
            source_id=f"{requirement}_PRIORITY",
        )

    deadline = decisions.get("deadline_seconds")
    stem = requirement.replace("REQ_", "").title().replace("_", "")
    system_contract = f"System{stem}Contract"
    invariants = tuple(
        AGInvariantSpec(
            invariant_id=_identifier(
                item.get("invariant_id"), "invariants.invariant_id"
            ),
            scope=system_contract,
            trigger_or_antecedent_ast=_conjunction_ast(
                item.get("antecedent"), "invariants.antecedent"
            ),
            required_consequent_ast=_conjunction_ast(
                item.get("consequent"), "invariants.consequent"
            ),
            source_kind=str(item["source_kind"]),
            source_id=str(item.get("source_id") or item["invariant_id"]),
        )
        for item in (decisions.get("invariants") or ())
        if isinstance(item, Mapping)
    )
    return AGChainSpec(
        source_requirement=requirement,
        package=f"{requirement}_AG",
        system_contract=system_contract,
        system_assumptions=tuple(
            _identifier(item, "system_assumptions")
            for item in decisions.get("system_assumptions", ())
        ),
        observation=observation,
        deadline=float(deadline) if deadline not in (None, "") else None,
        components=tuple(components),
        verification=f"{stem}Verification",
        system_observation_concepts=_expression_concepts(observation),
        pattern=str(decisions["safety_pattern"]),
        timing_origin=(
            _identifier(decisions.get("timing_origin"), "timing_origin")
            if decisions.get("timing_origin") else None
        ),
        priority=priority,
        invariants=invariants,
        selected_model_elements=tuple(sorted({
            *(item.concept for component in components
              for item in component.assumptions),
            *(component.guarantee for component in components),
            *(concept for component in components
              for concept in component.additional_guarantees),
            # an invariant binds the concepts it constrains, so they must be
            # selected too or the binding is incomplete
            *(name for item in invariants
              for name in _ast_concepts(item.trigger_or_antecedent_ast)),
            *(name for item in invariants
              for name in _ast_concepts(item.required_consequent_ast)),
            *_expression_concepts(observation),
        })),
    )
