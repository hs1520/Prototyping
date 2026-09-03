"""LLM-decided A/G specs, rendered deterministically (R2 generation mode C).

Across nine seeds the LLM-authored-SysML mode got the engineering right -
guarantee allocation agreed with frozen gold 6/6, assumption discharge 4/6 - but
lost on notation: unparseable output, invented keywords, guards on undeclared
concepts, missing realization links. No seed reached PASS.

So the model emits decisions only - which pattern the requirement instantiates,
what starts the timing, how the deadline is apportioned, which producer discharges
each assumption, how responses are ordered - and `ag_emitter` renders the SysML.
Convention conformance is then true by construction; correctness is not, since a
decision can be wrong and the evaluator still scores it against frozen human gold.

The component set, ownership and interfaces come from the frozen architecture
boundary, as in the LLM-authored mode.
"""
from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .ag_chains import (
    AGAssumptionSpec,
    AGRealizationPathSpec,
    AGChainSpec,
    AGComponentSpec,
    AGInvariantSpec,
    AGPrioritySpec,
)
from .ag_profile import (
    DERIVED_SOURCE_KIND,
    INVARIANT_PATTERNS,
    INVARIANT_SOURCE_KINDS,
    KNOWN_PATTERNS,
    LOCKED_UNTIL_RELEASE_PATTERN,
    TIMED_PATTERN,
    TRIGGERED_PATTERNS,
)
from .ag_emitter import _capitalise
from ..utils.sysml_text_utils import IDENTIFIER_RE, find_block_end

class DecisionError(ValueError):
    """The decisions are missing, malformed, or internally inconsistent."""


class DecisionFailureDisposition(str, Enum):
    """What the bounded generator may do after validation fails."""

    RETRYABLE_VALIDATION_ERROR = "RETRYABLE_VALIDATION_ERROR"
    NEEDS_ARCHITECTURE_INPUT = "NEEDS_ARCHITECTURE_INPUT"


class ArchitectureInputRequired(RuntimeError):
    """The decision cannot be completed without an unpublished architecture fact."""

    def __init__(self, decision_error: DecisionError):
        self.decision_error = decision_error
        super().__init__(str(decision_error))


def classify_decision_failure(
    error: DecisionError | str,
) -> DecisionFailureDisposition:
    """Classify a validator failure without weakening validation.

    Most failures are the model's own JSON or consistency choices and can be retried
    against the published boundary. A singleton timed response set cannot: neither the
    requirement nor the boundary enumerates the competing safety responses, so asking
    for a second member rewards invention (the measured ``OTHER_RESPONSE`` failure).
    New errors stay retryable until evidence shows they need a fact the inputs lack.
    """
    if str(error) in {
        "priority.members needs at least two responses",
        "priority response catalog has fewer than two provenance-backed responses",
    }:
        return DecisionFailureDisposition.NEEDS_ARCHITECTURE_INPUT
    return DecisionFailureDisposition.RETRYABLE_VALIDATION_ERROR


def extract_runtime_response_catalog(model_text: str) -> Dict[str, Any]:
    """Extract selectable safety responses from committed arbiter behavior.

    Gold-blind: entries are calls made by state entry actions in an existing
    arbitration state definition, so the response vocabulary stays tied to executable
    elements already in the model. A port, Boolean guarantee, or enum member with a
    plausible name is not a response.
    """
    entries: List[Dict[str, str]] = []
    for definition in re.finditer(
        r"\bstate\s+def\s+([A-Za-z_]\w*Arbiter\w*)\s*\{",
        model_text,
        re.IGNORECASE,
    ):
        outer_brace = model_text.index("{", definition.start())
        outer_end = find_block_end(model_text, outer_brace)
        if outer_end == -1:
            continue
        body = model_text[outer_brace + 1:outer_end]
        for state in re.finditer(r"\bstate\s+([A-Za-z_]\w*)\s*\{", body):
            state_brace = body.index("{", state.start())
            state_end = find_block_end(body, state_brace)
            if state_end == -1:
                continue
            state_body = body[state_brace + 1:state_end]
            for action in re.finditer(
                r"\bentry\s+action\s+([A-Za-z_]\w*)\s*:\s*"
                r"([A-Za-z_]\w*)\s*;",
                state_body,
            ):
                entries.append({
                    "response_id": action.group(2),
                    "source_kind": "EXISTING_MODEL_BEHAVIOR",
                    "source_id": (
                        f"{definition.group(1)}.{state.group(1)}."
                        f"{action.group(1)}"
                    ),
                })
    deduplicated = {
        item["response_id"]: item for item in entries
    }
    return {
        "artifact_role": "RUNTIME_RESPONSE_CATALOG",
        "source": "COMMITTED_SYSML_BEHAVIOR",
        "entries": list(deduplicated.values()),
    }


def extract_decisions(raw: str) -> Dict[str, Any]:
    """Pull the decision object out of a model response.

    Fails closed: a response without one well-formed JSON object is an error, not a
    partially guessed decision set.
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
    if not IDENTIFIER_RE.fullmatch(text):
        raise DecisionError(f"{field} must be a bare identifier, got {value!r}")
    # A decided name goes into SysML text, where a reserved word is a parser
    # error; cheaper to refuse it here.
    from .sysml_reserved import SYSML_RESERVED_WORDS
    if text in SYSML_RESERVED_WORDS:
        raise DecisionError(
            f"{field} must not be a SysML reserved word, got {value!r}; "
            "choose another identifier"
        )
    return text


def _bounded_expression(value: Any, field: str) -> str:
    """A bare concept, or a bounded Boolean expression over concepts.

    An invariant pattern's system guarantee is an expression, not one concept
    (REQ_SAFE_008 observes ``not powerOnInitialisation or payloadLocked``), so the
    observation accepts the same bounded subset the constraints use.
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
        if not IDENTIFIER_RE.fullmatch(token):
            raise DecisionError(
                f"{field} must use only concepts and not/and/or, got {value!r}"
            )
    return text


def _expression_concepts(expression: str) -> Tuple[str, ...]:
    """The concepts a bounded Boolean expression names, in order, deduplicated.

    The emitter declares one `attribute <concept> : Boolean;` per observation concept,
    so an expression observation is handed over as its concepts: `attribute a and b :
    Boolean;` does not parse. The hand-encoded chains split it explicitly, decisions
    did not, and a measured seed failed the raw syntax gate for it.
    """
    return tuple(dict.fromkeys(
        token
        for token in expression.replace("(", " ").replace(")", " ").split()
        if token not in ("not", "and", "or")
    ))


def _ast_concepts(node: Any) -> List[str]:
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

    Every invariant across the three encoded chains is a conjunction of literals, so
    the decision format is a list of ``{"concept": ..., "negated": ...}`` rather than a
    nested AST the model has to assemble.
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
    trigger-response transition: it starts locked, admits one authorised way out, and
    returns on a distinct event. The invariants name the locked and authorisation
    concepts and the boundary names the consumed events, so the lifecycle is derived
    rather than asked for and the model writes no state machine.
    """
    if decisions.get("safety_pattern") != LOCKED_UNTIL_RELEASE_PATTERN:
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
            locked = consequent[0]
        elif len(antecedent) == 1 and "lock" in consequent[0].lower():
            locked = locked or consequent[0]
        elif len(antecedent) == 1:
            authorisation = consequent[0]
    if not locked or not authorisation:
        return ()
    # only the component guaranteeing the lock gets a lock lifecycle; applied to
    # every component it built the authorisation gateway a machine out of
    # whatever inputs it consumed
    if locked not in [str(item) for item in produces]:
        return ()

    consumed = [
        str(concept)
        for concept in boundary_component.get("interfaces", {}).get("consumes", ())
    ]
    # events other than the authorisation power the component up and down; their
    # order comes from the boundary interface
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

    The boundary shows a component as ``Contract (part usage : Def)``, so a model may
    name the part instead. Naming the same component by its part or definition does not
    extend the architecture and is resolved; a name belonging to no component still
    fails closed.
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

    Checked against the boundary, not gold: a wrong allocation can still be well
    formed, so only incoherence is refused.
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

    if pattern in INVARIANT_PATTERNS:
        # an invariant pattern states its obligation as invariants; timing budgets
        # are mutually exclusive with it in the profile
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

    if pattern in TRIGGERED_PATTERNS:
        if pattern == TIMED_PATTERN:
            if decisions.get("deadline_seconds") in (None, ""):
                raise DecisionError("a timed pattern needs deadline_seconds")
        else:
            # An untimed triggered pattern apportions nothing: a deadline would make it
            # a timed chain under the wrong name, and composition would then check
            # budgets against a deadline the requirement never set.
            if decisions.get("deadline_seconds") not in (None, ""):
                raise DecisionError(
                    f"{pattern} is untimed and must not carry deadline_seconds"
                )
            budgeted = sorted(
                str(entry.get("component_id"))
                for entry in decided
                if isinstance(entry, Mapping)
                and entry.get("latency_budget_seconds") not in (None, "")
            )
            if budgeted:
                raise DecisionError(
                    f"{pattern} is untimed; no component may carry "
                    f"latency_budget_seconds, got {budgeted}"
                )
        priority = decisions.get("priority")
        if not isinstance(priority, Mapping):
            raise DecisionError(f"{pattern} needs a priority decision")
        members = [str(item) for item in priority.get("members", ())]
        selected = str(priority.get("selected_response") or "")
        catalog = boundary.get("response_catalog")
        if isinstance(catalog, Mapping):
            catalog_entries = [
                item for item in (catalog.get("entries") or ())
                if isinstance(item, Mapping) and item.get("response_id")
            ]
            catalog_members = {
                str(item["response_id"]) for item in catalog_entries
            }
            if len(catalog_members) < 2:
                raise DecisionError(
                    "priority response catalog has fewer than two "
                    "provenance-backed responses"
                )
            unknown = sorted(set(members) - catalog_members)
            omitted = sorted(catalog_members - set(members))
            if unknown or omitted:
                raise DecisionError(
                    "priority.members must exactly match the provenance-backed "
                    f"response catalog; unknown={unknown}, omitted={omitted}"
                )
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

    Structure (component identity, ownership, interfaces) comes from the boundary;
    judgement (pattern, timing, discharge wiring, ordering) from the decisions.
    Element naming is mechanical, so the model guesses no convention.
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

    # The arbiter is the component feeding the producer of the system observation.
    # Under a timed pattern the profile realizes it with the fixed arbitration
    # behaviour, so it is identified structurally rather than named by the model.
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
        # The emitter derives the discharge edge from the matching upstream guarantee,
        # so the decision only says whether an assumption is an environment input or is
        # discharged internally.
        # Concepts declared as typed lifecycle events are consumed but not assumed: a
        # mechanism that assumes its power-on event is locked only once that event has
        # occurred. The boundary merges both into one `consumes` list, so the split is
        # the author's, and a default-safe component cannot be assembled without it.
        events = {
            str(item) for item in entry.get("lifecycle_events", ())
            if isinstance(item, str)
        }
        assumptions = tuple(
            AGAssumptionSpec(
                concept=_identifier(item.get("concept"), "assumption.concept"),
                environment=(
                    item.get("discharged_by") in (None, "", "environment")
                    # a concept nothing in the boundary produces is environmental, however labelled
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
        group = entry.get("timing_segment_group")
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
        if not origin:
            # An untimed triggered pattern declares no interval and so no timing
            # origin, but the arbitration still fires on something: the trigger the
            # priority contract already names, stated once rather than twice.
            priority_decision = decisions.get("priority")
            if isinstance(priority_decision, Mapping):
                origin = str(priority_decision.get("trigger") or "")
        is_arbiter = name == arbiter
        if is_arbiter and origin:
            # the timing origin triggers the arbitration even though it is an
            # environment input rather than an upstream guarantee
            trigger_concept = origin
        behavior = "SafetyResponseArbitration" if is_arbiter else f"{stem}Behavior"
        # the responding action establishes every guarantee the state settles - for
        # the arbiter, the selection and the command it issues
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
            # concurrency is the author's to declare; undeclared stays serial
            timing_segment_group=(
                int(group) if isinstance(group, (int, float)) else None
            ),
        ))

    priority: Optional[AGPrioritySpec] = None
    decided_priority = decisions.get("priority")
    if isinstance(decided_priority, Mapping):
        members = tuple(str(item) for item in decided_priority.get("members", ()))
        selected = str(decided_priority.get("selected_response"))
        catalog = boundary.get("response_catalog")
        provenance_by_member = {
            str(item.get("response_id")): (
                str(item.get("source_kind") or ""),
                str(item.get("source_id") or ""),
            )
            for item in (
                catalog.get("entries", ())
                if isinstance(catalog, Mapping) else ()
            )
            if isinstance(item, Mapping) and item.get("response_id")
        }
        priority = AGPrioritySpec(
            response_set_id=_identifier(
                decided_priority.get("response_set_id") or "RESPONSE_SET_V1",
                "priority.response_set_id",
            ),
            members=members,
            edges=tuple(
                (selected, item) for item in members if item != selected
            ),
            # A timed chain states the trigger as its interval origin; an untimed one
            # states it on the priority decision instead.
            trigger=_identifier(
                decisions.get("timing_origin")
                or decided_priority.get("trigger"),
                "timing_origin",
            ),
            selected_response=selected,
            source_kind=DERIVED_SOURCE_KIND,
            source_id=f"{requirement}_PRIORITY",
            member_provenance=tuple(
                (member, *provenance_by_member[member])
                for member in members
                if member in provenance_by_member
            ),
        )

    deadline = decisions.get("deadline_seconds")
    margin = decisions.get("timing_margin_seconds")
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
        timing_margin=(
            float(margin) if margin not in (None, "") else None
        ),
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
            # an invariant binds the concepts it constrains, so select them too
            *(name for item in invariants
              for name in _ast_concepts(item.trigger_or_antecedent_ast)),
            *(name for item in invariants
              for name in _ast_concepts(item.required_consequent_ast)),
            *_expression_concepts(observation),
        })),
    )
