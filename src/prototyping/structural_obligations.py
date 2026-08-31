"""Requirement-traceable structural obligations for generated SysML models.

The whole-model plan already fixes typed components, ports, and connections.
This module compiles its requirement-tagged connection graph into stable
source-to-target obligations before SysML generation.  Later validation can
therefore evaluate the same paths for every candidate instead of deriving a
new scenario set from candidate-specific names or ports.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any, Iterable, Mapping, Sequence

from ..utils.req_id import normalise_req_id
from ..utils.sysml_text_utils import find_block_end


_STRUCTURAL_CATEGORIES = ("REQ_FUNC_", "REQ_SAFE_", "REQ_INTF_", "REQ_OPER_")

# The pure-text specialization parser lives in utils so that packages below
# `prototyping` in the dependency order (dse.diagnostics) can resolve `:>`
# without importing this package. Re-exported here for its existing callers.
from ..utils.sysml_text_utils import part_def_bases  # noqa: F401,E402


def specializes(def_name: str, target: str,
                bases: dict[str, tuple[str, ...]]) -> bool:
    """True when *def_name* is *target* or transitively specialises it.

    A usage retyped to a catalogue implementation (``Impl :> Planned``) is
    still, by the language's own subtyping, a usage of the planned
    definition; a reader that matches definition names exactly reports the
    planned component as missing and the implementation as unplanned on a
    model that is right. Measured on an archived end-to-end run.
    """
    seen: set[str] = set()
    frontier = [def_name]
    while frontier:
        current = frontier.pop()
        if current == target:
            return True
        if current in seen:
            continue
        seen.add(current)
        frontier.extend(bases.get(current, ()))
    return False



@dataclass(frozen=True)
class StructuralConnectionRef:
    source_component: str
    source_port: str
    target_component: str
    target_port: str
    item_type: str

    def key(self) -> tuple[str, str, str, str]:
        return (
            self.source_component,
            self.source_port,
            self.target_component,
            self.target_port,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "source_component": self.source_component,
            "source_port": self.source_port,
            "target_component": self.target_component,
            "target_port": self.target_port,
            "item_type": self.item_type,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StructuralConnectionRef":
        source = value.get("source")
        target = value.get("target")
        source_value = source if isinstance(source, Mapping) else {}
        target_value = target if isinstance(target, Mapping) else {}
        return cls(
            source_component=str(
                source_value.get("component")
                or value.get("source_component")
                or ""
            ).strip(),
            source_port=str(
                source_value.get("port")
                or value.get("source_port")
                or ""
            ).strip(),
            target_component=str(
                target_value.get("component")
                or value.get("target_component")
                or ""
            ).strip(),
            target_port=str(
                target_value.get("port")
                or value.get("target_port")
                or ""
            ).strip(),
            item_type=str(value.get("item_type") or "").strip(),
        )


@dataclass(frozen=True)
class RequirementRealizationPlan:
    """Source-anchored primary causal path for one frozen requirement."""

    requirement_id: str
    realization_kind: str
    trigger_concept: str
    effect_concept: str
    connection_path: tuple[StructuralConnectionRef, ...]
    owner_component: str = ""
    behavior_kind: str = ""
    behavior_name: str = ""
    source_digest: str = ""
    # The discrete response a functional requirement obliges, decided by the
    # planner from the requirement text and recorded here so that the plan
    # validator and the terminal closure gate read the same decision instead
    # of each inferring it from keywords. One of RESPONSE_INTENTS, a declared
    # domain intent accompanied by response_markers, or "" when the
    # requirement is not functional. "none" is a valid decision and means the
    # requirement obliges no discrete response (a continuous property, a
    # data-reception duty, a hover); "unverifiable" means a response is
    # obliged but no reachable-action marker can evidence it. Both must be
    # accompanied by a non-empty response_intent_rationale.
    response_intent: str = ""
    response_intent_rationale: str = ""
    # Declared evidence markers for an intent outside the built-in table:
    # lowercase name fragments by which a reachable state's action shows the
    # response. Required (and each lexically anchored in effect_concept) when
    # response_intent is out-of-vocabulary; ignored for built-in intents,
    # whose markers stay the checker's authority.
    response_markers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "realization_kind": self.realization_kind,
            "trigger_concept": self.trigger_concept,
            "effect_concept": self.effect_concept,
            "connection_path": [
                item.to_dict() for item in self.connection_path
            ],
            "owner_component": self.owner_component,
            "behavior_kind": self.behavior_kind,
            "behavior_name": self.behavior_name,
            "source_digest": self.source_digest,
            "response_intent": self.response_intent,
            "response_intent_rationale": self.response_intent_rationale,
            "response_markers": list(self.response_markers),
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "RequirementRealizationPlan":
        path = value.get("connection_path")
        if not isinstance(path, Sequence) or isinstance(path, (str, bytes)):
            path = ()
        return cls(
            requirement_id=normalise_req_id(
                str(value.get("requirement_id") or "")
            ),
            realization_kind=str(
                value.get("realization_kind") or ""
            ).strip().upper(),
            trigger_concept=str(value.get("trigger_concept") or "").strip(),
            effect_concept=str(value.get("effect_concept") or "").strip(),
            connection_path=tuple(
                StructuralConnectionRef.from_dict(item)
                for item in path
                if isinstance(item, Mapping)
            ),
            owner_component=str(
                value.get("owner_component") or ""
            ).strip(),
            behavior_kind=str(
                value.get("behavior_kind") or ""
            ).strip().upper(),
            behavior_name=str(
                value.get("behavior_name") or ""
            ).strip(),
            source_digest=str(value.get("source_digest") or "").strip(),
            response_intent=str(
                value.get("response_intent") or ""
            ).strip().lower(),
            response_intent_rationale=str(
                value.get("response_intent_rationale") or ""
            ).strip(),
            response_markers=tuple(
                marker
                for marker in (
                    str(item or "").strip().lower()
                    for item in (
                        value.get("response_markers")
                        if isinstance(
                            value.get("response_markers"), Sequence
                        )
                        and not isinstance(
                            value.get("response_markers"), (str, bytes)
                        )
                        else ()
                    )
                )
                if marker
            ),
        )


@dataclass(frozen=True)
class StructuralObligation:
    obligation_id: str
    requirement_id: str
    source_component: str
    target_component: str
    required_components: tuple[str, ...]
    required_connections: tuple[StructuralConnectionRef, ...]
    entry_kind: str
    trigger_concept: str = ""
    effect_concept: str = ""
    source_digest: str = ""
    provenance: str = "PLAN_TOPOLOGY"
    realization_kind: str = "CAUSAL_PATH"
    behavior_kind: str = ""
    behavior_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "obligation_id": self.obligation_id,
            "requirement_id": self.requirement_id,
            "source_component": self.source_component,
            "target_component": self.target_component,
            "required_components": list(self.required_components),
            "required_connections": [
                item.to_dict() for item in self.required_connections
            ],
            "entry_kind": self.entry_kind,
            "trigger_concept": self.trigger_concept,
            "effect_concept": self.effect_concept,
            "source_digest": self.source_digest,
            "provenance": self.provenance,
            "realization_kind": self.realization_kind,
            "behavior_kind": self.behavior_kind,
            "behavior_name": self.behavior_name,
        }


def requires_structural_path(requirement_id: str) -> bool:
    return str(requirement_id).upper().startswith(_STRUCTURAL_CATEGORIES)


_CONCEPT_STOPWORDS = {
    "a", "an", "and", "be", "by", "component", "current", "data",
    "for", "from", "in", "into", "of", "on", "or", "shall", "signal",
    "state", "status", "subsystem", "system", "the", "to", "when",
}


def _normalise_phrase(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def _stem(value: str) -> str:
    aliases = {
        "avoidance": "avoid",
        "collision": "collide",
        "detection": "detect",
        "failure": "fail",
        "failures": "fail",
        "navigation": "navigate",
        "propulsive": "propulsion",
    }
    if value in aliases:
        return aliases[value]
    if len(value) > 5 and value.endswith("ing"):
        return value[:-3]
    if len(value) > 4 and value.endswith("ed"):
        return value[:-2]
    if len(value) > 4 and value.endswith("s"):
        return value[:-1]
    return value


def _concept_terms(value: str) -> set[str]:
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(value))
    return {
        _stem(token)
        for token in re.findall(r"[A-Za-z0-9]+", words.lower())
        if token not in _CONCEPT_STOPWORDS
    }


#: Prepositional heads that mark an adjunct clause. A phrase headed by one of
#: these qualifies a behaviour — a tolerance, a deadline, a governing standard —
#: instead of naming one, so no component, port or state can lexically represent
#: it. Measured on the failed authoritative attempt of 2026-08-01, where the
#: planner was obliged to copy `'with a circular error probable (CEP) of less
#: than 1.0 metre'` and `'in accordance with the ASTM F3411-22 standard'`
#: verbatim out of the requirement and was then failed for not representing
#: them. `when`, `if` and `during` are deliberately absent: they introduce a
#: real trigger clause rather than an adjunct.
_ADJUNCT_HEADS = frozenset({"with", "within", "in", "per", "under", "by"})

#: Synonym groups bridging requirement prose and model identifiers, seeded from
#: the same run. `_stem` is morphological only, so it cannot connect a word to
#: its domain synonym: the port named `obstacleData` is exactly what carries a
#: `'collision threat'`, and what a requirement calls `'transmit'` a model calls
#: `telemetry`. Each group is evidence-backed rather than a general thesaurus.
_LEXICAL_BRIDGE_GROUPS = (
    frozenset({"collide", "threat", "obstacle"}),
    frozenset({"telemetry", "transmit"}),
)

#: Shortest stem allowed to stand for a longer one by prefix. Identifiers
#: abbreviate where requirements spell out — `navState` is the port that
#: realises `'navigate'` — and three characters is what `nav` needs.
_MIN_ABBREVIATION_STEM = 3

#: Function words that survive `_CONCEPT_STOPWORDS` and must never act as the
#: abbreviated side of a prefix match, or `not` would stand for `notification`.
_NON_ABBREVIATING = frozenset({
    "as", "at", "is", "it", "less", "no", "not", "of", "than", "to", "with",
})


def _bridged(terms: set[str]) -> set[str]:
    result = set(terms)
    for group in _LEXICAL_BRIDGE_GROUPS:
        if result & group:
            result |= group
    return result


def _represents(phrase_terms: set[str], vocabulary_terms: set[str]) -> bool:
    """Does the model vocabulary lexically stand for the requirement phrase?

    Exact stem overlap first, then the two ways the two vocabularies are known
    to diverge: identifiers abbreviate, and identifiers use the domain synonym.
    """
    if not phrase_terms:
        return True
    left = _bridged(phrase_terms)
    right = _bridged(vocabulary_terms)
    if left & right:
        return True
    for term in left:
        for other in right:
            short, long_ = sorted((term, other), key=len)
            if (
                len(short) >= _MIN_ABBREVIATION_STEM
                and short not in _NON_ABBREVIATING
                and long_.startswith(short)
            ):
                return True
    return False


def _adjunct_reason(value: str) -> str | None:
    """Why no owner can represent this phrase, or None if it names a behaviour."""
    words = _normalise_phrase(value).split()
    if words and words[0] in _ADJUNCT_HEADS:
        return f"{words[0]!r} heads an adjunct clause"
    return None


def _is_initialization_trigger(value: str) -> bool:
    """Recognise lifecycle triggers represented by a state-machine initial edge."""
    phrase = _normalise_phrase(value)
    return any(
        marker in phrase
        for marker in (
            "upon power on",
            "at power on",
            "on power on",
            "during power on",
            "upon startup",
            "at startup",
            "on startup",
            "upon initialization",
            "upon initialisation",
            "when initialized",
            "when initialised",
        )
    )


def _local_behavior_semantic_vocabulary(
    realization: RequirementRealizationPlan,
    planned_behaviors: Sequence[Any],
    behavior_obligations: Sequence[Any],
) -> tuple[set[str], set[str], bool]:
    """Collect typed trigger/effect evidence for one local realization."""
    trigger_terms: set[str] = set()
    effect_terms: set[str] = set()
    initialization_present = False
    for behavior in planned_behaviors:
        if (
            getattr(behavior, "owner", "") != realization.owner_component
            or getattr(behavior, "behavior_id", "") != realization.behavior_name
        ):
            continue
        initial_state = str(getattr(behavior, "initial_state", "") or "")
        initialization_present = bool(initial_state)
        effect_terms.update(_concept_terms(initial_state))
        for state in getattr(behavior, "states", ()) or ():
            effect_terms.update(_concept_terms(
                str(getattr(state, "state_id", "") or "")
            ))
            effect_terms.update(_concept_terms(
                str(getattr(state, "entry_action", "") or "")
            ))
            effect_terms.update(_concept_terms(
                str(getattr(state, "do_action", "") or "")
            ))
        for transition in getattr(behavior, "transitions", ()) or ():
            trigger_terms.update(_concept_terms(
                str(getattr(transition, "trigger", "") or "")
            ))
    for obligation in behavior_obligations:
        if (
            normalise_req_id(str(
                getattr(obligation, "requirement_id", "") or ""
            )) != realization.requirement_id
            or getattr(obligation, "owner_def", "")
            != realization.owner_component
            or getattr(obligation, "realization_kind", "")
            != "STATE_MACHINE"
        ):
            continue
        initialization_present = bool(
            getattr(obligation, "initial_state", None)
        )
        effect_terms.update(_concept_terms(str(
            getattr(obligation, "stable_behavior_id", "") or ""
        )))
        effect_terms.update(_concept_terms(str(
            getattr(obligation, "initial_state", "") or ""
        )))
        for transition in getattr(obligation, "transitions", ()) or ():
            trigger_terms.update(_concept_terms(str(
                getattr(transition, "trigger", "") or ""
            )))
            effect_terms.update(_concept_terms(str(
                getattr(transition, "target", "") or ""
            )))
            effect_terms.update(_concept_terms(str(
                getattr(transition, "action", "") or ""
            )))
    return trigger_terms, effect_terms, initialization_present


def _requirement_map(requirements: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for requirement in requirements:
        source = str(requirement or "").strip()
        match = re.search(
            r"\bREQ[-_][A-Za-z]+[-_]\d+\b",
            source,
            re.IGNORECASE,
        )
        if match is not None:
            result[normalise_req_id(match.group(0))] = source
    return result


def compile_source_anchored_structural_obligations(
    realizations: Sequence[RequirementRealizationPlan],
    components: Sequence[Any],
    connections: Sequence[Any],
    *,
    requirements: Sequence[str] = (),
    require_complete: bool = False,
    planned_behaviors: Sequence[Any] = (),
    behavior_obligations: Sequence[Any] = (),
) -> tuple[tuple[StructuralObligation, ...], tuple[str, ...]]:
    """Validate source-declared causal paths and freeze them as obligations.

    Unlike graph root/sink inference, this compiler never guesses causality
    from topology.  The trigger and effect must be copied from the frozen
    requirement and the ordered path must reuse exact, requirement-traced
    connections from the typed plan.
    """
    issues: list[str] = []
    requirement_sources = _requirement_map(requirements)
    structural_requirements = {
        req_id
        for req_id in requirement_sources
        if requires_structural_path(req_id)
    }
    component_by_name = {
        component.name: component for component in components
    }
    connection_by_key = {
        (
            item.source_component,
            item.source_port,
            item.target_component,
            item.target_port,
        ): item
        for item in connections
    }
    seen: set[str] = set()
    obligations: list[StructuralObligation] = []

    for index, realization in enumerate(realizations):
        prefix = f"requirement_realizations[{index}]"
        req_id = realization.requirement_id
        if not req_id:
            issues.append(f"{prefix}.requirement_id is required")
            continue
        if req_id in seen:
            issues.append(f"duplicate requirement realization for {req_id}")
            continue
        seen.add(req_id)
        if realization.realization_kind not in {
            "CAUSAL_PATH", "LOCAL_BEHAVIOR",
        }:
            issues.append(
                f"{prefix}.realization_kind must be CAUSAL_PATH or "
                "LOCAL_BEHAVIOR"
            )
        if requirements and req_id not in requirement_sources:
            issues.append(f"{prefix} references undeclared {req_id}")
        if not requires_structural_path(req_id):
            issues.append(
                f"{prefix} is not a FUNC, SAFE, INTF, or OPER requirement"
            )

        source_text = requirement_sources.get(req_id, "")
        source_digest = (
            hashlib.sha256(source_text.encode("utf-8")).hexdigest()
            if source_text else realization.source_digest
        )
        source_phrase = _normalise_phrase(source_text)
        for field_name, concept in (
            ("trigger_concept", realization.trigger_concept),
            ("effect_concept", realization.effect_concept),
        ):
            normalised = _normalise_phrase(concept)
            if not normalised:
                issues.append(f"{prefix}.{field_name} is required")
            elif source_text and normalised not in source_phrase:
                issues.append(
                    f"{prefix}.{field_name} must be an exact phrase copied "
                    f"from {req_id}"
                )

        trigger_terms = _concept_terms(realization.trigger_concept)
        effect_terms = _concept_terms(realization.effect_concept)
        concept_valid = True
        if not trigger_terms:
            issues.append(
                f"{prefix}.trigger_concept has no requirement-specific "
                "semantic term"
            )
            concept_valid = False
        if not effect_terms:
            issues.append(
                f"{prefix}.effect_concept has no requirement-specific "
                "semantic term"
            )
            concept_valid = False

        path = realization.connection_path
        if realization.realization_kind == "LOCAL_BEHAVIOR":
            local_valid = concept_valid
            if path:
                issues.append(
                    f"{prefix}.connection_path must be empty for "
                    "LOCAL_BEHAVIOR"
                )
                local_valid = False
            owner = component_by_name.get(realization.owner_component)
            if owner is None:
                issues.append(
                    f"{prefix}.owner_component is not a planned component"
                )
                local_valid = False
            elif req_id not in owner.requirements:
                issues.append(
                    f"{prefix}.owner_component is not allocated {req_id}"
                )
                local_valid = False
            if realization.behavior_kind not in {"STATE_DEF", "ACTION_DEF"}:
                issues.append(
                    f"{prefix}.behavior_kind must be STATE_DEF or ACTION_DEF"
                )
                local_valid = False
            if not re.fullmatch(
                r"[A-Za-z_]\w*", realization.behavior_name
            ):
                issues.append(
                    f"{prefix}.behavior_name is not a SysML identifier"
                )
                local_valid = False
            owner_vocabulary = " ".join((
                realization.owner_component,
                realization.behavior_name,
                str(getattr(owner, "responsibility", "")),
            ))
            owner_terms = _concept_terms(owner_vocabulary)
            (
                typed_trigger_terms,
                typed_effect_terms,
                initialization_present,
            ) = _local_behavior_semantic_vocabulary(
                realization,
                planned_behaviors,
                behavior_obligations,
            )
            lifecycle_trigger = _is_initialization_trigger(
                realization.trigger_concept
            )
            if (
                lifecycle_trigger
                and (
                    realization.behavior_kind != "STATE_DEF"
                    or not initialization_present
                )
            ):
                issues.append(
                    f"{prefix} lifecycle trigger "
                    f"{realization.trigger_concept!r} requires a typed "
                    "state behavior with an initial state"
                )
                local_valid = False
            elif (
                trigger_terms
                and not lifecycle_trigger
                and _adjunct_reason(realization.trigger_concept) is None
                and not _represents(
                    trigger_terms, owner_terms | typed_trigger_terms
                )
            ):
                issues.append(
                    f"{prefix} local behavior owner "
                    f"{realization.owner_component}.{realization.behavior_name}"
                    " does not represent trigger phrase "
                    f"{realization.trigger_concept!r}"
                )
                local_valid = False
            if (
                effect_terms
                and _adjunct_reason(realization.effect_concept) is None
                and not _represents(
                    effect_terms, owner_terms | typed_effect_terms
                )
            ):
                issues.append(
                    f"{prefix} local behavior owner "
                    f"{realization.owner_component}.{realization.behavior_name}"
                    " does not represent effect phrase "
                    f"{realization.effect_concept!r}"
                )
                local_valid = False
            if local_valid:
                obligations.append(StructuralObligation(
                    obligation_id=f"STRUCT_{req_id}_001",
                    requirement_id=req_id,
                    source_component=realization.owner_component,
                    target_component=realization.owner_component,
                    required_components=(realization.owner_component,),
                    required_connections=(),
                    entry_kind="SOURCE_ANCHORED_LOCAL_BEHAVIOR",
                    trigger_concept=realization.trigger_concept,
                    effect_concept=realization.effect_concept,
                    source_digest=source_digest,
                    provenance="FROZEN_REQUIREMENT_REALIZATION",
                    realization_kind="LOCAL_BEHAVIOR",
                    behavior_kind=realization.behavior_kind,
                    behavior_name=realization.behavior_name,
                ))
            continue

        if not path:
            issues.append(f"{prefix}.connection_path must not be empty")
            continue
        path_valid = concept_valid
        for edge_index, edge in enumerate(path):
            key = edge.key()
            planned = connection_by_key.get(key)
            if planned is None:
                issues.append(
                    f"{prefix}.connection_path[{edge_index}] is not an exact "
                    "planned connection"
                )
                path_valid = False
                continue
            if edge.item_type != planned.item_type:
                issues.append(
                    f"{prefix}.connection_path[{edge_index}].item_type does "
                    "not match the planned connection"
                )
                path_valid = False
            if req_id not in planned.requirements:
                issues.append(
                    f"{prefix}.connection_path[{edge_index}] is not traced "
                    f"to {req_id}"
                )
                path_valid = False
            if edge_index and (
                path[edge_index - 1].target_component
                != edge.source_component
            ):
                issues.append(
                    f"{prefix}.connection_path is not component-contiguous "
                    f"at index {edge_index}"
                )
                path_valid = False

        first = path[0]
        last = path[-1]
        source_component = component_by_name.get(first.source_component)
        target_component = component_by_name.get(last.target_component)
        source_vocabulary = " ".join((
            first.source_component,
            first.source_port,
            first.item_type,
            str(getattr(source_component, "responsibility", "")),
        ))
        target_vocabulary = " ".join((
            last.target_component,
            last.target_port,
            last.item_type,
            str(getattr(target_component, "responsibility", "")),
        ))
        if (
            trigger_terms
            and _adjunct_reason(realization.trigger_concept) is None
            and not _represents(
                trigger_terms, _concept_terms(source_vocabulary)
            )
        ):
            issues.append(
                f"{prefix} causal source {first.source_component}."
                f"{first.source_port} does not represent trigger phrase "
                f"{realization.trigger_concept!r}"
            )
            path_valid = False
        if (
            effect_terms
            and _adjunct_reason(realization.effect_concept) is None
            and not _represents(
                effect_terms, _concept_terms(target_vocabulary)
            )
        ):
            issues.append(
                f"{prefix} causal target {last.target_component}."
                f"{last.target_port} does not represent effect phrase "
                f"{realization.effect_concept!r}"
            )
            path_valid = False

        if not path_valid:
            continue
        required_components = (
            first.source_component,
            *(item.target_component for item in path),
        )
        obligations.append(StructuralObligation(
            obligation_id=f"STRUCT_{req_id}_001",
            requirement_id=req_id,
            source_component=first.source_component,
            target_component=last.target_component,
            required_components=tuple(required_components),
            required_connections=tuple(path),
            entry_kind="SOURCE_ANCHORED_CAUSAL_TRIGGER",
            trigger_concept=realization.trigger_concept,
            effect_concept=realization.effect_concept,
            source_digest=source_digest,
            provenance="FROZEN_REQUIREMENT_REALIZATION",
            realization_kind="CAUSAL_PATH",
        ))

    if require_complete:
        for req_id in sorted(structural_requirements - seen):
            issues.append(
                f"{req_id} has no source-anchored requirement realization"
            )
    return tuple(obligations), tuple(issues)


def compile_structural_obligations(
    components: Sequence[Any],
    connections: Sequence[Any],
    *,
    allocated_requirements: Iterable[str] = (),
) -> tuple[tuple[StructuralObligation, ...], tuple[str, ...]]:
    """Compile stable causal paths from a validated typed model plan.

    Connections are grouped by their requirement provenance.  Every simple
    root-to-sink path becomes one obligation, preserving parallel safety paths
    instead of collapsing them into an existential role-level check.
    """
    component_ports = {
        component.name: tuple(component.ports)
        for component in components
    }
    traced_requirements = sorted({
        requirement_id
        for connection in connections
        for requirement_id in connection.requirements
        if requires_structural_path(requirement_id)
    })
    allocated = {
        requirement_id
        for requirement_id in allocated_requirements
        if requires_structural_path(requirement_id)
    }
    issues = [
        f"{requirement_id} has no requirement-traceable structural path"
        for requirement_id in sorted(allocated - set(traced_requirements))
    ]
    obligations: list[StructuralObligation] = []

    for requirement_id in traced_requirements:
        tagged = sorted(
            (
                connection
                for connection in connections
                if requirement_id in connection.requirements
            ),
            key=lambda item: (
                item.source_component,
                item.source_port,
                item.target_component,
                item.target_port,
            ),
        )
        adjacency: dict[str, list[Any]] = {}
        indegree: dict[str, int] = {}
        outdegree: dict[str, int] = {}
        nodes: set[str] = set()
        for connection in tagged:
            source = connection.source_component
            target = connection.target_component
            nodes.update((source, target))
            adjacency.setdefault(source, []).append(connection)
            outdegree[source] = outdegree.get(source, 0) + 1
            indegree[target] = indegree.get(target, 0) + 1
            indegree.setdefault(source, 0)
            outdegree.setdefault(target, 0)

        roots = sorted(node for node in nodes if indegree.get(node, 0) == 0)
        sinks = {node for node in nodes if outdegree.get(node, 0) == 0}
        paths: list[tuple[Any, ...]] = []

        def visit(
            node: str,
            path: tuple[Any, ...],
            visited: frozenset[str],
        ) -> None:
            if node in sinks and path:
                paths.append(path)
                return
            for connection in adjacency.get(node, ()):
                target = connection.target_component
                if target in visited:
                    continue
                visit(
                    target,
                    (*path, connection),
                    visited | {target},
                )

        for root in roots:
            visit(root, (), frozenset({root}))

        # A closed directed loop has no root/sink pair.  Preserve its exact
        # planned edges as bounded one-edge obligations rather than inventing
        # an arbitrary break point.
        if not paths:
            paths = [(connection,) for connection in tagged]

        unique_paths: dict[
            tuple[tuple[str, str, str, str], ...],
            tuple[Any, ...],
        ] = {}
        for path in paths:
            key = tuple(
                (
                    item.source_component,
                    item.source_port,
                    item.target_component,
                    item.target_port,
                )
                for item in path
            )
            unique_paths.setdefault(key, path)

        for index, path in enumerate(unique_paths.values(), 1):
            source = path[0].source_component
            target = path[-1].target_component
            required_components = (
                source,
                *(item.target_component for item in path),
            )
            external_entry = any(
                port.external and port.direction in {"in", "inout"}
                for port in component_ports.get(source, ())
            )
            obligations.append(StructuralObligation(
                obligation_id=f"STRUCT_{requirement_id}_{index:03d}",
                requirement_id=requirement_id,
                source_component=source,
                target_component=target,
                required_components=tuple(required_components),
                required_connections=tuple(
                    StructuralConnectionRef(
                        source_component=item.source_component,
                        source_port=item.source_port,
                        target_component=item.target_component,
                        target_port=item.target_port,
                        item_type=item.item_type,
                    )
                    for item in path
                ),
                entry_kind=(
                    "EXTERNAL_BOUNDARY"
                    if external_entry
                    else "INTERNAL_SOURCE"
                ),
            ))

    return tuple(obligations), tuple(issues)


def validate_structural_obligations(
    model_text: str,
    obligations: Sequence[StructuralObligation],
    *,
    model_name: str,
) -> dict[str, Any]:
    """Validate frozen structural obligations against one terminal SysML model."""
    from ..simulation.extractor import extract_behavioral_graph

    graph = extract_behavioral_graph(
        model_text,
        root_package=model_name,
    )
    bases = part_def_bases(model_text)
    usages_by_definition: dict[str, list[str]] = {}
    for usage_name, part in graph.parts.items():
        indexed = {part.def_name}
        frontier = [part.def_name]
        while frontier:
            current = frontier.pop()
            for base in bases.get(current, ()):
                if base not in indexed:
                    indexed.add(base)
                    frontier.append(base)
        for def_name in indexed:
            bucket = usages_by_definition.setdefault(def_name, [])
            if usage_name not in bucket:
                bucket.append(usage_name)
    for usages in usages_by_definition.values():
        usages.sort()

    def resolve(component: str) -> tuple[str | None, str | None]:
        usages = usages_by_definition.get(component, ())
        conventional = component[:1].lower() + component[1:]
        if conventional in usages:
            return conventional, None
        if len(usages) == 1:
            return usages[0], None
        if not usages:
            return None, f"component definition {component} has no system usage"
        return None, (
            f"component definition {component} has ambiguous usages: "
            f"{', '.join(usages)}"
        )

    actual_connections = {
        (
            connection.source.split(".", 1)[0],
            connection.source.split(".", 1)[1],
            connection.target.split(".", 1)[0],
            connection.target.split(".", 1)[1],
        )
        for connection in graph.connections
        if "." in connection.source and "." in connection.target
    }
    results: list[dict[str, Any]] = []

    for obligation in obligations:
        issues: list[str] = []
        resolved: dict[str, str] = {}
        for component in obligation.required_components:
            usage, issue = resolve(component)
            if issue:
                issues.append(issue)
            elif usage is not None:
                resolved[component] = usage

        missing_behavior_elements: list[str] = []
        if obligation.realization_kind == "LOCAL_BEHAVIOR":
            owner_match = re.search(
                rf"\bpart\s+def\s+"
                rf"{re.escape(obligation.source_component)}\s*\{{",
                model_text,
            )
            owner_body = ""
            if owner_match is not None:
                opening = model_text.find(
                    "{", owner_match.start(), owner_match.end()
                )
                closing = find_block_end(model_text, opening)
                if closing != -1:
                    owner_body = model_text[opening + 1:closing]
            keyword = (
                "state" if obligation.behavior_kind == "STATE_DEF"
                else "action"
            )
            if not owner_body or re.search(
                rf"\b{keyword}\s+def\s+"
                rf"{re.escape(obligation.behavior_name)}\b",
                owner_body,
            ) is None:
                missing_behavior_elements.append(
                    f"{obligation.behavior_kind} "
                    f"{obligation.source_component}."
                    f"{obligation.behavior_name}"
                )
                issues.append(
                    "missing required local behavior "
                    + missing_behavior_elements[-1]
                )

        missing_connections: list[str] = []
        for connection in obligation.required_connections:
            source = resolved.get(connection.source_component)
            target = resolved.get(connection.target_component)
            if source is None or target is None:
                continue
            key = (
                source,
                connection.source_port,
                target,
                connection.target_port,
            )
            if key not in actual_connections:
                missing_connections.append(
                    f"{source}.{connection.source_port} -> "
                    f"{target}.{connection.target_port}"
                )
        if missing_connections:
            issues.extend(
                f"missing required connection {item}"
                for item in missing_connections
            )

        observed_path: list[str] = []
        if not missing_connections and not issues:
            if obligation.realization_kind == "LOCAL_BEHAVIOR":
                observed_path = [
                    resolved[obligation.source_component],
                    f"{obligation.behavior_kind} "
                    f"{obligation.behavior_name}",
                ]
            else:
                for index, connection in enumerate(
                    obligation.required_connections
                ):
                    source = resolved[connection.source_component]
                    target = resolved[connection.target_component]
                    segment = [
                        source,
                        f"{source}.{connection.source_port}",
                        f"{target}.{connection.target_port}",
                        target,
                    ]
                    observed_path.extend(
                        segment if index == 0 else segment[1:]
                    )

        results.append({
            **obligation.to_dict(),
            "status": "PASS" if not issues else "FAIL",
            "resolved_usages": resolved,
            "observed_path": observed_path,
            "missing_connections": missing_connections,
            "missing_behavior_elements": missing_behavior_elements,
            "issues": list(dict.fromkeys(issues)),
        })

    passed = sum(item["status"] == "PASS" for item in results)
    if not obligations:
        status = "UNVERIFIED"
    else:
        status = "PASS" if passed == len(results) else "FAIL"
    return {
        "schema_version": "2.0",
        "artifact_role": "REQUIREMENT_STRUCTURAL_OBLIGATION_VALIDATION",
        "status": status,
        "scenario_set_fixed": True,
        "model_name": model_name,
        "source_model_digest": hashlib.sha256(
            (model_text or "").encode("utf-8")
        ).hexdigest(),
        "passed": passed,
        "total": len(results),
        "results": results,
    }
