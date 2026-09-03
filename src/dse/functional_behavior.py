"""Behavioural verification of functional actuation/sequencing requirements (roadmap ①).

A functional requirement mandating a response (release the payload, return to base,
land, report health, navigate) has behavioural evidence only if a reachable state
produces that response - an entry action or a `send <Cmd> to <port>`. The response
action is checked, not the phase name, so "PhaseDelivery exists" is not read as
verified release logic. Outcomes per requirement with a recognised intent:
  behaviorally-verified : a reachable state's entry-action / sent-command matches.
  behavior-absent       : no reachable state produces the response (not modelled).
Requirements with no recognised functional intent go to the base classifier.

Scope: response-action reachability. For event-driven waypoint-update and
post-flight-report requirements it also checks the named causal trigger and requires
an explicit seconds-valued latency constraint. Other continuous guard semantics
(delivery distance, abort inhibition) remain scenario-execution roadmap items.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Dict, List, Protocol, Sequence, Set

from ..simulation.state_extractor import extract_state_machines
from .requirement_trace import dse_req_id, extract_requirement_trace
from .safety_behavior import (
    BEHAVIOR_ABSENT,
    BEHAVIORALLY_VERIFIED,
    is_safety_req,
    reachable_states,
)

_FUNC_INTENT = {
    # "deliver" alone is a noun modifier here ("delivery mission", "delivery
    # waypoint") and marked navigation requirements as release obligations in
    # LLM-extracted text, so only a phrase naming the act of delivering the
    # payload counts. Response-side markers keep "deliver": an action named
    # deliverPayload is unambiguous.
    "release":  (("release", "deliver the payload", "deliver payload", "drop",
                  "payload release"),
                 ("release", "deliver", "drop", "payloadcmd", "payloadrelease")),
    "return":   (("return to base", "return-to-base", "rtb", "return trajectory", "return-to-home",
                  "return to launch", "return-to-launch", "rtl", "back to the launch"),
                 # Model-side names: "return-to-base" produced returnToBase, while
                 # "navigate back to the launch coordinates" produced returnToLaunch,
                 # which no marker matched. RTL is also the autopilot's own term.
                 ("rtb", "rtl", "returnhome", "returntohome", "returntobase",
                  "returntolaunch", "returnlaunch", "gohome")),
    "land":     (("land", "landing", "touchdown"), ("land", "touchdown")),
    "navigate": (("navigate", "waypoint", "gps waypoint", "flight plan"),
                 ("navigate", "waypoint", "gotowaypoint", "nav")),
    "report":   (("health report", "status report", "post-flight", "health and status report"),
                 ("report", "healthreport", "postflight", "telemetryreport")),
    "self_test": (("self-test", "self test", "self-check", "self check"),
                  ("selftest", "selfcheck")),
}
# More-specific response intents win over context words: "transmit a
# post-flight health report after landing" needs REPORT, and a reachable
# LAND action is only the trigger, so it does not satisfy it.
_FUNC_INTENT_PRIORITY = (
    "report", "self_test", "release", "return", "land", "navigate"
)

# The built-in values a planner may record as a functional requirement's
# response intent. "none" means the requirement obliges no discrete response
# (a continuous property, a data-reception duty, a hover). "unverifiable"
# means a discrete response is obliged but no reachable-action marker can
# evidence it, so the gate holds the model to nothing while the matrix
# reports the uncovered obligation. A planner may also record an intent
# outside this set by declaring its own ``response_markers``, each lexically
# anchored in the requirement's copied effect phrase; the gate then runs the
# same reachable-action check against the declared markers.
RESPONSE_INTENTS: frozenset[str] = frozenset(_FUNC_INTENT) | {
    "none", "unverifiable",
}

# Words too generic to anchor a declared marker; a marker justified only by
# one of these is justified by nothing.
_ANCHOR_STOPWORDS = frozenset((
    "the", "a", "an", "of", "to", "and", "or", "for", "with", "within",
    "shall", "must", "will", "upon", "from", "into", "that", "this", "when",
    "after", "before", "system", "shall", "then", "its", "their", "each",
    "every", "all", "any",
))


def _normalise_marker(marker: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(marker or "").lower())


def _anchor_words(effect_concept: str) -> frozenset[str]:
    words = set()
    for word in re.findall(r"[a-z0-9]+", str(effect_concept or "").lower()):
        if len(word) < 4 or word in _ANCHOR_STOPWORDS:
            continue
        words.add(word)
        if word.endswith("s") and len(word) > 4:
            words.add(word[:-1])
    return frozenset(words)


def marker_anchored_in_effect(marker: str, effect_concept: str) -> bool:
    """True when a declared marker shares a content word with the copied effect phrase.

    Anti-self-grading rule for declared (out-of-vocabulary) intents: the LLM that plans
    the behaviours also declares their evidence, so the declaration has to derive from
    the requirement's own effect phrase - the lexical discipline the plan already applies
    to connection-path endpoints. "alert" is anchored in "alert operators within 5
    minutes"; "hovering" is not.
    """
    normalised = _normalise_marker(marker)
    if len(normalised) < 3:
        return False
    for word in _anchor_words(effect_concept):
        if word in normalised or normalised in word:
            return True
    return False


def response_markers(intent: str) -> frozenset[str] | None:
    """The model-side markers that satisfy a built-in planned intent, or None for "none" /
    "unverifiable" / declared (out-of-vocabulary) intents.
    """
    entry = _FUNC_INTENT.get((intent or "").lower())
    return frozenset(entry[1]) if entry else None


def planned_intents_from_model(model) -> "Dict[str, str]":
    """{requirement id: recorded response_intent} read from the generation plan a committed model
    carries in its metadata, or {} when the model carries no plan (legacy runs).
    """
    meta = getattr(model, "metadata", None) or {}
    plan = meta.get("whole_model_generation_plan") if isinstance(meta, dict) else None
    if not isinstance(plan, dict):
        return {}
    out: Dict[str, str] = {}
    for item in plan.get("requirement_realizations") or ():
        if not isinstance(item, dict):
            continue
        rid = str(item.get("requirement_id") or "").strip().upper().replace("-", "_")
        intent = str(item.get("response_intent") or "").strip().lower()
        if rid and intent:
            out[rid] = intent
    return out


def planned_behavior_bindings_from_model(model) -> "Dict[str, tuple]":
    """{requirement id: (owner_component, behavior_name)} read from the plan's
    requirement_realizations, or {} when the model carries no plan.

    The plan already records this binding (REQ_SAFE_008 ->
    PayloadMechanism.DeliveryAbortBehavior); the verification matrix uses it to select the
    anchoring machine directly rather than by initial-state-name vocabulary.
    """
    meta = getattr(model, "metadata", None) or {}
    plan = meta.get("whole_model_generation_plan") if isinstance(meta, dict) else None
    if not isinstance(plan, dict):
        return {}
    out: Dict[str, tuple] = {}
    for item in plan.get("requirement_realizations") or ():
        if not isinstance(item, dict):
            continue
        rid = str(item.get("requirement_id") or "").strip().upper().replace("-", "_")
        behavior = str(item.get("behavior_name") or "").strip()
        owner = str(item.get("owner_component") or "").strip()
        if rid and behavior:
            out[rid] = (owner, behavior)
    return out


def planned_markers_from_model(model) -> "Dict[str, frozenset[str]]":
    """{requirement id: declared response_markers} read from the plan a committed model
    carries in its metadata, or {} when the model carries no plan.

    Only realizations that declare markers appear; built-in intents carry none and
    keep the built-in table's authority.
    """
    meta = getattr(model, "metadata", None) or {}
    plan = meta.get("whole_model_generation_plan") if isinstance(meta, dict) else None
    if not isinstance(plan, dict):
        return {}
    out: Dict[str, frozenset[str]] = {}
    for item in plan.get("requirement_realizations") or ():
        if not isinstance(item, dict):
            continue
        rid = str(item.get("requirement_id") or "").strip().upper().replace("-", "_")
        raw = item.get("response_markers")
        if not rid or not isinstance(raw, (list, tuple)):
            continue
        markers = frozenset(
            m for m in (_normalise_marker(v) for v in raw) if m
        )
        if markers:
            out[rid] = markers
    return out


def planned_response_intent(
    req_id: str,
    req_text: str,
    planned: str | None,
    declared_markers: "Iterable[str] | None" = None,
) -> tuple[str, frozenset[str]] | None:
    """The response intent for one requirement, preferring the planner's recorded
    decision over keyword inference.

    ``planned`` is the ``response_intent`` the plan recorded, or None for legacy runs and
    requirements the plan did not cover. A recorded "none" means no discrete response is
    obliged and the gate asks for none; "unverifiable" means one is obliged but no
    reachable-action marker can evidence it, so the gate likewise holds the model to
    nothing (the matrix reports the obligation separately). An intent outside the built-in
    table is honoured when the plan declared its own ``response_markers``, checked the same
    way; a built-in intent always uses the built-in markers, so a planner cannot redefine
    what evidences release or navigate. With nothing usable recorded the keyword table
    decides, so runs planned before these fields existed keep their verdicts.
    """
    if planned is not None:
        planned = planned.strip().lower()
        if planned in ("none", "unverifiable"):
            return None
        markers = response_markers(planned)
        if markers is not None:
            return planned, markers
        if declared_markers:
            declared = frozenset(
                m for m in (_normalise_marker(v) for v in declared_markers) if m
            )
            if declared:
                return planned, declared
        # An unrecognised recorded value with no declared markers is treated
        # as absent rather than as a silent pass: fall through to inference.
    return functional_response_intent(req_id, req_text)


def functional_response_intent(
    req_id: str, req_text: str
) -> tuple[str, frozenset[str]] | None:
    """Keyword-inferred response intent, the fallback for plans that carry no recorded
    intent; see :func:`planned_response_intent`.

    The terminal closure gate credits a functional requirement only when a reachable
    state produces one of these markers, so the generation plan reads the same table.
    Two copies would let the plan freeze a model the gate then refuses, with no repair
    able to close the difference.
    """
    text = (req_text or "").lower()
    if "FUNC" not in (req_id or "").upper() or is_safety_req(req_id, text):
        return None
    for intent in _FUNC_INTENT_PRIORITY:
        keywords, responses = _FUNC_INTENT[intent]
        if any(keyword in text for keyword in keywords):
            return intent, frozenset(responses)
    return None


@dataclass(frozen=True)
class _ProducedResponse:
    names: frozenset[str]
    trigger_context: str


class ActionBodyEvidence(Protocol):
    """The action evidence strict diagnosis needs from an upstream audit."""

    name: str
    body_empty: bool


def _produced_response_records(model_text: str) -> List[_ProducedResponse]:
    out: List[_ProducedResponse] = []
    for sm in extract_state_machines(model_text):
        reach = reachable_states(sm)
        for s in sm.states:
            if s.name not in reach:
                continue
            names: Set[str] = set()
            if s.entry_action:
                names.add(s.entry_action.lower())
            if s.entry_action_def:
                names.add(s.entry_action_def.lower())
            # A sustained response is a `do action`; reading only entry actions hid
            # continuous activities (navigating, tracking, holding) the extractor had
            # already captured. `response_action_for_state` treats the two as one
            # executable response.
            if s.do_action:
                names.add(s.do_action.lower())
            if s.do_action_def:
                names.add(s.do_action_def.lower())
            for cmd, _port in s.sends:
                names.add(cmd.lower())
            if not names:
                continue
            incoming = [
                t for t in sm.transitions
                if not t.is_initial and t.target == s.name and t.source in reach
            ]
            context = " ".join(
                str(value or "")
                for t in incoming
                for value in (
                    t.name, t.source, t.target, t.accept_trigger,
                    " ".join(g.description() for g in t.guards),
                )
            ).lower()
            out.append(_ProducedResponse(frozenset(names), context))
    return out


def _has_required_trigger(req_text: str, record: _ProducedResponse) -> bool:
    """Reject a response action reached through an unrelated event.

    Narrow by design: only the causal qualifiers of the two functional sequencing
    families are enforced; other functional intents keep plain response-reachability
    semantics.
    """
    text = req_text.lower()
    context = re.sub(r"[^a-z0-9]+", "", record.trigger_context)
    if "health report" in text and any(k in text for k in ("landing", "post-flight")):
        # The trigger must denote the end of flight. "upon completion of the
        # automated landing sequence" produced AutomatedLandingCompleted, while
        # "upon mission completion" produced MissionCompletion, which no reading
        # of "land" admits. Accept a landing-completion trigger, or a completion
        # trigger that is not a start-of-flight event; a bare command, a power-on
        # or an arming trigger does not qualify. (`context` also carries the
        # transition name, so "land" alone is not enough: a transition named
        # transmitReportAfterLanding fired by GenericCommand does not pass.)
        if "land" in context and "complet" in context:
            return True
        return "complet" in context and not any(
            k in context for k in ("poweron", "startup", "arm", "takeoff", "launch")
        )
    if "waypoint" in text and any(k in text for k in (
        "modification command", "waypoint-modification", "revised waypoint",
    )):
        waypoint_event = "waypoint" in context and any(
            k in context for k in ("modification", "revision", "revised", "update")
        )
        valid_qualified = "valid" not in text or "valid" in context
        return waypoint_event and valid_qualified
    if any(k in text for k in ("self-test", "self test", "self-check", "self check")):
        # The marker match that produced this record already establishes the
        # self-test response; this rule only checks that the self-test state is
        # entered by a start-of-life event rather than an unrelated command.
        # Requiring "selftest" in the trigger context passed only when the
        # transition was named startSelfTest; powerOn accepting PowerOnEvent into
        # a state whose entry action is selfTest is the same design.
        return any(k in context for k in ("poweron", "powerup", "startup", "start", "boot", "init"))
    return True


_SECONDS_RE = re.compile(r"\bwithin\s+(\d+(?:\.\d+)?)\s*(?:seconds?|s)\b", re.IGNORECASE)
_TIME_ATTR_RE = re.compile(
    # A seconds-valued bound is emitted as `DurationValue` by the semantic
    # materialiser (requirement_semantics maps the unit "s" to that type) and
    # as `Real` by the typed plan. Recognising only `Real` hid the attribute
    # the constraint referenced, so a requirement with a timing anchor read as
    # having none.
    r"attribute\s+(\w+)\s*:\s*(?:Real|DurationValue|TimeValue)\s*=\s*"
    r"(\d+(?:\.\d+)?)\s*\[s\]\s*;",
    re.IGNORECASE,
)
_CONSTRAINT_RE = re.compile(r"assert\s+constraint\s+\w+\s*\{([^{}]+)\}", re.IGNORECASE)


def _has_required_timing_anchor(model_text: str, req_text: str) -> bool:
    match = _SECONDS_RE.search(req_text)
    if not match:
        return True
    expected = float(match.group(1))
    low = req_text.lower()
    family = "report" if "health report" in low else "waypoint" if "waypoint" in low else ""
    constraints = " ".join(_CONSTRAINT_RE.findall(model_text)).lower()
    for name, value in _TIME_ATTR_RE.findall(model_text):
        key = name.lower()
        if abs(float(value) - expected) > 1e-9:
            continue
        if family and family not in key:
            continue
        if not any(k in key for k in ("latency", "delay", "time")):
            continue
        if key in constraints:
            return True
    return False


def functional_behavior_status(
    model_text: str,
    requirements: List[str],
    planned_intents: "Mapping[str, str] | None" = None,
    planned_markers: "Mapping[str, Iterable[str]] | None" = None,
) -> Dict[str, str]:
    """{functional req_id: behaviour status} for FUNC requirements with a recognised
    actuation/sequencing intent; safety reqs go through safety_behavior.

    ``planned_intents`` maps requirement id -> the ``response_intent`` the generation plan
    recorded and decides which response each requirement is held to (a recorded "none"
    holds it to none); without it the keyword table decides, so runs planned before the
    field existed keep their verdicts. ``planned_markers`` carries the declared
    ``response_markers`` for intents outside the built-in table, checked the same way.
    """
    produced = _produced_response_records(model_text)
    has_state_machines = bool(extract_state_machines(model_text))
    trace = extract_requirement_trace(model_text, requirements)
    text = trace.source_by_id
    satisfied = trace.satisfied

    out: Dict[str, str] = {}
    for rid in satisfied:
        txt = text.get(rid, rid).lower()
        # The trace's ids are hyphenated (dse_req_id display form); every
        # producer of these mappings keys them REQ_XXX_NNN, so looking up the
        # raw rid ignored every recorded intent.
        key = rid.upper().replace("-", "_")
        planned = planned_intents.get(key) if planned_intents else None
        declared = planned_markers.get(key) if planned_markers else None
        intent = planned_response_intent(rid, txt, planned, declared)
        if intent is None:
            continue
        markers: Set[str] = set(intent[1])
        response_records = [
            record for record in produced
            if any(marker in name for name in record.names for marker in markers)
        ]
        hit = (
            any(_has_required_trigger(txt, record) for record in response_records)
            and _has_required_timing_anchor(model_text, txt)
        )
        if hit:
            out[rid] = BEHAVIORALLY_VERIFIED
        elif has_state_machines:
            out[rid] = BEHAVIOR_ABSENT
    return out


# ---------------------------------------------------------------------------
# Strict diagnosis: what the name-based rule credits, and what survives an
# owner- and type-aware reading of the same model.  Advisory only - nothing
# here changes `functional_behavior_status`, so archived runs stay reproducible.
# ---------------------------------------------------------------------------

NAME_MATCH_ONLY = "credited by a name match on an action that does nothing"
BARE_INVOCATION_ONLY = "the crediting state invokes no action definition"
FOREIGN_OWNER = "the crediting response belongs to another part"
NO_STRICT_DIFFERENCE = ""


def functional_behavior_diagnosis(
    model_text: str,
    requirements: List[str],
    action_records: Sequence[ActionBodyEvidence],
) -> Dict[str, Dict[str, str]]:
    """Per functional requirement: legacy verdict, strict verdict, and the gap.

    The legacy rule credits a requirement when a reachable state's action *name* contains
    an intent marker - which can be the usage label of a bare invocation with an empty
    definition, or the SysML library type ``Action`` a bare usage resolves to, and need
    not belong to the satisfying part. The strict reading requires the crediting state to
    invoke a named action definition, owned by the satisfying part, with a non-empty body.
    """
    legacy = functional_behavior_status(model_text, requirements)
    by_name = {record.name.lower(): record for record in action_records}
    trace = extract_requirement_trace(model_text, requirements)
    owners = {
        req_id: sorted(parts)[0]
        for req_id, parts in trace.owners.items()
        if parts
    }
    text = trace.source_by_id

    strict_support: Dict[str, List[tuple]] = {}
    for machine in extract_state_machines(model_text):
        reachable = reachable_states(machine)
        for state in machine.states:
            if state.name not in reachable:
                continue
            definition = state.entry_action_def or state.do_action_def
            record = by_name.get(str(definition).lower()) if definition else None
            names = {
                value.lower() for value in (
                    state.entry_action, state.do_action, definition
                ) if value
            }
            names.update(command.lower() for command, _ in state.sends)
            strict_support.setdefault(machine.owner_part or "", []).append(
                (frozenset(names), record)
            )

    out: Dict[str, Dict[str, str]] = {}
    for rid, legacy_status in legacy.items():
        markers: Set[str] = set()
        low = text.get(rid, rid).lower()
        for intent in _FUNC_INTENT_PRIORITY:
            keywords, response = _FUNC_INTENT[intent]
            if any(keyword in low for keyword in keywords):
                markers = set(response)
                break
        owner = owners.get(dse_req_id(rid))
        reason = NAME_MATCH_ONLY
        strict = BEHAVIOR_ABSENT
        if legacy_status != BEHAVIORALLY_VERIFIED:
            strict, reason = legacy_status, NO_STRICT_DIFFERENCE
        else:
            foreign_only = True
            for part, records in strict_support.items():
                for names, record in records:
                    if not any(marker in name
                               for name in names for marker in markers):
                        continue
                    if owner and part != owner:
                        continue
                    foreign_only = False
                    if record is None:
                        reason = BARE_INVOCATION_ONLY
                    elif record.body_empty:
                        reason = NAME_MATCH_ONLY
                    else:
                        strict, reason = BEHAVIORALLY_VERIFIED, NO_STRICT_DIFFERENCE
                        break
                if strict == BEHAVIORALLY_VERIFIED:
                    break
            if strict != BEHAVIORALLY_VERIFIED and foreign_only:
                reason = FOREIGN_OWNER
        out[rid] = {
            "legacy_status": legacy_status,
            "strict_status": strict,
            "status_difference_reason": reason,
        }
    return out
