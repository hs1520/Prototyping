"""Behavioural verification of FUNCTIONAL actuation/sequencing requirements (roadmap ①).

A functional requirement that mandates a RESPONSE (release the payload, return to base, land,
report health, navigate) has real behavioural evidence only if a reachable state actually
PRODUCES that response — an entry action or a `send <Cmd> to <port>`. We check the response
ACTION (not just a phase name), so "PhaseDelivery exists" doesn't masquerade as "release logic
verified". Outcomes per functional requirement with a recognised intent:
  behaviorally-verified : a reachable state's entry-action / sent-command matches the response.
  behavior-absent       : no reachable state anywhere produces the response (not modelled —
                          honestly flags the gap, e.g. release/health-report logic missing).
Requirements with no recognised functional intent are left to the base classifier.

Scope (honest): checks that the functional RESPONSE ACTION is reachable. For event-driven
waypoint-update and post-flight-report requirements it also checks the named causal trigger
and requires an explicit seconds-valued latency constraint. Other continuous guard semantics
(e.g. delivery distance and abort inhibition) remain scenario-execution roadmap items.
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

# functional intent → (keywords that mark the intent in the requirement text,
#                       markers that mark the produced response in a state's action/send/name)
_FUNC_INTENT = {
    # "deliver" alone is a noun modifier in this domain ("delivery mission",
    # "delivery waypoint") and marked navigation requirements as release
    # obligations when the requirement text was LLM-extracted rather than
    # frozen. Only a phrase that names the act of delivering the payload
    # counts; the response-side markers keep "deliver" because an action
    # named deliverPayload is unambiguous.
    "release":  (("release", "deliver the payload", "deliver payload", "drop",
                  "payload release"),
                 ("release", "deliver", "drop", "payloadcmd", "payloadrelease")),
    "return":   (("return to base", "return-to-base", "rtb", "return trajectory", "return-to-home",
                  "return to launch", "return-to-launch", "rtl", "back to the launch"),
                 # Model-side names: the frozen set said "return-to-base" and the
                 # models wrote returnToBase; an extracted set said "navigate back
                 # to the launch coordinates" and the models wrote returnToLaunch,
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
# More-specific response intents must win over context words.  For example,
# "transmit a post-flight health report after landing" requires REPORT; a
# reachable LAND action is only the trigger/context and must not satisfy it.
_FUNC_INTENT_PRIORITY = (
    "report", "self_test", "release", "return", "land", "navigate"
)

# The built-in values a planner may record as a functional requirement's
# response intent. "none" is a legitimate decision: the requirement obliges no
# discrete response (a continuous property, a data-reception duty, a hover).
# "unverifiable" is the honest record for the opposite failure: the requirement
# DOES oblige a discrete response, but no reachable-action marker can evidence
# it, so the gate holds the model to nothing while the matrix reports the
# uncovered obligation instead of pretending there is none. The set used to be
# closed because the marker table below was the only source of checkable
# evidence; a planner may now also record an intent outside this set by
# declaring its own ``response_markers``, each lexically anchored in the
# requirement's copied effect phrase — the gate then runs the same
# reachable-action check against the declared markers.
RESPONSE_INTENTS: frozenset[str] = frozenset(_FUNC_INTENT) | {
    "none", "unverifiable",
}

# Words too generic to anchor a declared marker: a marker justified only by
# one of these is a marker justified by nothing.
_ANCHOR_STOPWORDS = frozenset((
    "the", "a", "an", "of", "to", "and", "or", "for", "with", "within",
    "shall", "must", "will", "upon", "from", "into", "that", "this", "when",
    "after", "before", "system", "shall", "then", "its", "their", "each",
    "every", "all", "any",
))


def _normalise_marker(marker: str) -> str:
    """Lowercase alphanumeric form used for matching against action names."""
    return re.sub(r"[^a-z0-9]+", "", str(marker or "").lower())


def _anchor_words(effect_concept: str) -> frozenset[str]:
    """Content words of the copied effect phrase, with naive plural stems."""
    words = set()
    for word in re.findall(r"[a-z0-9]+", str(effect_concept or "").lower()):
        if len(word) < 4 or word in _ANCHOR_STOPWORDS:
            continue
        words.add(word)
        if word.endswith("s") and len(word) > 4:
            words.add(word[:-1])
    return frozenset(words)


def marker_anchored_in_effect(marker: str, effect_concept: str) -> bool:
    """True when a declared marker shares a content word with the copied
    effect phrase.

    This is the anti-self-grading rule for declared (out-of-vocabulary)
    intents: the same LLM that plans the behaviours also declares what counts
    as their evidence, so the declaration must be visibly derived from the
    requirement's own effect phrase — the same lexical-representation
    discipline the plan already applies to connection-path endpoints. A
    marker like "alert" is anchored in "alert operators within 5 minutes";
    "hovering" is not, however convenient it would be to match.
    """
    normalised = _normalise_marker(marker)
    if len(normalised) < 3:
        return False
    for word in _anchor_words(effect_concept):
        if word in normalised or normalised in word:
            return True
    return False


def response_markers(intent: str) -> frozenset[str] | None:
    """The model-side markers that satisfy a built-in planned intent, or None
    for "none" / "unverifiable" / declared (out-of-vocabulary) intents."""
    entry = _FUNC_INTENT.get((intent or "").lower())
    return frozenset(entry[1]) if entry else None


def planned_intents_from_model(model) -> "Dict[str, str]":
    """{requirement id: recorded response_intent} read from the generation plan
    a committed model carries in its metadata, or {} when the model carries no
    plan (legacy runs). Keys are normalised to the REQ_XXX_NNN form."""
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

    This is the requirement->behavior binding the plan already records
    (e.g. REQ_SAFE_008 -> PayloadMechanism.DeliveryAbortBehavior); the
    verification matrix uses it to select the anchoring machine directly
    instead of by initial-state-name vocabulary."""
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
    """{requirement id: declared response_markers} read from the generation
    plan a committed model carries in its metadata, or {} when the model
    carries no plan. Only realizations that declare markers appear; built-in
    intents carry none and keep the built-in table's authority."""
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
    """The response intent for one requirement, preferring the planner's
    recorded decision over keyword inference.

    ``planned`` is the ``response_intent`` the generation plan recorded for
    this requirement, or None when no plan is available (legacy runs, or a
    requirement the plan did not cover). A recorded "none" means the planner
    decided the requirement obliges no discrete response, and the gate then
    asks for none; "unverifiable" means a response is obliged but no
    reachable-action marker can evidence it, so the gate likewise holds the
    model to nothing (the matrix reports the uncovered obligation
    separately). A recorded intent outside the built-in table is honoured
    when the plan declared its own ``response_markers`` — the gate runs the
    same reachable-action check against them; a built-in intent always uses
    the built-in markers, so a planner cannot re-define what evidences
    release or navigate. Only when nothing usable was recorded does the
    keyword table decide, so archived runs planned before these fields
    existed keep the same verdicts they had.
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
    """Keyword-inferred response intent. Retained as the fallback for plans
    that carry no recorded intent; see :func:`planned_response_intent`.

    The terminal closure gate credits a functional requirement only when a
    reachable state produces one of these markers, so the generation plan has
    to read the same table. Two copies of it would let the plan freeze a model
    the gate then refuses, with no repair able to close the difference.
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
    """Responses produced by reachable states, with their incoming trigger context."""
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
            # A sustained response is a `do action`, and reading only entry
            # actions made the natural spelling of a continuous activity —
            # navigating, tracking, holding — invisible to this audit while
            # the extractor had captured it all along.  `response_action_for_
            # state` already treats the two as one executable response.
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

    This is intentionally narrow: only causal qualifiers present in the two
    functional sequencing families are enforced. Other functional intents keep
    their existing response-reachability semantics.
    """
    text = req_text.lower()
    context = re.sub(r"[^a-z0-9]+", "", record.trigger_context)
    if "health report" in text and any(k in text for k in ("landing", "post-flight")):
        # The trigger must denote the end of flight. The frozen set spelled it
        # "upon completion of the automated landing sequence" and models wrote
        # AutomatedLandingCompleted; an extracted set spelled the same
        # obligation "upon mission completion" and the model wrote
        # MissionCompletion, which no reading of "land" admits. Accept a
        # landing-completion trigger, or a completion trigger that is not a
        # start-of-flight event; a bare command, a power-on or an arming
        # trigger still does not qualify. (`context` also carries the
        # transition name, so "land" alone is not enough: a transition named
        # transmitReportAfterLanding fired by GenericCommand must not pass.)
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
        # The response being a self-test is already established by the marker
        # match that produced this record; what this rule guards is that the
        # self-test state is entered by a start-of-life event and not by an
        # unrelated command. Requiring "selftest" in the TRIGGER context as well
        # only passed when the transition happened to be named startSelfTest
        # (the frozen set's models); a transition named powerOn firing accept
        # PowerOnEvent into a state whose entry action is selfTest is the same
        # design and must pass.
        return any(k in context for k in ("poweron", "powerup", "startup", "start", "boot", "init"))
    return True


_SECONDS_RE = re.compile(r"\bwithin\s+(\d+(?:\.\d+)?)\s*(?:seconds?|s)\b", re.IGNORECASE)
_TIME_ATTR_RE = re.compile(
    # A seconds-valued bound is emitted as `DurationValue` by the semantic
    # materialiser (requirement_semantics maps the unit "s" to that type) and
    # as `Real` by the typed plan. Recognising only `Real` meant the attribute
    # the constraint actually referenced was invisible here, so a requirement
    # with a perfectly good timing anchor was reported as having none.
    r"attribute\s+(\w+)\s*:\s*(?:Real|DurationValue|TimeValue)\s*=\s*"
    r"(\d+(?:\.\d+)?)\s*\[s\]\s*;",
    re.IGNORECASE,
)
_CONSTRAINT_RE = re.compile(r"assert\s+constraint\s+\w+\s*\{([^{}]+)\}", re.IGNORECASE)


def _has_required_timing_anchor(model_text: str, req_text: str) -> bool:
    """Require an explicit seconds-valued bound linked by an assert constraint."""
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
    actuation/sequencing intent (excludes safety reqs — those go through safety_behavior).

    ``planned_intents`` maps requirement id -> the ``response_intent`` the
    generation plan recorded. When given, it decides which response each
    requirement is held to (a recorded "none" means: hold it to none). When
    absent, the keyword table decides, so archived runs planned before the
    field existed keep their verdicts. ``planned_markers`` carries the
    declared ``response_markers`` for intents outside the built-in table;
    the same reachable-action check then runs against them."""
    produced = _produced_response_records(model_text)
    has_state_machines = bool(extract_state_machines(model_text))
    trace = extract_requirement_trace(model_text, requirements)
    text = trace.source_by_id
    satisfied = trace.satisfied

    out: Dict[str, str] = {}
    for rid in satisfied:
        txt = text.get(rid, rid).lower()
        # The trace's ids are hyphenated (dse_req_id display form); every
        # producer of these mappings keys them REQ_XXX_NNN. Looking up the
        # raw rid silently ignored every recorded intent.
        key = rid.upper().replace("-", "_")
        planned = planned_intents.get(key) if planned_intents else None
        declared = planned_markers.get(key) if planned_markers else None
        intent = planned_response_intent(rid, txt, planned, declared)
        if intent is None:
            continue                                       # no recognised functional intent
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
            out[rid] = BEHAVIOR_ABSENT                     # response action not produced anywhere
    return out


# ---------------------------------------------------------------------------
# Strict diagnosis: what the name-based rule credits, and what survives an
# owner- and type-aware reading of the same model.  Advisory only — nothing
# here changes `functional_behavior_status`, so archived runs stay reproducible.
# ---------------------------------------------------------------------------

#: Why a requirement the legacy rule credits does not survive the strict one.
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

    The legacy rule credits a requirement when some reachable state's action
    *name* contains an intent marker.  The name can be the usage label of a bare
    invocation, whose definition is empty, or even the SysML library type
    ``Action`` that a bare usage resolves to — and it need not belong to the part
    that satisfies the requirement.  The strict reading requires the crediting
    state to invoke a named action definition, owned by the satisfying part,
    whose body is not empty.
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
