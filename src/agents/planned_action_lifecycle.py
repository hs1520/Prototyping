"""Deep module for the planned-action terminal lifecycle.

The generation pipeline owns phase ordering and model commits.  This module owns
the knowledge needed on either side of that real seam: which effects may be
materialised before the terminal commit, how the rewrite is guarded, and which
planned effects are audited against the final terminal snapshot.

The two effect selections are deliberately not the same.  Materialisation is
derived from the A/G specifications chosen by this run.  Audit evidence is read
from the model generation plan when it carries schema-10 ``action_effects`` and
falls back to the runtime derivation only when the plan carries none.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional, Sequence, Tuple

from ..dse.functional_behavior import functional_behavior_diagnosis
from ..prototyping.action_effects import (
    LEGACY_AUDIT,
    SEND_EVENT,
    PlannedActionEffect,
    parse_action_effects,
)
from ..prototyping.action_semantics import analyze_action_semantics
from ..simulation.syntax_checker import check_syntax
from ..utils.sysml_text_utils import find_block_end


# Scoped to one chain while the profile is proven end to end.  Widening this is
# a deliberate semantic change: every additional chain rewrites generated text.
_ACTION_EFFECT_CHAINS = ("REQ_SAFE_005",)
_OWNER_SCOPE_RE = re.compile(r"\bpart\s+def\s+([A-Za-z_]\w*)\s*\{")


@dataclass(frozen=True, slots=True)
class ActionDiagnostic:
    """One observable disposition without collapsing independent rewrites."""

    code: str
    requirement_id: str = ""
    owner_def: str = ""
    action_def: str = ""


@dataclass(frozen=True, slots=True)
class PlannedActionPreparation:
    """Immutable value carried from terminal commit to terminal snapshot."""

    model_text: str
    changed: bool
    syntax_disposition: str
    diagnostics: tuple[ActionDiagnostic, ...] = ()
    _audit_effects: tuple[PlannedActionEffect, ...] = field(
        default=(), repr=False, compare=False
    )


@dataclass(frozen=True, slots=True)
class PlannedActionObservation:
    """Read-only evidence from the final terminal model text."""

    _artifact_json: str = field(repr=False, compare=False)

    def to_artifact_dict(self) -> dict[str, Any]:
        """Return a fresh projection so callers cannot mutate stored evidence."""
        return json.loads(self._artifact_json)


def prepare_planned_actions(
    model_text: str,
    *,
    model_plan: Mapping[str, Any] | None,
    ag_plan: Mapping[str, Any] | None,
) -> PlannedActionPreparation:
    """Derive and safely materialise this run's bounded response actions.

    Call after every other behaviour writer and immediately before the existing
    terminal commit.  Expected model problems are returned as diagnostics; an
    unexpected implementation failure remains visible as an exception.
    """
    source = str(model_text or "")
    runtime_effects = _derive_runtime_effects(model_plan, ag_plan)
    plan_effects = parse_action_effects(
        model_plan.get("action_effects")
        if isinstance(model_plan, Mapping)
        else None
    )
    audit_effects = plan_effects or runtime_effects

    candidate, diagnostics = _materialize_runtime_effects(source, runtime_effects)
    if candidate == source:
        return PlannedActionPreparation(
            model_text=source,
            changed=False,
            syntax_disposition="NOT_CHECKED",
            diagnostics=diagnostics,
            _audit_effects=audit_effects,
        )

    before = check_syntax(source).total_errors()
    after = check_syntax(candidate).total_errors()
    if after > before:
        rolled_back = tuple(
            replace(
                item,
                code={
                    "SEND_BODY_APPLIED": "SEND_BODY_ROLLED_BACK",
                    "TYPED_USAGE_APPLIED": "TYPED_USAGE_ROLLED_BACK",
                }.get(item.code, item.code),
            )
            for item in diagnostics
        ) + (ActionDiagnostic(code="SYNTAX_REGRESSION_ROLLBACK"),)
        return PlannedActionPreparation(
            model_text=source,
            changed=False,
            syntax_disposition="REJECTED_SYNTAX_REGRESSION",
            diagnostics=rolled_back,
            _audit_effects=audit_effects,
        )

    return PlannedActionPreparation(
        model_text=candidate,
        changed=True,
        syntax_disposition="ACCEPTED_NON_REGRESSION",
        diagnostics=diagnostics,
        _audit_effects=audit_effects,
    )


def observe_terminal_actions(
    preparation: PlannedActionPreparation,
    terminal_model_text: str,
    *,
    requirements: Sequence[str],
    profile: str = LEGACY_AUDIT,
) -> PlannedActionObservation:
    """Audit the final terminal snapshot without changing qualification.

    ``terminal_model_text`` is intentionally supplied again: collaboration may
    accept a repair after the initial terminal commit.  The observed text, not
    the prepared candidate or the plan, is the semantic authority.
    """
    text = str(terminal_model_text or "")
    report = analyze_action_semantics(
        text,
        action_effect_plan=preparation._audit_effects,
        profile=profile,
    )
    payload = report.to_dict()
    payload["functional_behaviour_diagnosis"] = functional_behavior_diagnosis(
        text,
        list(requirements or ()),
        report.actions,
    )
    return PlannedActionObservation(
        _artifact_json=json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )


def _derive_runtime_effects(
    model_plan: Mapping[str, Any] | None,
    ag_plan: Mapping[str, Any] | None,
) -> tuple[PlannedActionEffect, ...]:
    """Use the A/G specifications this run actually decided.

    ``LLM_DECIDED_SPEC`` may rename every response.  The static chain library is
    only a template: archived models use an arbiter response name different from
    the library's default, so deriving from the library resolves to nothing.
    """
    plan = model_plan if isinstance(model_plan, Mapping) else {}
    architecture_components = plan.get("components") or ()
    connections = plan.get("connections") or ()
    specs = ag_plan.get("specs") or () if isinstance(ag_plan, Mapping) else ()
    effects: list[PlannedActionEffect] = []
    for spec in specs:
        if getattr(spec, "source_requirement", None) not in _ACTION_EFFECT_CHAINS:
            continue
        effects.extend(
            _derive_action_effects(spec, architecture_components, connections)
        )
    return tuple(effects)


def _derive_action_effects(
    chain: Any,
    components: Sequence[Mapping[str, Any]],
    connections: Sequence[Mapping[str, Any]] = (),
) -> tuple[PlannedActionEffect, ...]:
    """Join this run's selected A/G chain to its architecture plan."""
    by_name = {str(item.get("name")): item for item in components}
    produced = {
        _signal_for_guarantee(component.guarantee): component
        for component in chain.components
    }
    effects: list[PlannedActionEffect] = []
    for consumer in chain.components:
        signal = consumer.trigger_signal
        producer = produced.get(signal) if signal else None
        if producer is None or producer.owner_def == consumer.owner_def:
            continue
        action = producer.response_action
        effects.append(PlannedActionEffect(
            requirement_id=chain.source_requirement,
            owner_def=producer.owner_def,
            owner_behavior=producer.behavior,
            response_state=_response_state(chain, producer),
            action_def=action,
            usage_label="on" + action[0].upper() + action[1:],
            effect_kind=SEND_EVENT,
            event_type=signal,
            sender_port=_sender_port(
                producer.owner_def,
                consumer.owner_def,
                by_name,
                connections,
            ),
            consumer_owner_def=consumer.owner_def,
            consumer_behavior=consumer.behavior,
            accept_transition=f"accept{signal}",
            target_state=consumer.response_state,
        ))
    return tuple(effects)


def _signal_for_guarantee(guarantee: str) -> str:
    """Apply the A/G chain convention for an in-chain produced signal.

    A trigger with no producer under this convention is an environment input and
    is deliberately not turned into a planned action effect.
    """
    if not guarantee:
        return ""
    return guarantee[0].upper() + guarantee[1:] + "Signal"


def _response_state(chain: Any, producer: Any) -> str:
    """Resolve the emitted arbitration state rather than guessing its name."""
    from ..prototyping.ag_emitter import arbitration_response_state

    if getattr(producer, "behavior", None) == "SafetyResponseArbitration":
        emitted = arbitration_response_state(chain)
        if emitted:
            return emitted
    return producer.response_state


def _sender_port(
    producer: str,
    consumer: str,
    by_name: Mapping[str, Mapping[str, Any]],
    connections: Sequence[Mapping[str, Any]],
) -> str:
    for link in connections:
        if (
            str(link.get("source_component")) == producer
            and str(link.get("target_component")) == consumer
        ):
            return str(link.get("source_port") or "")
    inbound = {
        str(port.get("name"))
        for port in by_name.get(consumer, {}).get("ports", ())
        if str(port.get("direction")) in ("in", "inout")
    }
    for port in by_name.get(producer, {}).get("ports", ()):
        if (
            str(port.get("direction")) in ("out", "inout")
            and str(port.get("name")) in inbound
        ):
            return str(port.get("name"))
    return ""


def _materialize_runtime_effects(
    model_text: str,
    effects: Sequence[PlannedActionEffect],
) -> tuple[str, tuple[ActionDiagnostic, ...]]:
    text = model_text
    diagnostics: list[ActionDiagnostic] = []
    for effect in effects:
        if effect.effect_kind != SEND_EVENT or not effect.is_complete_identity():
            diagnostics.append(_diagnostic("INCOMPLETE_IDENTITY", effect))
            continue
        span = _owner_span(text, effect.owner_def)
        if span is None:
            diagnostics.append(_diagnostic("OWNER_NOT_FOUND", effect))
            continue
        opening, closing = span
        owner_body = text[opening:closing]
        updated, effect_diagnostics = _rewrite_owner_body(owner_body, effect)
        diagnostics.extend(effect_diagnostics)
        if updated != owner_body:
            text = text[:opening] + updated + text[closing:]
    return text, tuple(diagnostics)


def _diagnostic(code: str, effect: PlannedActionEffect) -> ActionDiagnostic:
    return ActionDiagnostic(
        code=code,
        requirement_id=effect.requirement_id,
        owner_def=effect.owner_def,
        action_def=effect.action_def,
    )


def _owner_span(model_text: str, owner_def: str) -> Optional[Tuple[int, int]]:
    for match in _OWNER_SCOPE_RE.finditer(model_text):
        if match.group(1) != owner_def:
            continue
        opening = model_text.find("{", match.start())
        closing = find_block_end(model_text, opening)
        if closing != -1:
            return opening, closing
    return None


def _rewrite_owner_body(
    owner_body: str,
    effect: PlannedActionEffect,
) -> tuple[str, tuple[ActionDiagnostic, ...]]:
    diagnostics: list[ActionDiagnostic] = []
    updated = owner_body
    action = re.compile(
        rf"\baction\s+def\s+{re.escape(effect.action_def)}\s*\{{"
    ).search(updated)
    if action is None:
        diagnostics.append(_diagnostic("ACTION_NOT_FOUND", effect))
    else:
        opening = updated.find("{", action.start())
        closing = find_block_end(updated, opening)
        if closing == -1:
            diagnostics.append(_diagnostic("ACTION_NOT_FOUND", effect))
        else:
            current = updated[opening + 1:closing]
            send = f"send {effect.event_type}() to {effect.sender_port};"
            if send in current:
                diagnostics.append(_diagnostic("SEND_ALREADY_PRESENT", effect))
            elif current.strip():
                diagnostics.append(_diagnostic("NONEMPTY_BODY", effect))
            else:
                updated = (
                    updated[:opening + 1]
                    + f" {send} "
                    + updated[closing:]
                )
                diagnostics.append(_diagnostic("SEND_BODY_APPLIED", effect))

    typed = f"entry action {effect.usage_label} : {effect.action_def};"
    if typed in updated:
        diagnostics.append(_diagnostic("TYPED_USAGE_ALREADY_PRESENT", effect))
    else:
        bare = re.compile(
            rf"\bentry\s+action\s+{re.escape(effect.action_def)}\s*;"
        )
        rewritten, count = bare.subn(typed, updated, count=1)
        if count:
            updated = rewritten
            diagnostics.append(_diagnostic("TYPED_USAGE_APPLIED", effect))
        else:
            diagnostics.append(_diagnostic("ENTRY_USAGE_NOT_FOUND", effect))
    return updated, tuple(diagnostics)
