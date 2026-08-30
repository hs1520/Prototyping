"""Verification strategy matrix: every requirement gets an explicit tier + method.

"SITL unmapped" is not "unverified": endurance closes at the datasheet tier,
phase sequencing is the behavioral-sim tier's job, and IP54/regulatory items are
inspection/analysis work outside any simulation toolchain. This module derives —
from artifacts that already exist (linker specs, Phase 8 per-requirement scopes,
model state machines, deterministic keyword rules) — a per-requirement assignment,
so the honest gap ("unassigned") is explicit and small instead of an undifferentiated
"20/36 unmapped" bucket. Each requirement is also decomposed into an observable
behaviour obligation and its explicit physical thresholds; requirement-level
``verified`` is gated on every such obligation being verified.

Method vocabulary follows the systems-engineering IADT convention
(Inspection / Analysis / Demonstration / Test).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .verification_obligations import (
    EvidenceClaim,
    ObligationResult,
    all_obligations_verified,
    compile_verification_obligations,
    evaluate_obligations,
    is_inhibition_requirement,
)

# Canonical tiers, ordered from executable test downwards. A requirement may hold
# several (e.g. max airspeed: L1 param + forward-flight analysis).
TIER_METHOD: Dict[str, str] = {
    "l2_sitl": "Test (native SITL)",
    "l2_sitl_planned": "Test (native SITL — planned, not executed)",
    "l2_sitl_failed": "Test (native SITL — failed)",
    "gazebo": "Test (Gazebo high-fidelity FDM)",
    "gazebo_partial": "Test (Gazebo FDM — partial requirement evidence)",
    "gazebo_failed": "Test (Gazebo high-fidelity FDM — failed)",
    "l1_param": "Inspection (config consistency)",
    "l1_param_planned": "Inspection (config consistency — planned, not executed)",
    "l1_param_failed": "Inspection (config consistency — failed)",
    "datasheet": "Analysis (manufacturer datasheet)",
    "datasheet_failed": "Analysis (manufacturer datasheet — requirement failed)",
    "forward_flight": "Analysis (lumped momentum model)",
    "forward_flight_failed": "Analysis (lumped momentum model — requirement failed)",
    "behavioral_sim": "Analysis (model-level simulation)",
    "behavioral_sim_failed": "Analysis (model-level simulation — failed)",
    "gazebo_deferred": "Test (Gazebo — planned, see SITL_INTEGRATION_DESIGN S8/T9)",
    "inspection_analysis": "Inspection/Analysis (outside simulation scope)",
    "planned_no_response": "Plan records no discrete response obliged",
    "planned_unverifiable_response": (
        "Plan records an obliged response the closure gate cannot check"
    ),
}

_VERIFIED_TIERS = {"l2_sitl", "gazebo", "l1_param", "datasheet", "forward_flight", "behavioral_sim"}
_FAILED_TIERS = {
    "l2_sitl_failed", "l1_param_failed", "gazebo_failed",
    "datasheet_failed", "forward_flight_failed", "behavioral_sim_failed",
}
_PLANNED_TIERS = {"l2_sitl_planned", "l1_param_planned", "gazebo_deferred"}

# Requirements that are inspection/analysis work in any real programme — the
# simulation toolchain honestly cannot test them.
_INSPECTION_KWS = (
    "comply", "compliance", "regulation", "easa", "faa", "astm", "ip54", "ip5",
    "ingress", "temperature", "certif", "material", "encrypt", "aes",
)

# Physics/conditions that need the Gazebo tier (S8 boundary): obstacle physics,
# one-motor-out dynamics, wind conditions, positional release conditions.
_GAZEBO_KWS = (
    "obstacle", "collision", "avoidance", "motor inoperative", "one motor",
    "propulsion unit", "single propulsion", "headwind", "tailwind", "crosswind",
    "gust",
)

#: What each L2 verification actually measures, and therefore which obligation
#: kinds its result may close. The default is the behaviour clause alone: a
#: check that watches a mode change or a servo position has not looked at any
#: numeric threshold, and must not be read as having closed one. A check that
#: times an interval against a limit taken from the model has.
_L2_CLOSES_KINDS = {
    "assert_waypoint_update_latency": {"behavior", "response_time"},
}


def _sim_claim(**kwargs) -> EvidenceClaim:
    """A claim produced by a simulation tier.

    No simulator tests an encryption scheme, an ingress rating or a regulatory
    approval, whatever else the same requirement also asserts. Excluding those
    clauses here means a wire-level MAVLink v2 test can close the protocol half
    of a requirement without appearing to have closed the encrypted-channel
    half — which is exactly what a single whole-sentence obligation used to let
    it do, in both directions.
    """
    kwargs.setdefault("clause_exclude_terms", frozenset(_INSPECTION_KWS))
    return EvidenceClaim(**kwargs)


_BEHAVIORAL_TEXT_KWS = ("phase", "sequence", "sequential", "state", "mode", "transition")
_INITIALIZATION_KWS = ("power-on", "power on", "default", "initial", "startup", "start-up")


@dataclass(frozen=True)
class MatrixRow:
    req_id: str
    text: str
    tiers: Tuple[str, ...]
    methods: Tuple[str, ...]
    status: str  # verified | partial | planned | failed | out-of-sim-scope | blocked | unassigned
    evidence: Tuple[str, ...] = field(default_factory=tuple)
    obligations: Tuple[ObligationResult, ...] = field(default_factory=tuple)


def _l2_strength(inject_kind: str, verify_kind: str) -> str:
    if inject_kind == "mavlink_command":
        return "actuator-existence"
    if verify_kind == "assert_arm_rejected":
        return "proxy-trigger"
    return "behavior-chain"


def _timing_actuation(low: str) -> bool:
    return ("within" in low and "second" in low
            and any(k in low for k in ("actuat", "release", "deploy", "lock")))


def _positional_release(low: str) -> bool:
    return "release" in low and ("metre" in low or "meter" in low)


def _norm_req_id(req_id: object) -> str:
    return str(req_id or "").replace("-", "_")


def _result_map(results) -> Dict[str, bool]:
    """Normalise TestResult objects or report dictionaries by requirement id."""
    mapped: Dict[str, bool] = {}
    for result in results or []:
        if isinstance(result, dict):
            rid = result.get("req_id")
            passed = result.get("passed")
        else:
            rid = getattr(result, "req_id", None)
            passed = getattr(result, "passed", None)
        if rid is not None and passed is not None:
            mapped[_norm_req_id(rid)] = bool(passed)
    return mapped


def _guard_signature(guard) -> tuple:
    return (
        getattr(guard, "kind", None),
        getattr(guard, "attribute", None),
        getattr(guard, "operator", None),
        getattr(guard, "threshold", None),
        getattr(guard, "enum_type", None),
        getattr(guard, "enum_value", None),
    )


def _record_behavioral_outcome(
    rid: str,
    outcomes: List[bool],
    tiers: Dict[str, set],
    evidence: Dict[str, List[str]],
    claims: Dict[str, List[EvidenceClaim]],
    description: str,
    claim_kinds=frozenset({"behavior"}),
) -> bool:
    """Persist real simulator evidence; return True when an outcome existed."""
    if not outcomes:
        return False
    if all(outcomes):
        tiers[rid].add("behavioral_sim")
        detail = f"{description} (PASS)"
        evidence[rid].append(detail)
        claims[rid].append(_sim_claim(
            description=detail, status="verified", kinds=frozenset(claim_kinds),
        ))
    else:
        tiers[rid].add("behavioral_sim_failed")
        detail = f"{description} (FAIL)"
        evidence[rid].append(detail)
        claims[rid].append(_sim_claim(
            description=detail, status="failed", kinds=frozenset(claim_kinds),
        ))
    return True


def build_matrix(model, realization: Optional[dict], requirement_evidence,
                 gazebo: Optional[dict] = None,
                 l1_results=None, l2_results=None,
                 planned_intents: Optional[Dict[str, str]] = None,
                 planned_markers: Optional[Dict[str, frozenset]] = None) -> List[MatrixRow]:
    """Derive the per-requirement verification assignment from existing artifacts.

    ``realization`` is the Phase 8 dict (``realization_run.json``'s "realization"
    value) or None when the latest run produced no recommendation.
    """
    from src.simulation.behavioral_sim import (
        run_behavioral_simulation,
        run_initialization_scenario,
    )
    from src.simulation.state_extractor import extract_state_machines
    from src.dse.functional_behavior import (
        BEHAVIOR_ABSENT,
        BEHAVIORALLY_VERIFIED,
        functional_behavior_status,
        planned_intents_from_model,
        planned_markers_from_model,
    )

    from src.sitl.requirement_linker import RequirementEvidenceBundle

    if not isinstance(requirement_evidence, RequirementEvidenceBundle):
        raise TypeError("build_matrix requires a RequirementEvidenceBundle")
    model_text = model.to_sysml_text() or ""
    model_digest = hashlib.sha256(model_text.encode("utf-8")).hexdigest()
    if requirement_evidence.model_digest != model_digest:
        raise ValueError(
            "requirement evidence does not match the model revision"
        )
    req_texts: Dict[str, str] = dict(requirement_evidence.requirement_texts)
    satisfy_map: Dict[str, List[str]] = {
        req_id: list(parts)
        for req_id, parts in requirement_evidence.satisfying_parts.items()
    }
    guard_assignment = requirement_evidence.guard_assignments
    universe = sorted(set(req_texts) | set(satisfy_map))

    tiers: Dict[str, set] = {r: set() for r in universe}
    evidence: Dict[str, List[str]] = {r: [] for r in universe}
    claims: Dict[str, List[EvidenceClaim]] = {r: [] for r in universe}
    blocked: set = set()

    # 1. Linker specs describe verification intent only.  They become verified
    # only when a matching execution result is supplied; otherwise they remain
    # explicitly planned.  This prevents dry-run/spec generation from becoming
    # a false green in the verification matrix.
    l1_by_req = _result_map(l1_results)
    l2_by_req = _result_map(l2_results)
    for spec in requirement_evidence.test_specs:
        rid = spec.req_id
        if rid not in tiers:
            continue
        if spec.tier == "L2":
            strength = _l2_strength(spec.inject.kind, spec.verify.kind)
            outcome = l2_by_req.get(_norm_req_id(rid))
            tier = "l2_sitl" if outcome is True else (
                "l2_sitl_failed" if outcome is False else "l2_sitl_planned"
            )
            tiers[rid].add(tier)
            state = "PASS" if outcome is True else "FAIL" if outcome is False else "planned, not executed"
            evidence[rid].append(
                f"L2 {spec.inject.kind}→{spec.verify.kind} ({strength}; {state})"
            )
            claims[rid].append(_sim_claim(
                description=evidence[rid][-1],
                status=(
                    "verified" if outcome is True
                    else "failed" if outcome is False
                    else "planned"
                ),
                kinds=frozenset(
                    _L2_CLOSES_KINDS.get(spec.verify.kind, {"behavior"})
                ),
            ))
        elif spec.tier == "L1":
            names = ", ".join(p.param_name for p in spec.params) or "params"
            outcome = l1_by_req.get(_norm_req_id(rid))
            tier = "l1_param" if outcome is True else (
                "l1_param_failed" if outcome is False else "l1_param_planned"
            )
            tiers[rid].add(tier)
            state = "PASS" if outcome is True else "FAIL" if outcome is False else "planned, not executed"
            # A spec whose every parameter is a static harness constant (e.g. a
            # port-matched protocol/config entry: SERIAL0_PROTOCOL, GPS_INJECT_TO)
            # carries no model-derived number. It is real config-level evidence
            # for the tier, but it must not close the requirement's obligations —
            # an encrypted-link requirement is not "verified" by one protocol
            # param. Model-derived L1 rows (threshold/attr readback) keep the
            # full-closure claim.
            # Unknown provenance counts as model-derived (only an affirmative
            # all-"static" param set downgrades to a config-level claim).
            model_derived = any(
                getattr(p, "source", "") != "static" for p in spec.params
            )
            suffix = "" if model_derived else " (config-level; obligations not closed)"
            evidence[rid].append(f"L1 param consistency ({state}): {names}{suffix}")
            claims[rid].append(_sim_claim(
                description=evidence[rid][-1],
                status=(
                    "verified" if outcome is True
                    else "failed" if outcome is False
                    else "planned"
                ),
                all_obligations=model_derived,
            ))
        elif spec.tier == "TRACE":
            blocked.add(rid)
            evidence[rid].append(f"TRACE blocked: {spec.notes}")

    # 2. Phase 8 per-requirement scopes → datasheet / forward_flight.
    for v in (realization or {}).get("per_requirement", []) or []:
        rid = str(v.get("req_id", "")).replace("-", "_")
        if rid not in tiers:
            continue
        scope = v.get("scope")
        if scope == "closure":
            if v.get("met") is True:
                tiers[rid].add("datasheet")
            elif v.get("met") is False:
                tiers[rid].add("datasheet_failed")
            extra = f" — {v.get('note')}" if v.get("note") else ""
            evidence[rid].append(
                f"datasheet closure: {v.get('family')} realized={v.get('realized_value')} "
                f"target={v.get('target')} met={v.get('met')}{extra}")
            family = str(v.get("family") or "").lower()
            family_kinds = {
                "time": {"behavior", "endurance"},
                "mass": {"behavior", "mass"},
                "payload": {"behavior", "payload", "hover_throttle_margin"},
            }.get(family, {"behavior", family})
            claims[rid].append(_sim_claim(
                description=evidence[rid][-1],
                status=(
                    "verified" if v.get("met") is True
                    else "failed" if v.get("met") is False
                    else "partial"
                ),
                kinds=frozenset(family_kinds),
            ))
        elif scope == "forward_flight":
            if v.get("met") is True:
                tiers[rid].add("forward_flight")
            elif v.get("met") is False:
                tiers[rid].add("forward_flight_failed")
            evidence[rid].append(
                f"forward-flight (lumped): {v.get('family')} realized={v.get('realized_value')} "
                f"met={v.get('met')}")
            family = str(v.get("family") or "").lower()
            claims[rid].append(_sim_claim(
                description=evidence[rid][-1],
                status=(
                    "verified" if v.get("met") is True
                    else "failed" if v.get("met") is False
                    else "partial"
                ),
                kinds=frozenset({"behavior", family}),
            ))
        elif scope == "deferred":
            evidence[rid].append(f"Phase 8 deferred: {v.get('note') or v.get('family')}")

    # 3. Behavioral-sim tier. Presence of a state machine is not evidence: the
    #    exact simulator scenario must pass. This prevents declaration-only
    #    anchors from turning an UNASSIGNED row into a false green.
    try:
        state_machines = extract_state_machines(model_text)
        behavioral = run_behavioral_simulation(model_text)
    except Exception:
        state_machines = []
        behavioral = None
    results_by_machine: Dict[str, List[bool]] = {}
    if behavioral is not None:
        known_names = {sm.name for sm in state_machines}
        for result in behavioral.scenario_results:
            if result.state_machine in known_names:
                results_by_machine.setdefault(result.state_machine, []).append(
                    bool(result.passed)
                )
    machines_by_owner: Dict[str, list] = {}
    for sm in state_machines:
        machines_by_owner.setdefault(sm.owner_part, []).append(sm)
    # The plan's recorded intents ride on the committed model's metadata; a
    # model rebuilt from its text (as the closure audit does) carries none, so
    # a caller that holds the plan passes the intents in.
    if planned_intents is None:
        planned_intents = planned_intents_from_model(model)
    if planned_markers is None:
        planned_markers = planned_markers_from_model(model)
    functional_status = {
        _norm_req_id(rid): status
        for rid, status in functional_behavior_status(
            model_text,
            [
                f"{rid.replace('_', '-')}: {req_texts.get(rid, '')}"
                for rid in universe
            ],
            planned_intents=planned_intents,
            planned_markers=planned_markers,
        ).items()
    }
    # A functional requirement the generation plan recorded as obliging no
    # discrete response is noted as such. This is evidence about the plan's
    # decision, not about the design; the row's status is unchanged here and
    # the terminal closure audit decides, together with the extractor's own
    # "no measurable criterion" flag, whether such a row is a model gap or a
    # requirement that offers nothing to anchor to. An "unverifiable" record
    # is the opposite honesty case — a response IS obliged but the gate has
    # no reachable-action marker that can evidence it — and gets its own
    # tier, so the uncovered obligation stays visible in the matrix instead
    # of masquerading as "no response obliged".
    for rid in universe:
        recorded_intent = planned_intents.get(_norm_req_id(rid))
        if recorded_intent == "none":
            tiers[rid].add("planned_no_response")
            evidence[rid].append(
                "generation plan records response_intent=none for this "
                "requirement (no discrete response obliged)"
            )
        elif recorded_intent == "unverifiable":
            tiers[rid].add("planned_unverifiable_response")
            evidence[rid].append(
                "generation plan records response_intent=unverifiable: a "
                "discrete response is obliged but no reachable-action "
                "marker can evidence it (gate-capability gap, not a model "
                "gap; the obligation remains open for external review)"
            )

    for rid in universe:
        # A TRACE-blocked requirement's guard assignment is the WRONG-family guard
        # the gate rejected — it must not earn a behavioral-sim tier from it.
        assigned = None if rid in blocked else guard_assignment.get(rid)
        # An accept-event pseudo-guard exists to route the SITL L2 spec; it is
        # NOT a behavioral-sim anchor. Falling through keeps the honest
        # initialization/functional routing: "default to locked upon power-on"
        # must be an initial-state invariant, not event reachability.
        if assigned is not None and getattr(assigned, "kind", "") == "accept_event":
            assigned = None
        low = f"{rid} {req_texts.get(rid, '')}".lower()
        if assigned:
            attr = assigned.attribute
            signature = assigned.signature
            matched_names = [
                sm.name
                for sm in machines_by_owner.get(assigned.part_name, [])
                if any(
                    _guard_signature(sm_guard) == signature
                    for transition in sm.transitions
                    for sm_guard in transition.guards
                )
            ]
            outcomes = [
                outcome
                for name in matched_names
                for outcome in results_by_machine.get(name, [])
            ]
            compiled = compile_verification_obligations(
                rid, req_texts.get(rid, "")
            )
            quantitative_kinds = {
                obligation.kind for obligation in compiled
                if obligation.kind != "behavior"
            }
            _record_behavioral_outcome(
                rid, outcomes, tiers, evidence, claims,
                f"model guard '{attr}' exercised at behavioral-sim tier",
                claim_kinds={
                    "behavior",
                    *(quantitative_kinds if len(quantitative_kinds) == 1 else ()),
                },
            )
            continue

        owners = satisfy_map.get(rid, [])
        owner_machines = [
            sm for owner in owners for sm in machines_by_owner.get(owner, [])
        ]

        # Default/initial-state requirements need initialization semantics, not
        # a fabricated fault transition. Select machines whose initial-state
        # name is actually mentioned by the requirement (e.g. Locked).
        #
        # Not, however, when the plan recorded a discrete response intent for
        # this requirement: "execute a power-on self-check" obliges a self-test
        # response and is anchored by the state that produces it, not by an
        # initial-state invariant. Without this guard the substring match below
        # bound such a requirement to whichever owner machine happened to start
        # in a state named PowerOn -- the flight-phase manager -- and reported
        # that machine's initialisation as the requirement's evidence.
        planned_intent = planned_intents.get(_norm_req_id(rid))
        obliges_response = bool(planned_intent) and planned_intent != "none"
        # An inhibition requirement is anchored by the response that is
        # withheld, never by initial-state semantics; one measured run routed
        # such a requirement here because its text contains "power-on"
        # (naming the phase, not a default state) and then failed a model
        # whose inhibition anchor existed and whose scenarios all passed.
        if (
            not obliges_response
            and not is_inhibition_requirement(low)
            and any(k in low for k in _INITIALIZATION_KWS)
        ):
            compact_low = "".join(ch for ch in low if ch.isalnum())
            init_candidates = [
                sm for sm in owner_machines
                if sm.initial_state
                and "".join(
                    ch for ch in sm.initial_state.lower() if ch.isalnum()
                ) in compact_low
            ]
            init_outcomes = [
                run_initialization_scenario(sm).passed for sm in init_candidates
            ]
            if _record_behavioral_outcome(
                rid, init_outcomes, tiers, evidence, claims,
                "initial/default-state invariant exercised at behavioral-sim tier",
            ):
                continue

        functional_outcome = functional_status.get(rid)
        if functional_outcome == BEHAVIOR_ABSENT:
            # A declared action or phase name is not evidence unless a reachable
            # state actually produces the functional response.
            evidence[rid].append(
                "functional response action is not produced by any reachable state"
            )
            continue
        if (functional_outcome == BEHAVIORALLY_VERIFIED
                or any(k in low for k in _BEHAVIORAL_TEXT_KWS)):
            outcomes = [
                outcome
                for sm in owner_machines
                for outcome in results_by_machine.get(sm.name, [])
            ]
            _record_behavioral_outcome(
                rid, outcomes, tiers, evidence, claims,
                "requirement-linked state-machine scenario exercised at behavioral-sim tier",
            )

    # 4/5. Keyword rules — inspection/analysis and Gazebo-planned. Only applied when
    #    the requirement has doc text (no text → nothing to judge by).
    for rid in universe:
        text = req_texts.get(rid, "")
        if not text.strip():
            continue
        low = f"{rid} {text}".lower()
        if any(k in low for k in _INSPECTION_KWS):
            tiers[rid].add("inspection_analysis")
            evidence[rid].append("inspection/analysis item (compliance/environment/materials)")
            # Scoped to the clauses that actually carry the untestable terms.
            # A blanket claim used to stamp out-of-sim-scope over every clause
            # of a compound requirement, so a single "encrypted" put MAVLink
            # protocol conformance — which SITL does test — out of scope too.
            claims[rid].append(EvidenceClaim(
                description=evidence[rid][-1], status="out-of-sim-scope",
                all_obligations=True,
                clause_terms=frozenset(_INSPECTION_KWS),
            ))
        if (any(k in low for k in _GAZEBO_KWS)
                or _positional_release(low) or _timing_actuation(low)):
            tiers[rid].add("gazebo_deferred")
            evidence[rid].append("needs Gazebo-tier physics (S8 boundary) — planned")
            claims[rid].append(_sim_claim(
                description=evidence[rid][-1], status="planned", all_obligations=True,
            ))

    # 6. Optional Gazebo high-fidelity results. A PASS upgrades only the exact
    #    requirement the Gazebo runner names; other Gazebo-planned requirements
    #    remain planned/partial so one high-fidelity result cannot greenwash an
    #    entire mixed-scope requirement.
    for item in (gazebo or {}).get("req_results", []) or []:
        rid = _norm_req_id(item.get("req_id"))
        if rid not in tiers:
            continue
        status = str(item.get("status", "")).upper()
        check = item.get("check") or item.get("name") or "Gazebo"
        message = item.get("message") or item.get("evidence") or ""
        suffix = f": {message}" if message else ""
        if status == "PASS":
            tiers[rid].discard("gazebo_deferred")
            tiers[rid].add("gazebo")
            evidence[rid].append(f"Gazebo PASS ({check}){suffix}")
            claims[rid].append(_sim_claim(
                description=evidence[rid][-1], status="verified", all_obligations=True,
            ))
        elif status == "FAIL":
            tiers[rid].add("gazebo_failed")
            evidence[rid].append(f"Gazebo FAIL ({check}){suffix}")
            claims[rid].append(_sim_claim(
                description=evidence[rid][-1], status="failed", all_obligations=True,
            ))
        elif status == "PARTIAL":
            tiers[rid].add("gazebo_deferred")
            tiers[rid].add("gazebo_partial")
            evidence[rid].append(f"Gazebo partial ({check}){suffix}")
            claims[rid].append(_sim_claim(
                description=evidence[rid][-1], status="partial", all_obligations=True,
            ))
        elif status in {"INCONCLUSIVE", "PLANNED", "SKIPPED", "SUSPENDED"}:
            tiers[rid].add("gazebo_deferred")
            evidence[rid].append(f"Gazebo {status.lower()} ({check}){suffix}")
            claims[rid].append(_sim_claim(
                description=evidence[rid][-1], status="planned", all_obligations=True,
            ))

    rows: List[MatrixRow] = []
    for rid in universe:
        t = tuple(sorted(tiers[rid]))
        has_verified = bool(set(t) & _VERIFIED_TIERS)
        if rid in blocked:
            status = "blocked"
        elif set(t) & _FAILED_TIERS:
            status = "failed"
        elif "gazebo_partial" in t:
            status = "partial"
        elif set(t) & _PLANNED_TIERS:
            status = "partial" if has_verified else "planned"
        elif has_verified:
            status = "verified"
        elif "gazebo_deferred" in t:
            status = "planned"
        elif "inspection_analysis" in t:
            status = "out-of-sim-scope"
        else:
            status = "unassigned"
        obligations = evaluate_obligations(
            compile_verification_obligations(rid, req_texts.get(rid, "")),
            claims[rid],
            blocked=rid in blocked,
        )
        if status == "verified" and not all_obligations_verified(obligations):
            status = "partial"
        rows.append(MatrixRow(
            req_id=rid,
            text=req_texts.get(rid, ""),
            tiers=t,
            methods=tuple(TIER_METHOD[x] for x in t),
            status=status,
            evidence=tuple(evidence[rid]),
            obligations=obligations,
        ))
    return rows


def summarize(rows: List[MatrixRow]) -> Dict[str, object]:
    by_status: Dict[str, int] = {}
    by_tier: Dict[str, int] = {}
    for r in rows:
        by_status[r.status] = by_status.get(r.status, 0) + 1
        for t in r.tiers:
            by_tier[t] = by_tier.get(t, 0) + 1
    obligations = [item for row in rows for item in row.obligations]
    return {
        "total": len(rows),
        "by_status": dict(sorted(by_status.items())),
        "by_tier": dict(sorted(by_tier.items())),
        "unassigned_req_ids": [r.req_id for r in rows if r.status == "unassigned"],
        "obligations_total": len(obligations),
        "obligations_verified": sum(item.status == "verified" for item in obligations),
    }


def to_markdown(rows: List[MatrixRow]) -> str:
    s = summarize(rows)
    lines = [
        "# Verification Strategy Matrix",
        "",
        "Every requirement is explicitly assigned to one or more verification tiers",
        "(IADT methods). \"Unassigned\" is the true honest gap.",
        "",
        f"- Total requirements: {s['total']}",
        "- By status: " + ", ".join(f"{k}={v}" for k, v in s["by_status"].items()),
        "- By tier: " + ", ".join(f"{k}={v}" for k, v in s["by_tier"].items()),
        f"- Verified obligations: {s['obligations_verified']}/{s['obligations_total']}",
        "",
        "| Requirement | Status | Obligation coverage | Tiers | Evidence |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        ev = "; ".join(r.evidence[:2]) or "—"
        verified = sum(item.status == "verified" for item in r.obligations)
        coverage = f"{verified}/{len(r.obligations)}"
        lines.append(
            f"| {r.req_id} | {r.status} | {coverage} | "
            f"{', '.join(r.tiers) or '—'} | {ev} |"
        )
    if s["unassigned_req_ids"]:
        lines += ["", "## Unassigned (honest gap)"]
        for r in rows:
            if r.status == "unassigned":
                lines.append(f"- **{r.req_id}**: {r.text[:120]}")
    return "\n".join(lines) + "\n"


def to_json(rows: List[MatrixRow]) -> Dict[str, object]:
    return {
        "summary": summarize(rows),
        "rows": [
            {
                "req_id": r.req_id,
                "text": r.text,
                "tiers": list(r.tiers),
                "methods": list(r.methods),
                "status": r.status,
                "evidence": list(r.evidence),
                "obligations": [
                    {
                        "obligation_id": item.obligation_id,
                        "clause": item.clause,
                        "kind": item.kind,
                        "status": item.status,
                        "evidence": list(item.evidence),
                    }
                    for item in r.obligations
                ],
            }
            for r in rows
        ],
    }
