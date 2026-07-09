"""Verification strategy matrix: every requirement gets an explicit tier + method.

"SITL unmapped" is not "unverified": endurance closes at the datasheet tier,
phase sequencing is the behavioral-sim tier's job, and IP54/regulatory items are
inspection/analysis work outside any simulation toolchain. This module derives —
from artifacts that already exist (linker specs, Phase 8 per-requirement scopes,
model state machines, deterministic keyword rules) — a per-requirement assignment,
so the honest gap ("unassigned") is explicit and small instead of an undifferentiated
"20/36 unmapped" bucket.

Method vocabulary follows the systems-engineering IADT convention
(Inspection / Analysis / Demonstration / Test).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Canonical tiers, ordered from executable test downwards. A requirement may hold
# several (e.g. max airspeed: L1 param + forward-flight analysis).
TIER_METHOD: Dict[str, str] = {
    "l2_sitl": "Test (native SITL)",
    "l1_param": "Inspection (config consistency)",
    "datasheet": "Analysis (manufacturer datasheet)",
    "forward_flight": "Analysis (lumped momentum model)",
    "behavioral_sim": "Analysis (model-level simulation)",
    "gazebo_deferred": "Test (Gazebo — planned, see SITL_INTEGRATION_DESIGN S8/T9)",
    "inspection_analysis": "Inspection/Analysis (outside simulation scope)",
}

_VERIFIED_TIERS = {"l2_sitl", "l1_param", "datasheet", "forward_flight", "behavioral_sim"}

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

_BEHAVIORAL_TEXT_KWS = ("phase", "sequence", "sequential", "state", "mode", "transition")


@dataclass(frozen=True)
class MatrixRow:
    req_id: str
    text: str
    tiers: Tuple[str, ...]
    methods: Tuple[str, ...]
    status: str  # verified | planned | out-of-sim-scope | blocked | unassigned
    evidence: Tuple[str, ...] = field(default_factory=tuple)


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


def build_matrix(model, realization: Optional[dict], linker) -> List[MatrixRow]:
    """Derive the per-requirement verification assignment from existing artifacts.

    ``realization`` is the Phase 8 dict (``realization_run.json``'s "realization"
    value) or None when the latest run produced no recommendation.
    """
    from src.simulation.state_extractor import extract_state_machines

    req_texts: Dict[str, str] = dict(getattr(linker, "_req_texts", {}) or {})
    satisfy_map: Dict[str, List[str]] = dict(getattr(linker, "_satisfy_map", {}) or {})
    guard_assignment = dict(getattr(linker, "_guard_assignment", {}) or {})
    universe = sorted(set(req_texts) | set(satisfy_map))

    tiers: Dict[str, set] = {r: set() for r in universe}
    evidence: Dict[str, List[str]] = {r: [] for r in universe}
    blocked: set = set()

    # 1. Linker specs → L1 / L2 / TRACE.
    for spec in linker.generate_test_specs():
        rid = spec.req_id
        if rid not in tiers:
            continue
        if spec.tier == "L2":
            strength = _l2_strength(spec.inject.kind, spec.verify.kind)
            tiers[rid].add("l2_sitl")
            evidence[rid].append(f"L2 {spec.inject.kind}→{spec.verify.kind} ({strength})")
        elif spec.tier == "L1":
            names = ", ".join(p.param_name for p in spec.params) or "params"
            tiers[rid].add("l1_param")
            evidence[rid].append(f"L1 param consistency: {names}")
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
            tiers[rid].add("datasheet")
            evidence[rid].append(
                f"datasheet closure: {v.get('family')} realized={v.get('realized_value')} "
                f"target={v.get('target')} met={v.get('met')}")
        elif scope == "forward_flight":
            tiers[rid].add("forward_flight")
            evidence[rid].append(
                f"forward-flight (lumped): {v.get('family')} realized={v.get('realized_value')} "
                f"met={v.get('met')}")
        elif scope == "deferred":
            evidence[rid].append(f"Phase 8 deferred: {v.get('note') or v.get('family')}")

    # 3. Behavioral-sim tier: the model declares the trigger chain (guard) or the
    #    satisfying part owns a state machine that the text is about.
    try:
        sm_owners = {sm.owner_part for sm in extract_state_machines(model.to_sysml_text() or "")}
    except Exception:
        sm_owners = set()
    for rid in universe:
        # A TRACE-blocked requirement's guard assignment is the WRONG-family guard
        # the gate rejected — it must not earn a behavioral-sim tier from it.
        assigned = None if rid in blocked else guard_assignment.get(rid)
        low = f"{rid} {req_texts.get(rid, '')}".lower()
        if assigned:
            g = assigned.get("guard")
            attr = getattr(g, "attribute", "?")
            tiers[rid].add("behavioral_sim")
            evidence[rid].append(f"model guard '{attr}' exercised at behavioral-sim tier")
        elif any(p in sm_owners for p in satisfy_map.get(rid, [])) and any(
                k in low for k in _BEHAVIORAL_TEXT_KWS):
            tiers[rid].add("behavioral_sim")
            evidence[rid].append("satisfying part's state machine exercised at behavioral-sim tier")

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
        if (any(k in low for k in _GAZEBO_KWS)
                or _positional_release(low) or _timing_actuation(low)):
            tiers[rid].add("gazebo_deferred")
            evidence[rid].append("needs Gazebo-tier physics (S8 boundary) — planned")

    rows: List[MatrixRow] = []
    for rid in universe:
        t = tuple(sorted(tiers[rid]))
        if set(t) & _VERIFIED_TIERS:
            status = "verified"
        elif rid in blocked:
            status = "blocked"
        elif "gazebo_deferred" in t:
            status = "planned"
        elif "inspection_analysis" in t:
            status = "out-of-sim-scope"
        else:
            status = "unassigned"
        rows.append(MatrixRow(
            req_id=rid,
            text=req_texts.get(rid, ""),
            tiers=t,
            methods=tuple(TIER_METHOD[x] for x in t),
            status=status,
            evidence=tuple(evidence[rid]),
        ))
    return rows


def summarize(rows: List[MatrixRow]) -> Dict[str, object]:
    by_status: Dict[str, int] = {}
    by_tier: Dict[str, int] = {}
    for r in rows:
        by_status[r.status] = by_status.get(r.status, 0) + 1
        for t in r.tiers:
            by_tier[t] = by_tier.get(t, 0) + 1
    return {
        "total": len(rows),
        "by_status": dict(sorted(by_status.items())),
        "by_tier": dict(sorted(by_tier.items())),
        "unassigned_req_ids": [r.req_id for r in rows if r.status == "unassigned"],
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
        f"- By status: " + ", ".join(f"{k}={v}" for k, v in s["by_status"].items()),
        f"- By tier: " + ", ".join(f"{k}={v}" for k, v in s["by_tier"].items()),
        "",
        "| Requirement | Status | Tiers | Evidence |",
        "|---|---|---|---|",
    ]
    for r in rows:
        ev = "; ".join(r.evidence[:2]) or "—"
        lines.append(f"| {r.req_id} | {r.status} | {', '.join(r.tiers) or '—'} | {ev} |")
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
            }
            for r in rows
        ],
    }
