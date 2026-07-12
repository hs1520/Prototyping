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
    "l2_sitl_planned": "Test (native SITL — planned, not executed)",
    "l2_sitl_failed": "Test (native SITL — failed)",
    "gazebo": "Test (Gazebo high-fidelity FDM)",
    "gazebo_partial": "Test (Gazebo FDM — partial requirement evidence)",
    "gazebo_failed": "Test (Gazebo high-fidelity FDM — failed)",
    "l1_param": "Inspection (config consistency)",
    "l1_param_planned": "Inspection (config consistency — planned, not executed)",
    "l1_param_failed": "Inspection (config consistency — failed)",
    "datasheet": "Analysis (manufacturer datasheet)",
    "forward_flight": "Analysis (lumped momentum model)",
    "behavioral_sim": "Analysis (model-level simulation)",
    "gazebo_deferred": "Test (Gazebo — planned, see SITL_INTEGRATION_DESIGN S8/T9)",
    "inspection_analysis": "Inspection/Analysis (outside simulation scope)",
}

_VERIFIED_TIERS = {"l2_sitl", "gazebo", "l1_param", "datasheet", "forward_flight", "behavioral_sim"}
_FAILED_TIERS = {"l2_sitl_failed", "l1_param_failed", "gazebo_failed"}
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

_BEHAVIORAL_TEXT_KWS = ("phase", "sequence", "sequential", "state", "mode", "transition")


@dataclass(frozen=True)
class MatrixRow:
    req_id: str
    text: str
    tiers: Tuple[str, ...]
    methods: Tuple[str, ...]
    status: str  # verified | partial | planned | failed | out-of-sim-scope | blocked | unassigned
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


def build_matrix(model, realization: Optional[dict], linker,
                 gazebo: Optional[dict] = None,
                 l1_results=None, l2_results=None) -> List[MatrixRow]:
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

    # 1. Linker specs describe verification intent only.  They become verified
    # only when a matching execution result is supplied; otherwise they remain
    # explicitly planned.  This prevents dry-run/spec generation from becoming
    # a false green in the verification matrix.
    l1_by_req = _result_map(l1_results)
    l2_by_req = _result_map(l2_results)
    for spec in linker.generate_test_specs():
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
        elif spec.tier == "L1":
            names = ", ".join(p.param_name for p in spec.params) or "params"
            outcome = l1_by_req.get(_norm_req_id(rid))
            tier = "l1_param" if outcome is True else (
                "l1_param_failed" if outcome is False else "l1_param_planned"
            )
            tiers[rid].add(tier)
            state = "PASS" if outcome is True else "FAIL" if outcome is False else "planned, not executed"
            evidence[rid].append(f"L1 param consistency ({state}): {names}")
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
        elif status == "FAIL":
            tiers[rid].add("gazebo_failed")
            evidence[rid].append(f"Gazebo FAIL ({check}){suffix}")
        elif status == "PARTIAL":
            tiers[rid].add("gazebo_deferred")
            tiers[rid].add("gazebo_partial")
            evidence[rid].append(f"Gazebo partial ({check}){suffix}")
        elif status in {"INCONCLUSIVE", "PLANNED", "SKIPPED", "SUSPENDED"}:
            tiers[rid].add("gazebo_deferred")
            evidence[rid].append(f"Gazebo {status.lower()} ({check}){suffix}")

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
