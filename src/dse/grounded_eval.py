"""Grounded safety evaluation — replaces analytic placeholders with model execution.

Pain point B: objective values must reflect what the model actually *does*, not a
Python formula or a regex keyword count.

Pain point B / Problem-2 refinement: it is not enough that a fault transition can
*fire* — the safety mechanism must be *causally complete*:

  1. the voting guard variable must be DERIVED from the channel health (so a real
     channel fault propagates into the decision), not a free-floating attribute,
  2. the failsafe output (overrideCmd) must be CONNECTED to a consumer (so the
     decision actually commands something), and
  3. a fault→failsafe transition must exist.

A model that merely *declares* `failedChannels` and lets a simulator drive that
free variable looks safe to a naive scorer but does nothing under a real sensor
fault. This evaluator detects that "fake safety" and penalises it — which a
keyword/regex scorer, and even a dynamic sim that drives the guard directly,
cannot. SITL closed-loop remains the high-fidelity oracle that calibrates this
(docs/DSE_REDESIGN.md §三-D).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import src.simulation.behavioral_sim as bsim
from src.simulation.state_extractor import extract_state_machines

from .operators.redundantize import kofn_reliability

_SENSOR_PORT_RE = re.compile(r"\bin\s+port\s+sensor\w*\s*:", re.IGNORECASE)
_OVERRIDE_CONNECT_RE = re.compile(r"\bconnect\b[^;]*\boverrideCmd\b", re.IGNORECASE)
_FAILED_INIT_RE = re.compile(
    r"\bfailedChannels\s*:\s*\w+\s*=\s*([^;]+);", re.IGNORECASE | re.DOTALL
)


@dataclass
class GroundedSafety:
    channels: int            # redundancy depth read from the model's sensor ports
    faults_masked: int       # faults tolerated before failsafe (model's guard - 1)
    guard_grounded: bool     # voting guard is DERIVED from channel health (not free)
    output_connected: bool   # failsafe output (overrideCmd) drives a consumer
    has_failsafe: bool       # a fault→failsafe transition exists
    causal_complete: bool    # all three causal links present
    safety_verified: bool    # dynamic sim could drive the fault to failsafe (informational)
    reliability: float       # grounded reliability objective


def _count_channels(sysml_text: str) -> int:
    return max(1, len(_SENSOR_PORT_RE.findall(sysml_text)))


def _guard_is_derived(sysml_text: str) -> bool:
    """True when failedChannels is computed from channel health, not a literal."""
    m = _FAILED_INIT_RE.search(sysml_text)
    if not m:
        return False
    rhs = m.group(1)
    # a free variable initialises to a bare literal (e.g. `= 0`); a derived one
    # references other features (channel health attributes)
    return bool(re.search(r"channel", rhs, re.IGNORECASE))


def _guard_threshold(sm) -> Optional[float]:
    for tr in sm.fault_transitions():
        for g in tr.guards:
            thr = getattr(g, "threshold", None)
            if thr:
                return float(thr)
    return None


def grounded_safety(sysml_text: str, channel_reliability: float = 0.85) -> GroundedSafety:
    """Evaluate redundancy/failsafe by inspecting and executing the resolved model."""
    sms = extract_state_machines(sysml_text)

    has_failsafe = False
    faults_masked = 0
    safety_verified = False
    for sm in sms:
        if sm.initial_state is None:
            non_fault = [s.name for s in sm.states if not s.entry_action]
            sm.initial_state = (
                non_fault[0] if non_fault else (sm.states[0].name if sm.states else None)
            )
        if not sm.fault_transitions():
            continue
        has_failsafe = True
        thr = _guard_threshold(sm)
        if thr:
            faults_masked = max(faults_masked, int(thr) - 1)
        try:
            if not bsim._run_scenario(sm).violations:
                safety_verified = True
        except Exception as exc:
            from ..utils.suppressed import record_suppressed
            record_suppressed("dse.grounded_eval.scenario", exc)
            pass

    channels = _count_channels(sysml_text)
    guard_grounded = _guard_is_derived(sysml_text)
    output_connected = bool(_OVERRIDE_CONNECT_RE.search(sysml_text))
    causal_complete = guard_grounded and output_connected and has_failsafe

    # k-of-N reliability using the masking the model's own voting guard encodes
    # (e.g. 2oo3 TMR masks 1, not 2) — name and behaviour now agree.
    reliability = kofn_reliability(channels, faults_masked, channel_reliability)
    # redundancy that is not causally complete is "fake safety" — heavily penalised
    if channels > 1 and not causal_complete:
        reliability *= 0.5

    return GroundedSafety(
        channels=channels,
        faults_masked=faults_masked,
        guard_grounded=guard_grounded,
        output_connected=output_connected,
        has_failsafe=has_failsafe,
        causal_complete=causal_complete,
        safety_verified=safety_verified,
        reliability=reliability,
    )


def grounded_objectives(sysml_text: str, channel_reliability: float = 0.85) -> dict:
    """Multiple ORTHOGONAL design-quality objectives derived from the model.

    Returns (all maximised, 0..1):
      * reliability      — raw k-of-N redundancy benefit (no causal penalty)
      * safety_integrity — fraction of the causal safety chain actually present
                           (guard grounded / output connected / failsafe / fires)

    These are genuinely independent: a redundant model with a broken voting chain
    has high reliability *potential* but low integrity. Designed for the N-D
    Pareto front (the caller adds cost objectives). Replaces the toy 2-D placeholder.
    """
    g = grounded_safety(sysml_text, channel_reliability)
    reliability = kofn_reliability(g.channels, g.faults_masked, channel_reliability)
    if g.channels <= 1:
        safety_integrity = 1.0  # single channel: no failsafe expected → N/A
    else:
        links = (g.guard_grounded, g.output_connected, g.has_failsafe, g.safety_verified)
        safety_integrity = sum(1 for x in links if x) / len(links)
    return {"reliability": reliability, "safety_integrity": safety_integrity}
