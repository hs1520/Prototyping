"""Requirement-verification coverage classifier.

A model can `satisfy` every requirement (allocation/intent) yet verify almost
none: `satisfy` is an allocation relation, and proof comes from verification (an
assert/calc-def that evaluates, a verification case, a behavioral check). This
classifier reports per requirement what evidence exists, so a baseline states its
verification maturity.

Levels (strongest -> weakest):
  flight-verified    : the recommended design was flown in Gazebo (stable hover,
                       real-motor calibrated) AND a real motor+prop datasheet meets
                       the target. Opt-in (RUN_GAZEBO); applies to the
                       endurance/flight requirement.
  analysis-verified  : an injected, Automator-evaluable `assert constraint` checks
                       it (endurance / MTOW / range - the physics axis we model).
  quantitative       : carries a numeric target -> coverable by the DSE->SITL
                       quantitative pipeline (verification case / L1 settable-param
                       range check).
  allocated-only     : has a `satisfy` but no quantitative target and no assert -
                       functional/safety/interface behaviour or protocol
                       conformance, not verified here (generation-side).
  unallocated        : declared but not even allocated (no satisfy).
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List

from .domain_objective import endurance_target, mass_limit, range_requirement
from .requirement_spec import extract_requirements
from .functional_behavior import functional_behavior_status
from .requirement_trace import dse_req_id, extract_requirement_trace
from .safety_behavior import safety_behavior_status

FLIGHT_VERIFIED = "flight-verified"
ANALYSIS_VERIFIED = "analysis-verified"
QUANTITATIVE = "quantitative"
ALLOCATED_ONLY = "allocated-only"
UNALLOCATED = "unallocated"

def classify_requirement_coverage(model_text: str, requirements: List[str],
                                  endurance_req: str = "REQ-PERF-002",
                                  dynamic: bool = False, gazebo: Dict = None) -> Dict[str, str]:
    """{req_id: evidence level} for every requirement declared in the model."""
    trace = extract_requirement_trace(model_text, requirements)
    declared = trace.declared
    satisfied = trace.satisfied

    # flight-verified (strongest): Gazebo flew the design and confirmed (a) endurance
    # (stable hover + real datasheet >= target) and (b) single-motor-failure
    # controllability, the redundancy requirement only physical flight sim can show.
    flight = set()
    if gazebo and gazebo.get("status") == "ok":
        et = endurance_target(requirements)
        if et > 0 and gazebo.get("datasheet_endurance_min", 0) >= et:
            flight.add(dse_req_id(endurance_req))
        if gazebo.get("motor_failure_tolerant") and gazebo.get("redundancy_req"):
            flight.add(dse_req_id(gazebo["redundancy_req"]))

    analysis = set()
    if "enduranceMeetsReq" in model_text and endurance_target(requirements) > 0:
        analysis.add(dse_req_id(endurance_req))
    if "mtowWithinReq" in model_text and mass_limit(requirements)[0]:
        analysis.add(dse_req_id(mass_limit(requirements)[0]))
    if "rangeMeetsReq" in model_text and range_requirement(requirements)[0]:
        analysis.add(dse_req_id(range_requirement(requirements)[0]))
    quant = {dse_req_id(s.req_id) for s in extract_requirements(requirements)}
    # safety + functional requirements get a behavioural status from state-machine
    # reachability (safety: fail-safe reachable; functional: response action
    # reachable), lifting them out of allocated-only. Safety wins when both apply.
    dyn = {}
    if dynamic:
        from .dynamic_behavior import dynamic_fire_by_part
        dyn = dynamic_fire_by_part(model_text)
    safety = safety_behavior_status(model_text, requirements, dynamic_fire=dyn)
    functional = functional_behavior_status(model_text, requirements)

    out: Dict[str, str] = {}
    for rid in declared:
        if rid in flight:
            out[rid] = FLIGHT_VERIFIED
        elif rid in analysis:
            out[rid] = ANALYSIS_VERIFIED
        elif rid in quant:
            out[rid] = QUANTITATIVE
        elif rid in safety:
            out[rid] = safety[rid]
        elif rid in functional:
            out[rid] = functional[rid]
        elif rid in satisfied:
            out[rid] = ALLOCATED_ONLY
        else:
            out[rid] = UNALLOCATED
    return out


def coverage_summary(cov: Dict[str, str]) -> str:
    from .safety_behavior import (BEHAVIOR_ABSENT, BEHAVIORALLY_VERIFIED,
                                  BEHAVIORALLY_VIOLATED, RESPONSE_COLLAPSED)
    c = Counter(cov.values())
    cc = c.get(RESPONSE_COLLAPSED, 0)
    return (f"requirement evidence ({len(cov)} reqs): "
            f"{c.get(FLIGHT_VERIFIED, 0)} flight-verified, "
            f"{c.get(ANALYSIS_VERIFIED, 0)} analysis-verified, "
            f"{c.get(QUANTITATIVE, 0)} quantitative-checkable, "
            f"{c.get(BEHAVIORALLY_VERIFIED, 0)} behaviorally-verified, "
            f"{c.get(BEHAVIORALLY_VIOLATED, 0)} behaviorally-violated, "
            + (f"{cc} response-collapsed, " if cc else "")
            + f"{c.get(BEHAVIOR_ABSENT, 0)} behavior-absent, "
            f"{c.get(ALLOCATED_ONLY, 0)} allocated-only (intent, UNVERIFIED), "
            f"{c.get(UNALLOCATED, 0)} unallocated")
