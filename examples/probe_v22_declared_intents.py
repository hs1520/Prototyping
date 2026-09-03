"""Cross-domain probe of the v22 declared-intent mechanism (one real LLM loop).

v22 (see experiment_arms.COMMON_GENERATION_PIPELINE_VERSION) lets a FUNC
realization declare `response_markers` for an intent outside the built-in
table, and record `unverifiable`/`none` for the two honesty cases. This runs
that contract against a live provider on building-management requirements,
where two of the four obliged responses (alert, unlock) have no built-in
intent. One system, one loop, no comparison.

Archived result (2026-08-28, vertex / gemini-3.1-pro-preview) in
examples/output/probe_v22_declared_intents_20260828/:
  - accepted_plan.json  : the plan that reached status PASS through the
    pipeline's bounded retry loop (TypedPlanGeneration, max 6 attempts).
    Recorded decisions: alert -> declared intent 'alert' markers ['alert'];
    unlock -> declared intent 'unlock' markers ['unlock']; self_test ->
    built-in, markers empty; monitor -> none with rationale. The declared
    responses are realized by reachable states (FaultAlert entry
    alertOperators; FireEmergency entry unlockDoors).
  - attempt1_payload.json : the raw first-attempt payload; its intent fields
    were already correct and the retries repaired unrelated state-declaration
    defects.

Re-running makes new LLM calls and writes to a fresh date-stamped directory.
"""
from __future__ import annotations

import datetime
import json
import pathlib
import sys

from src.prototyping.provider_factory import create_llm
from src.llm.chain_of_thought import ChainOfThoughtPrompter
from src.agents.typed_plan_generation import (
    DEFAULT_MAXIMUM_PLAN_ATTEMPTS,
    TypedModelPlanError,
    TypedPlanGeneration,
    TypedPlanRequest,
)

REQS = [
    "REQ-FUNC-001: The building management system shall alert the on-duty operators within 5 minutes of detecting an equipment fault.",
    "REQ-FUNC-002: The building management system shall unlock all emergency exit doors upon receiving a validated fire-alarm signal.",
    "REQ-FUNC-003: The building management system shall execute an automated self-test prior to entering service.",
    "REQ-FUNC-004: The building management system shall continuously monitor total energy consumption of the building.",
]


def main() -> int:
    stamp = datetime.date.today().strftime("%Y%m%d")
    out_dir = pathlib.Path(
        f"examples/output/probe_v22_declared_intents_{stamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    step = TypedPlanGeneration(
        ChainOfThoughtPrompter(create_llm(provider="vertex")),
        maximum_attempts=DEFAULT_MAXIMUM_PLAN_ATTEMPTS,
    )
    try:
        outcome = step.generate(TypedPlanRequest(
            system_name="SmartBuildingBMS",
            requirements=REQS,
            verbose=True,
        ))
    except TypedModelPlanError as exc:
        (out_dir / "failure.json").write_text(json.dumps({
            "error": str(exc),
            "plan_attempts": exc.plan_attempts,
        }, indent=2))
        print("FAILED CLOSED; attempts archived")
        return 1
    plan = outcome.plan
    (out_dir / "accepted_plan.json").write_text(
        json.dumps(plan.to_dict(), indent=2, ensure_ascii=False)
    )
    print("plan status:", plan.status)
    for r in plan.requirement_realizations:
        print(f"  {r.requirement_id}: intent={r.response_intent!r} "
              f"markers={list(r.response_markers)!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
