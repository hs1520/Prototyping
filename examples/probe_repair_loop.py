"""Run the Increment 3 repair loop for real, once, on a single chain.

DIAGNOSTIC ONLY — no manifest, no digest binding, not evidence.

Why it exists: automatic surgical repair is a single-chain capability by design
(§15), and every pilot now selects three chains, so `_build_multichain_ag_trace`
runs with `allow_repair=False` and every authorised model-semantic failure is
routed to a BLOCKED task. The consequence is easy to miss — **the Increment 3 exit
gate ("one authorised model-semantic failure is automatically routed, attempted,
and rechecked against the committed revision") is not exercised by the current
pilot configuration at all.** It was, when REQ_SAFE_005 was the only selected
chain.

So this drives that gate deliberately: take one chain out of a real committed
model, injure it in a way that routes to DEPENDENCY_CLOSED_SURGICAL_REPAIR, and run
the real path — route, envelope, bounded session, provider call, surgical merge,
preservation gates, target-diagnostic removal, re-extraction of the committed
revision. The provider is real; the injury is deliberate, which is why this is a
diagnostic and not an experiment (§13 excludes error injection from the core).

    .venv/bin/python examples/probe_repair_loop.py --confirm-external-call
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.prototyping.ag_assurance import route_failure_diagnostics  # noqa: E402
from src.prototyping.ag_contracts import check_ag_graph  # noqa: E402
from src.prototyping.ag_extractor import extract_ag_graph  # noqa: E402
from src.prototyping.ag_repair import (  # noqa: E402
    attempt_dependency_closed_ag_repair,
)
from src.prototyping.blackboard import Blackboard, RecordType  # noqa: E402
from src.prototyping.context_builder import ContextBuilder  # noqa: E402
from src.prototyping.provider_factory import create_llm  # noqa: E402
from src.prototyping.task_session import TaskSessionRegistry  # noqa: E402

DEFAULT_MODEL = (
    "examples/output/decided_roles_3chain_20260726/seed-0/R2-BBAG/"
    "shared_model_final.sysml"
)


def _single_chain(text: str, requirement: str) -> str:
    """One chain's package plus its authoritative requirement def.

    Repair is a single-chain capability, so a multi-chain model must be reduced
    before the loop is reachable at all.
    """
    package = re.search(
        rf"package {requirement}_AG \{{.*?\n\}}", text, re.S
    )
    if package is None:
        raise SystemExit(f"no {requirement}_AG package in {DEFAULT_MODEL}")
    source = re.search(
        rf"requirement def {requirement} \{{[^}}]*\}}", text
    )
    if source is None:
        raise SystemExit(f"no authoritative {requirement} in the model")
    return (
        f"package Drone {{\n    {source.group(0)}\n}}\n\n{package.group(0)}\n"
    )


def _injure(text: str) -> tuple[str, str]:
    """Remove one responding state's entry action: a model-semantic fault."""
    match = re.search(r"state (\w+) \{ entry action (set\w+); \}", text)
    if match is None:
        raise SystemExit("no entry action to remove")
    return text.replace(match.group(0), f"state {match.group(1)};"), match.group(2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--requirement", default="REQ_SAFE_005")
    parser.add_argument("--provider", default="vertex")
    parser.add_argument("--model-name", default="gemini-3.1-pro-preview")
    parser.add_argument("--confirm-external-call", action="store_true")
    args = parser.parse_args()
    if not args.confirm_external_call:
        parser.error("refusing to call a provider without --confirm-external-call")

    load_dotenv()
    chain = _single_chain(Path(args.model).read_text(), args.requirement)
    injured, removed_action = _injure(chain)
    print(f"single chain: {args.requirement} "
          f"({len(chain.splitlines())} lines); removed `{removed_action}`")

    report = check_ag_graph(extract_ag_graph(injured))
    print(f"verdict after injury: {report.verdict} "
          f"{sorted({d.code for d in report.errors()})}")
    routed = route_failure_diagnostics(
        report.diagnostics,
        source_requirement=report.source_requirement,
        realization_links=report.realization_links,
    )
    authorised = [
        item for item in routed["failures"] if item.get("repair_authorized")
    ]
    print(f"routed failures: {len(routed['failures'])}, "
          f"authorised for surgical repair: {len(authorised)}")
    if not authorised:
        print("nothing authorised — the loop is not reachable for this injury")
        return 1

    board = Blackboard("Drone")
    # the authoritative publication must byte-match the committed `doc` text: the
    # commit gate compares them and refuses otherwise, which it did on the first
    # attempt here — protection working, not an obstacle to route around
    doc = re.search(
        rf"requirement def {args.requirement} \{{[^}}]*?doc /\*(.*?)\*/",
        injured, re.S,
    )
    board.publish(
        RecordType.SOURCE, "requirements.authoritative", "RequirementsAgent",
        {"requirements": [
            f"{args.requirement.replace('_', '-')}: "
            + " ".join((doc.group(1) if doc else "").split())
        ]},
    )
    board.commit_model(
        injured, base_revision=0,
        base_digest=board.current_model.model_digest, producer="probe",
    )
    analysis = board.publish(
        RecordType.ANALYSIS, "analysis.ag_trace", "AGChecker",
        {"diagnostics": [item.as_dict() for item in report.diagnostics]},
    )
    failure_record = board.publish(
        RecordType.ANALYSIS, "diagnostic.failure", "AGFailureRouter",
        authorised[0],
    )

    builder = ContextBuilder(board)
    sessions = TaskSessionRegistry()
    decision = attempt_dependency_closed_ag_repair(
        llm=create_llm(provider=args.provider, model=args.model_name),
        board=board,
        context_builder=builder,
        sessions=sessions,
        failure_record_id=failure_record.record_id,
        analysis_record_id=analysis.record_id,
    )

    # the audit is why a rejection is actionable rather than opaque
    audit = next(
        (item.payload for item in reversed(board.records(topic="repair.decision"))
         if item.payload.get("audit")), {}
    ).get("audit") or {}
    if audit:
        print(f"\naudit               : llm_invoked={audit.get('llm_invoked')}, "
              f"responses={audit.get('response_count')}, "
              f"context={audit.get('context_mode')} "
              f"({audit.get('context_line_count')} lines)")
        for reason in audit.get("rejection_reasons") or ():
            print(f"  rejected because  : {reason}")
    gate = next(
        (item.payload.get("gate") for item in reversed(board.records(topic="repair.decision"))
         if item.payload.get("gate")), None
    )
    if gate:
        print(f"\ngate                : target={gate['target']} "
              f"removed={gate['target_removed']} regression_free={gate['regression_free']} "
              f"pattern={gate['pattern_verdict']}")
        print(f"  new diagnostics   : {gate['new_diagnostics']}")
        print(f"  remaining         : {gate['remaining_diagnostics']}")

    envelopes = builder.snapshot()["envelopes"]
    print(f"\ndecision            : {decision.status} ({decision.reason})")
    print(f"target removed      : {decision.target_diagnostic_removed}")
    print(f"regression free     : {decision.regression_free}")
    print(f"whole-model fallback: {decision.whole_model_fallback_used}")
    print(f"committed revision  : {decision.committed_model_revision} "
          f"(base {decision.base_model_revision})")
    for envelope in envelopes:
        body = envelope.get("model_context") or ""
        print(f"envelope            : {envelope.get('agent_role')}, "
              f"{len(body.splitlines())} lines, "
              f"{len(envelope.get('diagnostic_record_ids') or [])} diagnostics")
    if decision.committed_model_revision:
        after = check_ag_graph(extract_ag_graph(board.current_model.model_text))
        print(f"recheck of committed: {after.verdict} "
              f"{sorted({d.code for d in after.errors()})}")
    if decision.status != "ACCEPTED":
        snapshot = sessions.snapshot()
        for item in snapshot.get("sessions", []):
            for turn in item.get("transcript", []) or []:
                if turn.get("role") == "assistant":
                    print("\n--- what the repair agent actually returned ---")
                    print(str(turn.get("content"))[:900])
    print("DIAGNOSTIC ONLY — not experiment evidence")
    return 0 if decision.status == "ACCEPTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
