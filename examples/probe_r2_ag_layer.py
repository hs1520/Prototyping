"""Exercise the R2 A/G layer alone, against a base model an earlier run committed.

DIAGNOSTIC ONLY — this writes no manifest, freezes nothing, and binds no digests,
so nothing it prints is experiment evidence. Claims still come from
``run_revised_experiment.py``, which is a descriptive pilot and therefore requires
three distinct seeds by protocol.

The point is cost. A pilot spends most of its budget on design generation and
refinement, which is not what an A/G change touches: what changes is the decision
prompt, ``ag_decision``, ``ag_emitter`` and the checker. This runs exactly that
path —

    decision prompt -> LLM decisions -> ag_decision -> ag_emitter -> check_ag_graph

— at one LLM call per selected chain, so a defect can be found and fixed in a
minute instead of an hour, and the expensive multi-seed run happens once, at the
end, when the chains are already stable.

    .venv/bin/python examples/probe_r2_ag_layer.py \
        --base examples/output/<run>/seed-0/R1-BBCTX/shared_model_final.sysml \
        --provider vertex --model gemini-3.1-pro-preview \
        --confirm-external-call
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from run_revised_experiment import FROZEN_REQUIREMENTS  # noqa: E402
from src.agents.orchestrator import Orchestrator  # noqa: E402
from src.prototyping.ag_contracts import check_ag_graph  # noqa: E402
from src.prototyping.ag_extractor import extract_ag_graphs  # noqa: E402
from src.prototyping.provider_factory import create_llm  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", required=True,
        help="a committed base model (an arm's shared_model_final.sysml with no "
             "A/G layer, e.g. R1-BBCTX's)",
    )
    parser.add_argument("--provider", default="vertex")
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument("--out", default=None, help="write the merged model here")
    parser.add_argument(
        "--confirm-external-call", action="store_true",
        help="confirm this invocation is authorised to call a paid provider",
    )
    args = parser.parse_args()
    if not args.confirm_external_call:
        parser.error("refusing to call a provider without --confirm-external-call")

    load_dotenv()
    base_text = Path(args.base).read_text()
    if "_AG {" in base_text:
        parser.error(
            f"{args.base} already carries an A/G layer; pass a base model "
            "(an R0/R1 arm's committed model)"
        )

    orchestrator = Orchestrator(
        create_llm(provider=args.provider, model=args.model),
        revised_experiment_arm="R2-BBAG",
        r2_generation_mode="LLM_DECIDED_SPEC",
    )
    # the probe is not a frozen run; this only satisfies the arm's input record
    orchestrator.last_requirement_input = {
        "mode": "frozen", "requirement_set_digest": "probe",
    }
    print(f"base model: {args.base} ({len(base_text.splitlines())} lines)")
    merged = orchestrator._apply_ag_contract_layer(
        base_text, list(FROZEN_REQUIREMENTS)
    )

    failures = 0
    for graph in sorted(
        extract_ag_graphs(merged),
        key=lambda item: str(item.system.source_requirement),
    ):
        report = check_ag_graph(graph)
        codes = sorted({item.code for item in report.errors()})
        print(
            f"  {str(graph.system.source_requirement):14s} "
            f"{report.verdict:6s} {codes}"
        )
        failures += report.verdict != "PASS"

    if args.out:
        Path(args.out).write_text(merged)
        print(f"merged model written to {args.out}")
    print("DIAGNOSTIC ONLY — not experiment evidence")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
