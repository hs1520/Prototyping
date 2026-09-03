"""Ablation arm registry - each arm differs from FULL by one component.

An arm entry holds its pipeline configuration, whether it runs the DSE stage,
and whether it is a post-hoc replay (no LLM runs). ``run_ablation.py``
consumes it; ``analyze.py`` and the campaign manifest record its digest, so a
result traces back to the arm definition that produced it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class AblationArm:
    name: str
    ablated_component: str
    question: str
    pipeline_kwargs: Mapping[str, Any] = field(default_factory=dict)
    run_dse: bool = True
    posthoc: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "ablated_component": self.ablated_component,
            "question": self.question,
            "pipeline_kwargs": dict(self.pipeline_kwargs),
            "run_dse": self.run_dse,
            "posthoc": self.posthoc,
        }

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest()


BASELINE_ARM = "FULL"

ARMS: Dict[str, AblationArm] = {
    arm.name: arm
    for arm in (
        AblationArm(
            name="FULL",
            ablated_component="none (baseline — production default pipeline)",
            question="Baseline for every paired comparison; its own failure "
                     "rate quantifies current full-chain robustness.",
        ),
        AblationArm(
            name="NO-DSE",
            ablated_component="bilevel design-space exploration (explore() skipped)",
            question="Does DSE materially improve the final design over the "
                     "generate-only model?",
            run_dse=False,
        ),
        AblationArm(
            name="DSE-BILEVEL",
            ablated_component="LLM-declared variation points → catalog-operator "
                              "bilevel DSE",
            question="How do the two DSE modes compare on the same spec?",
            pipeline_kwargs={"dse_mode": "bilevel"},
        ),
        AblationArm(
            name="NO-REFINE",
            ablated_component="iterative refinement loop (max_iterations 4 → 1)",
            question="What does iterative refinement contribute beyond the "
                     "first evaluated candidate?",
            pipeline_kwargs={"max_iterations": 1},
        ),
        AblationArm(
            name="NO-SURGICAL",
            ablated_component="surgical block-level refinement (whole-model "
                              "rewrite only)",
            question="What do scoped block edits save vs full rewrites in "
                     "quality, preservation, and tokens?",
            pipeline_kwargs={"use_surgical_refinement": False},
        ),
        AblationArm(
            name="NO-DETFIX",
            ablated_component="deterministic fixer chain (Tier-0 syntax fixes; "
                              "connectivity direction/missing-connect fixes)",
            question="How many LLM calls and failures does the deterministic "
                     "repair layer absorb?",
            pipeline_kwargs={"use_deterministic_fixers": False},
        ),
        AblationArm(
            name="NO-REPAIR",
            ablated_component="combined repair stack: iterative refinement "
                              "(max_iterations 4 -> 1) AND surgical "
                              "block-level refinement (whole-model rewrite "
                              "only; surgical-hosted exit passes inactive, "
                              "closure retains its full-rewrite fallback)",
            question="Is the repair stack redundant, or does stripping both "
                     "the iteration budget and the surgical mechanism break "
                     "the run where single ablations were absorbed?",
            pipeline_kwargs={
                "max_iterations": 1,
                "use_surgical_refinement": False,
            },
        ),
        AblationArm(
            name="SINGLE-SHOT",
            ablated_component="multistep typed-plan generation → legacy "
                              "single-prompt whole-model generation "
                              "(plan-conformance gates inapplicable by "
                              "construction)",
            question="Is structured multistep decomposition necessary, or can "
                     "one prompt produce an admissible model?",
            pipeline_kwargs={"design_generation_mode": "single_shot"},
        ),
        AblationArm(
            name="W-UNIFORM",
            ablated_component="requirement-derived recommendation weights → "
                              "uniform weights (offline replay on FULL's "
                              "saved Pareto fronts)",
            question="Does requirement-traceable weighting actually change "
                     "which design is recommended?",
            posthoc=True,
        ),
    )
}

PAID_ARMS = [arm.name for arm in ARMS.values() if not arm.posthoc]


def registry_manifest() -> Dict[str, Any]:
    """Digest-stamped snapshot of every arm, for the campaign manifest."""
    return {
        name: {**arm.to_dict(), "digest": arm.digest()}
        for name, arm in ARMS.items()
    }
