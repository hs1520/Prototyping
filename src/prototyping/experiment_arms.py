"""Unambiguous experiment namespaces for the legacy and revised Option 2 work.

The implemented external-contract pilot historically used ``B0/B1/B2``.  The
revised Blackboard/A-G study deliberately uses ``R*`` labels so archived runs
from the two designs cannot be pooled by name.
"""
from __future__ import annotations

from enum import Enum


LEGACY_EXPERIMENT_NAMESPACE = "LEGACY_EXTERNAL_CONTRACT_V1"
REVISED_EXPERIMENT_NAMESPACE = "BLACKBOARD_AG_V1"
# v10 ran the five generation steps as one shared conversation. Measured on
# three paired seeds it took qualification from 3/3 to 0/3 at 2.1x the prompt
# cost, failing a different check each seed, and was reverted. v11 restores v9's
# independent single-turn generation calls; what it adds over v9 is board
# instrumentation only (an A/G-planning session, archived generation drafts,
# per-turn token accounting), none of which changes what the model is asked to
# produce. The three archived v10 runs are a negative result and must not be
# pooled with either neighbour.
# v12 fixes the attribute materialiser, which could not see a
# unit-suffixed type (`: LengthValue [m]`) and appended a duplicate
# declaration instead of normalising the existing one — the sole cause of
# the one NOT_QUALIFIED run in pilot_v11b_20260731. Generation prompts are
# unchanged; the terminal model is not, so v11 and v12 are not poolable.
# v13 puts A/G Boolean concepts under the typed plan. They are typed by the
# contract but were never planned attributes, so nothing owned their declared
# type and generation was free to write `attribute airborne : Real = 0.0;` —
# which it did in two measured runs, leaving the terminal gate as the first
# thing to see it and able only to fail the run. The gate stays fail-closed;
# what changes is that the existing planned-attribute materialiser now
# enforces the type during generation. v12 and v13 are not poolable.
# v14 corrects v13: it planned every Boolean A/G concept as an attribute,
# including the ones a component guarantees. Those are outputs, generation
# realises them as directed ports, and one measured run then had
# recoveryActuationPowerAvailable as both a port and an attribute. Only
# consumed concepts are planned now. v13 and v14 are not poolable.
# v15 closes the second writer of the same attributes. Two independent
# materialisers write the semantic-anchor attributes, and both decided
# "already present?" with a pattern blind to a unit-suffixed type; v12
# fixed one, and the duplicate came back through the other. The semantic
# binder now also sees a declaration with no initializer. v14 and v15 are
# not poolable.
# v16 also lets the plan own a planned port's TYPE. "Does this port exist?"
# looked only at the name, so a port written with the wrong type was neither
# added nor corrected, and conformance reported it as both a missing planned
# port and an unplanned one — four such pairs failed one measured run. A
# wrong DIRECTION is still reported rather than rewritten: that changes what
# the connections mean. v15 and v16 are not poolable.
# v17 rejects a planned attribute whose value type the pipeline cannot emit.
# One baseline run committed `lockState : StateEnum = Locked` and failed
# Syside on a type that is declared nowhere; the plan has no way to declare
# one, and validation had only checked the name was a well-formed
# identifier. The plan now fails closed, which puts the retry inside the
# bounded attempt budget. This changes generation, so v16 and v17 are not
# poolable.
# v18 rewrites C-style boolean negation deterministically in the syntax gate.
# `if !sensorFailure` is a parser error — SysML v2 spells negation `not` — and
# in pilot_n6_20260802/seed-3/R0-CURRENT it was the committed model's only
# error, which cost that run its qualification. One occurrence in eight
# archived pilots, and the A/G emitter already wrote `not`, so the convention
# was known to the system and simply never enforced on provider output. A model
# the gate previously rejected can now be repaired, so v17 and v18 are not
# poolable. `!=` is untouched: it is a legal inequality the emitter produces.
COMMON_GENERATION_PIPELINE_VERSION = "typed-whole-model-plan-v18"
R2_DETERMINISTIC_GENERATION_MODE = "DETERMINISTIC_SPEC_EMITTER"
R2_DETERMINISTIC_INTERVENTION_VERSION = (
    "r2-bbag-whole-model-guided-deterministic-v3"
)
# LLM-authored A/G is a SEPARATE intervention (design §15): the LLM authors the
# bounded A/G decomposition instead of the deterministic emitter. Version 7
# freezes that decomposition before ordinary architecture/behavior generation;
# its response vocabulary is therefore a design input implemented downstream,
# rather than being inferred post-hoc from already-generated arbiter behavior.
# It retains the fixed authoring and post-commit repair budgets.
# The post-commit repair budget remains fixed and independently audited.
# It must never be pooled with authored-v1/v2 or the deterministic mode.
# v9 changes the retry seam, not the budgets: the structured-decision turns are a
# real multi-turn conversation, so a rejected decision object comes back as the
# model's own assistant turn instead of being paraphrased into a fresh prompt.
# That changes what the model sees on every retry, so v8 and v9 are not pooled.
R2_LLM_AUTHORED_GENERATION_MODE = "LLM_AUTHORED_AG"
R2_LLM_AUTHORED_INTERVENTION_VERSION = (
    "r2-bbag-whole-model-guided-authored-v10"
)

# LLM-decided specs are a THIRD frozen intervention. The LLM emits the engineering
# decisions — pattern, timing origin and apportionment, discharge wiring, response
# ordering — and the deterministic emitter renders the SysML from them. Measurement
# motivated it: the LLM-authored mode agreed with gold on allocation 6/6 and
# discharge 4/6 while never once reaching a PASS, because it kept losing rounds to
# the notation rather than to the engineering. Here conformance holds by
# construction and only the decisions are judged. Separate frozen config; never
# pooled with either other mode.
# v6 makes the bounded validation retry a real multi-turn conversation: the
# rejected decision object is resent as the model's own assistant turn, so
# "keep everything that was already valid" refers to something it can read. The
# 9/9 evidence was produced under v5, where every retry was a fresh single-turn
# prompt carrying only a paraphrase of the rejection. v5 and v6 must never be
# pooled, and a v6 claim requires its own runs.
R2_LLM_DECIDED_GENERATION_MODE = "LLM_DECIDED_SPEC"
R2_LLM_DECIDED_INTERVENTION_VERSION = (
    "r2-bbag-whole-model-guided-decided-v7"
)

R2_GENERATION_MODES = (
    R2_DETERMINISTIC_GENERATION_MODE,
    R2_LLM_AUTHORED_GENERATION_MODE,
    R2_LLM_DECIDED_GENERATION_MODE,
)
# The one accepted (mode, version) pair per intervention; nothing else may run.
R2_INTERVENTION_VERSION_BY_MODE = {
    R2_DETERMINISTIC_GENERATION_MODE: R2_DETERMINISTIC_INTERVENTION_VERSION,
    R2_LLM_AUTHORED_GENERATION_MODE: R2_LLM_AUTHORED_INTERVENTION_VERSION,
    R2_LLM_DECIDED_GENERATION_MODE: R2_LLM_DECIDED_INTERVENTION_VERSION,
}


class RevisedExperimentArm(str, Enum):
    """Arms reserved for the supervisor-directed redesign."""

    CURRENT = "R0-CURRENT"
    BLACKBOARD_CONTEXT = "R1-BBCTX"
    SEMANTIC_ASSURANCE = "R2-BBAG"
    LONG_SESSION_DIAGNOSTIC = "R1-LONG"

    @classmethod
    def parse(cls, value: "RevisedExperimentArm | str") -> "RevisedExperimentArm":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().upper().replace("_", "-")
        aliases = {
            "R0": cls.CURRENT,
            "CURRENT": cls.CURRENT,
            "R1": cls.BLACKBOARD_CONTEXT,
            "BB-CTX": cls.BLACKBOARD_CONTEXT,
            "R2": cls.SEMANTIC_ASSURANCE,
            "BB-AG": cls.SEMANTIC_ASSURANCE,
            "R1-LONG": cls.LONG_SESSION_DIAGNOSTIC,
        }
        if normalized in aliases:
            return aliases[normalized]
        for arm in cls:
            if normalized == arm.value:
                return arm
        raise ValueError(f"unknown revised experiment arm: {value!r}")

    @property
    def implemented(self) -> bool:
        """The arm's intervention is wired and runnable end-to-end.

        R2-BBAG became runnable once the SysML A/G profile, extractor, and bounded
        compositional checker landed (Increment 2). Runnability is distinct from
        :attr:`evaluation_ready`.
        """
        return self in {
            self.CURRENT, self.BLACKBOARD_CONTEXT, self.SEMANTIC_ASSURANCE
        }

    @property
    def evaluation_ready(self) -> bool:
        """The arm can be scored and pooled into the controlled comparison.

        R0/R1 rest on deterministic coordination/context metrics. The R2-BBAG code
        runtime loop exists — A/G-aware generation (``ag_emitter``/``ag_chains``), the
        extractor/checker, orchestrator wiring, and the independent evaluator
        (``ag_evaluation.py``) all exist. Post-hoc readiness additionally requires
        the complete frozen configuration/boundary/gold/taxonomy/blind-label evidence
        chain; until that manifest clears, pooling R2 accuracy/F1 is not valid, so R2 stays
        runnable-but-not-poolable and is deliberately excluded here.
        """
        return self in {self.CURRENT, self.BLACKBOARD_CONTEXT}

    @property
    def uses_blackboard(self) -> bool:
        return self in {self.BLACKBOARD_CONTEXT, self.SEMANTIC_ASSURANCE}


def revised_arm_metadata(
    arm: RevisedExperimentArm, r2_generation_mode: str | None = None
) -> dict[str, object]:
    """Arm metadata for a run result.

    ``r2_generation_mode`` must be the mode the run actually executed. It used to
    be hardcoded to the deterministic intervention, which meant a run executing a
    different mode would report itself as deterministic — mislabelled evidence that
    the pooling gates would then accept, because they compare the recorded mode
    rather than observe the behaviour.
    """
    metadata: dict[str, object] = {
        "experiment_namespace": REVISED_EXPERIMENT_NAMESPACE,
        "configuration": arm.value,
        "implemented": arm.implemented,
        "evaluation_ready": arm.evaluation_ready,
        "common_generation_pipeline_version": COMMON_GENERATION_PIPELINE_VERSION,
        "blackboard_context_intervention": arm.uses_blackboard,
        "semantic_assurance_intervention": arm is RevisedExperimentArm.SEMANTIC_ASSURANCE,
        "global_long_session_diagnostic": (
            arm is RevisedExperimentArm.LONG_SESSION_DIAGNOSTIC
        ),
    }
    if arm is RevisedExperimentArm.SEMANTIC_ASSURANCE:
        mode = r2_generation_mode or R2_DETERMINISTIC_GENERATION_MODE
        if mode not in R2_INTERVENTION_VERSION_BY_MODE:
            raise ValueError(
                f"unknown r2_generation_mode {mode!r}; a run may not report an "
                "intervention that has no frozen version"
            )
        metadata.update({
            "r2_generation_mode": mode,
            # derived, never passed in: the recorded version must be the one bound
            # to the mode actually executed, or results from different
            # interventions could be pooled under one version
            "r2_intervention_version": R2_INTERVENTION_VERSION_BY_MODE[mode],
        })
    return metadata
