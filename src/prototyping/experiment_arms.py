"""Unambiguous experiment namespaces for the legacy and revised Option 2 work.

The implemented external-contract pilot historically used ``B0/B1/B2``.  The
revised Blackboard/A-G study deliberately uses ``R*`` labels so archived runs
from the two designs cannot be pooled by name.
"""
from __future__ import annotations

from enum import Enum


LEGACY_EXPERIMENT_NAMESPACE = "LEGACY_EXTERNAL_CONTRACT_V1"
REVISED_EXPERIMENT_NAMESPACE = "BLACKBOARD_AG_V1"
COMMON_GENERATION_PIPELINE_VERSION = "typed-whole-model-plan-v9"
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
R2_LLM_AUTHORED_GENERATION_MODE = "LLM_AUTHORED_AG"
R2_LLM_AUTHORED_INTERVENTION_VERSION = (
    "r2-bbag-whole-model-guided-authored-v8"
)

# LLM-decided specs are a THIRD frozen intervention. The LLM emits the engineering
# decisions — pattern, timing origin and apportionment, discharge wiring, response
# ordering — and the deterministic emitter renders the SysML from them. Measurement
# motivated it: the LLM-authored mode agreed with gold on allocation 6/6 and
# discharge 4/6 while never once reaching a PASS, because it kept losing rounds to
# the notation rather than to the engineering. Here conformance holds by
# construction and only the decisions are judged. Separate frozen config; never
# pooled with either other mode.
R2_LLM_DECIDED_GENERATION_MODE = "LLM_DECIDED_SPEC"
R2_LLM_DECIDED_INTERVENTION_VERSION = (
    "r2-bbag-whole-model-guided-decided-v5"
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
