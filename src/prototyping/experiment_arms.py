"""Unambiguous experiment namespaces for the legacy and revised Option 2 work.

The implemented external-contract pilot historically used ``B0/B1/B2``.  The
revised Blackboard/A-G study deliberately uses ``R*`` labels so archived runs
from the two designs cannot be pooled by name.
"""
from __future__ import annotations

from enum import Enum


LEGACY_EXPERIMENT_NAMESPACE = "LEGACY_EXTERNAL_CONTRACT_V1"
REVISED_EXPERIMENT_NAMESPACE = "BLACKBOARD_AG_V1"
R2_DETERMINISTIC_GENERATION_MODE = "DETERMINISTIC_SPEC_EMITTER"
R2_DETERMINISTIC_INTERVENTION_VERSION = (
    "r2-bbag-deterministic-spec-emitter-v1"
)


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


def revised_arm_metadata(arm: RevisedExperimentArm) -> dict[str, object]:
    metadata: dict[str, object] = {
        "experiment_namespace": REVISED_EXPERIMENT_NAMESPACE,
        "configuration": arm.value,
        "implemented": arm.implemented,
        "evaluation_ready": arm.evaluation_ready,
        "blackboard_context_intervention": arm.uses_blackboard,
        "semantic_assurance_intervention": arm is RevisedExperimentArm.SEMANTIC_ASSURANCE,
        "global_long_session_diagnostic": (
            arm is RevisedExperimentArm.LONG_SESSION_DIAGNOSTIC
        ),
    }
    if arm is RevisedExperimentArm.SEMANTIC_ASSURANCE:
        metadata.update({
            "r2_generation_mode": R2_DETERMINISTIC_GENERATION_MODE,
            "r2_intervention_version": R2_DETERMINISTIC_INTERVENTION_VERSION,
        })
    return metadata
