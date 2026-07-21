"""Unambiguous experiment namespaces for the legacy and revised Option 2 work.

The implemented external-contract pilot historically used ``B0/B1/B2``.  The
revised Blackboard/A-G study deliberately uses ``R*`` labels so archived runs
from the two designs cannot be pooled by name.
"""
from __future__ import annotations

from enum import Enum


LEGACY_EXPERIMENT_NAMESPACE = "LEGACY_EXTERNAL_CONTRACT_V1"
REVISED_EXPERIMENT_NAMESPACE = "BLACKBOARD_AG_V1"


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
        return self in {self.CURRENT, self.BLACKBOARD_CONTEXT}

    @property
    def uses_blackboard(self) -> bool:
        return self is self.BLACKBOARD_CONTEXT


def revised_arm_metadata(arm: RevisedExperimentArm) -> dict[str, object]:
    return {
        "experiment_namespace": REVISED_EXPERIMENT_NAMESPACE,
        "configuration": arm.value,
        "implemented": arm.implemented,
        "blackboard_context_intervention": arm is RevisedExperimentArm.BLACKBOARD_CONTEXT,
        "semantic_assurance_intervention": arm is RevisedExperimentArm.SEMANTIC_ASSURANCE,
        "global_long_session_diagnostic": (
            arm is RevisedExperimentArm.LONG_SESSION_DIAGNOSTIC
        ),
    }
