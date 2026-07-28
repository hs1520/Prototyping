from src.prototyping.generation_plan import ModelGenerationPlan
from src.prototyping.requirement_semantics import (
    NUMERIC_INVARIANT,
    RequirementSemanticObligation,
    compile_requirement_semantic_obligations,
)


_REQUIREMENT = (
    "REQ-FUNC-002: The system shall detect a stationary obstacle directly "
    "ahead within the forward sensor field of view and maintain at least "
    "5 metres of separation while avoiding it."
)


def _payload():
    return {
        "components": [
            {
                "name": "Perception",
                "responsibility": "Measures obstacle separation.",
                "requirements": ["REQ_FUNC_002"],
                "ports": [{
                    "name": "obstacleData",
                    "direction": "out",
                    "type": "ObstaclePort",
                }],
            },
            {
                "name": "Controller",
                "responsibility": "Maintains obstacle separation.",
                "requirements": ["REQ_FUNC_002"],
                "ports": [{
                    "name": "obstacleData",
                    "direction": "in",
                    "type": "ObstaclePort",
                }],
            },
        ],
        "connections": [{
            "source": {
                "component": "Perception",
                "port": "obstacleData",
            },
            "target": {
                "component": "Controller",
                "port": "obstacleData",
            },
            "item_type": "ObstaclePort",
            "requirements": ["REQ_FUNC_002"],
        }],
    }


def test_explicit_lower_bound_compiles_from_frozen_source():
    obligations = compile_requirement_semantic_obligations((_REQUIREMENT,))

    assert len(obligations) == 1
    obligation = obligations[0]
    assert obligation.obligation_id == "SEM_REQ_FUNC_002_001"
    assert obligation.requirement_id == "REQ_FUNC_002"
    assert obligation.kind == NUMERIC_INVARIANT
    assert obligation.subject_terms == ("separation",)
    assert obligation.operator == ">="
    assert obligation.threshold == 5.0
    assert obligation.unit == "m"
    assert len(obligation.source_digest) == 64


def test_non_matching_or_timed_requirement_is_not_overinterpreted():
    requirements = (
        "REQ-SAFE-005: deploy within 0.5 seconds after failure.",
        "REQ-FUNC-003: carry a payload.",
    )

    assert compile_requirement_semantic_obligations(requirements) == ()


def test_plan_carries_and_round_trips_frozen_semantic_obligation():
    plan = ModelGenerationPlan.from_payload(
        _payload(),
        requirements=(_REQUIREMENT,),
    )

    assert plan.status == "PASS"
    assert plan.schema_version == "4.0"
    assert len(plan.semantic_obligations) == 1
    restored = ModelGenerationPlan.from_dict(plan.to_dict())
    assert restored.semantic_obligations == plan.semantic_obligations
    assert restored.to_dict()["semantic_obligations"][0][
        "claim_boundary"
    ] == "MODEL_SEMANTIC_FIDELITY_NOT_PHYSICAL_PROOF"


def test_semantic_obligation_deserialisation_preserves_source_facts():
    original = compile_requirement_semantic_obligations((_REQUIREMENT,))[0]

    restored = RequirementSemanticObligation.from_dict(original.to_dict())

    assert restored == original
