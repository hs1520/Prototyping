from __future__ import annotations

from src.agents.generated_model_admission import (
    GeneratedModelAdmission,
    ModelAdmissionRequest,
)
from src.agents.requirements_agent import RequirementsAgent
from src.llm.chain_of_thought import CoTResult
from src.sysml.lite_model import build_lite_model


_REQ = (
    "REQ-SAFE-001: The system shall initiate return to base when the battery "
    "state of charge reaches 25 percent."
)


def test_missing_satisfy_stays_visible():
    text = """package D {
        requirement def REQ_SAFE_001 { doc /* battery return requirement */ }
        part def BatteryReturnController {
            attribute batteryStateOfCharge : Real = 100.0;
            action def initiateReturnToBase { }
        }
    }"""
    outcome = GeneratedModelAdmission(build_lite_model, None).accept(
        ModelAdmissionRequest(
            response=CoTResult(final_answer=text, extracted_sysml=text),
            system_name="D",
            requirements=[_REQ],
            generation_metadata={},
            is_refinement=True,
        )
    )
    model = outcome.model
    untraced = list(outcome.untraced_requirements)

    assert untraced == ["REQ_SAFE_001"]
    assert model.part_definitions[0].satisfy_relationships == []
    assert model.to_sysml_text() == text


def test_cache_no_fabricated_requirement():
    text = "package D { part def BatteryReturnController { } }"
    model = build_lite_model(text, model_name="D")

    RequirementsAgent.create_sysml_requirements(
        object.__new__(RequirementsAgent), [_REQ], model
    )

    assert model.requirement_definitions == []
    assert "REQ_SAFE_001" not in model.to_sysml_text()
