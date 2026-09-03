from src.agents.requirements_agent import RequirementsAgent
from src.llm.interface import MockLLM


def test_fixed_requirement_wins():
    manual = (
        "REQ-SAFE-001: The system shall initiate return to base when battery "
        "state of charge reaches 25%."
    )
    llm = MockLLM()
    llm.inject_response(
        manual + " [SEV:Major]\n"
        "REQ-FUNC-002: The system shall report health after landing."
    )
    agent = RequirementsAgent(llm)

    result = agent.run({
        "system_description": "A delivery aircraft performs a mission and reports its health. " * 2,
        "system_name": "Drone",
        "existing_requirements": [manual],
    })

    assert result.output.count(manual) == 1
    assert len([r for r in result.output if r.startswith("REQ-SAFE-001:")]) == 1
    assert any(r.startswith("REQ-FUNC-002:") for r in result.output)
    assert agent.validate_requirements(result.output)["valid"]
