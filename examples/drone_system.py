"""
Autonomous Drone System Prototyping Example.

Demonstrates the use of the AI-assisted MBSE prototyping framework
to rapidly design an autonomous package delivery drone system.

Usage:
    python examples/drone_system.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prototyping.pipeline import PrototypingPipeline
from src.prototyping.provider_factory import create_llm


DRONE_DESCRIPTION = """
Design an autonomous drone system for last-mile package delivery in urban environments.

The drone must:
- Navigate autonomously from a distribution hub to delivery addresses
- Avoid obstacles including buildings, trees, and other aircraft
- Operate in varying weather conditions (wind up to 15 m/s, light rain)
- Carry payloads up to 2.5 kg
- Return to base if battery is low or mission fails
- Communicate with ground control for mission updates

The system operates in a regulated airspace and must comply with safety standards.
"""

DRONE_REQUIREMENTS = [
    "REQ-FUNC-001: The drone shall navigate autonomously to GPS waypoints with < 1m precision",
    "REQ-FUNC-002: The drone shall detect and avoid obstacles within 15m range",
    "REQ-FUNC-003: The drone shall carry payloads of up to 2.5 kg",
    "REQ-FUNC-004: The drone shall communicate with ground control via encrypted link",
    "REQ-PERF-001: The drone shall maintain stable flight within 0.5° roll/pitch deviation",
    "REQ-PERF-002: The drone shall achieve minimum 25 minutes flight endurance at full payload",
    "REQ-PERF-003: The drone shall reach maximum airspeed of 15 m/s",
    "REQ-SAFE-001: The drone shall auto-land safely if battery charge drops below 15%",
    "REQ-SAFE-002: The drone shall activate emergency landing on communication loss > 10s",
    "REQ-SAFE-003: The drone shall not operate if sensor self-test fails at startup",
]


def main():
    print("=" * 70)
    print("AI-Assisted MBSE Rapid Prototyping: Autonomous Drone System")
    print("=" * 70)
    print()

    llm = create_llm(provider="github_copilot")
    print(f"Using LLM: {llm.__class__.__name__}")
    print()

    # Initialize the prototyping pipeline
    pipeline = PrototypingPipeline(
        llm=llm,
        quality_threshold=0.65,
        max_iterations=3,
    )

    # Run the complete prototyping pipeline
    result = pipeline.prototype_system(
        system_name="AutonomousDrone",
        description=DRONE_DESCRIPTION,
        additional_requirements=DRONE_REQUIREMENTS,
        mcts_iterations=30,
    )

    # Display results
    print("\n" + "=" * 70)
    print("PROTOTYPING RESULTS")
    print("=" * 70)

    print(f"\nSystem: {result['system_name']}")
    print(f"Final Quality Score: {result['final_score']:.3f}")
    print(f"Iterations completed: {result['iterations']}")

    print(f"\nRequirements extracted: {len(result['requirements'])}")
    for req in result['requirements'][:5]:
        print(f"  • {req}")
    if len(result['requirements']) > 5:
        print(f"  ... and {len(result['requirements']) - 5} more")

    print(f"\nModel Summary:")
    summary = result['model_summary']
    print(f"  Blocks (components): {summary['blocks_count']}")
    print(f"  Requirements: {summary['requirements_count']}")
    print(f"  Connectors: {summary['connectors_count']}")
    print(f"  Components: {', '.join(summary['blocks'])}")

    print(f"\nDesign Space Exploration:")
    dse = result['design_space_summary']
    print(f"  Configurations explored: {dse['configurations_evaluated']}")
    print(f"  Pareto-optimal designs: {dse['pareto_front_size']}")
    print(f"  Best overall score: {dse['best_overall_score']:.3f}")

    if result['evaluation_history']:
        print(f"\nConvergence history:")
        for ev in result['evaluation_history']:
            bar = "█" * int(ev['score'] * 20)
            print(f"  Iter {ev['iteration']}: {ev['score']:.3f} {bar}")

    print("\n" + "=" * 70)
    print("GENERATED SysML v2 MODEL")
    print("=" * 70)
    print(result['model_sysml'])

    # Also show design alternatives
    print("\n" + "=" * 70)
    print("DESIGN ALTERNATIVES ANALYSIS")
    print("=" * 70)
    alternatives = pipeline.explore_alternatives(
        system_name="AutonomousDrone",
        description=DRONE_DESCRIPTION,
        num_alternatives=2,
    )
    for i, alt in enumerate(alternatives, 1):
        print(f"\nAlternative {i}: {alt['name']}")
        print(f"  Reasoning steps: {alt['thought_steps']}")
        if alt['sysml']:
            sysml_preview = alt['sysml'][:200] + "..." if len(alt['sysml']) > 200 else alt['sysml']
            print(f"  SysML preview:\n{sysml_preview}")

    print("\n✓ Drone system prototyping complete!")


if __name__ == "__main__":
    main()
