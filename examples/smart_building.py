"""
Smart Building Management System Prototyping Example.

Demonstrates the use of the AI-assisted MBSE prototyping framework
to rapidly design a smart building management system (BMS).

Usage:
    python examples/smart_building.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prototyping.pipeline import PrototypingPipeline
from src.prototyping.provider_factory import create_llm
from src.dse.design_space import DesignParameter, DesignSpace, ParameterType
from src.dse.mcts import MCTSDesignExplorer


BUILDING_DESCRIPTION = """
Design a Smart Building Management System (BMS) for a modern office building with 20 floors.

The system must:
- Control HVAC (heating, ventilation, air conditioning) based on occupancy and weather
- Manage lighting in all zones automatically and allow manual override
- Monitor and optimize energy consumption across all systems
- Provide access control and security monitoring
- Alert building managers to equipment faults and anomalies
- Integrate with external weather services for predictive control
- Provide a dashboard for building managers

The building houses 500 employees and operates 24/7 with peak hours 8am-6pm.
"""

BUILDING_REQUIREMENTS = [
    "REQ-FUNC-001: The BMS shall maintain indoor temperature at 21±1°C in occupied zones",
    "REQ-FUNC-002: The BMS shall control lighting levels based on occupancy and daylight",
    "REQ-FUNC-003: The BMS shall monitor all energy consumption with 1-minute granularity",
    "REQ-FUNC-004: The BMS shall provide access control for all entry points",
    "REQ-FUNC-005: The BMS shall alert operators within 5 minutes of equipment fault",
    "REQ-PERF-001: The BMS shall reduce energy consumption by ≥ 20% vs baseline",
    "REQ-PERF-002: The BMS shall respond to manual overrides within 2 seconds",
    "REQ-PERF-003: The BMS dashboard shall update every 30 seconds",
    "REQ-SAFE-001: The BMS shall maintain emergency lighting on power failure",
    "REQ-SAFE-002: The BMS shall unlock all exit doors during fire alarm",
    "REQ-INTF-001: The BMS shall integrate with the city's smart grid API",
]


def demonstrate_mcts_standalone():
    """Demonstrate standalone MCTS design space exploration for BMS."""
    print("\n" + "=" * 60)
    print("Standalone MCTS Design Space Exploration for BMS")
    print("=" * 60)

    # Define BMS-specific design space
    space = DesignSpace(name="BMS_DesignSpace")
    space.add_parameter(DesignParameter(
        name="hvac_control_strategy",
        param_type=ParameterType.CATEGORICAL,
        default_value="reactive",
        choices=["reactive", "predictive", "adaptive_ml"],
        description="HVAC control algorithm",
    ))
    space.add_parameter(DesignParameter(
        name="sensor_density",
        param_type=ParameterType.DISCRETE,
        default_value=2,
        choices=[1, 2, 3, 4],
        description="Sensors per room (1=minimal, 4=dense)",
    ))
    space.add_parameter(DesignParameter(
        name="update_frequency_hz",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.033,  # ~30s updates
        min_value=0.016,
        max_value=1.0,
        unit="Hz",
        description="Control loop update frequency",
    ))
    space.add_parameter(DesignParameter(
        name="edge_computing",
        param_type=ParameterType.BOOLEAN,
        default_value=False,
        description="Use edge computing for local processing",
    ))
    space.add_parameter(DesignParameter(
        name="redundancy",
        param_type=ParameterType.CATEGORICAL,
        default_value="none",
        choices=["none", "backup_server", "full_redundancy"],
    ))

    def bms_evaluate(config):
        """Evaluate a BMS configuration across multiple quality attributes."""
        params = config.parameters
        scores = {}

        # Energy efficiency based on control strategy and sensor density
        strategy_score = {"reactive": 0.5, "predictive": 0.8, "adaptive_ml": 0.95}
        scores["energy_efficiency"] = strategy_score.get(
            params.get("hvac_control_strategy", "reactive"), 0.5
        ) * min(1.0, params.get("sensor_density", 2) / 4.0 * 0.5 + 0.5)

        # Reliability based on redundancy and edge computing
        redundancy_score = {"none": 0.6, "backup_server": 0.8, "full_redundancy": 0.95}
        scores["reliability"] = redundancy_score.get(
            params.get("redundancy", "none"), 0.6
        )
        if params.get("edge_computing", False):
            scores["reliability"] = min(1.0, scores["reliability"] + 0.05)

        # Responsiveness based on update frequency
        freq = params.get("update_frequency_hz", 0.033)
        scores["responsiveness"] = min(1.0, freq / 0.1)

        # Cost (inverse: higher cost = lower score)
        base_cost = 0.8
        if params.get("hvac_control_strategy") == "adaptive_ml":
            base_cost -= 0.1
        if params.get("edge_computing", False):
            base_cost -= 0.05
        if params.get("redundancy") == "full_redundancy":
            base_cost -= 0.15
        scores["cost_effectiveness"] = max(0.0, base_cost)

        return scores

    explorer = MCTSDesignExplorer(
        design_space=space,
        evaluation_function=bms_evaluate,
        exploration_constant=1.5,
        max_depth=4,
        random_seed=42,
    )

    print(f"Starting MCTS exploration...")
    best_config = explorer.search(num_iterations=50)

    tree_summary = explorer.get_exploration_tree_summary()
    print(f"Explored {tree_summary['total_configurations']} configurations")
    print(f"Tree nodes: {tree_summary['total_nodes']}")
    print(f"Max depth reached: {tree_summary['max_depth_reached']}")

    print(f"\nBest configuration: {best_config.name}")
    print(f"Parameters:")
    for k, v in best_config.parameters.items():
        print(f"  {k}: {v}")
    print(f"Scores:")
    for k, v in best_config.scores.items():
        print(f"  {k}: {v:.3f}")
    print(f"Overall score: {best_config.overall_score:.3f}")

    # Show Pareto front
    pareto = space.get_pareto_front()
    print(f"\nPareto-optimal front: {len(pareto)} configurations")
    for cfg in pareto[:3]:
        print(f"  {cfg.name}: score={cfg.overall_score:.3f}, params={cfg.parameters}")


def main():
    print("=" * 70)
    print("AI-Assisted MBSE Rapid Prototyping: Smart Building Management System")
    print("=" * 70)
    print()

    llm = create_llm(provider="vertex")
    print(f"Using LLM: {llm.__class__.__name__}")

    pipeline = PrototypingPipeline(
        llm=llm,
        quality_threshold=0.65,
        max_iterations=2,
    )

    result = pipeline.prototype_system(
        system_name="SmartBuildingBMS",
        description=BUILDING_DESCRIPTION,
        additional_requirements=BUILDING_REQUIREMENTS,
        mcts_iterations=20,
    )

    print("\n" + "=" * 70)
    print("PROTOTYPING RESULTS")
    print("=" * 70)

    print(f"\nSystem: {result['system_name']}")
    print(f"Final Quality Score: {result['final_score']:.3f}")
    print(f"Components designed: {result['model_summary']['blocks_count']}")

    print("\n" + "=" * 70)
    print("GENERATED SysML v2 MODEL (excerpt)")
    print("=" * 70)
    sysml = result['model_sysml']
    # Show first 50 lines
    lines = sysml.splitlines()
    print("\n".join(lines[:50]))
    if len(lines) > 50:
        print(f"... ({len(lines) - 50} more lines)")

    # Demonstrate standalone MCTS
    demonstrate_mcts_standalone()

    print("\n✓ Smart Building BMS prototyping complete!")


if __name__ == "__main__":
    main()
