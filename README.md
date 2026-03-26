# AI-Assisted Rapid Prototyping for Model Based Systems Engineering

A framework for using AI and Large Language Models (LLMs) to rapidly prototype cyber-physical systems using SysML v2, implementing design space exploration with Monte Carlo Tree Search, Chain of Thought prompting, and Retrieval Augmented Generation.

## Overview

This project explores the use of AI and LLMs for rapid prototyping in Model Based Systems Engineering (MBSE). It implements several advanced techniques:

| Technique | Implementation |
|-----------|---------------|
| **Chain of Thought (CoT) Prompting** | Structured step-by-step reasoning for design decisions |
| **Retrieval Augmented Generation (RAG)** | Domain knowledge base of SysML v2 patterns and design principles |
| **Multi-Agent Architecture** | Specialized agents for requirements, design, and evaluation |
| **Monte Carlo Tree Search (MCTS)** | Design space exploration balancing exploration vs. exploitation |
| **SysML v2 Modeling** | Full representation and serialization of SysML v2 elements |

## Architecture

```
src/
├── sysml/           # SysML v2 model representation
│   └── model.py     # Block, Port, Attribute, Requirement, Action, Connector, SysMLModel
├── llm/             # LLM interface and prompting
│   ├── interface.py        # Abstract LLM interface, OpenAI and Mock implementations
│   └── chain_of_thought.py # CoT prompting for design, evaluation, and refinement
├── rag/             # Retrieval Augmented Generation
│   ├── knowledge_base.py   # Built-in SysML v2 patterns, design principles, examples
│   └── retriever.py        # TF-IDF retrieval + LLM generation
├── dse/             # Design Space Exploration
│   ├── design_space.py     # Parameter types, configurations, Pareto front
│   ├── mcts.py             # Monte Carlo Tree Search with UCB1
│   └── evaluator.py        # Multi-criteria design quality evaluation
├── agents/          # Multi-agent framework
│   ├── base_agent.py       # Abstract agent with RAG augmentation
│   ├── requirements_agent.py # Requirements extraction and validation
│   ├── design_agent.py     # SysML v2 design generation and refinement
│   └── orchestrator.py     # Multi-agent orchestration pipeline
└── prototyping/     # High-level pipeline API
    └── pipeline.py         # PrototypingPipeline entry point
```

## Prototyping Pipeline

The framework implements a cyclic design process that combines forward and backward inference:

```
System Description
       ↓
[RequirementsAgent] → CoT + RAG → Structured Requirements
       ↓
[DesignAgent] → CoT + RAG → Initial SysML v2 Model
       ↓
[MCTS Explorer] → Design Space Exploration → Best Configuration
       ↓
[DesignEvaluator] → Multi-criteria Scoring
       ↓
[DesignAgent] ← Refinement feedback ← Score < threshold?
       ↓ (converged)
Final SysML v2 Model + Pareto-optimal Alternatives
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Using the Mock LLM (no API key required)

```python
from src.prototyping.pipeline import PrototypingPipeline

pipeline = PrototypingPipeline()
result = pipeline.prototype_system(
    system_name="AutonomousDrone",
    description="""
    An autonomous drone for package delivery that navigates to GPS waypoints,
    avoids obstacles, and auto-lands on low battery.
    """,
    additional_requirements=[
        "REQ-FUNC-001: The drone shall navigate to GPS waypoints with < 1m precision",
        "REQ-SAFE-001: The drone shall auto-land if battery drops below 15%",
    ],
)

print(result["model_sysml"])
print(f"Quality score: {result['final_score']:.3f}")
```

### Using the OpenAI API

```python
import os
from src.prototyping.pipeline import PrototypingPipeline, create_llm

llm = create_llm(use_openai=True, model="gpt-4o", api_key=os.environ["OPENAI_API_KEY"])
pipeline = PrototypingPipeline(llm=llm)
result = pipeline.prototype_system(...)
```

## Running Examples

```bash
# Autonomous drone system (uses mock LLM)
python examples/drone_system.py

# Smart building management system (uses mock LLM)
python examples/smart_building.py

# With real OpenAI API (requires OPENAI_API_KEY environment variable)
python examples/drone_system.py --openai
```

## Design Space Exploration with MCTS

The framework uses Monte Carlo Tree Search to navigate the design space, implementing the classic four phases:

1. **Selection** — Traverse the tree using UCB1 to find a promising leaf node
2. **Expansion** — Add a new child with a modified design configuration
3. **Simulation** — Evaluate random rollout from the expanded node
4. **Backpropagation** — Update all ancestor nodes with the simulation reward

```python
from src.dse.design_space import DesignSpace, DesignParameter, ParameterType
from src.dse.mcts import MCTSDesignExplorer

space = DesignSpace(name="MySystem_DesignSpace")
space.add_parameter(DesignParameter(
    name="control_strategy",
    param_type=ParameterType.CATEGORICAL,
    default_value="reactive",
    choices=["reactive", "predictive", "adaptive_ml"],
))
space.add_parameter(DesignParameter(
    name="update_rate_hz",
    param_type=ParameterType.CONTINUOUS,
    default_value=10.0,
    min_value=1.0,
    max_value=100.0,
))

def my_evaluator(config):
    return {
        "performance": compute_performance(config),
        "reliability": compute_reliability(config),
    }

explorer = MCTSDesignExplorer(space, my_evaluator, random_seed=42)
best = explorer.search(num_iterations=100)
pareto = space.get_pareto_front()
```

## SysML v2 Model Generation

The framework generates SysML v2 models in standard text notation:

```sysml
package AutonomousDrone {
    doc /* Auto-generated design for AutonomousDrone */

    // Requirements
    requirement REQ_FUNC_001 {
        doc /* The drone shall navigate autonomously to GPS waypoints with < 1m precision */
    }
    requirement REQ_SAFE_001 {
        doc /* The drone shall auto-land safely if battery charge drops below 15% */
    }

    // Part Definitions
    part def FlightController {
        port sensorIn : SensorPort;
        port commandOut : MotorPort;
        attribute loopRate : Real = 100.0 [Hz];
    }

    part def IMU {
        port dataOut : SensorPort;
        attribute samplingRate : Real = 1000.0 [Hz];
    }

    // Connections
    connect IMU.dataOut to FlightController.sensorIn;
}
```

## Multi-Agent Architecture

Each agent has a specialized role and can leverage RAG for domain knowledge:

- **RequirementsAgent** — Extracts structured requirements using CoT, validates completeness, classifies by type (FUNC/PERF/SAFE/INTF/CONS)
- **DesignAgent** — Generates and refines SysML v2 models, parses SysML text, uses heuristic fallback
- **Orchestrator** — Coordinates the full pipeline, manages iterative refinement until quality threshold

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_dse.py -v

# Run with coverage
python -m pytest tests/ --cov=src
```

## Design Principles

### Modular Architecture
Each module is independently testable and can be used standalone.

### Mock-First Development
The `MockLLM` class provides realistic responses for development and testing without API costs.

### Pareto-Optimal Design Exploration
The framework computes the Pareto front of explored designs, enabling multi-objective trade-off analysis between performance, reliability, cost, and other quality attributes.

### Traceable Requirements
All generated design elements include `satisfies` links back to requirements, maintaining end-to-end traceability.

## Project Structure

```
Prototyping/
├── README.md
├── requirements.txt
├── pytest.ini
├── src/
│   ├── sysml/           # SysML v2 model elements
│   ├── llm/             # LLM interface + Chain of Thought
│   ├── rag/             # Knowledge base + Retrieval
│   ├── dse/             # Design space + MCTS + Evaluator
│   ├── agents/          # Multi-agent framework
│   └── prototyping/     # High-level pipeline
├── tests/
│   ├── test_sysml.py    # 26 tests for SysML model
│   ├── test_llm.py      # 19 tests for LLM + CoT
│   ├── test_rag.py      # 23 tests for RAG knowledge base
│   ├── test_dse.py      # 25 tests for DSE + MCTS
│   └── test_agents.py   # 22 tests for multi-agent system
└── examples/
    ├── drone_system.py      # Autonomous drone prototyping demo
    └── smart_building.py    # Smart BMS prototyping demo
```

## References

- [SysML v2 Specification](https://www.omg.org/spec/SysML/)
- [Chain of Thought Prompting (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)
- [Retrieval Augmented Generation (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- [Monte Carlo Tree Search Survey](https://arxiv.org/abs/1208.4468)
- [Model Based Systems Engineering (MBSE)](https://www.incose.org/docs/default-source/working-groups/model-based-se/mbse-initiative-description.pdf)