# Using AI for Rapid Prototyping in MBSE

The pipeline takes a natural-language requirement set and carries it through four stages:

1. **Generation**: an LLM writes a generation plan and then a SysML v2 model in staged calls; programs check the plan, the syntax (SysIDE), plan conformance and behavioural execution, and admit the model.
2. **Exploration**: catalogue-derived propulsion alternatives are attached to the model, sized and estimated analytically, and compared on endurance and mass.
3. **Realisation**: feasible candidates are matched to catalogue components and their mass, payload and endurance budgets are recalculated from product data.
4. **Evidence production**: the realised design is executed in model-level behavioural simulation, ArduPilot SITL and Gazebo; results are joined to the source requirements in an evidence matrix.

## Setup

Python 3.10 or later.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in the values
```

`.env` variables:

| Variable | Purpose |
|---|---|
| `VERTEX_API_KEY` | Vertex AI access; the reported studies used `gemini-3.1-pro-preview` through Vertex |
| `GEMINI_API_KEY` | Alternative provider for the Gemini API |
| `LLM_TIMEOUT_SECONDS` | Per-request timeout in seconds |
| `SYSIDE_LICENSE_KEY` | Licence key for SysIDE Automator, read by the `syside` package itself |
| `LANGSMITH_API_KEY` | Optional. When set, every LLM call is traced to LangSmith under `LANGSMITH_PROJECT`; without it the provider client runs unwrapped |
| `PINECONE_API_KEY` | Only for the retrieval module in `src/rag/` |

SysIDE Automator, the SysML v2 parser and validator, is installed from PyPI as `syside`. It is a licensed product of Sensmetry: the package validates a licence key against Sensmetry's licence server when imported, so `SYSIDE_LICENSE_KEY` must be set and the machine needs network access on first use (a 30-day trial key is available from the Syside pricing page). The generation and exploration stages need only Python, SysIDE and an LLM provider. The execution stages need external tools:

- **ArduPilot SITL**: a locally built `arducopter` binary. The SITL runner looks in `~/ardupilot/build/sitl/bin/` and then `~/PycharmProjects/ardupilot/build/sitl/bin/`; the Gazebo harness uses the second path. `MAVProxy` is installed from `requirements.txt`.
- **Gazebo Harmonic**: run in Docker from `examples/Dockerfile.gazebo` (image name `headless_gazebo`). Copy the `models` directory of `ardupilot_gazebo` to `gazebo_poc/templates`; the SDF generator overlays its generated vehicle on those templates.

## Running

Generate a model and explore the design space for the delivery-drone case:

```bash
python examples/run_pipeline.py
```

Set `RUN_GAZEBO=1` to include the Gazebo stage. Outputs go to `examples/output/` (override with `PROTOTYPING_OUTPUT_DIR`).

Study A, the isolated end-to-end run with SITL, Gazebo and the evidence matrix, in order:

```bash
python examples/run_realization_report.py
python examples/run_sitl_feasibility.py
python examples/run_gazebo_feasibility.py
python examples/build_verification_matrix.py
python examples/finalize_authoritative_run.py
```

Study B, the component study over pipeline variants (three seeds) and the fault-injection study of the repair layers:

```bash
python experiments/ablation/run_ablation.py --seeds 3 --label <campaign-name>
python experiments/ablation/mutation_study.py --provider vertex --source-campaign <campaign-dir> --source-arm NO-DSE --source-seed 0
python experiments/ablation/analyze.py <campaign-dir>
```

The arm identifiers in the code (`SINGLE-SHOT`, `NO-REFINE`, `NO-SURGICAL`, `NO-DETFIX`, `NO-REPAIR`) correspond to `SINGLE-PROMPT`, `ONE-ITERATION`, `WHOLE-MODEL-REPAIR`, `NO-REWRITE-RULES` and `LIMITED-REPAIR` in the report.

Tests:

```bash
pytest
```

SITL flight tests are opt-in (`pytest -m sitl`) and need the ArduPilot binary.

## Repository layout

Pipeline code lives in `src/`, one package per stage; `data/` holds inputs rather than code:

| Path | Contents |
|---|---|
| `src/agents/`, `src/prototyping/`, `src/llm/` | Generation: staged authoring, generation plan, checks and repair, runtime board |
| `src/sysml/`, `src/simulation/` | SysML v2 text model, rewrite rules, behavioural execution |
| `src/dse/` | Exploration: analytical estimator, design space, search |
| `src/realization/` | Realisation: component catalogue (`catalog.py`), matching, budget closure |
| `src/app/`, `src/utils/`, `src/rag/` | Pipeline entry point, helpers, retrieval module (unused by the reported runs) |
| `data/` | SysML v2 release examples, read only by the retrieval module |
