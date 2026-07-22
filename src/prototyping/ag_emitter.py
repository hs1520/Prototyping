"""Deterministic emitter for the bounded A/G contract layer (Stage 2-3, §12).

A reviewed A/G *decomposition* (the Stage-2 human-reviewed chain structure) is
rendered into valid SysML v2 and merged into the model, so the committed model —
the sole semantic authority (§6.2) — actually carries the contracts and R2-BBAG
traces are non-empty. Emission is deterministic and syntax-gate-valid; it uses
only the validated convention (`requirement def` + `assume`/`require constraint`
+ `dependency decomposition`) and invents no keywords. The emitted layer is the
inverse of ``ag_extractor`` and round-trips to a checker PASS for a sound chain.

This renders the *contract* layer only. Behaviour that realises each guarantee
(state machines, actions) remains ordinary LLM generation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple


@dataclass(frozen=True)
class AGAssumptionSpec:
    concept: str
    environment: bool = False


@dataclass(frozen=True)
class AGComponentSpec:
    name: str                       # component requirement-def name
    guarantee: str                  # Boolean concept the component publishes
    assumptions: Tuple[AGAssumptionSpec, ...] = ()
    latency_budget: Optional[float] = None


@dataclass(frozen=True)
class AGChainSpec:
    source_requirement: str         # immutable stakeholder req id (provenance)
    package: str                    # SysML package name for the contract layer
    system_contract: str            # system requirement-def name
    system_assumptions: Tuple[str, ...]  # Boolean environment/trigger concepts
    observation: str                # observed system-level Boolean concept
    deadline: Optional[float]       # system deadline (maxLatency), seconds
    components: Tuple[AGComponentSpec, ...]


def _fmt(value: float) -> str:
    return repr(float(value))


def _dedup(concepts: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(c for c in concepts if c))


def emit_ag_package(spec: AGChainSpec) -> str:
    """Render one reviewed A/G chain as a valid SysML v2 package."""
    out: list[str] = [f"package {spec.package} {{"]

    # System contract.
    out.append(f"    requirement def {spec.system_contract} {{")
    out.append(
        f"        doc /* bounded A/G system contract for {spec.source_requirement} */"
    )
    for concept in _dedup([*spec.system_assumptions, spec.observation]):
        out.append(f"        attribute {concept} : Boolean;")
    if spec.deadline is not None:
        out.append(f"        attribute maxLatency : Real = {_fmt(spec.deadline)};")
    for concept in spec.system_assumptions:
        out.append(f"        assume constraint a_{concept} {{ {concept} }}")
    out.append(f"        require constraint g_observed {{ {spec.observation} }}")
    out.append("    }")

    # Component contracts.
    for comp in spec.components:
        out.append(f"    requirement def {comp.name} {{")
        concepts = _dedup([*(a.concept for a in comp.assumptions), comp.guarantee])
        for concept in concepts:
            out.append(f"        attribute {concept} : Boolean;")
        if comp.latency_budget is not None:
            out.append(
                f"        attribute latencyBudget : Real = {_fmt(comp.latency_budget)};"
            )
        for assumption in comp.assumptions:
            prefix = "env_" if assumption.environment else "a_"
            out.append(
                f"        assume constraint {prefix}{assumption.concept} "
                f"{{ {assumption.concept} }}"
            )
        out.append(
            f"        require constraint g_{comp.guarantee} {{ {comp.guarantee} }}"
        )
        out.append("    }")

    # Decomposition edges (unique names → no namespace-shadowing warning).
    for comp in spec.components:
        out.append(
            f"    dependency decompose{comp.name} "
            f"from {spec.system_contract} to {comp.name};"
        )
    out.append("}")
    return "\n".join(out)


def merge_ag_contracts(model_text: str, specs: Iterable[AGChainSpec]) -> str:
    """Append the emitted A/G contract packages to the model text.

    Multiple top-level packages are valid SysML; the extractor scans the whole
    text. The base model is preserved verbatim.
    """
    packages = [emit_ag_package(spec) for spec in specs]
    if not packages:
        return model_text
    return (model_text or "").rstrip() + "\n\n" + "\n\n".join(packages) + "\n"
