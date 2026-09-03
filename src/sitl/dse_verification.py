"""Layer 0+1 entry point: turn a DSE-recommended model into a verifiable artifact."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..dse.verification_builder import build_verification_cases
from .parameter_projection import (
    ParameterCheck,
    settable_parm_lines,
    validate_settable_parameters,
)


@dataclass
class DSEVerificationReport:
    verification_model: str
    verification_cases: List[str]
    parm_lines: List[str]
    l1_ok: bool
    l1_results: List[ParameterCheck] = field(default_factory=list)

    def summary(self) -> str:
        vc = len(self.verification_cases)
        l1 = f"{sum(r.ok for r in self.l1_results)}/{len(self.l1_results)}"
        return (f"{vc} verification case(s), L1 {l1} settable params in range, "
                f"{len(self.parm_lines)} .parm line(s)")


def build_dse_verification(model_text: str, requirements: List[str]) -> DSEVerificationReport:
    """Run the layer 0+1 loop on a DSE-recommended model.

    Cases come from the quantified requirements; L1 reads the settable-family
    values off the resolved model's variant attributes. Emergent families are
    deferred to L2 (see memory `sitl-family-param-mapping`).
    """
    vmodel, vnames = build_verification_cases(model_text, requirements)
    l1_ok, l1_results = validate_settable_parameters(model_text)
    parm = settable_parm_lines(model_text)
    return DSEVerificationReport(
        verification_model=vmodel,
        verification_cases=vnames,
        parm_lines=parm,
        l1_ok=l1_ok,
        l1_results=l1_results,
    )


def build_from_explore_result(result: Dict[str, Any]) -> DSEVerificationReport:
    """Convenience: run the loop on an Orchestrator.explore()/prototype() result dict."""
    return build_dse_verification(
        result.get("model_sysml", ""), result.get("requirements", []) or []
    )
