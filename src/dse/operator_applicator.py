"""Apply the chosen architecture to a model via valid-by-construction operators.

Item F step 4 (+ connectivity fix): when the bilevel DSE drives Phase 3, the
redundancy / failsafe structure is applied from the operators' Syside-verified
catalog SysML instead of the brittle regex injection + LLM grounding pass.

Connectivity-correct form: the catalog *type definitions* go in a nested namespace
package (collision-safe), but the safety monitor is added as an ADDRESSABLE part
usage in the MAIN model (``part bdseSafetyMonitor : BdseArchitecture::<Channel>;``).
Being a real part usage in the host model, it is within reach of the pipeline's
surgical connectivity_fixer, which wires it to the flight controller — unlike the
earlier isolated-nested-package form, which was valid but unreachable (reachability
collapsed to ~0 because the injected parts could not be addressed or wired).

The merged model is re-checked with Syside; any merge that would break parsing is
skipped (model left untouched), so this can never produce invalid SysML.
"""
from __future__ import annotations

import re
from typing import List, Tuple

from ..simulation.syntax_checker import check_syntax
from ..utils.sysml_text_utils import get_sysml_text
from .operators.redundantize import CATALOG

# pipeline redundancy_level -> operator variant
_LEVEL_TO_VARIANT = {"none": "single", "dual": "dual", "triple": "triple"}

_PREFIX = "Bdse"  # collision-safe prefix for the injected package-level type defs


def _merge(model_text: str, variant: str) -> Tuple[str, bool]:
    """Insert the channel type defs + an addressable safety-monitor part usage at
    the MAIN package level, then keep the merge only if it still parses.

    Type names are PREFIXED (not put in a sub-package) so the monitor is a plain
    ``part bdseSafetyMonitor : Bdse<Channel>;`` usage with a LOCAL type — which the
    behavioral-graph extractor recognises as an instance (a sub-package qualified
    type ``Pkg::Channel`` parsed fine but the extractor reported it as an unknown
    instance, so the connectivity_fixer's wires were rejected). The prefix keeps it
    collision-safe with the host model's own names.
    """
    name, _n, frag = CATALOG[variant]
    block = "        port def SensorSignal;\n        port def CommandSignal;\n" + frag
    for old in ("SensorSignal", "CommandSignal", name):
        block = re.sub(rf"\b{old}\b", _PREFIX + old, block)

    idx = model_text.rfind("}")
    if idx == -1:
        return model_text, False
    import_line = "" if "ScalarValues" in model_text else "    private import ScalarValues::*;\n"
    inject = (
        "\n" + import_line + block + "\n"
        f"    part bdseSafetyMonitor : {_PREFIX}{name};\n"
    )
    merged = model_text[:idx] + inject + model_text[idx:]
    if check_syntax(merged).has_errors:
        return model_text, False
    return merged, True


def apply_architecture(model, best_config) -> List[str]:
    """Apply redundancy/failsafe as an addressable part usage in the host model.

    Mutates ``model.metadata['last_sysml_text']``. Returns the applied decisions
    (empty if nothing applied or the merge was skipped to stay valid). The
    downstream connectivity_fixer wires ``bdseSafetyMonitor`` to the controller.
    """
    params = getattr(best_config, "parameters", {}) or {}
    level = str(params.get("redundancy_level", "none"))
    variant = _LEVEL_TO_VARIANT.get(level, "single")

    applied: List[str] = []
    text = get_sysml_text(model)

    if variant in ("dual", "triple"):
        text2, ok = _merge(text, variant)
        if ok:
            text = text2
            applied.append(f"redundancy={variant} (addressable part bdseSafetyMonitor)")

    if applied:
        if not hasattr(model, "metadata") or model.metadata is None:
            object.__setattr__(model, "metadata", {})
        model.metadata["last_sysml_text"] = text
        model.metadata["bilevel_operator_applied"] = applied
    return applied
