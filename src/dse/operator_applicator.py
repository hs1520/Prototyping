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
from .operators.protocol import CATALOG as PROTO_CATALOG
from .operators.redundantize import CATALOG

# pipeline redundancy_level -> operator variant
_LEVEL_TO_VARIANT = {"none": "single", "dual": "dual", "triple": "triple"}

_PREFIX = "Bdse"  # collision-safe prefix for the injected package-level type defs

# power-carrying port names are never retyped to a data protocol (domain separation)
_PWR_EXACT = re.compile(
    r"\b(power|pwr)(supply|in|out|bus|rail|link|feed|connector|line)\b",
    re.IGNORECASE,
)


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


def _merge_protocol(model_text: str, protocol: str) -> Tuple[str, bool, str]:
    """Retype generic data ports to the chosen catalog protocol signal.

    Injects the protocol operator's rich signal port def (item-typed payload;
    base ``Frame``/``Signal`` types Bdse-prefixed for collision safety), retypes
    directed ``DataPort``/``RfPort`` usages (power ports untouched), removes the
    stale semicolon-form generic defs, and keeps the result only if the whole
    model still parses — the same valid-by-construction gate as the redundancy
    merge.  Replaces the retired unvalidated regex injector
    (``apply_inject_protocol_to_sysml_text``).
    """
    key = re.sub(r"[^a-z0-9]", "", protocol.lower())
    if key not in PROTO_CATALOG:
        return model_text, False, ""
    signal, frame, _interop = PROTO_CATALOG[key]

    text = model_text
    # 1. Rich signal port def at package level (skip if the model has one)
    if not re.search(rf"\bport\s+def\s+{signal}\b", text):
        idx = text.rfind("}")
        if idx == -1:
            return model_text, False, ""
        defs = (
            f"    item def {_PREFIX}Frame;\n"
            f"    item def {frame} :> {_PREFIX}Frame;\n"
            f"    abstract port def {_PREFIX}Signal;\n"
            f"    port def {signal} :> {_PREFIX}Signal {{ in item payload : {frame}; }}\n"
        )
        text = text[:idx] + "\n" + defs + text[idx:]

    # 2. Retype directed generic data ports; power ports keep their type
    port_usage_re = re.compile(
        r"\b((?:in|out|inout)\s+port\s+(\w+)\s*:\s*)(DataPort|RFPort|RfPort)\b",
        re.IGNORECASE,
    )

    def _retype(m: re.Match) -> str:
        port_name = m.group(2).lower()
        if _PWR_EXACT.search(port_name) or port_name in ("power", "pwr"):
            return m.group(0)
        return f"{m.group(1)}{signal}"

    text = port_usage_re.sub(_retype, text)

    # 3. Drop stale semicolon-form generic port defs
    for generic in ("DataPort", "RFPort", "RfPort", "GenericPort"):
        text = re.sub(
            rf"^[ \t]*\bport\s+def\s+{generic}\s*;[ \t]*\n?",
            "", text, flags=re.IGNORECASE | re.MULTILINE,
        )

    if text == model_text:
        return model_text, False, ""
    if check_syntax(text).has_errors:
        return model_text, False, ""
    return text, True, signal


def apply_architecture(model, best_config) -> List[str]:
    """Apply the DSE architecture decisions via valid-by-construction merges.

    Redundancy/failsafe becomes an addressable part usage in the host model;
    the communication protocol retypes generic data ports to the catalog signal.
    Mutates ``model.metadata['last_sysml_text']``. Returns the applied decisions
    (empty if nothing applied or a merge was skipped to stay valid). The
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

    protocol = str(params.get("communication_protocol", "") or "")
    if protocol and protocol.lower() != "none":
        text3, ok, signal = _merge_protocol(text, protocol)
        if ok:
            text = text3
            applied.append(f"protocol={protocol} (validated retype → {signal})")

    if applied:
        if not hasattr(model, "metadata") or model.metadata is None:
            object.__setattr__(model, "metadata", {})
        model.metadata["last_sysml_text"] = text
        model.metadata["bilevel_operator_applied"] = applied
    return applied
