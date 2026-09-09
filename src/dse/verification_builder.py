"""Generate SysML v2 verification cases for a DSE-recommended model.

Layer 0 of the DSE->SITL plan: make the verification intent explicit and
traceable inside the SysML v2 model, per the standard's verification-case
framework. For every quantified requirement (one with a numeric target -
speed/endurance/mass...) it emits, against the model's system type:

    requirement def <Rid><Fam>Req { doc /* req + direction */ attribute target : Real = <v>; }
    requirement <rid><fam>Req : <Rid><Fam>Req;                       // usage
    verification def <Rid><Fam>Verification {
        subject s : <SystemType>;
        objective { verify <rid><fam>Req; }                          // verify the usage
    }

The numeric verdict is left to the execution layer (SITL .parm + L2 test compares
the measured value against `target`), so one case can run at several fidelities.
Syntax is Syside-verified; any emission that would not parse is discarded (model
returned unchanged).
"""
from __future__ import annotations

import re
from typing import List, Tuple

from ..simulation.syntax_checker import check_syntax
from .domain_objective import _PERF_FAMILIES, requirement_targets


def _find_system_type(model_text: str) -> str:
    best, best_n = "", -1
    for m in re.finditer(r"\bpart\s+def\s+(\w+)\s*(?::>[^{]*)?\{", model_text):
        name = m.group(1)
        brace = model_text.index("{", m.start())
        depth, i = 1, brace + 1
        while i < len(model_text) and depth:
            depth += (model_text[i] == "{") - (model_text[i] == "}")
            i += 1
        body = model_text[brace + 1 : i - 1]
        n = len(re.findall(r"\bpart\s+\w+\s*:", body))
        if n > best_n:
            best, best_n = name, n
    if best_n >= 1:
        return best
    # Flat package assembly (parts declared at package scope, no wrapping part def -
    # how the LLM often emits models): use the most-connected part's type as the
    # verification subject's representative.
    usage_type = dict(re.findall(r"\bpart\s+(\w+)\s*:\s*(\w+)\s*;", model_text))
    deg: dict = {}
    for a, b in re.findall(r"\bconnect\s+(\w+)\.\w+\s+to\s+(\w+)\.\w+", model_text, re.IGNORECASE):
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    for usage in sorted(deg, key=lambda u: -deg[u]):
        if usage in usage_type:
            return usage_type[usage]
    return ""


def _ident(rid: str, fam: str) -> str:
    core = "".join(p.capitalize() for p in re.split(r"[^A-Za-z0-9]", rid) if p)
    return f"{core}{fam.capitalize()}"


def build_verification_cases(
    model_text: str, requirements: List[str]
) -> Tuple[str, List[str]]:
    """Insert a verification case per (quantified requirement x family) into the model."""
    targets = requirement_targets(requirements)
    if not targets:
        return model_text, []
    system_type = _find_system_type(model_text)
    if not system_type:
        return model_text, []

    blocks: List[str] = []
    vdefs: List[str] = []
    seen: set = set()
    for rid, fts in targets.items():
        for fam, val in fts:
            base = _ident(rid, fam)
            if base in seen:
                continue
            seen.add(base)
            direction = ">=" if fam in _PERF_FAMILIES else "<="
            rdef, ruse = f"{base}Req", f"{base[0].lower()}{base[1:]}Req"
            vdef = f"{base}Verification"
            blocks.append(
                f"    requirement def {rdef} {{\n"
                f"        doc /* {rid}: {fam} {direction} {val} (verification target) */\n"
                f"        attribute target : Real = {float(val)};\n"
                f"    }}\n"
                f"    requirement {ruse} : {rdef};\n"
                f"    verification def {vdef} {{\n"
                f"        subject s : {system_type};\n"
                f"        objective {{ verify {ruse}; }}\n"
                f"    }}"
            )
            vdefs.append(vdef)

    if not blocks:
        return model_text, []

    idx = model_text.rfind("}")
    if idx == -1:
        return model_text, []
    addition = (
        "\n    // ── verification cases (auto-generated from quantified requirements) ──\n"
        + "\n".join(blocks)
        + "\n"
    )
    result = model_text[:idx] + addition + model_text[idx:]
    if check_syntax(result).has_errors:
        return model_text, []
    return result, vdefs
