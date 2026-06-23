"""Emit a self-contained, ANALYSABLE SysML v2 product-line fragment from a design.

This closes the gap where variant attributes were "bare" (no constraint / analysis /
satisfy). It mirrors the Python physics estimator into a SysML v2 ``calc def`` that
**Syside Automator can actually evaluate** (operators only — ``**`` for power/root), so
the model itself can compute emergent performance, check it against the requirement
(``assert constraint``), and trace it (``satisfy``) — not just carry numbers.

Single source of truth: the calc def's constants come from ``physics_estimator``, so the
SysML analysis and the Python DSE/calibration use the SAME physics (they evaluate equal).
"""
from __future__ import annotations

import math
import re
from typing import Optional, Tuple

from ..simulation.syntax_checker import check_syntax
from ..utils.sysml_text_utils import find_block_end
from .domain_objective import (
    DESIGN_DEFAULTS, DESIGN_FIELD_ATTR, endurance_target, variant_design_inputs,
)
from .physics_estimator import (
    AVIONICS_POWER_W, BASE_FRAME_KG, CELL_V, DesignInputs, ENERGY_DENSITY_WH_KG,
    ETA_DRIVE, FOM, G, RHO, ROTOR_MASS_COEF, USABLE,
)

_USAGE_RE = re.compile(r"\bpart\s+(\w+)\s*:\s*(\w+)\s*;")
# Endurance(...) argument order ↔ DesignInputs field name
_ARG_ORDER = ("battery_capacity_mah", "battery_cells", "rotor_count",
              "rotor_radius_m", "payload_mass_kg")


def endurance_calc_def(indent: str = "    ") -> str:
    """SysML v2 ``calc def Endurance`` mirroring physics_estimator.endurance_min.
    Constants are interpolated from the estimator → one source of truth."""
    pi = math.pi
    return (
        f"{indent}calc def Endurance {{\n"
        f"{indent}    in capacityMah : Real; in cells : Real; in rotorCount : Real;\n"
        f"{indent}    in rotorRadiusM : Real; in payloadKg : Real;\n"
        f"{indent}    attribute energyWh : Real = capacityMah / 1000.0 * cells * {CELL_V};\n"
        f"{indent}    attribute diskArea : Real = rotorCount * {pi} * (rotorRadiusM ** 2);\n"
        f"{indent}    attribute totalMass : Real = {BASE_FRAME_KG} + {ROTOR_MASS_COEF} * diskArea "
        f"+ energyWh / {ENERGY_DENSITY_WH_KG} + payloadKg;\n"
        f"{indent}    attribute hoverPower : Real = (totalMass * {G}) ** 1.5 "
        f"/ (((2.0 * {RHO} * diskArea) ** 0.5) * {FOM});\n"
        f"{indent}    attribute elecPower : Real = hoverPower / {ETA_DRIVE} + {AVIONICS_POWER_W};\n"
        f"{indent}    energyWh * {USABLE} / elecPower * 60.0\n"
        f"{indent}}}"
    )


def _invocation(d: DesignInputs) -> str:
    return (f"Endurance({float(d.battery_capacity_mah)}, {float(d.battery_cells)}, "
            f"{float(d.rotor_count)}, {d.rotor_radius_m}, {d.payload_mass_kg})")


def emit_endurance_analysis(
    d: DesignInputs, target_min: float, satisfy_req: str = "REQ-PERF-002",
    package_name: str = "AnalyzedProductLine", part_name: str = "AnalyzedDesign",
) -> Tuple[str, bool]:
    """A self-contained analysable SysML fragment: calc def + derived endurance +
    ``assert constraint`` against the requirement target + ``satisfy``. Returns
    (sysml, ok); ok=False (empty) if it would not parse."""
    inv = _invocation(d)
    rid = satisfy_req.replace("-", "_")
    sysml = (
        f"package {package_name} {{\n"
        f"{endurance_calc_def()}\n"
        f"    requirement def {rid} {{\n"
        f"        doc /* {satisfy_req}: endurance >= {target_min} min */\n"
        f"        attribute target : Real = {target_min};\n"
        f"    }}\n"
        f"    requirement {rid.lower()} : {rid};\n"
        f"    part def {part_name} {{\n"
        f"        attribute enduranceMin : Real = {inv};\n"
        f"        assert constraint enduranceMeetsReq {{ {inv} >= {target_min} }}\n"
        f"        satisfy {rid.lower()};\n"
        f"    }}\n"
        f"}}"
    )
    return (sysml, True) if not check_syntax(sysml).has_errors else ("", False)


def _root_part_def(text: str) -> Optional[Tuple[str, int, int]]:
    """(name, brace_index, close_brace_index) of the part def owning the most part
    usages — the system assembly the analysis closure attaches to."""
    best = None
    for m in re.finditer(r"\bpart\s+def\s+(\w+)\s*(?::>[^{]*)?\{", text):
        brace = text.index("{", m.start())
        end = find_block_end(text, brace)
        if end == -1:
            continue
        n = len(_USAGE_RE.findall(text[brace + 1:end]))
        if best is None or n > best[0]:
            best = (n, m.group(1), brace, end)
    return (best[1], best[2], best[3]) if best and best[0] >= 1 else None


def _inject_attr_into_type(text: str, type_name: str, attr: str, value: float) -> Tuple[str, bool]:
    m = re.search(rf"\bpart\s+def\s+{re.escape(type_name)}\s*(?::>[^{{]*)?\{{", text)
    if not m:
        return text, False
    end = find_block_end(text, text.index("{", m.start()))
    if end == -1:
        return text, False
    return text[:end] + f" attribute {attr} : Real = {value};" + text[end:], True


def _analysis_scope(text: str) -> Optional[Tuple[str, int, bool]]:
    """Where the closure attaches: (usages_body, close_brace_index, wrap_in_partdef).
    Prefer a system root part def (nested assembly); else fall back to package level
    (flat assembly — the closure goes into a new wrapper part def there)."""
    root = _root_part_def(text)
    if root is not None:
        _, brace, end = root
        return text[brace + 1:end], end, False
    pkg = re.search(r"\bpackage\s+\w+\s*\{", text)
    if not pkg:
        return None
    brace = text.index("{", pkg.start())
    end = find_block_end(text, brace)
    return (text[brace + 1:end], end, True) if end != -1 else None


def inject_endurance_analysis(
    model_text: str, requirements, capacity_mah: Optional[float] = None,
    satisfy_req: str = "REQ-PERF-002",
) -> Tuple[str, bool]:
    """Inject an Automator-evaluable endurance closure that REFERENCES the chosen
    subsystem parts' design attributes (``powerSystem.batteryCapacityMah``,
    ``propulsionSystem.rotorCount`` …): a ``calc def`` + derived ``enduranceMin`` +
    an ``assert constraint`` vs the endurance requirement + ``satisfy``.

    This is the in-model version of emit_endurance_analysis: instead of a self-contained
    fragment with literal numbers, the constraint is wired to the actual variant
    attributes, so the variant params finally participate in a constraint (closes the
    bare-attribute gap on the real DSE product). The inner-BO ``capacity_mah`` is written
    back into the CHOSEN power variant (not just the first one) so it can be referenced.
    Works for both a nested system part def and a flat package assembly (wrapped in a
    new ``DseEnduranceAnalysis`` part def, referencing the package-level part usages).

    Returns (model_text, ok); ok=False (unchanged) if there's no endurance target, no
    assembly, no design-input owners, or the result wouldn't parse."""
    target = endurance_target(list(requirements or []))
    if target <= 0:
        return model_text, False
    text = model_text
    sc = _analysis_scope(text)
    if sc is None:
        return model_text, False
    body, end, wrap = sc
    field_owner = {}                                 # design field -> (usage name, type)
    for uname, utype in _USAGE_RE.findall(body):
        for field in variant_design_inputs(text, utype):
            field_owner.setdefault(field, (uname, utype))
    if not field_owner:
        return model_text, False
    # battery capacity is the inner-BO variable; if the chosen power variant doesn't
    # declare it, write the chosen value INTO that variant's type so it's referenceable.
    if "battery_capacity_mah" not in field_owner and capacity_mah and "battery_cells" in field_owner:
        uname, utype = field_owner["battery_cells"]
        text, ok2 = _inject_attr_into_type(text, utype, "batteryCapacityMah", float(capacity_mah))
        if ok2:
            field_owner["battery_capacity_mah"] = (uname, utype)
            sc = _analysis_scope(text)               # text shifted by the insertion
            if sc is None:
                return model_text, False
            _, end, wrap = sc
    args = []
    for field in _ARG_ORDER:
        if field in field_owner:
            args.append(f"{field_owner[field][0]}.{DESIGN_FIELD_ATTR[field]}")
        else:
            args.append(str(float(DESIGN_DEFAULTS[field])))   # nothing owns it → literal default
    inv = f"Endurance({', '.join(args)})"
    rid = satisfy_req.replace("-", "_")
    lines = [endurance_calc_def(indent="        ")]
    if not re.search(rf"requirement\s+def\s+{rid}\b", text):   # don't redeclare an existing req def
        lines.append(f"        requirement def {rid} {{ attribute target : Real = {target}; }}")
    lines += [
        f"        requirement {rid.lower()} : {rid};",
        f"        attribute enduranceMin : Real = {inv};",
        f"        assert constraint enduranceMeetsReq {{ {inv} >= {target} }}",
        f"        satisfy {rid.lower()};",
    ]
    core = "\n".join(lines)
    note = "    // --- DSE analysis closure (Automator-evaluable; refs chosen variants) ---\n"
    if wrap:                                          # flat package → wrapper part def
        frag = f"\n{note}    part def DseEnduranceAnalysis {{\n{core}\n    }}\n"
    else:                                             # nested → straight into the root body
        frag = f"\n        {note}{core}\n"
    out = text[:end] + frag + text[end:]
    return (out, True) if not check_syntax(out).has_errors else (model_text, False)
