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
    DESIGN_DEFAULTS, DESIGN_FIELD_ATTR, endurance_target, mass_limit, max_rated_payload,
    range_requirement, variant_design_inputs,
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


def mtow_calc_def(indent: str = "    ") -> str:
    """SysML v2 ``calc def Mtow`` mirroring physics_estimator.total_mass_kg (emergent
    all-up mass: frame + propulsion + battery self-mass + payload)."""
    pi = math.pi
    return (
        f"{indent}calc def Mtow {{\n"
        f"{indent}    in capacityMah : Real; in cells : Real; in rotorCount : Real;\n"
        f"{indent}    in rotorRadiusM : Real; in payloadKg : Real;\n"
        f"{indent}    attribute energyWh : Real = capacityMah / 1000.0 * cells * {CELL_V};\n"
        f"{indent}    attribute diskArea : Real = rotorCount * {pi} * (rotorRadiusM ** 2);\n"
        f"{indent}    {BASE_FRAME_KG} + {ROTOR_MASS_COEF} * diskArea + energyWh / {ENERGY_DENSITY_WH_KG} + payloadKg\n"
        f"{indent}}}"
    )


def range_calc_def(indent: str = "    ") -> str:
    """SysML v2 ``calc def RangeM`` mirroring physics_estimator.range_m (endurance ×
    cruise speed). Reuses ``Endurance`` via a nested calc invocation (Automator-evaluable)."""
    return (
        f"{indent}calc def RangeM {{\n"
        f"{indent}    in capacityMah : Real; in cells : Real; in rotorCount : Real;\n"
        f"{indent}    in rotorRadiusM : Real; in payloadKg : Real; in cruiseSpeedMps : Real;\n"
        f"{indent}    Endurance(capacityMah, cells, rotorCount, rotorRadiusM, payloadKg) * 60.0 * cruiseSpeedMps\n"
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
    satisfy_req: str = "REQ-PERF-002", design: Optional[DesignInputs] = None,
) -> Tuple[str, bool]:
    """Inject an Automator-evaluable endurance/MTOW/range closure: ``calc def`` + derived
    ``enduranceMin``/``mtowKg`` + ``assert constraint`` vs the requirement + ``satisfy``.

    If ``design`` (the DSE's recommended DesignInputs) is given, the invocations use its
    AUTHORITATIVE values — the exact design the bilevel search scored and the trade study
    lists — so the closure can't diverge from the optimization (cross-part references are
    fragile when variation points conflict on a field, e.g. propulsion AND airframe both
    declaring rotorCount, or when a field's owner isn't a variation point). Without it,
    falls back to referencing the chosen variants' attributes (legacy path). Either way the
    design's parameters participate in a real, checkable constraint (closes the bare-attr
    gap). Works for a nested system part def and a flat package (wrapped in DseDesignAnalysis).

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
    # design field -> (usage name, type). LAST-wins to match architecture_design's merge
    # (.update): when two variation points declare the same field (e.g. propulsion AND
    # airframe both set rotorCount — a model smell), the closure must reference the SAME
    # owner the DSE actually scored, or the constraint diverges from the optimization.
    field_owner = {}
    for uname, utype in _USAGE_RE.findall(body):
        for field in variant_design_inputs(text, utype):
            field_owner[field] = (uname, utype)
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
    reqs = list(requirements or [])
    # REQ_PERF_002 mandates endurance at the MAXIMUM RATED PAYLOAD — evaluate the analysis
    # at that load (a requirement-driven condition), not the 0.5 kg default or a lighter
    # chosen payload variant. Literal, since it's a worst-case condition, not a design var.
    rated_payload = max_rated_payload(reqs)

    def ref(field: str) -> str:
        if field == "payload_mass_kg" and rated_payload > 0:
            return str(float(rated_payload))
        if field in field_owner:
            return f"{field_owner[field][0]}.{DESIGN_FIELD_ATTR[field]}"
        return str(float(DESIGN_DEFAULTS[field]))     # nothing owns it → literal default

    if design is not None:                            # authoritative: exactly what DSE scored
        pay = rated_payload if rated_payload > 0 else design.payload_mass_kg
        order_vals = {"battery_capacity_mah": design.battery_capacity_mah,
                      "battery_cells": design.battery_cells, "rotor_count": design.rotor_count,
                      "rotor_radius_m": design.rotor_radius_m, "payload_mass_kg": pay}
        base5 = ", ".join(str(float(order_vals[f])) for f in _ARG_ORDER)
        cruise = design.cruise_speed_mps
    else:                                             # legacy: reference chosen variant attrs
        base5 = ", ".join(ref(f) for f in _ARG_ORDER)
        cruise = None

    defs: list = []                                   # calc defs (shared analysis scope)
    decls: list = []                                  # requirement usages
    members: list = []                                # derived attrs + constraints + satisfy
    seen = set(re.findall(r"requirement\s+def\s+(\w+)", text))

    def add_metric(calc_def_src, rid_raw, attr, expr, op, bound, cname):
        rid = rid_raw.replace("-", "_")
        if calc_def_src:
            defs.append(calc_def_src)
        if rid not in seen:                           # don't redeclare an existing req def
            decls.append(f"        requirement def {rid} {{ attribute target : Real = {bound}; }}")
            seen.add(rid)
        decls.append(f"        requirement {rid.lower()} : {rid};")
        members.append(f"        attribute {attr} : Real = {expr};")
        members.append(f"        assert constraint {cname} {{ {expr} {op} {bound} }}")
        members.append(f"        satisfy {rid.lower()};")

    # endurance (perf, >=) — the required trigger
    add_metric(endurance_calc_def(indent="        "), satisfy_req,
               "enduranceMin", f"Endurance({base5})", ">=", target, "enduranceMeetsReq")
    # all-up mass / MTOW (cost, <=) — pick the loosest mass bound = the gross-mass req
    mass_rid, mass_bound = mass_limit(reqs)
    if mass_rid and mass_bound > 0:
        add_metric(mtow_calc_def(indent="        "), mass_rid,
                   "mtowKg", f"Mtow({base5})", "<=", mass_bound, "mtowWithinReq")
    # range (perf, >=) — only a genuine OPERATIONAL-range requirement (not altitude/
    # separation, which share the 'metre' unit), and only if the design has a cruise speed
    range_rid, range_tgt = range_requirement(reqs)
    has_cruise = (cruise is not None and cruise > 0) if design is not None \
        else ("cruise_speed_mps" in field_owner)
    if range_rid and range_tgt > 0 and has_cruise:
        cruise_arg = str(float(cruise)) if design is not None else ref("cruise_speed_mps")
        rinv = f"RangeM({base5}, {cruise_arg})"
        add_metric(range_calc_def(indent="        "), range_rid,
                   "rangeM", rinv, ">=", range_tgt, "rangeMeetsReq")

    core = "\n".join(defs + decls + members)
    note = "    // --- DSE analysis closure (Automator-evaluable; refs chosen variants) ---\n"
    if wrap:                                          # flat package → wrapper part def
        frag = f"\n{note}    part def DseDesignAnalysis {{\n{core}\n    }}\n"
    else:                                             # nested → straight into the root body
        frag = f"\n        {note}{core}\n"
    out = text[:end] + frag + text[end:]
    return (out, True) if not check_syntax(out).has_errors else (model_text, False)


def _sig(d: DesignInputs):
    return (d.battery_capacity_mah, d.battery_cells, d.rotor_count, d.rotor_radius_m,
            d.payload_mass_kg, d.cruise_speed_mps)


def trade_study(alternatives, requirements, recommended: Optional[DesignInputs] = None,
                bindings=None, indent: str = "    ") -> Tuple[str, bool]:
    """An ``analysis def DesignTradeStudy`` comparing the DSE Pareto alternatives. Each
    alternative's endurance/MTOW/(range) is computed by EMBEDDED, Automator-evaluable calc
    defs (mirrors physics_estimator). When ``bindings`` is given (index-aligned list of
    ``{point_id: impl_type_name}``), each alternative is FORMALLY bound to the variant
    definitions it's composed of via a nested ``part alt{i} { part <point> : <Impl>; … }``
    — object-level traceability (alt3 *uses* Hexa_medium + Power_6s), not a comment.
    ``alternatives`` is a list of DesignInputs. Returns (block, ok)."""
    alts = list(alternatives or [])
    if not alts:
        return "", False
    binds = list(bindings or [])
    reqs = list(requirements or [])
    end_tgt = endurance_target(reqs)
    mass_bound = mass_limit(reqs)[1]
    range_tgt = range_requirement(reqs)[1]
    has_range = range_tgt > 0 and any(a.cruise_speed_mps > 0 for a in alts)
    inner = indent + "    "
    rec_sig = _sig(recommended) if recommended is not None else None
    lines = [
        f"{indent}analysis def DesignTradeStudy {{",
        f"{inner}doc /* DSE Pareto front: {len(alts)} non-dominated design(s); each altN is "
        f"bound to the variant impls it uses; metrics by embedded calc defs (== "
        f"physics_estimator). RECOMMENDED = the picked design. */",
        endurance_calc_def(inner),
        mtow_calc_def(inner),
    ]
    if has_range:
        lines.append(range_calc_def(inner))
    for i, a in enumerate(alts):
        tag = " (RECOMMENDED)" if rec_sig is not None and _sig(a) == rec_sig else ""
        five = (f"{float(a.battery_capacity_mah)}, {float(a.battery_cells)}, "
                f"{float(a.rotor_count)}, {a.rotor_radius_m}, {a.payload_mass_kg}")
        b = binds[i] if i < len(binds) else {}
        uses = ", ".join(f"{pid}={impl}" for pid, impl in b.items()) or "—"
        lines.append(f"{inner}// alt{i}{tag}: uses {uses} | cap={a.battery_capacity_mah}mAh "
                     f"payload={a.payload_mass_kg}kg")
        if b:                                    # formal object-level binding to variant defs
            lines.append(f"{inner}part alt{i}Design {{")
            for pid, impl in b.items():
                lines.append(f"{inner}    part {pid} : {impl};")
            lines.append(f"{inner}}}")
        lines.append(f"{inner}attribute alt{i}_enduranceMin : Real = Endurance({five});")
        lines.append(f"{inner}attribute alt{i}_mtowKg : Real = Mtow({five});")
        if has_range:
            lines.append(f"{inner}attribute alt{i}_rangeM : Real = RangeM({five}, {a.cruise_speed_mps});")
        # per-alternative requirement satisfaction: AND of the hard bounds (Automator-
        # evaluable), so the trade study self-evidences which candidates are feasible —
        # the recommendation-vs-feasibility tension becomes a model fact next to (RECOMMENDED).
        clauses = []
        if end_tgt > 0:
            clauses.append(f"alt{i}_enduranceMin >= {end_tgt}")
        if mass_bound > 0:
            clauses.append(f"alt{i}_mtowKg <= {mass_bound}")
        if has_range:
            clauses.append(f"alt{i}_rangeM >= {range_tgt}")
        if clauses:
            lines.append(f"{inner}attribute alt{i}_meetsAllReqs : Boolean = "
                         + " and ".join(clauses) + ";")
    lines.append(f"{indent}}}")
    block = "\n".join(lines)
    # self-validate in a wrapper that stubs the bound impl types (they live in the real
    # model; inject_trade_study re-validates against the full assembly).
    stubs = "".join(f"    part def {impl};\n" for b in binds for impl in set(b.values()))
    ok = not check_syntax(f"package _C {{\n{stubs}{block}\n}}").has_errors
    return (block, True) if ok else ("", False)


def inject_trade_study(model_text: str, alternatives, requirements,
                       recommended: Optional[DesignInputs] = None, bindings=None) -> Tuple[str, bool]:
    """Inject the DesignTradeStudy analysis def at package level. Returns (model_text, ok);
    ok=False (unchanged) if there's nothing to compare or it wouldn't parse."""
    block, ok = trade_study(alternatives, requirements, recommended=recommended, bindings=bindings)
    if not ok:
        return model_text, False
    pkg = re.search(r"\bpackage\s+\w+\s*\{", model_text)
    if not pkg:
        return model_text, False
    end = find_block_end(model_text, model_text.index("{", pkg.start()))
    if end == -1:
        return model_text, False
    out = model_text[:end] + "\n" + block + "\n" + model_text[end:]
    return (out, True) if not check_syntax(out).has_errors else (model_text, False)
