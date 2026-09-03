"""Emit an analysable SysML v2 product-line fragment from a design.

Mirrors the Python physics estimator into a ``calc def`` Syside Automator can
evaluate (operators only, ``**`` for power/root), so the model computes emergent
performance, checks it with ``assert constraint`` and traces it with ``satisfy``.
Constants come from ``physics_estimator``, so the SysML analysis and the Python
DSE/calibration evaluate the same physics.
"""
from __future__ import annotations

import math
import re
from typing import Optional, Tuple

from ..simulation.syntax_checker import check_syntax
from ..utils.sysml_text_utils import PART_DEF_RE, find_block_end, named_block_span
from .domain_objective import (
    DESIGN_DEFAULTS, DESIGN_FIELD_ATTR, endurance_target, mass_limit, max_rated_payload,
    range_requirement, variant_design_inputs,
)
from .requirement_spec import ALTITUDE, RANGE, SPEED, extract_requirements
from .physics_estimator import (
    AVIONICS_POWER_W, BASE_FRAME_KG, CELL_V, DesignInputs, ENERGY_DENSITY_WH_KG,
    ETA_DRIVE, FOM, G, RHO, ROTOR_MASS_COEF, USABLE,
)

# Name of the tool-authored wrapper this emitter injects; validators use it to
# recognise the block as pipeline-owned rather than model content.
ANALYSIS_CLOSURE_DEF_NAME = "DseDesignAnalysis"

_USAGE_RE = re.compile(r"\bpart\s+(\w+)\s*:\s*(\w+)\s*;")
_ARG_ORDER = ("battery_capacity_mah", "battery_cells", "rotor_count",
              "rotor_radius_m", "payload_mass_kg")


def endurance_calc_def(indent: str = "    ") -> str:
    """SysML v2 ``calc def Endurance`` mirroring physics_estimator.endurance_min."""
    pi = math.pi
    return (
        f"{indent}calc def Endurance {{\n"
        f"{indent}    // addedMassKg = delivery payload + onboard component masses (NOT delivery\n"
        f"{indent}    // payload alone, which REQ_FUNC_003 bounds separately).\n"
        f"{indent}    in capacityMah : Real; in cells : Real; in rotorCount : Real;\n"
        f"{indent}    in rotorRadiusM : Real; in addedMassKg : Real;\n"
        f"{indent}    attribute energyWh : Real = capacityMah / 1000.0 * cells * {CELL_V};\n"
        f"{indent}    attribute diskArea : Real = rotorCount * {pi} * (rotorRadiusM ** 2);\n"
        f"{indent}    attribute totalMass : Real = {BASE_FRAME_KG} + {ROTOR_MASS_COEF} * diskArea "
        f"+ energyWh / {ENERGY_DENSITY_WH_KG} + addedMassKg;\n"
        f"{indent}    attribute hoverPower : Real = (totalMass * {G}) ** 1.5 "
        f"/ (((2.0 * {RHO} * diskArea) ** 0.5) * {FOM});\n"
        f"{indent}    attribute elecPower : Real = hoverPower / {ETA_DRIVE} + {AVIONICS_POWER_W};\n"
        f"{indent}    energyWh * {USABLE} / elecPower * 60.0\n"
        f"{indent}}}"
    )


def mtow_calc_def(indent: str = "    ") -> str:
    """SysML v2 ``calc def Mtow`` mirroring physics_estimator.total_mass_kg (emergent all-up mass:
    frame + propulsion + battery self-mass + payload).
    """
    pi = math.pi
    return (
        f"{indent}calc def Mtow {{\n"
        f"{indent}    in capacityMah : Real; in cells : Real; in rotorCount : Real;\n"
        f"{indent}    in rotorRadiusM : Real; in addedMassKg : Real;\n"
        f"{indent}    attribute energyWh : Real = capacityMah / 1000.0 * cells * {CELL_V};\n"
        f"{indent}    attribute diskArea : Real = rotorCount * {pi} * (rotorRadiusM ** 2);\n"
        f"{indent}    {BASE_FRAME_KG} + {ROTOR_MASS_COEF} * diskArea + energyWh / {ENERGY_DENSITY_WH_KG} + addedMassKg\n"
        f"{indent}}}"
    )


def range_calc_def(indent: str = "    ") -> str:
    """SysML v2 ``calc def RangeM`` mirroring physics_estimator.range_m (endurance x cruise speed)."""
    return (
        f"{indent}calc def RangeM {{\n"
        f"{indent}    in capacityMah : Real; in cells : Real; in rotorCount : Real;\n"
        f"{indent}    in rotorRadiusM : Real; in addedMassKg : Real; in cruiseSpeedMps : Real;\n"
        f"{indent}    Endurance(capacityMah, cells, rotorCount, rotorRadiusM, addedMassKg) * 60.0 * cruiseSpeedMps\n"
        f"{indent}}}"
    )


def _invocation(d: DesignInputs) -> str:
    return (f"Endurance({float(d.battery_capacity_mah)}, {float(d.battery_cells)}, "
            f"{float(d.rotor_count)}, {d.rotor_radius_m}, {d.payload_mass_kg})")


def emit_endurance_analysis(
    d: DesignInputs, target_min: float, satisfy_req: str = "REQ-PERF-002",
    package_name: str = "AnalyzedProductLine", part_name: str = "AnalyzedDesign",
) -> Tuple[str, bool]:
    """Analysable SysML fragment: calc def + derived endurance + ``assert constraint``
    against the requirement target + ``satisfy``.
    """
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
    best = None
    for m in PART_DEF_RE.finditer(text):
        brace = m.end() - 1
        end = find_block_end(text, brace)
        if end == -1:
            continue
        n = len(_USAGE_RE.findall(text[brace + 1:end]))
        if best is None or n > best[0]:
            best = (n, m.group(1), brace, end)
    return (best[1], best[2], best[3]) if best and best[0] >= 1 else None


def _inject_attr_into_type(text: str, type_name: str, attr: str, value: float) -> Tuple[str, bool]:
    span = named_block_span(text, "part", type_name)
    if span is None:
        return text, False
    end = span[1]
    return text[:end] + f" attribute {attr} : Real = {value};" + text[end:], True


def _analysis_scope(text: str) -> Optional[Tuple[str, int, bool]]:
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

    With ``design`` (the DSE's recommended DesignInputs) the invocations use its values,
    so the closure cannot diverge from the optimization; cross-part references are fragile
    when two variation points declare the same field, or when a field's owner is not a
    variation point. Without it, falls back to the chosen variants' attributes (legacy
    path). Handles a nested system part def and a flat package (wrapped in
    DseDesignAnalysis). Returns (model_text, ok); ok=False leaves the text unchanged when
    there is no endurance target, no assembly, no design-input owners, or the result would
    not parse.
    """
    target = endurance_target(list(requirements or []))
    if target <= 0:
        return model_text, False
    text = model_text
    sc = _analysis_scope(text)
    if sc is None:
        return model_text, False
    body, end, wrap = sc
    # design field -> (usage name, type). Last-wins, matching architecture_design's
    # .update merge: when two points declare the same field (both setting rotorCount),
    # the closure uses the owner the DSE scored, else the constraint diverges from it.
    field_owner = {}
    for uname, utype in _USAGE_RE.findall(body):
        for field in variant_design_inputs(text, utype):
            field_owner[field] = (uname, utype)
    if not field_owner and design is None:
        return model_text, False
    # battery capacity is the inner-BO variable; if the chosen power variant does not
    # declare it, write the value into that variant's type so it is referenceable.
    # Legacy path only: the bound design path references recommendedDesign.
    if design is None and "battery_capacity_mah" not in field_owner and capacity_mah \
            and "battery_cells" in field_owner:
        uname, utype = field_owner["battery_cells"]
        text, ok2 = _inject_attr_into_type(text, utype, "batteryCapacityMah", float(capacity_mah))
        if ok2:
            field_owner["battery_capacity_mah"] = (uname, utype)
            sc = _analysis_scope(text)               # text shifted by the insertion
            if sc is None:
                return model_text, False
            _, end, wrap = sc
    reqs = list(requirements or [])
    # REQ_PERF_002 mandates endurance at the maximum rated payload, so evaluate
    # at that load rather than the 0.5 kg default. Emitted as a literal because
    # it is a worst-case condition rather than a design variable.
    rated_payload = max_rated_payload(reqs)

    def ref(field: str) -> str:
        if field == "payload_mass_kg" and rated_payload > 0:
            return str(float(rated_payload))
        if field in field_owner:
            return f"{field_owner[field][0]}.{DESIGN_FIELD_ATTR[field]}"
        return str(float(DESIGN_DEFAULTS[field]))

    design_attr_lines = None
    if design is not None:
        # Model-internal binding: emit the scored design point as a `recommendedDesign`
        # part with unified-name attributes and have the analysis reference them
        # (Automator evaluates cross-part refs), so editing a design attribute flows
        # into the analysis. addedMassKg = design.payload_mass_kg already covers the
        # delivery payload + component masses.
        cruise = design.cruise_speed_mps
        design_attr_lines = (
            f"            attribute capacityMah : Real = {float(design.battery_capacity_mah)};\n"
            f"            attribute cells : Real = {float(design.battery_cells)};\n"
            f"            attribute rotorCount : Real = {float(design.rotor_count)};\n"
            f"            attribute rotorRadiusM : Real = {float(design.rotor_radius_m)};\n"
            f"            attribute addedMassKg : Real = {float(design.payload_mass_kg)};\n"
            + (f"            attribute cruiseSpeedMps : Real = {float(cruise)};\n" if cruise else ""))
        base5 = ("recommendedDesign.capacityMah, recommendedDesign.cells, "
                 "recommendedDesign.rotorCount, recommendedDesign.rotorRadiusM, "
                 "recommendedDesign.addedMassKg")
    else:
        base5 = ", ".join(ref(f) for f in _ARG_ORDER)
        cruise = None

    defs: list = []
    decls: list = []
    members: list = []
    satisfied: list = []
    seen = set(re.findall(r"requirement\s+def\s+(\w+)", text))

    def add_metric(calc_def_src, rid_raw, attr, expr, op, bound, cname):
        rid = rid_raw.replace("-", "_")
        if calc_def_src:
            defs.append(calc_def_src)
        if rid not in seen:
            decls.append(f"        requirement def {rid} {{ attribute target : Real = {bound}; }}")
            seen.add(rid)
        decls.append(f"        requirement {rid.lower()} : {rid};")
        members.append(f"        attribute {attr} : Real = {expr};")
        members.append(f"        assert constraint {cname} {{ {expr} {op} {bound} }}")
        # Evidence chain (issue #4): the design element satisfies the requirement and a
        # verification verifies it (objective -> verify), with the assert above as the
        # evaluable evidence.
        satisfied.append(rid.lower())
        # A `verification def` whose objective verifies a sibling requirement usage draws
        # a subsetting-accessibility warning (a definition does not feature its owner's
        # usages); the verification usage form is featured by the owning part, no warning.
        members.append(f"        verification {rid.lower()}_check {{ objective {rid.lower()}_obj "
                       f"{{ verify {rid.lower()}; }} }}")

    add_metric(endurance_calc_def(indent="        "), satisfy_req,
               "enduranceMin", f"Endurance({base5})", ">=", target, "enduranceMeetsReq")
    mass_rid, mass_bound = mass_limit(reqs)
    if mass_rid and mass_bound > 0:
        add_metric(mtow_calc_def(indent="        "), mass_rid,
                   "mtowKg", f"Mtow({base5})", "<=", mass_bound, "mtowWithinReq")
    # range (perf, >=) - operational-range requirements only (altitude/separation
    # share the 'metre' unit), and only when the design has a cruise speed
    range_rid, range_tgt = range_requirement(reqs)
    has_cruise = (cruise is not None and cruise > 0) if design is not None \
        else ("cruise_speed_mps" in field_owner)
    if range_rid and range_tgt > 0 and has_cruise:
        cruise_arg = "recommendedDesign.cruiseSpeedMps" if design is not None else ref("cruise_speed_mps")
        rinv = f"RangeM({base5}, {cruise_arg})"
        add_metric(range_calc_def(indent="        "), range_rid,
                   "rangeM", rinv, ">=", range_tgt, "rangeMeetsReq")

    # Traceability-only structure: quantified requirements the calc set cannot
    # evaluate without assumptions still get a requirement usage + `verification
    # def` whose doc names the tier holding the evidence. No `assert constraint`
    # is emitted: speed needs a drag/thrust model the estimator does not expose,
    # range needs a non-zero cruise speed, altitude is a geofence config bound.
    _TRACE_TIER = {
        SPEED: ("forward_flight tier (lumped momentum, datasheet power caps) "
                "plus L1 param consistency (WPNAV_SPEED)"),
        RANGE: "forward_flight tier (lumped momentum, datasheet power caps)",
        ALTITUDE: "L1 geofence parameter consistency (FENCE_ENABLE / FENCE_ALT_MAX)",
    }
    trace_decls: list = []
    trace_members: list = []
    trace_seen: set = set()
    for spec in extract_requirements(reqs):
        rid = spec.req_id.replace("-", "_")
        tier_note = _TRACE_TIER.get(spec.quantity)
        if tier_note is None or rid.lower() in satisfied or rid in trace_seen:
            continue
        trace_seen.add(rid)
        if rid not in seen:
            trace_decls.append(
                f"        requirement def {rid} {{ attribute target : Real = {spec.value}; }}")
            seen.add(rid)
        trace_decls.append(f"        requirement {rid.lower()} : {rid};")
        trace_members.append(
            f"        verification {rid.lower()}_check {{\n"
            f"            doc /* Traceability: verified at the {tier_note}. The model calc set\n"
            f"               cannot evaluate this quantity without assumed constants, so no\n"
            f"               assert is emitted here; execution evidence lives in the\n"
            f"               verification matrix. */\n"
            f"            objective {rid.lower()}_obj {{ verify {rid.lower()}; }}\n"
            f"        }}")

    if design_attr_lines is not None:                 # bound design point first (refs resolve to it)
        sat = "".join(f"            satisfy {r};\n" for r in satisfied)
        members.insert(0, "        part recommendedDesign {\n" + design_attr_lines + sat + "        }")
    else:
        members += [f"        satisfy {r};" for r in satisfied]

    def _compose(extra_decls: list, extra_members: list) -> str:
        core = "\n".join(defs + decls + extra_decls + members + extra_members)
        note = "    // --- DSE analysis closure (Automator-evaluable; analysis BINDS to recommendedDesign) ---\n"
        # The bound closure owns a `part recommendedDesign`, so it goes inside the
        # `part def DseDesignAnalysis` wrapper (excluded from the reachability graph) and
        # the connectivity refiner does not wire it up as a system component. Only the
        # legacy path (refs into the root scope, no parts) inlines into the root body.
        if wrap or design_attr_lines is not None:
            frag = (
                f"\n{note}    part def {ANALYSIS_CLOSURE_DEF_NAME} "
                f"{{\n{core}\n    }}\n"
            )
        else:
            frag = f"\n        {note}{core}\n"
        return text[:end] + frag + text[end:]

    # Two-stage syntax gate: if the traceability block trips the checker, fall back to
    # the evaluable-only fragment.
    out = _compose(trace_decls, trace_members)
    if (trace_decls or trace_members) and check_syntax(out).has_errors:
        out = _compose([], [])
    return (out, True) if not check_syntax(out).has_errors else (model_text, False)


def _sig(d: DesignInputs):
    return (d.battery_capacity_mah, d.battery_cells, d.rotor_count, d.rotor_radius_m,
            d.payload_mass_kg, d.cruise_speed_mps)


def trade_study(alternatives, requirements, recommended: Optional[DesignInputs] = None,
                bindings=None, indent: str = "    ") -> Tuple[str, bool]:
    """An ``analysis def DesignTradeStudy`` comparing the DSE Pareto alternatives."""
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
                     f"addedMass={a.payload_mass_kg}kg (delivery+components)")
        if b:
            lines.append(f"{inner}part alt{i}Design {{")
            for pid, impl in b.items():
                lines.append(f"{inner}    part {pid} : {impl};")
            lines.append(f"{inner}}}")
        lines.append(f"{inner}attribute alt{i}_enduranceMin : Real = Endurance({five});")
        lines.append(f"{inner}attribute alt{i}_mtowKg : Real = Mtow({five});")
        if has_range:
            lines.append(f"{inner}attribute alt{i}_rangeM : Real = RangeM({five}, {a.cruise_speed_mps});")
        # per-alternative requirement satisfaction: AND of the hard bounds (Automator-
        # evaluable), so the trade study records which candidates are feasible next to
        # (RECOMMENDED).
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
    # self-validate in a wrapper stubbing the bound impl types (they live in the real
    # model; inject_trade_study re-validates against the full assembly).
    stubs = "".join(f"    part def {impl};\n" for b in binds for impl in set(b.values()))
    ok = not check_syntax(f"package _C {{\n{stubs}{block}\n}}").has_errors
    return (block, True) if ok else ("", False)


def inject_trade_study(model_text: str, alternatives, requirements,
                       recommended: Optional[DesignInputs] = None, bindings=None) -> Tuple[str, bool]:
    """Inject the DesignTradeStudy analysis def at package level."""
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
