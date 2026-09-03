"""Domain-aware objective for LLM variation-DSE (requirement-target driven).

Generic design-quality dimensions cannot see a quad-vs-vtol trade-off, so the
front collapsed. Each variant is scored against the numeric targets of the
requirements its variation point links to: "cruise speed >= 20 m/s" -> (speed, 20),
a variant attribute `cruiseSpeedMps = 25` -> (speed, 25), matched by quantity
family, satisfaction = value/target. performance = mean satisfaction over the
linked targets; cost = the chosen variants' cost-family attributes
(mass/count/power). Only the family matching is heuristic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..simulation.syntax_checker import check_syntax
from ..utils.sysml_text_utils import PART_DEF_RE, find_block_end, named_block_span
from .physics_estimator import DesignInputs, estimate, total_mass_kg
from .requirement_spec import (
    ENDURANCE, MASS_MTOW, PAYLOAD, RANGE, _NUM_UNIT_RE, extract_requirements,
    max_spec, max_value, min_spec,
)

_FAMILY = {
    "speed": ["m/s", "mps", "kph", "km/h", "airspeed", "speed", "velocity", "cruise"],
    # endurance is minutes/hours; "second" is latency (a 1.0 s response-time
    # requirement), so seconds are excluded from this family - otherwise
    # endurance_target/inner-BO sizes for ~1 unit.
    "time": ["endurance", "duration", "hovertime", "flighttime", "minute", "min", "hour"],
    # length units only - a bare length cannot tell operational range from altitude or
    # wingspan, so range targets come from _range_requirement (positive range phrase,
    # vertical excluded), not from the unit. altitude/wingspan/baseline are excluded.
    "range": ["range", "distance", "km", "meter", "metre"],
    "accuracy": ["accuracy", "precision", "deviation", "resolution", "lines"],
    "mass": ["mass", "weight", "kg", "gram"],
    "count": ["count", "number", "rotor", "motor", "cell", "node", "channel"],
    "power": ["power", "watt", "consumption"],
}
_COST_FAMILIES = {"mass", "count", "power"}
_PERF_FAMILIES = {"speed", "time", "range", "accuracy"}
# Variant-explorable performance families: emergent metrics the estimator derives
# from design inputs. Settable families (speed->WPNAV_SPEED, count->EK3_SRC) are
# handled by L1 instead; no component variant owns cruise speed, so making it a
# variant family forces speed_sat=0 and collapses the front (see memory
# sitl-family-param-mapping). Range needs a cruise-speed input, so it is deferred.
_EMERGENT_PERF = {"time"}

# ── Design-input ontology (single source of truth) ──────────────────────
# Every design input a variant may declare, classified once so all
# variation-space regularization is driven from here. The DSE feeds these
# through the physics estimator, mirroring SITL, so the ranking is calibratable.
#   layer   : "outer" = a discrete variant choice; "inner" = a continuous
#             variable sized by the inner BO (stripped from variants for a
#             uniform interface)
#   concern : owning-concern keywords; a field declared by >1 point stays on the
#             point whose name/type matches (else first declarer). () = none
#   req_cond: non-empty -> the value is a requirement-driven evaluation condition
#             (payload at the maximum rated payload, per REQ_PERF_002)
#   family  : quantity family for cost-bound filtering (mass/count/power/speed;
#             "" = none). The exact field->family, so classification does not
#             fall back to _family_of substring matching (which read
#             "rotorRadiusM" as count).
@dataclass(frozen=True)
class DesignField:
    field: str
    attr: str
    default: float
    layer: str
    concern: Tuple[str, ...] = ()
    req_cond: str = ""
    family: str = ""


@dataclass(frozen=True)
class ObservedDesignAttribute:
    name: str
    value: float
    unit: str
    field: str = ""
    family: str = ""


@dataclass(frozen=True)
class ResolvedDesignAttributes:
    """A model's numeric attributes resolved through the design ontology."""

    attributes: Tuple[ObservedDesignAttribute, ...]
    field_values: Dict[str, float]
    family_values: Dict[str, Tuple[str, float]]


DESIGN_ONTOLOGY: Tuple[DesignField, ...] = (
    DesignField("payload_mass_kg", "massKg", 0.5, "outer", ("payload", "cargo"), "max_rated_payload", "mass"),
    DesignField("battery_capacity_mah", "batteryCapacityMah", 5000.0, "inner", ("power", "batter", "energy")),
    DesignField("battery_cells", "batteryCells", 4, "outer", ("power", "batter", "energy"), "", "count"),
    DesignField("rotor_count", "rotorCount", 4, "outer", ("propuls", "rotor", "motor", "prop"), "", "count"),
    DesignField("rotor_radius_m", "rotorRadiusM", 0.13, "outer", ("propuls", "rotor", "motor", "prop")),
    DesignField("cruise_speed_mps", "cruiseSpeedMps", 0.0, "outer", ("propuls", "speed", "cruise"), "", "speed"),
)

DESIGN_INPUTS: Tuple[Tuple[str, str], ...] = tuple((d.field, d.attr) for d in DESIGN_ONTOLOGY)
DESIGN_FIELD_ATTR = {d.field: d.attr for d in DESIGN_ONTOLOGY}
_DESIGN_ATTR_FIELD = {d.attr.lower(): d.field for d in DESIGN_ONTOLOGY}
DESIGN_DEFAULTS = {d.field: d.default for d in DESIGN_ONTOLOGY}
_FIELD_CONCERN = {d.field: d.concern for d in DESIGN_ONTOLOGY if d.concern}
DESIGN_FIELD_FAMILY = {d.field: d.family for d in DESIGN_ONTOLOGY if d.family}
_INNER_LOOP_FIELDS = tuple(d.field for d in DESIGN_ONTOLOGY if d.layer == "inner")

# `attribute <name> [: <Type>[::<Type>][ [unit] ]] = <number> [ [unit] ];`
# The type may be qualified (`ISQ::LengthValue`, written by the A/G emitter)
# and may carry a unit suffix; a pattern accepting only a bare single-word
# type skipped both.
_ATTR_RE = re.compile(
    r"\battribute\s+(\w+)"
    r"(?:\s*:\s*\w+(?:::\w+)*(?:\s*\[[^\]]*\])?)?"
    r"\s*=\s*([\d.]+)\s*(?:\[\s*'?([^\]']+)'?\s*\])?"
)
_REQ_ID_RE = re.compile(r"REQ[-_][A-Z]+[-_]\d+")

# ── Committed-design resolution (single source for read and write) ──────────
# A resolved model keeps every variant `part def X :> Base { ... }` (the Pareto
# alternatives the trade study references) plus one binding
# `part <usage> : <ChosenImpl>;` per variation point. The bindings define the
# committed design; an unbound specialised def is documented alternative space.
# Flat text scans cannot tell the two apart and reported the first-declared
# alternative as committed, so these helpers resolve the bindings instead.
_PART_USAGE_RE = re.compile(
    r"\bpart\s+(?!def\b)(\w+)\s*:\s*(\w+)\s*(?:\[[^\]]*\]\s*)?[;{]"
)
# Blocks whose part usages are not assembly bindings: the trade study binds each
# Pareto alternative as `part altN { part <point> : <Impl>; }` inside its
# `analysis def`, and calc/verification defs may declare attribute-shaped params.
_NON_ASSEMBLY_BLOCK_RE = re.compile(
    r"\b(?:analysis|calc|verification)\s+def\s+\w+[^{;]*\{"
)


def _non_assembly_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for m in _NON_ASSEMBLY_BLOCK_RE.finditer(text):
        end = find_block_end(text, m.end() - 1)
        if end != -1:
            spans.append((m.end() - 1, end))
    return spans


def _inside(pos: int, spans: List[Tuple[int, int]]) -> bool:
    return any(s < pos < e for s, e in spans)


def committed_bindings(model_text: str) -> List[Tuple[str, str]]:
    """Ordered ``(usage_name, type_name)`` assembly bindings of *model_text*.

    Scans ``part <name> : <Type>;`` usages outside analysis/calc/verification blocks,
    whose nested usages cite alternatives rather than the committed system. Order is
    document order; a last-wins merge is the caller's job.
    """
    text = str(model_text or "")
    masked = _non_assembly_spans(text)
    return [
        (m.group(1), m.group(2))
        for m in _PART_USAGE_RE.finditer(text)
        if not _inside(m.start(), masked)
    ]


def _unbound_variant_spans(text: str, bound_types: set) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for m in PART_DEF_RE.finditer(text):
        if m.group(1) in bound_types:
            continue
        if ":>" not in text[m.start():m.end()]:
            continue  # unspecialised def: base type, not an alternative
        end = find_block_end(text, m.end() - 1)
        if end != -1:
            spans.append((m.end() - 1, end))
    return spans


def _family_of(*tokens: str) -> str:
    blob = " ".join(t.lower() for t in tokens if t)
    for fam, pats in _FAMILY.items():
        if any(p in blob for p in pats):
            return fam
    return ""


def resolve_design_attributes(model_text: str) -> ResolvedDesignAttributes:
    """Apply parsing, defaults, field identity and family rules in one place.

    Downstream modules consume this result instead of the ontology's regexes, private
    maps, case rules or fallback classifier. Binding-aware: when the model declares
    assembly bindings, attributes inside an unbound specialised variant def (a retained
    Pareto alternative) and inside analysis/calc/verification blocks are excluded, since
    the bindings select the committed design. A model with no bindings (bare
    declarations, fixtures) keeps the flat scan.
    """
    text = str(model_text or "")
    bindings = committed_bindings(text)
    excluded: List[Tuple[int, int]] = []
    if bindings:
        bound_types = {t for _, t in bindings}
        excluded = (
            _unbound_variant_spans(text, bound_types)
            + _non_assembly_spans(text)
        )
    attributes: list[ObservedDesignAttribute] = []
    field_values: Dict[str, float] = dict(DESIGN_DEFAULTS)
    family_values: Dict[str, Tuple[str, float]] = {}
    seen_fields: set[str] = set()
    for match in _ATTR_RE.finditer(text):
        if excluded and _inside(match.start(), excluded):
            continue
        name, raw_value, unit = match.groups()
        value = float(raw_value)
        field = _DESIGN_ATTR_FIELD.get(name.lower(), "")
        family = DESIGN_FIELD_FAMILY.get(field, "") or _family_of(
            name, unit or ""
        )
        attributes.append(ObservedDesignAttribute(
            name=name,
            value=value,
            unit=unit or "",
            field=field,
            family=family,
        ))
        if field and field not in seen_fields:
            field_values[field] = value
            seen_fields.add(field)
        if family and family not in family_values:
            family_values[family] = (name, value)
    return ResolvedDesignAttributes(
        attributes=tuple(attributes),
        field_values=field_values,
        family_values=family_values,
    )


def requirement_targets(requirements: List[str]) -> Dict[str, List[Tuple[str, float]]]:
    """{req_id: [(family, target_value), ...]} from requirement text numerics.

    The requirement-id prefix (REQ-PERF-001) is stripped first so its digits are not
    read as targets; a number's family comes from its own unit.
    """
    out: Dict[str, List[Tuple[str, float]]] = {}
    for r in requirements or []:
        m = _REQ_ID_RE.search(r)
        if not m:
            continue
        rid = m.group(0).replace("_", "-")
        body = r.split(":", 1)[1] if ":" in r else r
        targets: List[Tuple[str, float]] = []
        for num, unit in _NUM_UNIT_RE.findall(body):
            fam = _family_of(unit or "")
            if fam:
                targets.append((fam, float(num)))
        if targets:
            out[rid] = targets
    return out


def objective_families(requirements: List[str]) -> List[str]:
    """Performance quantity-families present across the requirements (sorted)."""
    fams = set()
    for targets in requirement_targets(requirements).values():
        for fam, _ in targets:
            if fam in _EMERGENT_PERF:
                fams.add(fam)
    return sorted(fams)


def variant_design_inputs(model_text: str, type_name: str) -> Dict[str, float]:
    """{DesignInputs field: value} for the design-input attributes declared in part def ``type_name``
    (e.g. massKg -> mass_kg).
    """
    span = named_block_span(model_text, "part", type_name)
    if span is None:
        return {}
    body = model_text[span[0] + 1:span[1]]
    out: Dict[str, float] = {}
    for name, val, _unit in _ATTR_RE.findall(body):
        field = _DESIGN_ATTR_FIELD.get(name.lower())
        if field is not None:
            out[field] = float(val)
    return out


def architecture_design(vps, choices: Dict[str, str], model_text: str) -> DesignInputs:
    """Merge the chosen variants' design inputs into one DesignInputs (DESIGN_DEFAULTS fill whatever no
    variant declares).
    """
    merged: Dict[str, float] = dict(DESIGN_DEFAULTS)
    for vp in vps:
        if vp.point_id in choices:
            merged.update(variant_design_inputs(model_text, vp.type_of(choices[vp.point_id])))
    merged["battery_cells"] = int(merged["battery_cells"])
    merged["rotor_count"] = int(merged["rotor_count"])
    return DesignInputs(**merged)


def objective_names(requirements: List[str]) -> List[str]:
    """Objective vector names: one satisfaction per perf family + cost_efficiency."""
    return [f + "_sat" for f in objective_families(requirements)] + ["cost_efficiency"]


def _emergent_for_family(fam: str, metrics: Dict[str, float]) -> float:
    return {
        "speed": metrics.get("cruise_speed_mps", 0.0),
        "time": metrics.get("endurance_min", 0.0),
        "range": metrics.get("range_m", 0.0),
    }.get(fam, 0.0)


def objectives_from_design(di: DesignInputs, vps, choices: Dict[str, str],
                           requirements: List[str]) -> Dict[str, float]:
    """Per-family satisfaction + cost_efficiency from a complete DesignInputs (battery
    already chosen by a variant in the single-layer path, or by the inner BO in the
    bilevel path). Split out so both paths share one scoring rule.
    """
    metrics = estimate(di)
    req_index = requirement_targets(requirements)
    fams = objective_families(requirements)
    fam_sat: Dict[str, List[float]] = {f: [] for f in fams}
    for vp in vps:
        if vp.point_id not in choices:
            continue
        for rid in vp.requirements:
            rid = rid.replace("_", "-")
            for fam, target in req_index.get(rid, []):
                if fam in fam_sat and target > 0:
                    fam_sat[fam].append(min(1.0, _emergent_for_family(fam, metrics) / target))
    obj: Dict[str, float] = {
        f + "_sat": (sum(v) / len(v) if v else 0.0) for f, v in fam_sat.items()
    }
    # cost proxy = emergent all-up mass (bigger battery -> heavier -> costlier)
    obj["cost_efficiency"] = 1.0 / (1.0 + total_mass_kg(di) / 5.0)
    return obj


def design_arch_inputs(di: DesignInputs) -> Dict[str, float]:
    """The non-capacity design inputs; battery capacity is the inner-BO variable."""
    return {
        "payload_mass_kg": di.payload_mass_kg, "battery_cells": di.battery_cells,
        "rotor_count": di.rotor_count, "rotor_radius_m": di.rotor_radius_m,
        "cruise_speed_mps": di.cruise_speed_mps,
    }


# ── requirement -> analysis-metric mapping: queries structured specs (controlled
# vocabulary) rather than keyword greps. The disambiguations (altitude!=range,
# payload!=MTOW, second!=endurance) live once in requirement_spec, done by an LLM
# when available and by a rule extractor otherwise. ──────────────────────
def endurance_target(requirements: List[str]) -> float:
    """Endurance requirement target (minutes) for the inner BO; largest ">=" ENDURANCE spec."""
    return max_value(extract_requirements(requirements), ENDURANCE, ">=")


def max_rated_payload(requirements: List[str]) -> float:
    """Maximum rated payload mass (kg) - the largest PAYLOAD spec."""
    return max_value(extract_requirements(requirements), PAYLOAD)


def range_requirement(requirements: List[str]) -> Tuple[Optional[str], float]:
    """(req_id, target_metres) for a range capability requirement - the largest ">=" RANGE
    spec. A "<=" range (operational-radius / geofence limit) is a bound, not a capability
    RangeM satisfies, so it is excluded. (None, 0.0) if none.
    """
    s = max_spec(extract_requirements(requirements), RANGE, ">=")
    return (s.req_id, s.value) if s else (None, 0.0)


def mass_limit(requirements: List[str]) -> Tuple[Optional[str], float]:
    """(req_id, MTOW limit kg) - the tightest "<=" MASS_MTOW spec (gross take-off mass, not
    a payload sub-bound); (None, 0.0) if none. A conjunction of upper bounds is governed
    by its minimum: taking the largest admitted designs that violated the stricter one.
    """
    s = min_spec(extract_requirements(requirements), MASS_MTOW, "<=")
    return (s.req_id, s.value) if s else (None, 0.0)


def _point_fields(model_text: str, point) -> set:
    fields = set()
    for _, vtype in point.variants:
        if vtype:
            fields |= set(variant_design_inputs(model_text, vtype))
    return fields


def _strip_attr_from_type(text: str, type_name: str, attr: str) -> str:
    span = named_block_span(text, "part", type_name)
    if span is None:
        return text
    brace, end = span
    body = text[brace + 1:end]
    new_body = re.sub(rf"\s*attribute\s+{re.escape(attr)}\s*:\s*\w+\s*=\s*[^;]+;", "", body)
    return text[:brace + 1] + new_body + text[end:]


def normalize_variation_ownership(model_text: str, points) -> Tuple[str, List[str]]:
    """Deduplicate design-field ownership across variation points (C + A): a field declared by
    more than one point stays on its canonical owner (concern match, else first declarer)
    and is stripped from the other variant type defs, so the merge is unambiguous.
    """
    text = model_text
    notes: List[str] = []
    field_pts: Dict[str, list] = {}
    for p in points:
        for f in _point_fields(text, p):
            field_pts.setdefault(f, []).append(p)

    stripped_pts: set = set()
    for field, pts in field_pts.items():
        if len(pts) < 2:
            continue
        # A catalog seed encodes a coupled architecture tuple. Moving one of its
        # fields (cells especially) to an independent LLM point by concern ownership
        # creates rotor/prop/voltage cross-products the catalog never asserted, so
        # the seed owns every architecture field it declares; other LLM fields keep
        # the ontology-concern rule.
        seed_owners = [
            p for p in pts
            if "catalog architecture seed" in p.rationale.lower()
        ]
        if len(seed_owners) == 1:
            owner = seed_owners[0]
            why = f"{field} → mandatory coupled catalog architecture seed"
        else:
            kws = _FIELD_CONCERN.get(field, ())
            owners = [p for p in pts
                      if kws and any(
                          k in (p.point_id + " " + " ".join(t for _, t in p.variants)).lower()
                          for k in kws
                      )]
            if len(owners) == 1:
                owner, why = owners[0], f"{field} → {owners[0].point_id} concern"
            else:
                owner, why = pts[0], "first declarer (no/ambiguous concern match)"
        attr = DESIGN_FIELD_ATTR[field]
        for p in pts:
            if p is owner:
                continue
            for _, vtype in p.variants:
                if vtype and field in variant_design_inputs(text, vtype):
                    text = _strip_attr_from_type(text, vtype, attr)
                    stripped_pts.add(p.point_id)
        notes.append(f"design field '{field}' declared by {sorted({p.point_id for p in pts})}; "
                     f"kept in '{owner.point_id}' ({why}), stripped from "
                     f"{sorted({p.point_id for p in pts if p is not owner})}")

    for p in points:
        if p.point_id in stripped_pts and not _point_fields(text, p):
            notes.append(f"variation point '{p.point_id}' is now physics-inert "
                         f"(structural-only) after deduplication")

    if notes and check_syntax(text).has_errors:
        return model_text, ["variation-ownership normalization skipped (rewrite did not parse)"]
    return text, notes


def strip_inner_loop_attrs(model_text: str, points) -> Tuple[str, List[str]]:
    """Strip inner-loop-variable attributes (batteryCapacityMah) from every variant type,
    so all variants of a point share one design-input interface and keep only their
    discrete distinguishing attributes (batteryCells).
    """
    text = model_text
    stripped: Dict[str, list] = {}
    for p in points:
        for _, vtype in p.variants:
            if not vtype:
                continue
            present = variant_design_inputs(text, vtype)
            for field in _INNER_LOOP_FIELDS:
                if field in present:
                    text = _strip_attr_from_type(text, vtype, DESIGN_FIELD_ATTR[field])
                    stripped.setdefault(field, []).append(vtype)
    notes = [f"stripped inner-loop attribute '{DESIGN_FIELD_ATTR[f]}' from {sorted(set(v))} "
             f"(it is the inner-BO variable, sized per design — not a fixed variant attribute; "
             f"uniform variant interface)" for f, v in stripped.items()]
    if notes and check_syntax(text).has_errors:
        return model_text, ["inner-loop attr strip skipped (rewrite did not parse)"]
    return text, notes


def normalize_variation_space(model_text: str, points) -> Tuple[str, List[str]]:
    """Deterministic regularizer for the LLM-declared variation space, driven by
    DESIGN_ONTOLOGY: (1) dedup field ownership across variation points (each field on its
    canonical-concern owner, stripped from the rest), then (2) strip inner-loop variables
    (BO-sized, not variant choices) for a uniform per-point interface.
    """
    text, notes = normalize_variation_ownership(model_text, points)
    text, strip_notes = strip_inner_loop_attrs(text, points)
    return text, notes + strip_notes


def evaluation_overrides(requirements: List[str]) -> Dict[str, float]:
    """Requirement-driven evaluation values for design fields whose ontology marks a
    ``req_cond`` - payload is evaluated at the maximum rated payload (REQ_PERF_002) rather
    than the chosen/default variant value. Returns {field: value}, empty if none apply, so
    the DSE, inner BO and closure evaluate at the same conditions.
    """
    out: Dict[str, float] = {}
    for d in DESIGN_ONTOLOGY:
        if d.req_cond == "max_rated_payload":
            v = max_rated_payload(requirements)
            if v > 0:
                out[d.field] = v
    return out


def within_requirement_bounds(design: Dict[str, float], satisfies: List[str],
                              requirements: List[str]) -> bool:
    """True iff a variant's design inputs respect the cost upper bounds of the
    requirements it satisfies.

    Cost families (mass/count/power) stay <= the linked requirement's upper bound. Perf
    families (speed) are not filtered: they are settable parameters (WPNAV_SPEED, tuned by
    L1), so a declared cruise speed is not a hard constraint. Bounds come from the
    requirements, not from the LLM.
    """
    idx = requirement_targets(requirements)
    upper: Dict[str, float] = {}
    for rid in satisfies:
        rid = rid.replace("_", "-")
        for fam, val in idx.get(rid, []):
            if fam in _COST_FAMILIES:
                upper[fam] = min(upper.get(fam, val), val)
    for field, v in design.items():
        if not isinstance(v, (int, float)):
            continue
        fam = DESIGN_FIELD_FAMILY.get(str(field).strip().lower())
        if fam in upper and v > upper[fam]:
            return False
    return True


def architecture_objectives(vps, choices: Dict[str, str], model_text: str,
                            requirements: List[str]) -> Dict[str, float]:
    """Single-layer scoring: battery taken from the resolved model. Kept for the
    non-bilevel callers/tests; the bilevel path uses objectives_from_design with an
    inner-optimized capacity."""
    return objectives_from_design(
        architecture_design(vps, choices, model_text), vps, choices, requirements
    )
