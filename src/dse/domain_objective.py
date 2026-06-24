"""Domain-aware objective for LLM variation-DSE (requirement-target driven).

Generic design-quality dims can't see a quad-vs-vtol trade-off, so the front
collapsed. This scores each variant against the NUMERIC TARGETS of the
requirements its variation point links to:

  * a requirement like "cruise speed >= 20 m/s" -> (speed, 20)
  * a variant's attribute `cruiseSpeedMps = 25` -> (speed, 25)
  * matched by quantity FAMILY (speed/time/mass/...), satisfaction = value/target

performance = mean satisfaction of the linked requirement targets; cost = the
chosen variants' cost-family attributes (mass/count/power). The objectivity is
in the numbers (requirement targets x variant attributes); only the family
matching is heuristic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..simulation.syntax_checker import check_syntax
from ..utils.sysml_text_utils import find_block_end
from .physics_estimator import DesignInputs, estimate, total_mass_kg
from .requirement_spec import (
    ENDURANCE, MASS_MTOW, PAYLOAD, RANGE, extract_requirements, max_spec, max_value,
)

# quantity family -> substrings that imply it (checked in name + unit, lowercased)
_FAMILY = {
    "speed": ["m/s", "mps", "kph", "km/h", "airspeed", "speed", "velocity", "cruise"],
    # endurance is conventionally minutes/hours; "second" is LATENCY (e.g. a 1.0 s
    # response-time requirement) — a different quantity that must NOT be mistaken for
    # flight endurance, or endurance_target/inner-BO would size for ~1 unit. Seconds are
    # deliberately excluded from this family (they contribute no endurance objective).
    "time": ["endurance", "duration", "hovertime", "flighttime", "minute", "min", "hour"],
    # length UNITS only — a bare length cannot tell operational range from altitude /
    # separation / wingspan, so range TARGETS are extracted text-aware by _range_requirement
    # (positive range phrase, vertical excluded), NOT by unit alone. (Removed the nouns
    # altitude/wingspan/baseline: they are lengths but NOT operational range.)
    "range": ["range", "distance", "km", "meter", "metre"],
    "accuracy": ["accuracy", "precision", "deviation", "resolution", "lines"],
    "mass": ["mass", "weight", "kg", "gram"],
    "count": ["count", "number", "rotor", "motor", "cell", "node", "channel"],
    "power": ["power", "watt", "consumption"],
}
_COST_FAMILIES = {"mass", "count", "power"}
_PERF_FAMILIES = {"speed", "time", "range", "accuracy"}
# Variant-explorable performance families: EMERGENT metrics the estimator derives
# from design inputs (a real trade-off + SITL-calibratable). Settable families
# (speed→WPNAV_SPEED, count→EK3_SRC) are handled directly by L1, NOT as variant
# objective dimensions — no component variant "owns" cruise speed, so making it a
# variant family forces speed_sat≡0 and collapses the front (see memory
# sitl-family-param-mapping). Range needs a cruise-speed input → deferred.
_EMERGENT_PERF = {"time"}

# Canonical attribute name per family, used by the variant generator so the
# emitted SysML attributes parse back to the SAME family (generation ↔ parsing
# single source of truth).  Names are chosen so _family_of resolves each to ONE
# family unambiguously (no cross-family substring, e.g. avoid 'meter' for accuracy).
FAMILY_ATTR = {
    "speed": "speedMps",
    "time": "enduranceMinutes",
    "range": "rangeMeters",
    "accuracy": "precisionPct",
    "mass": "massKg",
    "count": "unitCount",
    "power": "powerWatts",
}

# ── Design-input ontology (single source of truth) ──────────────────────────────────
# Each design input the variants may declare, classified once so every variation-space
# regularization is driven from here (no scattered field lists). The DSE scores designs by
# feeding these through the physics estimator, mirroring what SITL produces — so the static
# ranking can be calibrated.
#   layer   : "outer" = a discrete variant choice (lives in variant defs)
#             "inner" = a continuous variable SIZED by the inner BO (must NOT be pinned in a
#                       variant — stripped for a uniform interface)
#   concern : owning-concern keywords; a field declared by >1 variation point is kept on the
#             point whose name/type matches these (else first declarer). () = no canonical owner
#   req_cond: non-empty → value is a requirement-driven evaluation condition (e.g. payload is
#             evaluated at the maximum rated payload, per REQ_PERF_002), not the variant value
#   family  : quantity family for cost-bound filtering (mass/count/power/speed; "" = none) —
#             the EXACT field→family, so variant-attribute classification no longer relies on
#             _family_of substring matching (which e.g. mis-read "rotorRadiusM" as count).
@dataclass(frozen=True)
class DesignField:
    field: str
    attr: str
    default: float
    layer: str
    concern: Tuple[str, ...] = ()
    req_cond: str = ""
    family: str = ""


DESIGN_ONTOLOGY: Tuple[DesignField, ...] = (
    DesignField("payload_mass_kg", "massKg", 0.5, "outer", ("payload", "cargo"), "max_rated_payload", "mass"),
    DesignField("battery_capacity_mah", "batteryCapacityMah", 5000.0, "inner", ("power", "batter", "energy")),
    DesignField("battery_cells", "batteryCells", 4, "outer", ("power", "batter", "energy"), "", "count"),
    DesignField("rotor_count", "rotorCount", 4, "outer", ("propuls", "rotor", "motor", "prop"), "", "count"),
    DesignField("rotor_radius_m", "rotorRadiusM", 0.13, "outer", ("propuls", "rotor", "motor", "prop")),
    DesignField("cruise_speed_mps", "cruiseSpeedMps", 0.0, "outer", ("propuls", "speed", "cruise"), "", "speed"),
)

# Derived views (kept for existing callers; all sourced from DESIGN_ONTOLOGY)
DESIGN_INPUTS: Tuple[Tuple[str, str], ...] = tuple((d.field, d.attr) for d in DESIGN_ONTOLOGY)
DESIGN_FIELD_ATTR = {d.field: d.attr for d in DESIGN_ONTOLOGY}        # field → SysML attr
_DESIGN_ATTR_FIELD = {d.attr.lower(): d.field for d in DESIGN_ONTOLOGY}  # lower attr → field
DESIGN_DEFAULTS = {d.field: d.default for d in DESIGN_ONTOLOGY}
_FIELD_CONCERN = {d.field: d.concern for d in DESIGN_ONTOLOGY if d.concern}
DESIGN_FIELD_FAMILY = {d.field: d.family for d in DESIGN_ONTOLOGY if d.family}  # field → cost family
_INNER_LOOP_FIELDS = tuple(d.field for d in DESIGN_ONTOLOGY if d.layer == "inner")

_NUM_UNIT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([A-Za-z/%°]+(?:\s*/\s*[A-Za-z]+)?)?")
_ATTR_RE = re.compile(r"\battribute\s+(\w+)\s*(?::\s*\w+)?\s*=\s*([\d.]+)\s*(?:\[\s*'?([^\]']+)'?\s*\])?")
_REQ_ID_RE = re.compile(r"REQ[-_][A-Z]+[-_]\d+")


def _family_of(*tokens: str) -> str:
    blob = " ".join(t.lower() for t in tokens if t)
    for fam, pats in _FAMILY.items():
        if any(p in blob for p in pats):
            return fam
    return ""


def requirement_targets(requirements: List[str]) -> Dict[str, List[Tuple[str, float]]]:
    """{req_id: [(family, target_value), ...]} from requirement text numerics.

    The requirement-id prefix (REQ-PERF-001) is stripped first so its digits are
    not mistaken for targets; a number's family comes from its own UNIT only.
    """
    out: Dict[str, List[Tuple[str, float]]] = {}
    for r in requirements or []:
        m = _REQ_ID_RE.search(r)
        if not m:
            continue
        rid = m.group(0).replace("_", "-")
        body = r.split(":", 1)[1] if ":" in r else r  # drop "REQ-...:" prefix
        targets: List[Tuple[str, float]] = []
        for num, unit in _NUM_UNIT_RE.findall(body):
            fam = _family_of(unit or "")   # family from the number's own unit
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
    """{DesignInputs field: value} for the design-input attributes declared in part
    def ``type_name`` (e.g. massKg → mass_kg). Non-design attributes are ignored."""
    pat = re.compile(rf"\bpart\s+def\s+{re.escape(type_name)}\s*(?::>[^{{]*)?\{{")
    m = pat.search(model_text)
    if not m:
        return {}
    brace = model_text.index("{", m.start())
    end = find_block_end(model_text, brace)
    body = model_text[brace + 1 : end] if end != -1 else ""
    out: Dict[str, float] = {}
    for name, val, _unit in _ATTR_RE.findall(body):
        field = _DESIGN_ATTR_FIELD.get(name.lower())
        if field is not None:
            out[field] = float(val)
    return out


def architecture_design(vps, choices: Dict[str, str], model_text: str) -> DesignInputs:
    """Merge the chosen variants' design inputs into one DesignInputs (DESIGN_DEFAULTS
    fill whatever no variant declares)."""
    merged: Dict[str, float] = dict(DESIGN_DEFAULTS)
    for vp in vps:
        if vp.point_id in choices:
            merged.update(variant_design_inputs(model_text, vp.type_of(choices[vp.point_id])))
    merged["battery_cells"] = int(merged["battery_cells"])
    merged["rotor_count"] = int(merged["rotor_count"])
    return DesignInputs(**merged)  # all-up mass emerges in the estimator


def objective_names(requirements: List[str]) -> List[str]:
    """Objective vector names: one satisfaction per perf family + cost_efficiency."""
    return [f + "_sat" for f in objective_families(requirements)] + ["cost_efficiency"]


def _emergent_for_family(fam: str, metrics: Dict[str, float]) -> float:
    """Map a requirement quantity-family to the estimator's emergent metric."""
    return {
        "speed": metrics.get("cruise_speed_mps", 0.0),
        "time": metrics.get("endurance_min", 0.0),
        "range": metrics.get("range_m", 0.0),
    }.get(fam, 0.0)


def objectives_from_design(di: DesignInputs, vps, choices: Dict[str, str],
                           requirements: List[str]) -> Dict[str, float]:
    """Per-family satisfaction + cost_efficiency from a COMPLETE DesignInputs (battery
    already chosen — by a variant in the single-layer path, or by the inner BO in the
    bilevel path). Split out so both paths share one scoring rule."""
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
    # cost proxy = emergent all-up mass (bigger battery → heavier → costlier)
    obj["cost_efficiency"] = 1.0 / (1.0 + total_mass_kg(di) / 5.0)
    return obj


def design_arch_inputs(di: DesignInputs) -> Dict[str, float]:
    """The NON-capacity design inputs — battery capacity is the inner-BO variable."""
    return {
        "payload_mass_kg": di.payload_mass_kg, "battery_cells": di.battery_cells,
        "rotor_count": di.rotor_count, "rotor_radius_m": di.rotor_radius_m,
        "cruise_speed_mps": di.cruise_speed_mps,
    }


# ── requirement → analysis-metric mapping: now queries over STRUCTURED specs (controlled
# vocabulary), not scattered keyword greps. The brittle disambiguations (altitude≠range,
# payload≠MTOW, second≠endurance) live once in requirement_spec; an LLM does them when
# available, a deterministic rule extractor otherwise. ────────────────────────────────────
def endurance_target(requirements: List[str]) -> float:
    """Endurance requirement target (minutes) for the inner BO; largest ENDURANCE spec. 0 if none."""
    return max_value(extract_requirements(requirements), ENDURANCE)


def max_rated_payload(requirements: List[str]) -> float:
    """Maximum rated payload mass (kg) — the largest PAYLOAD spec. REQ_PERF_002 ties endurance
    to this load, so endurance/MTOW are evaluated here, not at an arbitrary default. 0.0 if none."""
    return max_value(extract_requirements(requirements), PAYLOAD)


def range_requirement(requirements: List[str]) -> Tuple[Optional[str], float]:
    """(req_id, target_metres) for the operational-range requirement (largest RANGE spec);
    (None, 0.0) if none. Altitude/separation/sensor-range are NOT range (classified apart)."""
    s = max_spec(extract_requirements(requirements), RANGE)
    return (s.req_id, s.value) if s else (None, 0.0)


def mass_limit(requirements: List[str]) -> Tuple[Optional[str], float]:
    """(req_id, MTOW limit kg) — the largest MASS_MTOW spec (gross take-off mass), NOT a
    payload sub-bound; (None, 0.0) if no MTOW requirement."""
    s = max_spec(extract_requirements(requirements), MASS_MTOW)
    return (s.req_id, s.value) if s else (None, 0.0)


def _point_fields(model_text: str, point) -> set:
    fields = set()
    for _, vtype in point.variants:
        if vtype:
            fields |= set(variant_design_inputs(model_text, vtype))
    return fields


def _strip_attr_from_type(text: str, type_name: str, attr: str) -> str:
    """Remove `attribute <attr> : <T> = <v>;` from the body of part def <type_name>."""
    m = re.search(rf"\bpart\s+def\s+{re.escape(type_name)}\b[^{{]*\{{", text)
    if not m:
        return text
    brace = text.index("{", m.start())
    end = find_block_end(text, brace)
    if end == -1:
        return text
    body = text[brace + 1:end]
    new_body = re.sub(rf"\s*attribute\s+{re.escape(attr)}\s*:\s*\w+\s*=\s*[^;]+;", "", body)
    return text[:brace + 1] + new_body + text[end:]


def normalize_variation_ownership(model_text: str, points) -> Tuple[str, List[str]]:
    """Deduplicate design-field ownership across variation points (C + A): detect each
    design field declared by >1 variation point, keep it on its canonical owner (concern
    match; else first declarer), and STRIP it from the others' variant type defs so the
    resolved design is coherent and the merge unambiguous. Returns (new_text, notes); the
    notes (A) record every strip and any point left physics-inert. Best-effort: if the
    rewrite wouldn't parse, the original text is returned with an explanatory note."""
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
        kws = _FIELD_CONCERN.get(field, ())
        owners = [p for p in pts
                  if kws and any(k in (p.point_id + " " + " ".join(t for _, t in p.variants)).lower()
                                 for k in kws)]
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

    if notes and check_syntax(text).has_errors:        # never ship an unparsable rewrite
        return model_text, ["variation-ownership normalization skipped (rewrite did not parse)"]
    return text, notes


def strip_inner_loop_attrs(model_text: str, points) -> Tuple[str, List[str]]:
    """Strip inner-loop-variable attributes (e.g. batteryCapacityMah) from every variant
    type, so all variants of a point share ONE consistent design-input interface (only their
    discrete distinguishing attrs, e.g. batteryCells). Returns (new_text, notes); reverts if
    the rewrite wouldn't parse."""
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
    """Single deterministic regularizer for the LLM-declared variation space, all driven by
    DESIGN_ONTOLOGY: (1) dedup field ownership across variation points (keep each field on its
    canonical-concern owner, strip from the rest), then (2) strip inner-loop variables (BO-
    sized, not variant choices) for a uniform per-point interface. One entry point so new
    ontology-classified smells get a single mount point. Returns (text, notes); each sub-pass
    self-reverts if its rewrite wouldn't parse."""
    text, notes = normalize_variation_ownership(model_text, points)
    text, strip_notes = strip_inner_loop_attrs(text, points)
    return text, notes + strip_notes


def evaluation_overrides(requirements: List[str]) -> Dict[str, float]:
    """Requirement-driven evaluation values for design fields whose ontology marks a
    ``req_cond`` — e.g. payload is evaluated at the MAXIMUM RATED PAYLOAD (REQ_PERF_002),
    not the chosen/default variant value. Returns {field: value} (empty if none apply), so
    the DSE/inner-BO/closure all evaluate at the same requirement-mandated conditions."""
    out: Dict[str, float] = {}
    for d in DESIGN_ONTOLOGY:
        if d.req_cond == "max_rated_payload":
            v = max_rated_payload(requirements)
            if v > 0:
                out[d.field] = v
    return out


def within_requirement_bounds(design: Dict[str, float], satisfies: List[str],
                              requirements: List[str]) -> bool:
    """True iff a variant's design inputs respect the COST upper bounds of the
    requirements it SATISFIES — cost families (mass/count/power) must be ≤ the linked
    requirement's upper bound (e.g. a payload variant can't exceed the payload-mass
    limit). PERF families (speed) are NOT filtered: they're settable parameters
    (WPNAV_SPEED, tuned by L1), so a variant's declared cruise speed is not a hard
    constraint. Bounds come from the requirements, not from the LLM."""
    idx = requirement_targets(requirements)
    upper: Dict[str, float] = {}
    for rid in satisfies:
        rid = rid.replace("_", "-")
        for fam, val in idx.get(rid, []):
            if fam in _COST_FAMILIES:
                upper[fam] = min(upper.get(fam, val), val)   # tightest linked upper bound
    for field, v in design.items():
        if not isinstance(v, (int, float)):
            continue
        fam = DESIGN_FIELD_FAMILY.get(str(field).strip().lower())  # exact, ontology-driven
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
