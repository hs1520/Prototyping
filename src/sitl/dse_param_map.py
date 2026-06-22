"""Layer 1 of the DSE→SITL plan: map a DSE-recommended model's SETTABLE quantity
families to ArduPilot .parm entries, with static L1 validation (no SITL launch).

Only SETTABLE families are here — those that correspond to a real ArduPilot parameter.
EMERGENT families (mass / endurance / range) cannot be set as parameters; they are
physical outcomes measured in L2 instead. See memory `sitl-family-param-mapping`.

The settable-family value (in SI, read off the resolved model's variant attributes)
is converted to the parameter's native unit and range-checked against the parameter's
documented limits. Each result is traceable to the source attribute and the family.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..dse.domain_objective import _ATTR_RE, _family_of


@dataclass(frozen=True)
class ParamSpec:
    """An ArduPilot parameter a settable family maps to."""
    name: str
    factor: float       # SI family value × factor → parameter's native unit
    unit: str
    lo: float           # documented inclusive lower bound (native unit)
    hi: float           # documented inclusive upper bound (native unit)


# SETTABLE families only. Emergent families intentionally absent (see module doc).
FAMILY_TO_PARAM: Dict[str, ParamSpec] = {
    # cruise/waypoint speed: model carries m/s, ArduPilot WPNAV_SPEED is cm/s.
    "speed": ParamSpec("WPNAV_SPEED", factor=100.0, unit="cm/s", lo=20.0, hi=2000.0),
}


@dataclass(frozen=True)
class L1Result:
    family: str
    source_attr: str        # the model attribute the value came from
    param_name: str
    param_value: float
    si_value: float
    ok: bool
    message: str


def _model_family_values(model_text: str) -> Dict[str, Tuple[str, float]]:
    """{family: (source_attr_name, si_value)} for numeric attributes in the model.
    First attribute of each family wins (resolved models carry the chosen variant's)."""
    out: Dict[str, Tuple[str, float]] = {}
    for name, val, unit in _ATTR_RE.findall(model_text):
        fam = _family_of(name, unit or "")
        if fam and fam not in out:
            out[fam] = (name, float(val))
    return out


def map_settable_params(model_text: str) -> List[L1Result]:
    """Map the model's settable families to ArduPilot params + range-check them (L1).

    Emergent families present in the model are skipped (they belong to L2)."""
    found = _model_family_values(model_text)
    results: List[L1Result] = []
    for fam, (attr, si) in found.items():
        spec = FAMILY_TO_PARAM.get(fam)
        if spec is None:
            continue  # emergent / unsupported family → not a settable param
        value = round(si * spec.factor, 4)
        in_range = spec.lo <= value <= spec.hi
        msg = (
            f"{spec.name}={value} {spec.unit} (from {attr}={si})"
            if in_range
            else f"{spec.name}={value} {spec.unit} out of range [{spec.lo}, {spec.hi}]"
        )
        results.append(L1Result(fam, attr, spec.name, value, si, in_range, msg))
    return results


def generate_parm_lines(model_text: str) -> List[str]:
    """ArduPilot .parm lines for the model's in-range settable params."""
    return [
        f"{r.param_name:<20} {r.param_value}"
        for r in map_settable_params(model_text)
        if r.ok
    ]


def validate_l1(model_text: str) -> Tuple[bool, List[L1Result]]:
    """Static L1: True iff every mapped settable param is within range. No SITL."""
    results = map_settable_params(model_text)
    return (all(r.ok for r in results), results)
