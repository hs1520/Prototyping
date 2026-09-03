"""Authoritative projection from DSE design facts to ArduPilot parameters."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import List, Mapping, Tuple

from ..dse.domain_objective import resolve_design_attributes
from ..dse.physics_estimator import DesignInputs, estimate


@dataclass(frozen=True, slots=True)
class ParameterRule:
    family: str
    name: str
    factor: float
    unit: str
    lo: float
    hi: float


@dataclass(frozen=True)
class ParameterCheck:
    family: str
    source_attr: str
    param_name: str
    param_value: float
    si_value: float
    ok: bool
    message: str


@dataclass(frozen=True, slots=True)
class ParameterProjection:
    design_inputs: DesignInputs
    parm_lines: tuple[str, ...]
    predicted: Mapping[str, float]
    caveat: str


SETTABLE_PARAMETER_RULES: Mapping[str, ParameterRule] = MappingProxyType({
    "speed": ParameterRule(
        family="speed",
        name="WPNAV_SPEED",
        factor=100.0,
        unit="cm/s",
        lo=20.0,
        hi=2000.0,
    ),
})

FRAME_CLASS_BY_ROTOR_COUNT: Mapping[int, int] = MappingProxyType(
    {4: 1, 6: 2, 8: 3}
)


def design_inputs_from_model(model_text: str) -> DesignInputs:
    """Resolve model attributes and fill absent design fields with defaults."""
    merged = dict(resolve_design_attributes(model_text).field_values)
    merged["battery_cells"] = int(merged["battery_cells"])
    merged["rotor_count"] = int(merged["rotor_count"])
    return DesignInputs(**merged)


def check_settable_parameters(model_text: str) -> List[ParameterCheck]:
    """Compile and range-check all supported settable design families."""
    found = dict(resolve_design_attributes(model_text).family_values)
    results: List[ParameterCheck] = []
    for family, (source_attr, si_value) in found.items():
        rule = SETTABLE_PARAMETER_RULES.get(family)
        if rule is None:
            continue
        value = round(si_value * rule.factor, 4)
        in_range = rule.lo <= value <= rule.hi
        message = (
            f"{rule.name}={value} {rule.unit} (from {source_attr}={si_value})"
            if in_range
            else f"{rule.name}={value} {rule.unit} out of range "
            f"[{rule.lo}, {rule.hi}]"
        )
        results.append(
            ParameterCheck(
                family=family,
                source_attr=source_attr,
                param_name=rule.name,
                param_value=value,
                si_value=si_value,
                ok=in_range,
                message=message,
            )
        )
    return results


def validate_settable_parameters(
    model_text: str,
) -> Tuple[bool, List[ParameterCheck]]:
    checks = check_settable_parameters(model_text)
    return all(item.ok for item in checks), checks


def settable_parm_lines(model_text: str) -> List[str]:
    return [
        f"{item.param_name:<20} {item.param_value}"
        for item in check_settable_parameters(model_text)
        if item.ok
    ]


def design_parm_lines(
    design: DesignInputs,
    *,
    base_params: Mapping[str, object] | None = None,
) -> List[str]:
    """Compile full design-input parameters and merge profile defaults once."""
    lines = [f"{'BATT_CAPACITY':<20} {design.battery_capacity_mah:.0f}"]
    frame_class = FRAME_CLASS_BY_ROTOR_COUNT.get(design.rotor_count)
    if frame_class is not None:
        lines.append(f"{'FRAME_CLASS':<20} {frame_class}")
    if design.cruise_speed_mps > 0:
        lines.append(f"{'WPNAV_SPEED':<20} {design.cruise_speed_mps * 100:.0f}")
    present = _parameter_names(lines)
    for name, value in (base_params or {}).items():
        if name not in present:
            lines.append(f"{name:<20} {value}")
    return lines


def merge_base_parameters(
    parm_content: str,
    base_params: Mapping[str, object],
) -> str:
    """Append missing profile parameters using SITLBridge's stable format."""
    additions = [
        f"{name:<30} {value}  # base SITL param"
        for name, value in base_params.items()
        if name not in _parameter_names(parm_content.splitlines())
    ]
    if not additions:
        return parm_content
    return parm_content + "\n" + "\n".join(additions) + "\n"


def project_model_parameters(
    model_text: str,
    *,
    base_params: Mapping[str, object] | None = None,
) -> ParameterProjection:
    design = design_inputs_from_model(model_text)
    predicted = estimate(design)
    return ParameterProjection(
        design_inputs=design,
        parm_lines=tuple(design_parm_lines(design, base_params=base_params)),
        predicted=MappingProxyType(dict(predicted)),
        caveat=(
            "all-up mass is emergent ({:.2f} kg) and NOT set in SITL "
            "(fixed frame model) — calibration will surface the gap"
        ).format(predicted.get("total_mass_kg", 0.0)),
    )


def _parameter_names(lines: List[str]) -> set[str]:
    names: set[str] = set()
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            names.add(stripped.split()[0])
    return names
