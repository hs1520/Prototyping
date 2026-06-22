"""Layer 4 of the DSE→SITL plan: map a DSE-recommended model's DESIGN INPUTS to an
ArduPilot SITL configuration (.parm), plus the estimator's PREDICTED emergent metrics
that L2 real flight will be checked against (the calibration baseline).

Settable in SITL:
  * battery_capacity_mah → BATT_CAPACITY  (drives simulated endurance)
  * rotor_count          → FRAME_CLASS    (quad/hexa/octa)
  * cruise_speed_mps     → WPNAV_SPEED    (cm/s)

NOT settable: airframe MASS. ArduCopter SITL mass is fixed by the frame model, so the
estimator's EMERGENT all-up mass (frame+battery+payload) cannot be reproduced exactly
in SITL. This is a deliberate low-vs-high-fidelity gap the calibration (layer 6) will
expose — not something to fake with a bogus parameter (see memory sitl-family-param-mapping).
"""
from __future__ import annotations

from typing import Dict, List

from ..dse.domain_objective import DESIGN_DEFAULTS, _ATTR_RE, _DESIGN_ATTR_FIELD
from ..dse.physics_estimator import DesignInputs, estimate

# rotor count → ArduCopter FRAME_CLASS
FRAME_CLASS: Dict[int, int] = {4: 1, 6: 2, 8: 3}  # quad / hexa / octa


def model_design_inputs(model_text: str) -> DesignInputs:
    """Merge the design-input attributes found in *model_text* into a DesignInputs
    (DESIGN_DEFAULTS fill anything absent). First value of each field wins."""
    merged: Dict[str, float] = dict(DESIGN_DEFAULTS)
    seen: set = set()
    for name, val, _unit in _ATTR_RE.findall(model_text):
        field = _DESIGN_ATTR_FIELD.get(name.lower())
        if field is not None and field not in seen:
            merged[field] = float(val)
            seen.add(field)
    merged["battery_cells"] = int(merged["battery_cells"])
    merged["rotor_count"] = int(merged["rotor_count"])
    return DesignInputs(**merged)


def design_to_sitl_parm(d: DesignInputs) -> List[str]:
    """ArduPilot .parm lines for the SITL-settable design inputs."""
    lines = [f"{'BATT_CAPACITY':<20} {d.battery_capacity_mah:.0f}"]
    fc = FRAME_CLASS.get(d.rotor_count)
    if fc is not None:
        lines.append(f"{'FRAME_CLASS':<20} {fc}")
    if d.cruise_speed_mps > 0:
        lines.append(f"{'WPNAV_SPEED':<20} {d.cruise_speed_mps * 100:.0f}")
    return lines


def predicted_emergent(d: DesignInputs) -> Dict[str, float]:
    """The estimator's predicted emergent metrics — the baseline L2 real flight is
    compared against in calibration (layer 6)."""
    return estimate(d)


def sitl_plan(model_text: str) -> Dict[str, object]:
    """Full layer-4 artifact for a DSE-recommended model: SITL .parm + predicted
    emergent baseline + the mass-fidelity caveat."""
    d = model_design_inputs(model_text)
    return {
        "design_inputs": d,
        "sitl_parm": design_to_sitl_parm(d),
        "predicted": predicted_emergent(d),
        "caveat": "all-up mass is emergent ({:.2f} kg) and NOT set in SITL (fixed frame "
                  "model) — calibration will surface the gap".format(
                      predicted_emergent(d).get("total_mass_kg", 0.0)),
    }
