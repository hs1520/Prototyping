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
from typing import Tuple

from ..simulation.syntax_checker import check_syntax
from .physics_estimator import (
    AVIONICS_POWER_W, BASE_FRAME_KG, CELL_V, DesignInputs, ENERGY_DENSITY_WH_KG,
    ETA_DRIVE, FOM, G, RHO, ROTOR_MASS_COEF, USABLE,
)


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
