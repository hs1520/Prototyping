"""Emit SysML v2 realization artifacts for datasheet closure."""
from __future__ import annotations

import re
from typing import Tuple

from ..dse.physics_estimator import AVIONICS_POWER_W, USABLE
from ..simulation.syntax_checker import check_syntax
from ..sysml.writer import identifier
from ..utils.sysml_text_utils import find_block_end
from .closure import ClosureReport


def emit_realization_package(report: ClosureReport,
                             package_name: str = "RealizationPackage") -> Tuple[str, bool]:
    """Return a standalone realization package and whether it passes syntax checking."""
    if report.chosen is None:
        sysml = (
            f"package {package_name} {{\n"
            f"    part def UnrealizedDesign {{\n"
            f"        doc /* INFEASIBLE_REALIZATION: no catalog implementation passed interface checks */\n"
            f"        attribute realizedEnduranceMin : Real = 0.0;\n"
            f"        assert constraint realizationCloses {{ false }}\n"
            f"    }}\n"
            f"}}"
        )
        return (sysml, not check_syntax(sysml).has_errors)

    c = report.chosen
    rd = c.rd
    metrics = c.metrics
    combo_t = identifier(
        rd.combo.name, fallback="RealizedComponent", max_length=64
    )
    pack_t = identifier(
        rd.pack.name, fallback="RealizedComponent", max_length=64
    )
    frame_t = identifier(
        rd.frame.name, fallback="RealizedComponent", max_length=64
    )
    integration_t = identifier(
        rd.integration_bundle.name,
        fallback="RealizedComponent",
        max_length=64,
    )
    avionics_a = AVIONICS_POWER_W / metrics.pack_voltage_v
    asserts = []
    satisfies = []
    req_decls = []
    realized_attr = {
        "time": "realizedEnduranceMin",
        "range": "realizedRangeM",
        "mass": "realizedMassKg",
        "speed": "realizedCruiseSpeedMps",
    }
    for i, v in enumerate(report.per_requirement or ()):
        if getattr(v, "scope", "closure") != "closure":
            continue
        attr = realized_attr.get(v.family)
        if attr is None:
            # payload (hover-throttle margin) has no realized attribute on
            # RealizedDesign to assert against. The old fallback asserted
            # realizedEnduranceMin >= <payload target> — vacuously true — and
            # then `satisfy`d a requirement the closure report judged UNMET.
            # No assertion is better than a fabricated one.
            continue
        req_name = v.req_id.replace("-", "_")
        req_decls.append(
            f"    requirement def {req_name} {{ attribute target : Real = {float(v.target)}; }}\n"
            f"    requirement {req_name.lower()} : {req_name};"
        )
        op = "<=" if v.family == "mass" else ">="
        cname = f"realizationCloses{i}"
        asserts.append(f"        assert constraint {cname} {{ {attr} {op} {float(v.target)} }}")
        if v.met:
            # `satisfy` is a formal satisfaction claim; an UNMET verdict keeps
            # its (false) assert as the honest record but claims nothing.
            satisfies.append(f"        satisfy {v.req_id.replace('-', '_').lower()};")
    if not asserts:
        asserts.append("        assert constraint realizationCloses { realizedEnduranceMin >= 0.0 }")
    sysml = (
        f"package {package_name} {{\n"
        f"    part def {combo_t} {{\n"
        f"        doc /* source: {rd.combo.source_url} retrieved {rd.combo.retrieved} */\n"
        f"        attribute motorMassG : Real = {float(rd.combo.motor_mass_g)};\n"
        f"        attribute propMassG : Real = {float(rd.combo.prop_mass_g)};\n"
        f"        attribute hoverCurrentA : Real = {metrics.hover_current_per_motor_a:.9f};\n"
        f"        attribute hoverThrottle : Real = {metrics.hover_throttle:.9f};\n"
        f"        attribute curveVoltageV : Real = {float(rd.combo.voltage_v)};\n"
        f"        attribute publishedMaxThrustG : Real = {float(rd.combo.max_thrust_g())};\n"
        f"        attribute deratedMaxThrustG : Real = "
        f"{metrics.derated_max_thrust_per_motor_g:.9f};\n"
        f"    }}\n"
        f"    part def {pack_t} {{\n"
        f"        doc /* source: {rd.pack.source_url} retrieved {rd.pack.retrieved} */\n"
        f"        attribute capacityMah : Real = {float(rd.pack.capacity_mah)};\n"
        f"        attribute cells : Real = {float(rd.pack.cells)};\n"
        f"        attribute nominalVoltageV : Real = {metrics.pack_voltage_v:.9f};\n"
        f"        attribute massG : Real = {float(rd.pack.mass_g)};\n"
        f"    }}\n"
        f"    part def {frame_t} {{\n"
        f"        doc /* source: {rd.frame.source_url} retrieved {rd.frame.retrieved} */\n"
        f"        attribute massG : Real = {float(rd.frame.mass_g)};\n"
        f"        attribute arms : Real = {float(rd.frame.arms)};\n"
        f"    }}\n"
        f"    part def {integration_t} {{\n"
        f"        doc /* source: {rd.integration_bundle.source_url} retrieved "
        f"{rd.integration_bundle.retrieved}; conservative integration mass budget */\n"
        f"        attribute massG : Real = {float(rd.integration_bundle.mass_g)};\n"
        f"    }}\n"
        f"    part realizedPropulsion : {combo_t};\n"
        f"    part realizedPower : {pack_t};\n"
        f"    part realizedAirframe : {frame_t};\n"
        f"    part realizedIntegration : {integration_t};\n"
        + ("\n".join(req_decls) + "\n" if req_decls else "")
        + f"    calc def RealizedEndurance {{\n"
        f"        in capacityMah : Real; in hoverCurrentA : Real; in rotorCount : Real;\n"
        f"        in avionicsA : Real;\n"
        f"        (capacityMah / 1000.0 * {USABLE}) / (hoverCurrentA * rotorCount + avionicsA) * 60.0\n"
        f"    }}\n"
        f"    part def RealizedDesign {{\n"
        f"        attribute realizedEnduranceMin : Real = RealizedEndurance({float(rd.pack.capacity_mah)}, "
        f"{metrics.hover_current_per_motor_a:.9f}, {float(rd.rotor_count)}, {avionics_a:.9f});\n"
        f"        attribute realizedRangeM : Real = {metrics.range_m:.9f};\n"
        f"        attribute realizedMassKg : Real = {metrics.total_mass_kg:.9f};\n"
        f"        attribute packNominalVoltageV : Real = {metrics.pack_voltage_v:.9f};\n"
        f"        attribute motorCurveVoltageV : Real = {float(rd.combo.voltage_v):.9f};\n"
        f"        attribute voltageRatio : Real = {metrics.voltage_ratio:.9f};\n"
        f"        attribute realizedCruiseSpeedMps : Real = "
        f"{(metrics.range_m / (metrics.endurance_min * 60.0)) if metrics.endurance_min > 0 else 0.0:.9f};\n"
        + "\n".join(asserts) + "\n"
        + "\n".join(satisfies) + "\n"
        + "    }\n"
        "}"
    )
    return (sysml, not check_syntax(sysml).has_errors)


def inject_realization_analysis(model_text: str, report: ClosureReport) -> Tuple[str, bool]:
    """Inject RealizationPackage into a model package; return original text if syntax fails."""
    fragment, ok = emit_realization_package(report)
    if not ok:
        return model_text, False
    pkg = re.search(r"\bpackage\s+\w+\s*\{", model_text or "")
    if not pkg:
        return model_text, False
    brace = model_text.index("{", pkg.start())
    end = find_block_end(model_text, brace)
    if end == -1:
        return model_text, False
    body = fragment[fragment.index("{") + 1:fragment.rfind("}")]
    injected = model_text[:end] + "\n" + body + "\n" + model_text[end:]
    if check_syntax(injected).has_errors:
        return model_text, False
    return injected, True
