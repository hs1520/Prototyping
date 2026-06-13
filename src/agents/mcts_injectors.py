"""MCTS parameter → SysML model injection helpers.

All functions are pure (no Orchestrator state) and are extracted from
Orchestrator so the main class focuses on coordination rather than
text-level mutations.

Public API
----------
apply_best_config_to_model(best_config, model)
apply_inject_attrs_to_sysml_text(model)
apply_inject_protocol_to_sysml_text(model, best_config)
apply_inject_sensor_count_to_sysml_text(model, best_config)
build_mcts_design_constraints(best_config) -> str
"""
from __future__ import annotations

import re
from typing import List, Optional

from ..dse.design_space import DesignConfiguration
from ..sysml.model import ElementRef, SysMLModel
from ..utils.sysml_text_utils import find_block_end

# ── Part-classification keyword sets ────────────────────────────────────────
_CTRL_KWS   = {"controller", "flight", "control", "nav", "autopilot"}
_CF_KWS     = {"controlfrequency", "controlfreq", "loopfrequency", "samplingfrequency"}
_SENSOR_KWS = {"sensor", "perception", "detector", "camera", "lidar", "imu", "gps", "radar"}
_SAFETY_KWS = {"safety", "monitor", "fault", "health"}


def apply_best_config_to_model(
    best_config: DesignConfiguration,
    model: SysMLModel,
) -> None:
    """Write MCTS winning parameter decisions into the SysMLModel in-place."""
    params = best_config.parameters
    if not params:
        return

    # 1. Control frequency → add/update ONLY the dedicated controlFrequency attribute
    freq = params.get("control_frequency_hz")
    if freq is not None:
        freq_str = str(round(float(freq), 2))
        ctrl_part = None
        best_ctrl_score = -1
        for part in model.part_definitions:
            score = sum(
                1 for sr in part.satisfy_relationships
                if sr.target and any(
                    cat in (sr.target.name or "").upper()
                    for cat in ("FUNC", "PERF")
                )
            ) + (2 if any(kw in part.name.lower() for kw in _CTRL_KWS) else 0)
            if score > best_ctrl_score:
                best_ctrl_score = score
                ctrl_part = part
        if ctrl_part is not None:
            matched = False
            for attr in ctrl_part.attributes:
                if attr.name.lower() in _CF_KWS:
                    attr.default_value = freq_str
                    matched = True
                    break
            if not matched:
                if not hasattr(model, "metadata") or model.metadata is None:
                    object.__setattr__(model, "metadata", {})
                model.metadata.setdefault("mcts_inject_attrs", []).append(
                    {
                        "part": ctrl_part.name,
                        "attr": "controlFrequency",
                        "value": freq_str,
                        "unit": "Hz",
                    }
                )

    # 2. Redundancy level → doc annotation
    redundancy = str(params.get("redundancy_level", "none"))
    if redundancy != "none":
        tag = f"redundancy={redundancy}"
        if model.description and tag not in model.description:
            model.description = f"{model.description} [{tag}]"
        elif not model.description:
            model.description = f"[{tag}]"

    # 3. Communication protocol → typed ports
    _POWER_PORT_NAMES = {"powerport", "power", "powerout", "powerin"}
    protocol = params.get("communication_protocol")
    if protocol:
        _proto_id = re.sub(r"[^A-Za-z0-9]", "", str(protocol))
        protocol_type_name = f"{_proto_id}Signal"
        for part in model.part_definitions:
            for port in part.ports:
                existing = (port.type_ref.name or "") if port.type_ref else ""
                if existing.lower() in _POWER_PORT_NAMES:
                    continue
                if port.name.lower() in _POWER_PORT_NAMES:
                    continue
                generic_types = {"dataport", "data", "rfport", "rf", ""}
                if existing.lower() in generic_types:
                    port.type_ref = ElementRef(name=protocol_type_name)

    # 4. Recommended sensor count → model-level metadata
    num_sensors = params.get("num_sensors")
    if num_sensors is not None:
        if not hasattr(model, "metadata") or model.metadata is None:
            object.__setattr__(model, "metadata", {})
        model.metadata["recommended_sensor_count"] = int(num_sensors)
        model.metadata["mcts_best_config"] = best_config.name


def apply_inject_attrs_to_sysml_text(model: SysMLModel) -> None:
    """Inject mcts_inject_attrs entries into the stored SysML text."""
    meta = getattr(model, "metadata", None) or {}
    inject_attrs = meta.get("mcts_inject_attrs", [])
    sysml_text = meta.get("last_sysml_text", "")

    if not inject_attrs or not sysml_text:
        return

    result = sysml_text
    applied: List[str] = []

    for entry in inject_attrs:
        part_name = entry.get("part", "")
        attr_name = entry.get("attr", "")
        value = entry.get("value", "")
        unit = entry.get("unit", "")

        if not part_name or not attr_name:
            continue

        already_re = re.compile(
            rf"\battribute\s+{re.escape(attr_name)}\s*:", re.IGNORECASE
        )
        if already_re.search(result):
            continue

        part_def_re = re.compile(
            rf"\bpart\s+def\s+{re.escape(part_name)}\s*\{{"
        )
        m_part = part_def_re.search(result)
        if not m_part:
            continue

        brace_open = result.index("{", m_part.start())
        closing = find_block_end(result, brace_open)
        if closing == -1:
            continue

        unit_suffix = f" [{unit}]" if unit else ""
        attr_line = f"attribute {attr_name} : Real = {value}{unit_suffix};"

        result = (
            result[:closing]
            + "\n        // (controlFrequency injected by MCTS pipeline)\n"
            + "        " + attr_line + "\n    "
            + result[closing:]
        )
        applied.append(f"{part_name}.{attr_name}={value}")

    if applied:
        model.metadata["last_sysml_text"] = result
        model.metadata["mcts_injected_attrs_applied"] = applied


def apply_inject_protocol_to_sysml_text(
    model: SysMLModel,
    best_config: DesignConfiguration,
) -> None:
    """Replace generic DataPort/RFPort with the MCTS protocol signal type in SysML text."""
    meta = getattr(model, "metadata", None) or {}
    sysml_text = meta.get("last_sysml_text", "")
    if not sysml_text:
        return

    protocol = str(best_config.parameters.get("communication_protocol", ""))
    if not protocol or protocol.lower() == "none":
        return

    proto_id = re.sub(r"[^A-Za-z0-9]", "", protocol)
    if not proto_id:
        return

    signal_type = f"{proto_id}Signal"

    # 1. Add port def at package level if absent
    if not re.search(rf"\bport\s+def\s+{re.escape(signal_type)}\b", sysml_text):
        pkg_open_re = re.compile(r"(\bpackage\s+\w+\s*\{)")
        sysml_text = pkg_open_re.sub(
            rf"\1\n    port def {signal_type};",
            sysml_text,
            count=1,
        )

    # 2. Replace directed port type annotations
    port_usage_re = re.compile(
        r"\b((?:in|out|inout)\s+port\s+(\w+)\s*:\s*)"
        r"(DataPort|RFPort|RfPort)\b",
        re.IGNORECASE,
    )
    _PWR_EXACT = re.compile(
        r"\b(power|pwr)(supply|in|out|bus|rail|link|feed|connector|line)\b",
        re.IGNORECASE,
    )

    def _replace(m: re.Match) -> str:  # type: ignore[type-arg]
        port_name: str = m.group(2)
        pn_lower = port_name.lower()
        if _PWR_EXACT.search(pn_lower) or pn_lower in ("power", "pwr"):
            return m.group(0)
        return f"{m.group(1)}{signal_type}"

    result = port_usage_re.sub(_replace, sysml_text)

    # 3. Remove stale generic port def declarations
    for _generic_def in ("DataPort", "RFPort", "RfPort", "GenericPort"):
        result = re.sub(
            rf"^[ \t]*\bport\s+def\s+{_generic_def}\s*;[ \t]*\n?",
            "",
            result,
            flags=re.IGNORECASE | re.MULTILINE,
        )

    if result != sysml_text:
        model.metadata["last_sysml_text"] = result
        model.metadata["mcts_injected_protocol_signal"] = signal_type


def apply_inject_sensor_count_to_sysml_text(
    model: SysMLModel,
    best_config: DesignConfiguration,
) -> None:
    """Programmatically add extra part usages to reach the MCTS sensor count target."""
    meta = getattr(model, "metadata", None) or {}
    sysml_text = meta.get("last_sysml_text", "")
    if not sysml_text:
        return

    target = int(best_config.parameters.get("num_sensors", 0))
    if target <= 1:
        return

    sensor_part_name: Optional[str] = None
    for part in model.part_definitions:
        if any(kw in part.name.lower() for kw in _SENSOR_KWS):
            sensor_part_name = part.name
            break
    if sensor_part_name is None:
        return

    usage_re = re.compile(
        rf"\bpart\s+(?!def\b)\w+\s*:\s*{re.escape(sensor_part_name)}\s*;",
        re.IGNORECASE,
    )
    existing = len(usage_re.findall(sysml_text))
    if existing >= target:
        return

    any_usage_re = re.compile(r"\bpart\s+(?!def\b)\w+\s*:\s*\w+\s*;")
    last_usage_end = 0
    for m in any_usage_re.finditer(sysml_text):
        last_usage_end = m.end()
    if last_usage_end == 0:
        return

    primary_usage_re = re.compile(
        rf"\bpart\s+(\w+)\s*:\s*{re.escape(sensor_part_name)}\s*;",
        re.IGNORECASE,
    )
    primary_match = primary_usage_re.search(sysml_text)
    primary_instance = primary_match.group(1) if primary_match else None

    connect_re = re.compile(
        r"\bconnect\s+(\w+)\.(\w+)\s+to\s+(\w+)\.(\w+)\s*;",
        re.IGNORECASE,
    )
    occupied_targets: set = set()
    primary_out_ports: list = []
    if primary_instance:
        for cm in connect_re.finditer(sysml_text):
            src_inst, src_port, tgt_inst, tgt_port = cm.groups()
            tgt_key = f"{tgt_inst}.{tgt_port}"
            occupied_targets.add(tgt_key)
            if src_inst.lower() == primary_instance.lower():
                primary_out_ports.append((src_port, tgt_inst, tgt_port))

    new_lines: list = []
    new_connects: list = []
    actually_injected = 0

    for i in range(existing + 1, target + 1):
        unit_name = f"sensorUnit{i}"
        unit_connects: list = []

        for src_port, tgt_inst, tgt_port in primary_out_ports:
            tgt_key = f"{tgt_inst}.{tgt_port}"
            if tgt_key not in occupied_targets:
                unit_connects.append(
                    f"    connect {unit_name}.{src_port} to {tgt_inst}.{tgt_port};"
                )
                occupied_targets.add(tgt_key)

        if not unit_connects:
            continue

        new_lines.append(f"    part {unit_name} : {sensor_part_name};")
        new_connects.extend(unit_connects)
        actually_injected += 1

    if not new_lines:
        return

    usage_block = "\n" + "\n".join(new_lines)
    result = sysml_text[:last_usage_end] + usage_block + sysml_text[last_usage_end:]

    if new_connects:
        connect_block = (
            "\n    // MCTS-injected redundant sensor connects (fan-in-safe only):\n"
            + "\n".join(new_connects)
            + "\n"
        )
        last_brace = result.rfind("}")
        if last_brace != -1:
            result = result[:last_brace] + connect_block + result[last_brace:]

    model.metadata["last_sysml_text"] = result
    model.metadata["mcts_injected_sensor_units"] = actually_injected


def build_mcts_design_constraints(best_config: DesignConfiguration) -> str:
    """Translate MCTS best-config parameters into concrete SysML implementation guidance."""
    params = best_config.parameters
    if not params:
        return ""

    lines = [
        "MCTS Architectural Decisions"
        " (these must be faithfully implemented in the SysML model):"
    ]

    redundancy = str(params.get("redundancy_level", "none"))
    if redundancy == "triple":
        lines.append(
            "  • redundancy_level=triple  →  add a state def implementing 2-of-3 "
            "majority voting.  Use canonical SysML v2 syntax — `transition <name> "
            "first <state> if <guard> then <state>;` (NOT `from/to/when`, NOT `->`). "
            "Declare `action def emergencyStop {}` at the part-def top level (NOT "
            "inline inside an entry); reference it via `entry action stop : "
            "emergencyStop;` from the failsafe state.  Ground guard names by "
            "declaring matching Boolean attributes (channelAFailed / channelBFailed "
            "/ channelCFailed)."
        )
    elif redundancy == "dual":
        lines.append(
            "  • redundancy_level=dual    →  add a state def with primary/backup "
            "channels.  Use canonical SysML v2 syntax — `transition <name> first "
            "<state> if <guard> then <state>;` (NOT `from/to/when`, NOT `->`). "
            "Declare `action def emergencyStop {}` at the part-def top level (NOT "
            "inline); reference it via `entry action stop : emergencyStop;` from "
            "the failsafe state.  Ground guard names with Boolean attributes "
            "(primaryChannelFailed / backupChannelFailed)."
        )

    freq = params.get("control_frequency_hz")
    if freq is not None:
        lines.append(
            f"  • control_frequency_hz={freq}  →  ADD (or update) exactly ONE "
            f"attribute named `controlFrequency : Real = {freq} [Hz]` in the main "
            f"flight-controller / autopilot part def.\n"
            f"    ✗ DO NOT change `telemetryRate`, `gnssRate`, `updateRate`, "
            f"`sampleRate`, or any other existing rate/frequency attribute — "
            f"those values come from requirements and must stay unchanged."
        )

    protocol = params.get("communication_protocol")
    if protocol and str(protocol).lower() not in ("none", ""):
        _proto_id = re.sub(r"[^A-Za-z0-9]", "", str(protocol))
        lines.append(
            f"  • communication_protocol={protocol}  →  THREE mandatory steps:\n"
            f"    1. Add `port def {_proto_id}Signal;` at the package level.\n"
            f"    2. Change EVERY port currently typed as `DataPort` or `RfPort` "
            f"to `{_proto_id}Signal` (e.g. `in port gnssIn : {_proto_id}Signal;`).\n"
            f"    3. Leave `PowerPort`-typed ports unchanged — they carry "
            f"electrical power, not protocol data.\n"
            f"    ✗ Do NOT keep any `DataPort` or `RfPort` in the final model."
        )

    distributed = params.get("distributed_control")
    if distributed is True:
        lines.append(
            "  • distributed_control=True  →  split control logic across dedicated "
            "part defs — do NOT centralise into a single monolithic block"
        )
    elif distributed is False:
        lines.append(
            "  • distributed_control=False →  use a single centralised controller "
            "part def that owns all decision logic"
        )

    num_sensors = params.get("num_sensors")
    if num_sensors is not None:
        lines.append(
            f"  • num_sensors={num_sensors}              →  include exactly "
            f"{num_sensors} sensor-related part def(s) or part usage(s)"
        )

    return "\n".join(lines)
