"""DSE best-config -> SysML model injection helpers.

All functions are pure (no Orchestrator state), extracted so the main class
stays on coordination rather than text-level mutations.  Sensor redundancy
is applied by the valid-by-construction operators
(``dse.operator_applicator``); the legacy
``apply_inject_sensor_count_to_sysml_text`` was removed.

Public API
----------
apply_best_config_to_model(best_config, model)
apply_inject_attrs_to_sysml_text(model)
build_dse_design_constraints(best_config) -> str
"""
from __future__ import annotations

import re
from typing import List

from ..dse.design_space import DesignConfiguration
from ..sysml.model import ElementRef, SysMLModel
from ..utils.sysml_text_utils import named_block_span

_CTRL_KWS   = {"controller", "flight", "control", "nav", "autopilot"}
_CF_KWS     = {"controlfrequency", "controlfreq", "loopfrequency", "samplingfrequency"}
_SENSOR_KWS = {"sensor", "perception", "detector", "camera", "lidar", "imu", "gps", "radar"}


def apply_best_config_to_model(
    best_config: DesignConfiguration,
    model: SysMLModel,
) -> None:
    """Write MCTS winning parameter decisions into the SysMLModel in-place."""
    params = best_config.parameters
    if not params:
        return

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

    redundancy = str(params.get("redundancy_level", "none"))
    if redundancy != "none":
        tag = f"redundancy={redundancy}"
        if model.description and tag not in model.description:
            model.description = f"{model.description} [{tag}]"
        elif not model.description:
            model.description = f"[{tag}]"

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

    num_sensors = params.get("num_sensors")
    if num_sensors is not None:
        if not hasattr(model, "metadata") or model.metadata is None:
            object.__setattr__(model, "metadata", {})
        model.metadata["recommended_sensor_count"] = int(num_sensors)
        model.metadata["dse_best_config"] = best_config.name


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

        span = named_block_span(result, "part", part_name)
        if span is None:
            continue
        brace_open, closing = span

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


def build_dse_design_constraints(best_config: DesignConfiguration) -> str:
    """Translate DSE best-config parameters into SysML implementation guidance.

    Only the catalog decision keys produce guidance; a config with none of them
    (e.g. a variation-DSE config of variant choices) yields "", so no dangling
    header reaches the refinement prompt.
    """
    params = best_config.parameters
    if not params:
        return ""

    lines = [
        "DSE Architectural Decisions"
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

    if len(lines) == 1:
        return ""
    return "\n".join(lines)
