"""Normalized engineering-response bindings for the current UAV platform.

The requirement contract remains implementation-neutral.  This adapter links a
response concept to the existing SITL semantic tag, accepted model action/send
signatures and the observation used by verification.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


PLATFORM_BINDING_VERSION = "ardupilot-option2-2"


@dataclass(frozen=True)
class PlatformBinding:
    response_concept: str
    semantic_tag: str
    action_aliases: tuple[str, ...]
    command_aliases: tuple[str, ...]
    observation_concept: str
    preferred_tier: str
    canonical_commands: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_BINDINGS: tuple[PlatformBinding, ...] = (
    PlatformBinding(
        "deploy_parachute", "PARACHUTE_DEPLOY",
        ("deployparachute", "releaseparachute", "activateparachute", "parachute"),
        ("cmdparachute", "mavcmddoparachute", "parachutedeploy"),
        "parachute_deployed", "sitl", ("MAV_CMD_DO_PARACHUTE",),
    ),
    PlatformBinding(
        "prevent_arming", "SENSOR_ARMING_INHIBIT",
        ("inhibitarming", "preventarming", "blockarming", "rejectarming", "disarm"),
        (), "arming_rejected", "sitl",
    ),
    PlatformBinding(
        "alert_gcs", "SENSOR_GROUND_ALERT",
        ("alertgcs", "notifygcs", "sendfailurealert", "reportfailure"),
        (), "gcs_alert_observed", "sitl",
    ),
    PlatformBinding(
        "lock_payload", "PAYLOAD_ABORT_LOCK",
        ("lockpayload", "grabpayload", "holdpayload", "lockgripper"),
        ("cmdgrippergrab", "mavcmddogripper"), "payload_locked", "sitl",
        ("MAV_CMD_DO_GRIPPER",),
    ),
    PlatformBinding(
        "release_payload", "PAYLOAD_RELEASE",
        ("releasepayload", "deliverpayload", "opengripper"),
        ("cmdgripperrelease", "mavcmddogripper"), "payload_released", "behavioral",
        ("MAV_CMD_DO_GRIPPER",),
    ),
    PlatformBinding(
        "return_to_base", "GCS_LOSS_RTL",
        ("returntobase", "returntohome", "initiatebatteryrtb", "executereturntobase", "startrtl", "rtl", "rtb"),
        ("cmdrtl", "cmdreturntohome", "mavcmdnavreturntolaunch"),
        "vehicle_enters_rtl", "sitl", ("MAV_CMD_NAV_RETURN_TO_LAUNCH",),
    ),
    PlatformBinding(
        "controlled_landing", "GCS_LOSS_LAND",
        ("land", "safeland", "controlledland", "initiatelanding"),
        ("cmdland", "mavcmdnavland"), "vehicle_enters_land", "sitl",
        ("MAV_CMD_NAV_LAND",),
    ),
    PlatformBinding(
        "revise_waypoint_sequence", "WAYPOINT_PLAN_UPDATE",
        ("revisewaypointsequence", "updateflightplan", "incorporatewaypoints"),
        (), "active_flight_plan_updated", "behavioral",
    ),
    PlatformBinding(
        "transmit_health_report", "POST_FLIGHT_HEALTH_REPORT",
        ("transmithealthreport", "sendhealthreport", "reporthealth"),
        (), "health_report_received_by_gcs", "behavioral",
    ),
    PlatformBinding(
        "perform_self_test", "POWER_ON_SELF_TEST",
        ("performselftest", "runselftest", "executeselfcheck"),
        (), "self_test_completed", "behavioral",
    ),
    PlatformBinding(
        "maintain_controlled_flight", "SINGLE_MOTOR_OUT_STABILITY",
        ("maintaincontrolledflight", "faulttolerantcontrol", "motoroutcontrol"),
        (), "controlled_flight_stability", "gazebo",
    ),
)

_BY_CONCEPT = {binding.response_concept: binding for binding in _BINDINGS}


def normalize_symbol(value: str) -> str:
    """Case/punctuation-insensitive symbol form for semantic alias matching."""
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def platform_binding(response_concept: str) -> PlatformBinding | None:
    return _BY_CONCEPT.get(response_concept)


def action_matches(binding: PlatformBinding, *symbols: str) -> bool:
    observed = [normalize_symbol(symbol) for symbol in symbols if symbol]
    aliases = tuple(normalize_symbol(alias) for alias in binding.action_aliases)
    return any(
        alias and (alias == symbol or alias in symbol or symbol in alias)
        for alias in aliases for symbol in observed
    )


def command_matches(binding: PlatformBinding, *symbols: str) -> bool:
    observed = [normalize_symbol(symbol) for symbol in symbols if symbol]
    aliases = tuple(normalize_symbol(alias) for alias in binding.command_aliases)
    if not aliases:
        return True
    if not observed:
        return False
    return any(
        alias and (alias == symbol or alias in symbol or symbol in alias)
        for alias in aliases for symbol in observed
    )


def validate_catalogue_bindings() -> list[str]:
    """Return binding/tag discrepancies against the existing SITL catalogue."""
    try:
        from ..sitl.sitl_catalogue import _TAG_TO_ENTRY
    except Exception as exc:  # pragma: no cover - optional import boundary
        return [f"SITL catalogue unavailable: {exc}"]
    diagnostics = []
    for binding in _BINDINGS:
        # Behavioral-only tags intentionally have no ArduPilot catalogue entry.
        if binding.preferred_tier == "sitl" and binding.semantic_tag not in _TAG_TO_ENTRY:
            diagnostics.append(
                f"{binding.response_concept}: missing SITL tag {binding.semantic_tag}"
            )
    return diagnostics
