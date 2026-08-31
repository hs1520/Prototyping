"""Optional Gazebo high-fidelity feasibility check for the recommended realization.

This is the H-tier complement to native SITL. It verifies Gazebo dynamics such
as hover stability, forward dash measurements, and the one-motor-out check when
the model has that requirement. It does not change Phase 8 datasheet CLOSED
semantics and it does not validate endurance; endurance remains a datasheet
closure result.

Run:
  PYTHONPATH=. .venv/bin/python examples/run_gazebo_feasibility.py

For CI/static reporting without Docker:
  PYTHONPATH=. .venv/bin/python examples/run_gazebo_feasibility.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.dse.physics_estimator import DesignInputs, total_mass_kg
from src.prototyping.artifact_provenance import (
    evidence_reuse_allowed,
    validate_derived_provenance,
    validate_run_provenance,
)
from src.prototyping.artifact_store import (
    atomic_write_json,
    atomic_write_text,
    ensure_open_bundle,
    input_dir,
    latest_output_dir,
    output_dir,
)
from gazebo_poc.payload_transport_evidence import (
    TransportWindow,
    evaluate_payload_transport,
)
from gazebo_poc.single_motor_out_evidence import evaluate_single_motor_out
from gazebo_poc.model_mission import ModelAction

ROOT = Path(__file__).resolve().parents[1]
OUT = output_dir()
INPUT = input_dir()
SYSML_PATH = INPUT / "final_model.sysml"
RUN_JSON = INPUT / "realization_run.json"
REPORT_JSON = OUT / "gazebo_feasibility_report.json"
REPORT_MD = OUT / "gazebo_feasibility_report.md"


def _requirements_from_sysml(text: str) -> list[str]:
    out: list[str] = []
    for m in re.finditer(
        r"requirement\s+def\s+([A-Za-z_][\w]*)\s*\{(?P<body>.*?)\}",
        text,
        re.S,
    ):
        doc = re.search(r"doc\s*/\*(.*?)\*/", m.group("body"), re.S)
        if doc:
            out.append(f"{m.group(1).replace('_', '-')}: {doc.group(1).strip()}")
    return out


def _req_id(req: str) -> str | None:
    m = re.search(r"\b(REQ[-_][A-Z]+[-_]\d+)\b", req, re.I)
    return m.group(1).replace("_", "-").upper() if m else None


def _number_after(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, re.I)
    return float(match.group(1)) if match else None


def _planned_gazebo_reqs(requirements: list[str]) -> list[dict[str, Any]]:
    planned: list[dict[str, Any]] = []
    for req in requirements:
        rid = _req_id(req)
        if not rid:
            continue
        low = req.lower()
        check = None
        reason = None
        if (("navigate" in low or "navigation" in low)
                and ("cep" in low or "circular error" in low)):
            # Autonomous waypoint navigation, not a stick-flown dash. Whether
            # the rig can fly it at all is the first thing this check reports.
            check = "navigation_accuracy"
            reason = ("waypoint navigation accuracy requires autonomous "
                      "position-controlled flight")
        elif ("payload" in low and "lock" in low and "abort" in low):
            # An inhibition claim cannot be shown by a run in which the
            # inhibiting condition never held. It is only testable by driving
            # the generated logic WITH the condition active and observing that
            # the action does not occur.
            check = "delivery_abort_inhibition"
            reason = ("payload-lock inhibition under an active abort must be "
                      "driven and physically observed, not inferred")
        elif any(k in low for k in ("single propulsion", "one motor", "motor inoperative", "propulsion unit")):
            check = "single_motor_out"
            reason = "one-motor-out controllability requires high-fidelity dynamics"
        elif any(k in low for k in ("headwind", "tailwind", "crosswind", "gust")):
            check = "wind_condition"
            reason = "wind-field dynamics are Gazebo-tier and are not covered by native SITL"
        elif any(k in low for k in ("obstacle", "collision", "avoidance")):
            check = "obstacle_avoidance"
            reason = "obstacle/contact physics are Gazebo-tier"
        elif "release" in low and ("metre" in low or "meter" in low):
            check = "positional_release"
            reason = "positional payload-release error needs high-fidelity motion/trajectory"
        elif "within" in low and "second" in low and any(k in low for k in ("deploy", "release", "actuat", "lock")):
            check = "timed_actuation"
            reason = "timed actuator behavior needs synchronized high-fidelity execution"
        elif (("attitude" in low or "roll and pitch" in low)
              and ("rms" in low or "deviation" in low)):
            if any(k in low for k in ("payload", "transport")):
                check = "payload_attitude"
                reason = ("payload-carry attitude and hover-throttle margin need "
                          "closed-loop flight dynamics")
            else:
                check = "cruise_attitude"
                reason = "cruise attitude RMS needs closed-loop flight dynamics"
        elif ("airspeed" in low or "cruise" in low) and "at least" in low and "m/s" in low:
            check = "cruise_speed"
            reason = "nil-wind top-speed dash needs closed-loop flight dynamics"
        if check:
            item: dict[str, Any] = {
                "req_id": rid,
                "check": check,
                "message": reason,
                "requirement_text": req,
            }
            if check == "wind_condition":
                item["wind_mps"] = _number_after(
                    r"(?:headwind|tailwind|crosswind|gust)[^.;]{0,50}?(\d+(?:\.\d+)?)\s*m/s",
                    low,
                )
                item["min_groundspeed_mps"] = _number_after(
                    r"(?:minimum\s+)?forward\s+ground\s+speed(?:\s+of)?\s+(\d+(?:\.\d+)?)\s*m/s",
                    low,
                )
            elif check == "timed_actuation":
                item["max_delay_s"] = _number_after(
                    r"within\s+(\d+(?:\.\d+)?)\s*seconds?", low
                )
            elif check == "positional_release":
                item["max_error_m"] = _number_after(
                    r"within\s+(\d+(?:\.\d+)?)\s*(?:metres?|meters?)", low
                )
            elif check in ("cruise_attitude", "payload_attitude"):
                item["max_rms_deg"] = _number_after(
                    r"within\s+(\d+(?:\.\d+)?)\s*degree", low
                )
                if check == "cruise_attitude":
                    authorised = re.search(
                        r"(?:from|between)\s+(\d+(?:\.\d+)?)\s*(?:m/s)?\s*"
                        r"(?:to|and)\s+(\d+(?:\.\d+)?)\s*m/s",
                        low,
                    )
                    if authorised:
                        item["authorised_speed_range_mps"] = [
                            float(authorised.group(1)),
                            float(authorised.group(2)),
                        ]
                        item["authorised_speed_source"] = "requirement"
                if check == "payload_attitude":
                    item["min_margin_pct"] = _number_after(
                        r"margin of at least\s+(\d+(?:\.\d+)?)\s*percent", low
                    )
            elif check == "cruise_speed":
                item["min_speed_mps"] = _number_after(
                    r"at least\s+(\d+(?:\.\d+)?)\s*m/s", low
                )
            elif check == "navigation_accuracy":
                item["max_cep_m"] = _number_after(
                    r"less\s+than\s+(\d+(?:\.\d+)?)\s*(?:metres?|meters?)", low
                )
            elif check == "obstacle_avoidance":
                # Every number comes from the requirement's own text; the
                # scenario envelope is never invented here.
                item["detection_range_m"] = _number_after(
                    r"detection\s+no\s+later\s+than\s+(\d+(?:\.\d+)?)\s*(?:metres?|meters?)",
                    low,
                )
                item["min_separation_m"] = _number_after(
                    r"separation\s+of\s+at\s+least\s+(\d+(?:\.\d+)?)\s*(?:metres?|meters?)",
                    low,
                )
                item["approach_speed_mps"] = _number_after(
                    r"closing\s+speed\s+no\s+greater\s+than\s+(\d+(?:\.\d+)?)\s*m/s",
                    low,
                )
                item["contract_ready"] = all(
                    item.get(key) is not None
                    for key in ("detection_range_m", "min_separation_m",
                                "approach_speed_mps")
                )
                if not item["contract_ready"]:
                    item["semantic_gaps"] = [
                        f"{key} not stated in the requirement"
                        for key in ("detection_range_m", "min_separation_m",
                                    "approach_speed_mps")
                        if item.get(key) is None
                    ]
            planned.append(item)
    return planned


def _load_run() -> dict[str, Any]:
    if not RUN_JSON.exists():
        raise FileNotFoundError(f"{RUN_JSON} not found; run examples/run_realization_report.py first")
    return json.loads(RUN_JSON.read_text(encoding="utf-8"))


def _drift_value(chosen: dict[str, Any], name: str, fallback: float) -> float:
    for row in chosen.get("design_drift", []) or []:
        if row.get("name") == name and row.get("realized") is not None:
            return float(row["realized"])
    return float(fallback)


def _combo_max_thrust_g(chosen: dict[str, Any]) -> float | None:
    combo_name = chosen.get("combo")
    if not combo_name:
        return None
    try:
        from src.realization.catalog import DEFAULT_CATALOG

        for combo in DEFAULT_CATALOG.combos:
            if combo.name == combo_name:
                return float(combo.max_thrust_g())
    except Exception:
        return None
    return None


def _gazebo_design_from_run(run: dict[str, Any]) -> dict[str, Any]:
    design = DesignInputs(**(run.get("recommended_design_inputs") or {}))
    chosen = ((run.get("realization") or {}).get("chosen") or {})
    return {
        "rotor_count": int(design.rotor_count),
        "rotor_radius_m": _drift_value(chosen, "rotor_radius_m", design.rotor_radius_m),
        "battery_capacity_mah": _drift_value(chosen, "battery_capacity_mah", design.battery_capacity_mah),
        "battery_cells": _drift_value(chosen, "battery_cells", design.battery_cells),
        "mass_kg": float(chosen.get("total_mass_kg") or total_mass_kg(design)),
        "payload_mass_kg": float(design.payload_mass_kg),
        "max_thrust_g": _combo_max_thrust_g(chosen),
        "hover_throttle": chosen.get("hover_throttle"),
        "source": "Phase 8 realized components when available; DSE recommendation otherwise",
    }


def _single_motor_req(planned: list[dict[str, Any]]) -> str | None:
    for item in planned:
        if item.get("check") == "single_motor_out":
            return item.get("req_id")
    return None


#: A cruise claim phrased "at all authorised speeds" needs an envelope, not a
#: point. Three certified steady points spanning the authority range is the
#: minimum this harness will call a sweep; the reported RMS is the worst of them.
_MIN_SWEEP_POINTS = 3
#: How far the headroom under a bound must exceed the variation a sweep itself
#: showed, before the sweep may stand in for speeds nobody flew. 10x is a
#: deliberately generous extrapolation: the swept points must look flat next to
#: the margin, not merely happen to pass.
_ENVELOPE_MARGIN_FACTOR = 10.0
#: ...and the worst swept point must clear the bound by this fraction outright.
#: A result sitting just under the limit closes nothing, however flat the sweep.
_ENVELOPE_MARGIN_HEADROOM = 0.5
_CEP_REQUIRED_POINTS = 8


#: How many times the one-motor-out scenario is flown before a verdict is
#: recorded. The outcome is bistable — repeated flights of the same
#: configuration have both held a clean hover and sunk — so a single sample
#: cannot support a redundancy claim in either direction.
_MOTOR_OUT_REPEATS = 5


def _decided_by_model(gazebo: dict[str, Any] | None, what: str) -> bool:
    """True when the generated model fired the exact physical action."""
    if not gazebo or gazebo.get(f"{what}_decided_by") != "generated model":
        return False
    expected = {
        "payload_release": ModelAction.RELEASE_PAYLOAD.value,
        "parachute": ModelAction.DEPLOY_PARACHUTE.value,
    }[what]
    return any(
        decision.get("action_definition") == expected
        for decision in gazebo.get(f"{what}_decisions") or []
    )


def _model_owned_clause(gazebo: dict[str, Any], what: str) -> str:
    """Name the machine and action the generated model fired."""
    expected = {
        "payload_release": ModelAction.RELEASE_PAYLOAD.value,
        "parachute": ModelAction.DEPLOY_PARACHUTE.value,
    }[what]
    decisions = [
        decision
        for decision in gazebo.get(f"{what}_decisions") or []
        if decision.get("action_definition") == expected
    ]
    fired = "; ".join(
        f"{d.get('owner_part')}.{d.get('machine')} {d.get('from_state')}"
        f"->{d.get('to_state')} on {d.get('event')} firing {d.get('action')}"
        f" : {d.get('action_definition')}"
        for d in decisions
    )
    return (
        "the GENERATED model owned the decision: "
        + (fired or "no transition recorded")
        + " — the harness only supplied the event and actuated what the model fired"
    )


def _planned_check(planned: list[dict[str, Any]], check: str) -> dict[str, Any] | None:
    return next((item for item in planned if item.get("check") == check), None)


def _payload_timing_req(planned: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in planned:
        if item.get("check") != "timed_actuation":
            continue
        low = str(item.get("requirement_text") or "").lower()
        if "parachute" in low or "ballistic recovery" in low:
            continue
        if any(k in low for k in ("payload", "delivery", "gripper")):
            return item
    return None


def _parachute_timing_req(planned: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in planned:
        if item.get("check") != "timed_actuation":
            continue
        low = str(item.get("requirement_text") or "").lower()
        if "parachute" in low or "ballistic recovery" in low:
            return item
    return None


def _run_live_gazebo(gazebo_design: dict[str, Any], planned: list[dict[str, Any]],
                     include_single_motor_out: bool = False,
                     mission_model_text: str | None = None) -> dict[str, Any]:
    from gazebo_poc import run_flight
    from gazebo_poc.prop_theory import rpm_cross_check

    mass = float(gazebo_design["mass_kg"])
    rotor_count = int(gazebo_design["rotor_count"])
    rotor_radius = float(gazebo_design["rotor_radius_m"])
    capacity = float(gazebo_design["battery_capacity_mah"])
    max_thrust_g = gazebo_design.get("max_thrust_g")
    hover_throttle = gazebo_design.get("hover_throttle")
    wind_req = _planned_check(planned, "wind_condition") or {}
    timing_req = _payload_timing_req(planned)
    position_req = _planned_check(planned, "positional_release")
    parachute_req = _parachute_timing_req(planned)
    cruise_att_req = _planned_check(planned, "cruise_attitude")
    payload_att_req = _planned_check(planned, "payload_attitude")
    speed_req = _planned_check(planned, "cruise_speed")
    navigation_req = _planned_check(planned, "navigation_accuracy")
    wind_mps = float(wind_req.get("wind_mps") or 0.0)
    min_groundspeed = wind_req.get("min_groundspeed_mps")
    payload_release = timing_req is not None or position_req is not None
    payload_mass_kg = float(gazebo_design.get("payload_mass_kg") or 0.0)

    rc = run_flight.main(
        mass_kg=mass,
        rotor_radius=rotor_radius,
        capacity_mah=capacity,
        rotor_count=rotor_count,
        calibrate=True,
        max_thrust_g=max_thrust_g,
        hover_throttle=hover_throttle,
        wind_mps=wind_mps,
        wind_min_groundspeed_mps=min_groundspeed,
        payload_release=payload_release,
        payload_mass_kg=payload_mass_kg,
        positional_release=position_req is not None,
        positional_tolerance_m=float((position_req or {}).get("max_error_m") or 1.0),
        parachute_deploy=parachute_req is not None,
        parachute_max_delay_s=float((parachute_req or {}).get("max_delay_s") or 0.5),
        measure_attitude=(cruise_att_req is not None
                          or payload_att_req is not None),
        mission_model_text=mission_model_text,
        # extend the pre-wind (nil-wind) dash so top speed and cruise attitude
        # are sampled from a built-up plateau rather than the 7 s direction probe
        nilwind_dash_s=(
            15.0 if (speed_req is not None or cruise_att_req is not None) else 0.0
        ),
    )
    result = dict(run_flight.LAST_RESULT)
    result["return_code"] = rc
    result["rotor_count"] = rotor_count
    result["mass_kg"] = mass
    result["rotor_radius_m"] = rotor_radius
    result["capacity_mah"] = capacity
    if cruise_att_req is not None:
        result["cruise_authorised_speed_range_mps"] = cruise_att_req.get(
            "authorised_speed_range_mps"
        )
        result["cruise_authorised_speed_source"] = cruise_att_req.get(
            "authorised_speed_source"
        )
    if result.get("hover_rpm"):
        rpm = rpm_cross_check(float(result["hover_rpm"]), mass, rotor_count, rotor_radius)
        result.update({
            "rpm_theory": rpm.theory_rpm,
            "rpm_pct_diff": rpm.pct_diff,
            "rpm_within_ct_band": rpm.within_ct_band,
            "implied_ct": rpm.ct_implied,
        })

    # Navigation accuracy is a separate route-following campaign. Its flight
    # intentionally returns after the eight global waypoints, whereas the
    # primary flight must continue into cruise and wind sweeps; combining them
    # would silently suppress every later primary-flight measurement.
    if navigation_req is not None and result.get("hover_stable"):
        rc_navigation = run_flight.main(
            mass_kg=mass,
            rotor_radius=rotor_radius,
            capacity_mah=capacity,
            rotor_count=rotor_count,
            calibrate=True,
            max_thrust_g=max_thrust_g,
            hover_throttle=hover_throttle,
            navigation_accuracy=True,
            gps_horizontal_error_m=gazebo_design.get("gps_horizontal_error_m"),
        )
        navigation_result = dict(run_flight.LAST_RESULT)
        result["navigation_return_code"] = rc_navigation
        for key, value in navigation_result.items():
            if (key.startswith("cep_") or key.startswith("takeoff_")
                    or key == "guided_mode_held_after_arming"):
                result[key] = value

    # Obstacle avoidance is its own scenario: the vehicle must approach a
    # stationary threat and be seen to break off. It cannot ride along with the
    # cruise survey, so it is flown separately — like one-motor-out — and only
    # when the requirement itself states the whole envelope (detection range,
    # separation, closing speed). Nothing here invents a scenario.
    obstacle_req = _planned_check(planned, "obstacle_avoidance")
    if obstacle_req and obstacle_req.get("contract_ready") and result.get("hover_stable"):
        rc_obstacle = run_flight.main(
            mass_kg=mass,
            rotor_radius=rotor_radius,
            capacity_mah=capacity,
            rotor_count=rotor_count,
            calibrate=True,
            max_thrust_g=max_thrust_g,
            hover_throttle=hover_throttle,
            obstacle_avoidance=True,
            obstacle_detection_range_m=float(obstacle_req["detection_range_m"]),
            obstacle_min_separation_m=float(obstacle_req["min_separation_m"]),
            obstacle_approach_speed_mps=float(obstacle_req["approach_speed_mps"]),
        )
        obstacle_result = dict(run_flight.LAST_RESULT)
        result["obstacle_return_code"] = rc_obstacle
        for key, value in obstacle_result.items():
            if key.startswith("obstacle_"):
                result[key] = value

    # The inhibition scenario needs its own flight: a payload can only be
    # released once, so the abort case cannot share the nominal delivery.
    inhibition_req = _planned_check(planned, "delivery_abort_inhibition")
    if (inhibition_req and mission_model_text and payload_release
            and result.get("hover_stable")):
        rc_inhibit = run_flight.main(
            mass_kg=mass,
            rotor_radius=rotor_radius,
            capacity_mah=capacity,
            rotor_count=rotor_count,
            calibrate=True,
            max_thrust_g=max_thrust_g,
            hover_throttle=hover_throttle,
            payload_release=True,
            payload_mass_kg=payload_mass_kg,
            positional_release=position_req is not None,
            positional_tolerance_m=float((position_req or {}).get("max_error_m") or 1.0),
            mission_model_text=mission_model_text,
            delivery_abort_before_release=True,
        )
        inhibit_result = dict(run_flight.LAST_RESULT)
        result["abort_inhibition_req"] = str(inhibition_req["req_id"])
        result["abort_inhibition_return_code"] = rc_inhibit
        result["abort_inhibition_abort_active"] = inhibit_result.get("payload_abort_active")
        # None = the release identity never resolved on this model (harness
        # question unput); [] = resolved and genuinely unguarded. Coercing the
        # first into the second is how a guarded model gets blamed.
        result["abort_inhibition_release_guards"] = inhibit_result.get(
            "payload_release_guards")
        result["abort_inhibition_flags_unbound"] = inhibit_result.get(
            "payload_abort_flags_unbound") or []
        result["abort_inhibition_flags_raised"] = inhibit_result.get(
            "payload_abort_flags_raised") or {}
        result["abort_inhibition_release_detected"] = inhibit_result.get(
            "payload_release_detected")
        result["abort_inhibition_decisions"] = inhibit_result.get(
            "payload_release_decisions")
        result["abort_inhibition_observer_available"] = inhibit_result.get(
            "payload_observer_available")
        result["abort_inhibition_z_before_m"] = inhibit_result.get("payload_z_before_m")
        result["abort_inhibition_z_after_m"] = inhibit_result.get("payload_z_after_m")

    # Transport is its own flight. In the delivery flight the release block runs
    # before the cruise survey, so every cruise point there is flown empty —
    # measured: the payload sat at the takeoff point while the vehicle reached
    # 1253 m away. A scenario that carries the payload throughout is the only
    # way a cruise window is transport evidence.
    if payload_att_req is not None and payload_mass_kg > 0 and result.get("hover_stable"):
        rc_transport = run_flight.main(
            mass_kg=mass,
            rotor_radius=rotor_radius,
            capacity_mah=capacity,
            rotor_count=rotor_count,
            calibrate=True,
            max_thrust_g=max_thrust_g,
            hover_throttle=hover_throttle,
            payload_transport=True,
            payload_mass_kg=payload_mass_kg,
            measure_attitude=True,
        )
        transport_result = dict(run_flight.LAST_RESULT)
        result["transport_return_code"] = rc_transport
        # the transport flight's windows REPLACE the delivery flight's, which
        # are empty by construction
        result["transport_windows"] = transport_result.get("transport_windows")
        result["transport_cruise_points"] = transport_result.get(
            "cruise_sweep_steady_points")

    rid = _single_motor_req(planned) if include_single_motor_out else None
    if rid and result.get("hover_stable"):
        # One flight cannot settle this. The same configuration has held 1.17,
        # 6.27, 9.93 and 10.00 m with attitude RMS from 1.59 to 19.56 deg — one
        # of those a genuinely clean flight. A single sample of a bistable
        # outcome is a coin flip, and a coin flip is not evidence about a safety
        # requirement whichever way it lands.
        runs = []
        for attempt in range(_MOTOR_OUT_REPEATS):
            rc_fail = run_flight.main(
                mass_kg=mass,
                rotor_radius=rotor_radius,
                capacity_mah=capacity,
                rotor_count=rotor_count,
                calibrate=True,
                fail_rotor=0,
                max_thrust_g=max_thrust_g,
                hover_throttle=hover_throttle,
                # "maintain controlled flight" is an attitude claim before it is
                # an altitude one: a run held 9.93 m to +/-0.06 m while wobbling
                # 13.5 deg. This scenario is never flown without measuring both.
                measure_attitude=True,
            )
            attempt_result = dict(run_flight.LAST_RESULT)
            runs.append({
                "attempt": attempt,
                "return_code": rc_fail,
                "stable": bool(attempt_result.get("hover_stable")),
                "hover_alt_m": attempt_result.get("hover_alt_m"),
                "hover_throttle_pct": attempt_result.get("hover_throttle_pct"),
                "attitude_rms_deg": attempt_result.get("hover_attitude_rms_deg"),
            })
            print(f"[motor-out] attempt {attempt + 1}/{_MOTOR_OUT_REPEATS}: "
                  f"stable={runs[-1]['stable']} alt={runs[-1]['hover_alt_m']} "
                  f"attitude_rms={runs[-1]['attitude_rms_deg']}", flush=True)

        measured = [r for r in runs if r["attitude_rms_deg"] is not None]
        # Worst case, not average: a redundancy claim that only holds sometimes
        # does not hold. The worst run is the one with the largest attitude RMS.
        worst = max(measured, key=lambda r: float(r["attitude_rms_deg"])) if measured else None
        passes = [r for r in runs if r["stable"]]
        result["motor_failure_req"] = rid
        result["motor_failure_runs"] = runs
        result["motor_failure_attempts"] = len(runs)
        result["motor_failure_passes"] = len(passes)
        result["motor_failure_tolerant"] = len(passes) == len(runs) and bool(runs)
        result["motor_failure_deterministic"] = (
            len(passes) == 0 or len(passes) == len(runs)
        )
        result["motor_failure_return_code"] = worst["return_code"] if worst else None
        result["motor_failure_hover_alt_m"] = worst["hover_alt_m"] if worst else None
        result["motor_failure_hover_throttle_pct"] = (
            worst["hover_throttle_pct"] if worst else None
        )
        result["motor_failure_attitude_rms_deg"] = (
            worst["attitude_rms_deg"] if worst else None
        )
        result["motor_failure_attitude_limit_deg"] = (
            run_flight.LAST_RESULT.get("hover_attitude_limit_deg")
        )
    return result


def _req_results(gazebo: dict[str, Any] | None, planned: list[dict[str, Any]],
                 include_single_motor_out: bool = False) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    covered: set[str] = set()
    if gazebo and gazebo.get("motor_failure_req"):
        rid = str(gazebo["motor_failure_req"])
        att = gazebo.get("motor_failure_attitude_rms_deg")
        attempts = int(gazebo.get("motor_failure_attempts") or 0)
        runs = gazebo.get("motor_failure_runs") or []
        claim = evaluate_single_motor_out(runs) if runs else None
        passes = claim.sensitivity[0].passed_runs if claim is not None else 0

        def _spread(key, digits=2):
            values = [r.get(key) for r in runs if r.get(key) is not None]
            return ", ".join(f"{float(v):.{digits}f}" for v in values) or "none"
        covered.add(rid)
        results.append({
            "req_id": rid,
            "check": "single_motor_out",
            "status": "PARTIAL" if claim is not None else "INCONCLUSIVE",
            "criterion_evaluation": (
                "PASS" if claim.status == "verified" else "FAIL"
            ) if claim is not None else None,
            "criterion": asdict(claim.criterion) if claim is not None else None,
            "sensitivity": (
                [asdict(item) for item in claim.sensitivity]
                if claim is not None else []
            ),
            "attitude_rms_deg": att,
            "attempts": attempts,
            "passes": passes,
            "message": (
                "Gazebo one-motor-out attitude was not measured, so nothing here "
                "speaks to controlled flight — altitude alone cannot: repeated "
                "runs of this configuration held 1.17, 6.27, 9.93 and 10.00 m "
                "with attitude RMS from 1.59 to 19.56 deg"
                if att is None else
                f"{claim.description}. The 5 deg bound is an unaccepted "
                "engineering interpretation, not a value in REQ-SAFE-007, so "
                "the requirement verdict remains conditional. Attitude RMS "
                f"across flights: {_spread('attitude_rms_deg')} deg; steady "
                f"altitude: {_spread('hover_alt_m')} m at "
                f"{_spread('hover_throttle_pct', 0)}% throttle"
            ),
        })

    wind_req = _planned_check(planned, "wind_condition")
    if gazebo and wind_req and gazebo.get("wind_test_complete"):
        rid = str(wind_req["req_id"])
        speed = float(gazebo.get("wind_groundspeed_mps") or 0.0)
        minimum = float(wind_req.get("min_groundspeed_mps") or 0.0)
        wind = float(wind_req.get("wind_mps") or 0.0)
        alignment = gazebo.get("wind_headwind_alignment")
        aligned = alignment is not None and float(alignment) >= 0.9
        steady = (gazebo.get("wind_groundspeed_steady_state") or {})
        held = bool(steady.get("steady"))
        meets = aligned and held and speed >= minimum
        f_drag = gazebo.get("wind_drag_area_m2")
        # How wrong could the drag estimate be before the verdict flips? The
        # vehicle holds a fixed AIRSPEED at fixed tilt, so ground speed is
        # airspeed - wind, and airspeed scales as 1/sqrt(f). Stating this makes
        # the result checkable rather than dependent on trusting the Cd values.
        tolerated_f = None
        if f_drag and speed + wind > 0 and minimum + wind > 0:
            tolerated_f = float(f_drag) * ((speed + wind) / (minimum + wind)) ** 2
        covered.add(rid)
        results.append({
            "req_id": rid,
            "check": "wind_condition",
            # The headwind now acts through the airframe's geometry-derived
            # quadratic drag plate, evaluated by gz-sim's LiftDrag against
            # airspeed, and the ground speed held is certified steady state.
            "status": "PASS" if meets else "INCONCLUSIVE",
            "message": (
                f"Gazebo closed-loop flight HELD {speed:.2f} m/s ground speed against a "
                f"{wind:.1f} m/s headwind at full forward authority "
                f"(minimum {minimum:.1f}, opposition alignment={alignment}; "
                f"{steady.get('reason', 'steady state unknown')}, drift "
                f"{steady.get('drift_fraction')} of mean over "
                f"{steady.get('duration_s')} s, n={steady.get('samples')}); "
                f"drag is the geometry-derived flat-plate area f={f_drag} m^2 acting on "
                "airspeed, not a lumped linearization of it"
                + (f"; the verdict survives unless the true f exceeds "
                   f"{tolerated_f:.4f} m^2 ({tolerated_f / float(f_drag):.2f}x the estimate)"
                   if tolerated_f else "")
            ),
        })

    timing_req = _payload_timing_req(planned)
    if gazebo and timing_req and gazebo.get("payload_release_commanded"):
        rid = str(timing_req["req_id"])
        delay = gazebo.get("payload_release_delay_s")
        chain = gazebo.get("payload_coordinate_to_separation_delay_s")
        limit = float(timing_req.get("max_delay_s") or 0.0)
        detected = bool(gazebo.get("payload_release_detected"))
        observer_available = bool(gazebo.get("payload_observer_available"))
        if chain is not None:
            # Full requirement interval: coordinate condition satisfied →
            # physical separation. Who owned the decision decides the ceiling:
            # the generated logic consuming the event and firing its own action
            # is what makes this evidence about the model rather than about
            # Gazebo's ability to separate a joint.
            meets = detected and float(chain) <= limit
            message = (
                f"delivery-coordinate condition satisfied → physical detachable-joint "
                f"separation delay={float(chain):.3f} s (limit {limit:.1f} s; "
                f"command→separation {delay} s, detected={detected}); "
                + (_model_owned_clause(gazebo, "payload_release")
                   if _decided_by_model(gazebo, "payload_release") else
                   "the coordinate condition was evaluated by the harness, not by "
                   "generated mission logic")
            )
        else:
            meets = detected and delay is not None and float(delay) <= limit
            message = (
                f"MAV_CMD_DO_GRIPPER to physical detachable-joint separation "
                f"delay={delay} s (limit {limit:.1f} s, detected={detected}, "
                f"observer_available={observer_available}); "
                "coordinate-condition detection was not exercised"
            )
        covered.add(rid)
        results.append({
            "req_id": rid,
            "check": "timed_actuation",
            # Physical timing is measured either way; the model owning the
            # decision is what lifts the ceiling off PARTIAL.
            "status": (
                ("PASS" if _decided_by_model(gazebo, "payload_release") else "PARTIAL")
                if meets else "FAIL" if observer_available else "INCONCLUSIVE"
            ),
            "message": message,
        })

    position_req = _planned_check(planned, "positional_release")
    if gazebo and position_req and gazebo.get("payload_release_commanded"):
        rid = str(position_req["req_id"])
        error = gazebo.get("payload_release_position_error_m")
        limit = float(position_req.get("max_error_m") or 0.0)
        physical_position_observed = (
            gazebo.get("payload_release_position_basis")
            == "payload_ground_truth_at_separation"
            and bool(gazebo.get("payload_release_detected"))
            and error is not None
        )
        met = physical_position_observed and float(error) <= limit
        # "no delivery-abort condition is active" must be shown, not assumed:
        # absence of an abort flag is not evidence that none was active.
        abort_inactive = gazebo.get("delivery_abort_inactive") is True
        # The requirement bounds WHEN the release is commanded — "release the
        # payload when the current geographic position is within 1.0 metre" —
        # so the clause closes on where the vehicle TRULY was at the trigger.
        # Not the estimator's own figure for that moment (a system scoring
        # itself against its own estimate proves nothing), and not the payload's
        # position at separation, which the requirement does not bound.
        trigger_truth = gazebo.get("trigger_truth_error_m")
        separation_truth = gazebo.get("separation_truth_error_m")
        trigger_truth_observed = trigger_truth is not None
        covered.add(rid)
        results.append({
            "req_id": rid,
            "check": "positional_release",
            # The requirement is a conjunction: within 1.0 m of the waypoint AND
            # no abort active. Only the first half was ever checked, so a
            # release that happened during an abort would have closed it.
            "status": (
                ("PASS" if _decided_by_model(gazebo, "payload_release")
                 and abort_inactive else "PARTIAL")
                if (trigger_truth_observed and float(trigger_truth) <= limit)
                else "FAIL" if trigger_truth_observed else "INCONCLUSIVE"
            ),
            "trigger_truth_error_m": trigger_truth,
            "trigger_estimated_error_m": gazebo.get("trigger_estimated_error_m"),
            "separation_truth_error_m": separation_truth,
            "message": (
                (f"the vehicle was truly {trigger_truth} m from the designated "
                 f"waypoint when the release was decided (limit {limit:.1f} m) — "
                 "this is the clause the requirement states, and it is judged on "
                 "ground truth rather than the estimator's own figure for that "
                 f"moment, which read {gazebo.get('trigger_estimated_error_m')} m. "
                 f"The payload then separated {separation_truth} m from the "
                 f"waypoint, {gazebo.get('trigger_to_separation_s')} s later: the "
                 "requirement does not bound where the payload ends up, but a "
                 "release-on-arrival rule misses by roughly the approach speed "
                 "times the actuation delay, and that is worth seeing; "
                 if trigger_truth_observed else
                 "the vehicle's true position at the trigger was not observed, "
                 "and the estimator's own figure cannot stand in for it; ")
                + ("no delivery-abort condition was active, as the requirement's "
                   "second conjunct demands; "
                   if abort_inactive else
                   "the delivery-abort state was NOT shown inactive, so the "
                   "requirement's second conjunct is unverified; ")
                + (_model_owned_clause(gazebo, "payload_release")
                   if _decided_by_model(gazebo, "payload_release") else
                   "harness-triggered — this exercises trajectory/position/actuator "
                   "coupling but not generated mission-logic ownership of the trigger")
            ),
        })

    parachute_req = _parachute_timing_req(planned)
    if gazebo and parachute_req and gazebo.get("parachute_commanded"):
        rid = str(parachute_req["req_id"])
        delay = gazebo.get("parachute_deploy_delay_s")
        limit = float(parachute_req.get("max_delay_s") or 0.0)
        observed = bool(gazebo.get("parachute_model_observed"))
        observer_available = bool(gazebo.get("parachute_observer_available"))
        met = observed and delay is not None and float(delay) <= limit
        covered.add(rid)
        results.append({
            "req_id": rid,
            "check": "parachute_deploy_timing",
            "status": (
                ("PASS" if _decided_by_model(gazebo, "parachute") else "PARTIAL")
                if met else "FAIL" if observer_available else "INCONCLUSIVE"
            ),
            "message": (
                f"MAV_CMD_DO_PARACHUTE to Gazebo parachute model creation/attachment "
                f"delay={delay} s (limit {limit:.1f} s, observed={observed}); "
                + (_model_owned_clause(gazebo, "parachute")
                   + "; precedence over other safety responses is a separate "
                     "arbitration claim and is NOT established here"
                   if _decided_by_model(gazebo, "parachute") else
                   "critical-failure detection and precedence were not injected "
                   "by this subcheck")
            ),
        })
        precedence_status = gazebo.get("parachute_precedence_status")
        if precedence_status is not None:
            results.append({
                "req_id": rid,
                "check": "safety_precedence",
                "status": {
                    "verified": "PASS",
                    "failed": "FAIL",
                    "inconclusive": "INCONCLUSIVE",
                }[precedence_status],
                "message": (
                    "critical propulsion failure, sensor failure, low battery, "
                    "and communication loss were simultaneously active; "
                    f"{gazebo.get('parachute_precedence_description')}; "
                    "the competing response set was discovered from the generated "
                    "SafetyArbiter: "
                    f"{gazebo.get('parachute_precedence_competing_actions')}"
                ),
            })

    speed_req = _planned_check(planned, "cruise_speed")
    if gazebo and speed_req and gazebo.get("nilwind_dash_speed_mps") is not None:
        rid = str(speed_req["req_id"])
        speed = float(gazebo["nilwind_dash_speed_mps"])
        minimum = float(speed_req.get("min_speed_mps") or 0.0)
        steady = gazebo.get("nilwind_dash_steady_state") or {}
        held = bool(steady.get("steady"))
        swept = gazebo.get("cruise_sweep_speeds_mps") or []
        met = held and speed >= minimum
        covered.add(rid)
        results.append({
            "req_id": rid,
            "check": "cruise_speed",
            # Nil wind → ground speed equals airspeed; ALT_HOLD keeps level
            # flight. A speed may only be reported once its window plateaus:
            # a still-accelerating dash reports the dash duration, not the
            # vehicle. INCONCLUSIVE (not FAIL) when nothing reached steady
            # state, because no speed was actually measured.
            "status": ("PASS" if met else "FAIL" if held else "INCONCLUSIVE"),
            "message": (
                f"Gazebo nil-wind pitch sweep: the fastest speed the vehicle HELD was "
                f"{speed:.2f} m/s at RC{gazebo.get('nilwind_dash_pitch_rc')} "
                f"(requirement >= {minimum:.1f} m/s); "
                + (f"{len(swept)} certified steady points "
                   f"({', '.join(f'{v:.1f}' for v in swept)} m/s); "
                   if swept else "")
                + f"steady state: {steady.get('reason', 'not established')}, drift "
                f"{steady.get('drift_fraction')} of mean over {steady.get('duration_s')} s, "
                f"n={steady.get('samples')}; altitude held in ALT_HOLD, "
                "ground speed = airspeed in nil wind"
            ),
        })

    catt_req = _planned_check(planned, "cruise_attitude")
    if gazebo and catt_req and gazebo.get("cruise_attitude_rms_deg") is not None:
        rid = str(catt_req["req_id"])
        rms = float(gazebo["cruise_attitude_rms_deg"])
        limit = float(catt_req.get("max_rms_deg") or 0.0)
        points = int(gazebo.get("cruise_attitude_points") or 0)
        span = gazebo.get("cruise_attitude_speed_span_mps") or []
        swept = points >= _MIN_SWEEP_POINTS and len(span) == 2
        authorised = gazebo.get("cruise_authorised_speed_range_mps") or []
        authority_source = gazebo.get("cruise_authorised_speed_source")
        authority_defined = (
            authority_source in {"requirement", "generated_model"}
            and len(authorised) == 2
        )
        authority_covered = (
            swept and authority_defined
            and float(span[0]) <= float(authorised[0])
            and float(span[1]) >= float(authorised[1])
        )
        # When nobody declares the authorised envelope, "all authorised speeds"
        # is a universal quantifier over a set we cannot enumerate — and a
        # verdict that no finite sweep can ever satisfy has stopped measuring
        # the vehicle and started restating that we did not fly infinitely many
        # points. The sweep may still speak for the set when two things hold:
        #   * the top of the sweep IS the fastest speed the vehicle held, and it
        #     clears the cruise-speed requirement, so no authorised speed sits
        #     above everything measured — a faster one is not flyable;
        #   * an unsampled point breaching the bound would have to depart from
        #     the measured envelope by many times the variation the sweep itself
        #     showed across its whole speed range.
        # The second is a margin argument, not a shape argument. Requiring RMS
        # to rise monotonically instead was tried and rejected: the measured
        # sweep dips 0.0032 deg between two points — 0.6% of a 0.5 deg limit —
        # so monotonicity fits the noise, not the physics, and would refuse a
        # result with 15x margin over a wobble that means nothing.
        swept_speeds = gazebo.get("cruise_attitude_swept_speeds_mps") or []
        swept_rms = gazebo.get("cruise_attitude_swept_rms_deg") or []
        if not swept_rms:
            # cruise_sweep is where the per-point RMS is actually measured; the
            # lists above are a convenience the run may predate. Falling back
            # here is what lets a stored report be re-scored under a corrected
            # criterion without re-flying — which would also re-roll the
            # physics and stop the comparison from isolating the change.
            sweep = sorted(
                (pt for pt in (gazebo.get("cruise_sweep") or [])
                 if pt.get("steady") and pt.get("attitude_rms_deg") is not None),
                key=lambda pt: pt["speed_mps"],
            )
            swept_speeds = [pt["speed_mps"] for pt in sweep]
            swept_rms = [pt["attitude_rms_deg"] for pt in sweep]
        ceiling = gazebo.get("nilwind_dash_speed_mps")
        speed_bar = (_planned_check(planned, "cruise_speed") or {}).get("min_speed_mps")
        spread = (
            max(swept_rms) - min(swept_rms)
            if len(swept_rms) == len(swept_speeds) >= _MIN_SWEEP_POINTS else None
        )
        headroom = limit - rms
        margin_dominates_variation = (
            spread is not None
            and headroom >= _ENVELOPE_MARGIN_FACTOR * spread
            # and the worst point is not merely close to the bound: a result
            # sitting just under the limit closes nothing, however flat.
            and rms <= _ENVELOPE_MARGIN_HEADROOM * limit
        )
        reaches_ceiling = (
            swept and ceiling is not None
            and float(span[1]) >= float(ceiling) - 1e-6
        )
        clears_speed_bar = (
            speed_bar is not None and float(span[1]) >= float(speed_bar)
        )
        envelope_demonstrated = (
            not authority_defined and swept and reaches_ceiling
            and clears_speed_bar and margin_dominates_variation
        )
        closed = authority_covered or envelope_demonstrated
        met = rms <= limit
        covered.add(rid)
        results.append({
            "req_id": rid,
            "check": "cruise_attitude",
            # "At all authorised speeds" is a sweep. The reported RMS is the
            # WORST case over the swept envelope, so a pass covers every point
            # measured — but only a sweep may claim it; one point stays PARTIAL.
            "status": (
                # A breach measured on a certified steady cruise point is a
                # violation of the stated bound whether or not the envelope was
                # closed: the vehicle held that speed in level nil-wind flight,
                # which is the "steady cruise" the requirement talks about.
                # Reporting it as PARTIAL would hide a real violation behind a
                # scope technicality.
                "FAIL" if not met and (closed or swept)
                else "PASS" if met and closed
                else "PARTIAL"
            ),
            "message": (
                f"Gazebo cruise attitude RMS {rms:.3f} deg about the window mean, "
                f"WORST of {points} certified steady speed points spanning "
                + (f"{span[0]:.1f}-{span[1]:.1f} m/s" if len(span) == 2 else "one point")
                + f" (roll {gazebo.get('cruise_attitude_roll_rms_deg'):.3f} / pitch "
                f"{gazebo.get('cruise_attitude_pitch_rms_deg'):.3f} deg, "
                f"n={gazebo.get('cruise_attitude_samples')}; limit {limit:.1f} deg RMS)"
                + (
                    f"; authorised range {authorised[0]:.1f}-{authorised[1]:.1f} "
                    f"m/s from {authority_source} is covered"
                    if authority_covered else
                    f"; no authorised envelope is declared, but the sweep reaches "
                    f"{float(span[1]):.1f} m/s — the fastest speed the vehicle HELD, "
                    f"clearing the {float(speed_bar):.1f} m/s cruise requirement — so no "
                    f"authorised speed lies above everything measured, and RMS varies "
                    f"only {spread:.4f} deg across the whole sweep "
                    f"({', '.join(f'{r:.4f}' for _, r in sorted(zip(swept_speeds, swept_rms)))} "
                    f"deg) against {headroom:.4f} deg of headroom: an unsampled point "
                    f"would have to depart from the measured envelope by "
                    f"{headroom / spread:.0f}x that variation to breach the bound"
                    if envelope_demonstrated else
                    f"; fewer than {_MIN_SWEEP_POINTS} points — 'all authorised speeds' not swept"
                    if not swept else
                    "; authorised-speed envelope is not defined by the requirement "
                    "or generated model, and the sweep does not stand in for it: "
                    + ("no per-point RMS was reported, so the sweep's own "
                       "variation across speed is unknown"
                       if spread is None else
                       "the sweep stops short of the fastest speed the vehicle held"
                       if not reaches_ceiling else
                       "the sweep does not reach the cruise-speed requirement"
                       if not clears_speed_bar else
                       f"the worst point sits at {rms / limit:.0%} of the bound, too "
                       f"close for the sweep to speak for speeds nobody flew"
                       if rms > _ENVELOPE_MARGIN_HEADROOM * limit else
                       f"RMS varies {spread:.4f} deg across the sweep against only "
                       f"{headroom:.4f} deg of headroom, so an unsampled point could "
                       f"plausibly breach the bound")
                    if not authority_defined else
                    f"; measured span does not cover authorised range "
                    f"{authorised[0]:.1f}-{authorised[1]:.1f} m/s from {authority_source}"
                )
            ),
        })

    patt_req = _planned_check(planned, "payload_attitude")
    if (gazebo and patt_req and gazebo.get("hover_attitude_rms_deg") is not None
            and gazebo.get("hover_payload_attached") is True):
        rid = str(patt_req["req_id"])
        rms = float(gazebo["hover_attitude_rms_deg"])
        limit = float(patt_req.get("max_rms_deg") or 0.0)
        margin_req = patt_req.get("min_margin_pct")
        throttle = gazebo.get("hover_throttle_pct")
        margin = None if throttle is None else 100.0 - float(throttle)
        rms_ok = rms <= limit
        margin_ok = (
            margin_req is None
            or (margin is not None and margin >= float(margin_req))
        )
        met = rms_ok and margin_ok
        # "Transport" covers cruise, not just carrying in a hover. The cruise
        # sweep is flown before the payload is released, so when it is a
        # certified sweep with the payload aboard it closes the cruise half;
        # its worst-case RMS must clear the same limit.
        cruise_points = int(gazebo.get("cruise_attitude_points") or 0)
        cruise_span = gazebo.get("cruise_attitude_speed_span_mps") or []
        # Judged on the windows OBSERVED to be carrying, by the same module the
        # flight used — a window measured after separation is excluded however
        # the run was configured, and a run that never looked cannot claim to
        # have been transporting anything.
        transport = evaluate_payload_transport(
            [TransportWindow.from_dict(w) for w in
             (gazebo.get("transport_windows") or [])],
            attitude_limit_deg=limit,
        )
        loaded_windows = transport.sensitivity[0].passed_runs if transport.sensitivity else 0
        transport_swept = (
            transport.status in {"verified", "failed"}
            and loaded_windows >= _MIN_SWEEP_POINTS
        )
        cruise_ok = transport_swept and transport.status == "verified"
        covered.add(rid)
        results.append({
            "req_id": rid,
            "check": "payload_attitude",
            "status": (
                "PASS" if met and cruise_ok
                else "FAIL" if not met or (transport_swept and not cruise_ok)
                else "PARTIAL"
            ),
            "message": (
                f"Gazebo hover with {gazebo.get('payload_mass_kg')} kg payload attached: "
                f"pose-observed attachment distance "
                f"{gazebo.get('hover_payload_attachment_distance_m')} m; "
                f"attitude RMS {rms:.3f} deg (limit {limit:.1f} deg, "
                f"n={gazebo.get('hover_attitude_samples')}), hover throttle "
                f"{throttle}% -> margin {margin if margin is None else round(margin, 1)}%"
                + (f" (required >= {float(margin_req):.0f}%)" if margin_req is not None else "")
                + f"; {transport.description}"
                + (f" ({cruise_points} certified steady cruise points"
                   + (f" spanning {cruise_span[0]:.1f}-{cruise_span[1]:.1f} m/s"
                      if len(cruise_span) == 2 else "") + ")"
                   if cruise_points else "")
                + ("" if transport_swept else
                   f"; fewer than {_MIN_SWEEP_POINTS} windows observed carrying — "
                   "transport across the envelope is not demonstrated")
            ),
        })

    if gazebo and gazebo.get("abort_inhibition_req"):
        rid = str(gazebo["abort_inhibition_req"])
        released = gazebo.get("abort_inhibition_release_detected")
        observer = bool(gazebo.get("abort_inhibition_observer_available"))
        aborted = bool(gazebo.get("abort_inhibition_abort_active"))
        guards_value = gazebo.get("abort_inhibition_release_guards")
        guards_measured = guards_value is not None
        guards = list(guards_value or [])
        unbound = list(gazebo.get("abort_inhibition_flags_unbound") or [])
        raised = dict(gazebo.get("abort_inhibition_flags_raised") or {})
        # A release that fires past a guard the harness never fed says nothing
        # about the model: an unset boolean reads FALSE, so a correctly guarded
        # model behaves exactly like an unguarded one. Only an unguarded
        # transition, or a guard that WAS fed and fired anyway, is a finding
        # against the model.
        told = bool(raised) or not unbound
        blind = bool(guards) and not told
        fired = "; ".join(
            f"{d.get('machine')} {d.get('from_state')}->{d.get('to_state')} "
            f"firing {d.get('action')}"
            for d in (gazebo.get("abort_inhibition_decisions") or [])
        )
        covered.add(rid)
        results.append({
            "req_id": rid,
            "check": "delivery_abort_inhibition",
            "status": (
                "INCONCLUSIVE"
                if not (aborted and observer) or blind or not guards_measured
                else "FAIL" if released else "PASS"
            ),
            "message": (
                "abort condition offered to the generated logic BEFORE the "
                "delivery coordinate; "
                + (f"the model then fired {fired}; " if fired else "")
                + (
                    "the release identity did not resolve on this model, so "
                    "whether the release path is guarded was NOT MEASURED — "
                    "a harness resolution gap, not a model finding"
                    if not guards_measured else
                    f"the release path is guarded by {guards} but this run "
                    f"raised no flag it reads ({unbound} stayed unbound), so an "
                    "unset boolean read FALSE and the guarded model behaved "
                    "exactly like an unguarded one — this says nothing about "
                    "the model"
                    if blind else
                    f"the payload SEPARATED anyway "
                    f"(z {gazebo.get('abort_inhibition_z_before_m')} -> "
                    f"{gazebo.get('abort_inhibition_z_after_m')}) — "
                    + (
                        f"the release transition's guard {guards} was fed "
                        f"{raised} and the transition fired regardless"
                        if guards else
                        "the generated release transition carries no guard, so "
                        "an active abort does not inhibit it"
                    )
                    if released else
                    "the payload remained attached, so the inhibition holds"
                    + (f", with {guards} evaluated against {raised}"
                       if guards else
                       " — though the release transition carries no guard, so "
                       "the hold is not attributable to inhibition logic")
                )
            ),
        })

    nav_req = _planned_check(planned, "navigation_accuracy")
    if gazebo and nav_req:
        rid = str(nav_req["req_id"])
        accepted = gazebo.get("takeoff_command_accepted")
        method = gazebo.get("takeoff_method")
        autonomous = accepted is True and method == "guided_nav_takeoff"
        cep = gazebo.get("cep_m")
        samples = int(gazebo.get("cep_samples") or 0)
        error_source = gazebo.get("cep_horizontal_error_source")
        injected_error = gazebo.get("cep_horizontal_error_injected_m")
        raw_truth_error = gazebo.get("cep_raw_gnss_truth_error_m")
        gnss_represented = (
            error_source == "sim_gps_glitch_xy_campaign"
            and injected_error is not None and float(injected_error) > 0.0
            and raw_truth_error is not None and float(raw_truth_error) > 0.0
        )
        campaign_complete = samples == _CEP_REQUIRED_POINTS
        measurable = (
            autonomous and gnss_represented and campaign_complete and cep is not None
        )
        limit = float(nav_req.get("max_cep_m") or 0.0)
        covered.add(rid)
        results.append({
            "req_id": rid,
            "check": "navigation_accuracy",
            # No CEP is reported unless the vehicle actually navigated to a
            # commanded position. Saying WHY it could not is worth more than a
            # blank row, and it is checkable.
            # CEP is a GNSS-dominated quantity. A rig whose simulated GNSS
            # carries no error cannot support a navigation-accuracy verdict
            # however small the measured error is — that would be a floor
            # reported as a result.
            "status": (
                "PASS" if measurable and float(cep) < limit
                else "FAIL" if measurable
                else "INCONCLUSIVE"
            ),
            "cep_m": cep,
            "message": (
                f"CEP < {nav_req.get('max_cep_m')} m is not measured: autonomous "
                f"position-controlled flight is not available in this rig — the "
                f"autopilot REJECTED MAV_CMD_NAV_TAKEOFF (COMMAND_ACK result="
                f"{gazebo.get('takeoff_command_result')}, 0=accepted) and every "
                f"segment is flown by RC stick in ALT_HOLD (takeoff_method="
                f"{method}). A stick-flown dash cannot demonstrate navigation to "
                "a designated waypoint, so no CEP is reported rather than a "
                "number from a manoeuvre the requirement does not describe"
                if not autonomous else
                f"the horizontal-error campaign completed {samples} of "
                f"{_CEP_REQUIRED_POINTS} commanded global waypoints; all points "
                "are required before computing its median CEP"
                if autonomous and not campaign_complete else
                f"the vehicle flew {samples} commanded waypoints "
                f"autonomously and held them to a median ground-truth error of "
                f"{cep} m (max {gazebo.get('cep_max_error_m')} m, limit "
                f"{nav_req.get('max_cep_m')} m) — but this is NOT a navigation "
                "CEP: no explicit horizontal GNSS error campaign reached the raw "
                "sensor fix relative to Gazebo truth. What is measured is the position loop's tracking "
                "accuracy against ground truth; the GNSS error that dominates a "
                "real CEP is absent, so the requirement stays open"
                if not gnss_represented else
                f"CEP {cep} m over {samples} commanded global waypoints "
                f"(limit {nav_req.get('max_cep_m')} m) with GNSS error represented "
                f"by {error_source}: injected radius {injected_error} m and "
                f"raw-GNSS-to-Gazebo-truth median error {raw_truth_error} m"
            ),
        })

    obstacle_req = _planned_check(planned, "obstacle_avoidance")
    if gazebo and obstacle_req and gazebo.get("obstacle_req"):
        rid = str(obstacle_req["req_id"])
        lidar = bool(gazebo.get("obstacle_lidar_available"))
        detected = bool(
            gazebo.get("obstacle_detected_within_range")
            if "obstacle_detected_within_range" in gazebo
            else gazebo.get("obstacle_detected_within_15m")
        )
        timely = bool(gazebo.get("obstacle_detection_timely"))
        met = bool(gazebo.get("obstacle_avoidance_met"))
        covered.add(rid)
        results.append({
            "req_id": rid,
            "check": "obstacle_avoidance",
            "status": "PASS" if met else "FAIL" if lidar else "INCONCLUSIVE",
            "message": (
                "Gazebo gpu_lidar -> MAVLink DISTANCE_SENSOR -> ArduPilot proximity avoidance: "
                f"lidar={lidar}, detected={detected}, "
                f"first_detection={gazebo.get('obstacle_first_detection_distance_m')} m, "
                f"detection_timely={timely}, "
                f"response_onset={gazebo.get('obstacle_response_onset_distance_m')} m, "
                f"minimum_separation={gazebo.get('obstacle_min_distance_m')} m, "
                f"final_groundspeed={gazebo.get('obstacle_final_groundspeed_mps')} m/s, "
                f"response_observed={gazebo.get('obstacle_response_observed')}, "
                f"scenario_alignment={gazebo.get('obstacle_scenario_alignment')}"
            ),
        })

    for item in planned:
        if item["req_id"] in covered:
            continue
        if item.get("check") == "single_motor_out" and not include_single_motor_out:
            results.append({
                **item,
                "status": "SUSPENDED",
                "message": (
                    "one-motor-out architecture decision is intentionally suspended; "
                    "rerun with --include-single-motor-out to collect it"
                ),
            })
            continue
        if (
            item.get("check") == "obstacle_avoidance"
            and "contract_ready" in item
            and not item.get("contract_ready")
        ):
            results.append({
                **item,
                "status": "INCONCLUSIVE",
                "message": (
                    "obstacle verification contract is incomplete; refusing to invent "
                    "a scenario envelope or oracle: "
                    + "; ".join(item.get("semantic_gaps") or [])
                ),
            })
            continue
        results.append({
            **item,
            "status": "PLANNED",
            "message": item["message"] + "; no implemented Gazebo check has produced evidence yet",
        })
    return results


def _overall_status(live: dict[str, Any] | None, req_results: list[dict[str, Any]], dry_run: bool) -> str:
    if dry_run:
        return "PLANNED"
    if not live or live.get("return_code") not in (0, 7, 8):
        return "FAILED"
    if live.get("return_code") == 8:
        return "GAZEBO_MODEL_UNCALIBRATED"
    if live.get("return_code") == 7:
        return "DYNAMICS_INFEASIBLE"
    statuses = {str(r.get("status", "")).upper() for r in req_results}
    if "FAIL" in statuses:
        return "FAIL"
    if statuses and statuses <= {"PASS"}:
        return "PASS"
    if statuses & {"PASS", "PARTIAL", "INCONCLUSIVE", "PLANNED", "SKIPPED", "SUSPENDED"}:
        return "PARTIAL"
    return "PASS" if live.get("hover_stable") else "INFEASIBLE"


def _write_report(report: dict[str, Any]) -> None:
    atomic_write_json(REPORT_JSON, report)
    g = report.get("gazebo_result") or {}
    lines = [
        "# Gazebo Feasibility Report",
        "",
        "- Honesty redline: Gazebo checks dynamics/trajectory phenomena only; it does not validate datasheet endurance.",
        f"- Status: {report.get('status')}",
        f"- Design source: {report.get('gazebo_design', {}).get('source')}",
        f"- Realized mass: {report.get('gazebo_design', {}).get('mass_kg')} kg",
        f"- Rotor count/radius: {report.get('gazebo_design', {}).get('rotor_count')} / {report.get('gazebo_design', {}).get('rotor_radius_m')} m",
        f"- Motor max thrust: {report.get('gazebo_design', {}).get('max_thrust_g')} g",
        f"- Datasheet hover throttle seed: {report.get('gazebo_design', {}).get('hover_throttle')}",
        "",
        "## Dynamics",
        f"- Hover stable: {g.get('hover_stable')}",
        f"- Failure kind: {g.get('failure_kind')}",
        f"- Hover throttle: {g.get('hover_throttle_pct')} %",
        f"- Forward speed: {g.get('fwd_speed_mps')} m/s",
        f"- Backed-out drag area: {g.get('drag_area_m2')} m^2",
        f"- RPM cross-check: diff={g.get('rpm_pct_diff')} %, within Ct band={g.get('rpm_within_ct_band')}",
        f"- Thrust diagnostics: {g.get('thrust_diagnostics')}",
        f"- Wind check: requested={g.get('wind_requested_mps')} m/s, "
        f"groundspeed={g.get('wind_groundspeed_mps')} m/s, "
        f"alignment={g.get('wind_headwind_alignment')}",
        f"- Wind fidelity: {g.get('wind_fidelity')}",
        f"- Payload release: detected={g.get('payload_release_detected')}, "
        f"observer={g.get('payload_observer_available')}, "
        f"delay={g.get('payload_release_delay_s')} s, mass={g.get('payload_mass_kg')} kg",
        f"- Payload release position error: {g.get('payload_release_position_error_m')} m "
        f"(met={g.get('payload_release_position_met')})",
        f"- Parachute deployment: model_observed={g.get('parachute_model_observed')}, "
        f"delay={g.get('parachute_deploy_delay_s')} s",
        f"- Nil-wind dash: {g.get('nilwind_dash_speed_mps')} m/s mean "
        f"(peak {g.get('nilwind_dash_peak_mps')} m/s)",
        f"- Cruise attitude RMS: {g.get('cruise_attitude_rms_deg')} deg "
        f"(n={g.get('cruise_attitude_samples')}, "
        f"~{g.get('cruise_attitude_mean_speed_mps')} m/s)",
        f"- Hover attitude RMS: {g.get('hover_attitude_rms_deg')} deg "
        f"(with_payload={g.get('hover_attitude_with_payload')}, "
        f"n={g.get('hover_attitude_samples')})",
        f"- Coordinate→separation delay: "
        f"{g.get('payload_coordinate_to_separation_delay_s')} s",
        f"- Obstacle avoidance: lidar={g.get('obstacle_lidar_available')}, "
        f"minimum_separation={g.get('obstacle_min_distance_m')} m, "
        f"final_groundspeed={g.get('obstacle_final_groundspeed_mps')} m/s, "
        f"met={g.get('obstacle_avoidance_met')}",
        "",
        "## Requirement Results",
    ]
    for item in report.get("req_results", []):
        lines.append(f"- {item['req_id']}: {item['status']} ({item['check']}) — {item['message']}")
    atomic_write_text(REPORT_MD, "\n".join(lines) + "\n")


def build_report(dry_run: bool = False, include_single_motor_out: bool = False) -> dict[str, Any]:
    run = _load_run()
    if SYSML_PATH.exists():
        model_sysml = SYSML_PATH.read_text(encoding="utf-8")
        fresh, reason = validate_run_provenance(run, model_sysml=model_sysml)
        if not fresh:
            raise RuntimeError(f"STALE artifact set: {reason}")
        requirements = _requirements_from_sysml(model_sysml)
        requirements_source = str(SYSML_PATH)
    else:
        raise FileNotFoundError(
            f"{SYSML_PATH} not found; Gazebo must use the exact model from the realization run"
        )
    planned = _planned_gazebo_reqs(requirements)
    if not dry_run:
        previous_run = None
        previous_model = None
        previous_report = None
        try:
            previous_dir = latest_output_dir()
            if previous_dir != INPUT.resolve():
                previous_run = json.loads(
                    (previous_dir / "realization_run.json").read_text(
                        encoding="utf-8"
                    )
                )
                previous_model = (previous_dir / "final_model.sysml").read_text(
                    encoding="utf-8"
                )
                previous_report = json.loads(
                    (previous_dir / "gazebo_feasibility_report.json").read_text(
                        encoding="utf-8"
                    )
                )
        except FileNotFoundError:
            previous_dir = INPUT.resolve()
        if previous_dir != INPUT.resolve():
            current_ids = {
                str(item.get("req_id") or "").upper().replace("-", "_")
                for item in planned
            }
            previous_pass_ids = {
                str(item.get("req_id") or "").upper().replace("-", "_")
                for item in previous_report.get("req_results", ())
                if str(item.get("status") or "").upper() == "PASS"
            }
            fresh, _ = validate_derived_provenance(
                previous_report, previous_run, previous_model
            )
            if (
                fresh
                and previous_report.get("status") == "PASS"
                and current_ids == previous_pass_ids
                and evidence_reuse_allowed(
                    previous_run,
                    run,
                    previous_model,
                    model_sysml,
                    current_ids,
                    require_parm_match=False,
                )
            ):
                report = dict(previous_report)
                report.update({
                    "generated_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                    "requirements_source": requirements_source,
                    "recommended_design_inputs": run.get(
                        "recommended_design_inputs"
                    ),
                    "source_provenance": run.get("artifact_provenance"),
                    "evidence_origin_provenance": previous_run.get(
                        "artifact_provenance"
                    ),
                    "reused_from_run_id": (
                        previous_run.get("artifact_provenance") or {}
                    ).get("run_id"),
                    "req_results": [
                        {**item, "evidence_reused": True}
                        for item in previous_report.get("req_results", ())
                    ],
                })
                _write_report(report)
                return report
    gazebo_design = _gazebo_design_from_run(run)
    live = None if dry_run else _run_live_gazebo(
        gazebo_design, planned, include_single_motor_out=include_single_motor_out,
        # the model under verification owns the mission decisions; without this
        # the harness decides and the evidence can only speak for the physics
        mission_model_text=model_sysml,
    )
    req_results = _req_results(
        live, planned, include_single_motor_out=include_single_motor_out
    )
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": _overall_status(live, req_results, dry_run=dry_run),
        "dry_run": dry_run,
        "include_single_motor_out": include_single_motor_out,
        "honesty_redline": (
            "Gazebo verifies high-fidelity dynamics such as hover stability, forward dash, "
            "and one-motor-out behavior. Datasheet endurance/mass closure remains Phase 8; "
            "do not treat Gazebo as endurance validation."
        ),
        "requirements_source": requirements_source,
        "recommended_design_inputs": run.get("recommended_design_inputs"),
        "source_provenance": run.get("artifact_provenance"),
        "gazebo_design": gazebo_design,
        "gazebo_result": live or {},
        "req_results": req_results,
    }
    _write_report(report)
    return report


def reprocess_existing_report() -> dict[str, Any]:
    """Re-evaluate requirement statuses from an existing raw Gazebo result.

    This is useful when report classification or an observer parser is fixed:
    it never starts Docker and never invents measurements that were not captured.
    """
    if not REPORT_JSON.exists():
        raise FileNotFoundError(f"{REPORT_JSON} not found")
    report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    run = _load_run()
    if not SYSML_PATH.exists():
        raise FileNotFoundError(f"{SYSML_PATH} not found")
    model_sysml = SYSML_PATH.read_text(encoding="utf-8")
    from src.prototyping.artifact_provenance import validate_derived_provenance
    fresh, reason = validate_derived_provenance(report, run, model_sysml)
    if not fresh:
        raise RuntimeError(f"STALE artifact set: {reason}")
    if SYSML_PATH.exists():
        requirements = _requirements_from_sysml(model_sysml)
    else:
        from examples.drone_system_v2 import DRONE_REQUIREMENTS
        requirements = list(DRONE_REQUIREMENTS)
    planned = _planned_gazebo_reqs(requirements)
    live = report.get("gazebo_result") or {}
    include = bool(report.get("include_single_motor_out"))
    results = _req_results(live, planned, include_single_motor_out=include)
    report["req_results"] = results
    report["status"] = _overall_status(live, results, dry_run=False)
    report["reprocessed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["reprocess_note"] = "statuses recomputed from captured raw result; no Gazebo rerun"
    _write_report(report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Build the Gazebo plan/report without launching Docker/Gazebo.")
    # One-motor-out is collected by DEFAULT since 2026-07-16 (user decision:
    # unsuspended — a hexa surviving one motor out is a core safety claim).
    # --skip-single-motor-out restores the previous suspended behaviour.
    parser.add_argument(
        "--include-single-motor-out",
        dest="include_single_motor_out",
        action="store_true",
        default=True,
        help="Collect the one-motor-out architecture check (default: on).",
    )
    parser.add_argument(
        "--skip-single-motor-out",
        dest="include_single_motor_out",
        action="store_false",
        help="Suspend the one-motor-out check (pre-2026-07-16 default).",
    )
    parser.add_argument(
        "--reprocess-existing",
        action="store_true",
        help="Recompute statuses from the existing raw report without launching Docker.",
    )
    args = parser.parse_args(argv)
    ensure_open_bundle(OUT)
    report = (
        reprocess_existing_report()
        if args.reprocess_existing
        else build_report(
            dry_run=args.dry_run,
            include_single_motor_out=args.include_single_motor_out,
        )
    )
    print(json.dumps({
        "status": report["status"],
        "dry_run": report["dry_run"],
        "req_results": report["req_results"],
        "report_json": str(REPORT_JSON),
    }, indent=2, ensure_ascii=False), flush=True)
    return 0 if report["status"] in {
        "PASS", "PARTIAL", "INFEASIBLE", "DYNAMICS_INFEASIBLE",
        "GAZEBO_MODEL_UNCALIBRATED", "PLANNED",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
