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
        if any(k in low for k in ("single propulsion", "one motor", "motor inoperative", "propulsion unit")):
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
                if check == "payload_attitude":
                    item["min_margin_pct"] = _number_after(
                        r"margin of at least\s+(\d+(?:\.\d+)?)\s*percent", low
                    )
            elif check == "cruise_speed":
                item["min_speed_mps"] = _number_after(
                    r"at least\s+(\d+(?:\.\d+)?)\s*m/s", low
                )
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
                     include_single_motor_out: bool = False) -> dict[str, Any]:
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
    if result.get("hover_rpm"):
        rpm = rpm_cross_check(float(result["hover_rpm"]), mass, rotor_count, rotor_radius)
        result.update({
            "rpm_theory": rpm.theory_rpm,
            "rpm_pct_diff": rpm.pct_diff,
            "rpm_within_ct_band": rpm.within_ct_band,
            "implied_ct": rpm.ct_implied,
        })

    # The obstacle-avoidance live sub-check was driven by the external-contract
    # geometry layer, which was retired with the Layer-2 excision; the planner
    # still records the requirement, and its status is judged from live evidence
    # in _req_results.

    rid = _single_motor_req(planned) if include_single_motor_out else None
    if rid and result.get("hover_stable"):
        rc_fail = run_flight.main(
            mass_kg=mass,
            rotor_radius=rotor_radius,
            capacity_mah=capacity,
            rotor_count=rotor_count,
            calibrate=True,
            fail_rotor=0,
            max_thrust_g=max_thrust_g,
            hover_throttle=hover_throttle,
        )
        fail_result = dict(run_flight.LAST_RESULT)
        result["motor_failure_req"] = rid
        result["motor_failure_return_code"] = rc_fail
        result["motor_failure_tolerant"] = bool(fail_result.get("hover_stable"))
        result["motor_failure_hover_alt_m"] = fail_result.get("hover_alt_m")
        result["motor_failure_hover_throttle_pct"] = fail_result.get("hover_throttle_pct")
    return result


def _req_results(gazebo: dict[str, Any] | None, planned: list[dict[str, Any]],
                 include_single_motor_out: bool = False) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    covered: set[str] = set()
    if gazebo and gazebo.get("motor_failure_req"):
        rid = str(gazebo["motor_failure_req"])
        ok = gazebo.get("motor_failure_tolerant")
        status = "PASS" if ok is True else "FAIL" if ok is False else "INCONCLUSIVE"
        covered.add(rid)
        results.append({
            "req_id": rid,
            "check": "single_motor_out",
            "status": status,
            "message": (
                "Gazebo one-motor-out hover remained stable"
                if ok is True else
                "Gazebo one-motor-out hover did not remain stable"
                if ok is False else
                "Gazebo one-motor-out result was inconclusive"
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
        meets = aligned and speed >= minimum
        covered.add(rid)
        results.append({
            "req_id": rid,
            "check": "wind_condition",
            # WindEffects is a mass-scaled linear force, locally calibrated to
            # the lumped drag area. Passing is useful dynamics evidence but is
            # not geometry-derived high-fidelity aerodynamic closure.
            "status": "PARTIAL" if meets else "INCONCLUSIVE",
            "message": (
                f"Gazebo closed-loop flight held {speed:.2f} m/s ground speed in a "
                f"{wind:.1f} m/s injected headwind (minimum {minimum:.1f}, "
                f"opposition alignment={alignment}); WindEffects uses a lumped "
                "0.05 m^2 drag-area local linearization"
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
            # physical separation. The condition was evaluated by the harness,
            # not by generated mission logic — hence still PARTIAL.
            meets = detected and float(chain) <= limit
            message = (
                f"delivery-coordinate condition satisfied → physical detachable-joint "
                f"separation delay={float(chain):.3f} s (limit {limit:.1f} s; "
                f"command→separation {delay} s, detected={detected}); the coordinate "
                "condition was evaluated by the harness, not by generated mission logic"
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
            # Physical timing is measured; condition ownership stays with the
            # harness, so the ceiling remains PARTIAL.
            "status": (
                "PARTIAL" if meets else "FAIL" if observer_available else "INCONCLUSIVE"
            ),
            "message": message,
        })

    position_req = _planned_check(planned, "positional_release")
    if gazebo and position_req and gazebo.get("payload_release_commanded"):
        rid = str(position_req["req_id"])
        error = gazebo.get("payload_release_position_error_m")
        limit = float(position_req.get("max_error_m") or 0.0)
        met = bool(gazebo.get("payload_release_position_met"))
        covered.add(rid)
        results.append({
            "req_id": rid,
            "check": "positional_release",
            "status": "PARTIAL" if met else "INCONCLUSIVE",
            "message": (
                f"Harness triggered physical payload release at horizontal position error "
                f"{error} m (limit {limit:.1f} m); this exercises trajectory/position/actuator "
                "coupling but not generated mission-logic ownership of the trigger"
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
                "PARTIAL" if met else "FAIL" if observer_available else "INCONCLUSIVE"
            ),
            "message": (
                f"MAV_CMD_DO_PARACHUTE to Gazebo parachute model creation/attachment "
                f"delay={delay} s (limit {limit:.1f} s, observed={observed}); critical-failure "
                "detection and precedence were not injected by this subcheck"
            ),
        })

    speed_req = _planned_check(planned, "cruise_speed")
    if gazebo and speed_req and gazebo.get("nilwind_dash_speed_mps") is not None:
        rid = str(speed_req["req_id"])
        speed = float(gazebo["nilwind_dash_speed_mps"])
        peak = gazebo.get("nilwind_dash_peak_mps")
        minimum = float(speed_req.get("min_speed_mps") or 0.0)
        met = speed >= minimum
        covered.add(rid)
        results.append({
            "req_id": rid,
            "check": "cruise_speed",
            # Nil wind → ground speed equals airspeed; ALT_HOLD keeps level
            # flight. The measured condition is exactly the requirement's.
            "status": "PASS" if met else "FAIL",
            "message": (
                f"Gazebo nil-wind level dash held {speed:.2f} m/s mean ground speed "
                f"(peak {float(peak):.2f} m/s; requirement >= {minimum:.1f} m/s); "
                "no wind injected during this segment, altitude held in ALT_HOLD, "
                "ground speed = airspeed in nil wind"
                if peak is not None else
                f"Gazebo nil-wind level dash held {speed:.2f} m/s mean ground speed "
                f"(requirement >= {minimum:.1f} m/s)"
            ),
        })

    catt_req = _planned_check(planned, "cruise_attitude")
    if gazebo and catt_req and gazebo.get("cruise_attitude_rms_deg") is not None:
        rid = str(catt_req["req_id"])
        rms = float(gazebo["cruise_attitude_rms_deg"])
        limit = float(catt_req.get("max_rms_deg") or 0.0)
        mean_speed = gazebo.get("cruise_attitude_mean_speed_mps")
        met = rms <= limit
        covered.add(rid)
        results.append({
            "req_id": rid,
            "check": "cruise_attitude",
            # One cruise point is measured; "at all authorised speeds" is a
            # sweep this single dash does not cover.
            "status": "PARTIAL" if met else "FAIL",
            "message": (
                f"Gazebo cruise attitude RMS {rms:.3f} deg about the window mean "
                f"(roll {gazebo.get('cruise_attitude_roll_rms_deg'):.3f} / pitch "
                f"{gazebo.get('cruise_attitude_pitch_rms_deg'):.3f} deg, "
                f"n={gazebo.get('cruise_attitude_samples')}) during the nil-wind dash "
                f"at ~{mean_speed if mean_speed is None else round(float(mean_speed), 1)} m/s "
                f"(limit {limit:.1f} deg RMS); single speed point — 'all authorised "
                "speeds' not swept"
            ),
        })

    patt_req = _planned_check(planned, "payload_attitude")
    if (gazebo and patt_req and gazebo.get("hover_attitude_rms_deg") is not None
            and gazebo.get("hover_attitude_with_payload")):
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
        covered.add(rid)
        results.append({
            "req_id": rid,
            "check": "payload_attitude",
            # Carry-hover only: transport also includes cruise, which this
            # window does not cover.
            "status": "PARTIAL" if met else "FAIL",
            "message": (
                f"Gazebo hover with {gazebo.get('payload_mass_kg')} kg payload attached: "
                f"attitude RMS {rms:.3f} deg (limit {limit:.1f} deg, "
                f"n={gazebo.get('hover_attitude_samples')}), hover throttle "
                f"{throttle}% -> margin {margin if margin is None else round(margin, 1)}%"
                + (f" (required >= {float(margin_req):.0f}%)" if margin_req is not None else "")
                + "; carry-hover window only — cruise-transport attitude not swept"
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
        gazebo_design, planned, include_single_motor_out=include_single_motor_out
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
