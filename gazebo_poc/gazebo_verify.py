"""Pipeline entry point (opt-in, heavy): verify the DSE-recommended design in Gazebo.

Gated by RUN_GAZEBO=1 in run_pipeline. Takes the DSE's recommended DesignInputs, flies THAT
airframe in Gazebo (generate SDF → ArduCopter JSON FDM → hover + forward dash), and cross-checks
the measured hover RPM against prop theory + the datasheet hover endurance. ~5 min, needs Docker
+ the arducopter binary. Best-effort: any failure returns a 'skipped'/'failed' verdict, never
breaks the pipeline.

Only quad (rotor_count==4) is supported by the SDF generator at this stage.
"""
from __future__ import annotations

from typing import Any, Dict

from src.dse.physics_estimator import total_mass_kg


def verify_recommended_design(design) -> Dict[str, Any]:
    """Fly the recommended DesignInputs in Gazebo and cross-validate. Returns a result dict with
    'status' in {ok, infeasible, failed, skipped}."""
    if design is None:
        return {"status": "skipped", "reason": "no recommended design from DSE"}
    if getattr(design, "rotor_count", 4) != 4:
        return {"status": "skipped",
                "reason": f"rotor_count={design.rotor_count}: SDF generator supports quad only"}

    mass = total_mass_kg(design)
    try:
        from gazebo_poc import run_flight
        from gazebo_poc.cross_validate import cross_validate
        from gazebo_poc.datasheet_endurance import datasheet_endurance
        from gazebo_poc.prop_theory import rpm_cross_check
    except Exception as e:                      # pragma: no cover - import guard
        return {"status": "failed", "reason": f"import: {e!r}"}

    try:
        run_flight.main(mass_kg=mass, rotor_radius=design.rotor_radius_m,
                        capacity_mah=design.battery_capacity_mah)
    except Exception as e:
        return {"status": "failed", "reason": f"flight: {e!r}", "mass_kg": mass}

    r = dict(run_flight.LAST_RESULT)
    if not r.get("hover_rpm"):
        return {"status": "failed", "reason": "no hover telemetry captured", "mass_kg": mass}

    rpm = rpm_cross_check(r["hover_rpm"], mass, 4, design.rotor_radius_m)
    ds = datasheet_endurance(mass, 4, design.battery_capacity_mah)
    cv = cross_validate(mass, 4, design.battery_capacity_mah, gazebo_stable=r.get("hover_stable", False))
    return {
        "status": "ok" if r.get("hover_stable") else "infeasible",
        "mass_kg": round(mass, 2),
        "hover_stable": r.get("hover_stable"),
        "hover_throttle_pct": round(r.get("hover_throttle_pct", 0), 1),
        "hover_rpm": round(r.get("hover_rpm", 0)),
        "rpm_theory": round(rpm.theory_rpm),
        "rpm_pct_diff": round(rpm.pct_diff, 1),
        "rpm_within_ct_band": rpm.within_ct_band,
        "implied_ct": round(rpm.ct_implied, 3),
        "datasheet_endurance_min": round(ds.endurance_min, 1),
        "forward_speed_mps": round(r.get("fwd_speed_mps", 0), 1),
        "forward_power_w": round(r.get("fwd_power_w", 0)),
        "drag_area_m2": round(r.get("drag_area_m2", 0), 3),
        "cross_validation_consistent": cv.consistent,
    }


def summary_line(v: Dict[str, Any]) -> str:
    if v.get("status") in ("skipped", "failed"):
        return f"Gazebo verify: {v['status']} ({v.get('reason', '')})"
    return (f"Gazebo verify [{v['status']}]: {v['mass_kg']}kg hover "
            f"{'STABLE' if v['hover_stable'] else 'UNSTABLE'} @{v['hover_throttle_pct']}% "
            f"throttle, {v['hover_rpm']} RPM (vs prop theory {v['rpm_theory']}, "
            f"{v['rpm_pct_diff']:+}%, Ct {v['implied_ct']}); datasheet endurance "
            f"{v['datasheet_endurance_min']} min; fwd {v['forward_speed_mps']} m/s "
            f"→ drag f={v['drag_area_m2']} m²; cross-val consistent={v['cross_validation_consistent']}")
