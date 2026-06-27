"""Gazebo hi-fi PoC stage 3 — fly OUR airframe and read the trimmed hover (dynamics check).

Generates the parametric SDF, starts headless_gazebo with our models mounted over the iris
originals, launches ArduCopter SITL on the JSON FDM, arms + takes off + hovers, and reports the
trimmed hover throttle and altitude-hold stability. This is the DYNAMICS result (can our
mass/inertia airframe fly stably) — endurance comes from the datasheet model, not here.

Honest risk: on Docker Desktop (macOS) the FDM return path (plugin fdm_addr=127.0.0.1) may not
reach the host SITL; if so EKF never gets a position and we report that as an environment blocker
rather than a flight result.

Run: PYTHONPATH=. python gazebo_poc/run_flight.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from gazebo_poc.sdf_generator import generate_sdf

_IMG = "headless_gazebo"
_CONTAINER = "ai_prototyping_gazebo"
_HOME = "-35.363262,149.165237,584,0"
_ARDUCOPTER = os.path.expanduser("~/PycharmProjects/ardupilot/build/sitl/bin/arducopter")
# gz resolves model:// via GZ_SIM_RESOURCE_PATH=/ardupilot_gazebo/models — NOT the
# /usr/local/share copy (mounting there is a no-op; this was a real bug found in audit).
_MODEL_BASE = "/ardupilot_gazebo/models"
_PARM = """FRAME_CLASS 1
FRAME_TYPE 1
ARMING_CHECK 0
DISARM_DELAY 0
EK3_CHECK_SCALE 100
GPS_TYPE 1
GPS1_TYPE 1
SIM_GPS_DISABLE 0
SIM_GPS1_ENABLE 1
SIM_GPS_NUMSATS 18
EK3_SRC1_POSXY 3
EK3_SRC1_VELXY 3
EK3_SRC1_POSZ 1
COMPASS_USE 0
COMPASS_USE2 0
COMPASS_USE3 0
EK3_SRC1_YAW 0
INS_ACC_BODYFIX 0
"""


def _sh(*args, **kw):
    return subprocess.run(args, capture_output=True, text=True, **kw)


def _parse_rotor_velocity(path) -> float:
    """Mean |velocity| (rad/s) over all rotor_*_joint entries in a captured gz joint_state dump."""
    per = _parse_rotor_velocities(path)
    allv = [w for ws in per.values() for w in ws]
    return sum(allv) / len(allv) if allv else 0.0


def _parse_rotor_velocities(path):
    """{rotor_joint_name: [|velocity| rad/s, ...]} from a gz joint_state dump (per-rotor, so power
    ∝ n³ can be summed correctly — the mean underestimates it in forward flight)."""
    import re
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return {}
    per, cur = {}, None
    for ln in lines:
        s = ln.strip()
        m = re.match(r'name:\s*"(rotor_\d_joint)"', s)
        if m:
            cur = m.group(1); continue
        if s.startswith("velocity:") and cur:
            try:
                per.setdefault(cur, []).append(abs(float(s.split(":", 1)[1])))
            except ValueError:
                pass
            cur = None
    return per


def _per_rotor_power_w(path, diameter_m):
    """Total shaft power Σ Cp·ρ·n_i³·D⁵ from per-rotor mean speeds (last samples)."""
    from gazebo_poc.prop_theory import mechanical_power_w
    per = _parse_rotor_velocities(path)
    total = 0.0
    for ws in per.values():
        tail = ws[-50:] if ws else []
        if tail:
            rpm = (sum(tail) / len(tail)) * 9.5493
            total += mechanical_power_w(rpm, diameter_m)
    return total


def _cleanup(proc):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=6)
        except subprocess.TimeoutExpired:
            proc.kill()
    _sh("docker", "stop", _CONTAINER)


# last flight's measurements, for programmatic callers (gazebo_verify / pipeline integration)
LAST_RESULT: dict = {}


def main(mass_kg=5.5, rotor_radius=0.19, capacity_mah=16000, area_override=None) -> int:
    LAST_RESULT.clear()
    out = Path("gazebo_poc/generated")
    g = generate_sdf(mass_kg, 4, rotor_radius, Path("gazebo_poc/templates"), out,
                     area_override=area_override)
    print(f"[gen] mass={g.mass_kg}kg inertia={tuple(round(x,4) for x in g.inertia)} "
          f"area={0.002*g.area_scale:.6f} (scale={g.area_scale:.2f})", flush=True)

    _sh("docker", "rm", "-f", _CONTAINER)
    # Mount the individual model.sdf FILES (not the dirs) so the original meshes/config in the
    # image are preserved — mounting the whole dir hides iris_collision.stl → gz fails to load.
    so = str((out / "iris_with_standoffs" / "model.sdf").resolve())
    gm = str((out / "iris_with_gimbal" / "model.sdf").resolve())
    run = _sh("docker", "run", "-d", "--name", _CONTAINER, "-p", "9002:9002/udp",
              "-v", f"{so}:{_MODEL_BASE}/iris_with_standoffs/model.sdf",
              "-v", f"{gm}:{_MODEL_BASE}/iris_with_gimbal/model.sdf", _IMG)
    if run.returncode != 0:
        print("[docker] failed:", run.stderr, flush=True)
        return 2
    print("[docker] container up, gz sim loading our airframe ...", flush=True)
    time.sleep(8)

    parm = out / "poc.parm"
    parm.write_text(_PARM)
    if not Path(_ARDUCOPTER).exists():
        print("[sitl] arducopter binary missing:", _ARDUCOPTER, flush=True)
        _cleanup(None)
        return 2
    proc = subprocess.Popen(
        [_ARDUCOPTER, "--model", "JSON", "--speedup", "1", "-I0",
         "--home", _HOME, "--wipe", "--defaults", str(parm)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"[sitl] arducopter pid={proc.pid}, connecting ...", flush=True)
    time.sleep(5)                       # let arducopter bind TCP 5760 before we connect

    try:
        from pymavlink import mavutil
        m = mavutil.mavlink_connection("tcp:127.0.0.1:5760")
        if not m.wait_heartbeat(timeout=20):
            print("[sitl] no heartbeat", flush=True); _cleanup(proc); return 3
        m.mav.request_data_stream_send(m.target_system, m.target_component,
                                       mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1)
        ABS = getattr(mavutil.mavlink, "EKF_POS_HORIZ_ABS", 16)
        print("[sitl] heartbeat ok, waiting for EKF absolute position (FDM loop) ...", flush=True)
        ready, deadline = False, time.time() + 75
        while time.time() < deadline:
            s = m.recv_match(type="EKF_STATUS_REPORT", blocking=True, timeout=2)
            if s and (s.flags & ABS):
                ready = True; break
        if not ready:
            print("[RESULT] FDM loop did NOT close (no EKF position) — likely the Docker UDP "
                  "return path on macOS. Environment blocker, not a flight result.", flush=True)
            _cleanup(proc); return 4
        print("[sitl] EKF ready — FDM loop closed. ALT_HOLD + arm + climb (baro, no GPS) ...",
              flush=True)

        def hb():
            return m.recv_match(type="HEARTBEAT", blocking=True, timeout=2)

        def wait(cond, secs, label):
            end = time.time() + secs
            while time.time() < end:
                h = hb()
                if h and cond(h):
                    return True
            print(f"[sitl] timeout waiting for {label}", flush=True)
            return False

        def drain_status():
            while True:
                s = m.recv_match(type="STATUSTEXT", blocking=False)
                if not s:
                    break
                print(f"[sitl] STATUSTEXT: {s.text}", flush=True)

        def rc(throttle, pitch=1500):           # roll=neutral, ch2=pitch, ch3=throttle, yaw=neutral
            m.mav.rc_channels_override_send(m.target_system, m.target_component,
                                            1500, pitch, throttle, 1500, 0, 0, 0, 0)

        ALT_HOLD = 2
        rc(1000)                                # throttle low before arming
        m.set_mode("ALT_HOLD")
        if not wait(lambda h: h.custom_mode == ALT_HOLD, 8, "ALT_HOLD mode"):
            _cleanup(proc); return 5
        # let EKF finish tilt alignment before arming (else arm is rejected)
        print("[sitl] in ALT_HOLD; settling EKF before arm ...", flush=True)
        for _ in range(10):
            rc(1000); time.sleep(1); drain_status()

        ARMED = mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED

        def is_armed():
            for _ in range(6):                  # poll several heartbeats
                h = hb()
                if h and (h.base_mode & ARMED):
                    return True
            return False

        armed = False
        for attempt, p2 in (("normal", 0), ("force", 21196), ("force", 21196)):
            rc(1000)
            m.mav.command_long_send(m.target_system, m.target_component,
                                    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
                                    1, p2, 0, 0, 0, 0, 0)
            time.sleep(2); drain_status()
            if is_armed():
                armed = True; break
        if not armed:
            print("[RESULT] failed to arm", flush=True); _cleanup(proc); return 5
        print("[sitl] ARMED. climbing (throttle up) ...", flush=True)

        alt0 = None

        def baro_alt():
            v = m.recv_match(type="VFR_HUD", blocking=True, timeout=2)
            return v.alt if v else None

        a = baro_alt()
        alt0 = a if a is not None else 0.0

        # closed-loop altitude hold at 10 m via stick nudges (robust to RC-trim offset after
        # --wipe): below target → stick up, above → stick down, within band → neutral-ish.
        TGT = 10.0

        def alt_hold_stick(rel):
            err = TGT - rel
            if err > 1.0:
                return 1650
            if err < -1.0:
                return 1350
            return 1500 + int(max(-120, min(120, err * 120)))   # gentle proportional trim

        peak = 0.0
        thr, rels = [], []
        cap_proc, cap_path = None, Path("gazebo_poc/generated/jointstate.txt")
        topic = ("/world/iris_runway/model/iris_with_gimbal/model/"
                 "iris_with_standoffs/joint_state")    # nested model — has rotor_*_joint
        t_end = time.time() + 40
        while time.time() < t_end:
            v = m.recv_match(type="VFR_HUD", blocking=True, timeout=2)
            if v is None:
                continue
            rel = v.alt - alt0
            peak = max(peak, rel)
            rc(alt_hold_stick(rel))
            if time.time() > t_end - 18:          # sample steady-state in the final 18 s
                thr.append(v.throttle); rels.append(rel)
                if cap_proc is None:              # capture rotor RPM while hovering
                    cap_proc = subprocess.Popen(
                        ["docker", "exec", _CONTAINER, "gz", "topic", "-e", "-t", topic],
                        stdout=open(cap_path, "w"), stderr=subprocess.DEVNULL)
        if cap_proc:
            cap_proc.terminate()
        rotor_rad_s = _parse_rotor_velocity(cap_path)
        hover_rpm = rotor_rad_s * 9.5493
        print(f"[sitl] climb peak={peak:.2f} m; steady hover sampled. "
              f"rotor |omega|~{rotor_rad_s:.1f} rad/s ({hover_rpm:.0f} RPM)", flush=True)

        # --- forward-flight dash: pitch forward, hold altitude, measure speed + rotor RPM ---
        # (stage 6b — the regime where Gazebo beats hover thrust=weight; NB body drag is iris-shaped)
        print("[sitl] forward dash (pitch fwd, hold alt) ...", flush=True)
        fwd_cap = Path("gazebo_poc/generated/jointstate_fwd.txt")
        fcap, spds = None, []
        t_end = time.time() + 18
        while time.time() < t_end:
            v = m.recv_match(type="VFR_HUD", blocking=True, timeout=2)
            if v is None:
                continue
            rel = v.alt - alt0
            rc(alt_hold_stick(rel), pitch=1330)     # nose down → fly forward
            if time.time() > t_end - 9:             # steady-state last 9 s
                spds.append(v.groundspeed)
                if fcap is None:
                    fcap = subprocess.Popen(
                        ["docker", "exec", _CONTAINER, "gz", "topic", "-e", "-t", topic],
                        stdout=open(fwd_cap, "w"), stderr=subprocess.DEVNULL)
        if fcap:
            fcap.terminate()
        fwd_rad_s = _parse_rotor_velocity(fwd_cap)
        fwd_rpm = fwd_rad_s * 9.5493
        ns = max(1, len(spds) // 2)
        fwd_speed = sum(spds[-ns:]) / ns if spds else 0.0
        m.mav.rc_channels_override_send(m.target_system, m.target_component, *([0] * 8))
        _cleanup(proc)

        # per-rotor mechanical power (Σ Cp·ρ·n³·D⁵), analytical curve, and backed-out drag area
        from gazebo_poc.forward_flight import power_at_speed, effective_drag_area_from_power
        D = 2 * rotor_radius
        p_hover = _per_rotor_power_w(cap_path, D)
        p_fwd = _per_rotor_power_w(fwd_cap, D)
        p_model = power_at_speed(mass_kg, 4, rotor_radius, max(fwd_speed, 0.1)).power_w
        f_eff = effective_drag_area_from_power(p_fwd, max(fwd_speed, 0.1), mass_kg, 4, rotor_radius)
        LAST_RESULT.update(fwd_speed_mps=fwd_speed, fwd_power_w=p_fwd, hover_power_w=p_hover,
                           analytical_fwd_power_w=p_model, drag_area_m2=f_eff)
        print(f"[FWD] speed={fwd_speed:.1f} m/s  hover_rpm={hover_rpm:.0f}  fwd_rpm={fwd_rpm:.0f}",
              flush=True)
        print(f"[FWD] Gazebo power (per-rotor): hover {p_hover:.0f} W → forward {p_fwd:.0f} W  | "
              f"analytical@{fwd_speed:.0f}m/s = {p_model:.0f} W  | backed-out drag area "
              f"f={f_eff:.3f} m²", flush=True)
        if not thr:
            print("[RESULT] armed but no telemetry", flush=True); return 6
        n = max(1, len(thr) // 3)               # steady-state = last third
        hov_thr = sum(thr[-n:]) / n
        hov_alt = sum(rels[-n:]) / n
        band = max(rels[-n:]) - min(rels[-n:])
        flew = peak > 1.0
        stable = flew and band < 2.0
        LAST_RESULT.update(hover_stable=stable, hover_throttle_pct=hov_thr, hover_rpm=hover_rpm,
                           hover_alt_m=hov_alt, mass_kg=mass_kg, rotor_radius_m=rotor_radius,
                           rotor_count=4, capacity_mah=capacity_mah, ok=stable)
        print(f"[RESULT] climb_peak={peak:.2f}m  hover_alt={hov_alt:.2f}m  "
              f"alt_band=±{band/2:.2f}m  hover_throttle={hov_thr:.0f}%", flush=True)
        print(f"[RESULT] dynamics: "
              f"{'STABLE hover @ %.0f%% throttle' % hov_thr if stable else 'did NOT achieve stable hover'}",
              flush=True)
        return 0 if stable else 7
    except Exception as e:
        print("[error]", repr(e), flush=True); _cleanup(proc); return 1


if __name__ == "__main__":
    # optional: python run_flight.py <area_override>  (stage-4 calibrated-thrust flight)
    ao = float(sys.argv[1]) if len(sys.argv) > 1 else None
    sys.exit(main(area_override=ao))
