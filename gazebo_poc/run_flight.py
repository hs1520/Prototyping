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
    """Mean |velocity| (rad/s) of the rotor_*_joint entries in a captured gz joint_state dump.
    The msg lists 'name: "<joint>"' then 'velocity: <x>'; we keep velocities whose preceding
    joint name contains 'rotor'."""
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return 0.0
    vels, cur_rotor = [], False
    for ln in lines:
        s = ln.strip()
        if s.startswith("name:"):
            cur_rotor = "rotor" in s.lower()
        elif s.startswith("velocity:") and cur_rotor:
            try:
                vels.append(abs(float(s.split(":", 1)[1])))
            except ValueError:
                pass
    return sum(vels) / len(vels) if vels else 0.0


def _cleanup(proc):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=6)
        except subprocess.TimeoutExpired:
            proc.kill()
    _sh("docker", "stop", _CONTAINER)


def main(mass_kg=5.5, rotor_radius=0.19, capacity_mah=16000, area_override=None) -> int:
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

        def rc(throttle):                       # roll/pitch=neutral, ch3=throttle, yaw=neutral
            m.mav.rc_channels_override_send(m.target_system, m.target_component,
                                            1500, 1500, throttle, 1500, 0, 0, 0, 0)

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
        print(f"[sitl] climb peak={peak:.2f} m; steady hover sampled. "
              f"rotor |omega|~{rotor_rad_s:.1f} rad/s ({rotor_rad_s*9.5493:.0f} RPM)", flush=True)
        m.mav.rc_channels_override_send(m.target_system, m.target_component, *([0] * 8))
        _cleanup(proc)
        if not thr:
            print("[RESULT] armed but no telemetry", flush=True); return 6
        n = max(1, len(thr) // 3)               # steady-state = last third
        hov_thr = sum(thr[-n:]) / n
        hov_alt = sum(rels[-n:]) / n
        band = max(rels[-n:]) - min(rels[-n:])
        flew = peak > 1.0
        stable = flew and band < 2.0
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
