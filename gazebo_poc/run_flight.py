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

import math
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from gazebo_poc.sdf_generator import generate_sdf
from gazebo_poc.payload_transport_evidence import (
    PayloadAttachment,
    TransportWindow,
    observe_attachment,
)
from gazebo_poc.steady_state import steady_state

_IMG = "headless_gazebo"
_CONTAINER = "ai_prototyping_gazebo"
_HOME = "-35.363262,149.165237,584,0"
_HOME_LAT, _HOME_LON = (float(_HOME.split(",")[0]),
                        float(_HOME.split(",")[1]))
_FWD_PITCH = int(os.environ.get("FWD_PITCH", "1330"))   # RC2 for the forward dash (lower = faster)
_ARDUCOPTER = os.path.expanduser("~/PycharmProjects/ardupilot/build/sitl/bin/arducopter")
# gz resolves model:// via GZ_SIM_RESOURCE_PATH=/ardupilot_gazebo/models — NOT the
# /usr/local/share copy (mounting there is a no-op; this was a real bug found in audit).
_MODEL_BASE = "/ardupilot_gazebo/models"
_WORLD_PATH = "/ardupilot_gazebo/worlds/iris_runway.sdf"
_AIR_DENSITY = 1.2041
_DEFAULT_DRAG_AREA_M2 = 0.05
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
MOT_THST_EXPO 0.65
MOT_SPIN_ARM 0.10
MOT_SPIN_MIN 0.15
MOT_THST_HOVER {mot_thst_hover}
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


def _thrust_diagnostics(mass_kg: float, rotor_count: int, area: float,
                        max_rotor_rad_s: float, max_thrust_g: float | None):
    """Analytical thrust-margin check for the generated LiftDrag model.

    This is the "thrust sweep" without needing a second Gazebo run: the generated
    model uses T = kappa * area * omega^2 per rotor, so we can tell whether the
    SDF has enough static thrust before blaming the controller.
    """
    import math
    from gazebo_poc.prop_theory import G, IRIS_LIFTDRAG_KAPPA

    required_per_rotor_n = mass_kg * G / rotor_count
    model_max_per_rotor_n = IRIS_LIFTDRAG_KAPPA * area * max_rotor_rad_s ** 2
    if max_thrust_g is not None:
        datasheet_max_per_rotor_n = max_thrust_g / 1000.0 * G
    else:
        datasheet_max_per_rotor_n = model_max_per_rotor_n
    hover_rad_s = math.sqrt(required_per_rotor_n / (IRIS_LIFTDRAG_KAPPA * area))
    return {
        "required_per_rotor_n": required_per_rotor_n,
        "model_max_per_rotor_n": model_max_per_rotor_n,
        "datasheet_max_per_rotor_n": datasheet_max_per_rotor_n,
        "model_twr": model_max_per_rotor_n * rotor_count / (mass_kg * G),
        "datasheet_twr": datasheet_max_per_rotor_n * rotor_count / (mass_kg * G),
        "hover_rad_s_required": hover_rad_s,
        "hover_rpm_required": hover_rad_s * 9.5493,
        "hover_rad_s_fraction_of_max": hover_rad_s / max_rotor_rad_s if max_rotor_rad_s else None,
        "sdf_can_hover": model_max_per_rotor_n >= required_per_rotor_n,
    }


def _parm_text(frame_class: int, hover_throttle: float | None,
               gripper_servo: int | None = None,
               parachute_servo: int | None = None,
               extra_parms: str = "") -> str:
    hover = 0.5 if hover_throttle is None else max(0.2, min(0.8, float(hover_throttle)))
    text = _PARM.replace("FRAME_CLASS 1", f"FRAME_CLASS {frame_class}").format(
        mot_thst_hover=f"{hover:.3f}"
    )
    if gripper_servo is not None:
        text += (
            "GRIP_ENABLE 1\nGRIP_TYPE 1\n"
            f"SERVO{gripper_servo}_FUNCTION 28\n"
            "GRIP_RELEASE 2000\nGRIP_GRAB 1000\n"
        )
    if parachute_servo is not None:
        text += (
            "CHUTE_ENABLED 1\nCHUTE_TYPE 10\nCHUTE_DELAY_MS 0\n"
            "CHUTE_ALT_MIN 0\nCHUTE_SERVO_ON 2000\nCHUTE_SERVO_OFF 1000\n"
            f"SERVO{parachute_servo}_FUNCTION 27\n"
        )
    if extra_parms:
        # Per-scenario overlay. Position-controlled flight needs a yaw source,
        # which the base set deliberately does not have; that overlay must not
        # leak into the scenarios measured without it.
        text += extra_parms if extra_parms.endswith("\n") else extra_parms + "\n"
    return text


#: A dash is measured over the trailing window once the initial acceleration
#: transient has passed, and it runs until that window is steady or the cap is
#: reached. Reporting a speed from a still-accelerating dash is what made the
#: 2026-08-30 run's "cruise speed" a function of the dash duration.
#: A hover is "controlled" only if the attitude controller is still tracking.
#: Altitude alone cannot say so: a one-motor-out hexa was recorded holding
#: 9.93 m to +/-0.06 m while its attitude RMS was 13.5 deg — a wobble 1500x the
#: nominal 0.009 deg, which no reading of "maintain controlled flight" covers,
#: and which an altitude-only check passed.
#:
#: The bound is an engineering judgement, not a requirement value: an order of
#: magnitude above the 0.5 deg RMS the requirements ask of steady cruise, and
#: far below the tilt authority, so it separates "tracking with reduced margin"
#: from "not tracking". It is reported with every verdict so a reader can
#: disagree with it.
#:
#: This does NOT make one run sufficient. Four runs of the same one-motor-out
#: configuration produced attitude RMS of 1.59, 13.46 and 19.56 deg (one run
#: unmeasured) and steady altitudes of 1.17, 6.27, 9.93 and 10.00 m: the
#: scenario is bistable, and BOTH signals vary. Adding attitude catches a
#: flight altitude alone would pass; settling the requirement needs the
#: scenario repeated and the distribution reported.
_HOVER_ATTITUDE_RMS_LIMIT_DEG = 5.0


_DASH_SETTLE_S = 6.0
_DASH_WINDOW_S = 10.0
#: Raised from 45 s because the low-pitch points need it: at ~10 m/s the drag
#: force is small, the approach to terminal velocity is correspondingly slow,
#: and rc1420 was still at 1.06% drift when a 45 s cap cut it off — losing an
#: envelope point to the clock rather than to the vehicle. The criterion stays
#: where it is; the measurement gets the time it needs to meet it.
_DASH_MAX_S = 75.0

#: RC2 commands swept by the cruise survey, gentle to full forward authority
#: (1500 = neutral, 1100 = full). One stick position answers "how fast is the
#: vehicle at this stick position"; the requirements ask what the vehicle can
#: hold "at all authorised speeds", which is a sweep.
_DASH_SWEEP_PITCH = (1420, 1330, 1220, 1100)


def _trailing(samples, now: float, window_s: float = _DASH_WINDOW_S):
    """The trailing ``window_s`` of ``[(time, value)]``."""
    return [item for item in samples if item[0] >= now - window_s]




def _collect_ned(m, sink) -> None:
    """Append the current NED horizontal velocity, for headwind alignment."""
    pos = m.messages.get("LOCAL_POSITION_NED") if hasattr(m, "messages") else None
    if pos is not None:
        sink.append((float(pos.vx), float(pos.vy)))


def _hold_until_steady(m, rc, alt_hold_stick, alt0, pitch, *, label,
                       attitude=None, on_sample=None, max_s=_DASH_MAX_S):
    """Hold a fixed forward pitch until the trailing speed window plateaus.

    Returns ``(window, verdict, t_window)`` where ``window`` is the trailing
    ``[(time, groundspeed)]`` the verdict was computed over and ``t_window`` is
    its ``(start, end)``. A hold that hits the cap without plateauing returns a
    verdict with ``steady=False``; the caller must then report INCONCLUSIVE
    rather than a speed.
    """
    samples = []
    t_start = time.time()
    t_cap = t_start + max_s
    settled = False
    while time.time() < t_cap:
        v = m.recv_match(type="VFR_HUD", blocking=True, timeout=2)
        if v is None:
            continue
        rc(alt_hold_stick(v.alt - alt0), pitch=pitch)
        now = time.time()
        if now - t_start <= _DASH_SETTLE_S:
            continue                      # discard the acceleration transient
        samples.append((now, float(v.groundspeed)))
        if attitude is not None:
            attitude.sample(m)
        if on_sample is not None:
            on_sample(m)
        if (now - t_start > _DASH_SETTLE_S + _DASH_WINDOW_S
                and steady_state(_trailing(samples, now)).steady):
            settled = True
            break
    window = _trailing(samples, samples[-1][0]) if samples else []
    verdict = steady_state(window)
    span = (window[0][0], window[-1][0]) if window else (t_start, time.time())
    print(f"[dash] {label} pitch={pitch} after {time.time() - t_start:.1f}s: "
          f"{verdict.describe()}{'' if settled else ' [cap reached]'}", flush=True)
    return window, verdict, span


def _wind_force_scale(mass_kg: float, wind_mps: float,
                      drag_area_m2: float = _DEFAULT_DRAG_AREA_M2) -> float:
    """The superseded linearization of quadratic drag at the wind working point.

    Gazebo 8 WindEffects applies ``m*k*(wind-v)``. Choosing k this way makes its
    force at zero groundspeed equal ``0.5*rho*A*wind^2``.

    NOT used in flight any more. Because the force stays linear in ``wind - v``
    while true drag is quadratic in it, this under-predicts drag by a factor of
    ``(wind+v)/wind`` as the vehicle speeds up: the 2026-08-30 run reached
    28.9 m/s against a 15 m/s headwind — faster than it flew with no wind at
    all. The headwind now acts through the airframe's geometry-derived drag
    plate instead (``airframe_drag``), which gz-sim's LiftDrag evaluates against
    airspeed. This is retained so ``wind_force_scale_override`` can reproduce
    the old behaviour for comparison.
    """
    if mass_kg <= 0 or wind_mps <= 0 or drag_area_m2 <= 0:
        return 0.0
    return 0.5 * _AIR_DENSITY * drag_area_m2 * wind_mps / mass_kg


def _wind_world_text(stock_world: str, force_scale: float) -> str:
    if "gz::sim::systems::WindEffects" in stock_world:
        return stock_world
    block = f"""
    <wind><linear_velocity>0 0 0</linear_velocity></wind>
    <plugin filename="gz-sim-wind-effects-system"
      name="gz::sim::systems::WindEffects">
      <force_approximation_scaling_factor>{force_scale:.9f}</force_approximation_scaling_factor>
      <horizontal><magnitude><time_for_rise>0.5</time_for_rise></magnitude>
        <direction><time_for_rise>0.5</time_for_rise></direction></horizontal>
      <vertical><time_for_rise>0.5</time_for_rise></vertical>
    </plugin>
"""
    if "</world>" not in stock_world:
        raise ValueError("iris world has no </world> element")
    return stock_world.replace("</world>", block + "  </world>", 1)


def _prepare_wind_world(out: Path, force_scale: float) -> Path:
    stock = _sh("docker", "run", "--rm", "--entrypoint", "cat", _IMG, _WORLD_PATH)
    if stock.returncode != 0:
        raise RuntimeError(f"cannot read stock Gazebo world: {stock.stderr.strip()}")
    path = out / "iris_runway_wind.sdf"
    path.write_text(_wind_world_text(stock.stdout, force_scale))
    return path


def _vehicle_spawn_heading_deg(stock_world: str) -> float:
    """Read the airframe's world yaw so the obstacle follows its body +X axis."""
    match = re.search(
        r"<include>\s*<uri>model://iris_with_gimbal</uri>\s*"
        r"<pose\s+degrees=[\"']true[\"']>\s*"
        r"[-+\d.eE]+\s+[-+\d.eE]+\s+[-+\d.eE]+\s+"
        r"[-+\d.eE]+\s+[-+\d.eE]+\s+([-+\d.eE]+)\s*</pose>",
        stock_world,
        re.IGNORECASE,
    )
    if not match:
        raise ValueError("iris_with_gimbal spawn pose/yaw not found in Gazebo world")
    return float(match.group(1))


def _obstacle_world_text(stock_world: str, obstacle_distance_m: float = 25.0) -> str:
    import math

    heading_deg = _vehicle_spawn_heading_deg(stock_world)
    heading_rad = math.radians(heading_deg)
    obstacle_x_m = obstacle_distance_m * math.cos(heading_rad)
    obstacle_y_m = obstacle_distance_m * math.sin(heading_rad)
    block = f"""
    <model name="gazebo_test_obstacle">
      <static>true</static><pose degrees="true">{obstacle_x_m:.3f} {obstacle_y_m:.3f} 10 0 0 {heading_deg:.3f}</pose>
      <link name="wall"><collision name="collision"><geometry>
        <box><size>2 6 20</size></box>
      </geometry></collision><visual name="visual"><geometry>
        <box><size>2 6 20</size></box>
      </geometry></visual></link>
    </model>
"""
    if "</world>" not in stock_world:
        raise ValueError("iris world has no </world> element")
    return stock_world.replace("</world>", block + "  </world>", 1)


def _prepare_obstacle_world(out: Path, obstacle_distance_m: float = 25.0) -> Path:
    stock = _sh("docker", "run", "--rm", "--entrypoint", "cat", _IMG, _WORLD_PATH)
    if stock.returncode != 0:
        raise RuntimeError(f"cannot read stock Gazebo world: {stock.stderr.strip()}")
    path = out / "iris_runway_obstacle.sdf"
    path.write_text(_obstacle_world_text(stock.stdout, obstacle_distance_m))
    return path


def _prepare_payload_model(out: Path, payload_mass_kg: float) -> Path:
    src = Path("gazebo_poc/templates/all_models/payload_box/model.sdf").read_text()
    x, y, z = 0.12, 0.08, 0.06
    ixx = payload_mass_kg * (y * y + z * z) / 12.0
    iyy = payload_mass_kg * (x * x + z * z) / 12.0
    izz = payload_mass_kg * (x * x + y * y) / 12.0
    src = re.sub(r"<mass>[^<]+</mass>", f"<mass>{payload_mass_kg:.6f}</mass>", src, count=1)
    src = re.sub(
        r"<ixx>[^<]+</ixx><iyy>[^<]+</iyy><izz>[^<]+</izz>",
        f"<ixx>{ixx:.9f}</ixx><iyy>{iyy:.9f}</iyy><izz>{izz:.9f}</izz>",
        src,
        count=1,
    )
    path = out / "payload_box" / "model.sdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(src)
    return path


def _parse_model_z(output: str) -> float | None:
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    for pattern in (
        rf"XYZ\s*\[\s*{number}\s+{number}\s+({number})\s*\]",
        rf"Pose[^\n]*:\s*\n\s*\[\s*{number}\s+{number}\s+({number})\s*\]",
        rf"position\s*\[\s*{number}[,\s]+{number}[,\s]+({number})\s*\]",
    ):
        match = re.search(pattern, output, re.I)
        if match:
            return float(match.group(1))
    return None


def _parse_model_xyz(output: str) -> tuple[float, float, float] | None:
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    for pattern in (
        rf"XYZ\s*\[\s*({number})\s+({number})\s+({number})\s*\]",
        rf"XYZ\s*\(m\)[^\n]*\n\s*\[\s*({number})\s+({number})\s+({number})\s*\]",
    ):
        match = re.search(pattern, output, re.I)
        if match:
            return tuple(float(match.group(index)) for index in (1, 2, 3))
    return None


def _model_xyz(model: str) -> tuple[float, float, float] | None:
    result = _sh("docker", "exec", _CONTAINER, "gz", "model", "-m", model, "-p")
    return _parse_model_xyz(result.stdout) if result.returncode == 0 else None


#: Single source for the attachment bound — the evidence module owns it,
#: so the distance the harness reports and the distance the verdict uses
#: cannot drift apart.
from gazebo_poc.payload_transport_evidence import ATTACHED_MAX_DISTANCE_M as _PAYLOAD_ATTACHED_MAX_DISTANCE_M


def _payload_attachment_distance() -> float | None:
    vehicle = _model_xyz("iris_with_gimbal")
    payload = _model_xyz("payload_box")
    if vehicle is None or payload is None:
        return None
    return math.dist(vehicle, payload)


def _payload_z() -> float | None:
    result = _sh("docker", "exec", _CONTAINER, "gz", "model", "-m", "payload_box", "-p")
    return _parse_model_z(result.stdout) if result.returncode == 0 else None



def _parse_model_xy(output: str) -> tuple | None:
    """Ground-truth horizontal position from ``gz model -p``."""
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    match = re.search(rf"XYZ\s*\(m\)[^\n]*\]\s*\n\s*\[\s*({number})\s+({number})\s+{number}\s*\]",
                      output, re.I)
    if match:
        return float(match.group(1)), float(match.group(2))
    match = re.search(rf"\[\s*({number})\s+({number})\s+{number}\s*\]", output)
    return (float(match.group(1)), float(match.group(2))) if match else None


def _vehicle_xy() -> tuple | None:
    result = _sh("docker", "exec", _CONTAINER, "gz", "model",
                 "-m", "iris_with_gimbal", "-p")
    return _parse_model_xy(result.stdout) if result.returncode == 0 else None


def _payload_xy() -> tuple | None:
    result = _sh("docker", "exec", _CONTAINER, "gz", "model",
                 "-m", "payload_box", "-p")
    return _parse_model_xy(result.stdout) if result.returncode == 0 else None


#: Waypoints for the navigation-accuracy survey, as (north, east) metres from
#: the hover point. Eight points on a 15 m ring: CEP is a median, so it needs
#: several samples, and a ring exercises every heading rather than one axis.
_CEP_RING_RADIUS_M = 15.0
_CEP_POINTS = 8


def _cep_targets(radius_m: float = _CEP_RING_RADIUS_M, count: int = _CEP_POINTS):
    return [
        (radius_m * math.cos(2 * math.pi * i / count),
         radius_m * math.sin(2 * math.pi * i / count))
        for i in range(count)
    ]


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    if not n:
        return None
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


def _publish_wind(world_x: float, world_y: float) -> bool:
    payload = (
        f"linear_velocity: {{x: {world_x:.6f}, y: {world_y:.6f}, z: 0}}, "
        "enable_wind: true"
    )
    result = _sh(
        "docker", "exec", _CONTAINER, "gz", "topic",
        "-t", "/world/iris_runway/wind", "-m", "gz.msgs.Wind", "-p", payload,
    )
    return result.returncode == 0


def _start_model_observer(model_name: str):
    """Observe the world pose stream and timestamp first appearance of a model."""
    seen = {"at": None}
    event = threading.Event()
    proc = subprocess.Popen(
        [
            "docker", "exec", _CONTAINER, "gz", "topic", "-e",
            "-t", "/world/iris_runway/pose/info",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )

    def read_stream():
        assert proc.stdout is not None
        marker = f'name: "{model_name}"'
        for line in proc.stdout:
            if marker in line:
                seen["at"] = time.monotonic()
                event.set()
                return

    threading.Thread(target=read_stream, daemon=True).start()
    return proc, event, seen


def _lidar_scan_min(ranges, max_range_m: float = 15.0) -> float:
    finite = [float(v) for v in ranges if 0 < float(v) < 1e6]
    return min(finite) if finite else max_range_m


def _obstacle_requirement_met(
    lidar_available: bool,
    detection_timely: bool,
    response_observed: bool,
    response_onset_distance_m: float | None,
    response_threshold_m: float | None,
    min_distance_m: float,
    required_separation_m: float | None,
) -> bool:
    """Evaluate only criteria present in the requirement contract.

    In particular, an "initiate before X" requirement is not silently upgraded
    into "never cross X".  Minimum separation is checked only when the contract
    explicitly contains that invariant.
    """
    response_timely = (
        response_onset_distance_m is not None
        and (
            response_threshold_m is None
            or response_onset_distance_m >= response_threshold_m
        )
    )
    clearance_met = (
        required_separation_m is None
        or min_distance_m >= required_separation_m
    )
    return bool(
        lidar_available
        and detection_timely
        and response_observed
        and response_timely
        and clearance_met
    )


def _start_lidar_observer(max_range_m: float = 15.0):
    """Stream the generated Gazebo lidar and expose its latest minimum range."""
    latest = {"distance_m": None, "updated_at": None, "sample_count": 0}
    proc = subprocess.Popen(
        ["docker", "exec", _CONTAINER, "gz", "topic", "-e", "-t", "/forward_lidar"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )

    def read_stream():
        assert proc.stdout is not None
        ranges = []
        for line in proc.stdout:
            stripped = line.strip()
            if stripped.startswith("ranges:"):
                try:
                    value = float(stripped.split(":", 1)[1])
                    if value == value and value > 0:  # finite/non-NaN checked below
                        ranges.append(value)
                except ValueError:
                    continue
            elif ranges:
                # Gazebo reports +inf when no return is inside sensor range.
                # That is healthy "clear to max range", not missing data.
                latest["distance_m"] = _lidar_scan_min(ranges, max_range_m)
                latest["updated_at"] = time.monotonic()
                latest["sample_count"] += 1
                ranges = []

    threading.Thread(target=read_stream, daemon=True).start()
    return proc, latest


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


def _attitude_rms_deg(samples) -> dict:
    """Roll/pitch RMS deviation about the window mean, in degrees.

    "Attitude deviations ... RMS" in the requirements means variation around the
    trim/commanded attitude, not the absolute pitch of a forward dash — so each
    axis is centred on its own window mean before the RMS is taken.
    """
    import math
    n = len(samples)
    if n < 2:
        return {"n": n, "roll_rms_deg": None, "pitch_rms_deg": None, "rms_deg": None}
    # Samples are (time, roll, pitch[, target_roll, target_pitch]); a bare
    # (roll, pitch) is still accepted.
    def _actual(sample):
        return (sample[1], sample[2]) if len(sample) >= 3 else (sample[0], sample[1])

    def _target(sample):
        return (sample[3], sample[4]) if len(sample) >= 5 else (None, None)

    rolls = [_actual(s)[0] for s in samples]
    pitches = [_actual(s)[1] for s in samples]
    out = {}
    for name, vals in (("roll", rolls), ("pitch", pitches)):
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        out[f"{name}_rms_deg"] = math.degrees(math.sqrt(var))
    out["n"] = n
    # The historical number: deviation about the window's own mean. That is
    # JITTER around whatever attitude the vehicle settled at, and it is not what
    # "attitude deviations within 0.5 degree RMS" asks — a vehicle holding a
    # steady 12 deg error scores zero on it.
    out["jitter_rms_about_window_mean_deg"] = max(
        out["roll_rms_deg"], out["pitch_rms_deg"])

    # Deviation from the attitude the controller was COMMANDED to hold, which is
    # what the requirement bounds. Only available when ATTITUDE_TARGET was
    # sampled; never silently substituted by the jitter figure.
    paired = [s for s in samples if _target(s)[0] is not None]
    if len(paired) >= 2:
        for axis, index in (("roll", 0), ("pitch", 1)):
            errors = [_actual(s)[index] - _target(s)[index] for s in paired]
            mean_error = sum(errors) / len(errors)
            out[f"mean_{axis}_error_deg"] = math.degrees(mean_error)
            out[f"{axis}_rms_about_command_deg"] = math.degrees(
                math.sqrt(sum(e * e for e in errors) / len(errors)))
        out["rms_about_command_deg"] = max(
            out["roll_rms_about_command_deg"], out["pitch_rms_about_command_deg"])
        out["command_samples"] = len(paired)
    else:
        out["rms_about_command_deg"] = None
        out["command_samples"] = len(paired)

    # rms_deg stays the requirement-facing number, and it is the about-command
    # one when it exists. It is None rather than the jitter when it does not:
    # an unmeasured reference is not a small error.
    out["rms_deg"] = out["rms_about_command_deg"]
    return out


def _quaternion_roll_pitch(q) -> tuple:
    """Roll and pitch in radians from a MAVLink (w, x, y, z) quaternion."""
    if q is None or len(q) < 4:
        return None, None
    w, x, y, z = (float(value) for value in q[:4])
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    return roll, math.asin(sin_pitch)


class _AttitudeSampler:
    """Collect ATTITUDE messages off pymavlink's parse cache without stealing
    messages from the blocking recv_match loops (every parsed message lands in
    ``m.messages`` regardless of which filtered read consumed it)."""

    def __init__(self):
        self.samples = []
        self._last_boot_ms = None

    def sample(self, m) -> None:
        att = m.messages.get("ATTITUDE") if hasattr(m, "messages") else None
        if att is None:
            return
        boot_ms = getattr(att, "time_boot_ms", None)
        if boot_ms is not None and boot_ms == self._last_boot_ms:
            return
        self._last_boot_ms = boot_ms
        # The attitude the controller was COMMANDED to hold. Without it only
        # jitter can be computed, and jitter scores a steadily mis-trimmed
        # vehicle as perfect.
        target = m.messages.get("ATTITUDE_TARGET") if hasattr(m, "messages") else None
        target_roll = target_pitch = None
        if target is not None and getattr(target, "q", None):
            target_roll, target_pitch = _quaternion_roll_pitch(target.q)
        self.samples.append((
            time.time(), float(att.roll), float(att.pitch),
            target_roll, target_pitch,
        ))

    def between(self, t0: float, t1: float):
        """Samples inside a time window — used to restrict attitude RMS to the
        segment whose speed was certified steady."""
        return [s for s in self.samples if t0 <= s[0] <= t1]


def main(mass_kg=5.5, rotor_radius=0.19, capacity_mah=16000, area_override=None,
         rotor_count=4, calibrate=False, fail_rotor=None, max_thrust_g=None,
         hover_throttle=None, wind_mps=0.0, wind_min_groundspeed_mps=None,
         payload_release=False, payload_mass_kg=0.0,
         payload_transport=False,
         positional_release=False, positional_tolerance_m=1.0,
         parachute_deploy=False, parachute_max_delay_s=0.5,
         obstacle_avoidance=False, obstacle_detection_range_m=15.0,
         obstacle_response_threshold_m=5.0,
         obstacle_min_separation_m=None,
         obstacle_approach_speed_mps=1.5,
         measure_attitude=False,
         nilwind_dash_s=0.0,   # accepted for compatibility; the cruise
                               # survey now sweeps to steady state instead
         wind_force_scale_override=None,
         mission_model_text=None,
         delivery_abort_before_release=False,
         navigation_accuracy=False,
         gps_horizontal_error_m=None,
         extra_parms="") -> int:
    LAST_RESULT.clear()
    # Carrying a payload and releasing it are different scenarios, and the flight
    # order made that matter: the release block runs BEFORE the cruise survey, so
    # a run configured for release flies every cruise point already empty. The
    # SDF, the payload spawn and the gripper wiring key off "aboard"; only the
    # release action keys off "release".
    payload_aboard = (payload_release or payload_transport) and payload_mass_kg > 0
    # When the generated model is supplied it OWNS the mission decisions: the
    # harness offers events derived from telemetry and actuates only what the
    # model fires. Without it the harness decides, which is honest evidence of
    # physics but says nothing about the generated logic — the two cases are
    # labelled differently so the mapper can tell them apart.
    mission = None
    if mission_model_text:
        from gazebo_poc.model_mission import ModelAction, ModelDrivenMission
        mission = ModelDrivenMission(mission_model_text)
        LAST_RESULT["mission_provenance"] = mission.provenance()
        print(f"[model] mission decisions owned by the generated model: "
              f"{len(mission.machines)} machines, "
              f"{len(mission.accepted_events())} events", flush=True)
    out = Path("gazebo_poc/generated")
    tdir = Path("gazebo_poc/templates")
    frame_class = 1
    if rotor_count == 4 and not calibrate and fail_rotor is None:
        # legacy quad path (in-place iris edit, iris-default aero/T-W) — kept for back-compat.
        # NOTE: it has no fail_rotor support, so any motor-failure test must take the path below.
        g = generate_sdf(mass_kg, 4, rotor_radius, tdir, out, area_override=area_override)
        print(f"[gen] quad mass={g.mass_kg}kg inertia={tuple(round(x,4) for x in g.inertia)} "
              f"area={0.002*g.area_scale:.6f}", flush=True)
        area = 0.002 * g.area_scale
        mult = 838.0
    else:
        from gazebo_poc.multirotor_sdf import generate_multirotor_sdf
        from gazebo_poc.sdf_generator import IRIS_AREA, IRIS_MASS_KG, multirotor_inertia
        inertia = multirotor_inertia(mass_kg, rotor_count, rotor_radius)
        mult = 838.0
        if calibrate:
            # real-motor calibration: area matches the prop's Ct (→ physical hover RPM) and the
            # max rotor speed is set so full throttle = the real motor's max thrust (→ real T/W,
            # so the controller can park a high-rotor-count airframe cleanly).
            from gazebo_poc.component_data import MN5008_KV340_18x61 as motor
            from gazebo_poc.prop_theory import calibrated_area, calibrated_max_rad_s
            area = calibrated_area(2 * rotor_radius)
            thrust_g = motor.max_thrust_g() if max_thrust_g is None else max_thrust_g
            mult = calibrated_max_rad_s(area, thrust_g / 1000.0 * 9.81)
        else:
            area = (area_override if area_override is not None
                    else (mass_kg / IRIS_MASS_KG) * IRIS_AREA)
        # The Phase-8 total already includes delivery payload. When the payload
        # is represented as a detachable Gazebo model, remove it from the body
        # link so it is not counted twice.
        body_mass_kg = mass_kg
        if payload_aboard:
            body_mass_kg = max(0.1, mass_kg - payload_mass_kg)
        # Parasitic drag of the airframe that is actually drawn. Without it the
        # body has no drag at all: a forward dash never reaches terminal
        # velocity, so its "cruise speed" measures the dash duration rather
        # than the vehicle. Enable wind on the link unconditionally — with no
        # WindEffects plugin in the world it costs nothing, and it lets the
        # drag plate see airspeed rather than ground speed when there is wind.
        from gazebo_poc.airframe_drag import drag_breakdown
        drag = drag_breakdown(
            rotor_count, rotor_radius,
            payload_attached=payload_aboard,
        )
        LAST_RESULT["body_drag"] = drag.as_dict()
        _, _, frame_class = generate_multirotor_sdf(
            body_mass_kg, rotor_count, rotor_radius, inertia, area, tdir, out,
            max_rotor_rad_s=mult, fail_rotor=fail_rotor,
            enable_wind=True, payload_release=payload_aboard,
            parachute_deploy=parachute_deploy,
            forward_lidar=obstacle_avoidance,
            forward_lidar_range_m=obstacle_detection_range_m,
            body_drag_area_m2=drag.flat_plate_area_m2)
        print(f"[gen] {rotor_count}-rotor mass={mass_kg}kg inertia={tuple(round(x,4) for x in inertia)} "
              f"area={area:.6f} max_rotor={mult:.0f}rad/s FRAME_CLASS={frame_class} "
              f"f_drag={drag.flat_plate_area_m2:.6f}m2"
              f"{' [calibrated]' if calibrate else ''}"
              f"{f' [MOTOR {fail_rotor} FAILED]' if fail_rotor is not None else ''}", flush=True)
    thrust_diag = _thrust_diagnostics(mass_kg, rotor_count, area, mult, max_thrust_g)
    LAST_RESULT["thrust_diagnostics"] = thrust_diag
    print("[diag] thrust sweep: "
          f"TWR(model)={thrust_diag['model_twr']:.2f}, "
          f"hover_required={thrust_diag['hover_rpm_required']:.0f}RPM "
          f"({thrust_diag['hover_rad_s_fraction_of_max']:.2f} of max), "
          f"sdf_can_hover={thrust_diag['sdf_can_hover']}", flush=True)

    _sh("docker", "rm", "-f", _CONTAINER)
    # Mount the individual model.sdf FILES (not the dirs) so the original meshes/config in the
    # image are preserved — mounting the whole dir hides iris_collision.stl → gz fails to load.
    so = str((out / "iris_with_standoffs" / "model.sdf").resolve())
    gm = str((out / "iris_with_gimbal" / "model.sdf").resolve())
    docker_args = [
        "docker", "run", "-d", "--name", _CONTAINER, "-p", "9002:9002/udp",
        "-v", f"{so}:{_MODEL_BASE}/iris_with_standoffs/model.sdf",
        "-v", f"{gm}:{_MODEL_BASE}/iris_with_gimbal/model.sdf",
    ]
    wind_scale = None
    if wind_mps > 0:
        # The airframe now carries genuine quadratic drag (the BodyDrag plate),
        # which gz-sim's LiftDrag evaluates against airspeed. WindEffects' own
        # m*k*(wind-v) term is a LINEAR approximation of that same drag, so
        # leaving it on would double-count — and it under-predicts badly once
        # groundspeed exceeds the wind it was linearized at. It is kept only as
        # the channel that carries the wind field; the force it adds is zero
        # unless a caller explicitly asks for the legacy behaviour.
        wind_scale = (
            0.0 if wind_force_scale_override is None
            else float(wind_force_scale_override)
        )
        try:
            wind_world = _prepare_wind_world(out, wind_scale)
        except Exception as e:
            print(f"[wind] world preparation failed: {e}", flush=True)
            return 2
        docker_args += ["-v", f"{wind_world.resolve()}:{_WORLD_PATH}"]
    elif obstacle_avoidance:
        try:
            obstacle_world = _prepare_obstacle_world(out)
        except Exception as e:
            print(f"[obstacle] world preparation failed: {e}", flush=True)
            return 2
        docker_args += ["-v", f"{obstacle_world.resolve()}:{_WORLD_PATH}"]
    if payload_aboard:
        payload_model = _prepare_payload_model(out, payload_mass_kg)
        docker_args += ["-v", f"{payload_model.resolve()}:{_MODEL_BASE}/payload_box/model.sdf"]
    docker_args.append(_IMG)
    run = _sh(*docker_args)
    if run.returncode != 0:
        print("[docker] failed:", run.stderr, flush=True)
        return 2
    print("[docker] container up, gz sim loading our airframe ...", flush=True)
    time.sleep(8)

    parm = out / "poc.parm"
    gripper_channel = max(6, rotor_count) if payload_aboard else None
    gripper_servo = gripper_channel + 1 if gripper_channel is not None else None
    parachute_channel = max(6, rotor_count) + 1 if parachute_deploy else None
    parachute_servo = parachute_channel + 1 if parachute_channel is not None else None
    parm.write_text(_parm_text(
        frame_class,
        hover_throttle,
        gripper_servo=gripper_servo,
        parachute_servo=parachute_servo,
        extra_parms=extra_parms or "",
    ))
    if obstacle_avoidance:
        avoidance_margin = (
            float(obstacle_min_separation_m) + 1.0
            if obstacle_min_separation_m is not None
            else float(obstacle_response_threshold_m or 2.0)
        )
        with parm.open("a") as f:
            f.write(
                "PRX1_TYPE 2\nAVOID_ENABLE 2\n"
                f"AVOID_DIST_MAX {float(obstacle_detection_range_m):.2f}\n"
                f"AVOID_MARGIN {avoidance_margin:.2f}\n"
                "AVOID_BEHAVE 1\nAVOID_ALT_MIN 0\n"
            )
    if wind_scale is not None:
        LAST_RESULT.update({
            "wind_requested_mps": float(wind_mps),
            "wind_min_groundspeed_mps": (
                None if wind_min_groundspeed_mps is None else float(wind_min_groundspeed_mps)
            ),
            "wind_force_scale": wind_scale,
            "wind_drag_area_m2": (LAST_RESULT.get("body_drag") or {}).get(
                "flat_plate_area_m2"),
            "wind_fidelity": (
                "Gazebo closed-loop dynamics; the headwind acts through the "
                "airframe's geometry-derived quadratic drag plate, which "
                "gz-sim's LiftDrag evaluates against airspeed. WindEffects' "
                "linear m*k*(wind-v) term is disabled — it approximated the "
                "same drag and under-predicted it as groundspeed grew."
            ),
        })
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

        def set_parameter(name: str, value: float) -> None:
            m.mav.param_set_send(
                m.target_system, m.target_component,
                name.encode(), float(value),
                mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
            )
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                response = m.recv_match(
                    type="PARAM_VALUE", blocking=True, timeout=0.5,
                )
                if response is None:
                    continue
                response_name = response.param_id
                if isinstance(response_name, bytes):
                    response_name = response_name.decode()
                if str(response_name).rstrip("\x00") == name:
                    return
            raise RuntimeError(f"parameter update was not acknowledged: {name}")

        def wait(cond, secs, label):
            end = time.time() + secs
            while time.time() < end:
                h = hb()
                if h and cond(h):
                    return True
            print(f"[sitl] timeout waiting for {label}", flush=True)
            return False

        def mode_holds(mode_id, secs=6.0, consecutive=4):
            """True only when the mode is reported on N heartbeats in a row.

            A single matching heartbeat is not evidence the vehicle is IN a
            mode. ArduCopter accepts GUIDED while disarmed, reports it once,
            and reverts to STABILIZE — which read as success here for months
            and left every flight in a stick-flown fallback, because
            ModeStabilize has no user takeoff.
            """
            end = time.time() + secs
            run = 0
            while time.time() < end:
                h = hb()
                if h is None:
                    continue
                run = run + 1 if h.custom_mode == mode_id else 0
                if run >= consecutive:
                    return True
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
        GUIDED = 4
        rc(1000)                                # throttle low before arming
        m.set_mode("GUIDED")
        if not wait(lambda h: h.custom_mode == GUIDED, 8, "GUIDED mode"):
            _cleanup(proc); return 5
        # let EKF finish tilt alignment before arming (else arm is rejected)
        print("[sitl] in GUIDED; settling EKF before arm ...", flush=True)
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
        # GUIDED must be (re-)entered AFTER arming and must HOLD. Set while
        # disarmed it does not stick, and the vehicle arms in STABILIZE, whose
        # has_user_takeoff() is false — so MAV_CMD_NAV_TAKEOFF is refused with
        # a bare MAV_RESULT_FAILED and every flight falls back to stick control.
        m.set_mode("GUIDED")
        guided_held = mode_holds(GUIDED)
        LAST_RESULT["guided_mode_held_after_arming"] = guided_held
        print(f"[sitl] ARMED. GUIDED held after arming: {guided_held}; takeoff ...",
              flush=True)

        def baro_alt():
            v = m.recv_match(type="VFR_HUD", blocking=True, timeout=2)
            return v.alt if v else None

        # ArduCopter refuses a GUIDED takeoff unless position_ok() holds, which
        # when armed needs EKF_POS_HORIZ_ABS set and EKF_CONST_POS_MODE clear.
        # A bare MAV_RESULT_FAILED does not say which; these flags do.
        ekf = m.recv_match(type="EKF_STATUS_REPORT", blocking=True, timeout=5)
        if ekf is not None:
            bits = {
                "ATTITUDE": 1, "VELOCITY_HORIZ": 2, "VELOCITY_VERT": 4,
                "POS_HORIZ_REL": 8, "POS_HORIZ_ABS": 16, "POS_VERT_ABS": 32,
                "POS_VERT_AGL": 64, "CONST_POS_MODE": 128,
                "PRED_POS_HORIZ_REL": 256, "PRED_POS_HORIZ_ABS": 512,
            }
            present = sorted(name for name, bit in bits.items() if int(ekf.flags) & bit)
            position_ok = ("POS_HORIZ_ABS" in present
                           and "CONST_POS_MODE" not in present)
            LAST_RESULT["ekf_flags_at_takeoff"] = present
            LAST_RESULT["ekf_position_ok_at_takeoff"] = position_ok
            print(f"[sitl] EKF at takeoff: flags={present} → position_ok={position_ok}",
                  flush=True)
        else:
            LAST_RESULT["ekf_flags_at_takeoff"] = None
            print("[sitl] no EKF_STATUS_REPORT before takeoff", flush=True)

        a = baro_alt()
        alt0 = a if a is not None else 0.0

        TGT = 10.0
        m.mav.command_long_send(
            m.target_system,
            m.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0, 0, 0, 0, 0, 0,
            TGT,
        )
        LAST_RESULT["takeoff_method"] = "guided_nav_takeoff"
        # Whether the autopilot ACCEPTED the takeoff decides how to read a
        # failure to climb: a rejected command is a configuration problem, a
        # denied climb after acceptance is a control/thrust one. Without this
        # the fallback hides which.
        takeoff_ack = m.recv_match(type="COMMAND_ACK", blocking=True, timeout=3)
        if takeoff_ack is not None and takeoff_ack.command == mavutil.mavlink.MAV_CMD_NAV_TAKEOFF:
            LAST_RESULT["takeoff_command_result"] = int(takeoff_ack.result)
            LAST_RESULT["takeoff_command_accepted"] = (
                takeoff_ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED
            )
            print(f"[sitl] NAV_TAKEOFF ack result={takeoff_ack.result} "
                  f"(0=accepted)", flush=True)
        else:
            LAST_RESULT["takeoff_command_accepted"] = None
            print("[sitl] NAV_TAKEOFF produced no COMMAND_ACK", flush=True)

        def alt_hold_stick(rel):
            hover_rc = 1500 if hover_throttle is None else int(1100 + max(0.2, min(0.8, float(hover_throttle))) * 800)
            # proportional climb-rate command around the datasheet hover-throttle seed. The
            # old 1500±90 cap was too weak for borderline real-motor T/W cases.
            err = TGT - rel
            return hover_rc + int(max(-180, min(160, err * 35)))

        peak = 0.0
        thr, rels = [], []
        if measure_attitude:
            # Requested BEFORE the first measurement window, not before the
            # cruise survey: hover is a measurement window too, and a stream
            # asked for too late leaves those samples with no reference at all —
            # which correctly yields None, but under-measures the envelope.
            m.mav.command_long_send(
                m.target_system, m.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE_TARGET,
                100000, 0, 0, 0, 0, 0)
            time.sleep(0.5)
        hover_att = _AttitudeSampler() if measure_attitude else None
        cap_proc, cap_path = None, Path("gazebo_poc/generated/jointstate.txt")
        topic = ("/world/iris_runway/model/iris_with_gimbal/model/"
                 "iris_with_standoffs/joint_state")    # nested model — has rotor_*_joint
        t_end = time.time() + 45
        while time.time() < t_end:
            v = m.recv_match(type="VFR_HUD", blocking=True, timeout=2)
            if v is None:
                continue
            rel = v.alt - alt0
            peak = max(peak, rel)
            if time.time() > t_end - 18:          # sample steady-state in the final 18 s
                thr.append(v.throttle); rels.append(rel)
                if hover_att is not None:
                    hover_att.sample(m)
                if cap_proc is None:              # capture rotor RPM while hovering
                    cap_proc = subprocess.Popen(
                        ["docker", "exec", _CONTAINER, "gz", "topic", "-e", "-t", topic],
                        stdout=open(cap_path, "w"), stderr=subprocess.DEVNULL)
        if peak < TGT * 0.25:
            if cap_proc:
                cap_proc.terminate()
                cap_proc = None
            print("[sitl] GUIDED takeoff produced insufficient climb; falling back to "
                  "ALT_HOLD with datasheet-hover throttle seed ...", flush=True)
            LAST_RESULT["takeoff_method"] = "guided_nav_takeoff_then_alt_hold_fallback"
            thr, rels = [], []
            cap_path = Path("gazebo_poc/generated/jointstate_alt_hold.txt")
            m.set_mode("ALT_HOLD")
            wait(lambda h: h.custom_mode == ALT_HOLD, 6, "ALT_HOLD mode fallback")
            t_end = time.time() + 42
            while time.time() < t_end:
                v = m.recv_match(type="VFR_HUD", blocking=True, timeout=2)
                if v is None:
                    continue
                rel = v.alt - alt0
                peak = max(peak, rel)
                rc(alt_hold_stick(rel))
                if time.time() > t_end - 18:
                    thr.append(v.throttle); rels.append(rel)
                    if hover_att is not None:
                        hover_att.sample(m)
                    if cap_proc is None:
                        cap_proc = subprocess.Popen(
                            ["docker", "exec", _CONTAINER, "gz", "topic", "-e", "-t", topic],
                            stdout=open(cap_path, "w"), stderr=subprocess.DEVNULL)
        if cap_proc:
            cap_proc.terminate()
        if hover_att is not None:
            hover_rms = _attitude_rms_deg(hover_att.samples)
            hover_attachment = (
                observe_attachment(_model_xyz("iris_with_gimbal"),
                                   _model_xyz("payload_box"))
                if payload_aboard else PayloadAttachment(observed=False)
            )
            hover_payload_distance = hover_attachment.distance_m
            LAST_RESULT.update({
                "hover_attitude_samples": hover_rms["n"],
                "hover_attitude_roll_rms_deg": hover_rms["roll_rms_deg"],
                "hover_attitude_pitch_rms_deg": hover_rms["pitch_rms_deg"],
                "hover_attitude_rms_deg": hover_rms["rms_deg"],
                # Observed, not assumed: the release happens later in the same
                # flight, so a configuration flag cannot say what was aboard
                # when this window was sampled.
                "hover_attitude_with_payload": (
                    hover_payload_distance is not None
                    and hover_payload_distance <= _PAYLOAD_ATTACHED_MAX_DISTANCE_M
                ),
                "hover_payload_attachment_distance_m": hover_payload_distance,
                "hover_payload_vertical_separation_m": (
                    hover_attachment.vertical_separation_m),
                # Named for what it means: the payload was ATTACHED, not merely
                # that a pose was readable. The two were conflated, and a field
                # that says "observed" while meaning "attached" is how an
                # unloaded window gets counted as transport evidence.
                "hover_payload_attached": (
                    hover_attachment.attached
                    or False
                ),
            })
            print(f"[att] hover RMS: {hover_rms}", flush=True)
        rotor_rad_s = _parse_rotor_velocity(cap_path)
        hover_rpm = rotor_rad_s * 9.5493
        print(f"[sitl] climb peak={peak:.2f} m; steady hover sampled. "
              f"rotor |omega|~{rotor_rad_s:.1f} rad/s ({hover_rpm:.0f} RPM)", flush=True)

        # --- physical payload release timing ---
        # This checks command -> Gazebo detachable joint -> falling payload. It
        # does not claim that the aircraft's mission logic generated the command.
        if payload_release and payload_mass_kg > 0:
            # NOT payload_aboard: a transport scenario carries the payload
            # through the whole flight and never releases it, which is what
            # makes its cruise windows transport evidence.
            release_target_ned = None
            release_target_world = None
            condition_position_ned = None
            condition_position_world = None
            condition_position_error = None
            separation_position_world = None
            separation_position_error = None
            condition_met_mono = None
            if positional_release:
                m.set_mode("ALT_HOLD")
                wait(lambda h: h.custom_mode == ALT_HOLD, 6, "ALT_HOLD for release approach")
                velocity_samples = []
                last_pos = None
                probe_end = time.monotonic() + 3.0
                while time.monotonic() < probe_end:
                    hud = m.recv_match(type="VFR_HUD", blocking=True, timeout=1)
                    if hud is not None:
                        rc(alt_hold_stick(hud.alt - alt0), pitch=1420)
                    pos = m.recv_match(type="LOCAL_POSITION_NED", blocking=True, timeout=1)
                    if pos is not None:
                        last_pos = (float(pos.x), float(pos.y))
                        if pos.vx * pos.vx + pos.vy * pos.vy > 0.25:
                            velocity_samples.append((float(pos.vx), float(pos.vy)))
                if last_pos and velocity_samples:
                    vx = sum(v[0] for v in velocity_samples) / len(velocity_samples)
                    vy = sum(v[1] for v in velocity_samples) / len(velocity_samples)
                    norm = (vx * vx + vy * vy) ** 0.5
                    ux, uy = vx / norm, vy / norm
                    release_target_ned = (
                        last_pos[0] + ux * 5.0,
                        last_pos[1] + uy * 5.0,
                    )
                    world_position = _vehicle_xy()
                    if world_position is not None:
                        release_target_world = (
                            world_position[0] + ux * 5.0,
                            world_position[1] - uy * 5.0,
                        )
                    approach_end = time.monotonic() + 12.0
                    while time.monotonic() < approach_end:
                        hud = m.recv_match(type="VFR_HUD", blocking=True, timeout=1)
                        pos = m.recv_match(type="LOCAL_POSITION_NED", blocking=True, timeout=1)
                        if pos is None:
                            continue
                        condition_position_ned = (float(pos.x), float(pos.y))
                        condition_position_error = (
                            (condition_position_ned[0] - release_target_ned[0]) ** 2
                            + (condition_position_ned[1] - release_target_ned[1]) ** 2
                        ) ** 0.5
                        if hud is not None:
                            pitch = 1470 if condition_position_error < 2.0 else 1420
                            rc(alt_hold_stick(hud.alt - alt0), pitch=pitch)
                        if condition_position_error <= positional_tolerance_m:
                            # the delivery coordinate condition is first satisfied
                            # HERE — the PERF-005 clock starts at this moment.
                            condition_met_mono = time.monotonic()
                            condition_position_world = _vehicle_xy()
                            break
                rc(alt_hold_stick(rels[-1] if rels else TGT))

            # An abort condition, when the scenario asks for one, is offered to
            # the model BEFORE the coordinate event. REQ-SAFE-006 says the
            # payload stays locked whenever an abort is active regardless of
            # proximity, so a release that still happens here is a physical
            # observation of the requirement being violated.
            abort_active = False
            if mission is not None and delivery_abort_before_release:
                abort_fired = mission.offer(
                    "AbortConditionActive", time=time.monotonic())
                abort_active = True
                print(f"[model] abort offered; model fired "
                      f"{[d.action for d in abort_fired]}", flush=True)

            # Ask the model whether to release. The harness does NOT decide.
            release_decisions = ()
            if mission is not None:
                release_decisions = mission.offer(
                    "DeliveryCoordinateSatisfied", time=time.monotonic())
                print(f"[model] delivery coordinate offered; model fired "
                      f"{[d.action for d in release_decisions]}", flush=True)
            model_released = (
                mission is None
                or mission.performed(release_decisions, ModelAction.RELEASE_PAYLOAD)
            )

            payload_z0 = _payload_z()
            release_started = time.monotonic()
            if model_released:
                m.mav.command_long_send(
                    m.target_system,
                    m.target_component,
                    211,  # MAV_CMD_DO_GRIPPER
                    0,
                    0, 0, 0, 0, 0, 0, 0,  # gripper 0, RELEASE
                )
            else:
                print("[model] the generated logic declined to release; "
                      "the harness issues no command", flush=True)
            release_delay = None
            payload_z = payload_z0
            deadline = time.monotonic() + 4.0
            while time.monotonic() < deadline:
                time.sleep(0.1)
                rc(alt_hold_stick(rels[-1] if rels else TGT))
                payload_z = _payload_z()
                if (payload_z0 is not None and payload_z is not None
                        and payload_z < payload_z0 - 0.15):
                    release_delay = time.monotonic() - release_started
                    separation_position_world = _payload_xy()
                    if (release_target_world is not None
                            and separation_position_world is not None):
                        separation_position_error = math.hypot(
                            separation_position_world[0] - release_target_world[0],
                            separation_position_world[1] - release_target_world[1],
                        )
                    break
            coordinate_chain_delay = None
            if condition_met_mono is not None and release_delay is not None:
                # coordinate-condition satisfied → physical separation, the
                # full PERF-005 interval. With a mission model the condition is
                # consumed by the generated logic, which owns the decision to
                # actuate; the harness only observes the separation.
                coordinate_chain_delay = (
                    release_started + release_delay - condition_met_mono
                )
            LAST_RESULT.update({
                "payload_release_commanded": model_released,
                "payload_release_decided_by": (
                    "generated model" if mission is not None else "harness"),
                "payload_release_decisions": (
                    [d.as_dict() for d in release_decisions]
                    if mission is not None else None),
                # (adapter_constant, model_action) pairs performed() accepted
                # by causal role — the report must be able to recognise the
                # model's own spelling of the actuation, not just the
                # adapter's.
                "action_resolutions": (
                    [list(pair) for pair in mission.action_resolutions]
                    if mission is not None else None),
                "payload_abort_active": abort_active,
                # "no guard in the model" and "a guard the harness never fed"
                # look identical in the flight: both release. Recording which
                # one happened is what keeps the verdict from blaming the model
                # for a condition it was never told about.
                # Identity by causal role: the guards on whatever transitions
                # answer the delivery event THIS model declares. None means
                # the event did not resolve — the question was never put, and
                # no verdict may read that as "unguarded". (The old lookup
                # asked for the harness spelling 'actuateRelease' and reported
                # a guarded run3 as carrying no inhibition logic.)
                "payload_release_guards": (
                    (lambda g: None if g is None else list(g))(
                        mission.guards_for_event("DeliveryCoordinateSatisfied")
                    )
                    if mission is not None else []
                ),
                "payload_abort_flags_unbound": (
                    list(mission.unlatched_boolean_attributes())
                    if mission is not None else []
                ),
                "payload_abort_flags_raised": (
                    dict(mission.conditions) if mission is not None else {}
                ),
                "payload_release_inhibited": bool(
                    mission is not None and not release_decisions),
                "payload_observer_available": payload_z0 is not None,
                "payload_release_detected": release_delay is not None,
                "payload_release_delay_s": release_delay,
                "payload_condition_met": condition_met_mono is not None,
                "payload_coordinate_to_separation_delay_s": coordinate_chain_delay,
                "payload_z_before_m": payload_z0,
                "payload_z_after_m": payload_z,
                "payload_mass_kg": payload_mass_kg,
                "payload_release_target_ned_m": release_target_ned,
                "payload_release_target_world_m": release_target_world,
                "payload_condition_position_ned_m": condition_position_ned,
                "payload_condition_position_world_m": condition_position_world,
                "payload_condition_position_error_m": condition_position_error,
                "payload_release_position_world_m": separation_position_world,
                "payload_release_position_error_m": separation_position_error,
                "payload_release_position_met": (
                    separation_position_error is not None
                    and separation_position_error <= positional_tolerance_m
                ),
                # The three quantities the requirement's two conjuncts turn on,
                # kept apart on purpose: what the ESTIMATOR thought at the
                # trigger, where the payload TRULY was when it separated, and
                # how long lay between. Only the truth error may close the
                # position clause; the other two are what make it readable.
                "trigger_estimated_error_m": condition_position_error,
                # The missing cell: where the vehicle TRULY was when the model
                # decided. Without it the estimator's error and the actuation
                # lag cannot be told apart, and a 4 m miss looks like one
                # cause when it may be the other.
                "trigger_truth_error_m": (
                    math.hypot(
                        condition_position_world[0] - release_target_world[0],
                        condition_position_world[1] - release_target_world[1],
                    )
                    if condition_position_world is not None
                    and release_target_world is not None else None
                ),
                "separation_truth_error_m": separation_position_error,
                "trigger_to_separation_s": coordinate_chain_delay,
                # "no delivery-abort condition is active" is the requirement's
                # second conjunct, and it was never checked.
                "delivery_abort_inactive": not abort_active,
                "payload_release_position_basis": (
                    "payload_ground_truth_at_separation"
                    if separation_position_error is not None else None
                ),
                "payload_release_position_tolerance_m": positional_tolerance_m,
                "payload_release_fidelity": (
                    "generated PayloadReleaseBehavior consumed the delivery event "
                    "and its entry action drove MAV_CMD_DO_GRIPPER to Gazebo "
                    "detachable-joint physical separation"
                    if mission is not None else
                    "MAV_CMD_DO_GRIPPER to Gazebo detachable-joint physical separation"
                ),
            })
            print(
                f"[payload] release detected={release_delay is not None} "
                f"delay={release_delay} s z={payload_z0}->{payload_z}",
                flush=True,
            )

        if navigation_accuracy:
            # CEP is a NAVIGATION claim: where the vehicle actually ends up
            # versus where it was told to go. Comparing the EKF's own estimate
            # against the commanded target only measures the position
            # controller — the EKF believes it arrived even when it did not.
            # So the error is taken from Gazebo ground truth, and the EKF's
            # view is recorded beside it; the gap between them IS the
            # GPS/estimator contribution the requirement is about.
            if not guided_held:
                LAST_RESULT["cep_unavailable_reason"] = (
                    "GUIDED did not hold, so no commanded waypoint was flown"
                )
                print("[nav] GUIDED not held — navigation accuracy not measurable",
                      flush=True)
            else:
                m.set_mode("GUIDED")
                mode_holds(GUIDED)
                origin = m.recv_match(type="LOCAL_POSITION_NED", blocking=True, timeout=5)
                world0 = _vehicle_xy()
                if origin is None or world0 is None:
                    LAST_RESULT["cep_unavailable_reason"] = (
                        "no LOCAL_POSITION_NED or no Gazebo ground-truth pose"
                    )
                else:
                    n0, e0, d0 = float(origin.x), float(origin.y), float(origin.z)
                    # The requirement says "designated GPS waypoints", which are
                    # ABSOLUTE. Commanding local-NED offsets instead makes a GNSS
                    # bias unobservable: the local origin is derived from the same
                    # biased fix, so the error cancels and the measurement reports
                    # the control loop no matter how wrong the GNSS is. Measured
                    # directly: a 1.5 m injected bias showed up at the sensor
                    # (raw-versus-fused 1.508 m) and moved the local-frame result
                    # by 0.000 m.
                    #
                    # So the waypoints are designated in GLOBAL coordinates, and
                    # derived from Gazebo GROUND TRUTH rather than from the fix —
                    # a point designated on a map does not move because the
                    # receiver is biased.
                    true_n0, true_e0 = world0[0], -world0[1]
                    lat_scale = 1.0 / 111320.0
                    lon_scale = lat_scale / math.cos(math.radians(_HOME_LAT))
                    alt_rel = float(rels[-1]) if rels else TGT
                    legs = []
                    for leg_index, (dn, de) in enumerate(_cep_targets()):
                        injected_offset = None
                        if gps_horizontal_error_m is not None:
                            error_angle = 2.0 * math.pi * leg_index / _CEP_POINTS
                            error_north = float(gps_horizontal_error_m) * math.cos(error_angle)
                            error_east = float(gps_horizontal_error_m) * math.sin(error_angle)
                            set_parameter(
                                "SIM_GPS1_GLTCH_X",
                                error_north * lat_scale,
                            )
                            set_parameter(
                                "SIM_GPS1_GLTCH_Y",
                                error_east * lon_scale,
                            )
                            injected_offset = [
                                round(error_north, 4), round(error_east, 4),
                            ]
                        tn, te = n0 + dn, e0 + de      # local target, for arrival only
                        des_lat = _HOME_LAT + (true_n0 + dn) * lat_scale
                        des_lon = _HOME_LON + (true_e0 + de) * lon_scale
                        arrived = False
                        deadline = time.time() + 40.0
                        while time.time() < deadline:
                            m.mav.set_position_target_global_int_send(
                                0, m.target_system, m.target_component,
                                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                                0b0000111111111000,          # position only
                                int(des_lat * 1e7), int(des_lon * 1e7), alt_rel,
                                0, 0, 0, 0, 0, 0, 0, 0,
                            )
                            pos = m.recv_match(type="LOCAL_POSITION_NED",
                                               blocking=True, timeout=1)
                            if pos is None:
                                continue
                            if math.hypot(float(pos.x) - tn, float(pos.y) - te) < 1.5:
                                arrived = True
                                break
                        if not arrived:
                            print(f"[nav] leg {leg_index}: never reached the commanded "
                                  "point; excluded from CEP", flush=True)
                            continue
                        # settle before sampling: a fly-through is not an arrival
                        settle_end = time.time() + 4.0
                        while time.time() < settle_end:
                            m.mav.set_position_target_global_int_send(
                                0, m.target_system, m.target_component,
                                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                                0b0000111111111000,
                                int(des_lat * 1e7), int(des_lon * 1e7), alt_rel,
                                0, 0, 0, 0, 0, 0, 0, 0)
                            m.recv_match(type="LOCAL_POSITION_NED",
                                         blocking=True, timeout=1)
                        # How much GNSS noise actually reaches the link, and how
                        # much of it survives the estimator. Without this a CEP
                        # that ignores injected noise looks like a robust result
                        # instead of an unverified filtering claim.
                        truth = _vehicle_xy()
                        raw_spread = []
                        raw_truth_errors = []
                        for _ in range(12):
                            raw = m.recv_match(type="GPS_RAW_INT", blocking=True, timeout=1)
                            fused = m.recv_match(type="GLOBAL_POSITION_INT",
                                                 blocking=True, timeout=1)
                            if raw is None or fused is None:
                                continue
                            dlat = (raw.lat - fused.lat) * 1e-7 * 111320.0
                            dlon = ((raw.lon - fused.lon) * 1e-7 * 111320.0
                                    * math.cos(math.radians(fused.lat * 1e-7)))
                            raw_spread.append(math.hypot(dlat, dlon))
                            if truth is not None:
                                true_lat = _HOME_LAT + truth[0] * lat_scale
                                true_lon = _HOME_LON - truth[1] * lon_scale
                                raw_north_error = (
                                    raw.lat * 1e-7 - true_lat
                                ) * 111320.0
                                raw_east_error = (
                                    raw.lon * 1e-7 - true_lon
                                ) * 111320.0 * math.cos(math.radians(true_lat))
                                raw_truth_errors.append(math.hypot(
                                    raw_north_error, raw_east_error,
                                ))
                        pos = m.recv_match(type="LOCAL_POSITION_NED",
                                           blocking=True, timeout=3)
                        if pos is None or truth is None:
                            continue
                        # ArduPilotPlugin maps Gazebo world (x, y) to NED (x, -y)
                        expected = (world0[0] + dn, world0[1] - de)
                        err_truth = math.hypot(truth[0] - expected[0],
                                               truth[1] - expected[1])
                        err_ekf = math.hypot(float(pos.x) - tn, float(pos.y) - te)
                        legs.append({
                            "leg": leg_index,
                            "target_ned_m": [round(dn, 3), round(de, 3)],
                            "truth_error_m": round(err_truth, 4),
                            "ekf_error_m": round(err_ekf, 4),
                            "raw_gnss_minus_fused_m": (
                                round(_median(raw_spread), 4) if raw_spread else None
                            ),
                            "raw_gnss_truth_error_m": (
                                round(_median(raw_truth_errors), 4)
                                if raw_truth_errors else None
                            ),
                            "injected_gnss_offset_ne_m": injected_offset,
                        })
                        print(f"[nav] leg {leg_index}: ground-truth error "
                              f"{err_truth:.3f} m, EKF-reported {err_ekf:.3f} m",
                              flush=True)
                    truth_errors = [leg["truth_error_m"] for leg in legs]
                    ekf_errors = [leg["ekf_error_m"] for leg in legs]
                    LAST_RESULT.update({
                        "cep_legs": legs,
                        "cep_samples": len(legs),
                        "cep_m": _median(truth_errors),
                        "cep_max_error_m": max(truth_errors) if truth_errors else None,
                        "cep_ekf_m": _median(ekf_errors),
                        "cep_raw_gnss_scatter_m": _median(
                            [leg["raw_gnss_minus_fused_m"] for leg in legs
                             if leg.get("raw_gnss_minus_fused_m") is not None]
                        ),
                        "cep_raw_gnss_truth_error_m": _median(
                            [leg["raw_gnss_truth_error_m"] for leg in legs
                             if leg.get("raw_gnss_truth_error_m") is not None]
                        ),
                        "cep_horizontal_error_source": (
                            "sim_gps_glitch_xy_campaign"
                            if gps_horizontal_error_m is not None else None
                        ),
                        "cep_horizontal_error_injected_m": (
                            None if gps_horizontal_error_m is None
                            else float(gps_horizontal_error_m)
                        ),
                        "cep_basis": (
                            "median horizontal error between the commanded waypoint "
                            "and Gazebo ground truth; the EKF-reported error is "
                            "recorded alongside and their difference is the "
                            "estimator/GPS contribution. "
                            + ("no GNSS error was injected, and SITL has no random "
                               "horizontal GPS error to begin with, so this "
                               "EXCLUDES the term that dominates a real CEP — it "
                               "is a floor, not a navigation CEP"
                               if gps_horizontal_error_m is None else
                               f"a {float(gps_horizontal_error_m):.2f} m horizontal "
                               "GNSS error was swept through eight directions via "
                               "SIM_GPS1_GLTCH_X/Y; raw fixes are compared directly "
                               "with Gazebo truth before the navigation CEP is judged")
                        ),
                    })
                    print(f"[nav] CEP={LAST_RESULT['cep_m']} m over "
                          f"{len(legs)} commanded waypoints "
                          f"(EKF-reported median {LAST_RESULT['cep_ekf_m']} m)",
                          flush=True)
            m.mav.rc_channels_override_send(m.target_system, m.target_component, *([0] * 8))
            _cleanup(proc)
            n = max(1, len(thr) // 3)
            hov_thr = sum(thr[-n:]) / n
            hov_alt = sum(rels[-n:]) / n
            LAST_RESULT.update(hover_stable=True, hover_throttle_pct=hov_thr,
                               hover_alt_m=hov_alt, ok=True, failure_kind=None)
            return 0

        if obstacle_avoidance:
            sensor_range = float(obstacle_detection_range_m)
            approach_speed = float(obstacle_approach_speed_mps)
            max_distance_cm = max(20, int(sensor_range * 100))
            effective_response_threshold = (
                float(obstacle_response_threshold_m)
                if obstacle_response_threshold_m is not None
                else (
                    float(obstacle_min_separation_m)
                    if obstacle_min_separation_m is not None else None
                )
            )
            lidar_proc, lidar = _start_lidar_observer(sensor_range)
            lidar_deadline = time.monotonic() + 6.0
            while lidar.get("distance_m") is None and time.monotonic() < lidar_deadline:
                time.sleep(0.1)
            lidar_available = lidar.get("distance_m") is not None
            m.set_mode("GUIDED")
            wait(lambda h: h.custom_mode == GUIDED, 6, "GUIDED for obstacle test")
            # Prime AP_Proximity before asking the vehicle to move. Gazebo's
            # no-return scan is a valid max-range clear reading, not "No Data".
            warmup_end = time.monotonic() + 3.0
            while time.monotonic() < warmup_end:
                distance = float(lidar.get("distance_m") or sensor_range)
                m.mav.distance_sensor_send(
                    int(time.monotonic() * 1000) & 0xFFFFFFFF,
                    20,
                    max_distance_cm,
                    max(20, min(max_distance_cm, int(distance * 100))),
                    0,
                    0,
                    0,
                    0,
                )
                m.mav.set_position_target_local_ned_send(
                    int(time.monotonic() * 1000) & 0xFFFFFFFF,
                    m.target_system,
                    m.target_component,
                    1,
                    4039,
                    0, 0, 0,
                    0, 0, 0,
                    0, 0, 0,
                    0, 0,
                )
                time.sleep(0.1)
            min_distance = float("inf")
            first_detection_distance = None
            detection_timely = False
            response_observed = False
            response_onset_distance = None
            final_speed = None
            last_hud = None
            peak_pre_detection_speed = 0.0
            stopped_samples = 0
            obstacle_end = time.monotonic() + 25.0
            while time.monotonic() < obstacle_end:
                hud = m.recv_match(type="VFR_HUD", blocking=False)
                if hud is not None:
                    last_hud = hud
                distance = lidar.get("distance_m")
                age = (
                    time.monotonic() - float(lidar["updated_at"])
                    if lidar.get("updated_at") is not None else float("inf")
                )
                if distance is not None and age < 1.0:
                    distance = float(distance)
                    min_distance = min(min_distance, distance)
                    m.mav.distance_sensor_send(
                        int(time.monotonic() * 1000) & 0xFFFFFFFF,
                        20,
                        max_distance_cm,
                        max(20, min(max_distance_cm, int(distance * 100))),
                        0,
                        0,
                        0,  # MAV_SENSOR_ROTATION_NONE: body-forward
                        0,
                    )
                    if last_hud is not None:
                        final_speed = float(last_hud.groundspeed)
                    has_return = distance < sensor_range - 0.02
                    if not has_return and final_speed is not None:
                        peak_pre_detection_speed = max(
                            peak_pre_detection_speed, final_speed
                        )
                    if has_return and first_detection_distance is None:
                        first_detection_distance = distance
                        tolerance = max(0.25, sensor_range * 0.03)
                        detection_timely = distance >= sensor_range - tolerance
                    if (
                        first_detection_distance is not None
                        and final_speed is not None
                        and peak_pre_detection_speed >= approach_speed * 0.8
                        and final_speed <= peak_pre_detection_speed * 0.8
                    ):
                        if response_onset_distance is None:
                            response_onset_distance = distance
                        response_observed = True
                        stopped_samples = stopped_samples + 1 if final_speed < 0.25 else 0
                    else:
                        stopped_samples = 0
                    clearance_breached = (
                        obstacle_min_separation_m is not None
                        and distance < float(obstacle_min_separation_m)
                    )
                    response_late = (
                        effective_response_threshold is not None
                        and distance < effective_response_threshold
                        and response_onset_distance is None
                    )
                    if clearance_breached or response_late or stopped_samples >= 10:
                        break
                # The lidar boresight, MAVLink sensor orientation, velocity command,
                # and wall normal are all body-forward (+X). BODY_NED prevents a
                # future world-frame edit from silently recreating the old side-flight
                # scenario in which the wall entered the ±30° lidar only at ~3 m.
                m.mav.set_position_target_local_ned_send(
                    int(time.monotonic() * 1000) & 0xFFFFFFFF,
                    m.target_system,
                    m.target_component,
                    8,      # MAV_FRAME_BODY_NED
                    4039,   # velocity only; ignore pos/accel/yaw
                    0, 0, 0,
                    approach_speed, 0, 0,
                    0, 0, 0,
                    0, 0,
                )
                time.sleep(0.1)
            m.mav.set_position_target_local_ned_send(
                int(time.monotonic() * 1000) & 0xFFFFFFFF,
                m.target_system,
                m.target_component,
                1,
                4039,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0,
            )
            if lidar_proc.poll() is None:
                lidar_proc.terminate()
            lidar_available = int(lidar.get("sample_count") or 0) > 0
            obstacle_met = _obstacle_requirement_met(
                lidar_available,
                detection_timely,
                response_observed,
                response_onset_distance,
                effective_response_threshold,
                min_distance,
                obstacle_min_separation_m,
            )
            LAST_RESULT.update({
                # gate read by the feasibility mapper: an obstacle scenario was
                # actually flown, so its verdict may be judged from live evidence
                "obstacle_req": True,
                "obstacle_lidar_available": lidar_available,
                "obstacle_lidar_samples": int(lidar.get("sample_count") or 0),
                "obstacle_detected_within_range": first_detection_distance is not None,
                # Legacy compatibility key; new reports use the range-agnostic name.
                "obstacle_detected_within_15m": first_detection_distance is not None,
                "obstacle_detection_range_requirement_m": sensor_range,
                "obstacle_first_detection_distance_m": first_detection_distance,
                "obstacle_detection_timely": detection_timely,
                "obstacle_min_distance_m": None if min_distance == float("inf") else min_distance,
                "obstacle_final_groundspeed_mps": final_speed,
                "obstacle_response_observed": response_observed,
                "obstacle_response_onset_distance_m": response_onset_distance,
                "obstacle_response_threshold_m": effective_response_threshold,
                "obstacle_min_separation_requirement_m": obstacle_min_separation_m,
                "obstacle_approach_speed_mps": approach_speed,
                "obstacle_scenario_alignment": (
                    "wall normal derived from airframe spawn yaw; "
                    "body-forward lidar and BODY_NED +X velocity"
                ),
                "obstacle_avoidance_met": obstacle_met,
                "obstacle_fidelity": (
                    "Gazebo gpu_lidar -> MAVLink DISTANCE_SENSOR -> ArduPilot proximity avoidance"
                ),
            })
            m.mav.rc_channels_override_send(m.target_system, m.target_component, *([0] * 8))
            _cleanup(proc)
            n = max(1, len(thr) // 3)
            hov_thr = sum(thr[-n:]) / n if thr else 0.0
            hov_alt = sum(rels[-n:]) / n if rels else 0.0
            band = max(rels[-n:]) - min(rels[-n:]) if rels else float("inf")
            stable = hov_alt > TGT * 0.5 and band < 1.5
            LAST_RESULT.update(
                hover_stable=stable,
                hover_throttle_pct=hov_thr,
                hover_rpm=hover_rpm,
                hover_alt_m=hov_alt,
                mass_kg=mass_kg,
                rotor_radius_m=rotor_radius,
                rotor_count=rotor_count,
                capacity_mah=capacity_mah,
                ok=stable,
                failure_kind=None if stable else "GAZEBO_MODEL_UNCALIBRATED",
            )
            print(
                f"[obstacle] lidar={lidar_available} detection={first_detection_distance}m "
                f"timely={detection_timely} response_at={response_onset_distance}m "
                f"min_distance={LAST_RESULT['obstacle_min_distance_m']}m "
                f"final_speed={final_speed}m/s response={response_observed} met={obstacle_met}",
                flush=True,
            )
            return 0 if stable else 8

        # --- forward-flight survey: sweep pitch, hold each point to steady state ---
        # One stick position only answers "how fast is this stick position". The
        # requirements ask what the vehicle holds "at all authorised speeds", and
        # whether it can make headway against a headwind using the authority it
        # has. Both are sweeps, and every point must reach a certified plateau
        # before its speed may be reported.
        if payload_aboard:
            # SERVO_OUTPUT_RAW is not streamed by default, and a non-blocking
            # read of a message nobody sends returns None — which reads as "no
            # signal" rather than "not asked for".
            m.mav.command_long_send(
                m.target_system, m.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                mavutil.mavlink.MAVLINK_MSG_ID_SERVO_OUTPUT_RAW,
                200000, 0, 0, 0, 0, 0)
            time.sleep(0.5)
        print("[sitl] forward-flight survey (pitch sweep, hold alt) ...", flush=True)
        m.set_mode("ALT_HOLD")
        wait(lambda h: h.custom_mode == ALT_HOLD, 6, "ALT_HOLD mode for forward dash")
        fwd_cap = Path("gazebo_poc/generated/jointstate_fwd.txt")
        fcap, wind_vectors = None, []
        cruise_att = _AttitudeSampler() if measure_attitude else None

        sweep = []
        last_window = []
        for index, pitch in enumerate(_DASH_SWEEP_PITCH):
            final_point = index == len(_DASH_SWEEP_PITCH) - 1
            if final_point and wind_mps <= 0 and fcap is None:
                # capture rotor speed over the segment fwd_speed will describe
                fcap = subprocess.Popen(
                    ["docker", "exec", _CONTAINER, "gz", "topic", "-e", "-t", topic],
                    stdout=open(fwd_cap, "w"), stderr=subprocess.DEVNULL)
            # What the vehicle is CARRYING is observed per window, not inferred
            # once from configuration: the same flight releases the payload, so
            # a window flown after separation is not transport evidence however
            # the run was configured.
            attachment = observe_attachment(
                _model_xyz("iris_with_gimbal"), _model_xyz("payload_box"))
            # The gripper servo drives the detach topic through a
            # TriggeredPublisher, so its PWM separates "the joint let go" from
            # "something commanded the release" when a payload goes missing.
            _srv = m.recv_match(type="SERVO_OUTPUT_RAW", blocking=True, timeout=2)
            gripper_pwm = (
                getattr(_srv, f"servo{(gripper_channel or 6) + 1}_raw", None)
                if _srv is not None else None
            )
            if payload_aboard:
                # Absolute poses, not just their separation: a RELEASED payload
                # is left hundreds of metres behind, while one that merely lags
                # under acceleration stays within a few. Only the vehicle's own
                # displacement tells those apart, and the separation alone was
                # about to be read as a release.
                _veh = _model_xyz("iris_with_gimbal")
                _pay = _model_xyz("payload_box")
                print(f"[payload] {pitch}: vehicle={_veh} payload={_pay} "
                      f"dist={attachment.distance_m} "
                      f"attached={attachment.attached} "
                      f"gripper_pwm={gripper_pwm}", flush=True)
            trace = []

            def _watch(conn, _sink=trace):
                if not payload_aboard or len(_sink) >= 6:
                    return
                srv = conn.messages.get("SERVO_OUTPUT_RAW")
                _sink.append({
                    "distance_m": (observe_attachment(
                        _model_xyz("iris_with_gimbal"),
                        _model_xyz("payload_box")).distance_m),
                    "gripper_pwm": (
                        getattr(srv, f"servo{(gripper_channel or 6) + 1}_raw", None)
                        if srv is not None else None
                    ),
                })

            window, verdict, span = _hold_until_steady(
                m, rc, alt_hold_stick, alt0, pitch,
                label="cruise survey", attitude=cruise_att,
                on_sample=_watch if payload_aboard else None)
            point = {
                "pitch_rc": pitch,
                "steady": verdict.steady,
                "speed_mps": verdict.mean,
                "steady_state": verdict.as_dict(),
                "payload_attachment": attachment.as_dict(),
                "gripper_pwm": gripper_pwm,
                "payload_trace": trace,
            }
            if cruise_att is not None and verdict.steady:
                point_rms = _attitude_rms_deg(cruise_att.between(*span))
                point.update({
                    "attitude_rms_deg": point_rms["rms_deg"],
                    "attitude_roll_rms_deg": point_rms["roll_rms_deg"],
                    "attitude_pitch_rms_deg": point_rms["pitch_rms_deg"],
                    "attitude_samples": point_rms["n"],
                })
            sweep.append(point)
            if window:
                last_window = window
        LAST_RESULT["cruise_sweep"] = sweep
        # Transport windows: the hover sample plus every cruise point, each
        # carrying the attachment that was OBSERVED while it was measured.
        transport_windows = []
        if LAST_RESULT.get("hover_attitude_rms_deg") is not None:
            transport_windows.append(TransportWindow(
                label="hover",
                attachment=PayloadAttachment(
                    # "observed" is whether the poses could be read; whether
                    # that means attached is the module's call, on vertical
                    # separation — which must be carried through EVERY
                    # reconstruction or it silently decides "not attached".
                    observed=LAST_RESULT.get(
                        "hover_payload_attachment_distance_m") is not None,
                    distance_m=LAST_RESULT.get("hover_payload_attachment_distance_m"),
                    vertical_separation_m=LAST_RESULT.get(
                        "hover_payload_vertical_separation_m"),
                ),
                attitude_rms_deg=LAST_RESULT.get("hover_attitude_rms_deg"),
            ))
        for point in sweep:
            state = point.get("payload_attachment") or {}
            transport_windows.append(TransportWindow(
                label=f"cruise@rc{point['pitch_rc']}",
                attachment=PayloadAttachment(
                    observed=bool(state.get("observed")),
                    distance_m=state.get("distance_m"),
                    vertical_separation_m=state.get("vertical_separation_m"),
                ),
                attitude_rms_deg=point.get("attitude_rms_deg"),
                speed_mps=point.get("speed_mps"),
            ))
        LAST_RESULT["transport_windows"] = [w.as_dict() for w in transport_windows]
        LAST_RESULT["cruise_sweep_payload_attached"] = any(
            w.attachment.attached for w in transport_windows
        )
        if fcap:
            fcap.terminate()
            fcap = None

        steady_points = [pt for pt in sweep if pt["steady"] and pt["speed_mps"] is not None]
        LAST_RESULT["cruise_sweep_steady_points"] = len(steady_points)
        LAST_RESULT["cruise_sweep_points"] = len(sweep)
        if steady_points:
            best = max(steady_points, key=lambda pt: pt["speed_mps"])
            LAST_RESULT.update({
                # The fastest speed the vehicle *held*, across the swept
                # authority — not the speed it happened to have reached.
                "nilwind_dash_speed_mps": best["speed_mps"],
                "nilwind_dash_peak_mps": max(pt["speed_mps"] for pt in steady_points),
                "nilwind_dash_samples": best["steady_state"]["samples"],
                "nilwind_dash_steady_state": best["steady_state"],
                "nilwind_dash_pitch_rc": best["pitch_rc"],
                "cruise_sweep_speeds_mps": [pt["speed_mps"] for pt in steady_points],
            })
            print("[nilwind] certified cruise points: "
                  + ", ".join(f"{pt['speed_mps']:.1f} m/s @rc{pt['pitch_rc']}"
                              for pt in steady_points)
                  + f"; best={best['speed_mps']:.1f} m/s", flush=True)
        else:
            print("[nilwind] no cruise point reached steady state — no speed reported",
                  flush=True)

        rms_points = [pt for pt in steady_points if pt.get("attitude_rms_deg") is not None]
        if rms_points:
            # "at all authorised speeds" is a worst case over the sweep, not one
            # point: report the largest RMS and the span it was swept across.
            worst = max(rms_points, key=lambda pt: pt["attitude_rms_deg"])
            LAST_RESULT.update({
                "cruise_attitude_rms_deg": worst["attitude_rms_deg"],
                "cruise_attitude_roll_rms_deg": worst["attitude_roll_rms_deg"],
                "cruise_attitude_pitch_rms_deg": worst["attitude_pitch_rms_deg"],
                "cruise_attitude_samples": worst["attitude_samples"],
                "cruise_attitude_mean_speed_mps": worst["speed_mps"],
                "cruise_attitude_swept_speeds_mps": [pt["speed_mps"] for pt in rms_points],
                # Paired with the speeds above, ascending. "All authorised
                # speeds" is a claim about points nobody flew; the per-point
                # RMS is what says whether the worst case sits at the envelope
                # boundary or somewhere in the interior, and only the first of
                # those lets a bounded sweep speak for the unbounded set.
                "cruise_attitude_swept_rms_deg": [
                    pt["attitude_rms_deg"] for pt in rms_points
                ],
                "cruise_attitude_speed_span_mps": [
                    min(pt["speed_mps"] for pt in rms_points),
                    max(pt["speed_mps"] for pt in rms_points),
                ],
                "cruise_attitude_points": len(rms_points),
            })
            print(f"[att] cruise RMS worst-of-sweep {worst['attitude_rms_deg']:.4f} deg "
                  f"over {len(rms_points)} speed points "
                  f"({min(pt['speed_mps'] for pt in rms_points):.1f}-"
                  f"{max(pt['speed_mps'] for pt in rms_points):.1f} m/s)", flush=True)

        fwd_speed = (
            sum(v for _t, v in last_window) / len(last_window) if last_window else 0.0
        )
        LAST_RESULT["fwd_speed_steady_state"] = steady_state(last_window).as_dict()

        if wind_mps > 0:
            # Aim the wind against the direction the vehicle actually flies, then
            # ask the question the requirement asks: with the forward authority it
            # has, what ground speed can it HOLD against the headwind?
            # ArduPilot reports NED while Gazebo uses ENU-like world axes; the
            # ArduPilotPlugin transform maps (world x, world y) -> (NED x, -NED y).
            baseline_vectors = []
            t_aim = time.time() + 3.0
            while time.time() < t_aim:
                pos = m.recv_match(type="LOCAL_POSITION_NED", blocking=True, timeout=1)
                v = m.recv_match(type="VFR_HUD", blocking=True, timeout=1)
                if v is not None:
                    rc(alt_hold_stick(v.alt - alt0), pitch=_DASH_SWEEP_PITCH[-1])
                if pos is not None and (pos.vx * pos.vx + pos.vy * pos.vy) > 1.0:
                    baseline_vectors.append((float(pos.vx), float(pos.vy)))
            if baseline_vectors:
                vx = sum(v[0] for v in baseline_vectors) / len(baseline_vectors)
                vy = sum(v[1] for v in baseline_vectors) / len(baseline_vectors)
                norm = (vx * vx + vy * vy) ** 0.5
                ux, uy = vx / norm, vy / norm
                # Desired wind in NED is (-ux, -uy), hence Gazebo world
                # vector (-ux, +uy) under the plugin's y-axis inversion.
                world_wind = (-ux * wind_mps, uy * wind_mps)
                wind_ok = _publish_wind(*world_wind)
                LAST_RESULT.update({
                    "wind_injected": wind_ok,
                    "wind_world_xy_mps": list(world_wind),
                    "wind_ned_xy_mps": [-ux * wind_mps, -uy * wind_mps],
                    "wind_baseline_groundspeed_mps": norm,
                })
                print(
                    f"[wind] injected={wind_ok} {wind_mps:.1f}m/s headwind; "
                    f"baseline groundspeed={norm:.1f}m/s",
                    flush=True,
                )
            else:
                LAST_RESULT["wind_injected"] = False
                print("[wind] no forward velocity vector; wind injection skipped", flush=True)

            fcap = subprocess.Popen(
                ["docker", "exec", _CONTAINER, "gz", "topic", "-e", "-t", topic],
                stdout=open(fwd_cap, "w"), stderr=subprocess.DEVNULL)
            wind_window, wind_verdict, _span = _hold_until_steady(
                m, rc, alt_hold_stick, alt0, _DASH_SWEEP_PITCH[-1],
                label="headwind penetration",
                on_sample=lambda conn: _collect_ned(conn, wind_vectors),
                max_s=_DASH_MAX_S + 15.0)
            if fcap:
                fcap.terminate()
            fwd_speed = (
                sum(v for _t, v in wind_window) / len(wind_window) if wind_window else 0.0
            )
            LAST_RESULT.update({
                "wind_groundspeed_steady_state": wind_verdict.as_dict(),
                "wind_pitch_rc": _DASH_SWEEP_PITCH[-1],
                "wind_authority": "full forward pitch (maximum sustained headway)",
            })
            LAST_RESULT["fwd_speed_steady_state"] = wind_verdict.as_dict()

        fwd_rad_s = _parse_rotor_velocity(fwd_cap)
        fwd_rpm = fwd_rad_s * 9.5493
        if wind_mps > 0:
            wind_ned = LAST_RESULT.get("wind_ned_xy_mps")
            alignment = None
            if wind_ned and wind_vectors:
                vx = sum(v[0] for v in wind_vectors) / len(wind_vectors)
                vy = sum(v[1] for v in wind_vectors) / len(wind_vectors)
                vnorm = (vx * vx + vy * vy) ** 0.5
                wnorm = (wind_ned[0] ** 2 + wind_ned[1] ** 2) ** 0.5
                if vnorm > 0 and wnorm > 0:
                    alignment = -(vx * wind_ned[0] + vy * wind_ned[1]) / (vnorm * wnorm)
            LAST_RESULT.update({
                "wind_groundspeed_mps": fwd_speed,
                "wind_headwind_alignment": alignment,
                "wind_test_complete": bool(LAST_RESULT.get("wind_injected")) and alignment is not None,
            })

        if parachute_deploy:
            from gazebo_poc.safety_precedence_evidence import (
                evaluate_safety_precedence,
            )

            observer, chute_event, chute_seen = _start_model_observer("parachute_small")
            time.sleep(0.2)
            observer_available = observer.poll() is None
            # Offer the critical-propulsion-failure detection to the model and
            # deploy only if its SafetyMonitor fires. Precedence over other
            # safety responses (REQ-SAFE-005) is a separate arbitration claim
            # and is NOT established by this subcheck.
            chute_decisions = ()
            precedence_evidence = None
            # The clock starts BEFORE the hazard is offered, so the model's own
            # decision time is inside the interval. The requirement says
            # "within 0.5 seconds of DETECTING", and detection is where the
            # interval begins — starting it after the model has already decided
            # measures the actuator alone.
            chute_started = time.monotonic()
            if mission is not None:
                competing_actions = mission.action_definitions_for_machine(
                    "SafetyArbiter"
                )
                hazard_state = {
                    "propulsionCriticalFailure": True,
                    "sensorSelfTestFailed": True,
                    "batterySoc": 0.0,
                    "commLossTime": 20.0,
                }
                # Control: the identical hazard state with nothing withheld but
                # the winning condition, on a FRESH machine so no state carries
                # over. Offered under a name the model does not accept, so only
                # the guard-driven competitors are evaluated and the winner's
                # accept transition stays out of it. Without this, an arbiter
                # that suppresses nothing and one that suppresses correctly both
                # produce silence.
                control_mission = ModelDrivenMission(mission_model_text)
                control_decisions = control_mission.offer(
                    "__precedence_control__", time=time.monotonic(),
                    variables=dict(hazard_state, propulsionCriticalFailure=False),
                )
                control_fired = [
                    decision.action_definition for decision in control_decisions
                    if decision.action_definition is not None
                ]
                LAST_RESULT["parachute_precedence_control_fired"] = control_fired
                chute_decisions = mission.offer(
                    "CriticalPropulsionFailure", time=time.monotonic(),
                    variables=hazard_state,
                )
                chute_fired_defs = [
                    decision.action_definition
                    for decision in chute_decisions
                    if decision.action_definition is not None
                ]
                # The expected winner is the model's OWN parachute action:
                # when the failure event fired exactly one distinct action,
                # that action is the response by causal role (run3 names it
                # deployBallisticRecoveryParachute); the adapter constant is
                # only the no-decision fallback.
                winner_action = (
                    chute_fired_defs[0]
                    if len(set(chute_fired_defs)) == 1 and chute_fired_defs
                    else ModelAction.DEPLOY_PARACHUTE.value
                )
                precedence_evidence = evaluate_safety_precedence(
                    control_fired_action_definitions=control_fired,
                    fired_action_definitions=chute_fired_defs,
                    winner_action_definition=winner_action,
                    competing_action_definitions=competing_actions,
                )
                print(f"[model] critical propulsion failure offered; model fired "
                      f"{[d.action for d in chute_decisions]}", flush=True)
            model_deployed = (
                mission is None
                or mission.performed(chute_decisions, ModelAction.DEPLOY_PARACHUTE)
            )
            if model_deployed:
                m.mav.command_long_send(
                    m.target_system,
                    m.target_component,
                    208,  # MAV_CMD_DO_PARACHUTE
                    0,
                    2, 0, 0, 0, 0, 0, 0,  # PARACHUTE_ACTION_RELEASE
                )
            else:
                print("[model] the generated logic declined to deploy the "
                      "parachute; the harness issues no command", flush=True)
            chute_event.wait(timeout=max(2.0, parachute_max_delay_s + 1.0))
            chute_delay = (
                float(chute_seen["at"] - chute_started)
                if chute_seen.get("at") is not None else None
            )
            if observer.poll() is None:
                observer.terminate()
            LAST_RESULT.update({
                "parachute_commanded": model_deployed,
                "parachute_decided_by": (
                    "generated model" if mission is not None else "harness"),
                "parachute_decisions": (
                    [d.as_dict() for d in chute_decisions]
                    if mission is not None else None),
                "action_resolutions": (
                    [list(pair) for pair in mission.action_resolutions]
                    if mission is not None else None),
                "parachute_observer_available": observer_available,
                "parachute_model_observed": chute_delay is not None,
                "parachute_deploy_delay_s": chute_delay,
                "parachute_max_delay_s": parachute_max_delay_s,
                "parachute_precedence_status": (
                    precedence_evidence.status
                    if precedence_evidence is not None else None
                ),
                "parachute_precedence_description": (
                    precedence_evidence.description
                    if precedence_evidence is not None else None
                ),
                "parachute_precedence_competing_actions": (
                    list(precedence_evidence.declared_competing_actions)
                    if precedence_evidence is not None else []
                ),
                "parachute_precedence_competing_actions_fired": (
                    list(precedence_evidence.competing_actions_fired)
                    if precedence_evidence is not None else []
                ),
                "parachute_fidelity": (
                    "generated ParachuteDeploymentBehavior consumed the critical-"
                    "failure event and its entry action drove MAV_CMD_DO_PARACHUTE "
                    "to Gazebo ParachutePlugin model creation/attachment"
                    if mission is not None else
                    "MAV_CMD_DO_PARACHUTE to Gazebo ParachutePlugin model creation/attachment"
                ),
            })
            print(
                f"[parachute] model observed={chute_delay is not None} "
                f"delay={chute_delay} s limit={parachute_max_delay_s}s",
                flush=True,
            )
        if mission is not None:
            LAST_RESULT["mission_decisions"] = mission.decision_log()
            LAST_RESULT["mission_events_offered"] = [
                {"time": round(t, 6), "event": e} for t, e in mission.offered
            ]
        m.mav.rc_channels_override_send(m.target_system, m.target_component, *([0] * 8))
        _cleanup(proc)

        # per-rotor mechanical power (Σ Cp·ρ·n³·D⁵), analytical curve, and backed-out drag area
        from gazebo_poc.forward_flight import power_at_speed, effective_drag_area_from_power
        D = 2 * rotor_radius
        p_hover = _per_rotor_power_w(cap_path, D)
        p_fwd = _per_rotor_power_w(fwd_cap, D)
        p_model = None
        f_eff = None
        if wind_mps <= 0:
            p_model = power_at_speed(
                mass_kg, rotor_count, rotor_radius, max(fwd_speed, 0.1)
            ).power_w
            f_eff = effective_drag_area_from_power(
                p_fwd, max(fwd_speed, 0.1), mass_kg, rotor_count, rotor_radius
            )
        LAST_RESULT.update(fwd_speed_mps=fwd_speed, fwd_power_w=p_fwd, hover_power_w=p_hover,
                           analytical_fwd_power_w=p_model, drag_area_m2=f_eff)
        print(f"[FWD] speed={fwd_speed:.1f} m/s  hover_rpm={hover_rpm:.0f}  fwd_rpm={fwd_rpm:.0f}",
              flush=True)
        if wind_mps <= 0:
            print(f"[FWD] Gazebo power (per-rotor): hover {p_hover:.0f} W → forward {p_fwd:.0f} W  | "
                  f"analytical@{fwd_speed:.0f}m/s = {p_model:.0f} W  | backed-out drag area "
                  f"f={f_eff:.3f} m²", flush=True)
        else:
            print(f"[WIND] groundspeed={fwd_speed:.1f}m/s alignment="
                  f"{LAST_RESULT.get('wind_headwind_alignment')} under {wind_mps:.1f}m/s headwind; "
                  "drag-area backout suppressed because airspeed != groundspeed", flush=True)
        if not thr:
            print("[RESULT] armed but no telemetry", flush=True); return 6
        n = max(1, len(thr) // 3)               # steady-state = last third
        hov_thr = sum(thr[-n:]) / n
        hov_alt = sum(rels[-n:]) / n
        band = max(rels[-n:]) - min(rels[-n:])
        # stable = SUSTAINED altitude near target + small band. A crashed/grounded vehicle has
        # band≈0 (sitting on the ground) and a brief peak, so check the steady altitude is held
        # well above ground — this correctly fails a non-redundant frame after a motor loss.
        #
        # Altitude is necessary but not sufficient. Repeated one-motor-out runs
        # of the same configuration held 1.17, 6.27, 9.93 and 10.00 m with
        # attitude RMS from 1.59 to 19.56 deg — one of them holding altitude
        # beautifully while wobbling 13.5 deg. Requiring both closes that gap;
        # neither signal alone, and no single run, settles the requirement.
        att_rms = LAST_RESULT.get("hover_attitude_rms_deg")
        attitude_ok = (att_rms is None
                       or float(att_rms) <= _HOVER_ATTITUDE_RMS_LIMIT_DEG)
        LAST_RESULT["hover_attitude_limit_deg"] = _HOVER_ATTITUDE_RMS_LIMIT_DEG
        LAST_RESULT["hover_attitude_within_limit"] = (
            None if att_rms is None else attitude_ok
        )
        stable = hov_alt > TGT * 0.5 and band < 1.5 and attitude_ok
        if stable:
            failure_kind = None
            rc_result = 0
        elif thrust_diag.get("sdf_can_hover"):
            failure_kind = "GAZEBO_MODEL_UNCALIBRATED"
            rc_result = 8
        else:
            failure_kind = "DYNAMICS_INFEASIBLE"
            rc_result = 7
        LAST_RESULT.update(hover_stable=stable, hover_throttle_pct=hov_thr, hover_rpm=hover_rpm,
                           hover_alt_m=hov_alt, mass_kg=mass_kg, rotor_radius_m=rotor_radius,
                           rotor_count=rotor_count, capacity_mah=capacity_mah, ok=stable,
                           failure_kind=failure_kind)
        print(f"[RESULT] climb_peak={peak:.2f}m  hover_alt={hov_alt:.2f}m  "
              f"alt_band=±{band/2:.2f}m  hover_throttle={hov_thr:.0f}%", flush=True)
        att_note = (
            "attitude not measured" if att_rms is None else
            f"attitude RMS {float(att_rms):.2f} deg "
            f"(limit {_HOVER_ATTITUDE_RMS_LIMIT_DEG:.1f})"
        )
        print(f"[RESULT] dynamics: "
              f"{'STABLE hover @ %.0f%% throttle' % hov_thr if stable else 'did NOT achieve stable hover'}"
              f"; {att_note}", flush=True)
        return rc_result
    except Exception as e:
        print("[error]", repr(e), flush=True); _cleanup(proc); return 1


if __name__ == "__main__":
    # optional: python run_flight.py <area_override>  (stage-4 calibrated-thrust flight)
    ao = float(sys.argv[1]) if len(sys.argv) > 1 else None
    sys.exit(main(area_override=ao))
