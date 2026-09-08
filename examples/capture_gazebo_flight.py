"""Film the realised design flying in Gazebo, intact and with a rotor failed.

Adds fixed cameras to the Gazebo world and captures frames while the ordinary
`run_flight` flight runs, producing the verification-chapter stills and an HTML
player for the two flights. It measures nothing new: every number comes from
`run_flight.LAST_RESULT`, the same source the feasibility harness reads.

No GUI path exists: the image is headless (`gz sim -s --headless-rendering`,
ogre2 software rendering on aarch64) and neither host nor image has PIL or
ffmpeg, so frames come off a camera topic through the in-container gz-transport
bindings (see gazebo_capture_grab.py) and are PNG-encoded here with zlib.
Nothing in gazebo_poc/ is modified; the camera world reaches the flight through
a monkeypatch on `run_flight._sh` that adds bind-mounts to its `docker run`.
Three cameras: `camtele` on the 5.6-9.4 m hover band (intact vehicle), `camlow`
on 0-3.3 m (rotor-out vehicle, which never leaves that band), `camwide` for
context. `max_thrust_g` and `hover_throttle` come from the catalogue
realisation record, as in `run_gazebo_feasibility.py`; without them SITL flies
a generic thrust model, the intact vehicle hovers at 7.65 m on 22% instead of
8.6 m on 43%, and the rotor-out case reports a stable hover.

Archived result (2026-08-29) in
examples/output/probe_gazebo_flight_capture_20260829/:
  - gazebo-hover-nominal.png  : six rotors, hovering at 8.57 m on 42.8%
    throttle, 2816 rpm; altitude held to +/-0.05 m.
  - gazebo-hover-rotorout.png : one rotor disabled, climb peak 1.32 m, hover
    altitude 0.07 m, return code 8; no stable hover.
  - flight_results.json       : LAST_RESULT for both flights.
Runs f99ac140 and 00e4d333 both recorded motor_failure_tolerant=false with
return code 8 and REQ-SAFE-007 FAIL.

Run: PYTHONPATH=. python examples/probe_gazebo_flight_capture.py
Requires Docker with the `headless_gazebo` image and a free port 5760; an
authoritative run holds that port during its SITL stage.
"""
from __future__ import annotations

import base64
import datetime
import json
import pathlib
import struct
import subprocess
import sys
import threading
import time
import zlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gazebo_poc.run_flight as rf  # noqa: E402

GRABBER = pathlib.Path(__file__).with_name("gazebo_capture_grab.py")

DESIGN = dict(
    mass_kg=5.54,
    rotor_radius=0.2032,
    capacity_mah=16000,
    rotor_count=6,
    calibrate=True,
    max_thrust_g=1975.6381250000006,
    hover_throttle=0.5405405405405405,
)

CAMERA = """
    <model name="{name}">
      <static>true</static>
      <pose>{pose}</pose>
      <link name="l">
        <sensor name="c" type="camera">
          <always_on>1</always_on>
          <update_rate>{rate}</update_rate>
          <topic>{name}</topic>
          <camera>
            <horizontal_fov>{hfov}</horizontal_fov>
            <image><width>{w}</width><height>{h}</height></image>
            <clip><near>0.1</near><far>500</far></clip>
          </camera>
        </sensor>
      </link>
    </model>
"""

CAMERAS = [
    dict(name="camwide", pose="-11 0 4.5 0 0 0", w=640, h=360, hfov=1.65, rate=4),
    dict(name="camtele", pose="-11 0 1.5 0 -0.50 0", w=640, h=360, hfov=0.55, rate=4),
    dict(name="camlow", pose="-8 0 1.5 0 -0.06 0", w=640, h=360, hfov=0.80, rate=4),
]


def build_camera_world(out: pathlib.Path) -> pathlib.Path:
    """Stock runway world with the cameras appended."""
    stock = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "cat", rf._IMG, rf._WORLD_PATH],
        capture_output=True, text=True,
    )
    if stock.returncode != 0:
        raise RuntimeError(f"cannot read stock world: {stock.stderr.strip()}")
    body = "".join(CAMERA.format(**c) for c in CAMERAS)
    out.write_text(stock.stdout.replace("</world>", body + "\n  </world>"))
    return out


def fly(world: pathlib.Path, frames_dir: pathlib.Path, fail_rotor=None) -> dict:
    """Run one flight with the cameras mounted, capturing frames throughout."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    original_sh = rf._sh

    def sh_with_cameras(*args, **kw):
        if len(args) > 1 and args[0] == "docker" and args[1] == "run" and rf._IMG in args:
            args = list(args)
            i = args.index(rf._IMG)
            args[i:i] = [
                "-v", f"{world.resolve()}:{rf._WORLD_PATH}",
                "-v", f"{GRABBER.resolve()}:/grab.py",
                "-v", f"{frames_dir.resolve()}:/out",
            ]
            args = tuple(args)
        return original_sh(*args, **kw)

    def capture():
        for _ in range(120):
            state = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", rf._CONTAINER],
                capture_output=True, text=True,
            )
            if state.stdout.strip() == "true":
                break
            time.sleep(1)
        else:
            print("[film] container never came up", flush=True)
            return
        time.sleep(6)                      # let the sensors system come up
        topics = ",".join(c["name"] for c in CAMERAS)
        subprocess.run(
            ["docker", "exec", rf._CONTAINER, "python3", "/grab.py", topics, "700", "900"],
            capture_output=True, text=True,
        )

    rf._sh = sh_with_cameras
    worker = threading.Thread(target=capture, daemon=True)
    worker.start()
    try:
        kwargs = dict(DESIGN)
        if fail_rotor is not None:
            kwargs["fail_rotor"] = fail_rotor
        print(f"[film] flying {kwargs}", flush=True)
        code = rf.main(**kwargs)
    finally:
        rf._sh = original_sh
    worker.join(timeout=30)
    result = dict(rf.LAST_RESULT)
    result["return_code"] = code
    return result


def encode_png(rgb: bytes, w: int, h: int) -> bytes:
    raw = b"".join(b"\x00" + rgb[y*w*3:(y+1)*w*3] for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b""))


def halve(rgb: bytes, w: int, h: int) -> bytes:
    """2x2 box downsample."""
    ow, oh = w // 2, h // 2
    out = bytearray(ow * oh * 3)
    for y in range(oh):
        top, bottom, dst = (2*y)*w*3, (2*y+1)*w*3, y*ow*3
        for x in range(ow):
            a, b = top + 6*x, bottom + 6*x
            for c in range(3):
                out[dst + 3*x + c] = (rgb[a+c] + rgb[a+3+c] + rgb[b+c] + rgb[b+3+c]) >> 2
    return bytes(out)


def crop_still(frames_dir: pathlib.Path, cam: str, index: int,
               cx: int, cy: int, cw: int = 288, ch: int = 162) -> bytes:
    d = frames_dir / cam
    w, h, _ = (d / "meta.txt").read_text().split()
    w, h = int(w), int(h)
    src = sorted(d.glob("f*.raw"))[index].read_bytes()
    x0 = max(0, min(w - cw, cx - cw // 2))
    y0 = max(0, min(h - ch, cy - ch // 2))
    out = bytearray()
    for y in range(y0, y0 + ch):
        out += src[(y*w + x0)*3:(y*w + x0 + cw)*3]
    return encode_png(bytes(out), cw, ch)


def clip_frames(frames_dir: pathlib.Path, cam: str, lo: int, hi: int,
                keep: int = 72) -> tuple[list[str], int, int]:
    d = frames_dir / cam
    w, h, _ = (d / "meta.txt").read_text().split()
    w, h = int(w), int(h)
    files = sorted(d.glob("f*.raw"))[lo:hi]
    step = max(1, len(files) // keep)
    picked = files[::step][:keep]
    return ([base64.b64encode(encode_png(halve(f.read_bytes(), w, h), w//2, h//2)).decode()
             for f in picked], w // 2, h // 2)


def main() -> int:
    stamp = datetime.date.today().strftime("%Y%m%d")
    out_dir = ROOT / "examples/output" / f"probe_gazebo_flight_capture_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_frames"
    world = build_camera_world(out_dir / "camera_world.sdf")

    results = {}
    results["nominal"] = fly(world, work / "nominal")
    results["rotor_out"] = fly(world, work / "rotor_out", fail_rotor=0)

    (out_dir / "flight_results.json").write_text(
        json.dumps(results, indent=2, default=str)
    )

    # Stills: intact vehicle in the hover band, rotor-out vehicle on the runway.
    # Frame indices come from the archived capture; a fresh run may need others.
    (out_dir / "gazebo-hover-nominal.png").write_bytes(
        crop_still(work / "nominal", "camtele", 265, 296, 90))
    (out_dir / "gazebo-hover-rotorout.png").write_bytes(
        crop_still(work / "rotor_out", "camlow", 135, 325, 272))

    for name, res in results.items():
        print(f"{name}: stable={res.get('hover_stable')} "
              f"alt={res.get('hover_alt_m')} "
              f"throttle={res.get('hover_throttle_pct')} rc={res.get('return_code')}")
    print(f"artefacts in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
