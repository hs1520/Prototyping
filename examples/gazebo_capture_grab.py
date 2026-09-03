"""Frame subscriber that runs inside the Gazebo container.

Bind-mounted by probe_gazebo_flight_capture.py and run with `docker exec`,
because the gz-transport Python bindings live in the image, not on the host.
Writes raw RGB frames, one directory per camera topic.

argv: <topics-csv> <max_frames_each> <deadline_seconds>
"""
import pathlib
import sys
import time

from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image

topics = (sys.argv[1] if len(sys.argv) > 1 else "camwide").split(",")
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 700
seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 900.0

node = Node()
counts: dict[str, list[int]] = {}

for topic in topics:
    out = pathlib.Path("/out") / topic
    out.mkdir(parents=True, exist_ok=True)
    counts[topic] = [0]

    def make_cb(name: str, directory: pathlib.Path):
        def cb(msg: Image) -> None:
            n = counts[name][0]
            if n >= limit:
                return
            (directory / f"f{n:05d}.raw").write_bytes(msg.data)
            if n == 0:
                (directory / "meta.txt").write_text(
                    f"{msg.width} {msg.height} {msg.pixel_format_type}\n"
                )
            counts[name][0] = n + 1
        return cb

    if not node.subscribe(Image, "/" + topic, make_cb(topic, out)):
        print(f"subscribe failed: {topic}", flush=True)

print(f"subscribed {topics}", flush=True)
deadline = time.time() + seconds
while time.time() < deadline and any(c[0] < limit for c in counts.values()):
    time.sleep(0.25)
print("captured " + ", ".join(f"{t}={counts[t][0]}" for t in topics), flush=True)
