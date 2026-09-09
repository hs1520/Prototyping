"""Two-step acceptance gate for SITL --speedup adoption (native L2 suite)."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def run_leg(model_path: str, out_path: str, speedup: float) -> int:
    from src.sysml.lite_model import build_lite_model
    from src.sitl.sitl_bridge import ARDUPILOT_COPTER_PROFILE, SITLBridge

    model = build_lite_model(
        Path(model_path).read_text(), model_name="AutonomousDrone"
    )
    kwargs = {}
    if speedup != 1.0:
        # Only the converted bridge accepts this; the baseline leg runs without
        # it, as step 1 requires.
        kwargs["speedup"] = speedup
    bridge = SITLBridge(
        model,
        output_dir=str(REPO / "experiments" / "speedup_gate_results" / "sitl"),
        platform_profile=ARDUPILOT_COPTER_PROFILE,
        fdm_backend="native",
        **kwargs,
    )
    bridge.generate_l1()
    started = time.time()
    results = bridge.run_l2(per_test_sitl=True)
    wall_total = time.time() - started

    git_rev = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()
    payload = {
        "model": model_path,
        "speedup": speedup,
        "git_rev": git_rev,
        "wall_total_s": round(wall_total, 1),
        "results": [
            {
                "req_id": r.req_id,
                "passed": r.passed,
                "message": r.message,
                "duration_s": round(r.duration_s, 2),
            }
            for r in results
        ],
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n[gate] wrote {out}  (wall {wall_total:.0f}s)")
    for r in results:
        print(f"[gate]   {'PASS' if r.passed else 'FAIL'} {r.req_id} "
              f"({r.duration_s:.0f}s) {r.message[:90]}")
    return 0


_LATENCY_RE = re.compile(r"(\d+\.\d+)\s*s\b")


def compare(a_path: str, b_path: str, tolerance_s: float) -> int:
    a = json.loads(Path(a_path).read_text())
    b = json.loads(Path(b_path).read_text())
    a_by_id = {r["req_id"]: r for r in a["results"]}
    b_by_id = {r["req_id"]: r for r in b["results"]}
    failures: list[str] = []
    print(f"[gate] {a_path} (speedup {a['speedup']}, {a['git_rev']}) vs "
          f"{b_path} (speedup {b['speedup']}, {b['git_rev']})")
    for req_id in sorted(set(a_by_id) | set(b_by_id)):
        ra, rb = a_by_id.get(req_id), b_by_id.get(req_id)
        if ra is None or rb is None:
            failures.append(f"{req_id}: present in only one leg")
            continue
        verdict = "==" if ra["passed"] == rb["passed"] else "!!"
        if ra["passed"] != rb["passed"]:
            failures.append(
                f"{req_id}: verdict flipped "
                f"{ra['passed']} -> {rb['passed']} ({rb['message'][:80]})"
            )
        la = _LATENCY_RE.findall(ra["message"])
        lb = _LATENCY_RE.findall(rb["message"])
        lat_note = ""
        if la and lb:
            da, db = float(la[0]), float(lb[0])
            lat_note = f"  measured {da:.3f}s vs {db:.3f}s"
            if abs(da - db) > tolerance_s:
                failures.append(
                    f"{req_id}: measurement drift {da:.3f}s -> {db:.3f}s "
                    f"(> {tolerance_s}s tolerance)"
                )
        print(f"[gate]  {verdict} {req_id}: {ra['passed']}/{rb['passed']}"
              f"{lat_note}")
    speedup_ratio = (
        a["wall_total_s"] / b["wall_total_s"] if b["wall_total_s"] else 0
    )
    print(f"[gate] wall time {a['wall_total_s']}s -> {b['wall_total_s']}s "
          f"({speedup_ratio:.1f}x)")
    if failures:
        print("[gate] GATE FAILED:")
        for f in failures:
            print("[gate]   -", f)
        return 1
    print("[gate] GATE PASSED: verdicts identical, measurements within "
          f"{tolerance_s}s")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--model", default="examples/output/final_model.sysml")
    run_p.add_argument("--out", required=True)
    run_p.add_argument("--speedup", type=float, default=1.0)
    cmp_p = sub.add_parser("compare")
    cmp_p.add_argument("a")
    cmp_p.add_argument("b")
    cmp_p.add_argument("--tolerance-s", type=float, default=0.3)
    args = parser.parse_args()
    if args.cmd == "run":
        return run_leg(args.model, args.out, args.speedup)
    return compare(args.a, args.b, args.tolerance_s)


if __name__ == "__main__":
    raise SystemExit(main())
