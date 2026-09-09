"""Remove one evidence source at a time from the reported evidence matrix.

Rebuilds the matrix of examples/output/run3_authoritative_20260831 from its
archived artefacts with all four evidence sources, checks the result against
the Appendix G statuses, then drops the observed results of one source at a
time and re-aggregates. Nothing external is rerun. The deterministic
model-level behavioural simulation inside the matrix builder does run; for the
behavioural variant it is silenced by disabling every write of a behavioural
outcome. Removing a source removes its observed results only: SITL and Gazebo
test intents stay recorded as planned or deferred.

Usage: python experiments/source_removal.py <output.json>
"""
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.prototyping import verification_matrix as vm  # noqa: E402
from src.sitl.requirement_linker import RequirementLinker  # noqa: E402
from src.sysml.lite_model import build_lite_model  # noqa: E402

RUN = REPO / "examples/output/run3_authoritative_20260831"
INPUTS = (
    "final_model.sysml",
    "gazebo_feasibility_latest.json",
    "realization_per_requirement_rebuilt.json",
    "run_report.json",
    "sitl_l2_final_7of7.json",
)

# Requirement statuses as printed in Appendix G; the baseline must reproduce them.
APPENDIX_G = {
    "REQ-CONS-001": "verified", "REQ-CONS-002": "out-of-sim-scope",
    "REQ-CONS-003": "verified", "REQ-CONS-004": "out-of-sim-scope",
    "REQ-FUNC-001": "partial", "REQ-FUNC-002": "verified",
    "REQ-FUNC-003": "verified", "REQ-FUNC-004": "out-of-sim-scope",
    "REQ-FUNC-005": "verified", "REQ-FUNC-006": "verified",
    "REQ-FUNC-007": "verified", "REQ-FUNC-008": "partial",
    "REQ-INTF-001": "out-of-sim-scope", "REQ-INTF-002": "partial",
    "REQ-INTF-003": "out-of-sim-scope",
    "REQ-PERF-001": "verified", "REQ-PERF-002": "verified",
    "REQ-PERF-003": "verified", "REQ-PERF-004": "verified",
    "REQ-PERF-005": "verified", "REQ-PERF-006": "verified",
    "REQ-SAFE-001": "partial", "REQ-SAFE-002": "partial",
    "REQ-SAFE-003": "verified", "REQ-SAFE-004": "blocked",
    "REQ-SAFE-005": "verified", "REQ-SAFE-006": "partial",
    "REQ-SAFE-007": "failed", "REQ-SAFE-008": "verified",
}


def sha(name):
    return hashlib.sha256((RUN / name).read_bytes()).hexdigest()


def load_inputs():
    model = build_lite_model((RUN / "final_model.sysml").read_text(), model_name="AutonomousDrone")
    run = json.loads((RUN / "run_report.json").read_text())
    bundle = RequirementLinker(model, llm=None, plan_payload=run["whole_model_generation_plan"]).compile_evidence()
    realization = dict(run["realization"])
    rebuilt = json.loads((RUN / "realization_per_requirement_rebuilt.json").read_text())
    realization["per_requirement"] = rebuilt["per_requirement"]
    realization["forward_flight_ok"] = rebuilt.get("forward_flight_ok")
    gazebo = json.loads((RUN / "gazebo_feasibility_latest.json").read_text())
    sitl = json.loads((RUN / "sitl_l2_final_7of7.json").read_text())
    return model, bundle, realization, gazebo, sitl["l1_results"], sitl["results"]


def build(model, bundle, realization, gazebo, l1, l2, no_behaviour=False):
    original = vm._record_behavioral_outcome
    if no_behaviour:
        vm._record_behavioral_outcome = lambda *a, **k: False
    try:
        return vm.build_matrix(model, realization, bundle, gazebo=gazebo, l1_results=l1, l2_results=l2)
    finally:
        vm._record_behavioral_outcome = original


def digest(rows):
    status = {r.req_id.replace("_", "-"): r.status for r in rows}
    obligations = {
        f"{r.req_id.replace('_', '-')}/{o.obligation_id}": o.status
        for r in rows for o in r.obligations
    }
    return status, obligations, vm.summarize(rows)


def main(out_path):
    model, bundle, realization, gazebo, l1, l2 = load_inputs()
    base_rows = build(model, bundle, realization, gazebo, l1, l2)
    base, base_ob, base_summary = digest(base_rows)
    mismatch = [(k, APPENDIX_G[k], base.get(k)) for k in APPENDIX_G if base.get(k) != APPENDIX_G[k]]
    if mismatch:
        raise SystemExit(f"baseline does not reproduce Appendix G: {mismatch}")
    print("baseline reproduces Appendix G on all 29 requirements")
    print("baseline requirement counts:", dict(Counter(base.values())))
    print("baseline obligations: total", base_summary["obligations_total"],
          "in scope", base_summary["obligations_in_sim_scope"],
          "verified", base_summary["obligations_verified"])

    variants = {
        "catalogue calculations": dict(realization={"per_requirement": [], "forward_flight_ok": None}),
        "behavioural execution": dict(no_behaviour=True),
        "SITL": dict(l1=None, l2=None),
        "Gazebo": dict(gazebo=None),
    }
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True, cwd=REPO).stdout.strip()
    out = {
        "code_commit": commit,
        "inputs": {name: sha(name) for name in INPUTS},
        "baseline": {"status": base, "obligations": base_ob, "summary": base_summary},
        "variants": {},
    }
    for name, overrides in variants.items():
        kwargs = dict(model=model, bundle=bundle, realization=realization, gazebo=gazebo, l1=l1, l2=l2)
        kwargs.update(overrides)
        status, obligations, summary = digest(build(**kwargs))
        changed_req = {k: (base[k], status[k]) for k in base if status.get(k) != base[k]}
        changed_ob = {k: (base_ob[k], obligations.get(k)) for k in base_ob if obligations.get(k) != base_ob[k]}
        out["variants"][name] = {
            "status": status, "summary": summary,
            "changed_requirements": changed_req, "changed_obligations": changed_ob,
            "counts": dict(Counter(status.values())),
        }
        print(f"\n== without {name}: {len(changed_req)} requirement(s) change; "
              f"verified obligations {summary['obligations_verified']}/{summary['obligations_in_sim_scope']}")
        for k, (a, b) in sorted(changed_req.items()):
            print(f"   {k}: {a} -> {b}")
        print("   counts:", dict(Counter(status.values())))
    Path(out_path).write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
