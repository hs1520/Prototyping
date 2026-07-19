"""Finalize the authoritative manifest after same-run Gazebo and SITL reports exist."""
from __future__ import annotations

import base64
import json
import hashlib
import subprocess
import time
from pathlib import Path

from src.prototyping.artifact_provenance import (
    sha256_json,
    sha256_text,
    validate_derived_provenance,
    validate_run_provenance,
)
from src.prototyping.artifact_store import (
    atomic_write_json, atomic_write_text, ensure_open_bundle, output_dir, write_state,
)
from src.prototyping.research_conclusion import (
    derive_research_conclusion, to_markdown as conclusion_to_markdown,
)
from src.prototyping.failure_routing import classify_bundle_evidence


ROOT = Path(__file__).resolve().parents[1]
OUT = output_dir()


def _read(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def _file_sha256(name: str) -> str:
    return hashlib.sha256((OUT / name).read_bytes()).hexdigest()


def _req_id(value: str) -> str:
    return (value or "").upper().replace("-", "_")


def _infrastructure_failure_message(value: str) -> bool:
    low = str(value or "").lower()
    return any(token in low for token in (
        "launch failed", "启动失败", "timed out", "timeout", "异常:",
        "connection refused", "process exited", "docker failed",
    ))


def _terminal_evidence(gazebo: dict, sitl: dict, matrix: dict) -> dict[str, dict]:
    """Normalize executed evidence for deterministic post-run failure routing."""
    evidence: dict[str, dict] = {}

    def item(req_id: str) -> dict:
        return evidence.setdefault(_req_id(req_id), {
            "verifier_available": True,
            "infrastructure_ok": True,
            "oracle_executed": False,
            "oracle_passed": None,
        })

    for row in matrix.get("rows", ()):
        current = item(row.get("req_id", ""))
        status = str(row.get("status", "")).lower()
        if status in {"unassigned", "out-of-sim-scope"}:
            current["verifier_available"] = False
        elif status == "failed":
            current.update(oracle_executed=True, oracle_passed=False)
        elif status == "verified" and current["oracle_passed"] is not False:
            current.update(oracle_executed=True, oracle_passed=True)

    for result in sitl.get("safety_l2", ()):
        current = item(result.get("req_id", ""))
        message = f"{result.get('message', '')} {result.get('note', '')}"
        if _infrastructure_failure_message(message):
            current.update(infrastructure_ok=False, oracle_executed=False)
            continue
        passed = bool(result.get("passed"))
        current.update(oracle_executed=True, oracle_passed=passed)

    gazebo_failed_to_run = str(gazebo.get("status", "")).upper() == "FAILED"
    for result in gazebo.get("req_results", ()):
        current = item(result.get("req_id", ""))
        status = str(result.get("status", "")).upper()
        message = str(result.get("message", ""))
        if gazebo_failed_to_run and status == "PLANNED":
            current.update(infrastructure_ok=False, oracle_executed=False)
            continue
        if status == "FAIL":
            current.update(oracle_executed=True, oracle_passed=False)
        elif status == "PASS" and current["oracle_passed"] is not False:
            current.update(oracle_executed=True, oracle_passed=True)
        elif status == "INCONCLUSIVE" and any(token in message.lower() for token in (
            "observer", "not available", "unavailable", "no implemented",
        )):
            current["verifier_available"] = False
        elif status in {"PARTIAL", "PLANNED"}:
            current["verifier_available"] = True
        elif status == "SUSPENDED":
            current["verifier_available"] = False
    return evidence


def main() -> int:
    ensure_open_bundle(OUT)
    run = _read("realization_run.json")
    model = (OUT / "final_model.sysml").read_text(encoding="utf-8")
    parm = (OUT / "recommended.parm").read_text(encoding="utf-8")
    ok, reason = validate_run_provenance(run, model_sysml=model, parm_text=parm)
    if not ok:
        raise SystemExit(f"STALE base artifact set: {reason}")
    provenance = run["artifact_provenance"]
    canonical = _read("canonical_run.json")
    if canonical.get("run_id") != provenance["run_id"]:
        raise SystemExit("STALE canonical run: run_id mismatch")
    if canonical.get("artifact_provenance") != provenance:
        raise SystemExit("STALE canonical run: provenance mismatch")
    if canonical.get("requirements") != run.get("requirements"):
        raise SystemExit("STALE canonical run: exact requirement set mismatch")
    gazebo = _read("gazebo_feasibility_report.json")
    ok, reason = validate_derived_provenance(gazebo, run, model)
    if not ok:
        raise SystemExit(f"STALE Gazebo report: {reason}")
    sitl = _read("sitl_feasibility_report.json")
    if sitl.get("source_provenance") != provenance:
        raise SystemExit("STALE SITL report: source provenance does not match this run")
    matrix = _read("verification_matrix.json")
    if matrix.get("source_provenance") != provenance:
        raise SystemExit("STALE verification matrix: source provenance does not match this run")
    matrix_summary = matrix.get("summary") or sitl.get("verification_matrix")
    rows = matrix.get("rows") or []
    if len(rows) != len(run.get("requirements") or []):
        raise SystemExit(
            "INCOMPLETE verification matrix: row count does not match exact requirement set"
        )
    try:
        research_conclusion = derive_research_conclusion(run, gazebo, sitl, matrix)
    except ValueError as exc:
        raise SystemExit(f"INCONSISTENT research conclusion inputs: {exc}") from exc

    robustness_enabled = any((run.get("robustness_options") or {}).values())
    if robustness_enabled:
        contracts = _read("requirement_contracts.json")
        semantic_trace = _read("semantic_trace_report.json")
        repair_decisions = classify_bundle_evidence(
            contracts,
            semantic_trace,
            _terminal_evidence(gazebo, sitl, matrix),
        )
        prior_routes = _read("repair_decisions.json")
        for key in (
            "artifact_schema_version", "contract_library_version",
            "pattern_library_version", "platform_binding_version",
            "requirement_source_digests", "model_digest", "producing_stage",
            "semantic_repair_attempts",
        ):
            if key in prior_routes:
                repair_decisions[key] = prior_routes[key]
        repair_decisions["producing_stage"] = "post_high_fidelity_routing"
        metrics = _read("robustness_metrics.json")
        metrics["post_evidence_route_counts"] = repair_decisions["counts"]
        atomic_write_json(OUT / "repair_decisions.json", repair_decisions)
        atomic_write_json(OUT / "robustness_metrics.json", metrics)
        run["repair_decisions"] = repair_decisions
        run["robustness_metrics"] = metrics

    phase9 = {
        "gazebo": {
            "status": gazebo.get("status"),
            "source_provenance": gazebo.get("source_provenance"),
        },
        "sitl": {
            "flight_passed": (sitl.get("flight") or {}).get("passed"),
            "safety_status": (sitl.get("safety_verification") or {}).get("status"),
            "l2_passed": sum(bool(x.get("passed")) for x in sitl.get("safety_l2", [])),
            "l2_total": len(sitl.get("safety_l2", [])),
            "source_provenance": sitl.get("source_provenance"),
        },
        "verification_matrix": {"row_count": len(rows)},
    }
    run["phase9_hifi"] = phase9
    run["research_conclusion"] = {
        "artifact": "research_conclusion.json",
        "overall": research_conclusion["overall"],
    }
    atomic_write_json(OUT / "realization_run.json", run)
    atomic_write_json(OUT / "research_conclusion.json", research_conclusion)
    atomic_write_text(
        OUT / "research_conclusion.md", conclusion_to_markdown(research_conclusion)
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout
    atomic_write_text(OUT / "working_tree.patch", diff)
    untracked: dict[str, str] = {}
    untracked_archive: dict[str, dict[str, str]] = {}
    for line in dirty:
        if not line.startswith("?? "):
            continue
        rel = line[3:]
        path = ROOT / rel
        if path.is_file():
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            untracked[rel] = digest
            untracked_archive[rel] = {
                "sha256": digest,
                "encoding": "base64",
                "content": base64.b64encode(data).decode("ascii"),
            }
    atomic_write_json(OUT / "working_tree_untracked.json", {
        "format": "base64-file-map-v1",
        "files": untracked_archive,
    })
    code_state = {
        "git_commit": commit,
        "git_diff_sha256": sha256_text(diff),
        "dirty_worktree": dirty,
        "untracked_file_sha256": untracked,
    }
    code_state["code_state_sha256"] = sha256_json(code_state)
    artifact_files = [
        "canonical_run.json",
        "realization_run.json",
        "final_model.sysml",
        "recommended.parm",
        "gazebo_feasibility_report.json",
        "sitl_feasibility_report.json",
        "verification_matrix.json",
        "research_conclusion.json",
        "research_conclusion.md",
        "working_tree.patch",
        "working_tree_untracked.json",
    ]
    robustness_files = [
        "requirement_contracts.json",
        "safety_pattern_bindings.json",
        "semantic_trace_report.json",
        "failure_diagnostics.json",
        "repair_decisions.json",
        "robustness_metrics.json",
    ]
    if robustness_enabled:
        missing = [name for name in robustness_files if not (OUT / name).exists()]
        if missing:
            raise SystemExit(
                "INCOMPLETE robustness artifact set: " + ", ".join(missing)
            )
        artifact_files.extend(robustness_files)
    artifact_sha256 = {name: _file_sha256(name) for name in artifact_files}
    artifact_index = {
        "canonical_run": "canonical_run.json",
        "realization": "realization_run.json",
        "model": "final_model.sysml",
        "parm": "recommended.parm",
        "gazebo": "gazebo_feasibility_report.json",
        "sitl": "sitl_feasibility_report.json",
        "verification_matrix": "verification_matrix.json",
        "research_conclusion": "research_conclusion.json",
        "research_conclusion_markdown": "research_conclusion.md",
        "code_diff": "working_tree.patch",
        "untracked_code_archive": "working_tree_untracked.json",
    }
    if robustness_enabled:
        artifact_index.update({
            "requirement_contracts": "requirement_contracts.json",
            "safety_pattern_bindings": "safety_pattern_bindings.json",
            "semantic_trace": "semantic_trace_report.json",
            "failure_diagnostics": "failure_diagnostics.json",
            "repair_decisions": "repair_decisions.json",
            "robustness_metrics": "robustness_metrics.json",
        })
    authority = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifact_provenance": provenance,
        "provenance_validated": True,
        "code_state": code_state,
        "bundle_artifact_sha256": artifact_sha256,
        "phase8": {
            "verdict": (run.get("realization") or {}).get("verdict"),
            "chosen": (run.get("realization") or {}).get("chosen"),
            "failed_checks": (run.get("realization") or {}).get("failed_checks"),
        },
        "phase9": phase9,
        "verification_matrix_summary": matrix_summary,
        "research_conclusion": research_conclusion,
        "artifacts": artifact_index,
    }
    atomic_write_json(OUT / "authoritative_run.json", authority)
    write_state(OUT, "FINAL", run_id=provenance["run_id"], provenance_validated=True)
    print(json.dumps({
        "run_id": provenance["run_id"],
        "phase8": authority["phase8"]["verdict"],
        "gazebo": phase9["gazebo"]["status"],
        "sitl_flight": phase9["sitl"]["flight_passed"],
        "sitl_safety": phase9["sitl"]["safety_status"],
        "matrix": matrix_summary,
        "research_conclusion": research_conclusion["overall"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
