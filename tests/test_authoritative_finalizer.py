from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from src.prototyping.artifact_provenance import build_run_provenance
from src.prototyping.artifact_store import OUTPUT_DIR_ENV, write_state
from src.realization.matcher import mapping_policy


ROOT = Path(__file__).resolve().parents[1]
FINALIZER = ROOT / "examples" / "finalize_authoritative_run.py"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _bundle(path: Path) -> dict:
    model = "package D {}\n"
    parm = "FRAME_CLASS 1\n"
    requirements = [{"id": "REQ-1", "text": "The drone shall fly."}]
    run = {
        "requirements": requirements,
        "recommended_design_inputs": {
            "rotor_count": 4, "battery_cells": 6,
            "rotor_radius_m": 0.20, "battery_capacity_mah": 10000,
        },
        "realization": {
            "verdict": "CLOSED",
            "mapping_policy": mapping_policy(),
            "chosen": {
                "frame": "F", "rotor_count": 4, "battery_cells": 6,
                "rotor_radius_m": 0.20, "battery_capacity_mah": 10000,
                "design_drift": [
                    {"name": "rotor_radius_m", "expected": 0.20, "realized": 0.20,
                     "relative_delta": 0.0, "limit": 0.10, "within_limit": True},
                    {"name": "battery_capacity_mah", "expected": 10000, "realized": 10000,
                     "relative_delta": 0.0, "limit": 0.10, "within_limit": True},
                    {"name": "battery_cells", "expected": 6, "realized": 6,
                     "relative_delta": 0.0, "limit": 0.0, "within_limit": True},
                ],
            },
            "per_requirement": [{
                "req_id": "REQ-1", "scope": "closure", "family": "time", "met": True,
            }],
        },
    }
    run["artifact_provenance"] = build_run_provenance(
        model_sysml=model,
        recommended_design=run["recommended_design_inputs"],
        realization=run["realization"],
        requirements=requirements,
        parm_text=parm,
        run_id="test-authority",
    )
    provenance = run["artifact_provenance"]
    path.mkdir()
    (path / "final_model.sysml").write_text(model, encoding="utf-8")
    (path / "recommended.parm").write_text(parm, encoding="utf-8")
    _write_json(path / "realization_run.json", run)
    _write_json(path / "canonical_run.json", {
        "run_id": provenance["run_id"],
        "requirements": requirements,
        "artifact_provenance": provenance,
    })
    _write_json(path / "gazebo_feasibility_report.json", {
        "status": "PASS", "source_provenance": provenance,
        "req_results": [{"req_id": "REQ-1", "status": "PASS"}],
    })
    _write_json(path / "sitl_feasibility_report.json", {
        "source_provenance": provenance,
        "flight": {"passed": True},
        "safety_verification": {"status": "PASS"},
        "safety_l2": [{"passed": True}],
        "verification_matrix": {"total": 1, "by_status": {"verified": 1}},
    })
    _write_json(path / "verification_matrix.json", {
        "source_provenance": provenance,
        "summary": {
            "total": 1,
            "by_status": {"verified": 1},
            "by_tier": {"gazebo": 1},
            "unassigned_req_ids": [],
        },
        "rows": [{"req_id": "REQ-1", "status": "verified"}],
    })
    write_state(path, "EVIDENCE", run_id=provenance["run_id"])
    return run


def _finalize(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(FINALIZER)],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT), OUTPUT_DIR_ENV: str(path)},
        capture_output=True,
        text=True,
    )


def test_finalizer_closes_a_complete_same_run_bundle(tmp_path):
    bundle = tmp_path / "bundle"
    _bundle(bundle)

    result = _finalize(bundle)

    assert result.returncode == 0, result.stderr
    state = json.loads((bundle / "run_state.json").read_text())
    authority = json.loads((bundle / "authoritative_run.json").read_text())
    assert state["state"] == "FINAL"
    assert authority["provenance_validated"] is True
    assert authority["artifact_provenance"]["run_id"] == "test-authority"
    assert "working_tree.patch" in authority["bundle_artifact_sha256"]
    assert "working_tree_untracked.json" in authority["bundle_artifact_sha256"]
    archive = json.loads((bundle / "working_tree_untracked.json").read_text())
    assert archive["format"] == "base64-file-map-v1"
    assert authority["research_conclusion"]["overall"] == "SUPPORTED"
    assert (bundle / "research_conclusion.md").exists()


def test_finalizer_rejects_a_changed_requirement_set(tmp_path):
    bundle = tmp_path / "bundle"
    run = _bundle(bundle)
    run["requirements"][0]["text"] = "A different requirement."
    _write_json(bundle / "realization_run.json", run)

    result = _finalize(bundle)

    assert result.returncode != 0
    assert "requirement-set fingerprint" in result.stderr
    assert json.loads((bundle / "run_state.json").read_text())["state"] == "EVIDENCE"
    assert not (bundle / "authoritative_run.json").exists()
