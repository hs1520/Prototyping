from __future__ import annotations

import pytest

import hashlib
import json

import examples.run_option2_pilot as pilot
from examples.drone_system_v2 import DRONE_REQUIREMENTS
from examples.run_option2_pilot import main
from src.prototyping.requirement_contracts import build_contract_bundle
from src.prototyping.requirement_inputs import (
    build_frozen_requirement_set,
    normalise_requirement_id,
)
from src.prototyping.robustness_gold import (
    GOLD_SCHEMA_VERSION,
    OPTION2_GOLD_REQUIREMENT_IDS,
    ROUTING_POLICY,
)


def _reviewed_inputs():
    requirements = [
        requirement
        for req_id in OPTION2_GOLD_REQUIREMENT_IDS
        for requirement in DRONE_REQUIREMENTS
        if normalise_requirement_id(requirement) == req_id
    ]
    frozen = build_frozen_requirement_set(requirements, source="test")
    bundle = build_contract_bundle(requirements)
    rows = []
    for contract in bundle.contracts:
        rows.append({
            "req_id": contract.req_id,
            "source_text": contract.source_text,
            "source_digest": contract.source_digest,
            "review_status": "REVIEWED",
            "reviewed_contract": contract.to_dict(),
            "reviewed_trace_links": [],
        })
    gold = {
        "schema_version": GOLD_SCHEMA_VERSION,
        "review_mode": "SOURCE_FIRST",
        "selected_requirement_ids": list(OPTION2_GOLD_REQUIREMENT_IDS),
        "selected_source_fingerprint": hashlib.sha256(json.dumps(
            [
                (contract.req_id, contract.source_digest)
                for contract in bundle.contracts
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest(),
        "requirement_set_fingerprint": "requirement-set-test",
        "routing_policy": ROUTING_POLICY,
        "reviewer": "Supervisor",
        "reviewed_at": "2026-07-20T12:00:00Z",
        "requirements": rows,
    }
    return frozen, gold


def test_b2_pilot_rejects_a_single_outer_iteration(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--config", "B2", "--max-iterations", "1"])

    assert exc_info.value.code == 2
    assert "B2 requires --max-iterations >= 2" in capsys.readouterr().err


def test_multi_repetition_pilot_pairs_mcts_seeds_across_configurations(
    monkeypatch, tmp_path
):
    calls = []

    def fake_run(configuration, batch_dir, **kwargs):
        calls.append((configuration, kwargs))
        return {
            "configuration": configuration,
            "run_id": f"fixed-batch/{kwargs['run_name']}",
            "status": "COMPLETED",
        }

    monkeypatch.setattr(pilot, "_pilot_id", lambda: "fixed-batch")
    monkeypatch.setattr(pilot, "_run_one", fake_run)
    monkeypatch.setattr(
        pilot,
        "_git_metadata",
        lambda: {"commit": "abc", "worktree_dirty": False, "worktree_status": []},
    )
    monkeypatch.setattr(
        pilot,
        "_runtime_metadata",
        lambda: {
            "ready": True,
            "python_executable": "/test/python",
            "python_version": "3.12.0",
            "syside_available": True,
            "syside_version": "test",
            "syside_probe_part_count": 1,
            "error": None,
        },
    )

    result = main([
        "--repetitions", "3",
        "--mcts-seed-base", "40",
        "--output-root", str(tmp_path),
    ])

    assert result == 0
    assert len(calls) == 9
    assert [item[0] for item in calls[:3]] == ["B0", "B1", "B2"]
    assert [item[1]["mcts_seed"] for item in calls] == [
        40, 40, 40, 41, 41, 41, 42, 42, 42,
    ]
    assert [item[1]["generation_seed"] for item in calls] == [
        1000, 1000, 1000, 1001, 1001, 1001, 1002, 1002, 1002,
    ]
    assert [item[1]["run_name"] for item in calls[:3]] == [
        "b0-r01", "b1-r01", "b2-r01",
    ]
    batch = json.loads(
        (tmp_path / "fixed-batch" / "pilot_batch.json").read_text(
            encoding="utf-8"
        )
    )
    assert batch["planned_run_count"] == 9
    assert batch["completed_count"] == 9


def test_controlled_experiment_rejects_too_few_repetitions(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--experiment", "--repetitions", "2"])

    assert exc_info.value.code == 2
    assert "at least three repetitions" in capsys.readouterr().err


def test_controlled_experiment_rejects_a_dirty_worktree(monkeypatch, capsys):
    monkeypatch.setattr(
        pilot,
        "_git_metadata",
        lambda: {"commit": "abc", "worktree_dirty": True, "worktree_status": ["M x"]},
    )

    with pytest.raises(SystemExit) as exc_info:
        main(["--experiment", "--repetitions", "3"])

    assert exc_info.value.code == 2
    assert "clean committed worktree" in capsys.readouterr().err


def test_controlled_experiment_requires_reviewed_input_artifacts(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        pilot,
        "_git_metadata",
        lambda: {"commit": "abc", "worktree_dirty": False, "worktree_status": []},
    )
    monkeypatch.setattr(
        pilot, "_runtime_metadata", lambda: {"ready": True}
    )

    with pytest.raises(SystemExit) as exc_info:
        main(["--experiment", "--repetitions", "3"])

    assert exc_info.value.code == 2
    assert "--requirements-artifact and --gold" in capsys.readouterr().err


def test_reviewed_experiment_gate_binds_gold_to_exact_frozen_sources():
    frozen, gold = _reviewed_inputs()

    approval = pilot._validate_experiment_inputs(frozen, gold)

    assert approval["gate"] == (
        "APPROVED_FROZEN_REQUIREMENTS_AND_SOURCE_FIRST_GOLD"
    )
    assert approval["reviewed_contract_counts"] == {
        "READY": 10, "INCOMPLETE": 1, "UNSUPPORTED": 4,
    }
    assert approval["gold_used_as_pipeline_input"] is False


def test_reviewed_experiment_gate_rejects_source_drift():
    frozen, gold = _reviewed_inputs()
    gold["requirements"][0]["source_text"] += " changed"

    with pytest.raises(ValueError, match="differs from frozen input"):
        pilot._validate_experiment_inputs(frozen, gold)


def test_pilot_rejects_broken_syside_before_llm_calls(monkeypatch, capsys):
    monkeypatch.setattr(
        pilot,
        "_git_metadata",
        lambda: {"commit": "abc", "worktree_dirty": False, "worktree_status": []},
    )
    monkeypatch.setattr(
        pilot,
        "_runtime_metadata",
        lambda: {
            "ready": False,
            "python_executable": "/wrong/python",
            "syside_available": False,
            "syside_probe_part_count": 0,
            "error": "ModuleNotFoundError: No module named 'syside'",
        },
    )

    with pytest.raises(SystemExit) as exc_info:
        main(["--config", "B0"])

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "Syside runtime preflight failed before any LLM calls" in stderr
    assert "/wrong/python" in stderr
