from __future__ import annotations

import pytest

import json

import examples.run_option2_pilot as pilot
from examples.run_option2_pilot import main


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
