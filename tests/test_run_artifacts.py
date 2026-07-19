from __future__ import annotations

import json

import pytest

from src.prototyping.posthoc_evaluation import build_uniform_posthoc_evaluation
from src.prototyping.requirement_inputs import build_frozen_requirement_set
from src.prototyping.run_artifacts import load_archived_run


def test_archived_loader_binds_posthoc_and_pilot_metadata(tmp_path):
    requirement = "REQ-SAFE-005: Critical propulsion failure shall deploy a parachute."
    frozen = build_frozen_requirement_set([requirement], source="test")
    model = "package D {}"
    posthoc = build_uniform_posthoc_evaluation(
        model_text=model,
        model_name="D",
        frozen_requirements=frozen,
    )
    (tmp_path / "realization_run.json").write_text(
        json.dumps({"requirements": [requirement]}), encoding="utf-8"
    )
    (tmp_path / "final_model.sysml").write_text(model, encoding="utf-8")
    (tmp_path / "posthoc_evaluation.json").write_text(
        json.dumps(posthoc), encoding="utf-8"
    )
    (tmp_path / "pilot_metadata.json").write_text(
        json.dumps({
            "run_id": "pilot/B2/r01", "repetition": 1, "mcts_seed": 7,
            "generation_seed": 1007,
            "generation_seed_control": "PROVIDER_BEST_EFFORT",
        }),
        encoding="utf-8",
    )

    run = load_archived_run(tmp_path)

    assert run["posthoc_model_digest_verified"] is True
    assert run["posthoc_evaluation"]["model_digest"] == posthoc["model_digest"]
    assert run["run_id"] == "pilot/B2/r01"
    assert run["mcts_seed"] == 7
    assert run["generation_seed"] == 1007


def test_archived_loader_rejects_posthoc_for_a_different_model(tmp_path):
    requirement = "REQ-SAFE-005: Critical propulsion failure shall deploy a parachute."
    frozen = build_frozen_requirement_set([requirement], source="test")
    posthoc = build_uniform_posthoc_evaluation(
        model_text="package D {}",
        model_name="D",
        frozen_requirements=frozen,
    )
    (tmp_path / "realization_run.json").write_text("{}", encoding="utf-8")
    (tmp_path / "final_model.sysml").write_text(
        "package Different {}", encoding="utf-8"
    )
    (tmp_path / "posthoc_evaluation.json").write_text(
        json.dumps(posthoc), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="does not match final_model"):
        load_archived_run(tmp_path)


def test_archived_loader_rejects_a_mutating_posthoc_sidecar(tmp_path):
    requirement = "REQ-SAFE-005: Critical propulsion failure shall deploy a parachute."
    frozen = build_frozen_requirement_set([requirement], source="test")
    model = "package D {}"
    posthoc = build_uniform_posthoc_evaluation(
        model_text=model,
        model_name="D",
        frozen_requirements=frozen,
    )
    posthoc["mutation_permitted"] = True
    (tmp_path / "realization_run.json").write_text("{}", encoding="utf-8")
    (tmp_path / "final_model.sysml").write_text(model, encoding="utf-8")
    (tmp_path / "posthoc_evaluation.json").write_text(
        json.dumps(posthoc), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="not a read-only measurement"):
        load_archived_run(tmp_path)
