from __future__ import annotations

import json

import pytest

from src.prototyping.experiment_arms import RevisedExperimentArm, revised_arm_metadata
from src.prototyping.revised_pilot import RevisedPilotConfig, run_revised_pilot


_REQS = (
    "REQ-SAFE-005: deploy parachute within 0.5 seconds.",
    "REQ-FUNC-002: maintain obstacle separation.",
)


def _config(**overrides):
    values = {
        "provider": "vertex",
        "model": "test-model",
        "seeds": (0, 1, 2),
        "max_iterations": 1,
        "code_revision": "abc123",
        "requirements": _REQS,
    }
    values.update(overrides)
    return RevisedPilotConfig(**values)


class _FakeLLM:
    def __init__(self, instance, seed):
        self.instance = instance
        self.seed = seed


class _FakePipeline:
    fail = None

    def __init__(
        self, *, llm, max_iterations, verbose, revised_experiment_arm, **_kwargs
    ):
        self.llm = llm
        self.max_iterations = max_iterations
        self.arm = revised_experiment_arm
        self.orchestrator = self

    def generate(self, *, system_name, system_description, frozen_requirements):
        if self.fail == (self.llm.seed, self.arm):
            raise RuntimeError("synthetic run failure")
        arm = RevisedExperimentArm.parse(self.arm)
        revised = revised_arm_metadata(arm)
        score = 0.8 + self.llm.seed / 100 + (
            ("R0-CURRENT", "R1-BBCTX", "R2-BBAG").index(self.arm) / 1000
        )
        result = {
            "system_name": system_name,
            "final_score": score,
            "requirements": list(frozen_requirements["requirements"]),
            "requirement_input": dict(frozen_requirements),
            "revised_experiment": revised,
            "collaboration": ({"synthetic": True} if arm.uses_blackboard else None),
            "llm_usage": {"calls": 1, "total_tokens": 100 + self.llm.seed},
        }
        if self.arm == "R2-BBAG":
            result["ag_contract_graph"] = {
                "checker_version": "ag-mvp-1",
                "verdict": "PASS",
            }
        return result

    @staticmethod
    def build_run_report(result):
        revised = result["revised_experiment"]
        return {
            "system_name": result["system_name"],
            "final_score": result["final_score"],
            "experiment_namespace": revised["experiment_namespace"],
            "configuration": revised["configuration"],
            "revised_experiment": revised,
            "llm_usage": result["llm_usage"],
        }


def test_config_freezes_exact_three_seed_three_arm_protocol():
    manifest = _config().to_manifest()
    assert manifest["experiment_namespace"] == "BLACKBOARD_AG_V1"
    assert manifest["arms"] == ["R0-CURRENT", "R1-BBCTX", "R2-BBAG"]
    assert manifest["gold_input_permitted"] is False
    assert manifest["langsmith_permitted"] is False
    assert manifest["gazebo_permitted"] is False
    assert len(manifest["configuration_digest"]) == 64

    with pytest.raises(ValueError, match="three distinct seeds"):
        _config(seeds=(0, 0, 1))
    with pytest.raises(ValueError, match="exact R0/R1/R2"):
        _config(arms=("R0-CURRENT", "R2-BBAG", "R1-BBCTX"))


def test_external_execution_requires_explicit_authorization(tmp_path):
    with pytest.raises(PermissionError, match="explicit current-run authorization"):
        run_revised_pilot(
            _config(),
            tmp_path / "unauthorised",
            external_execution_authorized=False,
            llm_factory=lambda **_kwargs: None,
            pipeline_factory=_FakePipeline,
            artifact_writer=lambda *_args: {},
        )
    assert not (tmp_path / "unauthorised").exists()


def test_complete_pilot_archives_nine_isolated_runs_and_descriptive_summary(tmp_path):
    factory_calls = []
    llm_instances = []
    artifact_calls = []

    def llm_factory(**kwargs):
        factory_calls.append(kwargs)
        llm = _FakeLLM(len(llm_instances) + 1, kwargs["provider_kwargs"]["seed"])
        llm_instances.append(llm)
        return llm

    def artifact_writer(_result, out):
        artifact_calls.append(out)
        marker = out / "synthetic_artifact.json"
        marker.write_text("{}\n", encoding="utf-8")
        return {"synthetic_artifact": str(marker)}

    out = tmp_path / "pilot"
    manifest = run_revised_pilot(
        _config(),
        out,
        external_execution_authorized=True,
        llm_factory=llm_factory,
        pipeline_factory=_FakePipeline,
        artifact_writer=artifact_writer,
    )

    assert manifest["status"] == "COMPLETE"
    assert manifest["run_count"] == 9
    assert manifest["completed_run_count"] == 9
    assert manifest["pooling_permitted"] is False
    assert manifest["protocol"]["study_classification"] == "DESCRIPTIVE_PILOT"
    assert manifest["descriptive_summary"]["status"] == "DESCRIPTIVE_ONLY"
    assert "p_values" not in manifest["descriptive_summary"]
    assert len({item.instance for item in llm_instances}) == 9
    assert [call["provider_kwargs"]["seed"] for call in factory_calls] == [
        0, 0, 0, 1, 1, 1, 2, 2, 2,
    ]
    assert all(
        call["provider_kwargs"]["enable_langsmith"] is False
        for call in factory_calls
    )
    assert len(artifact_calls) == 6
    assert len(list(out.glob("seed-*/*/run_manifest.json"))) == 9
    on_disk = json.loads((out / "pilot_manifest.json").read_text(encoding="utf-8"))
    assert on_disk["configuration_digest"] == manifest["configuration_digest"]
    r2 = [row for row in manifest["runs"] if row["configuration"] == "R2-BBAG"]
    assert r2 and all(row["evaluation_ready"] is False for row in r2)


def test_failed_run_makes_pilot_incomplete_and_existing_output_is_not_overwritten(
    tmp_path,
):
    def llm_factory(**kwargs):
        return _FakeLLM(1, kwargs["provider_kwargs"]["seed"])

    class FailingPipeline(_FakePipeline):
        fail = (1, "R1-BBCTX")

    out = tmp_path / "failed-pilot"
    manifest = run_revised_pilot(
        _config(),
        out,
        external_execution_authorized=True,
        llm_factory=llm_factory,
        pipeline_factory=FailingPipeline,
        artifact_writer=lambda *_args: {},
    )
    assert manifest["status"] == "INCOMPLETE"
    assert manifest["completed_run_count"] == 8
    assert manifest["descriptive_summary"]["status"] == "INCOMPLETE"
    failed = [row for row in manifest["runs"] if row["status"] == "FAILED"]
    assert len(failed) == 1
    assert failed[0]["error_type"] == "RuntimeError"

    with pytest.raises(FileExistsError, match="already exists"):
        run_revised_pilot(
            _config(),
            out,
            external_execution_authorized=True,
            llm_factory=llm_factory,
            pipeline_factory=FailingPipeline,
            artifact_writer=lambda *_args: {},
        )
