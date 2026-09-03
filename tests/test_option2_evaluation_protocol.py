from __future__ import annotations

import pytest

from src.prototyping.evaluation_protocol import build_descriptive_pilot_manifest


def test_three_reps_descriptive():
    manifest = build_descriptive_pilot_manifest(["run-1", "run-2", "run-3"])
    assert manifest["n"] == 3
    assert manifest["study_classification"] == "DESCRIPTIVE_PILOT"
    assert manifest["confirmatory_inference"] is False
    assert manifest["confirmatory_p_values_permitted"] is False


def test_rejects_legacy_namespace_low_n():
    with pytest.raises(ValueError, match="legacy"):
        build_descriptive_pilot_manifest(
            ["a", "b", "c"],
            experiment_namespace="LEGACY_EXTERNAL_CONTRACT_V1",
        )
    with pytest.raises(ValueError, match="at least three"):
        build_descriptive_pilot_manifest(["a", "b"])
    with pytest.raises(ValueError, match="at least three"):
        build_descriptive_pilot_manifest(["a", "b", "b"])


def test_six_reps_stay_descriptive():
    manifest = build_descriptive_pilot_manifest(["a", "b", "c", "d", "e", "f"])

    assert manifest["n"] == 6
    assert manifest["study_classification"] == "DESCRIPTIVE_PILOT"
    assert manifest["confirmatory_inference"] is False
    assert manifest["confirmatory_p_values_permitted"] is False
