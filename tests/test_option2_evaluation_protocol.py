from __future__ import annotations

import pytest

from src.prototyping.evaluation_protocol import build_descriptive_pilot_manifest


def test_three_repetitions_are_descriptive_not_confirmatory():
    manifest = build_descriptive_pilot_manifest(["run-1", "run-2", "run-3"])
    assert manifest["n"] == 3
    assert manifest["study_classification"] == "DESCRIPTIVE_PILOT"
    assert manifest["confirmatory_inference"] is False
    assert manifest["confirmatory_p_values_permitted"] is False


def test_pilot_manifest_rejects_mixed_namespace_and_wrong_n():
    with pytest.raises(ValueError, match="legacy"):
        build_descriptive_pilot_manifest(
            ["a", "b", "c"],
            experiment_namespace="LEGACY_EXTERNAL_CONTRACT_V1",
        )
    with pytest.raises(ValueError, match="exactly three"):
        build_descriptive_pilot_manifest(["a", "b"])
