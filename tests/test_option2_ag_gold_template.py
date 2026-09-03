from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.prototyping.ag_chains import (
    REQ_SAFE_004_CHAIN,
    REQ_SAFE_005_CHAIN,
    REQ_SAFE_008_CHAIN,
)
from src.prototyping.ag_contracts import check_ag_graph
from src.prototyping.ag_evaluation import GOLD_ROLE, evaluate_ag_against_gold
from src.prototyping.ag_extractor import extract_ag_graph
from src.prototyping.ag_emitter import emit_ag_package
from src.prototyping.ag_gold_template import (
    GOLD_STATUS_DRAFT,
    build_gold_draft,
    validate_frozen_gold,
)
from src.simulation.syntax_checker import check_syntax


def _freeze(draft: dict) -> dict:
    import copy

    def strip(obj):
        if isinstance(obj, dict):
            return {
                k: strip(v) for k, v in obj.items()
                if not (str(k).startswith("_") and "review" in str(k))
            }
        if isinstance(obj, list):
            return [strip(i) for i in obj]
        return obj

    gold = strip(copy.deepcopy(draft))
    gold["status"] = "FROZEN"
    gold["reviewer"] = "Dr. Supervisor"
    gold["reviewed_date"] = "2026-07-30"
    gold["review_protocol"] = {
        "blind_to_runtime_verdict": True,
        "independent_human_review": True,
    }
    gold["requirement_set_digest"] = "a" * 64
    gold["architecture_boundary_digest"] = "b" * 64
    return gold


def test_flags_unfrozen_draft():
    problems = validate_frozen_gold(_draft())
    assert any("FROZEN" in p for p in problems)
    assert any("reviewer" in p for p in problems)
    assert any("_review markers" in p for p in problems)


def test_accepts_frozen_gold():
    assert validate_frozen_gold(_freeze(_draft())) == []


def test_flags_unresolved_discharge_edge():
    frozen = _freeze(_draft())
    frozen["discharge_edges"][0]["by"] = None
    problems = validate_frozen_gold(frozen)
    assert any("unresolved" in p for p in problems)


def test_rejects_static_failure_class():
    frozen = _freeze(_draft())
    frozen["requirement_set_digest"] = None
    frozen["failure_class"] = "NO_FAILURE"
    problems = validate_frozen_gold(frozen)
    assert any("requirement_set_digest" in p for p in problems)
    assert any("per-run blind label" in p for p in problems)


def test_rejects_priority_no_provenance():
    frozen = _freeze(_draft())
    frozen["priority"].pop("source_kind")
    frozen["priority"].pop("source_id")
    problems = validate_frozen_gold(frozen)
    assert any("priority response-set provenance" in p for p in problems)


def test_rejects_nested_labels_and_derived():
    frozen = _freeze(_draft())
    frozen["metadata"] = {"failure_class": "NO_FAILURE"}
    frozen["timing"]["metadata"] = {"within_deadline": True}
    problems = validate_frozen_gold(frozen)
    assert any("per-run blind label" in p for p in problems)
    assert any("evaluator-derived fields" in p for p in problems)


def test_rejects_binary_float_deadline():
    frozen = _freeze(_draft())
    frozen["system"]["deadline_s"] = 0.5
    problems = validate_frozen_gold(frozen)
    assert any("deadline_s must not appear" in problem for problem in problems)


def test_rejects_incomplete_facts():
    frozen = _freeze(_draft())
    frozen["realization_links"][0]["response_paths"][0]["source"] = ""
    frozen["observation_links"][0]["observation"] = ""
    problems = validate_frozen_gold(frozen)
    assert any("must set source/target/action" in problem for problem in problems)
    assert any(
        "must set contract/verification/observation" in problem
        for problem in problems
    )


def test_gold_schema_version():
    assert _draft()["schema_version"] == "3.0"


def test_pooling_gate_requires_frozen(tmp_path):
    from src.prototyping.ag_gold_template import frozen_gold_gate

    reqs = [
        "REQ-SAFE-004: prevent arming on self-test failure",
        "REQ-SAFE-005: deploy the parachute within 0.5 s",
    ]
    problems = frozen_gold_gate(reqs, gold_dir=str(tmp_path))
    assert any("REQ_SAFE_004" in p for p in problems)
    assert any("REQ_SAFE_005" in p for p in problems)

    import json
    from src.prototyping.ag_chains import REQ_SAFE_005_CHAIN
    frozen = _freeze(build_gold_draft(REQ_SAFE_005_CHAIN, source_text="REQ-SAFE-005: x"))
    (tmp_path / "REQ_SAFE_005_ag_gold.json").write_text(json.dumps(frozen))
    still = frozen_gold_gate(reqs, gold_dir=str(tmp_path))
    assert not any("REQ_SAFE_005" in p for p in still)
    assert any("REQ_SAFE_004" in p for p in still)

_SRC = (
    "REQ-SAFE-005: The system shall deploy the ballistic recovery parachute "
    "within 0.5 seconds of detecting a critical propulsion subsystem failure "
    "during flight, taking precedence over all other safety responses."
)
_DRAFT_FILE = Path("docs/gold/REQ_SAFE_005_ag_gold.draft.json")


def _draft() -> dict:
    return build_gold_draft(REQ_SAFE_005_CHAIN, source_text=_SRC)


def test_draft_evaluator_only_unfrozen():
    draft = _draft()
    assert draft["artifact_role"] == GOLD_ROLE == "EVALUATOR_GOLD"
    assert draft["status"] == GOLD_STATUS_DRAFT
    assert draft["reviewer"] is None and draft["reviewed_date"] is None
    assert draft["chain_id"] == "REQ_SAFE_005"


def test_draft_cites_source_digest():
    assert _draft()["source_digest"].startswith("d98469c950cc8d65")


def test_draft_cannot_be_scored():
    prediction = check_ag_graph(
        extract_ag_graph(
            "package Source { requirement def REQ_SAFE_005; }\n"
            + emit_ag_package(REQ_SAFE_005_CHAIN),
            revision=1,
        )
    ).to_dict()
    with pytest.raises(ValueError, match="FROZEN gold"):
        evaluate_ag_against_gold(prediction, _draft())


def test_no_runtime_checker_import():
    source = Path("src/prototyping/ag_gold_template.py").read_text(encoding="utf-8")
    imports = "\n".join(
        line for line in source.splitlines()
        if line.strip().startswith(("import ", "from "))
    )
    assert "ag_extractor" not in imports
    assert "ag_contracts" not in imports


def test_committed_draft_matches_generator():
    on_disk = json.loads(_DRAFT_FILE.read_text(encoding="utf-8"))
    assert on_disk["status"] == GOLD_STATUS_DRAFT
    assert on_disk["artifact_role"] == "EVALUATOR_GOLD"
    assert on_disk == _draft()


@pytest.mark.parametrize(
    ("chain", "expected_category"),
    [
        (REQ_SAFE_004_CHAIN, "invariant_agreement"),
        (REQ_SAFE_005_CHAIN, "timing_agreement"),
        (REQ_SAFE_008_CHAIN, "invariant_agreement"),
    ],
)
def test_chain_prediction_round_trips(
    chain, expected_category
):
    path = Path(f"docs/gold/{chain.source_requirement}_ag_gold.draft.json")
    draft = json.loads(path.read_text(encoding="utf-8"))
    assert draft == build_gold_draft(chain, source_text=draft["source_text"])
    model = (
        f"package Source {{ requirement def {chain.source_requirement} "
        "{ doc /* source */ } }\n"
        + emit_ag_package(chain)
    )
    syntax = check_syntax(
        model,
        fail_closed=True,
        filter_stdlib_diagnostics=False,
    )
    assert syntax.has_errors is False
    assert syntax.score == 1.0
    report = check_ag_graph(extract_ag_graph(model, revision=1))
    assert report.verdict == "PASS", [item.code for item in report.diagnostics]
    prediction = report.to_dict()
    result = evaluate_ag_against_gold(prediction, _freeze(draft))
    assert prediction["artifact_role"] == "RUNTIME_A_G_PREDICTION"
    assert result["metric_name"] == "decomposition_extraction_agreement"
    assert "f1" not in result
    assert result["guarantee_allocation"]["f1"] == 1.0
    assert result["assumption_discharge"]["f1"] == 1.0
    assert result["realization_link_agreement"]["f1"] == 1.0
    assert result["observation_link_agreement"]["f1"] == 1.0
    assert expected_category in result
    if expected_category == "invariant_agreement":
        category = result[expected_category]
        assert category["stakeholder"]["f1"] == 1.0
        assert category["student_derived_design_constraint"]["f1"] == 1.0
    else:
        assert result["timing_agreement"]["segment_prf"]["f1"] == 1.0
        assert result["priority_agreement"]["edge_prf"]["f1"] == 1.0
        assert result["priority_agreement"]["arbitration_topology_conforms"] is True


def test_three_chains_cover_five_categories():
    observed = {
        "guarantee_allocation",
        "assumption_discharge",
    }
    for chain in (
        REQ_SAFE_004_CHAIN,
        REQ_SAFE_005_CHAIN,
        REQ_SAFE_008_CHAIN,
    ):
        path = Path(f"docs/gold/{chain.source_requirement}_ag_gold.draft.json")
        draft = json.loads(path.read_text(encoding="utf-8"))
        model = (
            f"package Source {{ requirement def {chain.source_requirement} "
            "{ doc /* source */ } }\n"
            + emit_ag_package(chain)
        )
        prediction = check_ag_graph(extract_ag_graph(model, revision=1)).to_dict()
        result = evaluate_ag_against_gold(prediction, _freeze(draft))
        observed.update(
            key for key in (
                "timing_agreement",
                "priority_agreement",
                "invariant_agreement",
                "realization_link_agreement",
                "observation_link_agreement",
            )
            if key in result
        )
    assert observed == {
        "guarantee_allocation",
        "assumption_discharge",
        "timing_agreement",
        "priority_agreement",
        "invariant_agreement",
        "realization_link_agreement",
        "observation_link_agreement",
    }


def test_path_and_observation_penalised():
    model = (
        "package Source { requirement def REQ_SAFE_005 { doc /* source */ } }\n"
        + emit_ag_package(REQ_SAFE_005_CHAIN)
    )
    prediction = check_ag_graph(extract_ag_graph(model, revision=1)).to_dict()
    frozen = _freeze(_draft())
    prediction["graph"]["realization_links"][0]["response_paths"] = []
    prediction["graph"]["observation_links"] = []
    result = evaluate_ag_against_gold(prediction, frozen)
    assert result["realization_link_agreement"]["recall"] < 1.0
    assert result["realization_link_agreement"]["fn"] == 1
    assert result["observation_link_agreement"]["recall"] == 0.0
    assert result["observation_link_agreement"]["fn"] == 1
    assert "f1" not in result


def test_malformed_fact_fails_closed():
    model = (
        "package Source { requirement def REQ_SAFE_005 { doc /* source */ } }\n"
        + emit_ag_package(REQ_SAFE_005_CHAIN)
    )
    prediction = check_ag_graph(extract_ag_graph(model, revision=1)).to_dict()
    prediction["graph"]["realization_links"][0].pop("continuous_guarantee")
    with pytest.raises(ValueError, match="continuous_guarantee"):
        evaluate_ag_against_gold(prediction, _freeze(_draft()))


def test_draft_discharge_edges_resolved():
    assert all(e["by"] is not None for e in _draft()["discharge_edges"])
