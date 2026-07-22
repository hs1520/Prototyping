"""Evaluator-only A/G gold template — draft generation and consistency.

The draft is derived from the reviewed decomposition (not the checker, F3), carries
the EVALUATOR_GOLD role and DRAFT status, and is internally consistent with the
emit → extract → check pipeline (F1=1.0), which is what the supervisor reviews and
freezes.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.prototyping.ag_chains import REQ_SAFE_005_CHAIN
from src.prototyping.ag_contracts import check_ag_graph
from src.prototyping.ag_evaluation import GOLD_ROLE, evaluate_ag_against_gold
from src.prototyping.ag_extractor import extract_ag_graph
from src.prototyping.ag_emitter import emit_ag_package
from src.prototyping.ag_gold_template import (
    GOLD_STATUS_DRAFT,
    build_gold_draft,
)

_SRC = (
    "REQ-SAFE-005: The system shall deploy the ballistic recovery parachute "
    "within 0.5 seconds of detecting a critical propulsion subsystem failure "
    "during flight, taking precedence over all other safety responses."
)
_DRAFT_FILE = Path("docs/gold/REQ_SAFE_005_ag_gold.draft.json")


def _draft() -> dict:
    return build_gold_draft(REQ_SAFE_005_CHAIN, source_text=_SRC)


def test_draft_is_evaluator_only_and_unfrozen():
    draft = _draft()
    assert draft["artifact_role"] == GOLD_ROLE == "EVALUATOR_GOLD"
    assert draft["status"] == GOLD_STATUS_DRAFT
    assert draft["reviewer"] is None and draft["reviewed_date"] is None
    assert draft["chain_id"] == "REQ_SAFE_005"


def test_draft_cites_the_canonical_source_digest():
    assert _draft()["source_digest"].startswith("d98469c950cc8d65")


def test_draft_is_consistent_with_the_pipeline_prediction():
    # emit the reviewed chain, run the checker, and score its prediction against the
    # independently-built draft — a faithful pipeline agrees perfectly.
    prediction = check_ag_graph(
        extract_ag_graph(emit_ag_package(REQ_SAFE_005_CHAIN), revision=1)
    ).to_dict()
    result = evaluate_ag_against_gold(prediction, _draft())
    assert result["guarantee_allocation"]["f1"] == 1.0
    assert result["assumption_discharge"]["f1"] == 1.0
    assert result["assumption_discharge"]["tp"] == 5
    # the checker prediction carries no failure_class, so the evaluator omits it
    assert "failure_class_match" not in result


def test_generator_never_imports_the_runtime_checker():
    source = Path("src/prototyping/ag_gold_template.py").read_text(encoding="utf-8")
    imports = "\n".join(
        line for line in source.splitlines()
        if line.strip().startswith(("import ", "from "))
    )
    assert "ag_extractor" not in imports
    assert "ag_contracts" not in imports


def test_committed_draft_file_matches_the_generator_and_is_unfrozen():
    on_disk = json.loads(_DRAFT_FILE.read_text(encoding="utf-8"))
    assert on_disk["status"] == GOLD_STATUS_DRAFT
    assert on_disk["artifact_role"] == "EVALUATOR_GOLD"
    # regenerable: the committed draft equals a fresh generation
    assert on_disk == _draft()


def test_draft_discharge_edges_have_no_unresolved_sources_for_this_chain():
    # REQ_SAFE_005 is fully specified: every edge resolves to environment or an
    # upstream component (no reviewer-blocking None).
    assert all(e["by"] is not None for e in _draft()["discharge_edges"])
