"""Evaluator-only A/G gold template — draft generation and consistency.

The draft is derived from the reviewed decomposition (not the checker, F3), carries
the EVALUATOR_GOLD role and DRAFT status, and is internally consistent with the
emit → extract → check pipeline (F1=1.0), which is what the supervisor reviews and
freezes.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.prototyping.ag_chains import REQ_SAFE_005_CHAIN
from src.prototyping.ag_contracts import check_ag_graph
from src.prototyping.ag_evaluation import GOLD_ROLE, evaluate_ag_against_gold
from src.prototyping.ag_extractor import extract_ag_graph
from src.prototyping.ag_emitter import emit_ag_package
from src.prototyping.ag_gold_template import (
    GOLD_STATUS_DRAFT,
    build_gold_draft,
    validate_frozen_gold,
)


def _freeze(draft: dict) -> dict:
    """Mimic the supervisor's manual freeze: drop _review markers, set metadata."""
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
    return gold


def test_validator_flags_an_unfrozen_draft():
    problems = validate_frozen_gold(_draft())
    # a draft trips the status, reviewer, date, both flags, and leftover markers
    assert any("FROZEN" in p for p in problems)
    assert any("reviewer" in p for p in problems)
    assert any("_review markers" in p for p in problems)


def test_validator_accepts_a_completely_frozen_gold():
    assert validate_frozen_gold(_freeze(_draft())) == []


def test_validator_flags_an_unresolved_discharge_edge():
    frozen = _freeze(_draft())
    frozen["discharge_edges"][0]["by"] = None  # reviewer left one unresolved
    problems = validate_frozen_gold(frozen)
    assert any("unresolved" in p for p in problems)


def test_pooling_gate_requires_every_selected_chain_frozen(tmp_path):
    from src.prototyping.ag_gold_template import frozen_gold_gate

    reqs = [
        "REQ-SAFE-004: prevent arming on self-test failure",
        "REQ-SAFE-005: deploy the parachute within 0.5 s",
    ]
    # no frozen files yet -> the gate blocks and names both chains
    problems = frozen_gold_gate(reqs, gold_dir=str(tmp_path))
    assert any("REQ_SAFE_004" in p for p in problems)
    assert any("REQ_SAFE_005" in p for p in problems)

    # freeze only ONE chain -> the gate still blocks on the other (P2: all chains)
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


def test_draft_is_evaluator_only_and_unfrozen():
    draft = _draft()
    assert draft["artifact_role"] == GOLD_ROLE == "EVALUATOR_GOLD"
    assert draft["status"] == GOLD_STATUS_DRAFT
    assert draft["reviewer"] is None and draft["reviewed_date"] is None
    assert draft["chain_id"] == "REQ_SAFE_005"


def test_draft_cites_the_canonical_source_digest():
    assert _draft()["source_digest"].startswith("d98469c950cc8d65")


def test_draft_cannot_be_scored_before_independent_blind_freeze():
    prediction = check_ag_graph(
        extract_ag_graph(
            "package Source { requirement def REQ_SAFE_005; }\n"
            + emit_ag_package(REQ_SAFE_005_CHAIN),
            revision=1,
        )
    ).to_dict()
    with pytest.raises(ValueError, match="FROZEN gold"):
        evaluate_ag_against_gold(prediction, _draft())


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
