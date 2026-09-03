"""Independently frozen architecture boundary (freeze-review P1).

Owner allocation is a design decision, not derivable from the requirement text,
so the component decomposition is frozen as its own provenanced artifact, cited
by digest from the evaluator gold.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.prototyping.ag_chains import (
    REQ_SAFE_004_CHAIN,
    REQ_SAFE_005_CHAIN,
    REQ_SAFE_008_CHAIN,
)
from src.prototyping.ag_gold_template import build_gold_draft
from src.prototyping.architecture_boundary import (
    architecture_boundary_digest,
    build_architecture_boundary_draft,
    validate_frozen_boundary,
)


def _draft() -> dict:
    return build_architecture_boundary_draft(
        REQ_SAFE_008_CHAIN, requirement_set_digest="a" * 64
    )


def _freeze(draft: dict) -> dict:
    def strip(obj):
        if isinstance(obj, dict):
            return {
                k: strip(v) for k, v in obj.items()
                if not (str(k).startswith("_") and "review" in str(k))
            }
        if isinstance(obj, list):
            return [strip(i) for i in obj]
        return obj

    boundary = strip(copy.deepcopy(draft))
    for comp in boundary["components"]:
        comp["responsibility"] = f"owns {comp['interfaces']['produces'][0]}"
    boundary["status"] = "FROZEN"
    boundary["reviewer"] = "Dr. Supervisor"
    boundary["reviewed_date"] = "2026-07-30"
    boundary["review_protocol"] = {"independent_architecture_review": True}
    boundary["artifact_digest"] = None
    boundary["artifact_digest"] = architecture_boundary_digest(boundary)
    return boundary


def test_draft_carries_provenance():
    draft = _draft()
    assert draft["artifact_role"] == "ARCHITECTURE_BOUNDARY"
    assert draft["schema_version"] == "1.0"
    assert draft["chain_id"] == "REQ_SAFE_008"
    assert draft["requirement_set_digest"] == "a" * 64
    assert {c["component_id"] for c in draft["components"]} == {
        "ReleaseCommandGatewayContract", "PayloadLockMechanismContract",
    }
    assert {(a["owner"], a["guarantee"]) for a in draft["allocations"]} == {
        ("releaseCommandGateway", "authorisedReleaseCommandReceived"),
        ("payloadLockMechanism", "payloadLocked"),
        ("payloadLockMechanism", "authorisedUnlockOnly"),
        ("payloadLockMechanism", "deenergiseToLock"),
    }
    releaser = next(
        c for c in draft["components"]
        if c["component_id"] == "ReleaseCommandGatewayContract"
    )
    assert releaser["interfaces"]["produces"] == [
        "authorisedReleaseCommandReceived"
    ]


@pytest.mark.parametrize(
    ("spec", "path"),
    [
        (
            REQ_SAFE_004_CHAIN,
            "docs/gold/REQ_SAFE_004_architecture_boundary.draft.json",
        ),
        (
            REQ_SAFE_005_CHAIN,
            "docs/gold/REQ_SAFE_005_architecture_boundary.draft.json",
        ),
        (
            REQ_SAFE_008_CHAIN,
            "docs/gold/REQ_SAFE_008_architecture_boundary.draft.json",
        ),
    ],
)
def test_committed_drafts_match_gold(spec, path):
    committed = json.loads(Path(path).read_text(encoding="utf-8"))
    assert committed == build_architecture_boundary_draft(spec)
    assert committed["status"] == "DRAFT_FOR_SUPERVISOR_REVIEW"
    assert committed["review_protocol"]["independent_architecture_review"] is False
    assert committed["artifact_digest"] is None


def test_safe005_power_continuous():
    draft = build_architecture_boundary_draft(REQ_SAFE_005_CHAIN)
    power = next(
        component for component in draft["components"]
        if component["component_id"] == "RecoveryPowerSupplyContract"
    )
    assert power["interfaces"]["consumes"] == ["airborne"]
    assert power["interfaces"]["produces"] == [
        "recoveryActuationPowerAvailable"
    ]
    assert "trigger" in power["interfaces"]
    assert power["interfaces"]["trigger"] is None


@pytest.mark.parametrize(
    "spec",
    [REQ_SAFE_004_CHAIN, REQ_SAFE_005_CHAIN, REQ_SAFE_008_CHAIN],
)
def test_gold_binds_boundary_digest(spec):
    boundary = _freeze(
        build_architecture_boundary_draft(
            spec, requirement_set_digest="a" * 64
        )
    )
    gold = build_gold_draft(
        spec,
        source_text=f"{spec.source_requirement}: test-only source",
        requirement_set_digest="a" * 64,
        architecture_boundary_digest=boundary["artifact_digest"],
    )
    assert gold["status"] == "DRAFT_FOR_SUPERVISOR_REVIEW"
    assert gold["architecture_boundary_digest"] == boundary["artifact_digest"]
    boundary_allocations = {
        (item["owner"], item["contract"], item["guarantee"])
        for item in boundary["allocations"]
    }
    gold_allocations = {
        (item["owner"], item["contract"], item["guarantee"])
        for item in gold["allocations"]
    }
    assert gold_allocations == boundary_allocations


def test_validator_flags_unfrozen_draft():
    problems = validate_frozen_boundary(_draft())
    assert any("FROZEN" in p for p in problems)
    assert any("responsibility" in p for p in problems)
    assert any("artifact_digest" in p for p in problems)


def test_validator_accepts_frozen():
    assert validate_frozen_boundary(_freeze(_draft())) == []


def test_digest_detects_tampering():
    frozen = _freeze(_draft())
    frozen["allocations"][0]["owner"] = "someoneElse"
    problems = validate_frozen_boundary(frozen)
    assert any("does not match the content" in p for p in problems)


def test_validator_requires_provenance():
    frozen = _freeze(_draft())
    frozen["schema_version"] = "unknown"
    frozen["requirement_set_digest"] = None
    frozen["components"][0]["interfaces"].pop("trigger")
    frozen["allocations"][0]["contract"] = "UnknownContract"
    frozen["artifact_digest"] = architecture_boundary_digest(frozen)

    problems = validate_frozen_boundary(frozen)
    assert any("schema_version" in p for p in problems)
    assert any("requirement_set_digest" in p for p in problems)
    assert any("interfaces.trigger" in p for p in problems)
    assert any("unknown component" in p for p in problems)
    assert any("exactly match" in p for p in problems)


def test_validator_rejects_duplicates():
    frozen = _freeze(_draft())
    frozen["components"].append(copy.deepcopy(frozen["components"][0]))
    frozen["allocations"].append(copy.deepcopy(frozen["allocations"][0]))
    frozen["artifact_digest"] = architecture_boundary_digest(frozen)

    problems = validate_frozen_boundary(frozen)
    assert any("duplicate component_id" in p for p in problems)
    assert any("duplicate allocation" in p for p in problems)


def test_no_runtime_checker_import():
    src = Path("src/prototyping/architecture_boundary.py").read_text(encoding="utf-8")
    imports = "\n".join(
        l for l in src.splitlines() if l.strip().startswith(("import ", "from "))
    )
    assert "ag_contracts" not in imports and "ag_extractor" not in imports
