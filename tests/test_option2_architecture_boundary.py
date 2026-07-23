"""Independently frozen architecture boundary (freeze-review P1).

Owner allocation is a design decision, not derivable from the stakeholder
requirement text, so the component decomposition is frozen as its own provenanced
artifact that the evaluator gold cites by digest.
"""
from __future__ import annotations

import copy
from pathlib import Path

from src.prototyping.ag_chains import REQ_SAFE_008_CHAIN
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


def test_draft_carries_full_provenance_and_allocations_from_the_spec():
    draft = _draft()
    assert draft["artifact_role"] == "ARCHITECTURE_BOUNDARY"
    assert draft["schema_version"] == "1.0"
    assert draft["chain_id"] == "REQ_SAFE_008"
    assert draft["requirement_set_digest"] == "a" * 64
    # component ids + interfaces + owner->guarantee allocations are pre-filled
    assert {c["component_id"] for c in draft["components"]} == {
        "ReleaseAuthorityContract", "PayloadLockActuatorContract",
    }
    assert {(a["owner"], a["guarantee"]) for a in draft["allocations"]} == {
        ("releaseAuthority", "releaseAuthorised"),
        ("payloadLockActuator", "payloadReleased"),
    }
    releaser = next(
        c for c in draft["components"] if c["component_id"] == "ReleaseAuthorityContract"
    )
    assert releaser["interfaces"]["produces"] == ["releaseAuthorised"]


def test_validator_flags_an_unfrozen_draft():
    problems = validate_frozen_boundary(_draft())
    assert any("FROZEN" in p for p in problems)
    assert any("responsibility" in p for p in problems)
    assert any("artifact_digest" in p for p in problems)


def test_validator_accepts_a_completely_frozen_boundary():
    assert validate_frozen_boundary(_freeze(_draft())) == []


def test_digest_binding_detects_tampering_after_freeze():
    frozen = _freeze(_draft())
    # someone edits an allocation but keeps the old digest
    frozen["allocations"][0]["owner"] = "someoneElse"
    problems = validate_frozen_boundary(frozen)
    assert any("does not match the content" in p for p in problems)


def test_validator_requires_complete_schema_provenance_and_interface_binding():
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


def test_validator_rejects_duplicate_components_and_allocations():
    frozen = _freeze(_draft())
    frozen["components"].append(copy.deepcopy(frozen["components"][0]))
    frozen["allocations"].append(copy.deepcopy(frozen["allocations"][0]))
    frozen["artifact_digest"] = architecture_boundary_digest(frozen)

    problems = validate_frozen_boundary(frozen)
    assert any("duplicate component_id" in p for p in problems)
    assert any("duplicate allocation" in p for p in problems)


def test_module_never_imports_the_runtime_checker():
    src = Path("src/prototyping/architecture_boundary.py").read_text(encoding="utf-8")
    imports = "\n".join(
        l for l in src.splitlines() if l.strip().startswith(("import ", "from "))
    )
    assert "ag_contracts" not in imports and "ag_extractor" not in imports
