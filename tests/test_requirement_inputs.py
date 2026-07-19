from __future__ import annotations

import pytest

from src.prototyping.requirement_contracts import build_contract_bundle
from src.prototyping.requirement_inputs import (
    APPROVED_CONTRACT_PROTOCOL_VERSION,
    build_approved_contract_bundle,
    build_frozen_requirement_set,
    requirement_set_digest,
    resolve_frozen_requirement_set,
    validate_approved_contract_bundle,
)


REQ = (
    "REQ-SAFE-005: Critical propulsion failure shall deploy the parachute "
    "within 0.5 seconds."
)


def _approved(requirements=(REQ,)):
    bundle = build_contract_bundle(requirements).to_dict()
    bundle["approval"] = {
        "protocol_version": APPROVED_CONTRACT_PROTOCOL_VERSION,
        "status": "APPROVED",
        "reviewer": "Supervisor",
        "approved_at": "2026-07-19T10:00:00Z",
        "requirement_set_digest": requirement_set_digest(requirements),
    }
    return bundle


def test_frozen_requirement_artifact_round_trips_exact_text_and_digest():
    artifact = build_frozen_requirement_set([REQ], name="option2")

    requirements, canonical = resolve_frozen_requirement_set(artifact)

    assert requirements == [REQ]
    assert canonical["requirement_set_digest"] == artifact["requirement_set_digest"]


def test_frozen_artifact_rejects_content_changed_after_digest():
    artifact = build_frozen_requirement_set([REQ])
    artifact["requirements"][0] += " changed"

    with pytest.raises(ValueError, match="digest"):
        resolve_frozen_requirement_set(artifact)


def test_approved_contract_requires_source_and_human_approval_match():
    bundle, provenance = validate_approved_contract_bundle(_approved(), [REQ])

    assert bundle.contracts[0].req_id == "REQ_SAFE_005"
    assert provenance["reviewer"] == "Supervisor"


def test_approved_contract_builder_attaches_auditable_metadata():
    approved = build_approved_contract_bundle(
        build_contract_bundle([REQ]),
        [REQ],
        reviewer="Supervisor",
        approved_at="2026-07-19T10:00:00Z",
    )

    assert approved["approval"]["status"] == "APPROVED"
    assert approved["approval"]["reviewer"] == "Supervisor"


def test_evaluation_gold_is_rejected_as_pipeline_input():
    with pytest.raises(ValueError, match="evaluation gold cannot be used"):
        validate_approved_contract_bundle(
            {"review_mode": "SOURCE_FIRST", "requirements": []}, [REQ]
        )


def test_approved_contract_cannot_target_a_different_frozen_set():
    other = "REQ-SAFE-006: The system shall lock the payload on abort."

    with pytest.raises(ValueError, match="different requirement set"):
        validate_approved_contract_bundle(_approved(), [other])
