from __future__ import annotations

import pytest

from src.prototyping.requirement_inputs import (
    build_frozen_requirement_set,
    resolve_frozen_requirement_set,
)


REQ = (
    "REQ-SAFE-005: Critical propulsion failure shall deploy the parachute "
    "within 0.5 seconds."
)


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
