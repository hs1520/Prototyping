"""How much of the reviewed decomposition the frozen boundary already determines.

Both LLM modes score guarantee-allocation F1 = 1.0, and the decided mode also scores
assumption-discharge F1 = 1.0. Neither is evidence of generation capability: the
boundary hands over each component's owner and its consumes/produces interfaces, and
a trivial rule over those reproduces almost all of the reviewed wiring.

This is a standing guard, not a one-off measurement. If someone later reports those
categories as generation accuracy, or changes the boundary so that it supplies more
or less than it does today, the numbers asserted here move and the test says so.
See `docs/R2_GENERATION_FINDINGS.md`.
"""
from __future__ import annotations

import pytest

from src.prototyping import ag_chains
from src.prototyping.architecture_boundary import build_architecture_boundary_draft

_CHAINS = {
    "REQ_SAFE_004": ag_chains.REQ_SAFE_004_CHAIN,
    "REQ_SAFE_005": ag_chains.REQ_SAFE_005_CHAIN,
    "REQ_SAFE_008": ag_chains.REQ_SAFE_008_CHAIN,
}


def _determined_slots(chain) -> tuple[int, int]:
    """(slots the trivial boundary rule gets right, total reviewed slots)."""
    boundary = build_architecture_boundary_draft(chain)
    produced = {
        str(concept): str(item["component_id"])
        for item in boundary["components"]
        for concept in item["interfaces"]["produces"]
    }
    right = total = 0
    for component in chain.components:
        for assumption in component.assumptions:
            total += 1
            source = produced.get(assumption.concept)
            trivial = None if source in (None, component.name) else source
            reviewed = None if assumption.environment else source
            right += trivial == reviewed
    return right, total


@pytest.mark.parametrize(
    "chain_id,expected_right,expected_total",
    [("REQ_SAFE_004", 4, 4), ("REQ_SAFE_005", 5, 5), ("REQ_SAFE_008", 2, 2)],
)
def test_boundary_determines_the_discharge_wiring(
    chain_id, expected_right, expected_total
):
    right, total = _determined_slots(_CHAINS[chain_id])
    assert (right, total) == (expected_right, expected_total), (
        f"{chain_id}: the boundary now determines {right}/{total} assumption "
        "slots, not what docs/R2_GENERATION_FINDINGS.md records. Discharge "
        "agreement is only a capability claim for the slots it does NOT determine."
    )


def test_every_reviewed_slot_is_boundary_determined():
    """The headline: every reviewed assumption slot, on every chain, is determined.

    Counting only reviewed assumptions is the correct denominator — an
    `interface_input` is not a discharge obligation and including it both inflates
    the total and invents disagreements that no gold edge corresponds to.
    """
    right = sum(_determined_slots(chain)[0] for chain in _CHAINS.values())
    total = sum(_determined_slots(chain)[1] for chain in _CHAINS.values())
    assert (right, total) == (11, 11), (
        "discharge agreement carries no capability signal at all while this holds"
    )


def test_ownership_is_handed_over_outright():
    """Allocation F1 cannot be a capability claim at all: the owner of every
    component is stated in the boundary the generator is given."""
    for chain_id, chain in _CHAINS.items():
        boundary = build_architecture_boundary_draft(chain)
        for item in boundary["components"]:
            assert item["owner_def"], chain_id
            assert item["owner_usage"], chain_id
