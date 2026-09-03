"""Expected candidate-probe misses stay out of the suppressed channel.

The extractor probes several syside attribute names per node and merges the
results, so an absent name is the expected outcome of probing. Recording each
miss buried real errors under thousands of entries (5,265 + 324 in one pilot
run). Pinned here: extraction still works on the committed pilot model, and
one full pass records nothing.
"""
from __future__ import annotations

from pathlib import Path

from src.simulation.state_extractor import extract_state_machines
from src.utils.suppressed import reset_suppressed, suppressed_summary

_PILOT_MODEL = (
    Path(__file__).resolve().parents[1]
    / "experiments/ablation/results/20260829_120859_pilot/runs"
    / "FULL_seed0.final.sysml"
)


def test_extraction_without_noise_flood():
    reset_suppressed()
    machines = extract_state_machines(_PILOT_MODEL.read_text())

    assert len(machines) == 13
    total_sends = sum(
        len(getattr(state, "sends", []) or [])
        for machine in machines
        for state in machine.states
    )
    assert total_sends == 2

    noise = {
        key: value["count"]
        for key, value in (suppressed_summary() or {}).items()
        if key.startswith("simulation.state_extractor.")
    }
    assert noise == {}, (
        f"candidate-probe misses leaked into the suppressed channel: {noise}"
    )
