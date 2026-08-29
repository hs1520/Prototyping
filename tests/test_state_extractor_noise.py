"""Expected candidate-probe misses must not flood the suppressed channel.

The extractor probes several syside attribute names per node and merges the
results; a name absent in the current syside build is the expected outcome of
that probing, not an anomaly.  Recording each miss buried real errors under
thousands of entries (ablation pilot: 5,265 + 324 in one run — 81 extractions
x the per-pass miss count).  This pins both halves of the fix: extraction
still works on the committed pilot model, and one full pass records nothing.
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


def test_extraction_works_without_flooding_the_suppressed_channel():
    reset_suppressed()
    machines = extract_state_machines(_PILOT_MODEL.read_text())

    # Function preserved: the pilot model's known machine and send counts.
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
