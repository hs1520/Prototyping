"""Dynamic (scenario-level) behavioural verdicts per part — roadmap ① remaining.

Structural reachability (safety_behavior) credits a guarded transition that *exists* and is
graph-reachable. This goes further: it runs the behavioural simulator, which DRIVES each
guard variable across its threshold and checks the guarded transition actually FIRES — so a
guard that is present but unsatisfiable / not grounded (dynamic "fake safety") is caught.

Returns {owner_part: 'fired' | 'failed'}:
  fired  — a guarded transition on that part's state machine fired when its guard was driven.
  failed — a scenario for that part's state machine did NOT pass (response never fires).
Parts without a guarded scenario are absent from the dict (caller falls back to structural).

Note: this verifies the response FIRES under its trigger; it does NOT verify metric timing
("within 1 s / 1 m") — the model carries no temporal semantics, so timing stays out of scope.
"""
from __future__ import annotations

from typing import Dict

from ..simulation.behavioral_sim import run_behavioral_simulation
from ..simulation.state_extractor import extract_state_machines


def dynamic_fire_by_part(model_text: str) -> Dict[str, str]:
    """{owner_part: 'fired'|'failed'} from executing the state machines (drive guards, check
    the guarded transition fires). 'failed' wins over 'fired' for the same part."""
    sm_owner = {sm.name: sm.owner_part for sm in extract_state_machines(model_text)}
    verdict: Dict[str, str] = {}
    try:
        res = run_behavioral_simulation(model_text)
    except Exception:
        return verdict
    for s in res.scenario_results:
        owner = sm_owner.get(s.state_machine)
        if not owner:
            continue
        if not s.passed:
            verdict[owner] = "failed"
        elif s.trigger_step is not None and verdict.get(owner) != "failed":
            verdict.setdefault(owner, "fired")
    return verdict
