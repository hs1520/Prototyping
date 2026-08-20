"""The transition name is optional in SysML v2 (`transition (name first)? ...`).

Five readers required one, so a model that spelled its transitions without
names was silently misread: guards escaped the late-response check, guarded
transitions were not counted, the repair gate minted a bogus `first` identity
token, and a degraded parse lost accept triggers.  Each case here pins the
unnamed spelling against the named one.
"""
from __future__ import annotations

import re

_NAMED = """
state def M {
    entry; then idle;
    state idle;
    state done { entry action fire : doFire; }
    transition tGo first idle accept GoSignal if armed then done;
}
"""
_UNNAMED = _NAMED.replace("transition tGo first", "transition first")


def test_model_facts_voting_pattern_sees_unnamed_guarded_transitions():
    from src.dse.model_facts import _redundancy_facts
    wrap = "part def SafetyMonitor {{ {sm} }}"
    named = _redundancy_facts(wrap.format(sm=_NAMED))
    unnamed = _redundancy_facts(wrap.format(sm=_UNNAMED))
    assert unnamed.guard_names == named.guard_names
    assert named.guard_names, "the guard must be seen at all"


def test_evaluator_guarded_transition_sees_unnamed():
    from src.dse.evaluator import _GUARDED_TRANSITION
    assert len(_GUARDED_TRANSITION.findall(_NAMED)) == 1
    assert len(_GUARDED_TRANSITION.findall(_UNNAMED)) == 1


def test_ag_repair_tokens_mint_no_bogus_first_token():
    from src.prototyping.ag_repair import _behavior_tokens
    wrap = "package P {{ part def C {{ {sm} }} }}"
    named = _behavior_tokens(wrap.format(sm=_NAMED), "M")
    unnamed = _behavior_tokens(wrap.format(sm=_UNNAMED), "M")
    assert ("transition", "first") not in unnamed
    # the only difference between the two spellings is the name token itself
    assert named - unnamed == {("transition", "tGo")}


def test_requirement_semantics_guard_re_matches_unnamed():
    from src.prototyping.requirement_semantics import _TRANSITION_GUARD_RE
    block = """
    transition first monitoring if separation < minSeparation then avoidance;
    """
    matches = list(_TRANSITION_GUARD_RE.finditer(block))
    assert len(matches) == 1
    assert matches[0].group("name") is None
    assert matches[0].group("target") == "avoidance"


def test_state_extractor_accept_fallback_covers_unnamed(monkeypatch):
    from src.simulation import state_extractor as se
    machines = se.extract_state_machines(
        "package P {\n"
        "    item def GoSignal;\n"
        "    part def C {\n"
        "        attribute armed : Boolean;\n"
        "        action def doFire { }\n"
        + _UNNAMED
        + "    }\n"
        "}\n"
    )
    assert machines, "state machine must extract"
    accepts = [t.accept_trigger for m in machines for t in m.transitions
               if t.accept_trigger]
    assert accepts == ["GoSignal"]
