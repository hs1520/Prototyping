"""The transition name is optional in SysML v2 (`transition (name first)? ...`).

Five readers required one, so unnamed transitions were misread: guards escaped
the late-response check, guarded transitions went uncounted, the repair gate
minted a `first` identity token, and a degraded parse lost accept triggers.
Each case pins the unnamed spelling against the named one.
"""
from __future__ import annotations


_NAMED = """
state def M {
    entry; then idle;
    state idle;
    state done { entry action fire : doFire; }
    transition tGo first idle accept GoSignal if armed then done;
}
"""
_UNNAMED = _NAMED.replace("transition tGo first", "transition first")


def test_model_facts_sees_unnamed():
    from src.dse.model_facts import _redundancy_facts
    wrap = "part def SafetyMonitor {{ {sm} }}"
    named = _redundancy_facts(wrap.format(sm=_NAMED))
    unnamed = _redundancy_facts(wrap.format(sm=_UNNAMED))
    assert unnamed.guard_names == named.guard_names
    assert named.guard_names, "the guard must be seen at all"


def test_evaluator_sees_unnamed():
    from src.dse.evaluator import _GUARDED_TRANSITION
    assert len(_GUARDED_TRANSITION.findall(_NAMED)) == 1
    assert len(_GUARDED_TRANSITION.findall(_UNNAMED)) == 1


def test_repair_tokens_no_first():
    from src.prototyping.ag_repair import _behavior_tokens
    wrap = "package P {{ part def C {{ {sm} }} }}"
    named = _behavior_tokens(wrap.format(sm=_NAMED), "M")
    unnamed = _behavior_tokens(wrap.format(sm=_UNNAMED), "M")
    assert ("transition", "first") not in unnamed
    assert named - unnamed == {("transition", "tGo")}


def test_guard_regex_matches_unnamed():
    from src.prototyping.requirement_semantics import _TRANSITION_GUARD_RE
    block = """
    transition first monitoring if separation < minSeparation then avoidance;
    """
    matches = list(_TRANSITION_GUARD_RE.finditer(block))
    assert len(matches) == 1
    assert matches[0].group("name") is None
    assert matches[0].group("target") == "avoidance"


def test_state_extractor_unnamed(monkeypatch):
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
