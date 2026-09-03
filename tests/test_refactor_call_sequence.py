"""Golden: what each arm asks the provider during ``generate()``, and in what order.

Every prompt digest here moves when any prompt those calls assemble is edited,
so an intentional prompt change regenerates this golden in the same commit and
the commit states which digests moved and why. Running the module as a script
rewrites the golden; diff it before committing. The replay responses are
archived seed-0 responses from ``pilot_v17_20260801``, which let the real
``Orchestrator.generate()`` run offline. Two qualifications: R0-CURRENT archives
no replayable transcript, so its code path is driven by R1-BBCTX's answers and
its entry is "what R0 would ask had it been given R1's answers" - stable and
drift-sensitive, but not evidence about a real R0 run; and the pilot is pinned
at v17 while the pipeline is at v20, so pruning that directory would take this
test with it. ``VerificationAgent`` is outside the replay - it is not called
within ``generate()``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.agents.orchestrator import Orchestrator
from src.llm.interface import LLMInterface, LLMResponse, Message
from src.prototyping.experiment_arms import R2_LLM_DECIDED_GENERATION_MODE


_ROOT = Path(__file__).resolve().parents[1]
_PILOT = _ROOT / "examples/output/pilot_v17_20260801"
_GOLDEN = Path(__file__).with_name("golden_refactor_call_sequence.json")
_ARMS = ("R0-CURRENT", "R1-BBCTX", "R2-BBAG")


def _digest(parts: list[str]) -> str:
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _assistant_responses(arm: str) -> list[str]:
    # R0-CURRENT has no transcript of its own to replay; see the module
    # docstring for why, and for what that costs the R0 entry.
    transcript_arm = "R1-BBCTX" if arm == "R0-CURRENT" else arm
    path = _PILOT / "seed-0" / transcript_arm / "session_transcripts.jsonl"
    sessions = [json.loads(line) for line in path.read_text().splitlines()]
    roles = (
        ("DesignAgent",)
        if arm != "R2-BBAG"
        else ("AGPlanningAgent", "DesignAgent")
    )
    return [
        message["content"]
        for role in roles
        for session in sessions
        if session["agent_role"] == role
        for message in session["messages"]
        if message["role"] == "assistant"
    ]


class _RecordingReplayLLM(LLMInterface):
    RETRY_DELAYS = ()

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._issued = 0
        self.calls: list[dict] = []

    def _next_response(self) -> str:
        """Fail as a call-count change rather than as `StopIteration`.

        An arm asking for one more call than the archive holds is what this golden is
        for, and a bare `StopIteration` raised deep inside the orchestrator hides that.
        """
        if self._issued >= len(self._responses):
            raise AssertionError(
                f"arm asked for provider call {self._issued + 1} but the "
                f"archived replay holds {len(self._responses)}: the call "
                "sequence grew, so the golden and the recorded call count "
                "both need regenerating"
            )
        response = self._responses[self._issued]
        self._issued += 1
        return response

    def _complete_impl(
        self,
        messages: list[Message],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        self.calls.append({
            "ordinal": len(self.calls) + 1,
            "system_prompt_digest": _digest([
                message.content for message in messages
                if message.role == "system"
            ]),
            "user_prompt_digest": _digest([
                message.content for message in messages
                if message.role == "user"
            ]),
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        return LLMResponse(content=self._next_response(), model="replay")


def _frozen_requirements() -> dict:
    config = json.loads((_PILOT / "pilot_config.json").read_text())
    return config["frozen_requirement_set"]


def _record_arm(arm: str) -> list[dict]:
    llm = _RecordingReplayLLM(_assistant_responses(arm))
    kwargs = {
        "llm": llm,
        "revised_experiment_arm": arm,
        "max_iterations": 1,
        "quality_threshold": 0.75,
        "maximum_plan_attempts": 6,
    }
    if arm == "R2-BBAG":
        kwargs["r2_generation_mode"] = R2_LLM_DECIDED_GENERATION_MODE
    orchestrator = Orchestrator(**kwargs)
    orchestrator.generate(
        system_name="DeliveryUAV",
        system_description=(
            "An autonomous delivery UAV with ballistic parachute recovery and "
            "forward obstacle avoidance."
        ),
        frozen_requirements=_frozen_requirements(),
    )
    return llm.calls


def _run_with_agenda(arm: str) -> dict:
    llm = _RecordingReplayLLM(_assistant_responses(arm))
    kwargs = {
        "llm": llm,
        "revised_experiment_arm": arm,
        "max_iterations": 1,
        "quality_threshold": 0.75,
        "maximum_plan_attempts": 6,
    }
    if arm == "R2-BBAG":
        kwargs["r2_generation_mode"] = R2_LLM_DECIDED_GENERATION_MODE
    return Orchestrator(**kwargs).generate(
        system_name="DeliveryUAV",
        system_description=(
            "An autonomous delivery UAV with ballistic parachute recovery and "
            "forward obstacle avoidance."
        ),
        frozen_requirements=_frozen_requirements(),
    )


def _record_all() -> dict[str, list[dict]]:
    return {arm: _record_arm(arm) for arm in _ARMS}


def test_call_sequence_matches_golden():
    assert _record_all() == json.loads(_GOLDEN.read_text())


def test_agenda_activates_every_phase():
    result = _run_with_agenda("R2-BBAG")
    agenda = result["control_agenda"]
    registered = [item["name"] for item in agenda["registered_knowledge_sources"]]
    activated = [item["knowledge_source"] for item in agenda["activations"]]
    assert len(activated) == 21
    assert registered == list(reversed(activated))
    assert activated.index("pre_ag_simulation") < activated.index(
        "ag_contract_reconciliation"
    )


def test_exhausted_replay_reports_growth():
    llm = _RecordingReplayLLM(["only one archived answer"])
    messages = [Message(role="user", content="plan")]
    llm._complete_impl(messages, temperature=0.2, max_tokens=16)

    try:
        llm._complete_impl(messages, temperature=0.2, max_tokens=16)
    except AssertionError as exc:
        assert "call sequence grew" in str(exc)
        assert "provider call 2" in str(exc)
    else:
        raise AssertionError("exhausted replay did not report the growth")


if __name__ == "__main__":
    _GOLDEN.write_text(
        json.dumps(_record_all(), indent=2, sort_keys=True) + "\n"
    )
