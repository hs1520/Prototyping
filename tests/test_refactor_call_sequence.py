"""Characterize the provider-call sequence before the controller refactor.

The replay responses are archived seed-0 pilot responses.  They let the real
``Orchestrator.generate()`` pipeline run offline while this test records exactly
what each arm asks the provider, and in what order.
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
        self._responses = iter(responses)
        self.calls: list[dict] = []

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
        return LLMResponse(content=next(self._responses), model="replay")


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


def _record_all() -> dict[str, list[dict]]:
    return {arm: _record_arm(arm) for arm in _ARMS}


def test_refactor_provider_call_sequence_matches_golden():
    assert _record_all() == json.loads(_GOLDEN.read_text())


if __name__ == "__main__":
    _GOLDEN.write_text(
        json.dumps(_record_all(), indent=2, sort_keys=True) + "\n"
    )
