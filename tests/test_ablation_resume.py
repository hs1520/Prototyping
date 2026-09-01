"""Prefix-replay resume for ablation runs.

s0v12 hung on one provider request 34 minutes in; 187k tokens of identical
prefix work had to be re-bought because the pipeline holds no checkpoint.
Every run now captures (request digest, response) per call, and a resumed
run replays the matching prefix free, going live at the first divergence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/ablation"))

from run_ablation import PrefixReplayLLM, _capture_observer, _request_digest  # noqa: E402
from src.llm.interface import LLMInterface, LLMResponse, Message  # noqa: E402


class _CountingLLM(LLMInterface):
    def __init__(self):
        self.live_calls = 0

    def _complete_impl(self, messages, temperature, max_tokens):
        self.live_calls += 1
        return LLMResponse(content=f"live-{self.live_calls}", model="fake")


def _msg(text):
    return [Message(role="user", content=text)]


def test_capture_observer_appends_digested_calls(tmp_path):
    llm = _CountingLLM()
    calls_path = tmp_path / "calls.jsonl"
    llm.add_call_observer(_capture_observer(calls_path))

    llm.complete(_msg("hello"), temperature=0.2, max_tokens=64)
    llm.complete(_msg("world"), temperature=0.2, max_tokens=64)

    lines = [json.loads(l) for l in calls_path.read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0]["response"]["content"] == "live-1"
    assert lines[0]["request_digest"] == _request_digest(
        [{"role": "user", "content": "hello"}], 0.2, 64
    )


def test_matching_prefix_replays_free_then_goes_live(tmp_path):
    recorder = _CountingLLM()
    calls_path = tmp_path / "calls.jsonl"
    recorder.add_call_observer(_capture_observer(calls_path))
    recorder.complete(_msg("step one"), temperature=0.2, max_tokens=64)
    recorder.complete(_msg("step two"), temperature=0.2, max_tokens=64)
    recorded = [
        json.loads(l) for l in calls_path.read_text().splitlines()
    ]

    fresh = _CountingLLM()
    resumed = PrefixReplayLLM(fresh, recorded)

    first = resumed.complete(_msg("step one"), temperature=0.2, max_tokens=64)
    second = resumed.complete(_msg("step two"), temperature=0.2, max_tokens=64)
    third = resumed.complete(_msg("step three"), temperature=0.2, max_tokens=64)

    assert (first.content, second.content) == ("live-1", "live-2")  # replayed
    assert fresh.live_calls == 1          # only the third call went live
    assert third.content == "live-1"      # the fresh provider's first call
    assert resumed.replayed_calls == 2
    # the resumed run's OWN capture (on the fresh inner) records live calls,
    # and its ledger bills only them
    assert resumed.ledger.as_dict()["calls"] == 1


def test_divergence_mid_prefix_goes_live_and_stays_live(tmp_path):
    recorder = _CountingLLM()
    calls_path = tmp_path / "calls.jsonl"
    recorder.add_call_observer(_capture_observer(calls_path))
    recorder.complete(_msg("step one"), temperature=0.2, max_tokens=64)
    recorder.complete(_msg("step two"), temperature=0.2, max_tokens=64)
    recorded = [json.loads(l) for l in calls_path.read_text().splitlines()]

    fresh = _CountingLLM()
    resumed = PrefixReplayLLM(fresh, recorded)
    resumed.complete(_msg("step one"), temperature=0.2, max_tokens=64)
    resumed.complete(_msg("DIFFERENT"), temperature=0.2, max_tokens=64)
    # even a later request that WOULD match the archive stays live: replay
    # past a divergence would splice two different trajectories
    resumed.complete(_msg("step two"), temperature=0.2, max_tokens=64)

    assert resumed.replayed_calls == 1
    assert fresh.live_calls == 2
