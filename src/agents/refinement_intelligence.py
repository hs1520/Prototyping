"""Narrow ports for model-generation intelligence used by Refinement Closure."""
from __future__ import annotations

from collections import deque
from typing import Any, Deque, Iterable, Mapping, Protocol


class RefinementIntelligence(Protocol):
    def chat(self, prompt: str, *, system_prompt: str) -> str: ...

    def generate(self, payload: Mapping[str, Any]) -> Any: ...

    def evaluate(self, **payload: Any) -> Any: ...

    def evaluate_design(self, **payload: Any) -> Any: ...


class RuntimeRefinementIntelligence:
    """Production adapter; resolves runtime collaborators at call time.

    Late resolution preserves test injection while preventing the refinement
    implementation from depending on three unrelated runtime interfaces.
    """

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    def chat(self, prompt: str, *, system_prompt: str) -> str:
        return self._runtime.llm.chat(prompt, system_prompt=system_prompt)

    def generate(self, payload: Mapping[str, Any]) -> Any:
        return self._runtime.design_agent.run(dict(payload))

    def evaluate(self, **payload: Any) -> Any:
        return self._runtime.evaluator.evaluate(**payload)

    def evaluate_design(self, **payload: Any) -> Any:
        return self._runtime.cot.evaluate_design(**payload)

    def verdict_robustness(self, *args: Any, **kwargs: Any) -> Any:
        method = getattr(self._runtime.evaluator, "verdict_robustness", None)
        return method(*args, **kwargs) if method is not None else None


class ScriptedRefinementIntelligence:
    """Deterministic in-memory adapter for Refinement Closure tests.

    Tests script responses at the same intelligence boundary used in
    production instead of replacing individual refinement helpers.  Every
    invocation is retained in ``calls`` so interaction assertions stay at the
    module interface.
    """

    def __init__(
        self,
        *,
        chat: Iterable[Any] = (),
        generate: Iterable[Any] = (),
        evaluate: Iterable[Any] = (),
        evaluate_design: Iterable[Any] = (),
        verdict_robustness: Iterable[Any] = (),
    ) -> None:
        self._responses: dict[str, Deque[Any]] = {
            "chat": deque(chat),
            "generate": deque(generate),
            "evaluate": deque(evaluate),
            "evaluate_design": deque(evaluate_design),
            "verdict_robustness": deque(verdict_robustness),
        }
        self.calls: dict[str, list[Mapping[str, Any]]] = {
            name: [] for name in self._responses
        }

    def _next(self, channel: str, payload: Mapping[str, Any]) -> Any:
        self.calls[channel].append(dict(payload))
        queue = self._responses[channel]
        if not queue:
            raise AssertionError(
                f"unexpected RefinementIntelligence.{channel} call"
            )
        response = queue.popleft()
        return response(payload) if callable(response) else response

    def chat(self, prompt: str, *, system_prompt: str) -> str:
        return str(self._next("chat", {
            "prompt": prompt,
            "system_prompt": system_prompt,
        }))

    def generate(self, payload: Mapping[str, Any]) -> Any:
        return self._next("generate", payload)

    def evaluate(self, **payload: Any) -> Any:
        return self._next("evaluate", payload)

    def evaluate_design(self, **payload: Any) -> Any:
        return self._next("evaluate_design", payload)

    def verdict_robustness(self, *args: Any, **kwargs: Any) -> Any:
        payload = {"args": args, "kwargs": kwargs}
        if not self._responses["verdict_robustness"]:
            self.calls["verdict_robustness"].append(payload)
            return None
        return self._next("verdict_robustness", payload)
