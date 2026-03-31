"""
LLM interface module for AI-assisted MBSE prototyping.

Provides abstract interface and concrete implementations for LLM integration,
including support for Chain of Thought (CoT) prompting techniques.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.config import Config


@dataclass
class Message:
    """A single message in an LLM conversation."""
    role: str  # "system", "user", or "assistant"
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    content: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMInterface(ABC):
    """Abstract base class for LLM integrations."""

    @abstractmethod
    def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Generate a completion from the LLM."""

    def chat(self, user_message: str, system_prompt: str = "") -> str:
        """Simple single-turn chat interface."""
        messages: List[Message] = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=user_message))
        response = self.complete(messages)
        return response.content


class GeminiLLM(LLMInterface):
    """Google Gemini API-backed LLM implementation."""

    def __init__(
        self,
        model: str = "gemini-3-flash-preview",
        api_key: Optional[str] = None,
        use_test_key: Optional[bool] = None,
        enable_langsmith: bool = True,
    ):
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "google-genai package is required. Install with: pip install google-genai"
            ) from e

        self.model = model
        self.langsmith_enabled = False

        # Load and export runtime env from .env through centralized config.
        Config.setup_langsmith_env(use_test=use_test_key)

        selected_api_key = api_key or Config.get_gemini_api_key(use_test=use_test_key)
        if not selected_api_key:
            raise ValueError(
                "Gemini API key is missing. Please set GEMINI_API_KEY or GEMINI_API_KEY_TEST in .env."
            )

        base_client = genai.Client(api_key=selected_api_key)
        self.client = self._maybe_wrap_with_langsmith(base_client, enable_langsmith)

    def _maybe_wrap_with_langsmith(self, base_client: Any, enable_langsmith: bool) -> Any:
        """Wrap Gemini client with LangSmith when tracing is enabled and installed."""
        if not enable_langsmith or not Config.langsmith_enabled():
            return base_client

        try:
            from langsmith import wrappers

            wrapped_client = wrappers.wrap_gemini(
                base_client,
                tracing_extra={
                    "tags": ["gemini", "llm-interface"],
                    "metadata": {
                        "integration": "google-genai",
                        "model": self.model,
                    },
                },
            )
            self.langsmith_enabled = True
            return wrapped_client
        except ImportError:
            # Keep normal Gemini calls working even if LangSmith package is absent.
            return base_client

    def complete(
        self,
        messages: List[Message],
        temperature: float = 1.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Call Gemini and normalize structured response fields into LLMResponse."""
        response = self.client.models.generate_content(
            model=self.model,
            contents=[m.content for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content, finish_reason = self._extract_content_and_finish_reason(response)

        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        completion_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        total_tokens = int(getattr(usage, "total_token_count", 0) or 0)
        thoughts_token_count = int(getattr(usage, "thoughts_token_count", 0) or 0)

        model_name = (
            getattr(response, "model_version", None)
            or getattr(response, "model", None)
            or self.model
        )

        metadata: Dict[str, Any] = {
            "response_id": getattr(response, "response_id", None),
            "finish_reason": finish_reason,
            "total_tokens": total_tokens,
            "thoughts_token_count": thoughts_token_count,
            "langsmith_enabled": self.langsmith_enabled,
            "candidate_count": len(getattr(response, "candidates", None) or []),
        }

        return LLMResponse(
            content=content,
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            metadata=metadata,
        )

    @staticmethod
    def _extract_content_and_finish_reason(response: Any) -> tuple[str, str]:
        """Extract response text and finish reason across SDK response variants."""
        content = getattr(response, "text", None) or ""
        finish_reason = ""

        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return content, finish_reason

        first_candidate = candidates[0]
        finish_reason_obj = getattr(first_candidate, "finish_reason", None)
        finish_reason = getattr(finish_reason_obj, "value", None) or str(finish_reason_obj or "")

        if content:
            return content, finish_reason

        candidate_content = getattr(first_candidate, "content", None)
        parts = getattr(candidate_content, "parts", None) or []
        content = "".join(
            part_text
            for part_text in (getattr(part, "text", "") for part in parts)
            if part_text
        )
        return content, finish_reason


class OpenAILLM(LLMInterface):
    """OpenAI API-backed LLM implementation."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            ) from e

        self.model = model
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Call the OpenAI API to generate a completion."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[m.to_dict() for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )


class MockLLM(LLMInterface):
    """
    Mock LLM for testing and development without API keys.

    Returns structured SysML v2 responses based on simple keyword matching.
    """

    def __init__(self, model: str = "mock-gpt-4"):
        self.model = model
        self._call_count = 0
        self._responses: List[str] = []  # injectable responses for testing

    def inject_response(self, response: str) -> None:
        """Inject a specific response for the next call."""
        self._responses.append(response)

    def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Return a mock response based on the input messages."""
        self._call_count += 1

        if self._responses:
            content = self._responses.pop(0)
            return LLMResponse(
                content=content,
                model=self.model,
                prompt_tokens=100,
                completion_tokens=200,
            )

        # Build context-aware mock response
        user_content = " ".join(
            m.content for m in messages if m.role in ("user", "system")
        ).lower()

        content = self._generate_mock_response(user_content)
        return LLMResponse(
            content=content,
            model=self.model,
            prompt_tokens=100,
            completion_tokens=len(content.split()),
        )

    def _generate_mock_response(self, context: str) -> str:
        """Generate a contextually relevant mock response."""
        if "requirement" in context:
            return (
                "Let me analyze the requirements step by step.\n\n"
                "Step 1: Identify functional requirements\n"
                "The system must provide reliable operation with high availability.\n\n"
                "Step 2: Identify non-functional requirements\n"
                "Performance: latency < 100ms, throughput > 1000 req/s\n"
                "Safety: fail-safe behavior in all edge cases\n\n"
                "Requirements identified:\n"
                "- REQ-001: The system shall operate continuously\n"
                "- REQ-002: The system shall respond within 100ms\n"
                "- REQ-003: The system shall be fault-tolerant\n"
            )
        elif "sysml" in context or "block" in context or "part def" in context:
            return (
                "Thinking through the architecture...\n\n"
                "Step 1: Identify main system components\n"
                "The system needs a controller, sensors, and actuators.\n\n"
                "Step 2: Define interfaces\n"
                "Components communicate via well-defined ports.\n\n"
                "```sysml\n"
                "package SystemDesign {\n"
                "    part def Controller {\n"
                "        port commandOut : CommandPort;\n"
                "        port sensorIn : SensorPort;\n"
                "        attribute processingRate : Real = 1000.0 [Hz];\n"
                "    }\n"
                "    part def Sensor {\n"
                "        port dataOut : SensorPort;\n"
                "        attribute samplingRate : Real = 100.0 [Hz];\n"
                "    }\n"
                "    part def Actuator {\n"
                "        port commandIn : CommandPort;\n"
                "        attribute responseTime : Real = 10.0 [ms];\n"
                "    }\n"
                "}\n"
                "```\n"
            )
        elif "design space" in context or "option" in context or "alternative" in context:
            return (
                "Evaluating design alternatives...\n\n"
                "Option A: Centralized architecture\n"
                "- Pros: Simple coordination, easy debugging\n"
                "- Cons: Single point of failure, scalability limits\n"
                "- Score: 0.72\n\n"
                "Option B: Distributed architecture\n"
                "- Pros: High resilience, better scalability\n"
                "- Cons: Complex coordination, harder to debug\n"
                "- Score: 0.85\n\n"
                "Recommendation: Option B (distributed) due to higher resilience score.\n"
            )
        elif "evaluate" in context or "score" in context or "quality" in context:
            return (
                "Evaluation results:\n"
                "- Functional completeness: 0.88\n"
                "- Performance compliance: 0.75\n"
                "- Safety compliance: 0.90\n"
                "- Overall score: 0.84\n"
            )
        else:
            return (
                "I'll analyze this systematically.\n\n"
                "Step 1: Understanding the context\n"
                "The system requires careful design considering all constraints.\n\n"
                "Step 2: Proposing a solution\n"
                "Based on best practices for cyber-physical systems, I recommend\n"
                "a modular architecture with clear separation of concerns.\n\n"
                "Step 3: Verification\n"
                "The proposed solution satisfies the stated requirements.\n"
            )
