"""
LLM interface module for AI-assisted MBSE prototyping.

Provides abstract interface and concrete implementations for LLM integration,
including support for Chain of Thought (CoT) prompting techniques.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, cast

import requests

from src.config import Config
from .github_auth import GitHubAuthManager, GitHubCLIAuthError


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
        max_tokens: int = 8192,
    ) -> LLMResponse:
        """Call Gemini and normalize structured response fields into LLMResponse."""
        # google-genai expects generation settings in `config`, not top-level kwargs.
        response = self.client.models.generate_content(
            model=self.model,
            contents=[m.content for m in messages],
            config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
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

class GitHubCopilotLLM(LLMInterface):
    """GitHub Models/Copilot-backed LLM using GitHub CLI authentication."""

    def __init__(
        self,
        model: str = "openai/gpt-5",
        api_key: Optional[str] = None,
        base_url: str = "https://models.github.ai/inference",
        models_endpoint: str = "https://models.github.ai/catalog/models",
        auto_login: bool = True,
        enable_langsmith: bool = True,
        auth_manager: Optional[GitHubAuthManager] = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            ) from e

        self.model = model
        self.base_url = base_url
        self.models_endpoint = models_endpoint
        self.auto_login = auto_login
        self.langsmith_enabled = False
        self.auth_manager = auth_manager or GitHubAuthManager(auto_login=auto_login)
        self._openai_cls = OpenAI
        self._enable_langsmith = enable_langsmith

        # Reuse centralized env setup so GitHub Models calls can be traced like Gemini calls.
        Config.setup_langsmith_env()

        token = self.auth_manager.get_token(explicit_token=api_key)

        if not token:
            raise GitHubCLIAuthError(
                "No GitHub auth token found. Run `gh auth login --hostname github.com --git-protocol https --web`."
            )

        self.client = self._build_client(token)

    def _build_client(self, token: str) -> Any:
        """Create an OpenAI-compatible client and wrap with LangSmith when available."""
        base_client = self._openai_cls(api_key=token, base_url=self.base_url)
        return self._maybe_wrap_with_langsmith(
            base_client,
            getattr(self, "_enable_langsmith", False),
        )

    def _maybe_wrap_with_langsmith(self, base_client: Any, enable_langsmith: bool) -> Any:
        """Wrap OpenAI-compatible client with LangSmith when tracing is enabled and installed."""
        if not enable_langsmith or not Config.langsmith_enabled():
            return base_client

        try:
            from langsmith import wrappers

            wrapped_client = wrappers.wrap_openai(
                base_client,
                tracing_extra=cast(
                    Any,
                    {
                        "tags": ["github-copilot", "llm-interface"],
                        "metadata": {
                            "integration": "openai-compatible",
                            "provider": "github_copilot",
                            "model": self.model,
                            "base_url": self.base_url,
                        },
                    },
                ),
            )
            self.langsmith_enabled = True
            return wrapped_client
        except ImportError:
            return base_client

    def list_models(self, provider: Optional[str] = None, timeout: int = 30) -> List[str]:
        """List model IDs visible to the current GitHub token, optionally filtered by provider."""
        retried_auth = False

        while True:
            token = self.auth_manager.get_token()
            if not token:
                raise GitHubCLIAuthError(
                    "No GitHub auth token found. Run `gh auth login --hostname github.com --git-protocol https --web`."
                )

            response = requests.get(
                self.models_endpoint,
                headers={"Authorization": f"Bearer {token}"},
                timeout=timeout,
            )

            if response.status_code in (401, 403):
                if retried_auth or not self.auto_login:
                    response.raise_for_status()
                token = self.auth_manager.refresh_token()
                self.client = self._build_client(token)
                retried_auth = True
                continue

            response.raise_for_status()
            payload = response.json()

            if isinstance(payload, list):
                items = payload
            elif isinstance(payload, dict):
                raw_items = payload.get("data")
                items = raw_items if isinstance(raw_items, list) else []
            else:
                items = []

            model_ids: List[str] = sorted(
                {
                    item.get("id", "").strip()
                    for item in items
                    if isinstance(item, dict) and item.get("id")
                }
            )

            if not provider:
                return model_ids

            normalized = provider.strip().lower().rstrip("/")
            if not normalized:
                return model_ids

            prefix = f"{normalized}/"
            return [model_id for model_id in model_ids if model_id.lower().startswith(prefix)]

        return []

    def complete(
        self,
        messages: List[Message],
        temperature: float = 1,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Call GitHub Models chat completions using OpenAI-compatible schema."""
        retried_auth = False
        payload_messages = cast(Any, [m.to_dict() for m in messages])

        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=payload_messages,
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                )
                choice = response.choices[0]
                usage = response.usage
                return LLMResponse(
                    content=choice.message.content or "",
                    model=response.model,
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    metadata={
                        "provider": "github_copilot",
                        "base_url": self.base_url,
                        "langsmith_enabled": self.langsmith_enabled,
                    },
                )
            except Exception as exc:
                if retried_auth or not self.auto_login or not self.auth_manager.is_auth_error(exc):
                    raise

                token = self.auth_manager.refresh_token()
                self.client = self._build_client(token)
                retried_auth = True

class VertexLLM(LLMInterface):
    """Vertex-backed Gemini LLM implementation."""

    def __init__(
        self,
        model: str = "gemini-3.1-pro-preview",
        api_key: Optional[str] = None,
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

        Config.setup_langsmith_env()
        selected_api_key = api_key or Config.get_vertex_api_key()
        if not selected_api_key:
            raise ValueError(
                "Vertex API key is missing. Please set VERTEX_API_KEY in .env."
            )

        base_client = genai.Client(vertexai=True, api_key=selected_api_key)
        self.client = self._maybe_wrap_with_langsmith(base_client, enable_langsmith)

    def _maybe_wrap_with_langsmith(self, base_client: Any, enable_langsmith: bool) -> Any:
        """Wrap Vertex client with LangSmith when tracing is enabled and installed."""
        if not enable_langsmith or not Config.langsmith_enabled():
            return base_client

        try:
            from langsmith import wrappers

            wrapped_client = wrappers.wrap_gemini(
                base_client,
                tracing_extra={
                    "tags": ["vertex", "llm-interface"],
                    "metadata": {
                        "integration": "google-genai",
                        "provider": "vertex",
                        "model": self.model,
                    },
                },
            )
            self.langsmith_enabled = True
            return wrapped_client
        except ImportError:
            return base_client

    def complete(
        self,
        messages: List[Message],
        temperature: float = 1.0,
        max_tokens: int = 8192,
    ) -> LLMResponse:
        """Call Vertex Gemini and normalize response into LLMResponse."""
        response = self.client.models.generate_content(
            model=self.model,
            contents=[m.content for m in messages],
            config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )

        content = getattr(response, "text", None) or ""
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        completion_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)

        model_name = (
            getattr(response, "model_version", None)
            or getattr(response, "model", None)
            or self.model
        )

        return LLMResponse(
            content=content,
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            metadata={
                "provider": "vertex",
                "langsmith_enabled": self.langsmith_enabled,
            },
        )


class MockLLM(LLMInterface):
    """Deterministic mock provider for offline tests and local development."""

    def __init__(self):
        self._injected_response: Optional[str] = None
        self._call_count = 0

    def inject_response(self, response: str) -> None:
        """Inject the next response returned by complete()."""
        self._injected_response = response

    def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        self._call_count += 1

        if self._injected_response is not None:
            content = self._injected_response
            self._injected_response = None
            return LLMResponse(
                content=content,
                model="mock-llm",
                prompt_tokens=0,
                completion_tokens=0,
            )

        user_text = "\n".join(
            m.content for m in messages if m.role == "user"
        ).lower()

        if "requirement" in user_text:
            content = (
                "Extracted requirements:\n"
                "1. The system shall provide core functionality.\n"
                "2. The system shall be safe and reliable."
            )
        elif "sysml" in user_text or "block diagram" in user_text:
            content = (
                "```sysml\n"
                "package MockSystem {\n"
                "    part def Controller {}\n"
                "}\n"
                "```"
            )
        else:
            content = "Mock response: design candidate generated successfully."

        return LLMResponse(
            content=content,
            model="mock-llm",
            prompt_tokens=0,
            completion_tokens=0,
            metadata={"temperature": temperature, "max_tokens": max_tokens},
        )


