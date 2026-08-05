"""
LLM interface module for AI-assisted MBSE prototyping.

Provides abstract interface and concrete implementations for LLM integration,
including support for Chain of Thought (CoT) prompting techniques.

The base class is a template method: ``complete()`` resolves the default
temperature (low, for structured SysML output), retries transient provider
errors with jittered backoff, enforces a per-request timeout (provider side),
and records every call into a per-instance :class:`TokenLedger`.  Providers
implement ``_complete_impl()`` only.
"""

from __future__ import annotations

import os
import random
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import requests

from src.config import Config
from .github_auth import GitHubAuthManager, GitHubCLIAuthError


# Structured SysML/JSON generation wants determinism first; escalation helpers
# raise the temperature only when a low-temperature answer fails validation.
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 20480
# Escalation ladder used by *_with_escalation helpers.
ESCALATION_TEMPERATURES: Tuple[float, ...] = (DEFAULT_TEMPERATURE, 0.6, 1.0)


def _default_timeout_seconds(default: float = 120.0) -> float:
    """Per-request timeout, overridable via LLM_TIMEOUT_SECONDS."""
    try:
        return float(os.environ.get("LLM_TIMEOUT_SECONDS", str(default)))
    except ValueError:
        return default


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


@dataclass
class TokenLedger:
    """Cumulative usage accounting for one LLM instance (all calls in a run)."""
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retries: int = 0
    failures: int = 0
    elapsed_seconds: float = 0.0

    def record(self, response: LLMResponse, elapsed: float, retries: int) -> None:
        self.calls += 1
        self.prompt_tokens += int(response.prompt_tokens or 0)
        self.completion_tokens += int(response.completion_tokens or 0)
        self.retries += retries
        self.elapsed_seconds += elapsed

    def record_failure(self, elapsed: float, retries: int) -> None:
        self.failures += 1
        self.retries += retries
        self.elapsed_seconds += elapsed

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> Dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "retries": self.retries,
            "failures": self.failures,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
        }

    def summary(self) -> str:
        def fmt(n: int) -> str:
            return f"{n / 1000:.1f}k" if n >= 1000 else str(n)

        parts = [
            f"{self.calls} calls",
            f"{fmt(self.prompt_tokens)} prompt + {fmt(self.completion_tokens)} completion tokens",
            f"{self.elapsed_seconds:.0f}s in LLM",
        ]
        if self.retries:
            parts.append(f"{self.retries} retries")
        if self.failures:
            parts.append(f"{self.failures} failed calls")
        return ", ".join(parts)


class LLMInterface(ABC):
    """Abstract base class for LLM integrations.

    ``complete()`` is concrete: it resolves the default temperature, retries
    transient errors (429/5xx/timeouts) with jittered backoff, and records
    usage in :attr:`ledger`.  Subclasses implement :meth:`_complete_impl`.
    Duck-typed stand-ins that define their own ``complete`` keep working.
    """

    # Backoff schedule for transient errors; jittered ±25% per attempt.
    # Override per instance (e.g. in tests) to speed up or disable retries.
    RETRY_DELAYS: Tuple[float, ...] = (2.0, 5.0, 15.0, 40.0)

    _RETRYABLE_MARKERS: Tuple[str, ...] = (
        "429", "rate limit", "resource_exhausted", "resource exhausted",
        "500", "502", "503", "504", "unavailable", "overloaded",
        "timeout", "timed out", "deadline", "connection",
        "connecterror", "name resolution", "nodename nor servname",
        "temporarily", "server error", "internal error",
    )
    _RETRYABLE_STATUS = {429, 500, 502, 503, 504}

    @property
    def ledger(self) -> TokenLedger:
        """Cumulative token/call accounting for this instance (lazy, survives
        subclasses that never call super().__init__())."""
        led = self.__dict__.get("_ledger")
        if led is None:
            led = TokenLedger()
            self.__dict__["_ledger"] = led
        return led

    def add_call_observer(self, observer: Callable[[Dict[str, Any]], None]) -> int:
        """Register an application-owned transcript observer.

        Observers receive completed provider calls and are deliberately local
        to this LLM instance.  The caller must remove the observer when its
        bounded Agent task closes so transcripts cannot cross task/role scope.
        """
        observers = self.__dict__.setdefault("_call_observers", {})
        sequence = int(self.__dict__.get("_call_observer_sequence", 0)) + 1
        self.__dict__["_call_observer_sequence"] = sequence
        observers[sequence] = observer
        return sequence

    def remove_call_observer(self, observer_id: int) -> None:
        self.__dict__.setdefault("_call_observers", {}).pop(
            int(observer_id), None
        )

    @property
    def call_observer_errors(self) -> Tuple[str, ...]:
        return tuple(self.__dict__.get("_call_observer_errors", ()))

    def _notify_call_observers(
        self,
        *,
        messages: List[Message],
        response: LLMResponse,
        temperature: float,
        max_tokens: int,
        retries: int,
        conversation_id: Optional[str] = None,
        new_message_offset: int = 0,
        label: Optional[str] = None,
    ) -> None:
        event = {
            "messages": [message.to_dict() for message in messages],
            # Which pipeline stage issued this call, when the caller says so.
            # Observers archive by stage; nothing about the request depends on it.
            "label": label,
            # A multi-turn call resends every earlier turn.  Observers that
            # archive transcripts must append only what is new, or one bounded
            # conversation would be recorded O(n^2) times.  Offset 0 (the
            # default, and every single-turn call) means "all of it is new".
            "new_message_offset": int(new_message_offset),
            "conversation_id": conversation_id,
            "response": {
                "role": "assistant",
                "content": response.content,
                "model": response.model,
                "prompt_tokens": int(response.prompt_tokens or 0),
                "completion_tokens": int(response.completion_tokens or 0),
            },
            "temperature": temperature,
            "max_tokens": int(max_tokens),
            "retries": int(retries),
        }
        errors = self.__dict__.setdefault("_call_observer_errors", [])
        for observer in tuple(
            self.__dict__.setdefault("_call_observers", {}).values()
        ):
            try:
                observer(event)
            except Exception as exc:
                # Transcript bookkeeping must not turn a completed provider
                # call into a provider retry.  The task session records its own
                # terminal state and the orchestrator rejects it before commit.
                errors.append(f"{type(exc).__name__}: {exc}")

    def complete(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        *,
        conversation_id: Optional[str] = None,
        new_message_offset: int = 0,
        label: Optional[str] = None,
    ) -> LLMResponse:
        """Generate a completion with unified retry, timeout, and accounting.

        ``conversation_id``/``new_message_offset`` are transcript bookkeeping
        for multi-turn callers (see :class:`Conversation`); they do not change
        what is sent to the provider, which is always the full ``messages``.
        """
        resolved_temp = DEFAULT_TEMPERATURE if temperature is None else temperature
        delays = list(self.RETRY_DELAYS)
        retries = 0
        start = time.monotonic()
        while True:
            try:
                response = self._complete_impl(
                    messages, temperature=resolved_temp, max_tokens=max_tokens
                )
                self.ledger.record(response, time.monotonic() - start, retries)
                self._notify_call_observers(
                    messages=messages,
                    response=response,
                    temperature=resolved_temp,
                    max_tokens=max_tokens,
                    retries=retries,
                    conversation_id=conversation_id,
                    new_message_offset=new_message_offset,
                    label=label,
                )
                return response
            except Exception as exc:
                if not delays or not self._is_retryable(exc):
                    self.ledger.record_failure(time.monotonic() - start, retries)
                    raise
                delay = delays.pop(0) * random.uniform(0.75, 1.25)
                retries += 1
                print(
                    f"  [LLM] transient error ({type(exc).__name__}: "
                    f"{str(exc)[:120]}) — retry {retries} in {delay:.0f}s"
                )
                time.sleep(delay)

    @abstractmethod
    def _complete_impl(
        self,
        messages: List[Message],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Provider-specific single API call (no retry/accounting concerns)."""

    @classmethod
    def _is_retryable(cls, exc: Exception) -> bool:
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if isinstance(status, int) and status in cls._RETRYABLE_STATUS:
            return True
        text = f"{type(exc).__name__} {exc}".lower()
        return any(marker in text for marker in cls._RETRYABLE_MARKERS)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def chat(
        self,
        user_message: str,
        system_prompt: str = "",
        temperature: Optional[float] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """Simple single-turn chat interface."""
        messages: List[Message] = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=user_message))
        response = self.complete(messages, temperature=temperature, max_tokens=max_tokens)
        return response.content

    def complete_with_escalation(
        self,
        messages: List[Message],
        validate: Callable[[str], bool],
        temperatures: Tuple[float, ...] = ESCALATION_TEMPERATURES,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Tuple[LLMResponse, bool]:
        """Low-temperature-first completion with temperature escalation.

        Calls ``complete`` at each temperature in ``temperatures`` until
        ``validate(content)`` returns truthy.  Returns ``(response, ok)`` —
        on exhaustion the last response is returned with ``ok=False`` so the
        caller can apply its own fallback.
        """
        last: Optional[LLMResponse] = None
        for temp in temperatures:
            last = self.complete(messages, temperature=temp, max_tokens=max_tokens)
            try:
                if validate(last.content):
                    return last, True
            except Exception:
                pass  # validation failure at this temperature → escalate
        assert last is not None
        return last, False

    def chat_with_escalation(
        self,
        user_message: str,
        system_prompt: str = "",
        validate: Optional[Callable[[str], bool]] = None,
        temperatures: Tuple[float, ...] = ESCALATION_TEMPERATURES,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Tuple[str, bool]:
        """chat() variant of :meth:`complete_with_escalation`; returns (content, ok)."""
        if validate is None:
            return self.chat(user_message, system_prompt, max_tokens=max_tokens), True
        messages: List[Message] = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=user_message))
        response, ok = self.complete_with_escalation(
            messages, validate, temperatures=temperatures, max_tokens=max_tokens
        )
        return response.content, ok


class Conversation:
    """A bounded multi-turn conversation over a stateless provider API.

    The provider APIs used here (Vertex/Gemini ``generateContent``, the GitHub
    Models chat endpoint) keep no server-side session: continuity exists only
    because the caller resends the earlier turns.  This object owns those turns
    so the exact bytes the model saw stay application-owned, reproducible, and
    digest-recordable — the property §5.3 requires and a provider conversation
    id could not give.

    One instance belongs to one Agent role and one bounded task, mirroring the
    TaskSession rules.  It is deliberately not shared across roles or tasks.
    """

    def __init__(
        self,
        llm: "LLMInterface",
        system_prompt: str = "",
        *,
        conversation_id: Optional[str] = None,
    ):
        self._llm = llm
        self._system_prompt = str(system_prompt or "")
        self._turns: List[Message] = []
        self.conversation_id = str(
            conversation_id or f"conv-{uuid.uuid4().hex[:12]}"
        )

    @property
    def turns(self) -> Tuple[Message, ...]:
        """The ordered user/assistant turns, excluding the system prompt."""
        return tuple(self._turns)

    @property
    def assistant_turn_count(self) -> int:
        return sum(1 for message in self._turns if message.role == "assistant")

    def _rendered(self) -> List[Message]:
        head = (
            [Message(role="system", content=self._system_prompt)]
            if self._system_prompt else []
        )
        return head + list(self._turns)

    def send(
        self,
        user_message: str,
        temperature: Optional[float] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        label: Optional[str] = None,
    ) -> str:
        """Append a user turn, send the whole conversation, keep the reply.

        The assistant reply is retained, so the next ``send`` shows the model
        its own previous answer rather than a paraphrase of it.
        """
        # What an observer has already archived is everything the *previous*
        # call sent.  On the opening turn that is nothing — the system prompt
        # has not been recorded yet, so the offset must be 0 or it would be
        # dropped from the transcript entirely.
        offset = len(self._rendered()) if self._turns else 0
        self._turns.append(Message(role="user", content=str(user_message)))
        response = self._llm.complete(
            self._rendered(),
            temperature=temperature,
            max_tokens=max_tokens,
            conversation_id=self.conversation_id,
            new_message_offset=offset,
            label=label,
        )
        self._turns.append(
            Message(role="assistant", content=str(response.content))
        )
        return response.content


def _split_gemini_messages(
    messages: List[Message],
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Split messages into (system_instruction, role-tagged contents).

    google-genai carries the system prompt in ``config.system_instruction``;
    conversation turns are role-tagged Content dicts ("user" / "model").
    Flattening everything into anonymous strings (the previous behaviour)
    demotes the system prompt to ordinary user text.
    """
    system_parts = [m.content for m in messages if m.role == "system"]
    system_instruction = "\n\n".join(system_parts) if system_parts else None
    contents: List[Dict[str, Any]] = [
        {
            "role": "model" if m.role == "assistant" else "user",
            "parts": [{"text": m.content}],
        }
        for m in messages
        if m.role != "system"
    ]
    return system_instruction, contents


class GeminiLLM(LLMInterface):
    """Google Gemini API-backed LLM implementation."""

    def __init__(
        self,
        model: str = "gemini-3-flash-preview",
        api_key: Optional[str] = None,
        use_test_key: Optional[bool] = None,
        enable_langsmith: bool = True,
        timeout_seconds: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "google-genai package is required. Install with: pip install google-genai"
            ) from e

        self.model = model
        self.seed = seed
        self.langsmith_enabled = False
        self.timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else _default_timeout_seconds()
        )

        # Load and export runtime env from .env through centralized config.
        Config.setup_langsmith_env(use_test=use_test_key)

        selected_api_key = api_key or Config.get_gemini_api_key(use_test=use_test_key)
        if not selected_api_key:
            raise ValueError(
                "Gemini API key is missing. Please set GEMINI_API_KEY or GEMINI_API_KEY_TEST in .env."
            )

        base_client = genai.Client(
            api_key=selected_api_key,
            http_options={"timeout": int(self.timeout_seconds * 1000)},
        )
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

    def _complete_impl(
        self,
        messages: List[Message],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Call Gemini and normalize structured response fields into LLMResponse."""
        system_instruction, contents = _split_gemini_messages(messages)
        # google-genai expects generation settings in `config`, not top-level kwargs.
        config: Dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        seed = getattr(self, "seed", None)
        if seed is not None:
            config["seed"] = seed
        if system_instruction:
            config["system_instruction"] = system_instruction

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
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
            "seed": seed,
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
        timeout_seconds: Optional[float] = None,
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
        self.timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else _default_timeout_seconds()
        )

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
        client_kwargs: Dict[str, Any] = {"api_key": token, "base_url": self.base_url}
        timeout = getattr(self, "timeout_seconds", None)
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        base_client = self._openai_cls(**client_kwargs)
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

    def _complete_impl(
        self,
        messages: List[Message],
        temperature: float,
        max_tokens: int,
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

    ARCHITECTURE_MAX_TOKENS = 65536

    # Retry shared-capacity failures at most twice. A client deadline is not a
    # capacity failure and is deliberately rejected without replaying the same
    # expensive request.
    RETRY_DELAYS = (10.0, 30.0)

    def __init__(
        self,
        model: str = "gemini-3.1-pro-preview",
        api_key: Optional[str] = None,
        enable_langsmith: bool = True,
        timeout_seconds: Optional[float] = None,
        seed: Optional[int] = 0,
        thinking_level: str = "HIGH",
    ):
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "google-genai package is required. Install with: pip install google-genai"
            ) from e

        self.model = model
        self.seed = seed
        self.thinking_level = thinking_level.upper()
        if self.thinking_level not in {"LOW", "HIGH"}:
            raise ValueError("Vertex thinking_level must be LOW or HIGH")
        self.langsmith_enabled = False
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else _default_timeout_seconds(600.0)
        )

        Config.setup_langsmith_env()
        selected_api_key = api_key or Config.get_vertex_api_key()
        if not selected_api_key:
            raise ValueError(
                "Vertex API key is missing. Please set VERTEX_API_KEY in .env."
            )

        base_client = genai.Client(
            vertexai=True,
            api_key=selected_api_key,
            http_options={"timeout": int(self.timeout_seconds * 1000)},
        )
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

    def _complete_impl(
        self,
        messages: List[Message],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Call Vertex Gemini and normalize response into LLMResponse.

        Transient errors (429/RESOURCE_EXHAUSTED/5xx/timeouts) are retried by
        the LLMInterface.complete() wrapper — no bespoke retry loop here.
        """
        system_instruction, contents = _split_gemini_messages(messages)
        config: Dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "thinking_config": {
                "thinking_level": getattr(self, "thinking_level", "HIGH")
            },
        }
        seed = getattr(self, "seed", None)
        if seed is not None:
            config["seed"] = seed
        if system_instruction:
            config["system_instruction"] = system_instruction

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
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
                "seed": seed,
                "thinking_level": getattr(self, "thinking_level", "HIGH"),
            },
        )

    @classmethod
    def _is_retryable(cls, exc: Exception) -> bool:
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        text = f"{type(exc).__name__} {exc}".lower()
        if status == 504 or "deadline" in text or "timeout" in text:
            return False
        return super()._is_retryable(exc)


class MockLLM(LLMInterface):
    """Deterministic mock provider for offline tests and local development."""

    def __init__(self):
        self._injected_response: Optional[str] = None
        self._call_count = 0

    def inject_response(self, response: str) -> None:
        """Inject the next response returned by complete()."""
        self._injected_response = response

    def _complete_impl(
        self,
        messages: List[Message],
        temperature: float,
        max_tokens: int,
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
