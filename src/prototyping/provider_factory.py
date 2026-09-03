"""LLM provider factory module."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional

from ..llm.interface import (
    GeminiLLM,
    GitHubCopilotLLM,
    LLMInterface,
    MockLLM,
    VertexLLM,
)


LLM_PROVIDER_FACTORIES: Dict[str, Callable[..., LLMInterface]] = {
    "mock": MockLLM,
    "gemini": GeminiLLM,
    "github_copilot": GitHubCopilotLLM,
    "vertex": VertexLLM,
}

LLM_PROVIDER_ALIASES: Dict[str, str] = {
    "default": "mock",
    "test": "mock",
}

DEFAULT_LLM_MODELS: Dict[str, str] = {
    "gemini": "gemini-3-flash-preview",
    "github_copilot": "openai/gpt-4.1-mini",
    "vertex": "gemini-3.1-pro-preview",
}


def _normalize_provider_name(provider: Optional[str]) -> str:
    normalized = (provider or "mock").strip().lower().replace("-", "_")
    return LLM_PROVIDER_ALIASES.get(normalized, normalized)


def _build_constructor_kwargs(
    factory: Any,
    model: Optional[str],
    api_key: Optional[str],
    provider_kwargs: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    kwargs = dict(provider_kwargs or {})
    signature = inspect.signature(factory.__init__)
    parameters = signature.parameters
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )

    if model is not None and ("model" in parameters or accepts_var_kwargs):
        kwargs.setdefault("model", model)
    if api_key is not None and ("api_key" in parameters or accepts_var_kwargs):
        kwargs.setdefault("api_key", api_key)

    return kwargs


def _resolve_model_name(provider_name: str, model: Optional[str]) -> Optional[str]:
    if model is not None:
        return model
    return DEFAULT_LLM_MODELS.get(provider_name)


def register_llm_provider(name: str, provider_factory: Any) -> None:
    """Register a custom LLM provider for create_llm()."""
    LLM_PROVIDER_FACTORIES[_normalize_provider_name(name)] = provider_factory


def available_llm_providers() -> List[str]:
    """Return all currently registered canonical provider names."""
    return sorted(LLM_PROVIDER_FACTORIES.keys())


def create_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    provider_kwargs: Optional[Dict[str, Any]] = None,
) -> LLMInterface:
    """Create an LLM instance."""
    provider_name = _normalize_provider_name(provider)
    factory = LLM_PROVIDER_FACTORIES.get(provider_name)
    if factory is None:
        supported = ", ".join(available_llm_providers())
        raise ValueError(
            f"Unknown LLM provider '{provider}'. Supported providers: {supported}"
        )

    resolved_model = _resolve_model_name(provider_name, model)

    kwargs = _build_constructor_kwargs(factory, resolved_model, api_key, provider_kwargs)
    return factory(**kwargs)
