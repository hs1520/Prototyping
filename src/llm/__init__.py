"""LLM interface package for AI-assisted MBSE."""

from .chain_of_thought import (
    ChainOfThoughtPrompter,
    CoTResult,
    ThoughtStep,
)
from .github_auth import GitHubAuthManager, GitHubCLIAuthError
from .interface import (
    GitHubCopilotLLM,
    LLMInterface,
    LLMResponse,
    Message,
    MockLLM,
    OpenAILLM,
)

__all__ = [
    "ChainOfThoughtPrompter",
    "CoTResult",
    "GitHubAuthManager",
    "GitHubCLIAuthError",
    "GitHubCopilotLLM",
    "LLMInterface",
    "LLMResponse",
    "Message",
    "MockLLM",
    "OpenAILLM",
    "ThoughtStep",
]
