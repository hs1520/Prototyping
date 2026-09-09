"""LLM interface package for AI-assisted MBSE."""

from .chain_of_thought import (
    ChainOfThoughtPrompter,
    CoTResult,
    ThoughtStep,
)
from .github_auth import GitHubAuthManager, GitHubCLIAuthError
from .interface import (
    GeminiLLM,
    GitHubCopilotLLM,
    LLMInterface,
    LLMResponse,
    Message,
    MockLLM,
    TokenLedger,
    VertexLLM,
)

__all__ = [
    "ChainOfThoughtPrompter",
    "CoTResult",
    "GeminiLLM",
    "GitHubAuthManager",
    "GitHubCLIAuthError",
    "GitHubCopilotLLM",
    "LLMInterface",
    "LLMResponse",
    "Message",
    "MockLLM",
    "ThoughtStep",
    "TokenLedger",
    "VertexLLM",
]
