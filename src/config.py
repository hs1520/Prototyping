"""Configuration module for managing environment variables."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class Config:
    """Application configuration from environment variables."""

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_API_KEY_TEST: str = os.getenv("GEMINI_API_KEY_TEST", "")

    VERTEX_API_KEY: str = os.getenv("VERTEX_API_KEY", "")

    # Default to the test key to control cost.
    GEMINI_USE_TEST_KEY: str = os.getenv("GEMINI_USE_TEST_KEY", "true")

    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_TRACING: str = os.getenv("LANGSMITH_TRACING", "true")
    LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "true")
    LANGSMITH_ENDPOINT: str = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "default")

    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")

    @classmethod
    def _as_bool(cls, value: str) -> Optional[bool]:
        if value is None:
            return None

        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return None

    @classmethod
    def use_test_gemini_key(cls, override: Optional[bool] = None) -> bool:
        """Return whether runtime should use GEMINI_API_KEY_TEST."""
        if override is not None:
            return override

        configured = cls._as_bool(cls.GEMINI_USE_TEST_KEY)
        if configured is not None:
            return configured

        # Invalid switch value: fall back to the test key.
        return True

    @classmethod
    def get_gemini_api_key(cls, use_test: Optional[bool] = None) -> str:
        """Return selected Gemini API key with fallback to the non-test key."""
        select_test = cls.use_test_gemini_key(override=use_test)
        if select_test and cls.GEMINI_API_KEY_TEST:
            return cls.GEMINI_API_KEY_TEST
        return cls.GEMINI_API_KEY

    @classmethod
    def get_vertex_api_key(cls) -> str:
        """Return the Vertex API key."""
        return cls.VERTEX_API_KEY

    @classmethod
    def setup_langsmith_env(cls, use_test: Optional[bool] = None) -> None:
        """Export LangSmith-related env vars for runtime tracing."""
        if cls.LANGSMITH_API_KEY:
            os.environ["LANGSMITH_API_KEY"] = cls.LANGSMITH_API_KEY

        # Keep both flags for compatibility with current LangSmith/LangChain clients.
        os.environ["LANGSMITH_TRACING"] = cls.LANGSMITH_TRACING
        os.environ["LANGCHAIN_TRACING_V2"] = cls.LANGCHAIN_TRACING_V2
        os.environ["LANGSMITH_ENDPOINT"] = cls.LANGSMITH_ENDPOINT
        os.environ["LANGSMITH_PROJECT"] = cls.LANGSMITH_PROJECT

        selected_key = cls.get_gemini_api_key(use_test=use_test)
        if selected_key:
            os.environ["GEMINI_API_KEY"] = selected_key

        if cls.GEMINI_API_KEY_TEST:
            os.environ["GEMINI_API_KEY_TEST"] = cls.GEMINI_API_KEY_TEST

    @classmethod
    def langsmith_enabled(cls) -> bool:
        """Return whether LangSmith tracing is expected to be enabled."""
        tracing_on = cls.LANGSMITH_TRACING.lower() == "true" or cls.LANGCHAIN_TRACING_V2.lower() == "true"
        return bool(cls.LANGSMITH_API_KEY and tracing_on)
