"""
Configuration module for managing environment variables.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class Config:
    """Application configuration from environment variables."""

    # LLM Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_API_KEY_TEST: str = os.getenv("GEMINI_API_KEY_TEST", "")

    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    VERTEX_API_KEY: str = os.getenv("VERTEX_API_KEY", "")

    # Practical switch: use test key by default to control costs.
    GEMINI_USE_TEST_KEY: str = os.getenv("GEMINI_USE_TEST_KEY", "true")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))

    # LangSmith Configuration
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_TRACING: str = os.getenv("LANGSMITH_TRACING", "true")
    LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "true")
    LANGSMITH_ENDPOINT: str = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "default")

    # Database Configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./test.db")
    DATABASE_HOST: str = os.getenv("DATABASE_HOST", "localhost")
    DATABASE_PORT: int = int(os.getenv("DATABASE_PORT", "5432"))
    DATABASE_USER: str = os.getenv("DATABASE_USER", "user")
    DATABASE_PASSWORD: str = os.getenv("DATABASE_PASSWORD", "password")

    # Application Configuration
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "Prototyping")

    # RAG Configuration
    RAG_EMBEDDING_MODEL: str = os.getenv(
        "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    RAG_VECTOR_DB_PATH: str = os.getenv("RAG_VECTOR_DB_PATH", "./data/vector_db")

    # DSE Configuration
    DSE_MAX_ITERATIONS: int = int(os.getenv("DSE_MAX_ITERATIONS", "100"))
    DSE_EXPLORATION_FACTOR: float = float(os.getenv("DSE_EXPLORATION_FACTOR", "0.2"))

    # Vector Database Configuration
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")

    @classmethod
    def validate(cls) -> bool:
        """
        Validate critical configuration values.
        Returns True if valid, False otherwise.
        """
        if cls.DEBUG:
            print("⚠️  DEBUG mode is enabled")

        if not cls.LLM_API_KEY:
            print("⚠️  LLM_API_KEY is not set")
            return False

        return True

    @classmethod
    def _as_bool(cls, value: str) -> Optional[bool]:
        """Parse common truthy/falsey strings. Return None when unset/unknown."""
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

        # Fallback: prioritize saving cost if switch value is invalid.
        return True

    @classmethod
    def get_gemini_api_key(cls, use_test: Optional[bool] = None) -> str:
        """Return selected Gemini API key with fallback to the non-test key."""
        select_test = cls.use_test_gemini_key(override=use_test)
        if select_test and cls.GEMINI_API_KEY_TEST:
            return cls.GEMINI_API_KEY_TEST
        return cls.GEMINI_API_KEY
    
    @classmethod
    def get_anthropic_api_key(cls) -> str:
        """Return the Anthropic API key."""
        return cls.ANTHROPIC_API_KEY

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

        # Keep GEMINI_API_KEY as the active runtime key used by SDK defaults.
        selected_key = cls.get_gemini_api_key(use_test=use_test)
        if selected_key:
            os.environ["GEMINI_API_KEY"] = selected_key

        # Preserve explicit test key env var for debugging/introspection.
        if cls.GEMINI_API_KEY_TEST:
            os.environ["GEMINI_API_KEY_TEST"] = cls.GEMINI_API_KEY_TEST

    @classmethod
    def langsmith_enabled(cls) -> bool:
        """Return whether LangSmith tracing is expected to be enabled."""
        tracing_on = cls.LANGSMITH_TRACING.lower() == "true" or cls.LANGCHAIN_TRACING_V2.lower() == "true"
        return bool(cls.LANGSMITH_API_KEY and tracing_on)

    @classmethod
    def to_dict(cls) -> dict:
        """Return configuration as a dictionary (excluding sensitive data)."""
        return {
            "debug": cls.DEBUG,
            "log_level": cls.LOG_LEVEL,
            "project_name": cls.PROJECT_NAME,
            "llm_model": cls.LLM_MODEL,
            "database_url": cls.DATABASE_URL,
            "rag_embedding_model": cls.RAG_EMBEDDING_MODEL,
            "dse_max_iterations": cls.DSE_MAX_ITERATIONS,
        }


# Export configuration instance
config = Config()
