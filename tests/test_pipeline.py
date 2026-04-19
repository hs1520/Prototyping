"""Tests for provider-agnostic LLM creation in pipeline."""

from src.llm.interface import LLMResponse, MockLLM
from src.prototyping import pipeline as pipeline_module


class DummyLLM:
    """Simple test double used to validate provider registration and kwargs."""

    def __init__(self, model: str = "dummy-model", api_key: str | None = None, **kwargs):
        self.model = model
        self.api_key = api_key
        self.extra = kwargs

    def complete(self, messages, temperature: float = 0.7, max_tokens: int = 2048):
        return LLMResponse(content="ok", model=self.model)


class StrictDummyLLM:
    """Constructor without model/api_key used to test kwargs filtering."""

    def __init__(self, timeout: int = 5):
        self.timeout = timeout

    def complete(self, messages, temperature: float = 0.7, max_tokens: int = 2048):
        return LLMResponse(content="ok", model="strict")


def test_create_llm_defaults_to_mock():
    llm = pipeline_module.create_llm()
    assert isinstance(llm, MockLLM)


def test_create_llm_rejects_unknown_provider():
    try:
        pipeline_module.create_llm(provider="not-a-provider")
        assert False, "Expected ValueError for unknown provider"
    except ValueError as exc:
        assert "Unknown LLM provider" in str(exc)


def test_register_custom_provider_and_create(monkeypatch):
    monkeypatch.setitem(pipeline_module.LLM_PROVIDER_FACTORIES, "dummy", DummyLLM)

    llm = pipeline_module.create_llm(
        provider="dummy",
        model="dummy-v1",
        api_key="k-test",
        provider_kwargs={"region": "us-central1"},
    )

    assert isinstance(llm, DummyLLM)
    assert llm.model == "dummy-v1"
    assert llm.api_key == "k-test"
    assert llm.extra["region"] == "us-central1"


def test_provider_aliases_keep_only_default_and_test(monkeypatch):
    monkeypatch.setitem(pipeline_module.LLM_PROVIDER_FACTORIES, "mock", DummyLLM)

    llm_default = pipeline_module.create_llm(provider="default", model="default-model")
    llm_test = pipeline_module.create_llm(provider="test", model="test-model")

    assert isinstance(llm_default, DummyLLM)
    assert llm_default.model == "default-model"
    assert isinstance(llm_test, DummyLLM)
    assert llm_test.model == "test-model"


def test_constructor_kwargs_are_filtered(monkeypatch):
	monkeypatch.setitem(pipeline_module.LLM_PROVIDER_FACTORIES, "strict", StrictDummyLLM)

	llm = pipeline_module.create_llm(
		provider="strict",
		model="should-not-be-passed",
		api_key="should-not-be-passed",
		provider_kwargs={"timeout": 42},
	)

	assert isinstance(llm, StrictDummyLLM)
	assert llm.timeout == 42


def test_vertex_uses_default_model_when_not_provided(monkeypatch):
	monkeypatch.setitem(pipeline_module.LLM_PROVIDER_FACTORIES, "vertex", DummyLLM)

	llm = pipeline_module.create_llm(provider="vertex")

	assert isinstance(llm, DummyLLM)
	assert llm.model == "gemini-3-pro-preview"


def test_vertex_accepts_claude_model_passthrough(monkeypatch):
	monkeypatch.setitem(pipeline_module.LLM_PROVIDER_FACTORIES, "vertex", DummyLLM)

	llm = pipeline_module.create_llm(provider="vertex", model="claude-3-7-sonnet")

	assert isinstance(llm, DummyLLM)
	assert llm.model == "claude-3-7-sonnet"


