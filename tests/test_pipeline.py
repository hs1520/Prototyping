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


def test_provider_alias_works_with_legacy_use_openai(monkeypatch):
	monkeypatch.setitem(pipeline_module.LLM_PROVIDER_FACTORIES, "openai", DummyLLM)

	llm = pipeline_module.create_llm(use_openai=True, model="compat-model")
	assert isinstance(llm, DummyLLM)
	assert llm.model == "compat-model"

	llm_alias = pipeline_module.create_llm(provider="open_ai", model="alias-model")
	assert isinstance(llm_alias, DummyLLM)
	assert llm_alias.model == "alias-model"


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

