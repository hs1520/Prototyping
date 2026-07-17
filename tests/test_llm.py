"""Tests for the LLM interface and Chain of Thought prompting."""

from types import SimpleNamespace

import pytest
import src.llm.interface as interface_module
from src.llm.interface import (
    DEFAULT_TEMPERATURE,
    GeminiLLM,
    GitHubCopilotLLM,
    LLMInterface,
    LLMResponse,
    Message,
    MockLLM,
    TokenLedger,
    _split_gemini_messages,
)
from src.llm.chain_of_thought import ChainOfThoughtPrompter, CoTResult


class TestMessage:
    def test_creation(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_to_dict(self):
        msg = Message(role="assistant", content="Response")
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "Response"


class TestLLMResponse:
    def test_total_tokens(self):
        response = LLMResponse(content="Test", prompt_tokens=50, completion_tokens=30)
        assert response.total_tokens == 80

    def test_empty_response(self):
        response = LLMResponse(content="")
        assert response.content == ""
        assert response.total_tokens == 0


class TestMockLLM:
    def test_complete_returns_response(self):
        llm = MockLLM()
        messages = [Message(role="user", content="Design a sensor")]
        response = llm.complete(messages)
        assert isinstance(response, LLMResponse)
        assert len(response.content) > 0

    def test_inject_response(self):
        llm = MockLLM()
        llm.inject_response("Custom injected response")
        messages = [Message(role="user", content="Any question")]
        response = llm.complete(messages)
        assert response.content == "Custom injected response"

    def test_call_count(self):
        llm = MockLLM()
        messages = [Message(role="user", content="Test")]
        llm.complete(messages)
        llm.complete(messages)
        assert llm._call_count == 2

    def test_requirement_context(self):
        llm = MockLLM()
        messages = [Message(role="user", content="Extract requirements from description")]
        response = llm.complete(messages)
        assert "requirement" in response.content.lower()

    def test_sysml_context(self):
        llm = MockLLM()
        messages = [Message(role="user", content="Generate sysml block diagram")]
        response = llm.complete(messages)
        assert len(response.content) > 0

    def test_chat_interface(self):
        llm = MockLLM()
        response = llm.chat("How to design a CPS?", system_prompt="You are an expert")
        assert isinstance(response, str)
        assert len(response) > 0


class TestChainOfThoughtPrompter:
    @pytest.fixture
    def cot(self):
        llm = MockLLM()
        return ChainOfThoughtPrompter(llm)

    def test_extract_requirements(self, cot):
        result = cot.extract_requirements(
            "A drone system that delivers packages autonomously"
        )
        assert isinstance(result, CoTResult)
        assert len(result.final_answer) > 0

    def test_generate_design(self, cot):
        # Inject a response with SysML code
        cot.llm.inject_response(
            "Step 1: Design the architecture\n\n"
            "```sysml\n"
            "package DroneSystem {\n"
            "    part def FlightController {}\n"
            "}\n"
            "```\n"
        )
        result = cot.generate_design(
            system_name="Drone",
            requirements=["The drone shall fly autonomously"],
        )
        assert isinstance(result, CoTResult)
        assert result.extracted_sysml is not None
        assert "FlightController" in result.extracted_sysml

    def test_evaluate_design(self, cot):
        # Inject a response with JSON scores
        cot.llm.inject_response(
            "Evaluation:\n"
            "The design is mostly complete.\n"
            "```json\n"
            '{"completeness": 0.8, "consistency": 0.7, "performance": 0.9, "safety": 0.85, "overall": 0.81}\n'
            "```\n"
        )
        result = cot.evaluate_design(
            model_text="package Test { part def A {} }",
            requirements=["REQ-001: System shall work"],
        )
        assert isinstance(result, CoTResult)
        scores = result.get_scores()
        assert scores is not None
        assert "overall" in scores
        assert 0.0 <= scores["overall"] <= 1.0

    def test_refine_design(self, cot):
        result = cot.refine_design(
            model_text="package Test {}",
            feedback="Missing sensor components",
            issues=["No sensor defined"],
        )
        assert isinstance(result, CoTResult)
        assert len(result.final_answer) > 0

    def test_parse_cot_response_extracts_sysml(self, cot):
        text = (
            "Here is the design:\n"
            "```sysml\n"
            "package MySystem {\n"
            "    part def Component {}\n"
            "}\n"
            "```\n"
        )
        result = cot._parse_cot_response(text)
        assert result.extracted_sysml is not None
        assert "MySystem" in result.extracted_sysml

    def test_parse_cot_response_extracts_json(self, cot):
        text = (
            "Scores:\n"
            "```json\n"
            '{"completeness": 0.9, "overall": 0.85}\n'
            "```\n"
        )
        result = cot._parse_cot_response(text)
        assert result.extracted_json is not None
        assert result.extracted_json["overall"] == 0.85

    def test_self_consistency(self, cot):
        result = cot.self_consistency_generate(
            system_name="TestSystem",
            requirements=["REQ-001: System shall respond in < 100ms"],
            num_samples=2,
        )
        assert isinstance(result, CoTResult)
        assert result.metadata.get("num_candidates") == 2


class _FlakyLLM(LLMInterface):
    """Raises a configurable exception for the first N calls, then succeeds."""

    RETRY_DELAYS = (0.0, 0.0)  # no real sleeping in tests

    def __init__(self, failures: int = 1, exc: Exception = None):
        self._failures = failures
        self._exc = exc or RuntimeError("429 RESOURCE_EXHAUSTED")
        self.impl_calls = 0

    def _complete_impl(self, messages, temperature, max_tokens):
        self.impl_calls += 1
        if self.impl_calls <= self._failures:
            raise self._exc
        return LLMResponse(content="ok", prompt_tokens=10, completion_tokens=5)


class _TempSensitiveLLM(LLMInterface):
    """Returns valid JSON only at temperature >= 0.6 (escalation testing)."""

    def __init__(self):
        self.temperatures = []

    def _complete_impl(self, messages, temperature, max_tokens):
        self.temperatures.append(temperature)
        content = '{"ok": true}' if temperature >= 0.6 else "not json"
        return LLMResponse(content=content, prompt_tokens=1, completion_tokens=1)


class TestRetryAndTimeout:
    def test_transient_error_is_retried_then_succeeds(self):
        llm = _FlakyLLM(failures=2)
        response = llm.complete([Message(role="user", content="hi")])
        assert response.content == "ok"
        assert llm.impl_calls == 3
        assert llm.ledger.retries == 2
        assert llm.ledger.calls == 1

    def test_non_retryable_error_raises_immediately(self):
        llm = _FlakyLLM(failures=1, exc=ValueError("bad request: invalid schema"))
        with pytest.raises(ValueError):
            llm.complete([Message(role="user", content="hi")])
        assert llm.impl_calls == 1
        assert llm.ledger.failures == 1

    def test_retry_budget_exhaustion_reraises(self):
        llm = _FlakyLLM(failures=10)  # more failures than RETRY_DELAYS entries
        with pytest.raises(RuntimeError):
            llm.complete([Message(role="user", content="hi")])
        assert llm.impl_calls == len(llm.RETRY_DELAYS) + 1
        assert llm.ledger.failures == 1

    def test_retryable_detection_by_status_code(self):
        exc = RuntimeError("boom")
        exc.status_code = 503
        assert LLMInterface._is_retryable(exc)
        assert not LLMInterface._is_retryable(ValueError("plain bad input"))

    @pytest.mark.parametrize(
        "exc",
        [
            RuntimeError("ConnectError: [Errno 8] nodename nor servname provided"),
            RuntimeError("temporary failure in name resolution"),
        ],
    )
    def test_dns_connection_failures_are_retryable(self, exc):
        assert LLMInterface._is_retryable(exc)


class TestDefaultTemperature:
    def test_chat_defaults_to_low_temperature(self):
        llm = MockLLM()
        llm.chat("anything")
        # MockLLM records the resolved temperature in metadata
        response = llm.complete([Message(role="user", content="x")])
        assert response.metadata["temperature"] == DEFAULT_TEMPERATURE

    def test_explicit_temperature_is_respected(self):
        llm = MockLLM()
        response = llm.complete([Message(role="user", content="x")], temperature=0.9)
        assert response.metadata["temperature"] == 0.9


class TestEscalation:
    def test_escalates_until_validation_passes(self):
        llm = _TempSensitiveLLM()

        def validate(content: str) -> bool:
            import json
            json.loads(content)
            return True

        response, ok = llm.complete_with_escalation(
            [Message(role="user", content="json please")], validate
        )
        assert ok
        assert response.content == '{"ok": true}'
        # low temp first, then escalated
        assert llm.temperatures[0] == DEFAULT_TEMPERATURE
        assert llm.temperatures[-1] >= 0.6

    def test_exhaustion_returns_last_response_not_ok(self):
        llm = _TempSensitiveLLM()
        response, ok = llm.complete_with_escalation(
            [Message(role="user", content="x")],
            validate=lambda c: False,
        )
        assert not ok
        assert response is not None

    def test_chat_with_escalation_returns_content(self):
        llm = _TempSensitiveLLM()
        content, ok = llm.chat_with_escalation(
            "json please",
            validate=lambda c: c.startswith("{"),
        )
        assert ok
        assert content == '{"ok": true}'


class TestTokenLedger:
    def test_accumulates_tokens_and_calls(self):
        llm = _FlakyLLM(failures=0)
        llm.complete([Message(role="user", content="a")])
        llm.complete([Message(role="user", content="b")])
        assert llm.ledger.calls == 2
        assert llm.ledger.prompt_tokens == 20
        assert llm.ledger.completion_tokens == 10
        assert llm.ledger.total_tokens == 30

    def test_as_dict_and_summary(self):
        ledger = TokenLedger()
        ledger.record(LLMResponse(content="x", prompt_tokens=1500, completion_tokens=200),
                      elapsed=1.2, retries=1)
        d = ledger.as_dict()
        assert d["calls"] == 1
        assert d["total_tokens"] == 1700
        assert d["retries"] == 1
        text = ledger.summary()
        assert "1 calls" in text
        assert "1.5k" in text


class TestGeminiMessageSplit:
    def test_system_goes_to_instruction_and_roles_are_tagged(self):
        system_instruction, contents = _split_gemini_messages([
            Message(role="system", content="you are an expert"),
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi there"),
            Message(role="user", content="follow-up"),
        ])
        assert system_instruction == "you are an expert"
        assert [c["role"] for c in contents] == ["user", "model", "user"]
        assert contents[0]["parts"][0]["text"] == "hello"

    def test_no_system_message_yields_none(self):
        system_instruction, contents = _split_gemini_messages(
            [Message(role="user", content="hello")]
        )
        assert system_instruction is None
        assert len(contents) == 1


class TestGeminiLLM:
    def test_complete_uses_config_argument_for_generation_settings(self):
        captured_kwargs = {}

        class FakeModels:
            def generate_content(self, **kwargs):
                captured_kwargs.update(kwargs)
                return SimpleNamespace(
                    text="ok",
                    usage_metadata=None,
                    candidates=[],
                    model_version="gemini-test",
                )

        gemini = GeminiLLM.__new__(GeminiLLM)
        gemini.model = "gemini-test"
        gemini.langsmith_enabled = False
        gemini.client = SimpleNamespace(models=FakeModels())

        response = gemini.complete(
            [Message(role="user", content="hello")],
            temperature=0.25,
            max_tokens=123,
        )

        assert response.content == "ok"
        assert "temperature" not in captured_kwargs
        assert "max_tokens" not in captured_kwargs
        assert captured_kwargs["config"]["temperature"] == 0.25
        assert captured_kwargs["config"]["max_output_tokens"] == 123

    def test_system_prompt_is_sent_as_system_instruction(self):
        captured_kwargs = {}

        class FakeModels:
            def generate_content(self, **kwargs):
                captured_kwargs.update(kwargs)
                return SimpleNamespace(
                    text="ok",
                    usage_metadata=None,
                    candidates=[],
                    model_version="gemini-test",
                )

        gemini = GeminiLLM.__new__(GeminiLLM)
        gemini.model = "gemini-test"
        gemini.langsmith_enabled = False
        gemini.client = SimpleNamespace(models=FakeModels())

        gemini.complete([
            Message(role="system", content="you are an expert"),
            Message(role="user", content="hello"),
        ])

        assert captured_kwargs["config"]["system_instruction"] == "you are an expert"
        contents = captured_kwargs["contents"]
        assert len(contents) == 1
        assert contents[0]["role"] == "user"
        assert contents[0]["parts"][0]["text"] == "hello"


class TestGitHubCopilotLLMListModels:
    def test_list_models_filters_provider(self, monkeypatch):
        class FakeAuthManager:
            def get_token(self, explicit_token=None):
                return "t"

            def refresh_token(self):
                return "t2"

        class FakeResponse:
            status_code = 200

            @staticmethod
            def raise_for_status():
                return None

            @staticmethod
            def json():
                return {
                    "data": [
                        {"id": "openai/gpt-4.1-mini"},
                        {"id": "meta/llama-3.3-70b-instruct"},
                        {"id": "openai/gpt-4o-mini"},
                    ]
                }

        monkeypatch.setattr(interface_module.requests, "get", lambda *args, **kwargs: FakeResponse())

        llm = interface_module.GitHubCopilotLLM.__new__(interface_module.GitHubCopilotLLM)
        llm.auto_login = True
        llm.auth_manager = FakeAuthManager()
        llm.models_endpoint = "https://models.github.ai/catalog/models"
        llm.base_url = "https://models.github.ai/inference"
        llm._openai_cls = lambda **kwargs: SimpleNamespace(**kwargs)
        llm.client = None

        models = llm.list_models(provider="openai")
        assert models == ["openai/gpt-4.1-mini", "openai/gpt-4o-mini"]

    def test_list_models_retries_once_on_401(self, monkeypatch):
        class FakeAuthManager:
            def __init__(self):
                self.refresh_count = 0

            def get_token(self, explicit_token=None):
                return "t"

            def refresh_token(self):
                self.refresh_count += 1
                return "t2"

        class FakeUnauthorizedResponse:
            status_code = 401

            @staticmethod
            def raise_for_status():
                raise RuntimeError("unauthorized")

            @staticmethod
            def json():
                return {}

        class FakeOkResponse:
            status_code = 200

            @staticmethod
            def raise_for_status():
                return None

            @staticmethod
            def json():
                return {"data": [{"id": "openai/gpt-4.1-mini"}]}

        responses = [FakeUnauthorizedResponse(), FakeOkResponse()]

        def fake_get(*args, **kwargs):
            return responses.pop(0)

        monkeypatch.setattr(interface_module.requests, "get", fake_get)

        auth_manager = FakeAuthManager()
        llm = interface_module.GitHubCopilotLLM.__new__(interface_module.GitHubCopilotLLM)
        llm.auto_login = True
        llm.auth_manager = auth_manager
        llm.models_endpoint = "https://models.github.ai/catalog/models"
        llm.base_url = "https://models.github.ai/inference"
        llm._openai_cls = lambda **kwargs: SimpleNamespace(**kwargs)
        llm.client = None

        models = llm.list_models()
        assert models == ["openai/gpt-4.1-mini"]
        assert auth_manager.refresh_count == 1
