"""Tests for the LLM interface and Chain of Thought prompting."""

import time
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
    VertexLLM,
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

    def test_call_observer_archives_exact_messages_and_can_be_removed(self):
        llm = MockLLM()
        events = []
        observer_id = llm.add_call_observer(events.append)
        llm.chat("hello", system_prompt="system")
        assert len(events) == 1
        assert events[0]["messages"] == [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "hello"},
        ]
        assert events[0]["response"]["role"] == "assistant"
        llm.remove_call_observer(observer_id)
        llm.chat("not archived")
        assert len(events) == 1

    def test_call_observer_failure_does_not_retry_provider_call(self):
        llm = MockLLM()

        def fail(_event):
            raise RuntimeError("archive failed")

        llm.add_call_observer(fail)
        response = llm.chat("one provider call")
        assert response
        assert llm._call_count == 1
        assert llm.call_observer_errors == ("RuntimeError: archive failed",)


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

    def test_max_call_seconds_measures_one_attempt_not_the_retry_chain(self):
        """A cancelled call is only diagnosable against the client deadline if
        the recorded duration is one request, so retries and their sleeps must
        not accumulate into it."""
        import time

        class _SlowThenFast(LLMInterface):
            RETRY_DELAYS = (0.0, 0.0)

            def __init__(self):
                self.impl_calls = 0

            def _complete_impl(self, messages, temperature, max_tokens):
                self.impl_calls += 1
                time.sleep(0.05 if self.impl_calls == 1 else 0.0)
                if self.impl_calls == 1:
                    raise RuntimeError("429 RESOURCE_EXHAUSTED")
                return LLMResponse(content="ok", prompt_tokens=1, completion_tokens=1)

        llm = _SlowThenFast()
        llm.complete([Message(role="user", content="hi")])
        usage = llm.ledger.as_dict()
        assert usage["max_call_seconds"] >= 0.05
        assert usage["max_call_seconds"] <= usage["elapsed_seconds"]

    def test_a_failed_call_still_records_how_long_it_ran(self):
        """The archived failure context stores the provider's text and nothing
        else; without this the run cannot say whether the deadline was hit."""
        llm = _FlakyLLM(failures=10)
        with pytest.raises(RuntimeError):
            llm.complete([Message(role="user", content="hi")])
        assert llm.ledger.as_dict()["max_call_seconds"] >= 0.0
        assert llm.ledger.failures == 1

    def test_the_backoff_ladder_outlasts_a_quota_window(self):
        """Three of eighteen paired runs were lost to 429 on 2026-08-11 while
        429 was already retryable: the ladder gave up 62 s after the first
        refusal."""
        assert sum(LLMInterface.RETRY_DELAYS) >= 300

    def test_a_cancelled_vertex_request_is_retried(self):
        """The exact text a 2026-08-11 pilot failed on, twice.

        The provider error reaches the classifier as a plain RuntimeError with
        the status only in its message, so adding 499 to _RETRYABLE_STATUS
        would have been dead code.
        """
        from src.llm.interface import VertexLLM

        exc = RuntimeError(
            "Vertex provider error (ClientError): 499 CANCELLED. "
            "{'error': {'code': 499, 'message': 'The operation was "
            "cancelled.', 'status': 'CANCELLED'}} status_code=None code=499"
        )
        assert VertexLLM._is_retryable(exc) is True

    def test_this_clients_own_deadline_is_still_not_retried(self):
        """Retrying a request our own wall clock killed meets the same wall
        clock again, which is why timeouts are excluded."""
        from src.llm.interface import VertexLLM

        exc = TimeoutError(
            "Vertex request exceeded hard wall-clock timeout of 600 seconds"
        )
        assert VertexLLM._is_retryable(exc) is False

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

    def test_generation_seed_is_forwarded_in_sdk_config(self):
        captured_kwargs = {}

        class FakeModels:
            def generate_content(self, **kwargs):
                captured_kwargs.update(kwargs)
                return SimpleNamespace(
                    text="ok", usage_metadata=None, candidates=[],
                    model_version="gemini-test",
                )

        gemini = GeminiLLM.__new__(GeminiLLM)
        gemini.model = "gemini-test"
        gemini.seed = 1234
        gemini.langsmith_enabled = False
        gemini.client = SimpleNamespace(models=FakeModels())

        response = gemini.complete([Message(role="user", content="hello")])

        assert captured_kwargs["config"]["seed"] == 1234
        assert response.metadata["seed"] == 1234


class TestVertexLLM:
    def test_sdk_retry_is_disabled_and_timeout_is_forwarded(
        self, monkeypatch
    ):
        from google import genai

        captured = {}
        monkeypatch.setattr(
            genai,
            "Client",
            lambda **kwargs: captured.update(kwargs) or SimpleNamespace(),
        )
        monkeypatch.setattr(
            interface_module.Config, "setup_langsmith_env", lambda: None
        )
        monkeypatch.setattr(
            interface_module.Config,
            "get_vertex_api_key",
            lambda: "test-key",
        )

        vertex = VertexLLM(enable_langsmith=False, timeout_seconds=12.5)

        assert captured["http_options"] == {
            "timeout": 12500,
            "retry_options": {"attempts": 1},
        }
        assert vertex._process_isolation is True

    def test_generation_seed_is_forwarded_in_sdk_config(self):
        captured_kwargs = {}

        class FakeModels:
            def generate_content(self, **kwargs):
                captured_kwargs.update(kwargs)
                return SimpleNamespace(
                    text="ok", usage_metadata=None, model_version="vertex-test"
                )

        vertex = VertexLLM.__new__(VertexLLM)
        vertex.model = "vertex-test"
        vertex.seed = 4321
        vertex.langsmith_enabled = False
        vertex.client = SimpleNamespace(models=FakeModels())

        response = vertex.complete([Message(role="user", content="hello")])

        assert captured_kwargs["config"]["seed"] == 4321
        assert captured_kwargs["config"]["thinking_config"] == {
            "thinking_level": "HIGH"
        }
        assert response.metadata["seed"] == 4321
        assert response.metadata["thinking_level"] == "HIGH"

    def test_deadline_is_not_retried_but_capacity_failure_is(self):
        assert not VertexLLM._is_retryable(
            RuntimeError("504 DEADLINE_EXCEEDED")
        )
        assert not VertexLLM._is_retryable(RuntimeError("ReadTimeout"))
        assert not VertexLLM._is_retryable(RuntimeError("ConnectTimeout"))
        assert VertexLLM._is_retryable(RuntimeError("429 RESOURCE_EXHAUSTED"))

    def test_vertex_retry_count_is_bounded_to_two(self):
        assert VertexLLM.RETRY_DELAYS == (10.0, 30.0)
        assert VertexLLM.ARCHITECTURE_MAX_TOKENS == 65536

    def test_vertex_default_timeout_is_ten_minutes(self, monkeypatch):
        monkeypatch.delenv("LLM_TIMEOUT_SECONDS", raising=False)
        assert interface_module._default_timeout_seconds(600.0) == 600.0

    def test_hard_timeout_interrupts_blocked_provider_call(self):
        class BlockingModels:
            def generate_content(self, **kwargs):
                time.sleep(5.0)

        models = BlockingModels()
        vertex = VertexLLM.__new__(VertexLLM)
        vertex.model = "vertex-test"
        vertex.seed = 0
        vertex.thinking_level = "HIGH"
        vertex.timeout_seconds = 0.05
        vertex.langsmith_enabled = False
        vertex._process_isolation = True
        vertex.client = SimpleNamespace(models=models)
        started = time.monotonic()

        with pytest.raises(
            TimeoutError, match="hard wall-clock timeout"
        ):
            vertex.complete([Message(role="user", content="hello")])

        assert time.monotonic() - started < 0.5
        assert vertex.ledger.failures == 1

    def test_process_isolated_request_returns_response(self):
        class Models:
            def generate_content(self, **kwargs):
                return SimpleNamespace(
                    text="ok",
                    model_version="vertex-child",
                    usage_metadata=SimpleNamespace(
                        prompt_token_count=3,
                        candidates_token_count=7,
                    ),
                )

        vertex = VertexLLM.__new__(VertexLLM)
        vertex.model = "vertex-test"
        vertex.seed = 0
        vertex.thinking_level = "HIGH"
        vertex.timeout_seconds = 2.0
        vertex.langsmith_enabled = False
        vertex._process_isolation = True
        vertex.client = SimpleNamespace(models=Models())

        response = vertex.complete([
            Message(role="user", content="hello")
        ])

        assert response.content == "ok"
        assert response.model == "vertex-child"
        assert response.prompt_tokens == 3
        assert response.completion_tokens == 7


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
