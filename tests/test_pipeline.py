from src.llm.interface import LLMResponse, MockLLM
from src.app import pipeline as orchestration_module
from src.prototyping import provider_factory as provider_module
from src.sitl.sitl_bridge import BridgeReport, TestResult as SITLTestResult


class DummyLLM:
    def __init__(self, model: str = "dummy-model", api_key: str | None = None, **kwargs):
        self.model = model
        self.api_key = api_key
        self.extra = kwargs

    def complete(self, messages, temperature: float = 0.7, max_tokens: int = 2048):
        return LLMResponse(content="ok", model=self.model)


class StrictDummyLLM:
    def __init__(self, timeout: int = 5):
        self.timeout = timeout

    def complete(self, messages, temperature: float = 0.7, max_tokens: int = 2048):
        return LLMResponse(content="ok", model="strict")


def test_create_llm_defaults_to_mock():
    llm = provider_module.create_llm()
    assert isinstance(llm, MockLLM)


def test_create_llm_rejects_unknown():
    try:
        provider_module.create_llm(provider="not-a-provider")
        assert False, "Expected ValueError for unknown provider"
    except ValueError as exc:
        assert "Unknown LLM provider" in str(exc)


def test_register_custom_provider(monkeypatch):
    monkeypatch.setitem(provider_module.LLM_PROVIDER_FACTORIES, "dummy", DummyLLM)

    llm = provider_module.create_llm(
        provider="dummy",
        model="dummy-v1",
        api_key="k-test",
        provider_kwargs={"region": "us-central1"},
    )

    assert isinstance(llm, DummyLLM)
    assert llm.model == "dummy-v1"
    assert llm.api_key == "k-test"
    assert llm.extra["region"] == "us-central1"


def test_provider_aliases(monkeypatch):
    monkeypatch.setitem(provider_module.LLM_PROVIDER_FACTORIES, "mock", DummyLLM)

    llm_default = provider_module.create_llm(provider="default", model="default-model")
    llm_test = provider_module.create_llm(provider="test", model="test-model")

    assert isinstance(llm_default, DummyLLM)
    assert llm_default.model == "default-model"
    assert isinstance(llm_test, DummyLLM)
    assert llm_test.model == "test-model"


def test_constructor_kwargs_filtered(monkeypatch):
    monkeypatch.setitem(provider_module.LLM_PROVIDER_FACTORIES, "strict", StrictDummyLLM)

    llm = provider_module.create_llm(
        provider="strict",
        model="should-not-be-passed",
        api_key="should-not-be-passed",
        provider_kwargs={"timeout": 42},
    )

    assert isinstance(llm, StrictDummyLLM)
    assert llm.timeout == 42


def test_vertex_default_model(monkeypatch):
    monkeypatch.setitem(provider_module.LLM_PROVIDER_FACTORIES, "vertex", DummyLLM)

    llm = provider_module.create_llm(provider="vertex")

    assert isinstance(llm, DummyLLM)
    assert llm.model == "gemini-3.1-pro-preview"


def test_vertex_model_passthrough(monkeypatch):
    monkeypatch.setitem(provider_module.LLM_PROVIDER_FACTORIES, "vertex", DummyLLM)

    llm = provider_module.create_llm(provider="vertex", model="claude-3-7-sonnet")

    assert isinstance(llm, DummyLLM)
    assert llm.model == "claude-3-7-sonnet"


class DummyPineconeWrapper:
    def __init__(self, default_namespace: str = "ns"):
        self.default_namespace = default_namespace

    def search(self, *args, **kwargs):
        return {"matches": []}


def test_pipeline_uses_injected_llm():
    llm = DummyLLM()
    pipeline = orchestration_module.PrototypingPipeline(
        llm=llm,
        pinecone_wrapper=DummyPineconeWrapper(),
    )

    assert pipeline.llm is llm
    assert pipeline.pinecone.default_namespace == "ns"
    assert pipeline.orchestrator is not None
    assert pipeline.orchestrator.design_agent is not None


def test_pipeline_without_pinecone(monkeypatch):
    class ExplodingPinecone:
        def __init__(self, *a, **kw):
            raise ValueError("Pinecone API key is required.")

    monkeypatch.setattr(orchestration_module, "PineconeWrapper", ExplodingPinecone)
    pipeline = orchestration_module.PrototypingPipeline(llm=DummyLLM())
    assert pipeline.rag is None
    assert pipeline.pinecone is None
    assert pipeline.orchestrator is not None


def test_save_run_report_writes_json(tmp_path):
    pipeline = orchestration_module.PrototypingPipeline(
        llm=DummyLLM(), pinecone_wrapper=DummyPineconeWrapper()
    )
    path = pipeline.save_run_report(
        {"system_name": "T", "final_score": 0.9}, directory=str(tmp_path)
    )
    assert path is not None
    import json
    saved = json.loads((tmp_path / path.split("/")[-1]).read_text())
    assert saved["system_name"] == "T"


def test_run_report_serialisable():
    import json

    report = orchestration_module.PrototypingPipeline.build_run_report({
        "system_name": "T",
        "final_score": 0.9,
        "iterations": 2,
        "requirements": ["REQ-1", "REQ-2"],
        "requirement_input": {
            "mode": "frozen", "frozen": True, "requirement_set_digest": "abc"
        },
        "evaluation_history": [{"iteration": 1, "score": 0.9}],
        "best_config": {"redundancy_level": "dual"},
        "llm_usage": {"calls": 3, "total_tokens": 100},
    })
    assert report["system_name"] == "T"
    assert report["requirements_count"] == 2
    assert report["requirements"] == ["REQ-1", "REQ-2"]
    assert report["requirement_input"]["mode"] == "frozen"
    json.dumps(report)


def test_run_report_evidence_digest():
    class Sim:
        reachability_score = 0.5
        scenario_results = ()

        @staticmethod
        def passed_scenarios():
            return []

    consistency = {
        "status": "PASS",
        "final_score": 0.7,
        "simulation": {
            "reachability_score": 0.5,
            "scenarios_passed": 0,
            "scenarios_total": 0,
        },
    }
    report = orchestration_module.PrototypingPipeline.build_run_report({
        "requirements": [],
        "final_score": 0.7,
        "simulation_result": Sim(),
        "terminal_consistency": consistency,
    })

    assert report["terminal_consistency"] == consistency
    assert report["simulation"]["scenarios_total"] == 0


def test_run_report_collaboration():
    revised = {
        "experiment_namespace": "BLACKBOARD_AG_V1",
        "configuration": "R1-BBCTX",
    }
    collaboration = {
        "blackboard": {"current_model": {"revision": 2}},
        "contexts": {"envelopes": []},
        "task_sessions": {"sessions": []},
    }
    report = orchestration_module.PrototypingPipeline.build_run_report({
        "requirements": [],
        "revised_experiment": revised,
        "collaboration": collaboration,
    })
    assert report["experiment_namespace"] == "BLACKBOARD_AG_V1"
    assert report["configuration"] == "R1-BBCTX"
    assert report["collaboration"] == collaboration


def test_run_report_sitl_traceability():
    report = orchestration_module.PrototypingPipeline.build_run_report({
        "system_name": "T",
        "requirements": [],
        "sitl_report": BridgeReport(
            model_name="T",
            parm_file="T.parm",
            l1_results=[SITLTestResult("REQ_L1", "L1", True, "ok")],
            l2_results=[SITLTestResult("REQ_SAFE_005", "L2", False, "servo8_raw last=1000")],
            trace_results=[
                SITLTestResult(
                    "REQ_SAFE_003",
                    "TRACE",
                    False,
                    "traceability mismatch: requirement text implies GCS",
                )
            ],
        ),
    })

    assert report["sitl"]["l1_passed"] == 1
    assert report["sitl"]["safety_status"] == "PARTIAL"
    assert report["sitl"]["l2_passed"] == 0
    assert report["sitl"]["l2_total"] == 1
    assert report["sitl"]["traceability_blocked"] == 1
    assert report["sitl"]["traceability"][0]["req_id"] == "REQ_SAFE_003"


def test_run_report_realization():
    report = orchestration_module.PrototypingPipeline.build_run_report({
        "system_name": "T",
        "requirements": [],
        "recommended_by": "datasheet",
        "recommended_estimator_feasible": True,
        "variation_proposal_source": "llm",
        "realization": {
            "verdict": "CLOSED",
            "summary": "MEET-IN-THE-MIDDLE CLOSED — ...",
            "chosen": {"combo": "c", "pack": "p", "frame": "f"},
            "forward_flight_ok": True,
            "rank_preservation": {"n": 2.0},
            "resize_note": "",
            "per_requirement": [{"req_id": "REQ-PERF-002"}],
        },
    })

    assert report["realization"]["verdict"] == "CLOSED"
    assert report["realization"]["chosen"]["combo"] == "c"
    # per_requirement is the datasheet/forward-flight tier input and survives into
    # the archived report: the summary-only projection dropped it and the matrix
    # reported the missing input as "unassigned" requirements (run3, A2).
    assert report["realization"]["per_requirement"] == [
        {"req_id": "REQ-PERF-002"}
    ]
    assert report["recommended_by"] == "datasheet"
    assert report["recommended_estimator_feasible"] is True
    assert report["variation_proposal_source"] == "llm"
    import json
    json.dumps(report)
