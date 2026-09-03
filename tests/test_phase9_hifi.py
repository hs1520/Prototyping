from __future__ import annotations


import src.agents.phase9_hifi as phase9
from src.agents.orchestrator import Orchestrator
from src.dse.physics_estimator import DesignInputs


def _design():
    return DesignInputs(1.5, 16000, 6, 4, 18 * 0.0254 / 2, 0.0)


def test_mode_maps_to_layers():
    assert phase9._layers_for("sitl") == ["sitl"]
    assert phase9._layers_for("gazebo") == ["gazebo"]
    assert phase9._layers_for("both") == ["gazebo", "sitl"]
    assert phase9._layers_for("nonsense") == []
    assert phase9._layers_for(None) == []


def test_env_absence_skips_run(monkeypatch):
    monkeypatch.setattr(phase9, "_env_available", lambda layer: "arducopter SITL binary not found")
    called = {"subprocess": False}
    monkeypatch.setattr(phase9, "_run_subprocess",
                        lambda *a, **k: called.__setitem__("subprocess", True) or {})
    out = phase9.run_layer("sitl", _design(), "package D {}")
    assert out["status"] == "skipped"
    assert "not found" in out["reason"]
    assert called["subprocess"] is False


def test_run_layer_sitl_parses_report(monkeypatch, tmp_path):
    monkeypatch.setattr(phase9, "_env_available", lambda layer: None)
    monkeypatch.setattr(phase9, "_run_subprocess", lambda *a, **k: {"returncode": 0, "stdout": "", "stderr": ""})
    (tmp_path / "sitl_feasibility_report.json").write_text(
        '{"flight": {"passed": true}, "safety_verification": {"status": "PASS"}}',
        encoding="utf-8",
    )
    out = phase9.run_layer("sitl", _design(), "package D {}", output_dir=tmp_path)
    assert out["status"] == "ran"
    assert out["flight_passed"] is True
    assert out["safety_status"] == "PASS"
    assert "not endurance" in out["redline"]


def test_nonzero_exit_with_report_ran(monkeypatch, tmp_path):
    # gazebo exit 2 = a requirement failed but the run completed; a parseable
    # report is a result, not an "error".
    monkeypatch.setattr(phase9, "_env_available", lambda layer: None)
    monkeypatch.setattr(phase9, "_run_subprocess",
                        lambda *a, **k: {"returncode": 2, "stdout": "", "stderr": ""})
    (tmp_path / "gazebo_feasibility_report.json").write_text('{"status": "FAIL"}', encoding="utf-8")
    out = phase9.run_layer("gazebo", _design(), "package D {}", output_dir=tmp_path)
    assert out["status"] == "ran"
    assert out["gazebo_status"] == "FAIL"
    assert out["returncode"] == 2


def test_run_layer_no_report_is_error(monkeypatch, tmp_path):
    monkeypatch.setattr(phase9, "_env_available", lambda layer: None)
    monkeypatch.setattr(phase9, "_run_subprocess",
                        lambda *a, **k: {"returncode": 1, "stdout": "", "stderr": "boom"})
    out = phase9.run_layer("gazebo", _design(), "package D {}", output_dir=tmp_path)
    assert out["status"] == "error"
    assert out["gazebo_status"] is None
    assert (tmp_path / "final_model.sysml").exists()
    import json
    rj = json.loads((tmp_path / "realization_run.json").read_text())
    assert rj["recommended_design_inputs"]["rotor_count"] == 4


def test_none_mode_no_runner(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr("src.agents.phase9_hifi.run_hifi_closure",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1) or {})
    art = Orchestrator._phase9_hifi_artifact(None, _design(), "package D {}", [])
    assert art is None
    assert called["n"] == 0


def test_artifact_carries_redline(monkeypatch):
    monkeypatch.setattr(
        "src.agents.phase9_hifi.run_hifi_closure",
        lambda mode, design, model_text, requirements: {
            "mode": mode,
            "layers": [
                {"layer": "sitl", "status": "ran", "flight_passed": True, "safety_status": "PASS"},
                {"layer": "gazebo", "status": "skipped", "reason": "docker not available"},
            ],
        },
    )
    art = Orchestrator._phase9_hifi_artifact("both", _design(), "package D {}", [])
    assert art is not None
    assert "datasheet CLOSED unchanged" in art["summary"]
    assert "endurance never validated" in art["summary"]
    assert "sitl(flight=True, safety=PASS)" in art["summary"]
    assert "gazebo=skipped(docker not available)" in art["summary"]


def test_best_effort(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("subprocess exploded")

    monkeypatch.setattr("src.agents.phase9_hifi.run_hifi_closure", boom)
    art = Orchestrator._phase9_hifi_artifact("sitl", _design(), "package D {}", [])
    assert art is None


def test_rejects_unknown_mode():
    assert Orchestrator._phase9_hifi_artifact("teleport", _design(), "package D {}", []) is None


def test_defaults_off_overridable():
    assert Orchestrator(llm=object()).phase9_hifi is None
    assert Orchestrator(llm=object(), phase9_hifi="sitl").phase9_hifi == "sitl"
    assert Orchestrator(llm=object(), phase9_hifi="both").phase9_hifi == "both"
