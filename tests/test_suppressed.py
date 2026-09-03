from __future__ import annotations

from types import SimpleNamespace

from src.dse.variation_dse import run_variation_dse
from src.app.pipeline import PrototypingPipeline
from src.utils.suppressed import record_suppressed, reset_suppressed, suppressed_summary


class _UnprintableError(Exception):
    def __str__(self) -> str:
        raise RuntimeError("str failed")


def test_summary_counts_reset():
    reset_suppressed()
    record_suppressed("unit.point", ValueError("first"))
    record_suppressed("unit.point", TypeError("second"))

    summary = suppressed_summary()
    assert summary["unit.point"]["count"] == 2
    assert summary["unit.point"]["last"] == "TypeError: second"

    reset_suppressed()
    assert suppressed_summary() == {}


def test_unprintable_exception_recorded():
    reset_suppressed()
    record_suppressed("unit.unprintable", _UnprintableError())
    summary = suppressed_summary()
    assert summary["unit.unprintable"]["count"] == 1
    assert summary["unit.unprintable"]["last"] == "_UnprintableError: <unprintable exception>"


_VARIATION_MODEL = """package Drone {
    port def Sig;
    part def LiftIface { in port cmd : Sig; out port thrust : Sig; }
    part def QuadRotor :> LiftIface {
        in port cmd : Sig; out port thrust : Sig;
        attribute rotorCount : Real = 4.0;
        attribute batteryCells : Real = 4.0;
        attribute rotorRadiusM : Real = 0.1905;
    }
    part def HexaRotor :> LiftIface {
        in port cmd : Sig; out port thrust : Sig;
        attribute rotorCount : Real = 6.0;
        attribute batteryCells : Real = 6.0;
        attribute rotorRadiusM : Real = 0.2286;
    }
    part def Airframe {
        variation part liftArch : LiftIface {
            doc /* rationale: endurance vs catalogue availability; satisfies REQ-PERF-002 */
            variant part quad : QuadRotor;
            variant part hexa : HexaRotor;
        }
    }
}"""


def _model(text: str = _VARIATION_MODEL):
    return SimpleNamespace(metadata={"last_sysml_text": text})


def _front_with_two_members(self, iterations):
    return SimpleNamespace(members=[
        ({"liftArch": "quad"}, {"time_sat": 1.0, "cost_efficiency": 1.0}),
        ({"liftArch": "hexa"}, {"time_sat": 0.9, "cost_efficiency": 0.9}),
    ])


def test_dse_callback_failure_reported(monkeypatch):
    reset_suppressed()
    monkeypatch.setattr(
        "src.dse.variation_dse.MultiObjectiveMCTS.search",
        _front_with_two_members,
    )

    def _bad_realizability(_di):
        raise RuntimeError("catalog unavailable")

    res = run_variation_dse(
        _model(),
        requirements=["REQ-PERF-002: endurance at least 20 minutes."],
        realizability=_bad_realizability,
    )

    assert res is not None
    report = PrototypingPipeline.build_run_report({
        "system_name": "SuppressedDemo",
        "requirements": ["REQ-PERF-002: endurance at least 20 minutes."],
    })
    assert report["suppressed"]["variation_dse.realizability"]["count"] == 2
    assert report["suppressed"]["variation_dse.realizability"]["last"] == (
        "RuntimeError: catalog unavailable"
    )
    reset_suppressed()
