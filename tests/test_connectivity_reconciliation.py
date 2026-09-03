from __future__ import annotations

import pytest

from src.simulation.connectivity_reconciliation import (
    ConnectivityProposalError,
    reconcile_connectivity,
)


MODEL = """package Drone {
    port def DataPort { inout item d : DataItem; }
    port def PowerPort { inout item p : PowerItem; }
    part def Controller { in port dataIn : DataPort; }
    part def Sensor { out port dataOut : DataPort; }
    part def Power { out port powerOut : PowerPort; }
    part def System {
        part controller : Controller;
        part sensor : Sensor;
        part power : Power;
    }
}
"""


def _reconcile(propose):
    return reconcile_connectivity(
        MODEL,
        failed_scenarios=[
            {"name": "power_to_controller", "src": "power", "tgts": ["controller"]}
        ],
        propose=propose,
        connectivity_system_prompt="connect-system",
        port_system_prompt="port-system",
    )


def test_valid_connection_merged():
    calls: list[str] = []

    def propose(_prompt: str, system_prompt: str) -> str:
        calls.append(system_prompt)
        return "connect sensor.dataOut to controller.dataIn;"

    result = _reconcile(propose)

    assert result.disposition == "CONNECTIONS_PROPOSED"
    assert len(result.added_connections) == 1
    assert not result.added_ports
    assert calls == ["connect-system"]
    assert "connect sensor.dataOut to controller.dataIn;" in result.model_text


def test_missing_port_added():
    replies = iter(
        [
            "connect power.powerOut to controller.powerIn;",
            "Controller: in port powerIn : PowerPort;",
            "connect power.powerOut to controller.powerIn;",
        ]
    )

    result = _reconcile(lambda _prompt, _system: next(replies))

    assert result.disposition == "PORTS_AND_CONNECTIONS_PROPOSED"
    assert [item.name for item in result.added_ports] == ["powerIn"]
    assert len(result.added_connections) == 1
    assert "in port powerIn : PowerPort;" in result.model_text
    assert "connect power.powerOut to controller.powerIn;" in result.model_text
    assert any(item.code == "CONNECTION_REJECTED" for item in result.diagnostics)


def test_invalid_port_rejected():
    replies = iter(
        [
            "connect power.powerOut to controller.powerIn;",
            "UnknownPart: in port powerIn : PowerPort;",
        ]
    )

    result = _reconcile(lambda _prompt, _system: next(replies))

    assert result.disposition == "NO_VALID_PROPOSAL"
    assert result.model_text == MODEL
    assert any(item.code == "PORT_REJECTED" for item in result.diagnostics)


def test_dependency_failure_raises():
    replies = iter(
        [
            "connect power.powerOut to controller.powerIn;",
            "Controller: in port powerIn : PowerPort;",
        ]
    )

    def propose(_prompt: str, _system: str) -> str:
        try:
            return next(replies)
        except StopIteration as exc:
            raise RuntimeError("provider unavailable") from exc

    with pytest.raises(ConnectivityProposalError) as caught:
        _reconcile(propose)

    partial = caught.value.partial
    assert partial.disposition == "PORTS_ONLY_DEPENDENCY_FAILED"
    assert [item.name for item in partial.added_ports] == ["powerIn"]
    assert "in port powerIn : PowerPort;" in partial.model_text


@pytest.mark.parametrize("error", [TypeError("bug"), AssertionError("bug")])
def test_programming_errors_propagate(error):
    def propose(_prompt: str, _system: str) -> str:
        raise error

    with pytest.raises(type(error), match="bug"):
        _reconcile(propose)
