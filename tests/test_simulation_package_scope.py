"""System simulation must not flatten auxiliary A/G packages into the system."""
from __future__ import annotations

import pytest

from src.simulation import extractor
from src.simulation.validator import SimulationValidator


_MODEL_WITH_SHADOW_PACKAGE = """package DeliveryUAV {
    port def Signal;
    part def Producer { out port status : Signal; }
    part def Consumer { in port status : Signal; }
    part producer : Producer;
    part consumer : Consumer;
    connect producer.status to consumer.status;
}

package REQ_SAFE_001_AG {
    part def Producer;
    part producer : Producer;
}
"""


@pytest.mark.skipif(
    not extractor._SYSIDE_OK,
    reason="Syside is required for namespace-aware extraction",
)
def test_root_package_scope_excludes_shadow_parts_and_preserves_real_ports():
    graph = extractor.extract_behavioral_graph(
        _MODEL_WITH_SHADOW_PACKAGE,
        root_package="DeliveryUAV",
    )

    assert set(graph.parts) == {"producer", "consumer"}
    assert set(graph.ports) == {"producer.status", "consumer.status"}
    assert len(graph.connections) == 1


@pytest.mark.skipif(
    not extractor._SYSIDE_OK,
    reason="Syside is required for namespace-aware extraction",
)
def test_validator_uses_model_name_as_package_scope_when_that_package_exists():
    result = SimulationValidator().validate(
        _MODEL_WITH_SHADOW_PACKAGE,
        model_name="DeliveryUAV",
    )

    assert result.num_parts == 2
    assert result.num_ports == 2
    assert result.num_connections == 1
