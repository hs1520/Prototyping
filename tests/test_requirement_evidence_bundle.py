from __future__ import annotations

import hashlib

import pytest

from src.sitl.requirement_linker import RequirementLinker
from src.sitl.sitl_bridge import SITLBridge
from src.sysml.lite_model import build_lite_model


MODEL = """package D {
    requirement def REQ_SAFE_003 {
        doc /* Loss of the GCS link shall command landing. */
    }
    part def Controller {
        attribute gcsLinkAlive : Boolean = true;
        state def LinkBehavior {
            state Connected;
            state Landing;
            transition first Connected accept when not gcsLinkAlive then Landing;
        }
        satisfy requirement REQ_SAFE_003;
    }
}"""


def test_evidence_is_compiled_once_for_one_model_digest():
    model = build_lite_model(MODEL, model_name="D")
    linker = RequirementLinker(model)

    first = linker.compile_evidence()
    second = linker.compile_evidence()

    assert first is second
    assert first.model_digest == hashlib.sha256(MODEL.encode("utf-8")).hexdigest()
    assert first.requirement_texts["REQ_SAFE_003"].startswith("Loss of the GCS")
    assert first.satisfying_parts["REQ_SAFE_003"] == ("Controller",)


def test_evidence_maps_are_read_only_snapshots():
    evidence = RequirementLinker(
        build_lite_model(MODEL, model_name="D")
    ).compile_evidence()

    with pytest.raises(TypeError):
        evidence.requirement_texts["REQ_SAFE_003"] = "changed"
    with pytest.raises(TypeError):
        evidence.satisfying_parts["REQ_SAFE_003"] = ("Other",)
    with pytest.raises(AttributeError):
        evidence.coverage["unmapped_req_ids"].append("REQ_FAKE")


def test_bridge_cleanup_is_safe_before_gazebo_was_started(tmp_path):
    bridge = SITLBridge(
        build_lite_model(MODEL, model_name="D"),
        output_dir=str(tmp_path),
    )

    bridge._stop_gazebo()

    assert bridge._gazebo_started_by_us is False
