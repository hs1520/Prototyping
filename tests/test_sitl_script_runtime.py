from pathlib import Path

from src.sitl.requirement_linker import SITLTestSpec
from src.sitl.script_runtime import render_pymavlink_runtime
from src.sitl.sitl_bridge import SITLBridge
from src.sitl.sitl_specs import InjectSpec, VerifySpec


def _bridge(tmp_path: Path) -> SITLBridge:
    bridge = object.__new__(SITLBridge)
    bridge._connection_string = "tcp:127.0.0.1:5760"
    bridge._output_dir = tmp_path
    return bridge


def test_shared_pymavlink_runtime_is_valid_python():
    runtime = render_pymavlink_runtime("tcp:127.0.0.1:5760")

    compile(runtime, "<sitl-runtime>", "exec")
    assert runtime.count("def force_arm_and_takeoff(") == 1
    assert "def wait_mode(" in runtime


def test_both_generated_script_families_use_one_runtime(tmp_path):
    bridge = _bridge(tmp_path)
    generic = bridge._render_test_script(SITLTestSpec(
        req_id="REQ_SAFE_001",
        tier="L2",
        inject=InjectSpec(kind="noop"),
        verify=VerifySpec(kind="noop"),
        notes="test",
    ))
    accept = bridge._render_accept_mode_script("CMD_LAND", "LAND", True)

    compile(generic, "<generic-sitl-test>", "exec")
    compile(accept, "<accept-mode-test>", "exec")
    for rendered in (generic, accept):
        assert rendered.count("def force_arm_and_takeoff(") == 1
        assert "target_mm = int(altitude * 0.7 * 1000)" in rendered
