"""SITL --speedup adoption (plan A): sim-time clocks and scaled pacing.

Rules under test (see sitl_specs.TestContext docstring):
- waits/send-rates representing sim durations scale by wall/speedup;
- requirement-bound windows read the sim clock (time_boot_ms), wall clock only
  as watchdog, so latencies and strictness do not drift with the factor;
- the gazebo FDM backend is lockstepped externally and pins speedup to 1.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

from src.sitl.sim_clock import SimClock
from src.sitl.sitl_specs import TestContext


class _HookedMav:
    def __init__(self):
        self.message_hooks = []

    def feed(self, msg):
        for hook in self.message_hooks:
            hook(self, msg)


def test_clock_tracks_boot_ms():
    clock = SimClock()
    assert clock.now_s() is None
    clock.observe(SimpleNamespace(time_boot_ms=1500))
    clock.observe(SimpleNamespace(no_stamp=True))
    clock.observe(SimpleNamespace(time_boot_ms=900))
    assert clock.now_s() == 1.5
    clock.observe(SimpleNamespace(time_boot_ms=2500))
    assert clock.now_s() == 2.5
    assert clock.elapsed_s(1.5) == 1.0
    assert clock.elapsed_s(None) is None


def test_clock_installs_hook():
    mav = _HookedMav()
    clock = SimClock()
    clock.install(mav)
    mav.feed(SimpleNamespace(time_boot_ms=4000))
    assert clock.now_s() == 4.0


def test_context_attaches_clock():
    hooked = TestContext(mav=_HookedMav(), mavutil=SimpleNamespace())
    hooked.mav.feed(SimpleNamespace(time_boot_ms=1000))
    assert hooked.clock.now_s() == 1.0
    # Unit-test fakes have no message_hooks; installation is a no-op.
    bare = TestContext(mav=SimpleNamespace(), mavutil=SimpleNamespace())
    assert bare.clock.now_s() is None


def test_scaled_sleep_divides(monkeypatch):
    slept = []
    monkeypatch.setattr(
        "src.sitl.sitl_specs.time.sleep", lambda s: slept.append(s)
    )
    TestContext(
        mav=SimpleNamespace(), mavutil=SimpleNamespace(), speedup=5.0
    ).scaled_sleep(2.0)
    TestContext(
        mav=SimpleNamespace(), mavutil=SimpleNamespace()
    ).scaled_sleep(2.0)
    assert slept == [0.4, 2.0]


def test_window_expires_on_sim_time():
    mav = _HookedMav()
    ctx = TestContext(mav=mav, mavutil=SimpleNamespace(), speedup=10.0)
    mav.feed(SimpleNamespace(time_boot_ms=10_000))
    expired = ctx.sim_window(5.0, wall_margin_s=30.0)
    assert not expired()
    mav.feed(SimpleNamespace(time_boot_ms=10_100))
    assert not expired()
    mav.feed(SimpleNamespace(time_boot_ms=15_000))
    assert not expired()
    mav.feed(SimpleNamespace(time_boot_ms=15_200))
    assert expired()


def test_window_starts_at_first_stamp():
    mav = _HookedMav()
    ctx = TestContext(mav=mav, mavutil=SimpleNamespace())
    expired = ctx.sim_window(2.0, wall_margin_s=30.0)
    assert not expired()
    mav.feed(SimpleNamespace(time_boot_ms=60_000))
    assert not expired()
    mav.feed(SimpleNamespace(time_boot_ms=62_500))
    assert expired()


def test_window_ignores_earlier_gap():
    """The SAFE_003 regression: the GCS-loss warmup sends heartbeats for 15
    sim-seconds and reads nothing, so the clock's latest stamp predates the verify
    window by ~30 sim-seconds.

    Seeding the budget from that stale stamp expired a 25s window 1.5s after it
    opened and failed a passing failsafe test with '超时未切换到 LAND'.
    """
    mav = _HookedMav()
    ctx = TestContext(mav=mav, mavutil=SimpleNamespace())
    mav.feed(SimpleNamespace(time_boot_ms=52_344))
    expired = ctx.sim_window(25.0, wall_margin_s=30.0)
    assert not expired()
    mav.feed(SimpleNamespace(time_boot_ms=82_369))
    assert not expired()
    mav.feed(SimpleNamespace(time_boot_ms=107_000))
    assert not expired()
    mav.feed(SimpleNamespace(time_boot_ms=107_500))
    assert expired()


def test_window_wall_watchdog():
    ctx = TestContext(mav=SimpleNamespace(), mavutil=SimpleNamespace())
    expired = ctx.sim_window(0.05, wall_margin_s=0.0)
    assert not expired()
    time.sleep(0.08)
    assert expired()


def _bridge(**kwargs):
    from src.sysml.lite_model import build_lite_model
    from src.sitl.sitl_bridge import SITLBridge

    model = build_lite_model("package P {}", model_name="P")
    return SITLBridge(model, output_dir=str(kwargs.pop("tmp")), **kwargs)


def test_bridge_speedup_gazebo_pinned(tmp_path, capsys):
    import pytest

    native = _bridge(tmp=tmp_path / "n", speedup=5.0)
    assert native._speedup == 5.0

    gazebo = _bridge(tmp=tmp_path / "g", fdm_backend="gazebo", speedup=5.0)
    assert gazebo._speedup == 1.0
    assert "钉回 1.0" in capsys.readouterr().out

    with pytest.raises(ValueError):
        _bridge(tmp=tmp_path / "x", speedup=0.5)


class _AckMav:
    """ACKs a command only from the Nth send (the SAFE_005 shape: a send followed by
    a plain sleep vanished, one followed by a blocking recv pump was processed).
    """

    def __init__(self, ack_on_attempt=1, result=0):
        self.message_hooks = []
        self.target_system = 1
        self.target_component = 0
        self.sends = 0
        self._ack_on = ack_on_attempt
        self._result = result
        self._pending = None
        self.mav = SimpleNamespace(command_long_send=self._send)

    def _send(self, _sys, _comp, cmd, _conf, *params):
        self.sends += 1
        if self.sends >= self._ack_on:
            self._pending = SimpleNamespace(
                get_type=lambda: "COMMAND_ACK", command=cmd,
                result=self._result,
            )

    def recv_match(self, type=None, blocking=True, timeout=1):  # noqa: A002
        message, self._pending = self._pending, None
        return message


def test_command_retries_until_ack():
    from src.sitl.sitl_specs import _send_command_acked

    mav = _AckMav(ack_on_attempt=2)
    ctx = TestContext(mav=mav, mavutil=SimpleNamespace())
    result = _send_command_acked(ctx, 208, [2.0] + [0.0] * 6,
                                 ack_timeout_s=0.05)
    assert result == 0
    assert mav.sends == 2


def test_no_ack_raises():
    import pytest
    from src.sitl.sitl_specs import _send_command_acked

    mav = _AckMav(ack_on_attempt=99)
    ctx = TestContext(mav=mav, mavutil=SimpleNamespace())
    with pytest.raises(RuntimeError, match="no COMMAND_ACK"):
        _send_command_acked(ctx, 208, [2.0] + [0.0] * 6,
                            attempts=2, ack_timeout_s=0.05)
    assert mav.sends == 2


def test_eof_retryable():
    """A bare EOFError from the per-call Vertex worker's dying pipe is retryable.

    It used to abort the call outright (149k tokens lost on one NO-REFINE roll);
    a retry spawns a new worker.
    """
    from src.llm.interface import VertexLLM

    assert VertexLLM._is_retryable(EOFError())
    assert VertexLLM._is_retryable(RuntimeError("EOFError: "))
