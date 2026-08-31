"""SITL --speedup adoption (plan A): sim-time clocks and scaled pacing.

The rules under test (see sitl_specs.TestContext docstring):
- waits/send-rates that REPRESENT sim durations scale by wall/speedup;
- requirement-bound windows read the sim clock (time_boot_ms), wall clock
  only as watchdog — so measured latencies and verdict strictness do not
  drift with the speedup factor;
- the gazebo FDM backend is lockstepped to an external engine and pins
  speedup back to 1.
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


def test_sim_clock_tracks_time_boot_ms_monotonically():
    clock = SimClock()
    assert clock.now_s() is None
    clock.observe(SimpleNamespace(time_boot_ms=1500))
    clock.observe(SimpleNamespace(no_stamp=True))          # ignored
    clock.observe(SimpleNamespace(time_boot_ms=900))       # regression ignored
    assert clock.now_s() == 1.5
    clock.observe(SimpleNamespace(time_boot_ms=2500))
    assert clock.now_s() == 2.5
    assert clock.elapsed_s(1.5) == 1.0
    assert clock.elapsed_s(None) is None


def test_sim_clock_installs_a_message_hook():
    mav = _HookedMav()
    clock = SimClock()
    clock.install(mav)
    mav.feed(SimpleNamespace(time_boot_ms=4000))
    assert clock.now_s() == 4.0


def test_context_attaches_clock_and_survives_hookless_mavs():
    hooked = TestContext(mav=_HookedMav(), mavutil=SimpleNamespace())
    hooked.mav.feed(SimpleNamespace(time_boot_ms=1000))
    assert hooked.clock.now_s() == 1.0
    # Unit-test fakes have no message_hooks; installation is a no-op.
    bare = TestContext(mav=SimpleNamespace(), mavutil=SimpleNamespace())
    assert bare.clock.now_s() is None


def test_scaled_sleep_divides_by_speedup(monkeypatch):
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


def test_sim_window_expires_on_sim_budget_not_wall():
    mav = _HookedMav()
    ctx = TestContext(mav=mav, mavutil=SimpleNamespace(), speedup=10.0)
    mav.feed(SimpleNamespace(time_boot_ms=10_000))
    expired = ctx.sim_window(5.0, wall_margin_s=30.0)
    assert not expired()
    mav.feed(SimpleNamespace(time_boot_ms=10_100))  # budget starts here
    assert not expired()
    mav.feed(SimpleNamespace(time_boot_ms=15_000))  # 4.9 sim s — inside
    assert not expired()
    mav.feed(SimpleNamespace(time_boot_ms=15_200))  # 5.1 sim s — budget spent
    assert expired()


def test_sim_window_budget_starts_at_first_stamp():
    mav = _HookedMav()
    ctx = TestContext(mav=mav, mavutil=SimpleNamespace())
    expired = ctx.sim_window(2.0, wall_margin_s=30.0)
    assert not expired()                             # silence is not billed
    mav.feed(SimpleNamespace(time_boot_ms=60_000))   # clock starts late
    assert not expired()
    mav.feed(SimpleNamespace(time_boot_ms=62_500))   # 2.5 sim s after start
    assert expired()


def test_sim_window_does_not_bill_the_gap_before_it_opened():
    """The SAFE_003 regression, distilled: the GCS-loss warmup sends
    heartbeats for 15 sim-seconds and reads nothing, so the clock's latest
    stamp predates the verify window by ~30 sim-seconds. Seeding the budget
    from that stale stamp expired a 25s window 1.5s after it opened and
    turned a passing failsafe test into '超时未切换到 LAND'."""
    mav = _HookedMav()
    ctx = TestContext(mav=mav, mavutil=SimpleNamespace())
    mav.feed(SimpleNamespace(time_boot_ms=52_344))   # last pre-disconnect stamp
    expired = ctx.sim_window(25.0, wall_margin_s=30.0)
    assert not expired()
    mav.feed(SimpleNamespace(time_boot_ms=82_369))   # first post-reconnect stamp
    assert not expired()                             # gap is NOT billed
    mav.feed(SimpleNamespace(time_boot_ms=107_000))  # +24.6 sim s — inside
    assert not expired()
    mav.feed(SimpleNamespace(time_boot_ms=107_500))  # +25.1 sim s — spent
    assert expired()


def test_sim_window_wall_watchdog_terminates_stalled_clock():
    ctx = TestContext(mav=SimpleNamespace(), mavutil=SimpleNamespace())
    expired = ctx.sim_window(0.05, wall_margin_s=0.0)
    assert not expired()
    time.sleep(0.08)
    assert expired()                                 # old wall-timeout behavior


def _bridge(**kwargs):
    from src.sysml.lite_model import build_lite_model
    from src.sitl.sitl_bridge import SITLBridge

    model = build_lite_model("package P {}", model_name="P")
    return SITLBridge(model, output_dir=str(kwargs.pop("tmp")), **kwargs)


def test_bridge_native_carries_speedup_and_gazebo_pins_it(tmp_path, capsys):
    import pytest

    native = _bridge(tmp=tmp_path / "n", speedup=5.0)
    assert native._speedup == 5.0

    gazebo = _bridge(tmp=tmp_path / "g", fdm_backend="gazebo", speedup=5.0)
    assert gazebo._speedup == 1.0
    assert "钉回 1.0" in capsys.readouterr().out

    with pytest.raises(ValueError):
        _bridge(tmp=tmp_path / "x", speedup=0.5)
