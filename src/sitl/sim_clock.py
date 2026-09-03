"""Simulation-time clock for SITL evidence.

Under ``--speedup N`` the vehicle's clock runs Nx faster than the wall clock,
so a ``time.time()``/``time.monotonic()`` measurement of a sim-time duration
looks Nx better than it is; requirement-bound measurements read the vehicle's
own ``time_boot_ms`` from streamed MAVLink messages (ATTITUDE,
GLOBAL_POSITION_INT, POSITION_TARGET_GLOBAL_INT, ...). ``SimClock`` hooks a
pymavlink connection's ``message_hooks``, so every received message advances
the clock even when a ``recv_match`` filter drops it. ``now_s()`` is None
until the first stamped message, so callers keep a wall-clock watchdog for
that window and for clock stalls.
"""
from __future__ import annotations

from typing import Any, Optional


class SimClock:
    """Tracks vehicle simulation time from received ``time_boot_ms`` stamps."""

    def __init__(self) -> None:
        self._boot_ms: Optional[int] = None
        # Count of accepted stamps. A budget window binds its start to the
        # first stamp observed after the window opened: the latest stamp can
        # be arbitrarily old when the harness spends time sending without
        # receiving (the GCS-loss warmup sends heartbeats for 15s and reads
        # nothing), and billing that gap to the window expired a 25s budget
        # 1.5s after it opened.
        self.observations: int = 0

    def observe(self, msg: Any) -> None:
        stamp = getattr(msg, "time_boot_ms", None)
        if not isinstance(stamp, (int, float)) or stamp <= 0:
            return
        stamp = int(stamp)
        self.observations += 1
        # Monotonic max: a fresh SITL (per-test --wipe launch) starts near 0,
        # and each test installs a fresh clock, so a reboot mid-test is the
        # only regression source - ignore it rather than jump backwards.
        if self._boot_ms is None or stamp > self._boot_ms:
            self._boot_ms = stamp

    def install(self, mav: Any) -> None:
        """Advance this clock on every message *mav* receives."""
        hooks = getattr(mav, "message_hooks", None)
        if hooks is None:
            return
        hooks.append(lambda _mav, msg: self.observe(msg))

    def now_s(self) -> Optional[float]:
        """Latest observed sim time in seconds, or None before any stamp."""
        return None if self._boot_ms is None else self._boot_ms / 1000.0

    def elapsed_s(self, since_s: Optional[float]) -> Optional[float]:
        """Sim seconds since *since_s*; None when either endpoint is unknown."""
        now = self.now_s()
        if now is None or since_s is None:
            return None
        return now - since_s
