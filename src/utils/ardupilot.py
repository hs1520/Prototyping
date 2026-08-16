"""Host-environment discovery for the ArduPilot Copter SITL binary."""
from __future__ import annotations

from pathlib import Path


ARDUCOPTER_CANDIDATES = (
    Path("~/ardupilot/build/sitl/bin/arducopter").expanduser(),
    Path("~/PycharmProjects/ardupilot/build/sitl/bin/arducopter").expanduser(),
)


def find_arducopter_binary() -> str | None:
    """Return the first installed candidate, or ``None``."""
    return next(
        (str(path) for path in ARDUCOPTER_CANDIDATES if path.exists()),
        None,
    )


def default_arducopter_binary() -> str:
    """Return an installed binary, or the conventional local fallback path."""
    return find_arducopter_binary() or str(ARDUCOPTER_CANDIDATES[-1])
