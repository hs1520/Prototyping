"""Isolated, atomic artifact bundles for authoritative pipeline runs.

Scientific runs never write shared filenames.  A writer holds the repository-wide
lock, builds one run directory, finalizes it, and only then atomically advances the
``latest`` symlink.  Incomplete runs remain inspectable but can never become latest.
"""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any


OUTPUT_DIR_ENV = "PROTOTYPING_OUTPUT_DIR"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "examples" / "output"
RUNS_DIRNAME = "runs"
LATEST_NAME = "latest"
STATE_NAME = "run_state.json"


def output_root() -> Path:
    return DEFAULT_OUTPUT_ROOT


def output_dir() -> Path:
    """Resolve the writable artifact directory for a runner.

    Authoritative child processes always receive ``PROTOTYPING_OUTPUT_DIR`` and
    therefore write into their open run bundle.  Ad-hoc/standalone commands use
    the legacy scratch root by default; they must never silently select the
    published ``latest`` bundle because that bundle is immutable after FINAL.
    """
    configured = os.environ.get(OUTPUT_DIR_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return output_root()


def latest_output_dir() -> Path:
    """Resolve the latest published bundle for read-only consumers."""
    latest = output_root() / LATEST_NAME
    if latest.exists():
        return latest.resolve()
    return output_root()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
    )


def read_state(run_dir: Path) -> dict[str, Any]:
    path = run_dir / STATE_NAME
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_state(run_dir: Path, state: str, **extra: Any) -> None:
    atomic_write_json(run_dir / STATE_NAME, {
        "state": state,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **extra,
    })


def ensure_open_bundle(run_dir: Path) -> None:
    state = read_state(run_dir).get("state")
    if state == "FINAL":
        raise RuntimeError(f"authoritative run bundle is immutable after FINAL: {run_dir}")


class AuthoritativeRunLock:
    """Non-blocking repository-wide lock preventing concurrent authoritative writers."""

    def __init__(self, root: Path | None = None):
        self.root = root or output_root()
        self.path = self.root / ".authoritative.lock"
        self._handle = None

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._handle.seek(0)
            owner = self._handle.read().strip() or "unknown owner"
            self._handle.close()
            self._handle = None
            raise RuntimeError(f"another authoritative run is active ({owner})") from exc
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(json.dumps({"pid": os.getpid(), "started_at": time.time()}))
        self._handle.flush()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


def create_staging_bundle(root: Path | None = None) -> Path:
    root = root or output_root()
    runs = root / RUNS_DIRNAME
    runs.mkdir(parents=True, exist_ok=True)
    staging = runs / f".staging-{uuid.uuid4()}"
    staging.mkdir()
    write_state(staging, "BUILDING")
    return staging


def assign_run_id(staging: Path, run_id: str, root: Path | None = None) -> Path:
    """Atomically give a staging bundle its permanent run-id directory."""
    root = root or output_root()
    final = root / RUNS_DIRNAME / run_id
    if final.exists():
        raise FileExistsError(f"run bundle already exists: {final}")
    os.replace(staging, final)
    write_state(final, "EVIDENCE", run_id=run_id)
    return final


def publish_latest(run_dir: Path, root: Path | None = None) -> Path:
    """Atomically publish a validated FINAL bundle as ``latest``."""
    root = root or output_root()
    state = read_state(run_dir)
    if state.get("state") != "FINAL":
        raise RuntimeError("only a FINAL run bundle may be published")
    latest = root / LATEST_NAME
    temporary = root / f".{LATEST_NAME}.{uuid.uuid4()}"
    target = os.path.relpath(run_dir, root)
    os.symlink(target, temporary)
    os.replace(temporary, latest)
    return latest
