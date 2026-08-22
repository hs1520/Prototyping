"""Shared stub installer for heavy optional deps absent from a test environment.

Deliberately NOT a conftest hook: each test module that needs the stubs calls
``install_missing_dep_stubs()`` at import time, so installation order matches
the historical inline blocks exactly. A conftest-time install would run before
every module and flip ``import syside``-style availability probes (e.g. in
``test_analysis_emitter``) from a graceful skip to a run against an empty stub.
"""
import sys
from types import ModuleType


def install_missing_dep_stubs() -> None:
    """Stub dotenv/pinecone/syside only when the real package is unavailable."""
    for name, attrs in [
        ("dotenv", {"load_dotenv": lambda *a, **kw: None}),
        ("pinecone", {"Pinecone": type("Pinecone", (), {"__init__": lambda self, **kw: None})}),
        ("syside", {}),
    ]:
        if name not in sys.modules:
            try:  # prefer the real package — a stub here poisons later test files
                __import__(name)
                continue
            except ImportError:
                pass
            mod = ModuleType(name)
            for key, value in attrs.items():
                setattr(mod, key, value)
            sys.modules[name] = mod
