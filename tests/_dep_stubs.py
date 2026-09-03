"""Shared stub installer for heavy optional deps absent from a test environment.

Not a conftest hook: each test module that needs the stubs calls
``install_missing_dep_stubs()`` at import time, keeping installation order the
same as the inline blocks. A conftest install runs before every module and
flips ``import syside`` availability probes (e.g. ``test_analysis_emitter``)
from a skip to a run against an empty stub.
"""
import sys
from types import ModuleType


def install_missing_dep_stubs() -> None:
    for name, attrs in [
        ("dotenv", {"load_dotenv": lambda *a, **kw: None}),
        ("pinecone", {"Pinecone": type("Pinecone", (), {"__init__": lambda self, **kw: None})}),
        ("syside", {}),
    ]:
        if name not in sys.modules:
            try:  # prefer the real package; a stub poisons tests
                __import__(name)
                continue
            except ImportError:
                pass
            mod = ModuleType(name)
            for key, value in attrs.items():
                setattr(mod, key, value)
            sys.modules[name] = mod
