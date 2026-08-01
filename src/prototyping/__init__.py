"""Intervention layer: blackboard, A/G assurance, and evidence.

`PrototypingPipeline` used to be re-exported here.  It composes `agents`, so the
re-export made `import src.prototyping` load the agent package and closed an
`agents` -> `prototyping` -> `agents` cycle at import time.  It now lives in
`src.app`.
"""

from .provider_factory import (
    available_llm_providers,
    create_llm,
    register_llm_provider,
)

__all__ = [
    "create_llm",
    "register_llm_provider",
    "available_llm_providers",
]
