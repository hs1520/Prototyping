"""Composition roots: entry points that wire the agent and intervention layers.

`agents` holds the knowledge sources, `prototyping` the intervention layer
(blackboard, A/G, evidence) that coordinates them. A module constructing both
sits above the two, which is what stops `import src.prototyping` from pulling
the agent package in behind it.
"""

from .pipeline import PrototypingPipeline

__all__ = ["PrototypingPipeline"]
