"""Composition roots: entry points that wire the agent and intervention layers.

`agents` holds the knowledge sources; `prototyping` holds the intervention layer
(blackboard, A/G, evidence) that coordinates them.  A module that constructs both
belongs above the two, not inside either — keeping the runners here is what stops
`import src.prototyping` from pulling the agent package in behind it.
"""

from .pipeline import PrototypingPipeline

__all__ = ["PrototypingPipeline"]
