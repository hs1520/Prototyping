"""PrototypingPipeline.dse_mode selects which DSE path the entry point enables.

Default is "variation" (the production-ready LLM-variation path). Orchestrator's
own flags stay default-OFF; the pipeline opts the chosen path in.
"""
from __future__ import annotations

import inspect

from src.prototyping.pipeline import PrototypingPipeline


def test_default_mode_is_variation():
    assert inspect.signature(PrototypingPipeline.__init__).parameters["dse_mode"].default == "variation"
    assert PrototypingPipeline._dse_flags("variation") == (True, False)


def test_bilevel_mode_selects_catalog_path():
    assert PrototypingPipeline._dse_flags("bilevel") == (False, True)


def test_off_mode_disables_both():
    assert PrototypingPipeline._dse_flags("off") == (False, False)


def test_unknown_mode_falls_back_to_variation():
    assert PrototypingPipeline._dse_flags("nonsense") == (True, False)
    assert PrototypingPipeline._dse_flags("") == (True, False)
