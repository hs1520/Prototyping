from __future__ import annotations

import inspect

import pytest

from src.app.pipeline import PrototypingPipeline


def test_default_mode_variation():
    assert inspect.signature(PrototypingPipeline.__init__).parameters["dse_mode"].default == "variation"
    assert PrototypingPipeline._dse_flags("variation") is True


def test_bilevel_mode_selects_catalog_path():
    assert PrototypingPipeline._dse_flags("bilevel") is False


def test_off_mode_raises_scalar_removed():
    with pytest.raises(ValueError, match="removed"):
        PrototypingPipeline._dse_flags("off")


def test_unknown_mode_falls_back():
    assert PrototypingPipeline._dse_flags("nonsense") is True
    assert PrototypingPipeline._dse_flags("") is True
