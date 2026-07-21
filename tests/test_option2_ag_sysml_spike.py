"""Lock the syntax boundary chosen for the revised bounded A/G profile."""
from __future__ import annotations

from pathlib import Path

from src.simulation.syntax_checker import check_syntax


_AG_SPIKE = """package AGSpike {
    requirement def SystemContract {
        attribute failureDetected : Boolean;
        attribute deploymentLatency : Real;
        attribute maximumLatency : Real = 0.5;
        assume constraint { failureDetected }
        require constraint { deploymentLatency <= maximumLatency }
    }
    requirement def ComponentContract {
        attribute commandIssued : Boolean;
        require constraint { commandIssued }
    }
    dependency decomposition from SystemContract to ComponentContract;
}"""


def test_official_release_corpus_contains_assume_and_require_constraints():
    source = Path(
        __file__
    ).resolve().parents[1].joinpath(
        "data/SysML-v2-release-src/validation/08-Requirements/8-Requirements.sysml"
    ).read_text(encoding="utf-8")
    assert "assume constraint" in source
    assert "require constraint" in source


def test_selected_ag_profile_constructs_pass_the_project_syside_gate():
    result = check_syntax(_AG_SPIKE)
    assert result.has_errors is False
    assert result.score == 1.0
