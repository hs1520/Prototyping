"""Package layering stays acyclic after the P1 shared-helper moves."""
from __future__ import annotations

import ast
import collections
import itertools
from pathlib import Path


_SRC = Path(__file__).parents[1] / "src"


def _package_edges() -> collections.Counter[tuple[str, str]]:
    edges: collections.Counter[tuple[str, str]] = collections.Counter()
    for path in _SRC.rglob("*.py"):
        parts = path.relative_to(_SRC).parts
        if len(parts) < 2:
            continue
        package = parts[0]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 2:
                target = node.module.split(".")[0]
                if target != package:
                    edges[(package, target)] += 1
    return edges


def test_no_bidirectional_package_pairs():
    edges = _package_edges()
    packages = {name for edge in edges for name in edge}
    assert not [
        (left, right)
        for left, right in itertools.combinations(packages, 2)
        if edges[(left, right)] and edges[(right, left)]
    ]


def test_moved_helpers_are_not_still_defined_in_source_modules():
    forbidden = {
        "sysml/lite_model.py": {"_is_stdlib_sema_error"},
        "simulation/syntax_checker.py": {"_is_stdlib_sema_error"},
        "sitl/dse_calibration.py": {"CalibrationResult", "calibrate_ranking"},
    }
    for relative, names in forbidden.items():
        tree = ast.parse((_SRC / relative).read_text(encoding="utf-8"))
        defined = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert defined.isdisjoint(names), f"{relative}: {defined & names}"
