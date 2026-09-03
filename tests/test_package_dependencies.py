from __future__ import annotations

import ast
import collections
import itertools
import typing
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


def test_package_graph_acyclic():
    edges = _package_edges()
    graph: dict[str, set[str]] = collections.defaultdict(set)
    for source, target in edges:
        graph[source].add(target)

    visiting: list[str] = []
    visited: set[str] = set()

    def visit(package: str) -> None:
        if package in visiting:
            start = visiting.index(package)
            cycle = visiting[start:] + [package]
            raise AssertionError("package cycle: " + " -> ".join(cycle))
        if package in visited:
            return
        visiting.append(package)
        for target in graph.get(package, ()):
            visit(target)
        visiting.pop()
        visited.add(package)

    for package in set(graph) | {item for targets in graph.values() for item in targets}:
        visit(package)


def test_design_agent_annotations_resolve():
    from src.agents.assembly_finalization import AssemblyFinalizer

    hints = typing.get_type_hints(AssemblyFinalizer.finalize)
    assert hints["request"].__name__ == "AssemblyRequest"


def test_no_reimplemented_protocols():
    tree = ast.parse((_SRC / "agents/design_agent.py").read_text(encoding="utf-8"))
    design_agent = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DesignAgent"
    )
    methods = {
        node.name
        for node in design_agent.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert methods.isdisjoint({
        "_step1_architecture",
        "_step2_parts",
        "_step3_interfaces",
        "_step4_behavior",
        "_postprocess_assembly",
        "_build_step_query",
        "_inject_missing_part_defs",
        "_inject_missing_state_defs",
        "_inject_missing_item_defs",
        "_inject_missing_ag_obligation_defs",
        "_validate_connections",
        "_run_refinement",
        "_apply_semantic_fixes",
        "_fix_self_test_behavior_semantics",
        "_fix_functional_satisfy_ownership",
        "_build_refinement_query",
        "_apply_requirement_traceability",
    })
    source = (_SRC / "agents/design_agent.py").read_text(encoding="utf-8")
    assert "REFINEMENT_SYSTEM_PROMPT" not in source
    assert "original_system_prompt" not in source
    assert "SYSTEM_PROMPT" not in source
    assert ".system_prompt =" not in source

    prompting = (_SRC / "llm/chain_of_thought.py").read_text(
        encoding="utf-8"
    )
    assert "generation_conversation" not in prompting
    assert "self._conversation" not in prompting
    assert "self.system_prompt" not in prompting

    requirements = (_SRC / "agents/requirements_agent.py").read_text(
        encoding="utf-8"
    )
    assert ".system_prompt =" not in requirements


def test_moved_helpers_not_redefined():
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


def test_sitl_no_private_dse_import():
    for path in (_SRC / "sitl").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        private = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.level == 2
            and node.module
            and node.module.startswith("dse")
            for alias in node.names
            if alias.name.startswith("_")
        }
        assert not private, f"{path.relative_to(_SRC)} imports {sorted(private)}"
