"""
Monte Carlo Tree Search (MCTS) for design space exploration.

Implements MCTS adapted for navigating the design space of cyber-physical
systems, balancing exploration of new designs with exploitation of
promising ones.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .design_space import DesignConfiguration, DesignSpace, DesignParameter, ParameterType


@dataclass
class MCTSNode:
    """A node in the Monte Carlo Tree Search tree."""
    config: DesignConfiguration
    parent: Optional[MCTSNode] = None
    children: List[MCTSNode] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    depth: int = 0
    untried_actions: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_fully_expanded(self) -> bool:
        """True if all child actions have been tried."""
        return len(self.untried_actions) == 0

    @property
    def average_reward(self) -> float:
        """Average reward across all visits."""
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits

    def ucb1_score(self, exploration_constant: float = math.sqrt(2)) -> float:
        """
        Upper Confidence Bound (UCB1) score for node selection.

        Balances exploitation (high average reward) with exploration
        (less-visited nodes).
        """
        if self.visits == 0:
            return float("inf")
        if self.parent is None or self.parent.visits == 0:
            return self.average_reward
        exploitation = self.average_reward
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration

    def best_child(self, exploration_constant: float = math.sqrt(2)) -> MCTSNode:
        """Select the child with the highest UCB1 score."""
        return max(self.children, key=lambda c: c.ucb1_score(exploration_constant))


EvaluationFunction = Callable[[DesignConfiguration], Dict[str, float]]


class MCTSDesignExplorer:
    """
    Monte Carlo Tree Search for design space exploration.

    Adapts classic MCTS to navigate the continuous/discrete design space
    of MBSE models, using an evaluation function to score configurations.

    The four phases of MCTS:
    1. Selection: Traverse the tree using UCB1 to select a leaf node
    2. Expansion: Add a new child node with a modified configuration
    3. Simulation: Randomly sample and evaluate from the expanded node
    4. Backpropagation: Update ancestor nodes with the simulation result
    """

    def __init__(
        self,
        design_space: DesignSpace,
        evaluation_function: EvaluationFunction,
        exploration_constant: float = math.sqrt(2),
        max_depth: int = 5,
        random_seed: Optional[int] = None,
    ):
        self.design_space = design_space
        self.evaluate = evaluation_function
        self.exploration_constant = exploration_constant
        self.max_depth = max_depth
        self.rng = random.Random(random_seed)

        # Initialize with the default configuration as root
        root_config = design_space.get_default_configuration()
        root_config.scores = self.evaluate(root_config)
        self.root = MCTSNode(
            config=root_config,
            untried_actions=self._generate_actions(root_config),
        )

        # Diagnostics populated by search()
        self.iterations_run: int = 0
        self.early_stopped: bool = False

    def search(
        self,
        num_iterations: int = 100,
        patience: Optional[int] = None,
    ) -> DesignConfiguration:
        """Run MCTS for up to `num_iterations`.

        If `patience` is set, stop early when the best score has not improved
        for `patience` consecutive iterations.
        """
        best_so_far = -float("inf")
        no_improve = 0
        self.iterations_run = 0

        for iteration in range(num_iterations):
            self.iterations_run = iteration + 1

            # Phase 1: Selection
            node = self._select(self.root)

            # Phase 2: Expansion
            if not node.is_fully_expanded and node.depth < self.max_depth:
                node = self._expand(node)

            # Phase 3: Simulation
            reward = self._simulate(node)

            # Phase 4: Backpropagation
            self._backpropagate(node, reward)

            # Record the configuration in the design space
            if node.config not in self.design_space.configurations:
                self.design_space.add_configuration(node.config)

            # Early stopping check
            if patience is not None:
                current_best = (
                    self.design_space.get_best_configuration().overall_score
                    if self.design_space.configurations else 0.0
                )
                if current_best > best_so_far + 1e-3:
                    best_so_far = current_best
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    self.early_stopped = True
                    break

        return self.design_space.get_best_configuration() or self.root.config

    def get_exploration_tree_summary(self) -> Dict[str, Any]:
        """Return a summary of the explored design space."""
        all_nodes: List[MCTSNode] = []
        self._collect_nodes(self.root, all_nodes)

        return {
            "total_nodes": len(all_nodes),
            "total_configurations": len(self.design_space.configurations),
            "root_visits": self.root.visits,
            "max_depth_reached": max((n.depth for n in all_nodes), default=0),
            "best_score": self.design_space.get_best_configuration().overall_score
            if self.design_space.configurations else 0.0,
        }

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a leaf node using UCB1."""
        current = node
        while current.children and current.is_fully_expanded:
            current = current.best_child(self.exploration_constant)
        return current

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand the node by adding a new child with a modified configuration.

        Skips actions that produce infeasible configurations (per
        DesignSpace.constraints). Returns the parent node unchanged when no
        feasible action remains.
        """
        # Drain infeasible actions, picking the first feasible one
        feasible_action = None
        feasible_config = None
        # Snapshot to allow safe mutation of untried_actions during iteration
        candidates = list(node.untried_actions)
        self.rng.shuffle(candidates)
        for action in candidates:
            node.untried_actions.remove(action)
            candidate = node.config.copy_with_changes(action)
            if self.design_space.is_feasible(candidate.parameters):
                feasible_action = action
                feasible_config = candidate
                break
        if feasible_config is None:
            return node

        new_config = feasible_config
        new_config.name = f"config_{len(self.design_space.configurations)}"
        new_config.scores = self.evaluate(new_config)

        child = MCTSNode(
            config=new_config,
            parent=node,
            depth=node.depth + 1,
            untried_actions=self._generate_actions(new_config),
        )
        node.children.append(child)
        self.design_space.add_configuration(new_config)
        return child

    def _simulate(self, node: MCTSNode) -> float:
        """
        Run a random simulation (rollout) from the given node.

        Makes random parameter changes and evaluates the result.
        """
        current_config = node.config
        for _ in range(self.rng.randint(1, 3)):  # 1-3 random steps
            actions = self._generate_actions(current_config)
            if not actions:
                break
            # Only take steps that yield a FEASIBLE configuration.  Mirrors the
            # feasibility check in _expand: without it the rollout evaluates and
            # backpropagates infeasible configs (e.g. triple redundancy with 1
            # sensor), polluting the reward signal with points the search is not
            # allowed to select.
            self.rng.shuffle(actions)
            next_config = None
            for action in actions:
                candidate = current_config.copy_with_changes(action)
                if self.design_space.is_feasible(candidate.parameters):
                    next_config = candidate
                    break
            if next_config is None:
                break  # no feasible move from here — end the rollout
            current_config = next_config
            current_config.scores = self.evaluate(current_config)

        return current_config.overall_score

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """Backpropagate the reward up the tree."""
        current: Optional[MCTSNode] = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent

    def _generate_actions(self, config: DesignConfiguration) -> List[Dict[str, Any]]:
        """
        Generate possible parameter modifications from a configuration.

        Each action is a dict of {parameter_name: new_value}.
        """
        actions: List[Dict[str, Any]] = []
        for param in self.design_space.parameters:
            current_val = config.parameters.get(param.name, param.default_value)

            if param.param_type == ParameterType.CONTINUOUS:
                # Try increasing and decreasing by 10%
                step = (
                    (param.max_value - param.min_value) * 0.1
                    if param.min_value is not None and param.max_value is not None
                    else abs(current_val) * 0.1 + 0.1
                )
                for delta in [-step, step]:
                    new_val = param.clamp(current_val + delta)
                    if new_val != current_val:
                        actions.append({param.name: new_val})

            elif param.param_type in (ParameterType.DISCRETE, ParameterType.CATEGORICAL):
                if param.choices:
                    for choice in param.choices:
                        if choice != current_val:
                            actions.append({param.name: choice})

            elif param.param_type == ParameterType.BOOLEAN:
                actions.append({param.name: not current_val})

        return actions

    def _collect_nodes(
        self, node: MCTSNode, result: List[MCTSNode]
    ) -> None:
        """Recursively collect all nodes in the tree."""
        result.append(node)
        for child in node.children:
            self._collect_nodes(child, result)
