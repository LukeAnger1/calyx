"""Monte Carlo Tree Search over Calyx pass sequences (UCT).

A node holds a `CalyxState`; the edge into it is the pass that produced it. The
path from the root is a concrete pass sequence, so the best node found is a
reproducible recipe. Nodes are scored by a reward function (maximized) — plug in
`NodeHeuristic.score` (with lower-is-better metrics given negative weights).

Termination is guaranteed three independent ways, any one of which is enough to
stop the search:

  1. `iteration_budget` — `run()` executes at most this many MCTS iterations.
  2. `max_depth`        — no pass sequence exceeds this length; nodes at the
                          limit are terminal (never expanded), and rollouts are
                          bounded by it, so every iteration does finite work.
  3. `time_limit_s`     — optional wall-clock cutoff checked each iteration.

Illegal moves (transition returns None) are dropped, so a node can run out of
legal children and become a dead end — also terminal.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from calyx_mcts.state import CalyxState
from calyx_mcts.transition import Transition

RewardFn = Callable[[CalyxState], float]


class Node:
    __slots__ = (
        "state", "pass_applied", "parent", "depth", "children",
        "untried", "visits", "total_reward", "reward", "terminal",
    )

    def __init__(self, state, pass_applied, parent, depth, actions, max_depth, reward):
        self.state: CalyxState = state
        self.pass_applied: Optional[str] = pass_applied
        self.parent: Optional["Node"] = parent
        self.depth: int = depth
        self.children: dict[str, "Node"] = {}
        self.reward: float = reward           # cached reward(state)
        self.visits: int = 0
        self.total_reward: float = 0.0
        # A node at the depth limit gets no moves -> terminal.
        self.terminal: bool = depth >= max_depth
        self.untried: list[str] = [] if self.terminal else list(actions)

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.visits if self.visits else 0.0

    def pass_sequence(self) -> tuple[str, ...]:
        seq: list[str] = []
        node: Optional[Node] = self
        while node is not None and node.pass_applied is not None:
            seq.append(node.pass_applied)
            node = node.parent
        return tuple(reversed(seq))


@dataclass
class SearchResult:
    best_passes: tuple[str, ...]
    best_reward: float
    best_state: CalyxState
    iterations_run: int
    nodes_created: int
    root: Node = field(repr=False)


class MCTS:
    def __init__(
        self,
        transition: Transition,
        reward_fn: RewardFn,
        actions: Sequence[str],
        *,
        max_depth: int = 8,
        iteration_budget: int = 200,
        rollout_depth: int = 6,
        exploration: float = math.sqrt(2),
        time_limit_s: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if iteration_budget < 1:
            raise ValueError("iteration_budget must be >= 1")
        self.transition = transition
        self.reward_fn = reward_fn
        self.actions = tuple(actions)
        self.max_depth = max_depth
        self.iteration_budget = iteration_budget
        self.rollout_depth = rollout_depth
        self.exploration = exploration
        self.time_limit_s = time_limit_s
        self.rng = random.Random(seed)
        self._nodes_created = 0

    # -- node construction ---------------------------------------------------
    def _make_node(self, state, pass_applied, parent, depth) -> Node:
        self._nodes_created += 1
        return Node(
            state, pass_applied, parent, depth,
            actions=self.actions, max_depth=self.max_depth,
            reward=self.reward_fn(state),
        )

    # -- UCT phases ----------------------------------------------------------
    def _select(self, root: Node) -> Node:
        """Descend by UCT until a node that can be expanded or is terminal."""
        node = root
        while not node.terminal:
            if node.untried:          # expandable here
                return node
            if not node.children:     # dead end: fully expanded, no legal child
                node.terminal = True
                return node
            node = self._best_uct_child(node)
        return node

    def _best_uct_child(self, node: Node) -> Node:
        log_n = math.log(node.visits) if node.visits > 0 else 0.0
        best, best_val = None, -math.inf
        for child in node.children.values():
            if child.visits == 0:
                return child
            uct = child.mean_reward + self.exploration * math.sqrt(log_n / child.visits)
            if uct > best_val:
                best, best_val = child, uct
        return best  # type: ignore[return-value]

    def _expand(self, node: Node) -> Node:
        """Try untried passes until one is legal; add and return that child.

        If every remaining pass is illegal, the node is a dead end (terminal)
        and is returned unchanged.
        """
        while node.untried:
            idx = self.rng.randrange(len(node.untried))
            action = node.untried.pop(idx)
            child_state = self.transition(node.state, action)
            if child_state is None:
                continue  # illegal move, drop it
            child = self._make_node(child_state, action, node, node.depth + 1)
            node.children[action] = child
            return child
        # Untried exhausted. Only a dead end if no legal child was ever found;
        # otherwise the node stays selectable and we descend into its children.
        if not node.children:
            node.terminal = True
        return node

    def _rollout(self, node: Node) -> float:
        """Random legal passes from `node` to a depth cap; return best reward seen.

        We may stop optimizing at any point, so a path's value is the *best*
        state encountered along it, not just the endpoint. Bounded by both
        `max_depth` and `rollout_depth`, so it always terminates.
        """
        best = node.reward
        state = node.state
        depth = node.depth
        steps = 0
        while depth < self.max_depth and steps < self.rollout_depth:
            moved = False
            for action in self.rng.sample(self.actions, len(self.actions)):
                nxt = self.transition(state, action)
                if nxt is not None:
                    state, depth, steps, moved = nxt, depth + 1, steps + 1, True
                    best = max(best, self.reward_fn(state))
                    break
            if not moved:
                break  # dead end
        return best

    @staticmethod
    def _backpropagate(node: Node, value: float) -> None:
        cur: Optional[Node] = node
        while cur is not None:
            cur.visits += 1
            cur.total_reward += value
            cur = cur.parent

    # -- driver --------------------------------------------------------------
    def run(self, root_state: CalyxState) -> SearchResult:
        root = self._make_node(root_state, pass_applied=None, parent=None, depth=0)
        best_node = root
        start = time.monotonic()
        iterations = 0

        for iterations in range(1, self.iteration_budget + 1):
            if self.time_limit_s is not None and (time.monotonic() - start) >= self.time_limit_s:
                iterations -= 1
                break

            leaf = self._select(root)
            if not leaf.terminal and leaf.untried:
                leaf = self._expand(leaf)
            if leaf.reward > best_node.reward:
                best_node = leaf
            value = self._rollout(leaf)
            self._backpropagate(leaf, value)

        return SearchResult(
            best_passes=best_node.pass_sequence(),
            best_reward=best_node.reward,
            best_state=best_node.state,
            iterations_run=iterations,
            nodes_created=self._nodes_created,
            root=root,
        )
