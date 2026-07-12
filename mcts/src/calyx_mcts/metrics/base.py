"""Metric abstraction + a per-node heuristic that combines metrics.

The MCTS will score each node with a heuristic. A heuristic is a weighted
combination of independent `Metric`s so we can add signals (cycle count,
LUTs/FFs from synthesis, IR-size proxies, ...) without touching the search.

Right now only the area metric exists (see `area.py`); this module just defines
the shape everything else plugs into.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from calyx_mcts.state import CalyxState


class Metric(ABC):
    """A single scalar measurement of a Calyx program.

    Subclasses report a raw value in their own units (bits, cycles, LUTs, ...).
    Direction is a property of the metric, not the search: `higher_is_better`
    lets a heuristic (or a normalizer) orient the value correctly.
    """

    #: stable identifier, used as the key in heuristic breakdowns
    name: str = "metric"
    #: True if larger values are more desirable (area/cycles are False)
    higher_is_better: bool = False

    @abstractmethod
    def evaluate(self, state: CalyxState) -> float:
        """Return the raw metric value for `state`."""
        raise NotImplementedError


@dataclass
class NodeHeuristic:
    """Weighted sum of metrics → a single score for an MCTS node.

    `terms` pairs each metric with a weight. Encode direction in the weight:
    since lower area is better, give the area metric a negative weight so a
    smaller program yields a higher score. This keeps `Metric`s reporting
    honest raw values while the heuristic owns the "what do we want" policy.
    """

    terms: list[tuple[Metric, float]]

    def breakdown(self, state: CalyxState) -> dict[str, float]:
        """Per-metric weighted contributions (handy for debugging a node)."""
        return {
            metric.name: weight * metric.evaluate(state)
            for metric, weight in self.terms
        }

    def score(self, state: CalyxState) -> float:
        """Scalar node score = sum of weighted metric contributions."""
        return sum(self.breakdown(state).values())
