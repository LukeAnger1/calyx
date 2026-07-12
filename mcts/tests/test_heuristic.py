"""Tests for the per-node heuristic scaffolding.

Proves the extension point: a heuristic combines independent metrics into one
node score. Only the area metric exists today; a throwaway second metric here
stands in for the cycle-count / synthesis metrics we'll add, showing they slot
in without touching the search.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from calyx_mcts.metrics.area import AreaMetric  # noqa: E402
from calyx_mcts.metrics.base import Metric, NodeHeuristic  # noqa: E402
from calyx_mcts.state import CalyxState, PrimitiveInstance  # noqa: E402


class _ConstMetric(Metric):
    """Stand-in for a future metric (e.g. cycle count) with a fixed value."""

    def __init__(self, name, value, higher_is_better=False):
        self.name = name
        self._value = value
        self.higher_is_better = higher_is_better

    def evaluate(self, state):
        return self._value


def _state():
    return CalyxState(primitives=[PrimitiveInstance("std_reg", {"WIDTH": 32})])  # 32 bits


def test_single_metric_score():
    h = NodeHeuristic(terms=[(AreaMetric(), -1.0)])  # lower area -> higher score
    assert h.score(_state()) == -32


def test_breakdown_keys_and_values():
    h = NodeHeuristic(terms=[(AreaMetric(), -1.0), (_ConstMetric("cycles", 100), -0.5)])
    bd = h.breakdown(_state())
    assert bd == {"area_bits": -32.0, "cycles": -50.0}


def test_weighted_combination():
    h = NodeHeuristic(terms=[(AreaMetric(), -1.0), (_ConstMetric("cycles", 100), -0.5)])
    # -32 (area) + -50 (cycles) = -82
    assert h.score(_state()) == -82.0


def test_weights_encode_direction():
    # smaller program should score strictly higher under a lower-is-better weight
    small = CalyxState(primitives=[PrimitiveInstance("std_reg", {"WIDTH": 8})])
    big = CalyxState(primitives=[PrimitiveInstance("std_reg", {"WIDTH": 64})])
    h = NodeHeuristic(terms=[(AreaMetric(), -1.0)])
    assert h.score(small) > h.score(big)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
