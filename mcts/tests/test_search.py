"""Tests for the MCTS search — correctness and termination guarantees.

Uses a fast synthetic transition model (no `calyx` binary needed):
  - "cell-share"        halves the multiplier count (>=2 required), shrinking area
  - "dead-cell-removal" drops one adder (area-neutral here, but a legal branch)
  - "illegal-noop"      never applies (exercises pruning)

Runnable with pytest or as `python tests/test_search.py`.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from calyx_mcts.metrics.area import calyx_faithful_bits, estimate_area_bits  # noqa: E402
from calyx_mcts.metrics.base import NodeHeuristic  # noqa: E402
from calyx_mcts.metrics.area import AreaMetric  # noqa: E402
from calyx_mcts.search import MCTS  # noqa: E402
from calyx_mcts.state import CalyxState, PrimitiveInstance  # noqa: E402


# --- synthetic model --------------------------------------------------------

ACTIONS = ("cell-share", "dead-cell-removal", "illegal-noop")


def _mult_coster(inst):
    # charge multipliers ~width^2 so sharing them actually matters
    if inst.name == "std_mult_pipe":
        return float(inst.param_values()[0] ** 2)
    return calyx_faithful_bits(inst)


def _count(state, name):
    return sum(p.count for p in state.primitives if p.name == name)


def synthetic_transition(state, pass_name):
    prims = state.primitives
    new = None
    if pass_name == "cell-share":
        out, changed = [], False
        for p in prims:
            if p.name == "std_mult_pipe" and p.count >= 2 and not changed:
                out.append(PrimitiveInstance(p.name, dict(p.params), p.count // 2, p.external))
                changed = True
            else:
                out.append(p)
        new = out if changed else None
    elif pass_name == "dead-cell-removal":
        dropped, out = False, []
        for p in prims:
            if p.name == "std_add" and not dropped:
                dropped = True
                continue
            out.append(p)
        new = out if dropped else None
    elif pass_name == "illegal-noop":
        new = None
    if new is None:
        return None
    return CalyxState(primitives=new, applied_passes=state.applied_passes + (pass_name,))


def _reward(state):
    # maximize reward == minimize area (negative weight in the heuristic)
    return NodeHeuristic([(AreaMetric(coster=_mult_coster), -1.0)]).score(state)


def _root():
    return CalyxState(primitives=[
        PrimitiveInstance("std_reg", {"WIDTH": 32}),                 # 32
        PrimitiveInstance("std_mult_pipe", {"WIDTH": 8}, count=4),   # 4 * 64 = 256
        PrimitiveInstance("std_add", {"WIDTH": 32}, count=2),        # 0
    ])  # area 288 -> reward -288


def _all_nodes(root):
    stack, seen = [root], []
    while stack:
        n = stack.pop()
        seen.append(n)
        stack.extend(n.children.values())
    return seen


# --- correctness ------------------------------------------------------------

def test_finds_area_minimizing_sequence():
    mcts = MCTS(synthetic_transition, _reward, ACTIONS,
                max_depth=6, iteration_budget=400, seed=1)
    res = mcts.run(_root())
    # best reachable: multiplier count driven 4->2->1, area = 32 + 64 = 96
    assert res.best_reward == -96.0
    assert estimate_area_bits(res.best_state, coster=_mult_coster) == 96.0
    assert _count(res.best_state, "std_mult_pipe") == 1
    assert "cell-share" in res.best_passes


def test_best_is_at_least_root():
    mcts = MCTS(synthetic_transition, _reward, ACTIONS,
                max_depth=4, iteration_budget=100, seed=2)
    res = mcts.run(_root())
    assert res.best_reward >= _reward(_root())


# --- termination limits -----------------------------------------------------

def test_iteration_budget_is_respected():
    mcts = MCTS(synthetic_transition, _reward, ACTIONS,
                max_depth=5, iteration_budget=50, seed=3)
    res = mcts.run(_root())
    assert res.iterations_run <= 50


def test_never_exceeds_max_depth():
    mcts = MCTS(synthetic_transition, _reward, ACTIONS,
                max_depth=3, iteration_budget=300, seed=4)
    res = mcts.run(_root())
    assert all(n.depth <= 3 for n in _all_nodes(res.root))


def test_unbounded_growth_still_terminates():
    # transition that ALWAYS produces a new (bigger) state — no natural stop.
    # Only the depth + iteration limits keep the search finite.
    def grow(state, pass_name):
        prims = list(state.primitives) + [PrimitiveInstance("std_reg", {"WIDTH": 1})]
        return CalyxState(primitives=prims, applied_passes=state.applied_passes + (pass_name,))

    mcts = MCTS(grow, _reward, ("a", "b", "c"),
                max_depth=4, iteration_budget=120, rollout_depth=10, seed=5)
    res = mcts.run(_root())
    assert res.iterations_run == 120
    assert all(n.depth <= 4 for n in _all_nodes(res.root))
    # deepest node's sequence can't exceed the depth limit either
    assert max(len(n.pass_sequence()) for n in _all_nodes(res.root)) <= 4


def test_all_moves_illegal_returns_root():
    mcts = MCTS(lambda s, p: None, _reward, ("x", "y"),
                max_depth=5, iteration_budget=30, seed=6)
    res = mcts.run(_root())
    assert res.best_passes == ()
    assert res.iterations_run == 30
    assert res.nodes_created == 1  # only the root


def test_tiny_budget_runs_once():
    mcts = MCTS(synthetic_transition, _reward, ACTIONS,
                max_depth=5, iteration_budget=1, seed=7)
    res = mcts.run(_root())
    assert res.iterations_run == 1


def test_time_limit_zero_stops_immediately():
    mcts = MCTS(synthetic_transition, _reward, ACTIONS,
                max_depth=5, iteration_budget=10_000, time_limit_s=0.0, seed=8)
    res = mcts.run(_root())
    assert res.iterations_run == 0  # wall-clock cutoff hit before any iteration


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
