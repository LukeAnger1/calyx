#!/usr/bin/env python3
"""Run the pass-search and print stats — a baseline proof that it works.

By default it runs on a self-contained synthetic model (no `calyx` needed):

    python run_baseline.py
    python run_baseline.py --iterations 1000 --max-depth 10 --seed 0

Pass a real program to search over the actual compiler (needs `calyx` on PATH):

    python run_baseline.py --futil ../examples/futil/simple.futil --lib ..

"Proof" = the report compares the MCTS result against (a) doing nothing (the
root) and (b) a random-search baseline given the same evaluation budget, and —
in synthetic mode — against the known optimum.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from calyx_mcts.search import MCTS  # noqa: E402
from calyx_mcts.state import CalyxState  # noqa: E402


def random_search(transition, reward_fn, actions, root, *, max_depth, trials, rng):
    """Baseline: `trials` random legal pass sequences; keep the best state seen.

    Same shape of work as an MCTS rollout, without the tree — so beating it is
    evidence the tree search is doing something.
    """
    best_reward = reward_fn(root)
    best_passes: tuple[str, ...] = ()
    for _ in range(trials):
        state, depth = root, 0
        while depth < max_depth:
            moved = False
            for action in rng.sample(list(actions), len(actions)):
                nxt = transition(state, action)
                if nxt is not None:
                    state, depth, moved = nxt, depth + 1, True
                    r = reward_fn(state)
                    if r > best_reward:
                        best_reward, best_passes = r, state.applied_passes
                    break
            if not moved:
                break
    return best_reward, best_passes


def tree_stats(root):
    stack, nodes, max_depth = [root], 0, 0
    depth_hist: dict[int, int] = {}
    branch_total, branch_nodes = 0, 0
    while stack:
        n = stack.pop()
        nodes += 1
        max_depth = max(max_depth, n.depth)
        depth_hist[n.depth] = depth_hist.get(n.depth, 0) + 1
        if n.children:
            branch_total += len(n.children)
            branch_nodes += 1
        stack.extend(n.children.values())
    avg_branch = branch_total / branch_nodes if branch_nodes else 0.0
    return nodes, max_depth, avg_branch, depth_hist


def area_trace(transition, area_fn, root, passes):
    """Replay a pass sequence, recording the area after each step."""
    trace = [("(root)", area_fn(root))]
    state = root
    for p in passes:
        state = transition(state, p)
        if state is None:
            trace.append((f"{p} [ILLEGAL]", float("nan")))
            break
        trace.append((p, area_fn(state)))
    return trace


def build_synthetic():
    from calyx_mcts import demo_model as dm
    root = dm.make_root()
    return {
        "mode": "synthetic",
        "transition": dm.demo_transition,
        "reward": dm.demo_reward,
        "area": dm.demo_area,
        "actions": dm.DEMO_ACTIONS,
        "root": root,
        "optimal_area": dm.optimal_area(root),
    }


def build_real(futil, lib, calyx_bin):
    from calyx_mcts.passes import DEFAULT_PASSES
    from calyx_mcts.transition import CalyxResourcesTransition
    from calyx_mcts.metrics.area import estimate_area_bits

    trans = CalyxResourcesTransition(futil, lib_path=lib, calyx=calyx_bin)
    root = trans.initial_state()
    if root is None:
        sys.exit(f"error: `{calyx_bin}` failed to compile {futil} (is it on PATH? is --lib right?)")
    area = lambda s: estimate_area_bits(s)  # noqa: E731  (faithful bits)
    return {
        "mode": f"real ({futil})",
        "transition": trans,
        "reward": lambda s: -estimate_area_bits(s),
        "area": area,
        "actions": DEFAULT_PASSES,
        "root": root,
        "optimal_area": None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--rollout-depth", type=int, default=8)
    ap.add_argument("--exploration", type=float, default=1.41421356)
    ap.add_argument("--time-limit", type=float, default=None, help="seconds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--futil", default=None, help="real mode: path to a .futil program")
    ap.add_argument("--lib", default=None, help="real mode: path to import library root")
    ap.add_argument("--calyx", default="calyx", help="real mode: calyx binary")
    args = ap.parse_args()

    cfg = build_real(args.futil, args.lib, args.calyx) if args.futil else build_synthetic()
    root: CalyxState = cfg["root"]
    root_area = cfg["area"](root)

    mcts = MCTS(
        cfg["transition"], cfg["reward"], cfg["actions"],
        max_depth=args.max_depth, iteration_budget=args.iterations,
        rollout_depth=args.rollout_depth, exploration=args.exploration,
        time_limit_s=args.time_limit, seed=args.seed,
    )
    t0 = time.monotonic()
    res = mcts.run(root)
    elapsed = time.monotonic() - t0

    rng = random.Random(args.seed)
    rand_reward, rand_passes = random_search(
        cfg["transition"], cfg["reward"], cfg["actions"], root,
        max_depth=args.max_depth, trials=args.iterations, rng=rng,
    )

    best_area = cfg["area"](res.best_state)
    rand_area = -rand_reward
    nodes, tree_depth, avg_branch, depth_hist = tree_stats(res.root)

    def pct_drop(a):
        return 0.0 if root_area == 0 else 100.0 * (root_area - a) / root_area

    W = 62
    print("=" * W)
    print(f"  Calyx pass-search baseline  —  mode: {cfg['mode']}")
    print("=" * W)
    print(f"  iterations={args.iterations}  max_depth={args.max_depth}  "
          f"seed={args.seed}  c={args.exploration:.3f}")
    print(f"  wall-clock: {elapsed*1000:.1f} ms   ({res.iterations_run} iters run)")
    print("-" * W)
    print("  RESULTS (area in bits — lower is better)")
    print(f"    root (no passes)   : {root_area:>12,.0f}")
    print(f"    random search      : {rand_area:>12,.0f}   ({pct_drop(rand_area):+.1f}%)")
    print(f"    MCTS               : {best_area:>12,.0f}   ({pct_drop(best_area):+.1f}%)")
    if cfg["optimal_area"] is not None:
        opt = cfg["optimal_area"]
        reach = 100.0 if opt == root_area else 100.0 * (root_area - best_area) / (root_area - opt)
        print(f"    optimum (oracle)   : {opt:>12,.0f}   ({pct_drop(opt):+.1f}%)")
        print(f"    → MCTS reached {reach:.1f}% of the achievable area reduction")
    print("-" * W)
    print("  BEST PASS SEQUENCE (MCTS)")
    if res.best_passes:
        for step, (name, area) in enumerate(
            area_trace(cfg["transition"], cfg["area"], root, res.best_passes)
        ):
            arrow = "  " if step == 0 else "→ "
            print(f"    {arrow}{name:<22} area={area:>12,.0f}")
    else:
        print("    (none — root was already best)")
    print("-" * W)
    print("  SEARCH TREE")
    print(f"    nodes created      : {res.nodes_created}")
    print(f"    live tree nodes    : {nodes}")
    print(f"    tree depth reached : {tree_depth} / {args.max_depth}")
    print(f"    avg branching      : {avg_branch:.2f}")
    print(f"    depth histogram    : "
          + ", ".join(f"d{d}:{depth_hist[d]}" for d in sorted(depth_hist)))
    print("-" * W)
    beats_root = best_area < root_area
    beats_rand = best_area <= rand_area
    verdict = ("PASS — MCTS improved over the root"
               + (" and matched/beat random search" if beats_rand else
                  " (random search tied/edged it — raise --iterations)")
               ) if beats_root else "NO IMPROVEMENT — check the model/limits"
    print(f"  VERDICT: {verdict}")
    print("=" * W)
    return 0 if beats_root else 1


if __name__ == "__main__":
    sys.exit(main())
