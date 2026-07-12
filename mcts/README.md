# Calyx pass-exploration MCTS

Monte Carlo Tree Search over Calyx compiler-pass sequences. Each **node** holds
a Calyx program state; an **edge** applies one pass; nodes are scored by a
**heuristic** built from one or more **metrics**.

Status: search + area metric working; more metrics to come.

## Layout

```
mcts/
├── src/calyx_mcts/
│   ├── state.py            # PrimitiveInstance, CalyxState (what a node carries)
│   ├── passes.py           # DEFAULT_PASSES — the action space (real pass names)
│   ├── transition.py       # Transition protocol + CalyxResourcesTransition (real CLI)
│   ├── search.py           # MCTS (UCT) + Node + SearchResult
│   └── metrics/
│       ├── base.py         # Metric ABC + NodeHeuristic (weighted combine)
│       └── area.py         # AreaMetric: estimated hardware area in bits
└── tests/
    ├── test_area.py
    ├── test_heuristic.py
    └── test_search.py
```

## The search

`MCTS` runs standard UCT: **select** (descend by UCT) → **expand** (apply one
untried pass) → **rollout** (random legal passes to a depth cap, valued by the
*best* state seen, since you may stop optimizing anytime) → **backpropagate**.
A node's path from the root is a concrete pass sequence, so the returned best is
a reproducible recipe.

Reward is any `CalyxState -> float` (maximized) — pass `NodeHeuristic.score`.
Transitions are injected (`(state, pass) -> CalyxState | None`); `None` prunes
illegal / mis-ordered passes. `CalyxResourcesTransition` is the real one
(replays `calyx <src> -p ... -b resources`); tests use a synthetic model.

### Termination (guaranteed three ways)

| Limit | Effect |
|---|---|
| `iteration_budget` | hard cap on MCTS iterations (`run()` loops at most this many) |
| `max_depth` | no pass sequence longer than this; nodes at the limit are terminal; bounds every rollout |
| `time_limit_s` | optional wall-clock cutoff, checked per iteration |

Any one of these stops the search; a node whose legal moves are all exhausted
becomes a dead-end terminal as well.

## The area metric

`calyx_faithful_bits` reproduces Calyx's own `resources` backend
(`estimated_size`): it sums state-holding bits — registers (`W`) and memories
(`W × slots`) — and treats combinational logic (adders, muxes, multipliers) as
free. Cheap, no simulation/synthesis, and it accepts the real backend's CSV via
`parse_resources_csv`, so you can wire it to `calyx <file> -b resources` later.

Extending the cost model is just swapping the `coster` callable
(`PrimitiveInstance -> bits`) — e.g. charge a multiplier as `width²` — with no
changes to the metric or the (future) search. See
`test_custom_coster_extends_baseline`.

## Where this is going

- More metrics: cycle count (via cider/fud sim), synthesis LUT/FF/DSP/BRAM +
  timing (via `tools/report-parsing/synthrep`), IR-size proxies.
- Each becomes a `Metric`; `NodeHeuristic` combines them (weights encode
  direction — lower-is-better metrics get negative weights).
- Then the MCTS itself: `expand()` applies a pass (`calyx -p <pass> -b calyx`),
  `reward()` calls the heuristic.

## Running the tests

```bash
cd mcts
pytest                      # if pytest is installed
# or, with no dependencies:
python tests/test_area.py
python tests/test_heuristic.py
```
