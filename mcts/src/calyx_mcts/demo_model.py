"""A self-contained synthetic Calyx-like model for demoing the search.

No `calyx` binary or data file needed. The root looks like a small accelerator
(register file, multipliers, adders, a memory); the passes actually shrink the
area metric, so the search has something real to optimize and the numbers it
reports are meaningful and reproducible.

Area here uses the *extended* cost model (multipliers charged ~width², which the
faithful Calyx `resources` backend does not do) precisely so that a
resource-sharing pass has a measurable payoff — this is the extension seam from
`metrics/area.py` in action.
"""

from __future__ import annotations

from calyx_mcts.metrics.area import calyx_faithful_bits, estimate_area_bits
from calyx_mcts.state import CalyxState, PrimitiveInstance

#: Passes the demo search may apply. `canonicalize` is an always-legal,
#: area-neutral distractor (like the many real passes that don't touch area).
DEMO_ACTIONS: tuple[str, ...] = (
    "cell-share",         # share multipliers  (big area win)
    "infer-share",        # share registers    (area win)
    "dead-cell-removal",  # drop an adder      (area-neutral here; a legal branch)
    "canonicalize",       # no-op distractor   (always legal, no change)
)


def demo_coster(inst: PrimitiveInstance) -> float:
    """Charge multipliers ~width²; everything else at the faithful baseline."""
    if inst.name == "std_mult_pipe":
        return float(inst.param_values()[0] ** 2)
    return calyx_faithful_bits(inst)


def demo_area(state: CalyxState) -> float:
    return estimate_area_bits(state, coster=demo_coster)


def demo_reward(state: CalyxState) -> float:
    """Maximize reward == minimize area."""
    return -demo_area(state)


def make_root() -> CalyxState:
    """A small accelerator: register file, 4 multipliers, adders, a memory."""
    return CalyxState(primitives=[
        PrimitiveInstance("std_reg", {"WIDTH": 32}, count=8),                    # 256
        PrimitiveInstance("std_mult_pipe", {"WIDTH": 32}, count=4),              # 4096
        PrimitiveInstance("std_add", {"WIDTH": 32}, count=4),                    # 0
        PrimitiveInstance(
            "comb_mem_d2",
            {"WIDTH": 32, "D0_SIZE": 4, "D1_SIZE": 4, "D0_IDX_SIZE": 3, "D1_IDX_SIZE": 3},
            count=2,
        ),                                                                        # 1024
    ])  # total area = 5376


def _halve_first(prims, name):
    out, changed = [], False
    for p in prims:
        if p.name == name and p.count >= 2 and not changed:
            out.append(PrimitiveInstance(p.name, dict(p.params), p.count // 2, p.external))
            changed = True
        else:
            out.append(p)
    return out if changed else None


def demo_transition(state: CalyxState, pass_name: str):
    """Apply a demo pass; return the new state or None (illegal / no effect)."""
    prims = state.primitives
    new = None
    if pass_name == "cell-share":
        new = _halve_first(prims, "std_mult_pipe")
    elif pass_name == "infer-share":
        new = _halve_first(prims, "std_reg")
    elif pass_name == "dead-cell-removal":
        out, dropped = [], False
        for p in prims:
            if p.name == "std_add" and p.count >= 1 and not dropped:
                dropped = True
                if p.count > 1:
                    out.append(PrimitiveInstance(p.name, dict(p.params), p.count - 1, p.external))
            else:
                out.append(p)
        new = out if dropped else None
    elif pass_name == "canonicalize":
        new = list(prims)  # always legal, no structural change
    if new is None:
        return None
    return CalyxState(primitives=new, applied_passes=state.applied_passes + (pass_name,))


def optimal_area(root: CalyxState) -> float:
    """Greedy fixpoint of the area-reducing passes = the reachable optimum.

    The reductions are monotone and independent, so greedily sharing until
    nothing changes reaches the true minimum — a ground truth to score the
    search against.
    """
    state = root
    while True:
        best_next = None
        for action in ("cell-share", "infer-share"):
            nxt = demo_transition(state, action)
            if nxt is not None and (best_next is None or demo_area(nxt) < demo_area(best_next)):
                best_next = nxt
        if best_next is None or demo_area(best_next) >= demo_area(state):
            return demo_area(state)
        state = best_next
