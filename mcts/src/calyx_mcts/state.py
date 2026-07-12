"""State that an MCTS node carries.

For now a node's state is just the set of hardware primitives instantiated by
the Calyx program (enough to estimate area). This will grow: the raw `.futil`
text, the sequence of passes applied to reach this node, and cached results of
other metrics (cycle count, synthesis numbers) all belong here eventually.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PrimitiveInstance:
    """One (kind of) primitive instantiated in a Calyx program.

    `params` mirrors the ordered parameter binding Calyx reports for a
    primitive, e.g. `std_reg(32)` -> {"WIDTH": 32} and
    `comb_mem_d2(32, 4, 4, 3, 3)` -> {"WIDTH": 32, "D0_SIZE": 4, ...}.
    Order matters: memory area rules read parameters positionally, exactly as
    the Calyx `resources` backend does.

    `count` lets a single instance stand in for N identical primitives (the
    `resources` backend aggregates by (name, params), so this matches its
    output directly).
    """

    name: str
    params: dict[str, int] = field(default_factory=dict)
    count: int = 1
    external: bool = False

    def param_values(self) -> list[int]:
        """Parameter values in declaration order (dicts preserve insertion order)."""
        return list(self.params.values())


@dataclass
class CalyxState:
    """The Calyx program a node represents.

    Metrics read from this. Right now only `primitives` is populated (from the
    `resources` backend), but the extra fields are the hooks for the metrics and
    the per-node heuristic we'll add next.
    """

    primitives: list[PrimitiveInstance] = field(default_factory=list)
    futil: str | None = None
    #: passes applied (in order) to reach this state from the root
    applied_passes: tuple[str, ...] = ()
