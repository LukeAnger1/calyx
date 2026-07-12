"""Area estimate for a Calyx program.

The baseline (`calyx_faithful_bits`) reproduces exactly what Calyx's own
`resources` backend computes in `estimated_size`: it sums the state-holding
bits — registers and memories — and treats every combinational primitive as
free. That is a deliberately loose estimate (the backend even carries a
`TODO: Add other primitives`), but it's cheap, needs no simulation or
synthesis, and is a reasonable first search signal.

The "coster" is a plain `PrimitiveInstance -> bits` function, so extending the
model later (charge multipliers ~width^2, adders ~width, etc.) is just swapping
in a different callable — no changes to the metric or the search. See
`tests/test_area.py` for how to layer a custom cost onto the baseline.
"""

from __future__ import annotations

import csv
import io
import math
import re
from typing import Callable

from calyx_mcts.metrics.base import Metric
from calyx_mcts.state import CalyxState, PrimitiveInstance

#: A cost model: bits contributed by ONE instance (count is applied by the caller).
Coster = Callable[[PrimitiveInstance], float]

_MEM_RE = re.compile(r"^(?:comb|seq)_mem_d([1-4])$")


def calyx_faithful_bits(inst: PrimitiveInstance) -> float:
    """Bits for one instance, matching Calyx's `resources` backend.

    - `std_reg(W)`            -> W
    - `(comb|seq)_mem_dN(W, s0..s_{N-1}, ...)` -> W * (s0 * ... * s_{N-1})
      (only the N size params count; the trailing IDX_SIZE params are ignored,
       exactly as the backend does)
    - anything else (adders, logic, muxes, multipliers, ...) -> 0
    """
    values = inst.param_values()
    if not values:
        return 0.0
    if inst.name == "std_reg":
        return float(values[0])
    m = _MEM_RE.match(inst.name)
    if m:
        dims = int(m.group(1))
        width = values[0]
        slots = math.prod(values[1 : 1 + dims]) if len(values) > dims else 1
        return float(width * slots)
    return 0.0


def estimate_area_bits(
    state: CalyxState,
    *,
    coster: Coster = calyx_faithful_bits,
    include_external: bool = False,
) -> float:
    """Total estimated area (in bits) for `state`.

    External memories are tracked separately by the backend and excluded by
    default here (they're off-chip); pass `include_external=True` to fold them
    in.
    """
    total = 0.0
    for inst in state.primitives:
        if inst.external and not include_external:
            continue
        total += inst.count * coster(inst)
    return total


class AreaMetric(Metric):
    """Estimated hardware area, in bits. Lower is better."""

    name = "area_bits"
    higher_is_better = False

    def __init__(
        self,
        coster: Coster = calyx_faithful_bits,
        *,
        include_external: bool = False,
    ) -> None:
        self.coster = coster
        self.include_external = include_external

    def evaluate(self, state: CalyxState) -> float:
        return estimate_area_bits(
            state, coster=self.coster, include_external=self.include_external
        )


def _parse_params(field: str) -> dict[str, int]:
    """Parse the `resources` CSV Parameters column, e.g. 'WIDTH: 32. SIZE: 4. '."""
    params: dict[str, int] = {}
    for chunk in field.split("."):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        key, _, val = chunk.partition(":")
        params[key.strip()] = int(val.strip())
    return params


def parse_resources_csv(text: str) -> list[PrimitiveInstance]:
    """Build primitives from `calyx <file> -b resources` CSV output.

    Header: Primitive, Count, External?, Parameters. This lets you feed the real
    backend's output straight into `estimate_area_bits`; tests use synthetic
    CSVs so they run without the compiler.
    """
    reader = csv.DictReader(io.StringIO(text))
    out: list[PrimitiveInstance] = []
    for row in reader:
        out.append(
            PrimitiveInstance(
                name=row["Primitive"].strip(),
                params=_parse_params(row.get("Parameters", "") or ""),
                count=int(row["Count"]),
                external=row.get("External?", "no").strip().lower() == "yes",
            )
        )
    return out
