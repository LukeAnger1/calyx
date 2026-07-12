"""Tests for the area estimate.

Runnable two ways:
    pytest                     # from the mcts/ folder
    python tests/test_area.py  # no pytest needed
"""

import os
import sys

# Make `import calyx_mcts` work under plain `python tests/test_area.py` too.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from calyx_mcts.metrics.area import (  # noqa: E402
    AreaMetric,
    calyx_faithful_bits,
    estimate_area_bits,
    parse_resources_csv,
)
from calyx_mcts.state import CalyxState, PrimitiveInstance  # noqa: E402


def test_register_area_is_bitwidth():
    reg = PrimitiveInstance("std_reg", {"WIDTH": 32})
    assert calyx_faithful_bits(reg) == 32


def test_combinational_primitive_is_free():
    # matches the compiler: only state (regs/mems) counts toward the estimate
    add = PrimitiveInstance("std_add", {"WIDTH": 32})
    mult = PrimitiveInstance("std_mult_pipe", {"WIDTH": 32})
    assert calyx_faithful_bits(add) == 0
    assert calyx_faithful_bits(mult) == 0


def test_memory_area_is_width_times_slots():
    # comb_mem_d2(WIDTH, D0_SIZE, D1_SIZE, D0_IDX_SIZE, D1_IDX_SIZE)
    # bits = WIDTH * D0_SIZE * D1_SIZE; the IDX_SIZE params are ignored.
    mem = PrimitiveInstance(
        "comb_mem_d2",
        {"WIDTH": 32, "D0_SIZE": 4, "D1_SIZE": 4, "D0_IDX_SIZE": 3, "D1_IDX_SIZE": 3},
    )
    assert calyx_faithful_bits(mem) == 32 * 16


def test_seq_mem_d1_and_d3_dimensions():
    m1 = PrimitiveInstance("seq_mem_d1", {"WIDTH": 8, "SIZE": 10, "IDX_SIZE": 4})
    assert calyx_faithful_bits(m1) == 8 * 10
    m3 = PrimitiveInstance(
        "seq_mem_d3",
        {"WIDTH": 16, "D0": 2, "D1": 3, "D2": 4, "I0": 1, "I1": 2, "I2": 2},
    )
    assert calyx_faithful_bits(m3) == 16 * (2 * 3 * 4)


def test_count_multiplies_total():
    state = CalyxState(primitives=[PrimitiveInstance("std_reg", {"WIDTH": 8}, count=5)])
    assert estimate_area_bits(state) == 8 * 5


def test_total_over_mixed_program():
    state = CalyxState(
        primitives=[
            PrimitiveInstance("std_reg", {"WIDTH": 32}, count=3),  # 96
            PrimitiveInstance("std_add", {"WIDTH": 32}),           # 0
            PrimitiveInstance("comb_mem_d1", {"WIDTH": 32, "SIZE": 16, "IDX": 4}),  # 512
        ]
    )
    assert estimate_area_bits(state) == 96 + 0 + 512


def test_external_excluded_by_default_and_included_on_request():
    state = CalyxState(
        primitives=[
            PrimitiveInstance("std_reg", {"WIDTH": 32}),  # internal, 32
            PrimitiveInstance("seq_mem_d1", {"WIDTH": 32, "SIZE": 8, "IDX": 3}, external=True),  # 256
        ]
    )
    assert estimate_area_bits(state) == 32
    assert estimate_area_bits(state, include_external=True) == 32 + 256


def test_area_metric_direction_and_value():
    metric = AreaMetric()
    state = CalyxState(primitives=[PrimitiveInstance("std_reg", {"WIDTH": 16})])
    assert metric.name == "area_bits"
    assert metric.higher_is_better is False
    assert metric.evaluate(state) == 16


def test_custom_coster_extends_baseline():
    # The extension seam: charge a multiplier as ~width^2, keep everything else
    # at the faithful baseline. No changes to AreaMetric or the search needed.
    def coster(inst):
        if inst.name == "std_mult_pipe":
            return float(inst.param_values()[0] ** 2)
        return calyx_faithful_bits(inst)

    state = CalyxState(
        primitives=[
            PrimitiveInstance("std_reg", {"WIDTH": 8}),          # 8 (baseline)
            PrimitiveInstance("std_mult_pipe", {"WIDTH": 8}),    # 64 (custom)
        ]
    )
    assert estimate_area_bits(state, coster=coster) == 8 + 64
    # baseline still treats the multiplier as free
    assert estimate_area_bits(state) == 8


def test_parse_resources_csv_matches_estimate():
    csv_text = (
        "Primitive,Count,External?,Parameters\n"
        "std_reg,3,no,WIDTH: 32. \n"
        "comb_mem_d2,1,no,WIDTH: 32. D0_SIZE: 4. D1_SIZE: 4. D0_IDX_SIZE: 3. D1_IDX_SIZE: 3. \n"
        "std_add,2,no,WIDTH: 32. \n"
    )
    prims = parse_resources_csv(csv_text)
    assert len(prims) == 3
    assert prims[0].count == 3 and prims[0].params["WIDTH"] == 32
    state = CalyxState(primitives=prims)
    assert estimate_area_bits(state) == (3 * 32) + (32 * 16) + 0


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
