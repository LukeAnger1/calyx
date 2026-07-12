"""How an MCTS edge turns one Calyx state into the next by applying a pass.

The search depends only on the `Transition` protocol
(`(state, pass_name) -> CalyxState | None`, where `None` means "illegal move,
prune"), so tests inject a fast synthetic model and real runs use the compiler.

`CalyxResourcesTransition` is the real one: it replays the whole pass sequence
through the `calyx` binary and reads the `resources` backend to recover the
program's primitives (enough for the area metric). Replaying from source each
time avoids the pitfalls of re-parsing emitted `.futil` between passes.
"""

from __future__ import annotations

import subprocess
from typing import Optional, Protocol, runtime_checkable

from calyx_mcts.metrics.area import parse_resources_csv
from calyx_mcts.state import CalyxState


@runtime_checkable
class Transition(Protocol):
    def __call__(self, state: CalyxState, pass_name: str) -> Optional[CalyxState]:
        """Apply `pass_name`; return the new state, or None if the move is illegal."""
        ...


class CalyxResourcesTransition:
    """Real transition backed by the `calyx` CLI + `resources` backend.

    Each state records the pass sequence applied so far; applying a new pass
    re-runs `calyx <source> -p ... -b resources` with the extended sequence.
    A non-zero exit (a pass that errors or is mis-ordered) yields `None`.
    """

    def __init__(self, source_path: str, lib_path: Optional[str] = None,
                 calyx: str = "calyx", timeout_s: float = 60.0) -> None:
        self.source_path = source_path
        self.lib_path = lib_path
        self.calyx = calyx
        self.timeout_s = timeout_s

    def _run(self, passes: tuple[str, ...]) -> Optional[CalyxState]:
        cmd = [self.calyx, self.source_path, "-b", "resources"]
        if self.lib_path:
            cmd += ["-l", self.lib_path]
        for p in passes:
            cmd += ["-p", p]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout_s
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None
        if proc.returncode != 0:
            return None
        return CalyxState(
            primitives=parse_resources_csv(proc.stdout),
            applied_passes=passes,
        )

    def initial_state(self) -> Optional[CalyxState]:
        """Root state: the program with no extra optimization passes applied."""
        return self._run(())

    def __call__(self, state: CalyxState, pass_name: str) -> Optional[CalyxState]:
        return self._run(state.applied_passes + (pass_name,))
