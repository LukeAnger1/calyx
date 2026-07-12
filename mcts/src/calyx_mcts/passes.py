"""The MCTS action space: Calyx optimization passes.

These are real pass names (the string each pass's `name()` returns, usable as
`calyx -p <name>`). The set is curated to *optimization / rewrite* passes that
meaningfully trade off area vs. latency and are interesting to reorder — not the
mandatory lowering steps (`tdcc`, `compile-*`, `*-insertion`, `lower-guards`,
...) that a normal pipeline runs unconditionally.

Illegal or ill-ordered choices are handled by the transition model returning
`None` (the compiler errors), so the search can include anything here safely.
"""

from __future__ import annotations

import subprocess

#: Curated optimization passes to explore/reorder.
DEFAULT_PASSES: tuple[str, ...] = (
    "comb-prop",
    "constant-port-prop",
    "dead-assign-removal",
    "dead-group-removal",
    "dead-cell-removal",
    "cell-share",
    "infer-share",
    "collapse-control",
    "group2seq",
    "group2invoke",
    "schedule-compaction",
    "static-inference",
    "static-promotion",
    "simplify-with-control",
    "simplify-guards",
    "canonicalize",
)


def list_passes_from_cli(calyx: str = "calyx") -> list[str]:
    """Enumerate every pass the installed `calyx` binary knows about.

    Convenience for building a larger action space from the real compiler;
    falls back to raising if `calyx` isn't on PATH. Not used by the tests.
    """
    out = subprocess.run(
        [calyx, "--list-passes"], capture_output=True, text=True, check=True
    ).stdout
    names: list[str] = []
    for line in out.splitlines():
        line = line.strip()
        # lines look like "  pass-name : description" or bare "pass-name"
        if not line or line.endswith(":") or " " not in line and "-" not in line:
            continue
        token = line.split()[0].strip(":")
        if token and all(c.islower() or c.isdigit() or c == "-" for c in token):
            names.append(token)
    return names
