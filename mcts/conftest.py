"""Put the src/ layout on the import path so `import calyx_mcts` works.

Lets `pytest` run from the `mcts/` folder without an editable install.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
