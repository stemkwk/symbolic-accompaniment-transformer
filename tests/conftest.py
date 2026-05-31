"""pytest fixtures + path setup for the test suite.

Primary path setup is ``pythonpath = ["."]`` in ``pyproject.toml`` (puts the
project root on sys.path so ``from scripts.prepare_data import ...`` works —
scripts/ is a namespace dir, not an installed package). This conftest keeps a
belt-and-suspenders insert so the suite also works if pytest is invoked without
the pyproject config picked up.

It also guards against test_dynamics.py's ``sys.path.insert(0, scripts/)`` /
``sys.path.pop(0)`` dance accidentally dropping the root between modules.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent   # tests/ -> project root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
