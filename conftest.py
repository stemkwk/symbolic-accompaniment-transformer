"""Root conftest.py — pytest session-wide fixtures and sys.path setup.

Ensures the project root is always on sys.path *before* any test module is
imported, so that ``from scripts.prepare_data import ...`` works regardless of
collection order.  test_dynamics.py does a ``sys.path.insert(0, scripts/)``
followed by ``sys.path.pop(0)``; without this file that pop would accidentally
remove the root that pytest prepended, breaking the ``scripts.*`` namespace
imports in test_basics.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
