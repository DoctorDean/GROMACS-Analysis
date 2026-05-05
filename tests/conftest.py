"""
tests/conftest.py
=================
Pytest configuration for the mdanalysis-gromacs test suite.

Adds the repository root to ``sys.path`` so that ``sim_prep``,
``utils``, and ``gromacs_analysis`` are importable when pytest is
invoked from any directory — including from the repo root, a CI
runner, or a temporary directory.
"""

import sys
from pathlib import Path

# Insert repo root at the front of sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
