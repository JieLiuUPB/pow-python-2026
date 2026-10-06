"""Compatibility entry point for the repository-level :mod:`pow_simulation`.

The implementation lives at the repository root.  This module keeps the
historical ``OCW.pow_simulation`` import path and direct script invocation
working for tests and downstream users.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from pow_simulation import *  # noqa: F401,F403,E402
from pow_simulation import main as _main  # noqa: E402


if __name__ == "__main__":
    _main()
