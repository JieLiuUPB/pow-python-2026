"""Compatibility entry point for the repository-level :mod:`ocw_w_sweep`.

The implementation lives at the repository root.  This module preserves both
the historical ``OCW.ocw_w_sweep`` import path and direct script invocation.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from ocw_w_sweep import *  # noqa: F401,F403,E402
from ocw_w_sweep import main as _main  # noqa: E402


if __name__ == "__main__":
    _main()
