"""Pytest bootstrap for the plain-module data_generation package.

Tests import ``generation_runtime`` / ``generation_v2`` / ... as top-level
modules, so this directory must be on ``sys.path``; the pipeline also imports
repository packages such as ``confidence_test``, so the repo root is added too.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for entry in (HERE, HERE.parent):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))
