from __future__ import annotations

import sys
from pathlib import Path

REVIEW = Path(__file__).resolve().parents[3]
for path in (REVIEW, REVIEW.parent):
    if str(path) not in sys.path: sys.path.insert(0, str(path))
