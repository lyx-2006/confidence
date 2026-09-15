from __future__ import annotations

import sys
from pathlib import Path


REVIEW_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = REVIEW_ROOT.parent
for path in (REPOSITORY_ROOT, REVIEW_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

