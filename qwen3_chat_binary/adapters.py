"""Explicit imports of the reviewed Qwen3 hook implementation."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
REVIEW_ROOT = REPOSITORY_ROOT / "qwen3 review"
if str(REVIEW_ROOT) not in sys.path:
    sys.path.insert(0, str(REVIEW_ROOT))

from Steering.hooks import SelectedHiddenCapture  # noqa: E402
from layer_metacognition.model_adapter import AdditiveActivationHook  # noqa: E402

__all__ = ["AdditiveActivationHook", "SelectedHiddenCapture"]

