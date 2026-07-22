"""Experiment confidence schema derived from the existing shared constants.

The colour-pool module may annotate display labels with numeric intervals.
This experiment's wire format requires the canonical class names, so display
suffixes are removed without changing the source module or class ordering.
"""

from __future__ import annotations

import re

from confidence_analysis import (
    CLASS_MIDPOINTS as SOURCE_CLASS_MIDPOINTS,
    CONFIDENCE_CLASSES as SOURCE_CONFIDENCE_CLASSES,
    CONFIDENCE_CLASS_TEXT as SOURCE_CONFIDENCE_CLASS_TEXT,
)


def _canonical_label(label: str) -> str:
    return re.sub(r"\s*\(\s*\d(?:\.\d+)?\s*-\s*\d(?:\.\d+)?\s*\)\s*$", "", label).strip()


CONFIDENCE_CLASSES = [_canonical_label(label) for label in SOURCE_CONFIDENCE_CLASSES]
CLASS_MIDPOINTS = [float(value) for value in SOURCE_CLASS_MIDPOINTS]

_normalized_lines = [
    re.sub(
        r"\s*\(\s*\d(?:\.\d+)?\s*-\s*\d(?:\.\d+)?\s*\)\s*$",
        "",
        line,
    )
    for line in SOURCE_CONFIDENCE_CLASS_TEXT.splitlines()
]
CONFIDENCE_CLASS_TEXT = "\n".join(_normalized_lines)

if len(CONFIDENCE_CLASSES) != 10 or len(set(CONFIDENCE_CLASSES)) != 10:
    raise RuntimeError(f"Expected ten distinct confidence classes, got {CONFIDENCE_CLASSES}")
if len(CLASS_MIDPOINTS) != len(CONFIDENCE_CLASSES):
    raise RuntimeError("Confidence class and midpoint counts differ")
