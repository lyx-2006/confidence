from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SHORT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = SHORT_ROOT.parent.parent
for candidate in (REPOSITORY_ROOT, SHORT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from capture.short_prompt import CLASS_LIST_ANCHOR, PHASE1_TEMPLATE_SHORT
from capture_reverse.reverse_prompt import PHASE1_TEMPLATE_SHORT_REVERSE, phase1_prompt_short_reverse
from capture_reverse.scoring import REVERSE_MIDPOINTS, reverse_soft_sa_from_logits
from dp_SA.config import MIDPOINTS


def test_reverse_template_changes_only_direction_sentence():
    ordinary = PHASE1_TEMPLATE_SHORT.replace(
        "A higher class indicates stronger image contribution. A lower class indicates stronger text contribution.",
        "A higher class indicates stronger text contribution. A lower class indicates stronger image contribution.",
    )
    assert PHASE1_TEMPLATE_SHORT_REVERSE == ordinary
    assert PHASE1_TEMPLATE_SHORT_REVERSE.count(CLASS_LIST_ANCHOR) == 1
    rendered = phase1_prompt_short_reverse("Q", "T", "A")
    assert "Question:\nQ" in rendered and "Text clue:\nT" in rendered and "**Answer**: A" in rendered


def test_reverse_midpoint_table_is_exact_reversal():
    assert REVERSE_MIDPOINTS == tuple(reversed(MIDPOINTS))
    assert REVERSE_MIDPOINTS[0] == MIDPOINTS[8]
    assert REVERSE_MIDPOINTS[8] == MIDPOINTS[0]


def test_reverse_score_raw_and_canonical_labels():
    logits = np.full(20, -100.0)
    ids = list(range(9))
    logits[0] = 100.0
    result = reverse_soft_sa_from_logits(logits, ids)
    assert result["raw_argmax_class"] == 0
    assert result["argmax_hard_class"] == 8
    assert result["soft_sa_image_score"] == REVERSE_MIDPOINTS[0]
    logits[0], logits[8] = -100.0, 100.0
    result = reverse_soft_sa_from_logits(logits, ids)
    assert result["raw_argmax_class"] == 8
    assert result["argmax_hard_class"] == 0
    assert result["soft_sa_image_score"] == REVERSE_MIDPOINTS[8]

