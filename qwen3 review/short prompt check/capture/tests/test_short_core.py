from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from transformers import AutoProcessor

SHORT_ROOT = Path(__file__).resolve().parents[2]
REVIEW_ROOT = SHORT_ROOT.parent
REPOSITORY_ROOT = REVIEW_ROOT.parent
for candidate in (REPOSITORY_ROOT, REVIEW_ROOT, SHORT_ROOT):
    if str(candidate) not in sys.path: sys.path.insert(0, str(candidate))

from capture.positions import external_positions, locate_short_phase1_positions
from capture.run import _context
from capture.short_prompt import CLASS_LIST_ANCHOR, PHASE1_TEMPLATE_SHORT
from steering.analyze import family_mean_ci
from steering.run import select_short_manifests


def test_short_template_has_exact_direction_and_anchor():
    assert "A higher class indicates stronger image contribution" in PHASE1_TEMPLATE_SHORT
    assert PHASE1_TEMPLATE_SHORT.count(CLASS_LIST_ANCHOR) == 1
    assert "0,1,2,3,4,5,6,7,8" in CLASS_LIST_ANCHOR


@pytest.mark.processor
def test_real_fast_processor_and_merged_cle():
    model = REPOSITORY_ROOT / "qwen-3-vl" / "model"
    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    assert processor.tokenizer.is_fast and processor.image_processor.is_fast
    row = json.loads(next(open(REVIEW_ROOT / "capture" / "results.jsonl", encoding="utf-8")))
    _prompt, _rendered, _messages, inputs, located = _context(processor, row)
    positions = external_positions(located)
    order = [positions[name]["processed_index"] for name in ("LAT", "PANL", "CLE", "SAC")]
    assert order == sorted(order) and len(set(order)) == 4
    assert "\n" in positions["CLE"]["token_text"]
    assert positions["CLE"]["newline_merge_policy"] == "processed_token_containing_target_newline"


def test_short_selection_uses_short_scores_and_counts():
    rows = []
    for index in range(180):
        rows.append({"status": "completed", "valid_class": True, "case_id": f"c{index}", "item_id": str(index),
                     "prior_index": 0, "condition": "conflict_easy", "version": "v4",
                     "argmax_hard_class": 0 if index < 70 else 8,
                     "soft_sa_image_score": index / 180, "phase0_correct": False, "answer_length": 4})
    construction, test, summary = select_short_manifests(rows)
    assert summary["construction_counts"] == {"high_image": 25, "high_text": 25}
    assert summary["test_count"] == 80
    assert summary["test_selection"] == "80_item_disjoint_cases_minimizing_abs_soft_sa_minus_0.5"
    distances = [row["distance_to_midpoint"] for row in test]
    assert distances == sorted(distances)
    assert not ({r["item_id"] for r in construction} & {r["item_id"] for r in test})


def test_family_bootstrap_is_deterministic():
    rows = [{"item_id": str(i), "test_side": "image_side", "value": float(i)} for i in range(10)]
    first = family_mean_ci(rows, lambda row: row["value"], repeats=100, seed=42)
    second = family_mean_ci(rows, lambda row: row["value"], repeats=100, seed=42)
    assert first == second and first[0] == pytest.approx(4.5)
