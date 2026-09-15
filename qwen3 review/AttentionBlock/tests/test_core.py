from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch

from AttentionBlock.analyze import bh_fdr
from AttentionBlock.config import EXPERIMENTS
from AttentionBlock.contracts import add_cle_plus_1, trial_key
from AttentionBlock.run import class_margin, edge_for_condition
from AttentionBlock.selection import select_manifest


def _rows() -> list[dict]:
    rows = []
    for index in range(140):
        rows.append({
            "status": "completed", "valid_class": True, "item_id": str(index),
            "prior_index": 0, "condition": "conflict_easy", "version": "v4",
            "argmax_hard_class": 0 if index < 70 else 8,
            "phase0_normalized_answer": f"color{index % 12}", "case_id": f"c{index}",
        })
    return rows


def test_balanced_item_disjoint_selection_is_deterministic():
    first, summary = select_manifest(_rows())
    second, _ = select_manifest(_rows())
    assert [row["case_id"] for row in first] == [row["case_id"] for row in second]
    assert Counter(row["test_side"] for row in first) == {"text_side": 50, "image_side": 50}
    assert len({row["item_id"] for row in first}) == 100
    assert summary["unique_item_count"] == 100


def test_frozen_inclusive_overlapping_windows():
    assert EXPERIMENTS["CLE2SAC"].windows == ((12, 16), (16, 20), (20, 24), (24, 28))
    assert EXPERIMENTS["PANL2CLE"].windows == ((8, 12), (12, 16), (16, 20), (20, 24))
    for spec in EXPERIMENTS.values():
        assert all(left[1] == right[0] for left, right in zip(spec.windows, spec.windows[1:]))
        assert all(len(range(start, end + 1)) == end - start + 1 for start, end in spec.windows)


def test_edge_direction_and_controls():
    positions = {"P1_SAC": 20, "P1_CLASS_LIST_END": 15,
                 "P1_CLASS_LIST_END_PLUS_1": 16, "P1_PANL": 7, "P1_PANL_PLUS_1": 8}
    assert edge_for_condition("CLE2SAC", "C1_main_block", positions)[0].pairs == ((20, 15),)
    assert edge_for_condition("CLE2SAC", "C2_source_plus_1_control", positions)[0].pairs == ((20, 16),)
    assert edge_for_condition("PANL2CLE", "C1_main_block", positions)[0].pairs == ((15, 7),)
    assert edge_for_condition("PANL2CLE", "C2_source_plus_1_control", positions)[0].pairs == ((15, 8),)


class _Tokenizer:
    def decode(self, values, **_kwargs):
        return f"T{values[0]}"


def test_cle_plus_one_and_causal_order():
    located = {
        "P1_PANL": {"processed_index": 2}, "P1_PANL_PLUS_1": {"processed_index": 3},
        "P1_CLASS_LIST_END": {"processed_index": 8}, "P1_SAC": {"processed_index": 12},
    }
    result = add_cle_plus_1(located, _Tokenizer(), torch.arange(20)[None])
    assert result["P1_CLASS_LIST_END_PLUS_1"]["processed_index"] == 9
    assert result["P1_CLASS_LIST_END_PLUS_1"]["token_id"] == 9


def test_metric_contracts_and_trial_key():
    logits = list(map(float, range(9)))
    assert class_margin(logits, 8) == pytest.approx(4.5)
    assert trial_key("c", "C0_clean") == "c|C0_clean"
    assert trial_key("c", "C1", (12, 16)) == "c|C1|L12-16"
    q = bh_fdr([0.01, 0.04, 0.03, 0.2])
    assert len(q) == 4 and all(0 <= value <= 1 for value in q)

