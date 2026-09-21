from __future__ import annotations

import numpy as np
import pytest

from qwen3_chat_binary.config import VECTOR_NORM_FRACTION
from qwen3_chat_binary.prepare_answer_matched import (
    CANONICAL_ANSWERS,
    fixed_split,
    loao_scaled_direction,
    sa_group,
    smoke_subset,
)


def _rows() -> list[dict]:
    labels = ("0", "1", "2", "3", "4", "tie")
    pairs = ("hard_text_easy_image", "balanced", "hard_image_easy_text")
    rows = []
    for index in range(396):
        rows.append({
            "status": "completed", "case_id": f"case_{index:04d}",
            "phase0_normalized_answer": CANONICAL_ANSWERS[index // 33],
            "predicted_label": labels[index % len(labels)],
            "pair_type": pairs[index % len(pairs)],
        })
    return rows


def test_fixed_answer_matched_split_is_deterministic_and_disjoint() -> None:
    construction, test, summary = fixed_split(_rows())
    construction_again, test_again, _summary_again = fixed_split(_rows())
    assert len(construction) == 317
    assert len(test) == 79
    assert [row["case_id"] for row in construction] == [row["case_id"] for row in construction_again]
    assert [row["case_id"] for row in test] == [row["case_id"] for row in test_again]
    assert {row["case_id"] for row in construction}.isdisjoint({row["case_id"] for row in test})
    assert set(summary["eligible_answers"]) == set(CANONICAL_ANSWERS)
    assert set(row["answer"] for row in test) == set(CANONICAL_ANSWERS)


def test_smoke_subset_covers_answers_pair_types_and_sa_groups() -> None:
    _construction, test, _summary = fixed_split(_rows())
    smoke = smoke_subset(test)
    assert len(smoke) == 20
    assert set(row["answer"] for row in smoke) == set(CANONICAL_ANSWERS)
    assert set(row["pair_type"] for row in smoke) == {
        "hard_text_easy_image", "balanced", "hard_image_easy_text",
    }
    assert set(row["sa_group"] for row in smoke) == {"text", "image", "neutral"}


def test_sa_group_uses_hard_five_class_labels() -> None:
    assert sa_group({"predicted_label": "0"}) == "text"
    assert sa_group({"predicted_label": "1"}) == "text"
    assert sa_group({"predicted_label": "2"}) == "neutral"
    assert sa_group({"predicted_label": "tie"}) == "neutral"
    assert sa_group({"predicted_label": "3"}) == "image"
    assert sa_group({"predicted_label": "4"}) == "image"


def test_loao_direction_excludes_recipient_and_has_three_percent_norm() -> None:
    directions = {
        "black": np.asarray([8.0, 0.0], np.float32),
        "blue": np.asarray([0.0, 3.0], np.float32),
        "brown": np.asarray([0.0, 6.0], np.float32),
        "cyan": np.asarray([3.0, 0.0], np.float32),
    }
    hidden = {
        answer: [np.asarray([3.0, 4.0], np.float32), np.asarray([0.0, 5.0], np.float32)]
        for answer in directions
    }
    raw, scaled, included, metadata = loao_scaled_direction(directions, hidden, "black")
    assert included == ["blue", "brown", "cyan"]
    assert np.allclose(raw, np.mean([directions[answer] for answer in included], axis=0))
    assert np.linalg.norm(scaled) == pytest.approx(metadata["mean_residual_norm"] * VECTOR_NORM_FRACTION)
