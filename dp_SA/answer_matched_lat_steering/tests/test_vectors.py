from __future__ import annotations

import numpy as np
import pytest

from dp_SA.answer_matched_lat_steering.vectors import (
    family_equal_mean, loao_direction, recipient_target_norm, scale_direction, shuffled_answer_direction,
)
from dp_SA.answer_matched_lat_steering.config import VECTOR_NORM_FRACTION


def test_family_equal_mean_does_not_weight_record_rich_family_more():
    mean, count = family_equal_mean({"a": [np.array([0.0]), np.array([2.0])], "b": [np.array([5.0])]})
    assert count == 2 and mean[0] == pytest.approx(3.0)


def test_loao_excludes_recipient_without_prenormalization():
    raw, included = loao_direction({"red": np.array([100.0, 0.0]), "blue": np.array([1.0, 0.0]), "green": np.array([3.0, 0.0]), "cyan": np.array([5.0, 0.0])}, "red")
    assert included == ["blue", "cyan", "green"]
    assert raw[0] == pytest.approx(3.0)


def test_scaling_and_family_cell_shuffle_are_deterministic():
    scaled = scale_direction(np.array([3.0, 4.0]), 0.3)
    assert np.linalg.norm(scaled) == pytest.approx(0.3)
    cells = [("a", "high_text", np.array([0.0, 1.0])), ("b", "high_text", np.array([1.0, 1.0])), ("c", "high_image", np.array([3.0, 1.0])), ("d", "high_image", np.array([4.0, 1.0]))]
    first, counts = shuffled_answer_direction(cells, seed="42")
    second, _ = shuffled_answer_direction(cells, seed="42")
    assert np.array_equal(first, second) and counts == {"high_text": 2, "high_image": 2}


def test_loao_rejects_fewer_than_three_other_answers():
    with pytest.raises(ValueError, match="only 2"):
        loao_direction({"red": np.ones(2), "blue": np.ones(2), "green": np.ones(2)}, "red")


def test_recipient_target_norm_uses_only_loao_included_answers():
    cells = {
        ("f1", "red", "high_text"): np.array([100.0, 0.0]),
        ("f2", "blue", "high_text"): np.array([3.0, 4.0]),
        ("f3", "green", "high_image"): np.array([0.0, 10.0]),
    }
    value = recipient_target_norm(cells, ["blue", "green"])
    assert value == pytest.approx(VECTOR_NORM_FRACTION * 7.5)
