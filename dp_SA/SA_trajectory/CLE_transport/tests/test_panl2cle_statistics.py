from __future__ import annotations

import numpy as np
import pytest

from dp_SA.SA_trajectory.CLE_transport.panl2cle_statistics import (
    distribution_distances, enrich_trial, probabilities, sa_state, sign_flip_p,
)


def test_probability_distances_and_states():
    logits = np.arange(9, dtype=float)
    assert probabilities(logits).sum() == pytest.approx(1.0)
    js, tv = distribution_distances(logits, logits)
    assert js == pytest.approx(0.0) and tv == pytest.approx(0.0)
    assert sa_state(.001) == "SA_stable"
    assert sa_state(.00101) == "SA_up"
    assert sa_state(-.00101) == "SA_down"


def test_enriched_hard_shift_and_sign_flip():
    row = {"experiment": "PANL2CLE", "case_id": "a", "family_id": "f",
           "item_id": "a", "answer": "brown", "test_side": "high_image",
           "condition": "C1_main_block", "window_name": "W1", "window_start": 8,
           "window_end": 12, "delta_soft_sa": .02,
           "clean_class_logits": list(range(9)), "blocked_class_logits": list(range(1, 10)),
           "clean_hard_sa_class": 2, "blocked_hard_sa_class": 5}
    result = enrich_trial(row)
    assert result["ordinal_shift"] == 3 and result["abs_ordinal_shift"] == 3
    assert result["hard_shift_up"] == 1 and result["large_ordinal_shift_rate"] == 1
    manifest = {"f": {"test_answer": "brown", "test_side": "high_image"}}
    assert 0 < sign_flip_p({"f": 1.0}, "answer_equal_macro", manifest, repeats=20) <= 1
