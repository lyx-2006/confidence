from __future__ import annotations

import numpy as np

from ..analyze import _random_cosines, casewise_effects, shared_family_draws, summarize


def _trial(alpha, value):
    return {"seed": 45, "case_id": "c", "family_id": "f", "item_id": "i", "fixed_answer": "red", "direction": "confidence_raw", "alpha": alpha, "final_soft_sa": value, "anchor_panl_sa": value, "anchor_lat_confidence": value, "seed_panl_sa": value, "seed_lat_confidence": value}


def test_symmetric_effect():
    rows = casewise_effects([_trial(-0.5, 1.0), _trial(0.5, 3.0)])
    assert all(row["effect"] == 1.0 for row in rows)


def test_shared_bootstrap_is_deterministic():
    first, first_hash = shared_family_draws(["a", "b"], 10)
    second, second_hash = shared_family_draws(["b", "a"], 10)
    assert first == second and first_hash == second_hash


def test_answer_equal_macro_weights_answers_equally():
    rows = [
        {"family_id": "f1", "fixed_answer": "red", "effect": 0.0},
        {"family_id": "f2", "fixed_answer": "blue", "effect": 2.0},
        {"family_id": "f3", "fixed_answer": "blue", "effect": 4.0},
    ]
    draws, _ = shared_family_draws(["f1", "f2", "f3"], 5)
    assert summarize(rows, "answer_equal_macro", draws)["mean_effect"] == 1.5


def test_family_micro_weights_families_equally():
    rows = [
        {"family_id": "f1", "fixed_answer": "red", "effect": 0.0},
        {"family_id": "f1", "fixed_answer": "red", "effect": 2.0},
        {"family_id": "f2", "fixed_answer": "blue", "effect": 5.0},
    ]
    draws, _ = shared_family_draws(["f1", "f2"], 5)
    assert summarize(rows, "family_micro", draws)["mean_effect"] == 3.0


def test_alpha_zero_rows_are_not_treated_as_direction_effects():
    zero = _trial(0.0, 2.0); zero["direction"] = "shared_alpha_zero"
    assert len(casewise_effects([zero, _trial(-0.5, 1.0), _trial(0.5, 3.0)])) == 5


def test_random_cosine_reference_is_deterministic_and_signed():
    first = _random_cosines(20, 32)
    second = _random_cosines(20, 32)
    assert np.array_equal(first, second)
    assert np.any(first < 0) and np.any(first > 0)
