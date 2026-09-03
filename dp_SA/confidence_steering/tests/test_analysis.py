from __future__ import annotations

from dp_SA.confidence_steering.analyze import _plot_by_direction, family_draws, summarize


def test_shared_family_bootstrap_and_aggregation_modes() -> None:
    rows = [
        {"family_id": "f1", "item_id": "a", "fixed_answer": "red", "answer_origin": "follow_text", "condition": "conflict_easy", "value": 1.0},
        {"family_id": "f1", "item_id": "b", "fixed_answer": "red", "answer_origin": "follow_image", "condition": "conflict_hard", "value": 3.0},
        {"family_id": "f2", "item_id": "c", "fixed_answer": "blue", "answer_origin": "follow_text", "condition": "conflict_easy", "value": 5.0},
    ]
    draws, fingerprint = family_draws(rows, 20)
    assert len(draws) == 20 and fingerprint == family_draws(rows, 20)[1]
    result = summarize(rows, "value", "family_micro", draws)
    assert result["mean_delta"] == 3.5 and result["family_count"] == 2


def test_plot_accepts_smoke_alpha_subset(tmp_path) -> None:
    rows = []
    for direction in ("confidence_raw", "confidence_perp_difficulty", "confidence_perp_sa", "confidence_perp_difficulty_sa"):
        for layer in (12, 14):
            for alpha in (-2.0, 0.0, 2.0):
                rows.append({"group": "answer_equal_macro", "direction": direction, "layer": layer,
                             "alpha": alpha, "mean_delta": 0.0, "ci95_low": -0.1, "ci95_high": 0.1})
    destinations = _plot_by_direction(rows, tmp_path / "final", "delta")
    assert len(destinations) == 4
    assert all(destination.is_file() for destination in destinations)
