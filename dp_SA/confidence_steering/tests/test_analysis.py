from __future__ import annotations

from dp_SA.confidence_steering.analyze import _plot_by_direction, component_additivity_rows, family_draws, symmetric_rows, summarize
from dp_SA.confidence_steering.run_spec import normalize_run_spec


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


def test_plot_writes_only_selected_natural_directions(tmp_path) -> None:
    directions = ("confidence_raw", "confidence_parallel_sa", "confidence_perp_sa_natural_scale")
    rows = [
        {"group": "answer_equal_macro", "direction": direction, "layer": layer,
         "alpha": alpha, "mean_delta": 0.0, "ci95_low": -0.1, "ci95_high": 0.1}
        for direction in directions for layer in (14, 16) for alpha in (-2.0, -1.0, 0.0, 1.0, 2.0)
    ]
    destinations = _plot_by_direction(rows, tmp_path / "panl", "delta", directions)
    assert {path.stem for path in destinations} == set(directions)
    assert {path.stem for path in (tmp_path / "panl").glob("*.png")} == set(directions)


def test_dynamic_symmetric_doses_and_component_additivity() -> None:
    spec = normalize_run_spec(
        ["confidence_raw", "confidence_parallel_sa", "confidence_perp_sa_natural_scale"],
        [14], [-2, -1, 0, 1, 2])
    rows = []
    for case, family, answer in (("a", "f1", "red"), ("b", "f2", "blue")):
        base = {"case_id": case, "item_id": case, "family_id": family, "condition": "conflict_easy", "answer_origin": "follow_text", "fixed_answer": answer, "layer": 14}
        for alpha in spec["alphas"]:
            for direction, factor in (("confidence_raw", 3.0), ("confidence_parallel_sa", 1.0), ("confidence_perp_sa_natural_scale", 2.0)):
                rows.append({**base, "direction": direction, "alpha": alpha, "delta_final_soft_sa": factor * alpha, "delta_panl_probe_sa": factor * alpha})
    draws, _ = family_draws(rows, 20)
    symmetric = symmetric_rows(rows, spec, draws, ["delta_final_soft_sa"])
    assert {row["dose"] for row in symmetric} == {1.0, 2.0}
    additivity = component_additivity_rows(rows, spec, draws)
    assert additivity and max(abs(row["additivity_error"]) for row in additivity) < 1e-12
