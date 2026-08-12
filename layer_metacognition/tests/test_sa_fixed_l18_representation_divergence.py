from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from layer_metacognition.run_sa_fixed_l18_representation_divergence import (
    REQUIRED_OUTPUTS,
    _existing_complete,
    validate_output,
)
from layer_metacognition.sa_formation.fixed_l18_representation_divergence import (
    AttributionFoldDirection,
    CONTEXTS,
    SourceUseFoldModel,
    association_with_bootstrap,
    direction_geometry,
    fit_source_use_directions,
    fixed_fold_item_bootstrap_indices,
    output_root,
    specialization_contrasts,
)
from layer_metacognition.sa_formation.reliance_external_representation import (
    ExplicitNuisanceEncoder,
)


def _development_rows(n: int = 40) -> tuple[list[dict], np.ndarray]:
    rng = np.random.default_rng(11)
    latent = rng.normal(size=n)
    hidden = rng.normal(scale=0.15, size=(n, 12))
    hidden[:, 0] += 1.8 * latent
    hidden[:, 1] -= 0.8 * latent
    rows = []
    for index in range(n):
        side = "image" if index % 2 else "text"
        rows.append(
            {
                "case_id": f"dev_{index:03d}",
                "item_id": str(index),
                "fold": index % 5,
                "answer_star": "blue" if side == "image" else "green",
                "answer_star_side": side,
                "condition": "conflict_easy" if index % 3 else "conflict_hard",
                "difficulty": "easy" if index % 3 else "hard",
                "prior_strength": float((index % 4) / 5),
                "full_margin": float(1 + index % 7),
                "behavior_delete_imageward": float(latent[index] + 0.1 * rng.normal()),
                "behavior_replace_imageward": float(0.8 * latent[index] + 0.1 * rng.normal()),
            }
        )
    return rows, hidden


def test_fixed_l18_fit_is_nested_development_only_and_saves_preprocessing(
    tmp_path: Path,
) -> None:
    rows, hidden = _development_rows()
    models, oof, audit = fit_source_use_directions(
        rows, hidden, tmp_path / "directions"
    )
    assert sorted(models) == [0, 1, 2, 3, 4]
    assert len(oof) == len(rows)
    assert audit["all_outer_item_overlaps_empty"] is True
    assert audit["all_alpha_selection_development_only"] is True
    assert audit["confirmatory_values_used_for_fit_or_selection"] is False
    for fold, model in models.items():
        assert model.fold == fold
        assert model.alpha in {0.1, 1.0, 10.0, 100.0, 1000.0}
        assert model.audit["heldout_fold_excluded_from_fit"] is True
        assert model.audit["item_overlap"] == []
        assert np.isclose(np.linalg.norm(model.unit_direction), 1.0)
        assert model.train_z_sd > 0
        entry = tmp_path / "directions" / (
            f"fold_{fold}_layer_18_post_answer_raw_choice_coupled.npz"
        )
        with np.load(entry, allow_pickle=False) as payload:
            assert "hidden_nuisance_beta" in payload
            assert "target_nuisance_beta" in payload
            assert "feature_mean" in payload
            assert "feature_scale" in payload
    assert (tmp_path / "directions" / "index.json").is_file()


def _dummy_u(fold: int, direction: np.ndarray) -> SourceUseFoldModel:
    hidden = len(direction)
    return SourceUseFoldModel(
        fold=fold,
        alpha=1.0,
        ridge_coefficient=direction.copy(),
        ridge_intercept=0.0,
        raw_direction=direction.copy(),
        unit_direction=direction.copy(),
        hidden_nuisance_beta=np.zeros((1, hidden)),
        feature_mean=np.zeros(hidden),
        feature_scale=np.ones(hidden),
        target_nuisance_beta=np.zeros((1, 2)),
        target_residual_mean=np.zeros(2),
        target_residual_scale=np.ones(2),
        nuisance_encoder=ExplicitNuisanceEncoder(
            answer_vocabulary=("blue", "green"),
            answer_reference="blue",
            prior_mean=0.0,
            prior_scale=1.0,
            margin_mean=0.0,
            margin_scale=1.0,
            columns=(
                "intercept",
                "choice_image",
                "choice_other",
                "difficulty_hard",
                "prior_strength",
                "full_margin",
                "answer=green",
            ),
        ),
        train_z_mean=0.0,
        train_z_sd=1.0,
        audit={},
    )


def _dummy_a(tmp_path: Path, fold: int, direction: np.ndarray) -> AttributionFoldDirection:
    source = tmp_path / f"a_{fold}.npz"
    np.savez(source, value=np.asarray(fold))
    return AttributionFoldDirection(
        fold=fold,
        d_raw=direction.copy(),
        d_unit=direction.copy(),
        raw_intercept=0.0,
        train_z_mean=0.0,
        train_z_sd=1.0,
        source_file=source,
        source_sha256="synthetic",
    )


def test_direction_geometry_reports_signed_cosine_and_orthogonal_retention(
    tmp_path: Path,
) -> None:
    u = np.asarray([1.0, 0.0, 0.0])
    a = np.asarray([0.2, np.sqrt(0.96), 0.0])
    source = {fold: _dummy_u(fold, u) for fold in range(5)}
    attribution = {fold: _dummy_a(tmp_path, fold, a) for fold in range(5)}
    geometry = direction_geometry(source, attribution)
    assert geometry["cosine_mean"] == pytest.approx(0.2)
    assert geometry["orthogonalized_direction_retention_minimum"] == pytest.approx(
        np.sqrt(0.96)
    )
    for row in geometry["folds"]:
        assert abs(row["u_perp_dot_a"]) < 1e-12
        assert abs(row["a_perp_dot_u"]) < 1e-12


def _bootstrap_rows(n: int = 40) -> list[dict]:
    return [
        {
            "case_id": f"case_{index}",
            "item_id": str(index),
            "fold": index % 5,
            "answer_star_side": "image" if index % 2 else "text",
        }
        for index in range(n)
    ]


def test_fixed_fold_bootstrap_and_paired_centered_specialization_are_deterministic() -> None:
    rows = _bootstrap_rows()
    sides = np.asarray([row["answer_star_side"] for row in rows])
    b = np.linspace(-2, 2, len(rows)) + 0.3 * (sides == "image")
    v = np.sin(np.linspace(-2, 2, len(rows))) - 0.2 * (sides == "image")
    u = b + 0.03 * np.cos(np.arange(len(rows)))
    a = v + 0.03 * np.sin(np.arange(len(rows)))
    samples1 = fixed_fold_item_bootstrap_indices(rows, iterations=40, seed=42)
    samples2 = fixed_fold_item_bootstrap_indices(rows, iterations=40, seed=42)
    assert all(np.array_equal(x, y) for x, y in zip(samples1, samples2))
    association = association_with_bootstrap(
        u, b, sides, samples1, centered=True
    )
    assert association["cluster_unit"] == "item_id"
    assert association["stratified_by"] == "fixed_fold"
    assert association["centering_refit_in_each_bootstrap_sample"] is True
    contrasts = specialization_contrasts(
        u, a, b, v, sides, samples1, centered=True
    )
    double = contrasts["symmetric_double_specialization"]
    assert double["estimate"] > 0
    assert double["paired_same_item_resamples"] is True
    assert double["valid_bootstrap"] == 40


def test_context_contract_output_path_and_existing_output_protection(
    tmp_path: Path,
) -> None:
    assert CONTEXTS == (
        "answer_only_pre_answer",
        "answer_only_post_answer",
        "postquery_prefix",
        "joint_common9",
        "joint_core_consensus",
    )
    expected = output_root(tmp_path)
    assert validate_output(tmp_path) == expected
    with pytest.raises(ValueError, match="fixed"):
        validate_output(tmp_path, str(tmp_path / "wrong"))

    expected.mkdir(parents=True)
    for name in REQUIRED_OUTPUTS:
        path = expected / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    (expected / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "config_fingerprint": "config",
                "input_aggregate_sha256": "input",
                "n": 76,
            }
        ),
        encoding="utf-8",
    )
    reused = _existing_complete(
        expected, config_fingerprint="config", input_sha256="input"
    )
    assert reused is not None and reused["n"] == 76
    with pytest.raises(FileExistsError, match="different"):
        _existing_complete(
            expected, config_fingerprint="changed", input_sha256="input"
        )
