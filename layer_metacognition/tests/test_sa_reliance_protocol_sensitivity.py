from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from layer_metacognition.sa_formation.reliance_protocol_sensitivity import (
    ProtocolSensitivityConfig,
    apply_full_margin_calibration,
    fit_full_margin_calibration,
    run_reliance_protocol_sensitivity,
)


def _rows(split: str, start: int, n: int, *, seed: int) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    donor_pool = [str(value) for value in range(900, 920)]
    for offset in range(n):
        item = start + offset
        fold = offset % 5
        side = "image" if offset % 2 else "text"
        answer = "red" if offset % 3 else "blue"
        difficulty = "hard" if offset % 4 else "easy"
        prior = float((offset % 3) / 5.0)
        margin = float(0.5 + (offset % 7))
        latent = float(rng.normal())
        # The two methods share latent reliance but have opposing margin
        # nuisance.  Removing Full margin should improve their agreement.
        deletion = latent + 1.5 * margin + 1.0 * (side == "image") + rng.normal(scale=0.15)
        replacement = latent - 1.2 * margin + 0.8 * (side == "image") + rng.normal(scale=0.15)
        donor_noise_1 = float(rng.normal(scale=0.6))
        donor_noise_2 = float(rng.normal(scale=0.6))
        rows.append(
            {
                "split": split,
                "case_id": f"{split}_{item}",
                "item_id": str(item),
                "fold": fold,
                "answer_star": answer,
                "answer_star_side": side,
                "difficulty": difficulty,
                "prior_strength": prior,
                "full_margin": margin,
                "behavior_delete_imageward": deletion,
                "behavior_replace_imageward": replacement,
                "behavior_replace_imageward_d1": replacement + donor_noise_1,
                "behavior_replace_imageward_d2": replacement + donor_noise_2,
                "donor1_item_id": donor_pool[(2 * offset) % len(donor_pool)],
                "donor2_item_id": donor_pool[(2 * offset + 1) % len(donor_pool)],
                "graded_residual_delete": deletion,
                "graded_residual_replace": replacement,
            }
        )
    return rows


def test_full_margin_calibration_is_development_fold_only_and_non_mutating() -> None:
    development = _rows("development", 0, 50, seed=1)
    original = [dict(row) for row in development]
    calibration = fit_full_margin_calibration(
        development,
        answer_vocabulary=("blue", "red"),
    )
    assert "full_margin" in calibration["nuisance"]["feature_names"]
    for fold, audit in calibration["folds"].items():
        assert audit["train_n"] == 40
        assert audit["development_test_n"] == 10
        held_out = {row["item_id"] for row in development if row["fold"] == int(fold)}
        assert not held_out.intersection(
            row["item_id"] for row in development if row["fold"] != int(fold)
        )
    transformed = apply_full_margin_calibration(development, calibration)
    assert development == original
    assert all("full_margin_residual_delete" in row for row in transformed)
    assert all("full_margin_residual_delete" not in row for row in development)


def test_unseen_vocabulary_column_is_explicitly_rank_deficient() -> None:
    development = _rows("development", 0, 50, seed=11)
    calibration = fit_full_margin_calibration(
        development,
        answer_vocabulary=("blue", "green", "red"),
    )
    for audit in calibration["folds"].values():
        assert audit["design_rank"] < audit["design_columns"]
        assert audit["condition_number"] is None


def test_sensitivity_writes_only_caller_path_and_preserves_original_gate(tmp_path: Path) -> None:
    development = _rows("development", 0, 50, seed=2)
    confirmatory = _rows("confirmatory", 100, 35, seed=3)
    output = tmp_path / "nested" / "sensitivity.json"
    result = run_reliance_protocol_sensitivity(
        development,
        confirmatory,
        output,
        config=ProtocolSensitivityConfig(bootstrap_iterations=40, loo_top_n=3),
        answer_vocabulary=("blue", "red"),
    )
    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert result["original_gate"]["overwritten"] is False
    assert result["original_gate"]["replacement_authorized"] is False
    assert result["input_audit"]["item_overlap_n"] == 0
    assert sum(result["input_audit"]["confirmatory_n_by_fold"].values()) == len(confirmatory)
    # The calibration embedded in the report remains reusable and its
    # fingerprint is not invalidated by application-only audit fields.
    apply_full_margin_calibration(confirmatory, result["calibration"])
    confirm = result["splits"]["confirmatory"]
    assert confirm["full_margin_adjusted"]["pearson"] > 0.75
    assert confirm["full_margin_adjusted_loo"]["n"] == len(confirmatory)
    assert confirm["donors"]["slots"] == 2 * len(confirmatory)
    assert confirm["donors"]["maximum_reuse"] > 1
    assert confirm["donors"]["raw_reliability"]["icc_two_donor_average"] is not None
    assert "warning" in confirm["donors"]["iid_spearman_brown_extrapolation_from_adjusted_single_donor"]


def test_sensitivity_rejects_development_confirmatory_item_overlap(tmp_path: Path) -> None:
    development = _rows("development", 0, 25, seed=4)
    confirmatory = _rows("confirmatory", 20, 25, seed=5)
    with pytest.raises(ValueError, match="overlap"):
        run_reliance_protocol_sensitivity(
            development,
            confirmatory,
            tmp_path / "should_not_exist.json",
            config=ProtocolSensitivityConfig(bootstrap_iterations=0),
            answer_vocabulary=("blue", "red"),
        )
    assert not (tmp_path / "should_not_exist.json").exists()
