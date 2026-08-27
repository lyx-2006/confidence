from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dp_SA.panl_information.analyze import _plots, _wide_probe_tables, fit_parameterizations
from dp_SA.panl_information.config import LAYERS, POSITIONS
from dp_SA.panl_information.metrics import item_bootstrap, pooled_r2
from dp_SA.panl_information.probe_utils import fixed_oof_cluster_permutation
from dp_SA.panl_information.train_decision_probe import decision_records
from dp_SA.panl_information.train_difficulty_probes import _fit_oof


def _rows(items: int = 10) -> list[dict[str, object]]:
    rows = []
    for item in range(items):
        for condition in ("conflict_easy", "conflict_hard"):
            dt = (item + 1) / (items + 1)
            di = .15 + .1 * (condition == "conflict_hard") + .03 * ((item * item + 1) % 7)
            rows.append({"case_id": f"{item}__prior_0__{condition}", "item_id": str(item), "prior_index": 0, "condition": condition, "outer_fold": item % 5, "d_text": dt, "d_image": di, "G": dt - di, "U": (dt + di) / 2, "Hard": int(condition == "conflict_hard"), "final_sa": .1 + .5 * (dt - di) + .2 * ((dt + di) / 2), "decision_side": "follow_image" if item % 2 else "follow_text", "phase0_normalized_answer": "red"})
    return rows


def test_parameterizations_are_equivalent() -> None:
    result = fit_parameterizations(_rows(), "final_sa", repeats=20, seed=42)
    assert result["mapping_audit"]["status"] == "passed"
    assert result["mapping_audit"]["max_fitted_difference"] <= 1e-10


def test_cluster_bootstrap_keeps_items() -> None:
    rows = _rows(5)
    result = item_bootstrap(rows, lambda sample: float(np.mean([row["final_sa"] for row in sample])), repeats=20, seed=42)
    assert result["item_count"] == 5 and result["valid_repeats"] == 20


def test_continuous_oof_probe_has_zero_item_leakage(tmp_path: Path) -> None:
    rows = _rows(10); rng = np.random.default_rng(42); latent = np.asarray([row["d_text"] for row in rows], dtype=float)
    X = np.column_stack([latent, latent * 2, rng.normal(0, .01, len(rows))]); paths = {fold: tmp_path / f"fold{fold}.joblib" for fold in range(5)}
    prediction = _fit_oof(X, latent, rows, paths)
    assert np.isfinite(prediction).all() and pooled_r2(latent, prediction) > .8
    assert all(path.is_file() for path in paths.values())
    for fold in range(5):
        train = {row["item_id"] for row in rows if row["outer_fold"] != fold}; test = {row["item_id"] for row in rows if row["outer_fold"] == fold}
        assert not train.intersection(test)


def test_decision_exclusion_and_cluster_permutation() -> None:
    rows = _rows(10) + [{"case_id": "99__prior_0__conflict_easy", "item_id": "99", "decision_side": None, "decision_exclusion_reason": "matches_neither"}]
    selected, excluded = decision_records(rows)
    assert len(excluded) == 1 and len(selected) == 20
    truth = np.asarray([1 if row["decision_side"] == "follow_image" else 0 for row in selected]); prediction = truth * .8 + .1
    result = fixed_oof_cluster_permutation(selected, truth, prediction, target="decision", metric=lambda y, p: float(np.mean((p >= .5) == y)), repeats=100, seed=42)
    assert result["unique_target_key_count"] == len(selected)
    assert 0 < result["p_value"] <= 1


def _synthetic_probe_payloads() -> tuple[dict, dict]:
    difficulty_metrics = []
    decision_metrics = []
    for position in POSITIONS:
        for layer in LAYERS:
            for target in ("text", "image"):
                difficulty_metrics.append({"target": target, "position": position, "layer": layer, "r2": .1, "r2_ci": {"lower": .01, "upper": .2}, "spearman": .2, "pearson": .2, "mae": 1.0, "sample_count": 20})
            decision_metrics.append({"position": position, "layer": layer, "balanced_accuracy": .6, "balanced_accuracy_ci": {"lower": .51, "upper": .7}, "auroc": .65, "auroc_ci": {"lower": .52, "upper": .75}, "accuracy": .6, "macro_f1": .6, "log_loss": .6, "majority_baseline": .5, "answer_identity_baseline": .55, "difficulty_only_baseline": .56, "sample_count": 20, "item_count": 10})
    difficulty = {"metrics": difficulty_metrics, "onsets": {target: {position: {"layer": 10, "layers": [10, 12]} for position in POSITIONS} for target in ("text", "image")}}
    decision = {"metrics": decision_metrics, "onsets": {position: {"layer": 10, "layers": [10, 12]} for position in POSITIONS}, "baselines": {"majority": {"balanced_accuracy": .5}}}
    return difficulty, decision


def test_wide_tables_and_three_figures(tmp_path: Path) -> None:
    for directory in ("tables", "figures"): (tmp_path / directory).mkdir()
    difficulty, decision = _synthetic_probe_payloads(); _wide_probe_tables(tmp_path, difficulty, decision); _plots(tmp_path, difficulty, decision)
    with (tmp_path / "tables" / "difficulty_probe.csv").open() as handle:
        header = handle.readline().strip().split(",")
    assert header[1] == "P1_AC_L10" and header[-1] == "P1_SAC_L27"
    figures = sorted(path.name for path in (tmp_path / "figures").glob("*.png"))
    assert figures == ["decision_probe_accuracy.png", "difficulty_probe_R2.png", "difficulty_probe_spearman.png"]
