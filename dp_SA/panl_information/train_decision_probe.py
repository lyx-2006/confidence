from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from layer_metacognition.probe.probe_models import build_current_answer_baseline, build_hidden_state_probe, choose_regularization_C

from .config import ARTIFACT_NAMES, BOOTSTRAP_REPEATS, C_GRID, LAYERS, PERMUTATION_REPEATS, POSITIONS, RESULTS_ROOT, SEED
from .io_utils import atomic_json, atomic_jsonl, ensure_output_layout, stage_update
from .metrics import bh_fdr
from .probe_utils import classification_metrics, feature_matrix, fixed_oof_cluster_permutation, load_probe_rows, majority_oof, stable_seed


def decision_records(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected, excluded = [], []
    for row in rows:
        if row.get("decision_side") in {"follow_text", "follow_image"}:
            selected.append(row)
        else:
            excluded.append({"case_id": row["case_id"], "item_id": row["item_id"], "reason": row.get("decision_exclusion_reason") or "invalid_decision_side"})
    return selected, excluded


def _hidden_oof(X: np.ndarray, y: np.ndarray, rows: Sequence[dict[str, Any]], paths: dict[int, Path]) -> tuple[np.ndarray, dict[str, Any]]:
    probability = np.empty(len(rows), dtype=float); details = {}
    for fold in range(5):
        train = np.asarray([int(row["outer_fold"]) != fold for row in rows]); test = ~train
        selected_C, selection = choose_regularization_C(X[train], [str(value) for value in y[train]], [str(rows[i]["item_id"]) for i in np.flatnonzero(train)], seed=stable_seed(SEED, "decision", fold), c_grid=C_GRID, n_splits=3)
        model = build_hidden_state_probe(selected_C); model.fit(X[train], y[train])
        probability[test] = model.predict_proba(X[test])[:, list(model.named_steps["classifier"].classes_).index(1)]
        paths[fold].parent.mkdir(parents=True, exist_ok=True); joblib.dump(model, paths[fold])
        details[str(fold)] = {"selected_C": selected_C, "selection": selection}
    return probability, details


def _answer_identity_oof(rows: Sequence[dict[str, Any]], y: np.ndarray) -> np.ndarray:
    X = np.asarray([[str(row["phase0_normalized_answer"])] for row in rows], dtype=object); probability = np.empty(len(rows))
    for fold in range(5):
        train = np.asarray([int(row["outer_fold"]) != fold for row in rows]); test = ~train
        model = build_current_answer_baseline(); model.fit(X[train], y[train])
        classes = list(model.named_steps["classifier"].classes_); probability[test] = model.predict_proba(X[test])[:, classes.index(1)]
    return probability


def _difficulty_oof(rows: Sequence[dict[str, Any]], y: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    X = np.asarray([[float(row["G"]), float(row["U"]), float(row["Hard"])] for row in rows]); probability = np.empty(len(rows)); details = {}
    for fold in range(5):
        train = np.asarray([int(row["outer_fold"]) != fold for row in rows]); test = ~train
        selected_C, selection = choose_regularization_C(X[train], [str(value) for value in y[train]], [str(rows[i]["item_id"]) for i in np.flatnonzero(train)], seed=stable_seed(SEED, "difficulty_baseline", fold), c_grid=C_GRID, n_splits=3)
        model = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(C=selected_C, penalty="l2", class_weight="balanced", solver="lbfgs", max_iter=5000))])
        model.fit(X[train], y[train]); classes = list(model.named_steps["classifier"].classes_)
        probability[test] = model.predict_proba(X[test])[:, classes.index(1)]
        details[str(fold)] = {"selected_C": selected_C, "selection": selection}
    return probability, details


def train_decision_probe(
    root: Path, *, bootstrap: int = BOOTSTRAP_REPEATS, permutations: int = PERMUTATION_REPEATS,
    positions: Sequence[str] = POSITIONS, layers: Sequence[int] = LAYERS,
) -> dict[str, Any]:
    all_rows, capture = load_probe_rows(root); rows, excluded = decision_records(all_rows)
    y = np.asarray([1 if row["decision_side"] == "follow_image" else 0 for row in rows], dtype=int)
    if len(set(y.tolist())) != 2:
        raise ValueError("Decision probe requires both follow_text and follow_image")
    atomic_jsonl(root / "artifacts" / "decision_excluded_records.jsonl", excluded)
    majority_probability, majority_details = majority_oof(rows, y)
    answer_probability = _answer_identity_oof(rows, y)
    difficulty_probability, difficulty_details = _difficulty_oof(rows, y)
    baseline = {
        "majority": classification_metrics(rows, y, majority_probability, repeats=bootstrap, seed=stable_seed(SEED, "majority")),
        "answer_identity": classification_metrics(rows, y, answer_probability, repeats=bootstrap, seed=stable_seed(SEED, "answer")),
        "difficulty_only": classification_metrics(rows, y, difficulty_probability, repeats=bootstrap, seed=stable_seed(SEED, "difficulty")),
        "majority_fold_details": majority_details, "difficulty_fold_details": difficulty_details,
    }
    atomic_json(root / "artifacts" / "decision_baselines.json", baseline)
    metrics: list[dict[str, Any]] = []; predictions: list[dict[str, Any]] = []
    stage_update(root, "decision_probe", "running", cell_count=len(positions) * len(layers), sample_count=len(rows))
    for position in positions:
        for layer in layers:
            X = feature_matrix(root, rows, capture, position, int(layer))
            paths = {fold: root / "artifacts" / "probe_models" / f"decision__{position}__L{layer}__fold{fold}.joblib" for fold in range(5)}
            probability, fold_details = _hidden_oof(X, y, rows, paths)
            cell = classification_metrics(rows, y, probability, repeats=bootstrap, seed=stable_seed(SEED, "decision", position, layer))
            permutation = fixed_oof_cluster_permutation(rows, y, probability, target="decision", metric=lambda truth, score: float(balanced_accuracy_score(truth.astype(int), score >= 0.5)), repeats=permutations, seed=stable_seed(SEED, "decision", position, layer, "permutation"))
            cell.update({"position": position, "layer": int(layer), "folds": fold_details, "permutation": permutation, "majority_baseline": baseline["majority"]["balanced_accuracy"], "answer_identity_baseline": baseline["answer_identity"]["balanced_accuracy"], "difficulty_only_baseline": baseline["difficulty_only"]["balanced_accuracy"]})
            metrics.append(cell)
            for row, truth, score in zip(rows, y, probability, strict=True):
                predictions.append({"case_id": row["case_id"], "item_id": row["item_id"], "prior_index": row["prior_index"], "condition": row["condition"], "outer_fold": row["outer_fold"], "target": "follow_image" if truth else "follow_text", "target_binary": int(truth), "probability_follow_image": float(score), "prediction": "follow_image" if score >= 0.5 else "follow_text", "position": position, "layer": int(layer)})
    adjusted = bh_fdr([row["permutation"]["p_value"] for row in metrics])
    for row, q in zip(metrics, adjusted, strict=True):
        row["permutation"]["q_value"] = float(q); row["permutation"]["fdr_significant"] = bool(q < 0.05)
    onsets = {}
    for position in positions:
        cells = sorted((row for row in metrics if row["position"] == position), key=lambda row: row["layer"]); onset = None
        for first, second in zip(cells, cells[1:]):
            if all(cell["balanced_accuracy"] > cell["majority_baseline"] and cell["permutation"]["q_value"] < 0.05 and cell["auroc"] > 0.5 and cell["direction"] > 0 for cell in (first, second)):
                onset = {"layer": first["layer"], "layers": [first["layer"], second["layer"]]}; break
        onsets[position] = onset
    atomic_jsonl(root / "artifacts" / ARTIFACT_NAMES["decision_oof"], predictions)
    summary = {"status": "complete", "metrics": metrics, "onsets": onsets, "baselines": baseline, "sample_count": len(rows), "item_count": len({str(row['item_id']) for row in rows}), "excluded_count": len(excluded), "prediction_count": len(predictions)}
    atomic_json(root / "artifacts" / "decision_probe_metrics.json", summary)
    stage_update(root, "decision_probe", "complete", prediction_count=len(predictions))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv); root = ensure_output_layout(RESULTS_ROOT, resume=args.resume)
    print(json.dumps(train_decision_probe(root), ensure_ascii=False))
    return 0


if __name__ == "__main__": raise SystemExit(main())
