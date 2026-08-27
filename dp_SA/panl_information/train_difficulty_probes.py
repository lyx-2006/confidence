from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ARTIFACT_NAMES, BOOTSTRAP_REPEATS, LAYERS, PERMUTATION_REPEATS, POSITIONS, RESULTS_ROOT, RIDGE_ALPHA, SEED
from .io_utils import atomic_json, atomic_jsonl, ensure_output_layout, stage_update
from .metrics import bh_fdr, pooled_r2
from .probe_utils import feature_matrix, fixed_oof_cluster_permutation, load_probe_rows, regression_metrics, stable_seed


def _fit_oof(X: np.ndarray, y: np.ndarray, rows: Sequence[dict[str, Any]], model_paths: dict[int, Path]) -> np.ndarray:
    prediction = np.empty(len(rows), dtype=float)
    for fold in range(5):
        train = np.asarray([int(row["outer_fold"]) != fold for row in rows])
        test = ~train
        if not train.any() or not test.any():
            raise ValueError(f"Outer fold {fold} is empty")
        mean, scale = float(y[train].mean()), float(y[train].std(ddof=0))
        if scale <= 0:
            raise ValueError(f"Outer-train target is constant in fold {fold}")
        model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=RIDGE_ALPHA, solver="lsqr"))])
        model.fit(X[train], (y[train] - mean) / scale)
        prediction[test] = model.predict(X[test]) * scale + mean
        model_paths[fold].parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "target_mean": mean, "target_scale": scale, "fold": fold}, model_paths[fold])
    return prediction


def _metadata_baseline(rows: Sequence[dict[str, Any]], y: np.ndarray, fields: Sequence[str]) -> tuple[np.ndarray, dict[str, Any]]:
    X = np.asarray([[str(row[field]) for field in fields] for row in rows], dtype=object)
    prediction = np.empty(len(rows), dtype=float)
    fold_details = {}
    for fold in range(5):
        train = np.asarray([int(row["outer_fold"]) != fold for row in rows]); test = ~train
        mean, scale = float(y[train].mean()), float(y[train].std(ddof=0))
        transform = ColumnTransformer([("categories", OneHotEncoder(handle_unknown="ignore"), list(range(len(fields))))])
        model = Pipeline([("features", transform), ("ridge", Ridge(alpha=RIDGE_ALPHA, solver="lsqr"))])
        model.fit(X[train], (y[train] - mean) / scale)
        prediction[test] = model.predict(X[test]) * scale + mean
        fold_details[str(fold)] = {"target_mean": mean, "target_scale": scale}
    return prediction, fold_details


def train_difficulty_probes(
    root: Path, *, bootstrap: int = BOOTSTRAP_REPEATS, permutations: int = PERMUTATION_REPEATS,
    positions: Sequence[str] = POSITIONS, layers: Sequence[int] = LAYERS,
) -> dict[str, Any]:
    rows, capture = load_probe_rows(root)
    metrics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    stage_update(root, "difficulty_probes", "running", cell_count=2 * len(positions) * len(layers))
    for target in ("text", "image"):
        field = f"{target}_model_perceived_difficulty"
        y = np.asarray([float(row[field]) for row in rows], dtype=float)
        metadata_fields = ("prior_bin", "text_answer") if target == "text" else ("condition", "image_answer")
        metadata_prediction, metadata_folds = _metadata_baseline(rows, y, metadata_fields)
        metadata_metrics = regression_metrics(rows, y, metadata_prediction, repeats=bootstrap, seed=stable_seed(SEED, target, "metadata"))
        atomic_json(root / "artifacts" / f"{target}_difficulty_metadata_baseline.json", {"fields": list(metadata_fields), "metrics": metadata_metrics, "folds": metadata_folds})
        for position in positions:
            for layer in layers:
                X = feature_matrix(root, rows, capture, position, int(layer))
                paths = {fold: root / "artifacts" / "probe_models" / f"{target}__{position}__L{layer}__fold{fold}.joblib" for fold in range(5)}
                prediction = _fit_oof(X, y, rows, paths)
                cell = regression_metrics(rows, y, prediction, repeats=bootstrap, seed=stable_seed(SEED, target, position, layer))
                permutation = fixed_oof_cluster_permutation(rows, y, prediction, target=target, metric=pooled_r2, repeats=permutations, seed=stable_seed(SEED, target, position, layer, "permutation"))
                cell.update({"target": target, "position": position, "layer": int(layer), "permutation": permutation, "metadata_baseline_r2": metadata_metrics["r2"]})
                metrics.append(cell)
                for row, truth, predicted in zip(rows, y, prediction, strict=True):
                    predictions.append({"case_id": row["case_id"], "item_id": row["item_id"], "prior_index": row["prior_index"], "condition": row["condition"], "outer_fold": row["outer_fold"], "target_name": target, "target": float(truth), "prediction": float(predicted), "position": position, "layer": int(layer)})
    for target in ("text", "image"):
        selected = [row for row in metrics if row["target"] == target]
        adjusted = bh_fdr([row["permutation"]["p_value"] for row in selected])
        for row, q in zip(selected, adjusted, strict=True):
            row["permutation"]["q_value"] = float(q)
            row["permutation"]["fdr_significant"] = bool(q < 0.05)
    onsets: dict[str, dict[str, Any]] = {}
    for target in ("text", "image"):
        onsets[target] = {}
        for position in positions:
            cells = sorted((row for row in metrics if row["target"] == target and row["position"] == position), key=lambda row: row["layer"])
            onset = None
            for first, second in zip(cells, cells[1:]):
                if all(cell["r2"] > 0 and cell["r2_ci"]["lower"] is not None and cell["r2_ci"]["lower"] > 0 and cell["spearman"] > 0 for cell in (first, second)):
                    onset = {"layer": first["layer"], "layers": [first["layer"], second["layer"]]}; break
            onsets[target][position] = onset
    atomic_jsonl(root / "artifacts" / ARTIFACT_NAMES["difficulty_oof"], predictions)
    summary = {"status": "complete", "metrics": metrics, "onsets": onsets, "prediction_count": len(predictions), "sample_count": len(rows), "item_count": len({str(row['item_id']) for row in rows})}
    atomic_json(root / "artifacts" / "difficulty_probe_metrics.json", summary)
    stage_update(root, "difficulty_probes", "complete", prediction_count=len(predictions))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv); root = ensure_output_layout(RESULTS_ROOT, resume=args.resume)
    print(json.dumps(train_difficulty_probes(root), ensure_ascii=False))
    return 0


if __name__ == "__main__": raise SystemExit(main())
