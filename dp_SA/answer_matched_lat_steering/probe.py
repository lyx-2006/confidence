from __future__ import annotations

import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import BOOTSTRAP_REPEATS, LAYERS, POSITIONS, RESULTS_ROOT, SEED, SMOKE_BOOTSTRAP_REPEATS, SMOKE_LAYERS
from .io_utils import atomic_csv, atomic_json, atomic_jsonl, canonical_hash, load_jsonl, sha256_file


METRICS = ("r2", "pearson", "spearman", "mae")


def _hidden(root: Path, row: dict[str, Any], position: str, layer: int) -> np.ndarray:
    key = f"{position}__L{layer}"
    with np.load(root / row["hidden_file"]) as payload:
        if key not in payload.files:
            raise KeyError(f"Missing hidden key {key} for {row['case_id']}")
        return np.asarray(payload[key], dtype=np.float32)


def _metrics(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    pearson = np.nan if np.ptp(y) == 0 or np.ptp(pred) == 0 else pearsonr(y, pred).statistic
    spearman = np.nan if len(np.unique(y)) < 2 or len(np.unique(pred)) < 2 else spearmanr(y, pred).statistic
    return np.asarray([
        r2_score(y, pred),
        pearson,
        spearman,
        mean_absolute_error(y, pred),
    ], dtype=float)


def shared_family_draws(families: Sequence[str], repeats: int, seed: int) -> tuple[list[str], np.ndarray]:
    ordered = sorted(set(map(str, families)))
    if len(ordered) < 2:
        raise ValueError("Family bootstrap requires at least two families")
    return ordered, np.random.default_rng(seed).integers(0, len(ordered), size=(repeats, len(ordered)))


def cluster_bootstrap_metrics(
    y: np.ndarray,
    pred: np.ndarray,
    family: Sequence[str],
    ordered_families: Sequence[str],
    draws: np.ndarray,
) -> np.ndarray:
    by_family = {name: np.flatnonzero(np.asarray(family) == name) for name in ordered_families}
    output = []
    for draw in draws:
        indices = np.concatenate([by_family[ordered_families[index]] for index in draw])
        values = _metrics(y[indices], pred[indices])
        if np.isfinite(values).all():
            output.append(values)
    if not output:
        raise ValueError("No finite family bootstrap repeats")
    return np.asarray(output)


def paired_cluster_bootstrap_difference(y: np.ndarray, left: np.ndarray, right: np.ndarray, family: Sequence[str], ordered_families: Sequence[str], draws: np.ndarray) -> np.ndarray:
    family_array = np.asarray(family); by_family = {name: np.flatnonzero(family_array == name) for name in ordered_families}; output = []
    for draw in draws:
        indices = np.concatenate([by_family[ordered_families[index]] for index in draw]); left_value = _metrics(y[indices], left[indices]); right_value = _metrics(y[indices], right[indices])
        if np.isfinite(left_value).all() and np.isfinite(right_value).all(): output.append(right_value - left_value)
    if not output: raise ValueError("No finite paired family bootstrap repeats")
    return np.asarray(output)


def _atomic_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        joblib.dump(value, temporary)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _split_indices(records: Sequence[dict[str, Any]], family_to_fold: dict[str, int], fold: int) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    train = np.asarray([family_to_fold[str(row["family_id"])] != fold for row in records])
    test = ~train
    if not train.any() or not test.any():
        raise ValueError(f"Probe fold {fold} has an empty train/test side")
    audit = {}
    for field, label in (("family_id", "family"), ("item_id", "item"), ("image_sha256", "image_hash")):
        left = {str(records[i][field]) for i in np.flatnonzero(train)}
        right = {str(records[i][field]) for i in np.flatnonzero(test)}
        audit[f"{label}_leakage_count"] = len(left & right)
    if any(audit.values()):
        raise ValueError(f"Probe leakage at fold {fold}: {audit}")
    return train, test, audit


def run_probe(*, output_root: Path = RESULTS_ROOT, smoke: bool = False, resume: bool = False, repeats: int | None = None) -> dict[str, Any]:
    root = Path(output_root)
    clean_path = root / "artifacts" / "diagnostics" / "clean_capture.jsonl"
    candidate_path = root / "artifacts" / "manifests" / "candidate_manifest.jsonl"
    fold_path = root / "artifacts" / "manifests" / "fold_assignments.jsonl"
    candidates = load_jsonl(candidate_path)
    clean = {str(row["case_id"]): row for row in load_jsonl(clean_path) if row.get("status") == "completed"}
    if {str(row["case_id"]) for row in candidates} != set(clean):
        raise ValueError("Probe requires complete joint clean capture")
    records = [{**row, **{k: clean[str(row["case_id"])][k] for k in ("hidden_file", "soft_sa_image_score")}} for row in candidates]
    fold_rows = load_jsonl(fold_path)
    family_to_fold = {str(row["family_id"]): int(row["fold"]) for row in fold_rows}
    if {str(row["family_id"]) for row in records} - set(family_to_fold):
        raise ValueError("Candidate family is missing a fold assignment")
    folds = sorted({value for value in family_to_fold.values() if value >= 0})
    if not smoke and folds != list(range(15)):
        raise ValueError(f"Formal probe requires folds 0..14, got {folds}")
    layers = SMOKE_LAYERS if smoke else LAYERS
    repeats = int(repeats or (SMOKE_BOOTSTRAP_REPEATS if smoke else BOOTSTRAP_REPEATS))
    config = {
        "format_version": 1, "smoke_only": smoke, "positions": list(POSITIONS), "layers": list(layers),
        "folds": folds, "bootstrap_repeats": repeats, "seed": SEED,
        "clean_sha256": sha256_file(clean_path), "candidate_sha256": sha256_file(candidate_path), "fold_sha256": sha256_file(fold_path),
    }
    config["fingerprint"] = canonical_hash(config)
    progress_path = root / "progress" / "probe_progress.json"
    table_path = root / "tables" / "table2_lat_panl_probe.csv"
    oof_path = root / "artifacts" / "probe" / "oof_predictions.jsonl"
    if progress_path.exists():
        previous = json.loads(progress_path.read_text())
        if previous.get("config_fingerprint") != config["fingerprint"]:
            raise ValueError("Probe resume fingerprint mismatch")
        if resume and previous.get("status") == "complete" and table_path.is_file() and oof_path.is_file():
            for relative, expected_sha in previous.get("model_files", {}).items():
                path = root / relative
                if not path.is_file() or sha256_file(path) != expected_sha: raise ValueError(f"Probe fold model fingerprint mismatch: {relative}")
            return {**previous, "resumed_noop": True}
        if not resume:
            raise FileExistsError("Probe artifacts exist; use --resume")

    atomic_json(progress_path, {"status": "running", "config_fingerprint": config["fingerprint"]})
    y_all = np.asarray([float(row["soft_sa_image_score"]) for row in records])
    family_all = np.asarray([str(row["family_id"]) for row in records])
    evaluated_mask = np.asarray([family_to_fold[family] in folds for family in family_all])
    eval_families, draws = shared_family_draws(family_all[evaluated_mask].tolist(), repeats, SEED + 2000)
    predictions: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    prediction_cache: dict[tuple[str, int], np.ndarray] = {}
    split_audits = []
    for fold in folds:
        _train, _test, audit = _split_indices(records, family_to_fold, fold)
        split_audits.append({"fold": fold, **audit})
    for position in POSITIONS:
        for layer in layers:
            X = np.stack([_hidden(root, row, position, int(layer)) for row in records])
            pred = np.full(len(records), np.nan, dtype=float)
            for fold in folds:
                train, test, _audit = _split_indices(records, family_to_fold, fold)
                model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0, solver="lsqr"))])
                model.fit(X[train], y_all[train])
                pred[test] = model.predict(X[test])
                model_path = root / "artifacts" / "probe" / "fold_models" / f"{position}__L{layer}__fold_{fold:02d}.joblib"
                _atomic_joblib(model_path, model)
                for index in np.flatnonzero(test):
                    predictions.append({"case_id": records[index]["case_id"], "family_id": records[index]["family_id"], "item_id": str(records[index]["item_id"]), "position": position, "layer": int(layer), "fold": fold, "target": float(y_all[index]), "prediction": float(pred[index]), "model_file": str(model_path.relative_to(root)), "model_sha256": sha256_file(model_path), "config_fingerprint": config["fingerprint"]})
            if not np.isfinite(pred[evaluated_mask]).all():
                raise ValueError(f"Incomplete OOF predictions for {position} L{layer}")
            observed = _metrics(y_all[evaluated_mask], pred[evaluated_mask])
            if not np.isfinite(observed).all():
                raise ValueError(f"Non-finite probe metric for {position} L{layer}")
            boot = cluster_bootstrap_metrics(y_all[evaluated_mask], pred[evaluated_mask], family_all[evaluated_mask], eval_families, draws)
            ci = np.percentile(boot, [2.5, 97.5], axis=0)
            metric_rows.append({
                "position": position, "layer": int(layer),
                "r2": observed[0], "r2_ci_low": ci[0, 0], "r2_ci_high": ci[1, 0],
                "pearson": observed[1], "pearson_ci_low": ci[0, 1], "pearson_ci_high": ci[1, 1],
                "spearman": observed[2], "spearman_ci_low": ci[0, 2], "spearman_ci_high": ci[1, 2],
                "mae": observed[3], "mae_ci_low": ci[0, 3], "mae_ci_high": ci[1, 3],
                "sample_count": int(evaluated_mask.sum()), "family_count": len(eval_families), "valid_bootstrap_repeats": len(boot),
            })
            prediction_cache[position, int(layer)] = pred.copy()
    contrasts = []
    for layer in layers:
        lat = prediction_cache["P1_LAT", int(layer)][evaluated_mask]
        panl = prediction_cache["P1_PANL", int(layer)][evaluated_mask]
        lat_observed = _metrics(y_all[evaluated_mask], lat); panl_observed = _metrics(y_all[evaluated_mask], panl)
        difference = paired_cluster_bootstrap_difference(y_all[evaluated_mask], lat, panl, family_all[evaluated_mask], eval_families, draws); valid = len(difference)
        for index, metric in enumerate(METRICS):
            low, high = np.percentile(difference[:, index], [2.5, 97.5])
            contrasts.append({"layer": int(layer), "metric": metric, "panl_minus_lat": float(panl_observed[index] - lat_observed[index]), "ci_low": float(low), "ci_high": float(high), "valid_bootstrap_repeats": valid})
    atomic_jsonl(oof_path, predictions)
    atomic_csv(table_path, metric_rows)
    atomic_csv(root / "artifacts" / "diagnostics" / "probe_position_contrasts.csv", contrasts)
    atomic_json(root / "artifacts" / "diagnostics" / "probe_split_audit.json", {"status": "passed", "folds": split_audits})
    model_files = {str(path.relative_to(root)): sha256_file(path) for path in sorted((root / "artifacts" / "probe" / "fold_models").glob("*.joblib"))}
    summary = {"status": "complete", "smoke_only": smoke, "cell_count": len(metric_rows), "expected_cell_count": len(POSITIONS) * len(layers), "prediction_count": len(predictions), "bootstrap_repeats": repeats, "model_files": model_files, "config_fingerprint": config["fingerprint"], "resumed_noop": False}
    atomic_json(progress_path, summary)
    return summary
