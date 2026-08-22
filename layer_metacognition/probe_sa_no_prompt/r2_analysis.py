"""Additive pooled OOF R-squared analysis for the no-SA probe."""

from __future__ import annotations

import csv
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from layer_metacognition.hidden_state_store import atomic_write_json, atomic_write_text, load_jsonl
from layer_metacognition.probe.common import iter_jsonl
from layer_metacognition.probe.hidden_state_loader import HiddenStateLoader
from layer_metacognition.probe.provenance import canonical_fingerprint, sha256_file
from layer_metacognition.probe_sa_no_prompt import DEFAULT_POSITIONS
from layer_metacognition.probe_sa_no_prompt.train_no_sa_probes import _feature_matrix, _outer_rows


R2_COHORT = "answer_matched"
R2_PREDICTION_TASK = "hard_midpoint"
R2_MODEL = {"type": "standard_scaler_plus_ridge", "alpha": 1.0, "solver": "lsqr"}


def load_hard_midpoint_map(config_path: str | Path) -> dict[str, float]:
    """Load the authoritative hard-class midpoints from the joint experiment config."""
    path = Path(config_path)
    config = json.loads(path.read_text(encoding="utf-8"))
    classes = config.get("source_attribution_classes")
    midpoints = config.get("source_attribution_midpoints")
    if not isinstance(classes, list) or not isinstance(midpoints, list):
        raise ValueError(
            f"Joint config must define source_attribution_classes and "
            f"source_attribution_midpoints: {path}"
        )
    if not classes or len(classes) != len(midpoints):
        raise ValueError(f"Joint config hard classes/midpoints are empty or length-mismatched: {path}")
    mapping: dict[str, float] = {}
    for raw_label, raw_midpoint in zip(classes, midpoints, strict=True):
        label = str(raw_label)
        if not label or label in mapping:
            raise ValueError(f"Joint config has an empty or duplicate hard class: {label!r}")
        if not isinstance(raw_midpoint, (int, float)) or not math.isfinite(float(raw_midpoint)):
            raise ValueError(f"Joint config has a non-finite midpoint for hard class {label!r}")
        mapping[label] = float(raw_midpoint)
    return mapping


def pooled_r2(true: Sequence[float], predicted: Sequence[float]) -> float | None:
    """Compute pooled R² without clipping; return None when it is undefined."""
    y_true = np.asarray(true, dtype=np.float64)
    y_pred = np.asarray(predicted, dtype=np.float64)
    if y_true.shape != y_pred.shape or y_true.ndim != 1:
        raise ValueError("R² inputs must be same-length one-dimensional arrays")
    if len(y_true) < 2 or not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        return None
    denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    if denominator <= 0.0:
        return None
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denominator)


def clustered_bootstrap_r2(
    rows: Sequence[dict[str, Any]],
    *,
    true_field: str,
    predicted_field: str,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap item clusters and compute a pooled R² interval."""
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row["item_id"])].append(row)
    items = sorted(buckets)
    if not items:
        return {"lower": None, "upper": None, "valid_repeats": 0, "item_count": 0, "repeats": int(repeats), "sampling_unit": "item_id"}
    counts_per_item: list[int] = []
    sums: list[float] = []
    squared_sums: list[float] = []
    squared_errors: list[float] = []
    for item in items:
        true = np.asarray([float(row[true_field]) for row in buckets[item]], dtype=np.float64)
        predicted = np.asarray([float(row[predicted_field]) for row in buckets[item]], dtype=np.float64)
        if not np.isfinite(true).all() or not np.isfinite(predicted).all():
            raise ValueError(f"Non-finite R² prediction values for item {item}")
        counts_per_item.append(len(true))
        sums.append(float(true.sum()))
        squared_sums.append(float(np.square(true).sum()))
        squared_errors.append(float(np.square(true - predicted).sum()))
    rng = np.random.default_rng(int(seed))
    sampled_counts = rng.multinomial(
        len(items),
        np.full(len(items), 1.0 / len(items), dtype=np.float64),
        size=int(repeats),
    ).astype(np.float64)
    sample_n = sampled_counts @ np.asarray(counts_per_item, dtype=np.float64)
    sample_sum = sampled_counts @ np.asarray(sums, dtype=np.float64)
    sample_squared_sum = sampled_counts @ np.asarray(squared_sums, dtype=np.float64)
    sample_sse = sampled_counts @ np.asarray(squared_errors, dtype=np.float64)
    sample_sst = sample_squared_sum - np.square(sample_sum) / sample_n
    valid = sample_sst > 0.0
    values = 1.0 - sample_sse[valid] / sample_sst[valid]
    if not len(values):
        lower = upper = None
    else:
        lower, upper = (float(value) for value in np.percentile(values, [2.5, 97.5]))
    return {
        "lower": lower,
        "upper": upper,
        "valid_repeats": int(len(values)),
        "item_count": len(items),
        "repeats": int(repeats),
        "sampling_unit": "item_id",
    }


def _prediction_key(row: dict[str, Any]) -> tuple[str, str, int, int, str]:
    return (
        str(row["cohort"]),
        str(row["position"]),
        int(row["layer"]),
        int(row["fold"]),
        str(row["no_sa_case_id"]),
    )


def _fit_midpoint_ridge(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_rows: Sequence[dict[str, Any]],
    test_rows: Sequence[dict[str, Any]],
    midpoint_map: dict[str, float],
) -> list[dict[str, Any]]:
    train_y = np.asarray([midpoint_map[str(row["hard_label"])] for row in train_rows], dtype=np.float64)
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=float(R2_MODEL["alpha"]), solver=str(R2_MODEL["solver"]))),
        ]
    )
    model.fit(train_x, train_y)
    predicted = np.asarray(model.predict(test_x), dtype=np.float64)
    return [
        {
            "cohort": R2_COHORT,
            "task": R2_PREDICTION_TASK,
            "no_sa_case_id": row["no_sa_case_id"],
            "joint_case_id": row["joint_case_id"],
            "item_id": row["item_id"],
            "prior_index": int(row["prior_index"]),
            "condition": row["condition"],
            "true_label": str(row["hard_label"]),
            "true_midpoint": float(midpoint_map[str(row["hard_label"])]),
            "predicted_midpoint": float(value),
        }
        for row, value in zip(test_rows, predicted, strict=True)
    ]


def _write_jsonl_append(path: Path, rows: Iterable[dict[str, Any]], existing: set[tuple[str, str, int, int, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            key = _prediction_key(row)
            if key in existing:
                continue
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            existing.add(key)
        handle.flush()


def _ensure_hard_midpoint_predictions(
    output: Path,
    config: dict[str, Any],
    midpoint_map: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    join_path = output / "join_records.jsonl"
    split_path = output / "split_assignments.json"
    if not join_path.is_file() or not split_path.is_file():
        raise FileNotFoundError("R² analysis requires join_records.jsonl and split_assignments.json")
    records = [row for row in iter_jsonl(join_path) if bool(row.get("answer_match"))]
    if not records:
        raise ValueError("R² analysis found no answer-matched records")
    missing_labels = sorted({str(row["hard_label"]) for row in records} - set(midpoint_map))
    if missing_labels:
        raise ValueError(f"Joint config has no midpoint for observed hard labels: {missing_labels}")
    assignment = json.loads(split_path.read_text(encoding="utf-8"))
    if assignment.get("group_key") != "item_id" or not isinstance(assignment.get("item_to_fold"), dict):
        raise ValueError("R² analysis requires the existing item_id split assignment")
    n_splits = int(config["n_splits"])
    if int(assignment.get("n_splits", -1)) != n_splits:
        raise ValueError("R² analysis split count differs from the original probe")
    item_to_fold = {str(key): int(value) for key, value in assignment["item_to_fold"].items()}
    if any(item_to_fold.get(str(row["item_id"])) is None for row in records):
        raise ValueError("R² analysis has answer-matched items without an existing fold assignment")

    source_config_path = Path(config["joint_config_path"])
    immutable = {
        "format_version": 1,
        "cohort": R2_COHORT,
        "conditions": list(config["conditions"]),
        "layers": [int(value) for value in config["layers"]],
        "positions": [str(value) for value in config["positions"]],
        "n_splits": n_splits,
        "seed": int(config["seed"]),
        "split_assignments_sha256": sha256_file(split_path),
        "join_records_sha256": sha256_file(join_path),
        "joint_config_path": str(source_config_path),
        "joint_config_sha256": sha256_file(source_config_path),
        "hard_label_midpoints": midpoint_map,
        "model": R2_MODEL,
    }
    immutable["config_fingerprint"] = canonical_fingerprint(immutable)
    r2_config_path = output / "r2_run_config.json"
    if r2_config_path.is_file():
        saved = json.loads(r2_config_path.read_text(encoding="utf-8"))
        if saved.get("config_fingerprint") != immutable["config_fingerprint"]:
            raise ValueError("Cannot resume R² analysis because its immutable configuration differs")
    else:
        atomic_write_json(r2_config_path, {**immutable, "status": "running"})

    prediction_path = output / "predictions" / "hard_midpoint_oof_predictions.jsonl"
    old_rows = load_jsonl(prediction_path, repair_trailing=True) if prediction_path.is_file() else []
    existing: set[tuple[str, str, int, int, str]] = set()
    for row in old_rows:
        key = _prediction_key(row)
        if key in existing:
            raise ValueError(f"Duplicate hard-midpoint OOF prediction key: {key}")
        existing.add(key)
    progress_path = output / "r2_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {}
    completed = set(str(value) for value in progress.get("completed_jobs", []))
    expected_jobs = len(config["positions"]) * len(config["layers"]) * n_splits
    loader = HiddenStateLoader(Path(config["no_sa_experiment_dir"]), cache_size=2)

    def checkpoint(status: str) -> None:
        atomic_write_json(
            progress_path,
            {
                "status": status,
                "total_job_count": expected_jobs,
                "completed_job_count": len(completed),
                "prediction_count": len(existing),
                "completed_jobs": sorted(completed),
            },
        )

    checkpoint("running")
    for position in config["positions"]:
        for layer in config["layers"]:
            usable, matrix, failures = _feature_matrix(loader, records, layer=int(layer), position=str(position))
            if failures or len(usable) != len(records):
                raise ValueError(
                    f"Hard-midpoint R² feature loading failed for {position}/L{layer}: "
                    f"{len(failures)} failures"
                )
            ordinal = {row["no_sa_case_id"]: index for index, row in enumerate(usable)}
            for fold in range(n_splits):
                job = f"{R2_COHORT}|{R2_PREDICTION_TASK}|{position}|{int(layer)}|{fold}"
                train_rows, test_rows = _outer_rows(usable, item_to_fold, fold=fold)
                expected_cases = {str(row["no_sa_case_id"]) for row in test_rows}
                present_cases = {
                    key[-1]
                    for key in existing
                    if key[:4] == (R2_COHORT, str(position), int(layer), fold)
                }
                if present_cases - expected_cases:
                    raise ValueError(f"Hard-midpoint OOF predictions contain unexpected cases for {job}")
                if present_cases == expected_cases and expected_cases:
                    completed.add(job)
                    continue
                train_x = matrix[[ordinal[row["no_sa_case_id"]] for row in train_rows]]
                test_x = matrix[[ordinal[row["no_sa_case_id"]] for row in test_rows]]
                rows = _fit_midpoint_ridge(train_x, test_x, train_rows, test_rows, midpoint_map)
                for row in rows:
                    row.update({"position": str(position), "layer": int(layer), "fold": fold})
                    if item_to_fold[str(row["item_id"])] != fold:
                        raise AssertionError("Hard-midpoint prediction is not OOF for its item")
                _write_jsonl_append(prediction_path, rows, existing)
                completed.add(job)
                checkpoint("running")
    checkpoint("training_complete")
    atomic_write_json(r2_config_path, {**immutable, "status": "training_complete", "prediction_count": len(existing)})
    return load_jsonl(prediction_path, repair_trailing=False), immutable


def _summarize_r2(
    rows: Sequence[dict[str, Any]],
    *,
    config: dict[str, Any],
    true_field: str,
    predicted_field: str,
    seed_offset: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    item_to_fold = {
        str(key): int(value)
        for key, value in json.loads((Path(config["output_dir"]) / "split_assignments.json").read_text(encoding="utf-8"))["item_to_fold"].items()
    }
    for row in rows:
        if str(row.get("cohort")) != R2_COHORT:
            continue
        if item_to_fold[str(row["item_id"])] != int(row["fold"]):
            raise ValueError("R² metric input contains a non-OOF prediction")
        grouped[(str(row["position"]), int(row["layer"]))].append(row)
    results: list[dict[str, Any]] = []
    for position in config["positions"]:
        for layer in config["layers"]:
            selected = grouped.get((str(position), int(layer)), [])
            if not selected:
                raise ValueError(f"Missing OOF R² predictions for {position}/L{layer}")
            r2 = pooled_r2(
                [float(row[true_field]) for row in selected],
                [float(row[predicted_field]) for row in selected],
            )
            bootstrap = clustered_bootstrap_r2(
                selected,
                true_field=true_field,
                predicted_field=predicted_field,
                repeats=int(config["bootstrap_repeats"]),
                seed=int(config["seed"]) + seed_offset + int(layer) * 101 + list(config["positions"]).index(position),
            )
            results.append(
                {
                    "position": str(position),
                    "layer": int(layer),
                    "n": len(selected),
                    "r2": r2,
                    "ci_low": bootstrap["lower"],
                    "ci_high": bootstrap["upper"],
                    "bootstrap_valid_repeats": bootstrap["valid_repeats"],
                    "item_count": bootstrap["item_count"],
                }
            )
    return results


def _write_metric(output: Path, name: str, target: str, rows: Sequence[dict[str, Any]]) -> tuple[Path, Path]:
    result_dir = output / "results" / R2_COHORT
    json_path = result_dir / f"{name}.json"
    csv_path = result_dir / f"{name}.csv"
    atomic_write_json(
        json_path,
        {
            "format_version": 1,
            "cohort": R2_COHORT,
            "target": target,
            "metric": "pooled_oof_r2",
            "prediction_scope": "outer_test_folds_only",
            "combinations": list(rows),
        },
    )
    fields = ["position", "layer", "n", "r2", "ci_low", "ci_high", "bootstrap_valid_repeats", "item_count"]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(csv_path, buffer.getvalue())
    return json_path, csv_path


def _plot_r2(rows: Sequence[dict[str, Any]], output: Path, *, title: str) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for position in DEFAULT_POSITIONS:
        selected = sorted(
            (int(row["layer"]), row)
            for row in rows
            if row["position"] == position and row["r2"] is not None
        )
        if not selected:
            continue
        layers = [layer for layer, _row in selected]
        values = [float(row["r2"]) for _layer, row in selected]
        lower = [row["ci_low"] for _layer, row in selected]
        upper = [row["ci_high"] for _layer, row in selected]
        axis.plot(layers, values, marker="o", label=position.upper())
        if all(value is not None for value in lower + upper):
            axis.fill_between(layers, lower, upper, alpha=0.10)
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Decoder layer (zero-based)")
    axis.set_ylabel("Pooled OOF R²")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _peaks(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    peaks: dict[str, dict[str, Any]] = {}
    for position in DEFAULT_POSITIONS:
        selected = [row for row in rows if row["position"] == position and row["r2"] is not None]
        if selected:
            best = max(selected, key=lambda row: float(row["r2"]))
            peaks[position] = {"layer": int(best["layer"]), "r2": float(best["r2"])}
    return peaks


def run_r2_analysis(output_dir: str | Path) -> dict[str, Any]:
    """Train only the missing midpoint probe and write additive R² artifacts."""
    output = Path(output_dir).resolve()
    config_path = output / "run_config.json"
    prediction_path = output / "predictions" / "oof_predictions.jsonl"
    if not config_path.is_file() or not prediction_path.is_file():
        raise FileNotFoundError("R² analysis requires the completed no-SA run config and OOF predictions")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if R2_COHORT not in config.get("cohorts", []):
        raise ValueError("R² analysis requires the existing answer_matched cohort")
    if set(config.get("conditions", [])) != {"conflict_easy", "conflict_hard"}:
        raise ValueError("R² analysis requires the conflict-only no-SA run")
    joint_config_path = Path(config.get("joint_config_path") or Path(config["joint_experiment_dir"]) / "config.json")
    midpoint_map = load_hard_midpoint_map(joint_config_path)
    hard_rows, r2_config = _ensure_hard_midpoint_predictions(output, config, midpoint_map)
    soft_rows = [
        row
        for row in iter_jsonl(prediction_path)
        if row.get("cohort") == R2_COHORT and row.get("task") == "soft_score"
    ]
    hard_results = _summarize_r2(
        hard_rows,
        config=config,
        true_field="true_midpoint",
        predicted_field="predicted_midpoint",
        seed_offset=10_000,
    )
    soft_results = _summarize_r2(
        soft_rows,
        config=config,
        true_field="true_score",
        predicted_field="predicted_score",
        seed_offset=20_000,
    )
    hard_json, hard_csv = _write_metric(
        output, "hard_midpoint_r2", "hard_label_interval_midpoint", hard_results
    )
    soft_json, soft_csv = _write_metric(output, "soft_score_r2", "soft_image_score", soft_results)
    hard_plot = output / "plots" / "answer_matched_hard_midpoint_r2.png"
    soft_plot = output / "plots" / "answer_matched_soft_score_r2.png"
    _plot_r2(hard_results, hard_plot, title="No-SA prompt: hard-label midpoint OOF R²")
    _plot_r2(soft_results, soft_plot, title="No-SA prompt: soft-score OOF R²")
    summary = {
        "format_version": 1,
        "status": "complete",
        "cohort": R2_COHORT,
        "prediction_scope": "outer_test_folds_only",
        "bootstrap": {"sampling_unit": "item_id", "repeats": int(config["bootstrap_repeats"]), "confidence_interval": 0.95},
        "hard_label_midpoints": midpoint_map,
        "hard_midpoint_peaks": _peaks(hard_results),
        "soft_score_peaks": _peaks(soft_results),
        "outputs": {
            "hard_json": str(hard_json),
            "hard_csv": str(hard_csv),
            "soft_json": str(soft_json),
            "soft_csv": str(soft_csv),
            "hard_plot": str(hard_plot),
            "soft_plot": str(soft_plot),
        },
        "r2_config_fingerprint": r2_config["config_fingerprint"],
    }
    atomic_write_json(output / "r2_summary.json", summary)
    progress_path = output / "r2_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress.update({"status": "complete", "hard_combination_count": len(hard_results), "soft_combination_count": len(soft_results), "hard_prediction_count": len(hard_rows), "soft_prediction_count": len(soft_rows)})
    atomic_write_json(progress_path, progress)
    atomic_write_json(output / "r2_run_config.json", {**r2_config, "status": "complete", "summary_path": str(output / "r2_summary.json")})
    return summary


__all__ = [
    "clustered_bootstrap_r2",
    "load_hard_midpoint_map",
    "pooled_r2",
    "run_r2_analysis",
]
