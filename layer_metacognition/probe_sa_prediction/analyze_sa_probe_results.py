#!/usr/bin/env python3
"""Aggregate SA OOF predictions, write tables, and plot layer trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)

from layer_metacognition.hidden_state_store import atomic_write_json
from layer_metacognition.probe.common import iter_jsonl

from . import DEFAULT_OUTPUT_DIR, SA_CLASSES, prediction_key


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _correlation(function: Callable[..., Any], left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return None
    result = function(left, right)
    value = getattr(result, "statistic", result[0])
    return _finite(value)


def hard_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "invalid", "reason": "no_predictions"}
    true = [str(row["true_label"]) for row in rows]
    predicted = [str(row["predicted_label"]) for row in rows]
    probabilities = np.asarray(
        [
            [float(row["class_probabilities"].get(label, 0.0)) for label in SA_CLASSES]
            for row in rows
        ],
        dtype=np.float64,
    )
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Hard OOF probabilities do not sum to one")
    support = Counter(true)
    observed_labels = [label for label in SA_CLASSES if support[label] > 0]
    per_class_auc: dict[str, float | None] = {}
    valid_auc: list[float] = []
    for class_index, label in enumerate(SA_CLASSES):
        binary = np.asarray([value == label for value in true], dtype=np.int64)
        if len(set(binary.tolist())) < 2:
            per_class_auc[label] = None
            continue
        value = float(roc_auc_score(binary, probabilities[:, class_index]))
        per_class_auc[label] = value
        valid_auc.append(value)
    return {
        "status": "valid",
        "sample_count": len(rows),
        "item_count": len({str(row["item_id"]) for row in rows}),
        "accuracy": float(accuracy_score(true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "macro_f1": float(
            f1_score(true, predicted, labels=observed_labels, average="macro", zero_division=0)
        ),
        "macro_ovr_auroc": float(np.mean(valid_auc)) if valid_auc else None,
        "per_class_auroc": per_class_auc,
        "class_support": {label: int(support[label]) for label in SA_CLASSES},
        "auroc_evaluable_class_count": len(valid_auc),
        "configured_class_count": len(SA_CLASSES),
        "observed_class_count": len(observed_labels),
    }


def soft_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "invalid", "reason": "no_predictions"}
    true = np.asarray([float(row["true_score"]) for row in rows], dtype=np.float64)
    predicted = np.asarray([float(row["predicted_score"]) for row in rows], dtype=np.float64)
    if not bool(np.isfinite(true).all() and np.isfinite(predicted).all()):
        raise ValueError("Soft OOF predictions contain NaN or Inf")
    return {
        "status": "valid",
        "sample_count": len(rows),
        "item_count": len({str(row["item_id"]) for row in rows}),
        "pearson": _correlation(pearsonr, true, predicted),
        "spearman": _correlation(spearmanr, true, predicted),
        "mae": float(mean_absolute_error(true, predicted)),
        "r2": float(r2_score(true, predicted)) if len(rows) >= 2 else None,
        "prediction_below_zero_count": int(np.sum(predicted < 0.0)),
        "prediction_above_one_count": int(np.sum(predicted > 1.0)),
        "prediction_outside_unit_interval_fraction": float(
            np.mean((predicted < 0.0) | (predicted > 1.0))
        ),
        "prediction_clipping_applied": False,
    }


def _fold_summary(
    rows: Sequence[dict[str, Any]],
    metric_function: Callable[[Sequence[dict[str, Any]]], dict[str, Any]],
    metric_names: Sequence[str],
) -> dict[str, Any]:
    by_fold: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_fold[int(row["fold"])].append(row)
    fold_metrics = [
        {"fold": fold, **metric_function(by_fold[fold])}
        for fold in sorted(by_fold)
    ]
    aggregate: dict[str, Any] = {}
    for name in metric_names:
        values = [float(row[name]) for row in fold_metrics if row.get(name) is not None]
        aggregate[name] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values, ddof=0)) if values else None,
            "valid_fold_count": len(values),
        }
    return {"folds": fold_metrics, "unweighted_fold_aggregate": aggregate}


def _trajectory_test(values: dict[int, float | None]) -> dict[str, Any]:
    usable = sorted((layer, value) for layer, value in values.items() if value is not None)
    if not usable:
        return {"first_layer": None, "last_layer": None, "delta": None, "layer_metric_spearman": None}
    layers = np.asarray([row[0] for row in usable], dtype=np.float64)
    metrics = np.asarray([float(row[1]) for row in usable], dtype=np.float64)
    return {
        "first_layer": int(layers[0]),
        "last_layer": int(layers[-1]),
        "first_value": float(metrics[0]),
        "last_value": float(metrics[-1]),
        "delta": float(metrics[-1] - metrics[0]),
        "layer_metric_spearman": _correlation(spearmanr, layers, metrics),
    }


def _mean(values: Sequence[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return float(np.mean(valid)) if valid else None


def hypothesis_analysis(
    hard: Sequence[dict[str, Any]],
    soft: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    hard_map = {(row["position"], int(row["layer"])): row["pooled_oof"].get("balanced_accuracy") for row in hard}
    soft_map = {(row["position"], int(row["layer"])): row["pooled_oof"].get("spearman") for row in soft}
    early_layers = (10, 12, 14, 16)

    def early(metric_map: dict[tuple[str, int], float | None]) -> dict[str, Any]:
        post_answer = _mean([metric_map.get((position, layer)) for position in ("panl", "sac") for layer in early_layers])
        answer = _mean([metric_map.get((position, layer)) for position in ("ac", "lat") for layer in early_layers])
        return {
            "panl_sac_mean": post_answer,
            "ac_lat_mean": answer,
            "difference": None if post_answer is None or answer is None else float(post_answer - answer),
            "direction_consistent": None if post_answer is None or answer is None else bool(post_answer > answer),
        }

    h2: dict[str, Any] = {}
    for position in ("ac", "lat"):
        h2[position] = {
            "hard_balanced_accuracy": _trajectory_test(
                {layer: hard_map.get((position, layer)) for layer in sorted({key[1] for key in hard_map})}
            ),
            "soft_spearman": _trajectory_test(
                {layer: soft_map.get((position, layer)) for layer in sorted({key[1] for key in soft_map})}
            ),
        }

    def position_ranking(metric_map: dict[tuple[str, int], float | None]) -> dict[str, Any]:
        positions = sorted({key[0] for key in metric_map})
        means = {
            position: _mean([value for (name, _layer), value in metric_map.items() if name == position])
            for position in positions
        }
        valid_means = {key: value for key, value in means.items() if value is not None}
        best_mean = max(valid_means, key=valid_means.get) if valid_means else None
        valid_cells = {key: value for key, value in metric_map.items() if value is not None}
        best_cell = max(valid_cells, key=valid_cells.get) if valid_cells else None
        return {
            "position_means": means,
            "best_mean_position": best_mean,
            "sac_best_mean": None if best_mean is None else best_mean == "sac",
            "global_best": None if best_cell is None else {"position": best_cell[0], "layer": best_cell[1], "value": valid_cells[best_cell]},
        }

    return {
        "interpretation": "Directional OOF evidence only; linear decodability is not causal evidence.",
        "hypothesis_1_early_panl_sac_before_ac_lat": {
            "early_layers": list(early_layers),
            "hard_balanced_accuracy": early(hard_map),
            "soft_spearman": early(soft_map),
        },
        "hypothesis_2_ac_lat_increase_with_layer": h2,
        "hypothesis_3_sac_is_best": {
            "hard_balanced_accuracy": position_ranking(hard_map),
            "soft_spearman": position_ranking(soft_map),
        },
    }


def _plot_trajectory(rows: Sequence[dict[str, Any]], *, metric: str, title: str, ylabel: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    for position in ("ac", "lat", "panl", "sac"):
        selected = sorted(
            (int(row["layer"]), row["pooled_oof"].get(metric))
            for row in rows
            if row["position"] == position and row["pooled_oof"].get(metric) is not None
        )
        if selected:
            axis.plot([row[0] for row in selected], [row[1] for row in selected], marker="o", label=position.upper())
    axis.set_xlabel("Decoder layer (zero-based)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_analysis(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    config_path = output / "run_config.json"
    predictions_path = output / "predictions" / "oof_predictions.jsonl"
    progress_path = output / "progress.json"
    for path in (config_path, predictions_path, progress_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required SA probe artifact does not exist: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    predictions = list(iter_jsonl(predictions_path))
    seen: set[tuple[str, str, int, int, str]] = set()
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        key = prediction_key(row)
        if key in seen:
            raise ValueError(f"Duplicate OOF prediction key: {key}")
        seen.add(key)
        grouped[(str(row["task"]), str(row["position"]), int(row["layer"]))].append(row)

    hard_results: list[dict[str, Any]] = []
    soft_results: list[dict[str, Any]] = []
    for position in config["positions"]:
        for layer in config["layers"]:
            hard_rows = grouped.get(("hard_label", position, int(layer)), [])
            soft_rows = grouped.get(("soft_score", position, int(layer)), [])
            hard_results.append(
                {
                    "position": position,
                    "layer": int(layer),
                    "pooled_oof": hard_metrics(hard_rows),
                    **_fold_summary(
                        hard_rows,
                        hard_metrics,
                        ("accuracy", "balanced_accuracy", "macro_f1", "macro_ovr_auroc"),
                    ),
                }
            )
            soft_results.append(
                {
                    "position": position,
                    "layer": int(layer),
                    "pooled_oof": soft_metrics(soft_rows),
                    **_fold_summary(
                        soft_rows,
                        soft_metrics,
                        ("pearson", "spearman", "mae", "r2"),
                    ),
                }
            )
    results_dir = output / "results"
    atomic_write_json(
        results_dir / "hard_label_results.json",
        {
            "format_version": 1,
            "target": "parsed_label",
            "class_space": list(SA_CLASSES),
            "primary_metric": "balanced_accuracy",
            "combinations": hard_results,
        },
    )
    atomic_write_json(
        results_dir / "soft_score_results.json",
        {
            "format_version": 1,
            "target": "soft_image_score",
            "primary_metric": "spearman",
            "prediction_clipping": False,
            "combinations": soft_results,
        },
    )

    hard_by_key = {(row["position"], row["layer"]): row for row in hard_results}
    soft_by_key = {(row["position"], row["layer"]): row for row in soft_results}
    table_dir = output / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    csv_path = table_dir / "layer_position_summary.csv"
    fields = (
        "position", "layer", "hard_sample_count", "hard_accuracy",
        "hard_balanced_accuracy", "hard_auroc", "hard_macro_f1",
        "soft_sample_count", "soft_spearman", "soft_pearson", "soft_mae", "soft_r2",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for position in config["positions"]:
            for layer in config["layers"]:
                hard = hard_by_key[(position, int(layer))]["pooled_oof"]
                soft = soft_by_key[(position, int(layer))]["pooled_oof"]
                writer.writerow(
                    {
                        "position": position.upper(),
                        "layer": int(layer),
                        "hard_sample_count": hard.get("sample_count"),
                        "hard_accuracy": hard.get("accuracy"),
                        "hard_balanced_accuracy": hard.get("balanced_accuracy"),
                        "hard_auroc": hard.get("macro_ovr_auroc"),
                        "hard_macro_f1": hard.get("macro_f1"),
                        "soft_sample_count": soft.get("sample_count"),
                        "soft_spearman": soft.get("spearman"),
                        "soft_pearson": soft.get("pearson"),
                        "soft_mae": soft.get("mae"),
                        "soft_r2": soft.get("r2"),
                    }
                )

    _plot_trajectory(
        hard_results,
        metric="accuracy",
        title="Final SA hard-label prediction across layers",
        ylabel="OOF accuracy",
        path=output / "plots" / "hard_label_layer_trajectory.png",
    )
    _plot_trajectory(
        soft_results,
        metric="spearman",
        title="Final SA soft-score prediction across layers",
        ylabel="OOF Spearman correlation",
        path=output / "plots" / "soft_score_layer_trajectory.png",
    )
    valid_hard = [row for row in hard_results if row["pooled_oof"].get("balanced_accuracy") is not None]
    valid_soft = [row for row in soft_results if row["pooled_oof"].get("spearman") is not None]
    best_hard = max(valid_hard, key=lambda row: row["pooled_oof"]["balanced_accuracy"]) if valid_hard else None
    best_soft = max(valid_soft, key=lambda row: row["pooled_oof"]["spearman"]) if valid_soft else None
    summary = {
        "format_version": 1,
        "source_experiment_dir": config["experiment_dir"],
        "hard_sa_prediction": [
            {
                "position": row["position"].upper(),
                "layer": row["layer"],
                "balanced_accuracy": row["pooled_oof"].get("balanced_accuracy"),
                "auroc": row["pooled_oof"].get("macro_ovr_auroc"),
                "accuracy": row["pooled_oof"].get("accuracy"),
                "macro_f1": row["pooled_oof"].get("macro_f1"),
            }
            for row in hard_results
        ],
        "soft_sa_prediction": [
            {
                "position": row["position"].upper(),
                "layer": row["layer"],
                "spearman": row["pooled_oof"].get("spearman"),
                "pearson": row["pooled_oof"].get("pearson"),
                "mae": row["pooled_oof"].get("mae"),
                "r2": row["pooled_oof"].get("r2"),
            }
            for row in soft_results
        ],
        "best_hard": None if best_hard is None else {
            "position": best_hard["position"].upper(),
            "layer": best_hard["layer"],
            **best_hard["pooled_oof"],
        },
        "best_soft": None if best_soft is None else {
            "position": best_soft["position"].upper(),
            "layer": best_soft["layer"],
            **best_soft["pooled_oof"],
        },
        "hypothesis_analysis": hypothesis_analysis(hard_results, soft_results),
        "oof_prediction_count": len(predictions),
        "aggregation": {
            "primary": "pooled_oof",
            "fold_summary": "unweighted_mean_and_population_std",
        },
    }
    atomic_write_json(output / "summary.json", summary)
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["status"] = "complete"
    progress["analysis_combination_count"] = len(hard_results) + len(soft_results)
    atomic_write_json(progress_path, progress)
    config["status"] = "complete"
    config["summary_path"] = str(output / "summary.json")
    atomic_write_json(config_path, config)
    return {
        "status": "complete",
        "hard_combination_count": len(hard_results),
        "soft_combination_count": len(soft_results),
        "best_hard": summary["best_hard"],
        "best_soft": summary["best_soft"],
        "output_dir": str(output),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_analysis(args.output_dir)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
