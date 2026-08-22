"""Analyze no-SA OOF predictions with item-clustered bootstrap intervals."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from layer_metacognition.hidden_state_store import atomic_write_json
from layer_metacognition.probe.common import iter_jsonl
from layer_metacognition.probe_sa_prediction.analyze_sa_probe_results import hard_metrics, soft_metrics
from layer_metacognition.probe_sa_no_prompt import (
    DEFAULT_COHORTS,
    DEFAULT_POSITIONS,
    SA_CLASSES,
    prediction_key,
)
from layer_metacognition.probe_sa_no_prompt.r2_analysis import run_r2_analysis


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _corr(function: Callable[..., Any], left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return None
    result = function(left, right)
    statistic = getattr(result, "statistic", result[0])
    return _finite(statistic)


def _hard_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result = hard_metrics(rows)
    if rows:
        true = [str(row["true_label"]) for row in rows]
        predicted = [str(row["predicted_label"]) for row in rows]
        result["confusion_matrix"] = confusion_matrix(true, predicted, labels=list(SA_CLASSES)).tolist()
    else:
        result["confusion_matrix"] = [[0 for _ in SA_CLASSES] for _ in SA_CLASSES]
    return result


def _item_buckets(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row["item_id"])].append(row)
    return dict(buckets)


def _percentile(values: Sequence[float]) -> dict[str, Any]:
    finite = np.asarray([value for value in values if math.isfinite(float(value))], dtype=np.float64)
    if not len(finite):
        return {"lower": None, "upper": None, "valid_repeats": 0}
    return {"lower": float(np.percentile(finite, 2.5)), "upper": float(np.percentile(finite, 97.5)), "valid_repeats": int(len(finite))}


def clustered_bootstrap_hard(rows: Sequence[dict[str, Any]], *, repeats: int, seed: int) -> dict[str, Any]:
    buckets = _item_buckets(rows)
    items = sorted(buckets)
    if not items:
        return {"accuracy": _percentile([]), "majority_accuracy": _percentile([]), "accuracy_delta": _percentile([]), "balanced_accuracy": _percentile([]), "item_count": 0}
    rng = np.random.default_rng(int(seed))
    probe_values: list[float] = []
    baseline_values: list[float] = []
    delta_values: list[float] = []
    balanced_values: list[float] = []
    for _ in range(int(repeats)):
        sampled_items = rng.choice(items, size=len(items), replace=True)
        sample = [row for item in sampled_items for row in buckets[str(item)]]
        probe = np.asarray([str(row["predicted_label"]) == str(row["true_label"]) for row in sample], dtype=np.float64)
        baseline = np.asarray([bool(row.get("majority_correct")) for row in sample], dtype=np.float64)
        probe_values.append(float(probe.mean()))
        baseline_values.append(float(baseline.mean()))
        delta_values.append(float(probe.mean() - baseline.mean()))
        try:
            balanced_values.append(float(balanced_accuracy_score([row["true_label"] for row in sample], [row["predicted_label"] for row in sample])))
        except ValueError:
            pass
    return {"accuracy": _percentile(probe_values), "majority_accuracy": _percentile(baseline_values), "accuracy_delta": _percentile(delta_values), "balanced_accuracy": _percentile(balanced_values), "item_count": len(items), "repeats": int(repeats), "sampling_unit": "item_id"}


def clustered_bootstrap_soft(rows: Sequence[dict[str, Any]], *, repeats: int, seed: int) -> dict[str, Any]:
    buckets = _item_buckets(rows)
    items = sorted(buckets)
    if not items:
        return {"spearman": _percentile([]), "pearson": _percentile([]), "mae": _percentile([]), "item_count": 0}
    rng = np.random.default_rng(int(seed))
    values: dict[str, list[float]] = {"spearman": [], "pearson": [], "mae": []}
    for _ in range(int(repeats)):
        sampled_items = rng.choice(items, size=len(items), replace=True)
        sample = [row for item in sampled_items for row in buckets[str(item)]]
        true = np.asarray([float(row["true_score"]) for row in sample], dtype=np.float64)
        predicted = np.asarray([float(row["predicted_score"]) for row in sample], dtype=np.float64)
        correlation_s = _corr(spearmanr, true, predicted)
        correlation_p = _corr(pearsonr, true, predicted)
        if correlation_s is not None:
            values["spearman"].append(correlation_s)
        if correlation_p is not None:
            values["pearson"].append(correlation_p)
        values["mae"].append(float(np.mean(np.abs(true - predicted))))
    return {metric: _percentile(metric_values) for metric, metric_values in values.items()} | {"item_count": len(items), "repeats": int(repeats), "sampling_unit": "item_id"}


def _stable_seed(seed: int, *parts: object) -> int:
    digest = hashlib.sha256("|".join([str(seed), *(str(part) for part in parts)]).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def _detect_onset(values: dict[int, float | None], *, consecutive: int = 2) -> dict[str, Any]:
    ordered = sorted((int(layer), value) for layer, value in values.items())
    for index in range(0, len(ordered) - consecutive + 1):
        window = ordered[index : index + consecutive]
        if all(value is not None and float(value) > 0.0 for _layer, value in window):
            return {"layer": int(window[0][0]), "layers": [int(layer) for layer, _value in window], "criterion": "95% bootstrap CI lower bound > 0", "consecutive_layers_required": consecutive}
    return {"layer": None, "layers": [], "criterion": "95% bootstrap CI lower bound > 0", "consecutive_layers_required": consecutive}


def _plot_hard(rows: Sequence[dict[str, Any]], output: Path, cohort: str) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for position in DEFAULT_POSITIONS:
        selected = sorted((int(row["layer"]), row) for row in rows if row["position"] == position and row["pooled_oof"].get("accuracy") is not None)
        if not selected:
            continue
        layers = [layer for layer, _row in selected]
        values = [float(row["pooled_oof"]["accuracy"]) for _layer, row in selected]
        lower = [row["bootstrap"].get("accuracy", {}).get("lower") for _layer, row in selected]
        upper = [row["bootstrap"].get("accuracy", {}).get("upper") for _layer, row in selected]
        axis.plot(layers, values, marker="o", label=position.upper())
        if all(value is not None for value in lower + upper):
            axis.fill_between(layers, lower, upper, alpha=0.10)
    baseline_rows = [row for row in rows if row["pooled_oof"].get("accuracy") is not None and row["bootstrap"].get("majority_accuracy", {}).get("lower") is not None]
    if baseline_rows:
        axis.axhline(float(np.mean([row["pooled_oof"].get("majority_accuracy", 0.0) for row in baseline_rows])), color="black", linestyle="--", linewidth=1.2, label="outer-train majority baseline")
    axis.set_xlabel("Decoder layer (zero-based)")
    axis.set_ylabel("Pooled OOF accuracy")
    axis.set_title(f"No-SA prompt ({cohort}, conflict-only): hard accuracy")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_soft(rows: Sequence[dict[str, Any]], output: Path, cohort: str) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for position in DEFAULT_POSITIONS:
        selected = sorted((int(row["layer"]), row) for row in rows if row["position"] == position and row["pooled_oof"].get("spearman") is not None)
        if not selected:
            continue
        layers = [layer for layer, _row in selected]
        values = [float(row["pooled_oof"]["spearman"]) for _layer, row in selected]
        lower = [row["bootstrap"].get("spearman", {}).get("lower") for _layer, row in selected]
        upper = [row["bootstrap"].get("spearman", {}).get("upper") for _layer, row in selected]
        axis.plot(layers, values, marker="o", label=position.upper())
        if all(value is not None for value in lower + upper):
            axis.fill_between(layers, lower, upper, alpha=0.10)
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Decoder layer (zero-based)")
    axis.set_ylabel("Pooled OOF Spearman")
    axis.set_title(f"No-SA prompt ({cohort}, conflict-only): soft-score Spearman")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _joint_comparison(output: Path, config: dict[str, Any], cohort_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    joint_summary_path = Path(config["joint_experiment_dir"]) / "stage_sa_prediction_probe" / "summary.json"
    if not joint_summary_path.is_file():
        return {"status": "skipped", "reason": f"joint summary not found: {joint_summary_path}"}
    joint = json.loads(joint_summary_path.read_text(encoding="utf-8"))
    figure, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    shared = {"AC", "LAT", "PANL"}
    for metric_index, (metric, label) in enumerate((("accuracy", "accuracy"), ("spearman", "Spearman"))):
        axis = axes[metric_index]
        for source_name, rows in (("No-SA answer-matched", cohort_rows.get("answer_matched", [])), ("No-SA all-joined", cohort_rows.get("all_joined", []))):
            for position in ("ac", "lat", "panl"):
                selected = sorted((int(row["layer"]), row["pooled_oof"].get(metric)) for row in rows if row["position"] == position and row["pooled_oof"].get(metric) is not None)
                if selected:
                    axis.plot([value[0] for value in selected], [value[1] for value in selected], marker="o", label=f"{source_name} {position.upper()}")
        key = "hard_sa_prediction" if metric == "accuracy" else "soft_sa_prediction"
        for position in shared:
            selected = sorted((int(row["layer"]), row.get(metric)) for row in joint.get(key, []) if str(row.get("position", "")).upper() == position and row.get(metric) is not None)
            if selected:
                axis.plot([value[0] for value in selected], [value[1] for value in selected], linestyle=":", linewidth=1.4, label=f"Joint prompt {position}")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Decoder layer (zero-based)")
    axes[0].set_title("Joint prompt vs No-SA prompt (conflict-only; AC/LAT/PANL)")
    axes[0].legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(output / "plots" / "joint_vs_no_sa_common_positions.png", dpi=180)
    plt.close(figure)
    return {"status": "complete", "joint_summary_path": str(joint_summary_path), "positions": sorted(shared)}


def run_analysis(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    (output / "plots").mkdir(parents=True, exist_ok=True)
    config_path = output / "run_config.json"
    predictions_path = output / "predictions" / "oof_predictions.jsonl"
    if not config_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError("No-SA Probe requires run_config.json and predictions/oof_predictions.jsonl")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    predictions = list(iter_jsonl(predictions_path))
    seen: set[tuple[Any, ...]] = set()
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        key = prediction_key(row)
        if key in seen:
            raise ValueError(f"Duplicate OOF prediction key: {key}")
        seen.add(key)
        grouped[(str(row["cohort"]), str(row["task"]), str(row["position"]), int(row["layer"]))].append(row)
    cohort_results: dict[str, dict[str, list[dict[str, Any]]]] = {}
    onset: dict[str, Any] = {}
    table_rows: list[dict[str, Any]] = []
    for cohort in config["cohorts"]:
        hard_results: list[dict[str, Any]] = []
        soft_results: list[dict[str, Any]] = []
        for position in config["positions"]:
            for layer in config["layers"]:
                hard_rows = grouped.get((cohort, "hard_label", position, int(layer)), [])
                soft_rows = grouped.get((cohort, "soft_score", position, int(layer)), [])
                seed = _stable_seed(int(config["seed"]), cohort, position, layer)
                hard = _hard_metrics(hard_rows)
                if hard_rows:
                    hard["majority_accuracy"] = float(np.mean([bool(row["majority_correct"]) for row in hard_rows]))
                else:
                    hard["majority_accuracy"] = None
                hard_row = {"cohort": cohort, "position": position, "layer": int(layer), "pooled_oof": hard, "bootstrap": clustered_bootstrap_hard(hard_rows, repeats=int(config["bootstrap_repeats"]), seed=seed)}
                soft_row = {"cohort": cohort, "position": position, "layer": int(layer), "pooled_oof": soft_metrics(soft_rows), "bootstrap": clustered_bootstrap_soft(soft_rows, repeats=int(config["bootstrap_repeats"]), seed=seed + 1)}
                hard_results.append(hard_row)
                soft_results.append(soft_row)
                table_rows.append({"cohort": cohort, "position": position.upper(), "layer": int(layer), "hard_sample_count": hard.get("sample_count"), "hard_accuracy": hard.get("accuracy"), "hard_accuracy_delta": (None if hard.get("majority_accuracy") is None or hard.get("accuracy") is None else float(hard["accuracy"] - hard["majority_accuracy"])), "hard_balanced_accuracy": hard.get("balanced_accuracy"), "hard_macro_f1": hard.get("macro_f1"), "hard_auroc": hard.get("macro_ovr_auroc"), "soft_sample_count": soft_row["pooled_oof"].get("sample_count"), "soft_spearman": soft_row["pooled_oof"].get("spearman"), "soft_pearson": soft_row["pooled_oof"].get("pearson"), "soft_mae": soft_row["pooled_oof"].get("mae"), "soft_r2": soft_row["pooled_oof"].get("r2")})
        result_dir = output / "results" / cohort
        atomic_write_json(result_dir / "hard_label_results.json", {"format_version": 1, "target": "parsed_label", "primary_metric": "accuracy", "secondary_metrics": ["balanced_accuracy", "macro_f1", "macro_ovr_auroc"], "combinations": hard_results})
        atomic_write_json(result_dir / "soft_score_results.json", {"format_version": 1, "target": "soft_image_score", "primary_metric": "spearman", "prediction_clipping": False, "combinations": soft_results})
        _plot_hard(hard_results, output / "plots" / f"{cohort}_hard_accuracy.png", cohort)
        _plot_soft(soft_results, output / "plots" / f"{cohort}_soft_spearman.png", cohort)
        hard_onset = {}
        soft_onset = {}
        for position in config["positions"]:
            hard_onset[position] = _detect_onset({int(row["layer"]): row["bootstrap"].get("accuracy_delta", {}).get("lower") for row in hard_results if row["position"] == position})
            soft_onset[position] = _detect_onset({int(row["layer"]): row["bootstrap"].get("spearman", {}).get("lower") for row in soft_results if row["position"] == position})
        cohort_results[cohort] = {"hard": hard_results, "soft": soft_results}
        onset[cohort] = {"hard_accuracy_delta": hard_onset, "soft_spearman": soft_onset}
    table_dir = output / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    fields = list(table_rows[0]) if table_rows else ["cohort", "position", "layer"]
    with (table_dir / "layer_position_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table_rows)
    match_rows = []
    manifest = json.loads((output / "join_manifest.json").read_text(encoding="utf-8")) if (output / "join_manifest.json").is_file() else {}
    for condition, rate in manifest.get("answer_match_rate_by_condition", {}).items():
        match_rows.append({"dimension": "condition", "value": condition, "answer_match_rate": rate})
    for prior, rate in manifest.get("answer_match_rate_by_prior_index", {}).items():
        match_rows.append({"dimension": "prior_index", "value": prior, "answer_match_rate": rate})
    with (table_dir / "answer_match_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dimension", "value", "answer_match_rate"])
        writer.writeheader()
        writer.writerows(match_rows)
    comparison = _joint_comparison(output, config, {cohort: cohort_results[cohort]["hard"] for cohort in cohort_results})
    r2_analysis = run_r2_analysis(output)
    summary = {"format_version": 1, "status": "complete", "experiment": "conflict-only_no_sa_prompt", "interpretation": "No-SA probe positivity means hidden states contain linearly decodable information related to final SA; it does not identify a causal formation location.", "primary_metrics": {"hard": "pooled_oof_accuracy_delta_vs_outer_train_majority_baseline", "soft": "pooled_oof_spearman"}, "additional_metrics": {"hard_midpoint": "pooled_oof_r2", "soft_score": "pooled_oof_r2"}, "cohorts": {cohort: {"hard": cohort_results[cohort]["hard"], "soft": cohort_results[cohort]["soft"]} for cohort in cohort_results}, "onset": onset, "comparison": comparison, "r2_analysis": r2_analysis, "oof_prediction_count": len(predictions), "input_failures": manifest.get("input_failure_count", 0), "unmatched_count": manifest.get("unmatched_count", 0)}
    atomic_write_json(output / "onset.json", {"definition": "Earliest layer with a positive 95% bootstrap CI lower bound for two consecutive measured layers; descriptive decodability only, not causal formation.", "results": onset})
    atomic_write_json(output / "summary.json", summary)
    config["status"] = "complete"
    config["summary_path"] = str(output / "summary.json")
    atomic_write_json(config_path, config)
    progress_path = output / "progress.json"
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        progress.update({"status": "complete", "analysis_combination_count": sum(len(value["hard"]) + len(value["soft"]) for value in cohort_results.values())})
        atomic_write_json(progress_path, progress)
    return {"status": "complete", "cohort_count": len(cohort_results), "prediction_count": len(predictions), "output_dir": str(output), "comparison": comparison}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--r2-only",
        action="store_true",
        help="Create only the additive hard-midpoint and soft-score OOF R² artifacts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_r2_analysis(args.output_dir) if args.r2_only else run_analysis(args.output_dir)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
