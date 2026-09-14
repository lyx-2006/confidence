from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from experiment_config import BOOTSTRAP_REPEATS, RESULTS_ROOT, SEED
from io_utils import atomic_json, atomic_jsonl, load_jsonl


def bh_fdr(pvalues: Sequence[float]) -> list[float]:
    values = np.asarray(pvalues, dtype=float)
    if not len(values):
        return []
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Invalid p-values")
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(values) / np.arange(1, len(values) + 1))[::-1]
    )[::-1]
    output = np.empty(len(values))
    output[order] = np.minimum(adjusted, 1.0)
    return output.tolist()


def _one_sided_p(values: np.ndarray, positive: bool) -> float:
    bad = np.count_nonzero(values <= 0 if positive else values >= 0)
    return float((bad + 1) / (len(values) + 1))


def _bootstrap_cell(rows: Sequence[dict[str, Any]], repeats: int, seed: int) -> dict[str, Any]:
    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_item[str(row["item_id"])].append(row)
    items = sorted(by_item)
    if not items:
        raise ValueError("Cannot bootstrap an empty steering cell")
    alpha_values = sorted({float(row["alpha"]) for row in rows})
    alpha_min, alpha_max = alpha_values[0], alpha_values[-1]
    if alpha_min >= 0 or alpha_max <= 0:
        raise ValueError("Dose response requires negative and positive alpha values")

    def statistics(sampled_items: Sequence[str]) -> tuple[float, float, float]:
        sampled = [row for item in sampled_items for row in by_item[item]]
        x = np.asarray([float(row["alpha"]) for row in sampled])
        y = np.asarray([float(row["delta_soft_sa"]) for row in sampled])
        slope = float(np.polyfit(x, y, 1)[0])
        plus = float(np.mean([float(row["delta_soft_sa"]) for row in sampled if float(row["alpha"]) == alpha_max]))
        minus = float(np.mean([float(row["delta_soft_sa"]) for row in sampled if float(row["alpha"]) == alpha_min]))
        return slope, plus, minus

    observed = statistics(items)
    rng = np.random.default_rng(seed)
    boot = np.asarray(
        [statistics(rng.choice(items, size=len(items), replace=True).tolist()) for _ in range(repeats)]
    )
    component_pvalues = [
        _one_sided_p(boot[:, 0], True),
        _one_sided_p(boot[:, 1], True),
        _one_sided_p(boot[:, 2], False),
    ]
    return {
        "slope": observed[0],
        "positive_extreme_alpha": alpha_max,
        "positive_extreme_mean_delta": observed[1],
        "negative_extreme_alpha": alpha_min,
        "negative_extreme_mean_delta": observed[2],
        "slope_ci": np.percentile(boot[:, 0], [2.5, 97.5]).tolist(),
        "positive_extreme_ci": np.percentile(boot[:, 1], [2.5, 97.5]).tolist(),
        "negative_extreme_ci": np.percentile(boot[:, 2], [2.5, 97.5]).tolist(),
        "component_pvalues": component_pvalues,
        "intersection_union_p": max(component_pvalues),
        "bootstrap_repeats": repeats,
        "item_count": len(items),
    }


def directional_gate(
    rows: Sequence[dict[str, Any]], repeats: int = BOOTSTRAP_REPEATS, seed: int = SEED
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "completed" and row.get("direction_type") == "true":
            groups[(str(row["position"]), int(row["layer"]))].append(row)
    metrics = []
    for index, (position, layer) in enumerate(sorted(groups)):
        metrics.append(
            {"position": position, "layer": layer, **_bootstrap_cell(groups[(position, layer)], repeats, seed + index)}
        )
    q_values = bh_fdr([row["intersection_union_p"] for row in metrics])
    selected = []
    for row, q_value in zip(metrics, q_values):
        row["q_value"] = q_value
        row["point_direction_valid"] = (
            row["slope"] > 0
            and row["positive_extreme_mean_delta"] > 0
            and row["negative_extreme_mean_delta"] < 0
        )
        row["selected"] = bool(row["point_direction_valid"] and q_value < 0.05)
        if row["selected"]:
            selected.append(dict(row))
    return selected, metrics


def _mean_ci(rows: Sequence[dict[str, Any]], repeats: int, seed: int) -> tuple[float, list[float]]:
    by_item: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_item[str(row["item_id"])].append(float(row["delta_soft_sa"]))
    items = sorted(by_item)
    values = [value for group in by_item.values() for value in group]
    rng = np.random.default_rng(seed)
    boot = [
        float(np.mean([value for item in rng.choice(items, size=len(items), replace=True) for value in by_item[str(item)]]))
        for _ in range(repeats)
    ]
    return float(np.mean(values)), np.percentile(boot, [2.5, 97.5]).tolist()


def build_metrics(rows: Sequence[dict[str, Any]], repeats: int, seed: int) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "completed":
            continue
        for group in ("all", str(row.get("test_side", "unknown"))):
            groups[(row["position"], int(row["layer"]), row["direction_type"], float(row["alpha"]), group)].append(row)
    output = []
    for index, (key, values) in enumerate(sorted(groups.items(), key=lambda pair: str(pair[0]))):
        mean, interval = _mean_ci(values, repeats, seed + index)
        output.append(
            {
                "position": key[0],
                "layer": key[1],
                "direction_type": key[2],
                "alpha": key[3],
                "group": key[4],
                "mean_delta_soft_sa": mean,
                "ci_low": interval[0],
                "ci_high": interval[1],
                "hard_class_change_rate": sum(bool(row["hard_class_changed"]) for row in values) / len(values),
                "hard_class_mean_delta": float(np.mean([row["hard_class_delta"] for row in values])),
                "invalid_probability_count": sum(
                    not math.isfinite(float(row["probability_sum"]))
                    or abs(float(row["probability_sum"]) - 1) > 1e-6
                    for row in values
                ),
                "sample_count": len(values),
                "item_count": len({str(row["item_id"]) for row in values}),
            }
        )
    return output


def build_dose_metrics(rows: Sequence[dict[str, Any]], repeats: int, seed: int) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "completed":
            continue
        for group in ("all", str(row.get("test_side", "unknown"))):
            groups[(row["position"], int(row["layer"]), row["direction_type"], group)].append(row)
    return [
        {
            "position": key[0],
            "layer": key[1],
            "direction_type": key[2],
            "group": key[3],
            **_bootstrap_cell(values, repeats, seed + 5000 + index),
        }
        for index, (key, values) in enumerate(sorted(groups.items(), key=lambda pair: str(pair[0])))
    ]


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plots(root: Path, metrics: Sequence[dict[str, Any]], config: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    true_all = [row for row in metrics if row["direction_type"] == "true" and row["group"] == "all"]
    positions = list(config["positions"])
    layers = list(map(int, config["layers"]))
    figure, axes = plt.subplots(2, 3, figsize=(16, 8), squeeze=False)
    for axis, position in zip(axes.flat, positions):
        for alpha in sorted({row["alpha"] for row in true_all}):
            data = sorted(
                [row for row in true_all if row["position"] == position and row["alpha"] == alpha],
                key=lambda row: row["layer"],
            )
            axis.plot([row["layer"] for row in data], [row["mean_delta_soft_sa"] for row in data], marker="o", label=f"a={alpha:g}")
        axis.axhline(0, color="black", linestyle="--", linewidth=0.8)
        axis.set_title(position)
        axis.set_xlabel("zero-based Gemma layer")
    axes[0, 0].set_ylabel("mean delta soft SA")
    axes[1, 0].set_ylabel("mean delta soft SA")
    axes[0, 2].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(root / "steering_delta_soft_by_layer.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    for layer in layers:
        data = sorted(
            [row for row in true_all if row["position"] == "P1_PANL" and row["layer"] == layer],
            key=lambda row: row["alpha"],
        )
        axis.plot([row["alpha"] for row in data], [row["mean_delta_soft_sa"] for row in data], marker="o", label=f"L{layer}")
    axis.axhline(0, color="black", linestyle="--")
    axis.set_xlabel("alpha")
    axis.set_ylabel("mean delta soft SA")
    axis.legend(ncol=max(1, min(4, len(layers))))
    figure.tight_layout()
    figure.savefig(root / "steering_dose_response_panl.png", dpi=160)
    plt.close(figure)

    extrema = [row for row in true_all if abs(row["alpha"]) == max(abs(float(value)) for value in config["alphas"])]
    best = [max([row for row in extrema if row["position"] == position], key=lambda row: abs(row["mean_delta_soft_sa"])) for position in positions]
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.bar([row["position"] for row in best], [row["mean_delta_soft_sa"] for row in best])
    axis.axhline(0, color="black", linestyle="--")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(root / "steering_position_comparison.png", dpi=160)
    plt.close(figure)

    shuffled = [row for row in metrics if row["position"] == "P1_PANL" and row["group"] == "all"]
    figure, axis = plt.subplots(figsize=(8, 5))
    plotted = False
    for layer in config.get("shuffled_layers", []):
        for direction in ("true", "shuffled"):
            data = sorted(
                [row for row in shuffled if row["layer"] == layer and row["direction_type"] == direction],
                key=lambda row: row["alpha"],
            )
            if data:
                plotted = True
                axis.plot([row["alpha"] for row in data], [row["mean_delta_soft_sa"] for row in data], marker="o", label=f"L{layer} {direction}")
    axis.axhline(0, color="black", linestyle="--")
    if plotted:
        axis.legend()
    else:
        axis.text(0.5, 0.5, "No shuffled-control layers configured", ha="center", va="center", transform=axis.transAxes)
    figure.tight_layout()
    figure.savefig(root / "steering_shuffled_control.png", dpi=160)
    plt.close(figure)


def _ood_summary(root: Path, rows: Sequence[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    manifests = load_jsonl(root / "steering" / "test_manifest.jsonl")
    cells = []
    for position in config["positions"]:
        for layer in config["layers"]:
            vectors = []
            for row in manifests:
                with np.load(root / row["hidden_file"]) as payload:
                    vectors.append(np.asarray(payload[f"{position}__L{layer}"], dtype=np.float64))
            natural_cosine, natural_ratio = [], []
            for left, right in zip(vectors[::2], vectors[1::2]):
                natural_cosine.append(float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))))
                natural_ratio.append(float(np.linalg.norm(right) / np.linalg.norm(left)))
            steered = [
                row for row in rows
                if row.get("status") == "completed" and row["position"] == position and int(row["layer"]) == layer
            ]
            cells.append(
                {
                    "position": position,
                    "layer": layer,
                    "natural_pair_count": len(natural_cosine),
                    "natural_cosine_percentiles": np.percentile(natural_cosine, [5, 50, 95]).tolist(),
                    "natural_norm_ratio_percentiles": np.percentile(natural_ratio, [5, 50, 95]).tolist(),
                    "steered_cosine_percentiles": np.percentile([row["activation_cosine"] for row in steered], [5, 50, 95]).tolist(),
                    "steered_norm_ratio_percentiles": np.percentile([row["activation_norm_ratio"] for row in steered], [5, 50, 95]).tolist(),
                }
            )
    return {"definition": "paired clean test activations in stable manifest order", "cells": cells}


def run_analysis(output_root: Path, bootstrap: int = BOOTSTRAP_REPEATS) -> dict[str, Any]:
    steering_dir = output_root / "steering"
    config = json.loads((steering_dir / "config.json").read_text())
    rows = load_jsonl(steering_dir / "predictions.jsonl")
    selected, gate_metrics = directional_gate(rows, bootstrap, SEED)
    metrics = build_metrics(rows, bootstrap, SEED)
    dose_metrics = build_dose_metrics(rows, bootstrap, SEED)
    _write_csv(steering_dir / "metrics.csv", metrics)
    _write_csv(steering_dir / "dose_response_metrics.csv", dose_metrics)
    atomic_json(steering_dir / "metrics.json", {"cells": metrics, "dose_response": dose_metrics})
    atomic_jsonl(steering_dir / "significant_cells.jsonl", selected)
    atomic_json(steering_dir / "directional_gate.json", {"cells": gate_metrics, "selected_count": len(selected)})
    ood = [
        {key: row[key] for key in ("position", "layer", "alpha", "direction_type", "case_id", "activation_cosine", "activation_norm_ratio")}
        for row in rows if row.get("status") == "completed"
    ]
    _write_csv(steering_dir / "ood_diagnostics.csv", ood)
    atomic_json(steering_dir / "ood_summary.json", _ood_summary(output_root, rows, config))
    _plots(steering_dir, metrics, config)
    summary = {
        "status": "complete",
        "prediction_count": len([row for row in rows if row.get("status") == "completed"]),
        "significant_cell_count": len(selected),
        "bootstrap_repeats": bootstrap,
    }
    atomic_json(steering_dir / "analysis_summary.json", summary)
    return summary
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze Gemma activation steering")
    parser.add_argument("--output-root", default=str(RESULTS_ROOT))
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPEATS)
    args = parser.parse_args(argv)
    run_analysis(Path(args.output_root), args.bootstrap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
