from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from dp_SA.io_utils import atomic_json, load_jsonl

from .config import ANSWER_MATCHED_OUTPUT_ROOT, SEED
from .prepare_answer_matched import VARIANT


BOOTSTRAP_REPEATS = 2000


def _bootstrap_answer_equal(rows: Sequence[dict[str, Any]], seed_key: str) -> tuple[float, float]:
    grouped: dict[str, np.ndarray] = {}
    for answer in sorted({str(row["test_answer"]) for row in rows}):
        grouped[answer] = np.asarray([
            float(row["delta_image_attribution_score"])
            for row in rows if str(row["test_answer"]) == answer
        ])
    seed = int(hashlib.sha256(f"{SEED}|{seed_key}".encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAP_REPEATS, dtype=float)
    for index in range(BOOTSTRAP_REPEATS):
        estimates[index] = np.mean([
            rng.choice(values, size=len(values), replace=True).mean()
            for values in grouped.values()
        ])
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _stats(rows: Sequence[dict[str, Any]], seed_key: str) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "defined": False, "reason": "empty_group"}
    deltas = np.asarray([float(row["delta_image_attribution_score"]) for row in rows])
    answers = sorted({str(row["test_answer"]) for row in rows})
    answer_means = [
        float(np.mean([
            float(row["delta_image_attribution_score"])
            for row in rows if str(row["test_answer"]) == answer
        ]))
        for answer in answers
    ]
    ci_low, ci_high = _bootstrap_answer_equal(rows, seed_key)
    return {
        "n": len(rows), "defined": True, "answer_count": len(answers),
        "case_equal_delta_mean": float(deltas.mean()),
        "case_equal_delta_std": float(deltas.std()),
        "answer_equal_delta_mean": float(np.mean(answer_means)),
        "answer_equal_bootstrap_ci_low": ci_low,
        "answer_equal_bootstrap_ci_high": ci_high,
        "label_change_rate": float(np.mean([bool(row["label_changed"]) for row in rows])),
        "midpoint_cross_rate": float(np.mean([bool(row["crossed_midpoint"]) for row in rows])),
        "steered_score_mean": float(np.mean([float(row["steered_image_attribution_score"]) for row in rows])),
        "steered_label_probability_mass_mean": float(np.mean([
            float(row["steered_label_probability_mass"]) for row in rows
        ])),
        "actual_perturbation_norm_mean": float(np.mean([
            float(row["actual_perturbation_norm"]) for row in rows
        ])),
        "hook_applied_counts": sorted({
            int(row["hook_diagnostics"]["steering_applied_count"]) for row in rows
        }),
        "alpha_zero_max_abs_delta": (
            float(np.max(np.abs(deltas))) if float(rows[0]["alpha"]) == 0.0 else None
        ),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plots(rows: Sequence[dict[str, Any]], summary: Sequence[dict[str, Any]], root: Path) -> list[str]:
    overall = [row for row in summary if row["group"] == "overall"]
    outputs: list[str] = []
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len({float(row["alpha"]) for row in overall})))
    for position in ("LAT", "PANL", "CLE"):
        subset = [row for row in overall if row["position"] == position]
        figure, axis = plt.subplots(figsize=(7.5, 5.2))
        for color, alpha in zip(colors, sorted({float(row["alpha"]) for row in subset})):
            cells = sorted((row for row in subset if float(row["alpha"]) == alpha), key=lambda row: int(row["layer"]))
            axis.plot([row["layer"] for row in cells], [row["answer_equal_delta_mean"] for row in cells], marker="o", color=color, label=f"alpha={alpha:g}")
        axis.axhline(0, color="black", linewidth=.8)
        axis.set_xlabel("Zero-based decoder layer")
        axis.set_ylabel("Answer-equal mean ΔSA")
        axis.set_title(f"{position}: answer-matched layer response")
        axis.grid(alpha=.2); axis.legend(fontsize=8, ncol=2); figure.tight_layout()
        path = root / VARIANT / "figures" / f"layer_curves_{position}.png"
        path.parent.mkdir(parents=True, exist_ok=True); figure.savefig(path, dpi=180); plt.close(figure)
        outputs.append(str(path))
    return outputs


def analyze(output_root: Path = ANSWER_MATCHED_OUTPUT_ROOT) -> dict[str, Any]:
    root = output_root.resolve()
    predictions_path = root / VARIANT / "tables" / "predictions.jsonl"
    rows = [row for row in load_jsonl(predictions_path) if row.get("status") == "completed"]
    if not rows:
        raise ValueError("No completed answer-matched predictions")
    groups: dict[tuple[str, int, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cell = (str(row["position"]), int(row["layer"]), float(row["alpha"]))
        groups[cell + ("overall",)].append(row)
        groups[cell + (f'clean_sa_group:{row["clean_sa_group"]}',)].append(row)
        groups[cell + (f'pair_type:{row["pair_type"]}',)].append(row)
    summary_rows: list[dict[str, Any]] = []
    for (position, layer, alpha, group), values in sorted(groups.items()):
        summary_rows.append({
            "variant": VARIANT, "direction": "matched_loao", "position": position,
            "layer": layer, "alpha": alpha, "group": group,
            **_stats(values, f"{position}|{layer}|{alpha}|{group}"),
        })
    _write_csv(root / VARIANT / "tables" / "cell_summary.csv", summary_rows)
    figures = _plots(rows, summary_rows, root)
    overall = [row for row in summary_rows if row["group"] == "overall"]
    strongest = max(overall, key=lambda row: abs(float(row["answer_equal_delta_mean"])))
    result = {
        "status": "complete", "completed_predictions": len(rows),
        "unique_cases": len({row["case_id"] for row in rows}),
        "cell_count": len(overall), "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "hook_applied_counts": sorted({int(row["hook_diagnostics"]["steering_applied_count"]) for row in rows}),
        "alpha_zero_max_abs_delta": max(
            abs(float(row["delta_image_attribution_score"])) for row in rows if float(row["alpha"]) == 0.0
        ),
        "strongest_absolute_answer_equal_cell": strongest,
        "figures": figures,
    }
    atomic_json(root / VARIANT / "tables" / "analysis_summary.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze native answer-matched LOAO steering")
    parser.add_argument("--output-root", type=Path, default=ANSWER_MATCHED_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    print(json.dumps(analyze(args.output_root), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
