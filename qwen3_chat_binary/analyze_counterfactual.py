from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

from dp_SA.io_utils import atomic_json, load_jsonl

from .config import COUNTERFACTUAL_OUTPUT_ROOT, VARIANTS


def metrics(x: np.ndarray, y: np.ndarray, *, midpoint: float = 0.5) -> dict[str, Any]:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return {"n": len(x), "defined": False, "reason": "insufficient_or_constant"}
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    residual = float(np.sum((y - fitted) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    mask = (x != midpoint) & (y != midpoint)
    return {
        "n": len(x), "defined": True,
        "pearson": float(pearsonr(x, y).statistic),
        "spearman": float(spearmanr(x, y).statistic),
        "kendall_tau_b": float(kendalltau(x, y).statistic),
        "r2": 1.0 - residual / total if total else None,
        "slope": float(slope), "intercept": float(intercept),
        "mae": float(np.mean(np.abs(y - x))),
        "rmse": float(np.sqrt(np.mean((y - x) ** 2))),
        "direction_n": int(mask.sum()),
        "direction_agreement": float(np.mean((x[mask] > 0.5) == (y[mask] > 0.5))) if mask.any() else None,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else ["empty"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def plot_scatter(rows: list[dict[str, Any]], x_key: str, y_key: str, path: Path, title: str) -> None:
    colors = {"hard_text_easy_image": "#d95f02", "balanced": "#7570b3", "hard_image_easy_text": "#1b9e77"}
    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    for group, color in colors.items():
        subset = [row for row in rows if row["pair_type"] == group]
        axis.scatter([row[x_key] for row in subset], [row[y_key] for row in subset], s=22, alpha=.68, label=group, color=color)
    x = np.asarray([row[x_key] for row in rows], float)
    y = np.asarray([row[y_key] for row in rows], float)
    if len(x) > 1 and np.std(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(x.min(), x.max(), 100)
        axis.plot(grid, intercept + slope * grid, "k--", linewidth=1.5, label="linear fit")
    axis.plot([x.min(), x.max()], [x.min(), x.max()], color="gray", linestyle=":", label="y=x")
    axis.set_xlabel("CMA image attribution")
    axis.set_ylabel("Verbal SA image attribution")
    axis.set_title(title)
    axis.grid(alpha=.2)
    axis.legend(fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def analyze(root: Path) -> dict[str, Any]:
    trials = {
        row["case_id"]: row for row in load_jsonl(root / "tables" / "four_cell_results.jsonl")
        if row.get("status") == "completed" and row.get("original_answer_reproduced")
    }
    construction = {row["case_id"]: row for row in load_jsonl(root / "tables" / "construction_manifest.jsonl")}
    combined: list[dict[str, Any]] = []
    comparison: list[dict[str, Any]] = []
    non_extreme_comparison: list[dict[str, Any]] = []
    for variant in VARIANTS:
        sa_rows = {
            row["case_id"]: row for row in load_jsonl(
                root.parent / "Capture" / variant / "tables" / "results.jsonl"
            ) if row.get("status") == "completed"
        }
        variant_rows: list[dict[str, Any]] = []
        for case_id in sorted(set(trials) & set(sa_rows)):
            trial, sa, build = trials[case_id], sa_rows[case_id], construction[case_id]
            prob = trial["cma_probability"]["image_share"]
            logit = trial["cma_logit"]["image_share"]
            if prob is None or logit is None:
                continue
            row = {
                "case_id": case_id, "variant": variant, "pair_type": trial["pair_type"],
                "answer_source": trial["answer_source"],
                "distractor_color_collision": trial["distractor_color_collision"],
                "third_color": trial["third_color"],
                "sa_image": float(sa["image_attribution_score"]),
                "cma_probability_image": float(prob),
                "cma_logit_image": float(logit),
                "image_probability_drop": trial["image_only_probability_drop"],
                "text_probability_drop": trial["text_only_probability_drop"],
                "interaction_probability": trial["cma_probability"]["interaction"],
                "interaction_logit": trial["cma_logit"]["interaction"],
                "text_entropy_delta": build["text_match_deltas"]["entropy_delta"],
                "text_probability_delta": build["text_match_deltas"]["target_probability_delta"],
            }
            variant_rows.append(row)
            combined.append(row)
        groups = {"overall": variant_rows}
        for value in ("hard_text_easy_image", "balanced", "hard_image_easy_text"):
            groups[f"pair_type:{value}"] = [row for row in variant_rows if row["pair_type"] == value]
        for value in ("text", "image", "other"):
            groups[f"answer_source:{value}"] = [row for row in variant_rows if row["answer_source"] == value]
        for value in (False, True):
            groups[f"distractor_collision:{str(value).lower()}"] = [row for row in variant_rows if row["distractor_color_collision"] is value]
        for group, values in groups.items():
            for kind, x_key, y_key in (
                ("probability", "cma_probability_image", "sa_image"),
                ("logit", "cma_logit_image", "sa_image"),
            ):
                score = metrics(
                    np.asarray([row[x_key] for row in values], float),
                    np.asarray([row[y_key] for row in values], float),
                    midpoint=0.5,
                )
                comparison.append({
                    "variant": variant, "group": group, "cma_kind": kind,
                    "cma_scale": "image_share_0_1", "verbal_sa_scale": "image_share_0_1",
                    **score,
                })
        for kind, x_key in (
            ("probability", "cma_probability_image"),
            ("logit", "cma_logit_image"),
        ):
            values = [row for row in variant_rows if 0.1 < row[x_key] < 0.9]
            score = metrics(
                np.asarray([row[x_key] for row in values], float),
                np.asarray([row["sa_image"] for row in values], float),
                midpoint=0.5,
            )
            non_extreme_comparison.append({
                "variant": variant, "cma_kind": kind,
                "filter": "0.1<cma_image_share<0.9",
                "cma_scale": "image_share_0_1", "verbal_sa_scale": "image_share_0_1",
                **score,
            })
        write_csv(root / variant / "tables" / "case_scores.csv", variant_rows)
        plot_scatter(variant_rows, "cma_probability_image", "sa_image", root / variant / "figures" / "probability_cma_vs_sa.png", f"{variant}: probability CMA vs SA")
        plot_scatter(variant_rows, "cma_logit_image", "sa_image", root / variant / "figures" / "logit_cma_vs_sa.png", f"{variant}: logit CMA image share vs verbal SA")
    write_csv(root / "tables" / "cma_sa_case_scores.csv", combined)
    write_csv(root / "tables" / "cma_sa_comparison.csv", comparison)
    write_csv(root / "tables" / "cma_sa_non_extreme_comparison.csv", non_extreme_comparison)
    plot_scatter(
        combined, "cma_probability_image", "sa_image",
        root / "figures" / "prob" / "cma_sa_combined.png",
        "Both boundary variants: probability CMA vs SA",
    )
    plot_scatter(
        combined, "cma_logit_image", "sa_image",
        root / "figures" / "logit" / "cma_sa_combined.png",
        "Both boundary variants: logit CMA image share vs verbal SA",
    )
    statuses = Counter(row.get("status") for row in load_jsonl(root / "tables" / "construction_manifest.jsonl"))
    run_statuses = Counter(row.get("status") for row in load_jsonl(root / "tables" / "four_cell_results.jsonl"))
    summary = {
        "construction_statuses": dict(statuses), "run_statuses": dict(run_statuses),
        "valid_trials": len(trials), "comparison_rows": len(comparison),
        "non_extreme_filter": "0.1<cma_image_share<0.9",
        "non_extreme": non_extreme_comparison,
        "analysis_scale": {
            "cma": "image_share_0_1",
            "verbal_sa": "image_share_0_1",
            "signed_minus1_to1_used": False,
        },
        "overall": [row for row in comparison if row["group"] == "overall"],
    }
    atomic_json(root / "tables" / "analysis_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze counterfactual CMA against verbal SA")
    parser.add_argument("--output-root", type=Path, default=COUNTERFACTUAL_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    print(json.dumps(analyze(args.output_root.resolve()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
