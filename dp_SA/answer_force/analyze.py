from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress, pearsonr, spearmanr

from dp_SA.io_utils import atomic_json, load_jsonl

from .config import BOOTSTRAP_REPEATS, CONDITIONS, MIDPOINTS, SEED


GROUPS = ("all", "original_text", "original_image", "conflict_easy", "conflict_hard")


def _finite(value: Any) -> float:
    output = float(value)
    if not math.isfinite(output):
        raise ValueError(f"non-finite analysis value: {value!r}")
    return output


def _mean_sem(values: Sequence[float]) -> tuple[float, float | None]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean()) if len(array) else float("nan")
    sem = float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else None
    return mean, sem


def _correlation(left: Sequence[float], right: Sequence[float], method: str) -> float:
    if len(left) < 2 or len(set(left)) < 2 or len(set(right)) < 2:
        return float("nan")
    result = pearsonr(left, right) if method == "pearson" else spearmanr(left, right)
    return float(result.statistic)


def item_metrics(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_item: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in results:
        if row.get("status") != "completed":
            continue
        item = str(row["item_id"])
        condition = str(row["condition"])
        if condition not in CONDITIONS:
            raise ValueError(f"unknown Answer-force condition: {condition}")
        if condition in by_item[item]:
            raise ValueError(f"duplicate condition for item {item}: {condition}")
        by_item[item][condition] = row
    output: list[dict[str, Any]] = []
    for item, grid in sorted(by_item.items()):
        if set(grid) != set(CONDITIONS):
            raise ValueError(f"item {item} does not have all three conditions")
        clean, opposite, unrelated = grid["clean"], grid["force_opposite"], grid["force_unrelated"]
        direction = float(clean["forced_direction"])
        row: dict[str, Any] = {
            "item_id": item, "case_id": clean["case_id"], "origin": clean["origin"], "difficulty": clean["difficulty"],
            "forced_direction": direction, "clean_answer_token_length": int(clean["answer_token_length"]),
            "opposite_answer_token_length": int(opposite["answer_token_length"]),
            "unrelated_answer_token_length": int(unrelated["answer_token_length"]),
            "opposite_token_length_difference": abs(int(opposite["answer_token_length"]) - int(clean["answer_token_length"])),
            "unrelated_token_length_difference": abs(int(unrelated["answer_token_length"]) - int(clean["answer_token_length"])),
            "clean_panl_sa": _finite(clean["panl_sa"]), "clean_final_sa": _finite(clean["final_soft_sa"]),
            "clean_final_hard_class": int(clean["final_hard_class"]), "clean_panl_pseudo_hard_class": int(clean["panl_pseudo_hard_class"]),
        }
        for condition, value in (("opposite", opposite), ("unrelated", unrelated)):
            panl_delta = _finite(value["panl_sa"]) - row["clean_panl_sa"]
            final_delta = _finite(value["final_soft_sa"]) - row["clean_final_sa"]
            prefix = condition
            row.update({
                f"{prefix}_panl_delta": panl_delta, f"{prefix}_panl_abs_delta": abs(panl_delta),
                f"{prefix}_final_delta": final_delta, f"{prefix}_final_abs_delta": abs(final_delta),
                f"{prefix}_final_hard_change": int(int(value["final_hard_class"]) != row["clean_final_hard_class"]),
                f"{prefix}_panl_pseudo_hard_change": int(int(value["panl_pseudo_hard_class"]) != row["clean_panl_pseudo_hard_class"]),
                f"{prefix}_panl_final_alignment_gap": abs(_finite(value["panl_sa"]) - _finite(value["final_soft_sa"])),
                f"{prefix}_final_hard_class": int(value["final_hard_class"]),
                f"{prefix}_panl_pseudo_hard_class": int(value["panl_pseudo_hard_class"]),
            })
            if condition == "opposite":
                row["opposite_panl_oriented_delta"] = direction * panl_delta
                row["opposite_final_oriented_delta"] = direction * final_delta
                row["opposite_final_hard_toward_force"] = int(direction * (int(value["final_hard_class"]) - row["clean_final_hard_class"]) > 0)
                row["opposite_panl_pseudo_hard_toward_force"] = int(direction * (int(value["panl_pseudo_hard_class"]) - row["clean_panl_pseudo_hard_class"]) > 0)
            else:
                # Unrelated answers have no modal identity of their own, but
                # retain the recipient's text↔image direction so the paired
                # specificity contrast is expressed on one common axis.
                row["unrelated_panl_oriented_delta"] = direction * panl_delta
                row["unrelated_final_oriented_delta"] = direction * final_delta
        row["paired_abs_contrast_panl"] = row["opposite_panl_abs_delta"] - row["unrelated_panl_abs_delta"]
        row["paired_abs_contrast_final"] = row["opposite_final_abs_delta"] - row["unrelated_final_abs_delta"]
        row["paired_specificity_contrast_panl"] = row["opposite_panl_oriented_delta"] - row["unrelated_panl_oriented_delta"]
        row["paired_specificity_contrast_final"] = row["opposite_final_oriented_delta"] - row["unrelated_final_oriented_delta"]
        row["opposite_panl_final_sign_agreement"] = int(np.sign(row["opposite_panl_delta"]) == np.sign(row["opposite_final_delta"]))
        row["unrelated_panl_final_sign_agreement"] = int(np.sign(row["unrelated_panl_delta"]) == np.sign(row["unrelated_final_delta"]))
        output.append(row)
    if not output:
        raise ValueError("no completed Answer-force items to analyze")
    return output


def _group_rows(rows: Sequence[Mapping[str, Any]], group: str) -> list[Mapping[str, Any]]:
    if group == "all":
        return list(rows)
    if group in {"original_text", "original_image"}:
        origin = group.removeprefix("original_")
        return [row for row in rows if row["origin"] == origin]
    difficulty = group.removeprefix("conflict_")
    return [row for row in rows if row["difficulty"] == difficulty]


def _bundle(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    def mean(name: str) -> float:
        values = [float(row[name]) for row in rows]
        return float(np.mean(values)) if values else float("nan")

    output = {
        "opposite_panl_signed_delta": mean("opposite_panl_delta"), "opposite_panl_absolute_delta": mean("opposite_panl_abs_delta"),
        "opposite_panl_oriented_delta": mean("opposite_panl_oriented_delta"), "opposite_final_signed_delta": mean("opposite_final_delta"),
        "opposite_final_absolute_delta": mean("opposite_final_abs_delta"), "opposite_final_oriented_delta": mean("opposite_final_oriented_delta"),
        "opposite_final_hard_change_rate": mean("opposite_final_hard_change"), "opposite_final_toward_force_rate": mean("opposite_final_hard_toward_force"),
        "opposite_panl_pseudo_hard_change_rate": mean("opposite_panl_pseudo_hard_change"), "opposite_panl_pseudo_hard_toward_force_rate": mean("opposite_panl_pseudo_hard_toward_force"),
        "unrelated_panl_signed_delta": mean("unrelated_panl_delta"), "unrelated_panl_absolute_delta": mean("unrelated_panl_abs_delta"),
        "unrelated_panl_oriented_delta": mean("unrelated_panl_oriented_delta"),
        "unrelated_final_signed_delta": mean("unrelated_final_delta"), "unrelated_final_absolute_delta": mean("unrelated_final_abs_delta"),
        "unrelated_final_oriented_delta": mean("unrelated_final_oriented_delta"),
        "unrelated_final_hard_change_rate": mean("unrelated_final_hard_change"), "unrelated_panl_pseudo_hard_change_rate": mean("unrelated_panl_pseudo_hard_change"),
        "paired_abs_contrast_panl": mean("paired_abs_contrast_panl"), "paired_abs_contrast_final": mean("paired_abs_contrast_final"),
        "paired_specificity_contrast_panl": mean("paired_specificity_contrast_panl"), "paired_specificity_contrast_final": mean("paired_specificity_contrast_final"),
        "opposite_panl_final_pearson": _correlation([float(row["opposite_panl_delta"]) for row in rows], [float(row["opposite_final_delta"]) for row in rows], "pearson"),
        "opposite_panl_final_spearman": _correlation([float(row["opposite_panl_delta"]) for row in rows], [float(row["opposite_final_delta"]) for row in rows], "spearman"),
        "opposite_panl_final_sign_agreement": mean("opposite_panl_final_sign_agreement"),
        "unrelated_panl_final_pearson": _correlation([float(row["unrelated_panl_delta"]) for row in rows], [float(row["unrelated_final_delta"]) for row in rows], "pearson"),
        "unrelated_panl_final_spearman": _correlation([float(row["unrelated_panl_delta"]) for row in rows], [float(row["unrelated_final_delta"]) for row in rows], "spearman"),
        "unrelated_panl_final_sign_agreement": mean("unrelated_panl_final_sign_agreement"),
        "opposite_panl_alignment_gap": mean("opposite_panl_final_alignment_gap"), "opposite_final_alignment_gap": mean("opposite_panl_final_alignment_gap"),
        "unrelated_panl_alignment_gap": mean("unrelated_panl_final_alignment_gap"), "unrelated_final_alignment_gap": mean("unrelated_panl_final_alignment_gap"),
    }
    return output


def _metric_values(rows: Sequence[Mapping[str, Any]], name: str) -> list[float]:
    direct = {
        "opposite_panl_signed_delta": "opposite_panl_delta", "opposite_panl_absolute_delta": "opposite_panl_abs_delta", "opposite_panl_oriented_delta": "opposite_panl_oriented_delta",
        "opposite_final_signed_delta": "opposite_final_delta", "opposite_final_absolute_delta": "opposite_final_abs_delta", "opposite_final_oriented_delta": "opposite_final_oriented_delta",
        "opposite_final_hard_change_rate": "opposite_final_hard_change", "opposite_final_toward_force_rate": "opposite_final_hard_toward_force", "opposite_panl_pseudo_hard_change_rate": "opposite_panl_pseudo_hard_change", "opposite_panl_pseudo_hard_toward_force_rate": "opposite_panl_pseudo_hard_toward_force",
        "unrelated_panl_signed_delta": "unrelated_panl_delta", "unrelated_panl_absolute_delta": "unrelated_panl_abs_delta", "unrelated_final_signed_delta": "unrelated_final_delta", "unrelated_final_absolute_delta": "unrelated_final_abs_delta", "unrelated_final_hard_change_rate": "unrelated_final_hard_change", "unrelated_panl_pseudo_hard_change_rate": "unrelated_panl_pseudo_hard_change",
        "unrelated_panl_oriented_delta": "unrelated_panl_oriented_delta", "unrelated_final_oriented_delta": "unrelated_final_oriented_delta", "paired_abs_contrast_panl": "paired_abs_contrast_panl", "paired_abs_contrast_final": "paired_abs_contrast_final", "paired_specificity_contrast_panl": "paired_specificity_contrast_panl", "paired_specificity_contrast_final": "paired_specificity_contrast_final", "opposite_panl_final_sign_agreement": "opposite_panl_final_sign_agreement", "unrelated_panl_final_sign_agreement": "unrelated_panl_final_sign_agreement",
        "opposite_panl_alignment_gap": "opposite_panl_final_alignment_gap", "opposite_final_alignment_gap": "opposite_panl_final_alignment_gap", "unrelated_panl_alignment_gap": "unrelated_panl_final_alignment_gap", "unrelated_final_alignment_gap": "unrelated_panl_final_alignment_gap",
    }
    if name in direct:
        return [float(row[direct[name]]) for row in rows]
    return []


def bootstrap_aggregates(rows: Sequence[dict[str, Any]], *, repeats: int = BOOTSTRAP_REPEATS, seed: int = SEED) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    point_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for group_index, group in enumerate(GROUPS):
        cohort = _group_rows(rows, group)
        if not cohort:
            continue
        point = _bundle(cohort)
        rng = np.random.default_rng(seed + 1009 * (group_index + 1))
        values: dict[str, list[float]] = {name: [] for name in point}
        for repeat in range(repeats):
            indices = rng.integers(0, len(cohort), size=len(cohort))
            sampled = [cohort[int(index)] for index in indices]
            bundle = _bundle(sampled)
            bundle_row = {"group": group, "repeat": repeat, **bundle}
            bootstrap_rows.append(bundle_row)
            for name, value in bundle.items():
                if math.isfinite(float(value)):
                    values[name].append(float(value))
        for name, point_value in point.items():
            sampled_values = values[name]
            metric_values = _metric_values(cohort, name)
            if metric_values:
                mean, sem = _mean_sem(metric_values)
            elif len(sampled_values) > 1:
                # Correlations and other derived statistics do not have a
                # scalar item-level value; report their bootstrap standard
                # error while retaining the same joint resamples/CI.
                mean, sem = float(point_value), float(np.std(sampled_values, ddof=1))
            else:
                mean, sem = float(point_value), None
            ci_low, ci_high = (None, None) if not sampled_values else map(float, np.percentile(np.asarray(sampled_values), [2.5, 97.5]))
            point_rows.append({"group": group, "metric": name, "mean": float(point_value), "sem": sem, "ci_low": ci_low, "ci_high": ci_high, "sample_count": len(cohort), "valid_bootstrap_repeats": len(sampled_values), "bootstrap_repeats": repeats})
    return point_rows, bootstrap_rows


def _regression_row(
    cohort: Sequence[Mapping[str, Any]], y_name: str, x_name: str, *, group: str,
    condition: str, endpoint: str, outcome: str, subset: str = "all",
) -> dict[str, Any]:
    x = np.asarray([float(row[x_name]) for row in cohort], dtype=float)
    y = np.asarray([float(row[y_name]) for row in cohort], dtype=float)
    base = {"group": group, "condition": condition, "endpoint": endpoint, "outcome": outcome, "subset": subset, "covariate": "abs_answer_token_length_difference", "sample_count": len(cohort)}
    if len(cohort) < 3 or len(set(x.tolist())) < 2:
        return {**base, "status": "undefined", "intercept": None, "slope": None, "slope_stderr": None, "p_value": None, "r_squared": None, "ci_low": None, "ci_high": None}
    fit = linregress(x, y)
    return {**base, "status": "complete", "intercept": float(fit.intercept), "slope": float(fit.slope), "slope_stderr": float(fit.stderr), "p_value": float(fit.pvalue), "r_squared": float(fit.rvalue ** 2), "ci_low": float(fit.slope - 1.96 * fit.stderr), "ci_high": float(fit.slope + 1.96 * fit.stderr)}


def regression_results(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for group in GROUPS:
        cohort = _group_rows(rows, group)
        for condition in ("opposite", "unrelated"):
            x = f"{condition}_token_length_difference"
            for endpoint in ("panl", "final"):
                for outcome, suffix in (("signed", "delta"), ("absolute", "abs_delta")):
                    output.append(_regression_row(cohort, f"{condition}_{endpoint}_{suffix}", x, group=group, condition=condition, endpoint=endpoint, outcome=outcome))
                if condition == "opposite":
                    output.append(_regression_row(cohort, f"opposite_{endpoint}_oriented_delta", x, group=group, condition=condition, endpoint=endpoint, outcome="oriented"))
                    matched = [row for row in cohort if int(row["opposite_token_length_difference"]) == 0]
                    output.append(_regression_row(matched, f"opposite_{endpoint}_oriented_delta", x, group=group, condition=condition, endpoint=endpoint, outcome="oriented", subset="clean_opposite_token_equal"))
    return output


def correlation_results(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a standalone PANL/final correlation table.

    The signed and absolute outcomes are reported separately for every
    requested cohort and intervention condition.  This keeps the correlation
    estimates easy to consume without parsing the wide aggregate table.
    """
    output: list[dict[str, Any]] = []
    for group in GROUPS:
        cohort = _group_rows(rows, group)
        for condition in ("opposite", "unrelated"):
            for outcome, suffix in (("signed_delta", "delta"), ("absolute_delta", "abs_delta")):
                panl = [float(row[f"{condition}_panl_{suffix}"]) for row in cohort]
                final = [float(row[f"{condition}_final_{suffix}"]) for row in cohort]
                output.append({
                    "group": group,
                    "condition": condition,
                    "outcome": outcome,
                    "pearson": _correlation(panl, final, "pearson"),
                    "spearman": _correlation(panl, final, "spearman"),
                    "sign_agreement": (
                        float(np.mean([row[f"{condition}_panl_final_sign_agreement"] for row in cohort]))
                        if outcome == "signed_delta" and cohort else None
                    ),
                    "sample_count": len(cohort),
                })
    return output


def clean_force_soft_sa_correlations(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Correlate clean soft SA with each forced-condition soft SA.

    Rows are intentionally pooled across original text/image origins.  PANL
    uses the reconstructed continuous probe value and final uses the midpoint
    soft score, so both endpoints remain on their native continuous scales.
    """
    output: list[dict[str, Any]] = []
    for condition in ("opposite", "unrelated"):
        for endpoint in ("panl", "final"):
            clean_values = [float(row[f"clean_{endpoint}_sa"]) for row in rows]
            forced_values = [
                float(row[f"clean_{endpoint}_sa"]) + float(row[f"{condition}_{endpoint}_delta"])
                for row in rows
            ]
            output.append({
                "comparison": "clean_vs_force",
                "condition": f"force_{condition}",
                "endpoint": endpoint,
                "pearson": _correlation(clean_values, forced_values, "pearson"),
                "spearman": _correlation(clean_values, forced_values, "spearman"),
                "sample_count": len(rows),
            })
    return output


def specificity_results(
    aggregate: Sequence[Mapping[str, Any]],
    bootstrap: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select the four paired directional effects and their bootstrap CIs."""
    wanted = {
        "opposite_panl_oriented_delta": "D_panl_opp",
        "unrelated_panl_oriented_delta": "D_panl_unrel",
        "paired_specificity_contrast_panl": "C_panl",
        "opposite_final_oriented_delta": "D_final_opp",
        "unrelated_final_oriented_delta": "D_final_unrel",
        "paired_specificity_contrast_final": "C_final",
    }
    output: list[dict[str, Any]] = []
    for row in aggregate:
        label = wanted.get(str(row["metric"]))
        if label is None:
            continue
        output.append({
            "group": row["group"], "metric": label,
            "mean": row["mean"], "sem": row["sem"],
            "ci_low": row["ci_low"], "ci_high": row["ci_high"],
            "sample_count": row["sample_count"],
            "valid_bootstrap_repeats": row["valid_bootstrap_repeats"],
            "bootstrap_repeats": row["bootstrap_repeats"],
        })
    return output


def hard_class_directional_proportions(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Report unchanged/toward-force/wrong-way proportions for opposite."""
    output: list[dict[str, Any]] = []
    for group in GROUPS:
        cohort = _group_rows(rows, group)
        for endpoint, clean_key, forced_key in (
            ("final", "clean_final_hard_class", "opposite_final_hard_class"),
            ("panl_pseudo_hard", "clean_panl_pseudo_hard_class", "opposite_panl_pseudo_hard_class"),
        ):
            categories = {"unchanged": 0, "toward_force": 0, "wrong_way": 0}
            for row in cohort:
                signed_class_delta = float(row["forced_direction"]) * (int(row[forced_key]) - int(row[clean_key]))
                if signed_class_delta == 0:
                    categories["unchanged"] += 1
                elif signed_class_delta > 0:
                    categories["toward_force"] += 1
                else:
                    categories["wrong_way"] += 1
            n = len(cohort)
            for category, count in categories.items():
                output.append({
                    "group": group, "condition": "force_opposite", "endpoint": endpoint,
                    "category": category, "proportion": (float(count) / n if n else None),
                    "count": count, "sample_count": n,
                })
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    buffer = io.StringIO()
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    else:
        buffer.write("\n")
    _atomic_text(path, buffer.getvalue())


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_figure(path: Path, figure: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".png", dir=path.parent)
    os.close(fd)
    try:
        figure.savefig(temporary, format="png", dpi=160)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _plot_grouped(output: Path, aggregate: Sequence[Mapping[str, Any]], metrics: Sequence[str], filename: str, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, origin in zip(axes, ("original_text", "original_image")):
        selected = [row for row in aggregate if row["group"] == origin and row["metric"] in metrics]
        labels = [str(row["metric"]).replace("_", "\n") for row in selected]
        values = [float(row["mean"]) for row in selected]
        lows = [float(row["ci_low"]) if row["ci_low"] is not None else value for row, value in zip(selected, values)]
        highs = [float(row["ci_high"]) if row["ci_high"] is not None else value for row, value in zip(selected, values)]
        axis.errorbar(range(len(values)), values, yerr=[np.asarray(values) - lows, highs - np.asarray(values)], fmt="o", capsize=3)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=8)
        axis.set_title(origin.replace("original_", "original "))
    fig.suptitle(title)
    fig.tight_layout()
    _atomic_figure(output / filename, fig)
    plt.close(fig)


def _plot_scatter(output: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {"text": "tab:blue", "image": "tab:orange"}
    for condition, marker in (("opposite", "o"), ("unrelated", "s")):
        for origin in ("text", "image"):
            cohort = [row for row in rows if row["origin"] == origin]
            ax.scatter([row[f"{condition}_panl_delta"] for row in cohort], [row[f"{condition}_final_delta"] for row in cohort], c=colors[origin], marker=marker, alpha=0.7, label=f"{condition}/{origin}")
    values = [abs(float(value)) for row in rows for value in (row["opposite_panl_delta"], row["opposite_final_delta"], row["unrelated_panl_delta"], row["unrelated_final_delta"])]
    bound = max(values, default=1.0)
    ax.plot([-bound, bound], [-bound, bound], "k--", linewidth=0.8)
    stats_lines = []
    for condition in ("opposite", "unrelated"):
        x = [float(row[f"{condition}_panl_delta"]) for row in rows]
        y = [float(row[f"{condition}_final_delta"]) for row in rows]
        stats_lines.append(
            f"{condition}: r={_correlation(x, y, 'pearson'):.2f}, "
            f"rho={_correlation(x, y, 'spearman'):.2f}"
        )
    ax.text(0.02, 0.98, "\n".join(stats_lines), transform=ax.transAxes, va="top", fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.7"})
    ax.set_xlabel("PANL delta"); ax.set_ylabel("Final delta"); ax.legend(fontsize=8); fig.tight_layout(); _atomic_figure(output / "panl_vs_final_delta_scatter.png", fig); plt.close(fig)


def _plot_original_forced(output: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for axis, origin in zip(axes, ("text", "image")):
        cohort = [row for row in rows if row["origin"] == origin]
        for endpoint, color in (("panl", "tab:blue"), ("final", "tab:orange")):
            axis.scatter([row[f"clean_{endpoint}_sa"] for row in cohort], [row[f"opposite_{endpoint}_delta"] + row[f"clean_{endpoint}_sa"] for row in cohort], color=color, label=endpoint)
        axis.set_title(f"original {origin}"); axis.set_xlabel("clean SA"); axis.set_ylabel("opposite SA"); axis.legend()
    fig.tight_layout(); _atomic_figure(output / "original_vs_forced_sa.png", fig); plt.close(fig)


def _plot_absolute_delta_overall(output: Path, aggregate: Sequence[Mapping[str, Any]]) -> None:
    """Plot absolute SA deltas for all items without an origin split."""
    metrics = (
        ("opposite_panl_absolute_delta", "opposite / PANL"),
        ("opposite_final_absolute_delta", "opposite / final"),
        ("unrelated_panl_absolute_delta", "unrelated / PANL"),
        ("unrelated_final_absolute_delta", "unrelated / final"),
    )
    selected = {
        str(row["metric"]): row
        for row in aggregate
        if row["group"] == "all" and row["metric"] in {metric for metric, _label in metrics}
    }
    labels = [label for metric, label in metrics if metric in selected]
    values = [float(selected[metric]["mean"]) for metric, _label in metrics if metric in selected]
    lows = [float(selected[metric]["ci_low"]) if selected[metric]["ci_low"] is not None else value for metric, value in ((metric, float(selected[metric]["mean"])) for metric, _label in metrics if metric in selected)]
    highs = [float(selected[metric]["ci_high"]) if selected[metric]["ci_high"] is not None else value for metric, value in ((metric, float(selected[metric]["mean"])) for metric, _label in metrics if metric in selected)]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.errorbar(
        range(len(values)), values,
        yerr=[np.asarray(values) - np.asarray(lows), np.asarray(highs) - np.asarray(values)],
        fmt="o", capsize=4, color="tab:blue",
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    axis.set_ylabel("Absolute delta SA")
    axis.set_title("Absolute PANL/final SA delta (all items; bootstrap 95% CI)")
    fig.tight_layout()
    _atomic_figure(output / "delta_sa_absolute_overall.png", fig)
    _atomic_figure(output / "panl_final_absolute_delta_overall.png", fig)
    plt.close(fig)


def _summary(rows: Sequence[dict[str, Any]], aggregate: Sequence[Mapping[str, Any]], matching: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    def lower(group: str, metric: str) -> float | None:
        values = [row for row in aggregate if row["group"] == group and row["metric"] == metric]
        return None if not values or values[0]["ci_low"] is None else float(values[0]["ci_low"])

    def upper(group: str, metric: str) -> float | None:
        values = [row for row in aggregate if row["group"] == group and row["metric"] == metric]
        return None if not values or values[0]["ci_high"] is None else float(values[0]["ci_high"])

    checks = {
        "opposite_panl_oriented_positive": (lower("all", "opposite_panl_oriented_delta") or 0.0) > 0,
        "opposite_final_oriented_positive": (lower("all", "opposite_final_oriented_delta") or 0.0) > 0,
        "opposite_final_changed": (lower("all", "opposite_final_hard_change_rate") or 0.0) > 0,
        "opposite_panl_changed": (lower("all", "opposite_panl_absolute_delta") or 0.0) > 0,
        "opposite_panl_final_absolute_contrast_positive": (lower("all", "paired_abs_contrast_panl") or 0.0) > 0 and (lower("all", "paired_abs_contrast_final") or 0.0) > 0,
        "opposite_panl_direction_positive": (lower("all", "opposite_panl_oriented_delta") or 0.0) > 0,
        "opposite_final_direction_positive": (lower("all", "opposite_final_oriented_delta") or 0.0) > 0,
        "panl_specificity_positive": (lower("all", "paired_specificity_contrast_panl") or 0.0) > 0,
        "final_specificity_positive": (lower("all", "paired_specificity_contrast_final") or 0.0) > 0,
        "unrelated_panl_direction_positive": (lower("all", "unrelated_panl_oriented_delta") or 0.0) > 0,
        "unrelated_final_direction_positive": (lower("all", "unrelated_final_oriented_delta") or 0.0) > 0,
        "unrelated_panl_unchanged": (upper("all", "unrelated_panl_absolute_delta") or 0.0) <= 0,
        "unrelated_final_unchanged": (upper("all", "unrelated_final_absolute_delta") or 0.0) <= 0,
        "both_endpoints_unchanged": (upper("all", "opposite_panl_absolute_delta") or 0.0) <= 0 and (upper("all", "opposite_final_absolute_delta") or 0.0) <= 0,
    }
    summary = {"status": "complete", "item_count": len(rows), "groups": list(GROUPS), "conclusion_checks": checks, "unrelated_matching": dict(matching)}
    lines = ["# Delayed-SA Answer-force summary", "", f"Items analyzed: {len(rows)}.", "", "## Interpretation", ""]
    if checks["opposite_panl_direction_positive"] and checks["opposite_final_direction_positive"] and checks["panl_specificity_positive"] and checks["final_specificity_positive"]:
        lines.append("Opposite is positively directional for both PANL and final, and both specificity contrasts are positive: this supports an additional modal-identity effect beyond generic answer replacement.")
    elif checks["opposite_panl_direction_positive"] or checks["opposite_final_direction_positive"]:
        lines.append("Opposite affects at least one directional SA endpoint, but the paired specificity contrast does not have a positive 95% CI lower bound; answer replacement is not more modal-specific than unrelated replacement.")
    elif checks["unrelated_panl_direction_positive"] or checks["unrelated_final_direction_positive"]:
        lines.append("Unrelated answer replacement is directionally positive on at least one endpoint, consistent with general recalibration toward the opposite/midpoint rather than a uniquely modal opposite effect.")
    elif checks["opposite_panl_oriented_positive"] and checks["opposite_final_oriented_positive"]:
        lines.append("Both PANL and final oriented deltas are positive: the result supports answer-content-conditioned internal PANL SA and verbal SA.")
    elif checks["opposite_panl_final_absolute_contrast_positive"] and checks["unrelated_panl_unchanged"] and checks["unrelated_final_unchanged"]:
        lines.append("Opposite changes exceed unrelated absolute changes while unrelated endpoints remain unchanged: this supports a modality-specific answer-conditioned SA update.")
    elif checks["opposite_final_changed"] and not checks["opposite_panl_changed"]:
        lines.append("Final output changes without PANL change: this is more consistent with an immediate post-PANL recomputation or direct SAC readout.")
    elif checks["opposite_panl_changed"] and not checks["opposite_final_changed"]:
        lines.append("PANL changes without final change: PANL contains answer-conditioned information that does not stably reach the final output.")
    elif checks["both_endpoints_unchanged"]:
        lines.append("Neither PANL nor final SA changes: the current result does not support fixed answer content as the main determinant of delayed verbal SA.")
    else:
        lines.append("The current result does not establish that fixed answer content is the main determinant of delayed verbal SA.")
    lines += ["", "Opposite and unrelated conditions are compared to distinguish modality-directed effects from generic answer replacement disturbance.", "Correlation or hard-label changes alone are not interpreted as evidence that PANL fully mediates Answer→SAC."]
    return summary, "\n".join(lines) + "\n"


def analyze(output_root: Path, *, repeats: int = BOOTSTRAP_REPEATS, seed: int = SEED) -> dict[str, Any]:
    output = Path(output_root).resolve()
    rows = item_metrics(load_jsonl(output / "results.jsonl"))
    aggregate, bootstrap = bootstrap_aggregates(rows, repeats=repeats, seed=seed)
    matched_rows = [row for row in rows if int(row["opposite_token_length_difference"]) == 0]
    matched_aggregate, _matched_bootstrap = bootstrap_aggregates(matched_rows, repeats=repeats, seed=seed + 7919) if matched_rows else ([], [])
    regressions = regression_results(rows)
    correlations = correlation_results(rows)
    clean_force_correlations = clean_force_soft_sa_correlations(rows)
    specificity = specificity_results(aggregate, bootstrap)
    hard_proportions = hard_class_directional_proportions(rows)
    _write_csv(output / "item_level_metrics.csv", rows)
    _write_csv(output / "aggregate_metrics.csv", aggregate)
    _write_csv(output / "bootstrap_results.csv", bootstrap)
    _write_csv(output / "regression_results.csv", regressions)
    _write_csv(output / "correlations.csv", correlations)
    _write_csv(output / "clean_force_soft_sa_correlations.csv", clean_force_correlations)
    _write_csv(output / "specificity_metrics.csv", specificity)
    _write_csv(output / "hard_class_directional_proportions.csv", hard_proportions)
    _write_csv(output / "token_matched_aggregate_metrics.csv", matched_aggregate)
    matching = json.loads((output / "unrelated_matching_diagnostics.json").read_text(encoding="utf-8")) if (output / "unrelated_matching_diagnostics.json").is_file() else {}
    summary, markdown = _summary(rows, aggregate, matching)
    summary["token_length_sensitivity"] = {
        "clean_opposite_token_equal_count": len(matched_rows),
        "total_item_count": len(rows),
        "subset_definition": "opposite_answer_token_length == clean_answer_token_length",
    }
    summary["clean_force_soft_sa_correlations"] = clean_force_correlations
    summary["specificity_metrics"] = specificity
    summary["hard_class_directional_proportions"] = hard_proportions
    atomic_json(output / "summary.json", summary)
    _atomic_text(output / "summary.md", markdown)
    _plot_grouped(output, aggregate, ("opposite_panl_signed_delta", "opposite_final_signed_delta"), "panl_final_signed_delta.png", "Signed delta")
    _plot_grouped(output, aggregate, ("opposite_panl_absolute_delta", "opposite_final_absolute_delta"), "panl_final_absolute_delta.png", "Absolute delta")
    _plot_grouped(output, aggregate, ("opposite_panl_oriented_delta", "opposite_final_oriented_delta"), "opposite_directional_effect.png", "Opposite directional effect")
    _plot_grouped(output, aggregate, ("opposite_final_hard_change_rate", "opposite_final_toward_force_rate"), "hard_label_change_rate.png", "Final hard-label change")
    _plot_scatter(output, rows)
    _plot_original_forced(output, rows)
    _plot_absolute_delta_overall(output, aggregate)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze delayed-SA Answer-force results")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPEATS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    analyze(args.output_root, repeats=args.bootstrap, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
