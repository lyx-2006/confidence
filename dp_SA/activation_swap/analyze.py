from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import BOOTSTRAP_REPEATS, DEFAULT_POSITIONS, SMOKE_BOOTSTRAP_REPEATS
from .metrics import (
    bh_fdr, bootstrap_values, condition_summary, paired_effect, paired_rows, sign_flip_p,
    stratified_effect_summary,
)
from .utils import atomic_json, canonical_hash, load_jsonl, stable_seed


METRICS = {
    "soft_sa": "swap_soft_sa",
    "hard_midpoint": "swap_hard_midpoint",
    "fixed_clean_class_margin": "swap_fixed_clean_class_margin",
    "first_token_change": "first_token_changed",
}
TARGET_LAYERS = {14, 16}


def _atomic_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
    temporary.replace(path)


def _stats_for_group(pairs: Sequence[dict[str, Any]], group: str, *, repeats: int, seed: int) -> dict[str, Any]:
    subset = pairs if group == "all" else [row for row in pairs if row["recipient_side"] == group]
    if not subset:
        raise ValueError(f"empty analysis group {group}")
    if group == "all":
        return stratified_effect_summary(subset, repeats=repeats, seed=seed)
    return bootstrap_values([float(row["effect"]) for row in subset], repeats=repeats, seed=seed)


def _condition_stats(rows: Sequence[dict[str, Any]], metric: str, *, repeats: int, seed: int) -> list[dict[str, Any]]:
    output = []
    for condition in ("I_from_I", "I_from_T", "T_from_T", "T_from_I"):
        subset = [row for row in rows if row["condition"] == condition]
        if not subset:
            continue
        stats = bootstrap_values([float(row[metric]) for row in subset], repeats=repeats,
                                 seed=stable_seed(seed, "condition", condition, metric))
        output.append({"analysis": "condition", "condition": condition, "metric": metric, **stats})
    return output


def _lodo_effect(rows: Sequence[dict[str, Any]], group: str) -> float | None:
    if group == "image_side":
        values = [float(row["effect"]) for row in rows if row["recipient_side"] == "image_side"]
        return float(np.mean(values)) if values else None
    if group == "text_side":
        values = [float(row["effect"]) for row in rows if row["recipient_side"] == "text_side"]
        return float(np.mean(values)) if values else None
    image = [float(row["effect"]) for row in rows if row["recipient_side"] == "image_side"]
    text = [float(row["effect"]) for row in rows if row["recipient_side"] == "text_side"]
    return float(0.5 * (np.mean(image) + np.mean(text))) if image and text else None


def _contrast_rows(grouped: dict[tuple[str, int], list[dict[str, Any]]], *, position: str, control: str,
                   layer: int, metric: str) -> list[dict[str, Any]]:
    left = {(str(row["recipient_case_id"]), row["swap_kind"]): row for row in grouped.get((position, layer), [])}
    right = {(str(row["recipient_case_id"]), row["swap_kind"]): row for row in grouped.get((control, layer), [])}
    pairs = []
    for key in sorted(set(left) & set(right)):
        l, r = left[key], right[key]
        pairs.append({"recipient_case_id": key[0], "swap_kind": key[1], "recipient_side": l["recipient_side"],
                      "left": float(l[metric]), "right": float(r[metric])})
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in pairs:
        by_case[row["recipient_case_id"]][row["swap_kind"]] = row
    output = []
    for case_id, pair in sorted(by_case.items()):
        if set(pair) != {"same", "cross"}:
            continue
        # The effect already carries the recipient-side orientation; subtracting
        # the control effect therefore preserves the direction convention.
        if metric == "first_token_changed":
            left_effect = pair["cross"]["left"] - pair["same"]["left"]
            right_effect = pair["cross"]["right"] - pair["same"]["right"]
        else:
            left_effect = pair["same"]["left"] - pair["cross"]["left"]
            right_effect = pair["same"]["right"] - pair["cross"]["right"]
        if pair["same"]["recipient_side"] == "text_side" and metric not in {
            "swap_fixed_clean_class_margin", "first_token_changed"
        }:
            left_effect = -left_effect
            right_effect = -right_effect
        output.append({"recipient_case_id": case_id, "recipient_side": pair["same"]["recipient_side"],
                       "effect": float(left_effect - right_effect), "left_effect": float(left_effect),
                       "right_effect": float(right_effect)})
    return output


def _save_diagnostics(output: Path, rows: Sequence[dict[str, Any]]) -> None:
    diagnostics = []
    for row in rows:
        diag = row.get("activation_diagnostics", {})
        diagnostics.append({
            "trial_key": row["trial_key"], "recipient_case_id": row["recipient_case_id"],
            "donor_case_id": row["donor_case_id"], "recipient_side": row["recipient_side"],
            "condition": row["condition"], "swap_kind": row["swap_kind"],
            "position": row["position"], "layer": row["layer"],
            "recipient_norm": diag.get("recipient_norm"), "donor_norm": diag.get("donor_norm"),
            "norm_ratio": diag.get("norm_ratio"), "abs_log_norm_ratio": diag.get("abs_log_norm_ratio"),
            "cosine_distance": diag.get("cosine_distance"), "target_exact_after_cast": diag.get("target_exact_after_cast"),
            "hook_count": diag.get("hook_count"), "applied_count": diag.get("applied_count"),
            "anomaly_warning": bool(diag.get("norm_ratio", 1.0) < 0.5 or diag.get("norm_ratio", 1.0) > 2.0 or
                                    diag.get("cosine_distance", 0.0) > 0.1),
        })
    _atomic_csv(output / "activation_diagnostics.csv", diagnostics)


def _activation_comparisons(
    grouped: dict[tuple[str, int], list[dict[str, Any]]], *, repeats: int, seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Matched activation-drift comparisons and cross-only PANL warnings."""
    output: list[dict[str, Any]] = []
    layers = sorted({layer for _position, layer in grouped})
    for layer in layers:
        indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
        for position in ("P1_PANL", "P1_PANL_PLUS_1"):
            for row in grouped.get((position, layer), []):
                indexed[(str(row["recipient_case_id"]), str(row["swap_kind"]), position)] = row
        case_ids = sorted({key[0] for key in indexed})
        for metric in ("norm_ratio", "abs_log_norm_ratio", "cosine_distance"):
            comparisons: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for case_id in case_ids:
                keys = {(kind, position): indexed.get((case_id, kind, position))
                        for kind in ("same", "cross")
                        for position in ("P1_PANL", "P1_PANL_PLUS_1")}
                if any(row is None for row in keys.values()):
                    continue
                values = {(kind, position): float(keys[(kind, position)]["activation_diagnostics"][metric])
                          for kind, position in keys}
                side = str(keys[("same", "P1_PANL")]["recipient_side"])
                effects = {
                    "PANL_cross_minus_same": values[("cross", "P1_PANL")] - values[("same", "P1_PANL")],
                    "PANL_PLUS_1_cross_minus_same": values[("cross", "P1_PANL_PLUS_1")] - values[("same", "P1_PANL_PLUS_1")],
                    "cross_PANL_minus_PANL_PLUS_1": values[("cross", "P1_PANL")] - values[("cross", "P1_PANL_PLUS_1")],
                    "same_PANL_minus_PANL_PLUS_1": values[("same", "P1_PANL")] - values[("same", "P1_PANL_PLUS_1")],
                    "cross_only_PANL_interaction": (
                        values[("cross", "P1_PANL")] - values[("same", "P1_PANL")]
                        - values[("cross", "P1_PANL_PLUS_1")] + values[("same", "P1_PANL_PLUS_1")]
                    ),
                }
                for comparison, effect in effects.items():
                    comparisons[comparison].append({"recipient_case_id": case_id, "recipient_side": side,
                                                    "effect": float(effect)})
            for comparison, rows in comparisons.items():
                for group in ("all", "image_side", "text_side"):
                    subset = rows if group == "all" else [row for row in rows if row["recipient_side"] == group]
                    if not subset:
                        continue
                    stats = (stratified_effect_summary(
                                subset, repeats=repeats,
                                seed=stable_seed(seed, "activation", str(layer), metric, comparison, group))
                             if group == "all" else
                             bootstrap_values(
                                [row["effect"] for row in subset], repeats=repeats,
                                seed=stable_seed(seed, "activation", str(layer), metric, comparison, group)))
                    output.append({"analysis": "activation_matched_comparison", "layer": layer,
                                   "diagnostic_metric": metric, "comparison": comparison, "group": group,
                                   "p_raw": (sign_flip_p(
                                       [row["effect"] for row in subset], repeats=repeats,
                                       seed=stable_seed(seed, "activation-p", str(layer), metric, comparison))
                                       if group == "all" else None), **stats})

    lookup = {(int(row["layer"]), row["diagnostic_metric"], row["comparison"], row["group"]): row
              for row in output}
    warnings: list[dict[str, Any]] = []
    for layer in layers:
        for metric in ("abs_log_norm_ratio", "cosine_distance"):
            cross = lookup.get((layer, metric, "cross_PANL_minus_PANL_PLUS_1", "all"))
            same = lookup.get((layer, metric, "same_PANL_minus_PANL_PLUS_1", "all"))
            interaction = lookup.get((layer, metric, "cross_only_PANL_interaction", "all"))
            warning = bool(cross and same and interaction and cross["ci_low"] > 0 and
                           same["ci_low"] <= 0 and interaction["ci_low"] > 0)
            warnings.append({"layer": layer, "diagnostic_metric": metric,
                             "cross_only_panl_drift_warning": warning,
                             "cross_panl_control_ci_low": None if cross is None else cross["ci_low"],
                             "same_panl_control_ci_low": None if same is None else same["ci_low"],
                             "interaction_ci_low": None if interaction is None else interaction["ci_low"]})
    return output, warnings


def _plot(output: Path, condition_rows: Sequence[dict[str, Any]], metrics: Sequence[dict[str, Any]], contrasts: Sequence[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
        "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    positions = tuple(DEFAULT_POSITIONS)
    position_titles = {"P1_PANL": "PANL", "P1_PANL_PLUS_1": "PANL + 1", "P1_SAC": "SAC"}
    condition_style = {
        "I_from_I": ("#0072B2", "o", "Image ← Image (same)"),
        "I_from_T": ("#D55E00", "s", "Image ← Text (cross)"),
        "T_from_I": ("#CC79A7", "D", "Text ← Image (cross)"),
        "T_from_T": ("#009E73", "^", "Text ← Text (same)"),
    }
    offsets = {"I_from_I": -0.30, "I_from_T": -0.10, "T_from_T": 0.10, "T_from_I": 0.30}

    def errorbar(ax: Any, part: Sequence[dict[str, Any]], *, color: str, marker: str = "o", label: str | None = None,
                 x_offset: float = 0.0) -> None:
        ordered = sorted(part, key=lambda row: int(row["layer"]))
        if not ordered:
            return
        x = np.asarray([float(row["layer"]) + x_offset for row in ordered])
        y = np.asarray([float(row["mean"]) for row in ordered])
        low = np.asarray([float(row["ci_low"]) for row in ordered])
        high = np.asarray([float(row["ci_high"]) for row in ordered])
        ax.errorbar(x, y, yerr=np.vstack([y - low, high - y]), fmt=marker, linestyle="none",
                    color=color, markeredgecolor="white", markeredgewidth=.7, markersize=6,
                    elinewidth=1.35, capsize=2.5, capthick=1.1, label=label, zorder=3)

    def target_layers(ax: Any) -> None:
        for layer in (14, 16):
            ax.axvspan(layer - .42, layer + .42, color="#F0E442", alpha=.12, linewidth=0, zorder=0)

    # Raw factorial condition means.  Only the soft-SA condition summaries
    # belong in this figure; other metrics use different units and scales.
    soft_condition_rows = [row for row in condition_rows if row.get("metric") == "swap_soft_sa"]
    soft_low = min(float(row["ci_low"]) for row in soft_condition_rows)
    soft_high = max(float(row["ci_high"]) for row in soft_condition_rows)
    soft_padding = max(.02, .08 * (soft_high - soft_low))
    soft_ylim = (max(0.0, soft_low - soft_padding), min(1.0, soft_high + soft_padding))
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5), sharey=True, constrained_layout=True)
    for ax, position in zip(axes, positions):
        target_layers(ax)
        for condition, (color, marker, label) in condition_style.items():
            part = [row for row in condition_rows if row["position"] == position and
                    row["condition"] == condition and row.get("metric") == "swap_soft_sa"]
            errorbar(ax, part, color=color, marker=marker, label=label, x_offset=offsets[condition])
        ax.set_title(position_titles[position]); ax.set_xlabel("Decoder layer")
        ax.set_xticks(sorted({int(row["layer"]) for row in condition_rows if row["position"] == position}))
        ax.set_ylim(*soft_ylim); ax.grid(axis="y", alpha=.25); ax.grid(axis="x", alpha=.10)
    axes[0].set_ylabel("Mean swapped soft SA (k/8)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Factorial activation-swap outcomes", fontsize=14, fontweight="semibold")
    fig.savefig(output / "soft_sa_swap_by_layer.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    # Separate panels keep the small PANL effects visible despite the much
    # larger late-layer SAC effect.  Each panel therefore has its own y scale.
    position_colors = {"P1_PANL": "#D55E00", "P1_PANL_PLUS_1": "#0072B2", "P1_SAC": "#009E73"}
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.2), constrained_layout=True)
    for ax, position in zip(axes, positions):
        target_layers(ax)
        part = [row for row in metrics if row.get("analysis") == "paired_effect" and
                row.get("position") == position and row.get("metric") == "soft_sa" and row.get("group") == "all"]
        errorbar(ax, part, color=position_colors[position])
        ax.axhline(0, color="#333333", lw=.9, zorder=1)
        ax.set_title(position_titles[position]); ax.set_xlabel("Decoder layer")
        ax.set_xticks([int(row["layer"]) for row in sorted(part, key=lambda row: int(row["layer"]))])
        ax.grid(axis="y", alpha=.25); ax.grid(axis="x", alpha=.10)
    axes[0].set_ylabel("Oriented soft-SA effect\n(positive = toward donor side)")
    fig.suptitle("Recipient-paired activation-swap effect", fontsize=14, fontweight="semibold")
    fig.savefig(output / "oriented_swap_effect_by_layer.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    contrast_specs = (
        ("PANL_minus_PANL_PLUS_1", "#D55E00", "PANL − PANL + 1"),
        ("SAC_minus_PANL_PLUS_1", "#009E73", "SAC − PANL + 1"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), constrained_layout=True)
    for ax, (contrast, color, title) in zip(axes, contrast_specs):
        target_layers(ax)
        part = [row for row in contrasts if row["contrast"] == contrast and
                row["metric"] == "soft_sa" and row["group"] == "all"]
        errorbar(ax, part, color=color)
        ax.axhline(0, color="#333333", lw=.9, zorder=1)
        ax.set_title(title); ax.set_xlabel("Decoder layer")
        ax.set_xticks([int(row["layer"]) for row in sorted(part, key=lambda row: int(row["layer"]))])
        ax.grid(axis="y", alpha=.25); ax.grid(axis="x", alpha=.10)
    axes[0].set_ylabel("Matched soft-SA contrast")
    fig.suptitle("Position specificity", fontsize=14, fontweight="semibold")
    fig.savefig(output / "panl_vs_control_contrast.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    for ax, metric, title in zip(axes, ("hard_midpoint", "fixed_clean_class_margin", "first_token_change"),
                                 ("Hard midpoint", "Fixed-clean-class margin", "First-token change rate")):
        target_layers(ax)
        part = [row for row in metrics if row.get("analysis") == "paired_effect" and
                row.get("position") == "P1_PANL" and row.get("metric") == metric and row.get("group") == "all"]
        errorbar(ax, part, color="#D55E00")
        ax.axhline(0, color="#333333", lw=.9, zorder=1)
        ax.set_title(title); ax.set_xlabel("Decoder layer")
        ax.set_xticks([int(row["layer"]) for row in sorted(part, key=lambda row: int(row["layer"]))])
        ax.grid(axis="y", alpha=.25); ax.grid(axis="x", alpha=.10)
    axes[0].set_ylabel("Recipient-paired effect")
    fig.suptitle("Supporting PANL metrics", fontsize=14, fontweight="semibold")
    fig.savefig(output / "supporting_metrics.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    def line_band(ax: Any, part: Sequence[dict[str, Any]], *, color: str, label: str,
                  marker: str = "o", linestyle: str = "-", show_ci: bool = True) -> None:
        ordered = sorted(part, key=lambda row: int(row["layer"]))
        if not ordered:
            return
        x = np.asarray([int(row["layer"]) for row in ordered])
        y = np.asarray([float(row["mean"]) for row in ordered])
        low = np.asarray([float(row["ci_low"]) for row in ordered])
        high = np.asarray([float(row["ci_high"]) for row in ordered])
        if show_ci:
            ax.fill_between(x, low, high, color=color, alpha=.14, linewidth=0, zorder=1)
        ax.plot(x, y, color=color, marker=marker, linestyle=linestyle, linewidth=1.8,
                markersize=5.5, markeredgecolor="white", markeredgewidth=.7,
                label=label, zorder=3)

    short_condition = {
        "I_from_I": "I2I", "I_from_T": "I2T", "T_from_I": "T2I", "T_from_T": "T2T",
    }
    # Condition-specific change from the recipient clean fixed-class margin.
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.2), constrained_layout=True)
    for ax, position in zip(axes, positions):
        target_layers(ax)
        for condition, (color, marker, _long_label) in condition_style.items():
            part = [row for row in metrics if row.get("analysis") == "condition_logit_change" and
                    row.get("position") == position and row.get("condition") == condition]
            line_band(ax, part, color=color, marker=marker, label=short_condition[condition], show_ci=False)
        ax.axhline(0, color="#333333", lw=.9, zorder=2)
        ax.set_title(position_titles[position]); ax.set_xlabel("Decoder layer")
        ax.set_xticks(sorted({int(row["layer"]) for row in metrics if row.get("position") == position}))
        ax.grid(axis="y", alpha=.25); ax.grid(axis="x", alpha=.10)
    axes[0].set_ylabel("Fixed-clean logit-margin change\n(swap − clean)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Logit change by condition and position", fontsize=14, fontweight="semibold")
    fig.savefig(output / "logit_change_diff_by_position.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    # Condition-specific category-change rates.  The common 0–1 scale
    # preserves direct comparability across positions.
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.2), sharey=True, constrained_layout=True)
    for ax, position in zip(axes, positions):
        target_layers(ax)
        for condition, (color, marker, _long_label) in condition_style.items():
            part = [row for row in condition_rows if row.get("metric") == "first_token_changed" and
                    row.get("position") == position and row.get("condition") == condition]
            line_band(ax, part, color=color, marker=marker, label=short_condition[condition], show_ci=False)
        ax.set_title(position_titles[position]); ax.set_xlabel("Decoder layer")
        ax.set_xticks(sorted({int(row["layer"]) for row in condition_rows if row.get("position") == position}))
        ax.set_ylim(-.02, 1.02); ax.grid(axis="y", alpha=.25); ax.grid(axis="x", alpha=.10)
    axes[0].set_ylabel("First-token change rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("First-token change rate by condition and position", fontsize=14, fontweight="semibold")
    fig.savefig(output / "token_change_rate_by_position.png", dpi=220, bbox_inches="tight"); plt.close(fig)


def analyze(output_root: Path, *, repeats: int, seed: int) -> dict[str, Any]:
    output = output_root.resolve()
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    matching = json.loads((output / "matching_diagnostics.json").read_text(encoding="utf-8"))
    swap_rows = load_jsonl(output / "swap_predictions.jsonl")
    expected = int(config["expected_swap_forwards"])
    if len(swap_rows) != expected or len({row["trial_key"] for row in swap_rows}) != expected:
        raise RuntimeError(f"incomplete swap grid: {len(swap_rows)}/{expected}")
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in swap_rows:
        grouped[(str(row["position"]), int(row["layer"]))].append(row)
    metric_rows: list[dict[str, Any]] = []
    paired_cell_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    condition_logit_change_rows: list[dict[str, Any]] = []
    arm_summary_rows: list[dict[str, Any]] = []
    p_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (position, layer), rows in sorted(grouped.items()):
        cases: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for raw in rows:
            cases[str(raw["recipient_case_id"])][str(raw["swap_kind"])] = raw
        for case_id, pair in sorted(cases.items()):
            if set(pair) != {"same", "cross"}:
                raise ValueError(f"recipient {case_id} lacks same/cross pair at {position} L{layer}")
            same, cross = pair["same"], pair["cross"]
            side = str(same["recipient_side"])
            paired = {
                "recipient_case_id": case_id, "recipient_item_id": same["recipient_item_id"],
                "recipient_side": side, "position": position, "layer": layer,
                "same_condition": same["condition"], "cross_condition": cross["condition"],
                "same_donor_case_id": same["donor_case_id"], "cross_donor_case_id": cross["donor_case_id"],
                "same_normalized_answer_match": same.get("matching", {}).get("normalized_answer_match"),
                "cross_normalized_answer_match": cross.get("matching", {}).get("normalized_answer_match"),
                "normalized_answer_pair_match": bool(
                    same.get("matching", {}).get("normalized_answer_match") and
                    cross.get("matching", {}).get("normalized_answer_match")
                ),
            }
            for metric, field in METRICS.items():
                same_value, cross_value = float(same[field]), float(cross[field])
                paired[f"same_{metric}"] = same_value
                paired[f"cross_{metric}"] = cross_value
                paired[f"{metric}_effect"] = paired_effect(same_value, cross_value, side, metric)
            paired_cell_rows.append(paired)
        for condition in ("I_from_I", "I_from_T", "T_from_I", "T_from_T"):
            subset = [raw for raw in rows if raw["condition"] == condition]
            changes = [float(raw["swap_fixed_clean_class_margin"]) -
                       float(raw["clean_fixed_clean_class_margin"]) for raw in subset]
            stats = bootstrap_values(
                changes, repeats=repeats,
                seed=stable_seed(seed, "condition-logit-change", position, str(layer), condition),
            )
            condition_logit_change_rows.append({
                "analysis": "condition_logit_change", "position": position, "layer": layer,
                "condition": condition, "metric": "fixed_clean_class_margin_change", **stats,
            })
        for metric, field in METRICS.items():
            condition_rows.extend({"position": position, "layer": layer, **row}
                                  for row in _condition_stats(rows, field, repeats=repeats,
                                                               seed=stable_seed(seed, "condition", position, str(layer), metric)))
            pairs = paired_rows(rows, field)
            if metric == "first_token_change":
                for arm in ("same", "cross"):
                    arm_rows = [raw for raw in rows if raw["swap_kind"] == arm]
                    for group in ("all", "image_side", "text_side"):
                        subset = arm_rows if group == "all" else [raw for raw in arm_rows if raw["recipient_side"] == group]
                        if group == "all":
                            stats = stratified_effect_summary(
                                [{"recipient_side": raw["recipient_side"], "effect": float(raw[field])} for raw in subset],
                                repeats=repeats,
                                seed=stable_seed(seed, "arm", position, str(layer), metric, arm, group))
                        else:
                            stats = bootstrap_values(
                                [float(raw[field]) for raw in subset], repeats=repeats,
                                seed=stable_seed(seed, "arm", position, str(layer), metric, arm, group))
                        arm_summary_rows.append({"analysis": "swap_arm_rate", "position": position,
                                                 "layer": layer, "metric": metric, "arm": arm,
                                                 "group": group, **stats})
            for group in ("all", "image_side", "text_side"):
                subset = pairs if group == "all" else [item for item in pairs if item["recipient_side"] == group]
                stats = _stats_for_group(pairs, group, repeats=repeats,
                                         seed=stable_seed(seed, "effect", position, str(layer), metric, group))
                p_value = sign_flip_p([float(item["effect"]) for item in subset], repeats=repeats,
                                      seed=stable_seed(seed, "signflip", position, str(layer), metric, group)) if group == "all" else None
                row = {"analysis": "paired_effect", "position": position, "layer": layer,
                       "metric": metric, "group": group, "p_raw": p_value,
                       "targeted_confirmatory": bool(position == "P1_PANL" and layer in TARGET_LAYERS), **stats}
                metric_rows.append(row)
                if group == "all":
                    p_candidates[metric].append(row)
    for metric, candidates in p_candidates.items():
        exploratory = [row for row in candidates if not row["targeted_confirmatory"]]
        q_values = bh_fdr([float(row["p_raw"]) for row in exploratory]) if exploratory else []
        for row, q in zip(exploratory, q_values):
            row["q_bh"] = q
            row["fdr_family"] = f"{metric}:non_target_position_layer_cells"
        for row in candidates:
            row.setdefault("q_bh", None); row.setdefault("fdr_family", None)
    contrast_rows: list[dict[str, Any]] = []
    for layer in sorted({layer for _position, layer in grouped}):
        for metric, field in METRICS.items():
            for control, contrast_name in (("P1_PANL_PLUS_1", "PANL_minus_PANL_PLUS_1"), ("P1_PANL_PLUS_1", "SAC_minus_PANL_PLUS_1")):
                position = "P1_PANL" if contrast_name.startswith("PANL") else "P1_SAC"
                raw_pairs = _contrast_rows(grouped, position=position, control=control, layer=layer, metric=field)
                if not raw_pairs:
                    continue
                for group in ("all", "image_side", "text_side"):
                    subset = raw_pairs if group == "all" else [row for row in raw_pairs if row["recipient_side"] == group]
                    stats = stratified_effect_summary(subset, repeats=repeats,
                                                      seed=stable_seed(seed, "contrast", contrast_name, str(layer), metric, group)) if group == "all" else bootstrap_values([row["effect"] for row in subset], repeats=repeats, seed=stable_seed(seed, "contrast", contrast_name, str(layer), metric, group))
                    contrast_rows.append({"analysis": "position_contrast", "contrast": contrast_name, "layer": layer,
                                          "metric": metric, "group": group, "p_raw": sign_flip_p([row["effect"] for row in subset], repeats=repeats, seed=stable_seed(seed, "contrast-p", contrast_name, str(layer), metric)) if group == "all" else None,
                                          "targeted_confirmatory": bool(layer in TARGET_LAYERS and contrast_name == "PANL_minus_PANL_PLUS_1"), **stats})
    _atomic_csv(output / "swap_metrics.csv", paired_cell_rows)
    _atomic_csv(output / "bootstrap_results.csv", [*metric_rows, *condition_rows, *arm_summary_rows])
    _atomic_csv(output / "position_contrasts.csv", contrast_rows)
    _atomic_csv(output / "logit_change_diff_by_position.csv", condition_logit_change_rows)
    _atomic_csv(output / "token_change_rate_by_position.csv", [
        row for row in condition_rows if row.get("metric") == "first_token_changed"
    ])
    _save_diagnostics(output, swap_rows)
    activation_comparisons, activation_drift_checks = _activation_comparisons(
        grouped, repeats=repeats, seed=seed
    )
    _atomic_csv(output / "activation_diagnostic_comparisons.csv", activation_comparisons)

    # Sensitivity analysis is restricted to complete recipient pairs for which
    # both the same- and cross-side donors have the same normalized answer as
    # the recipient.  This is intentionally a filter on the frozen mapping,
    # never a re-matching step.
    normalized_rows: list[dict[str, Any]] = []
    for (position, layer), rows in sorted(grouped.items()):
        for metric, field in METRICS.items():
            pairs = paired_rows(rows, field)
            matched = [pair for pair in pairs
                       if bool(pair["same_case"].get("matching", {}).get("normalized_answer_match"))
                       and bool(pair["cross_case"].get("matching", {}).get("normalized_answer_match"))]
            for group in ("all", "image_side", "text_side"):
                subset = matched if group == "all" else [item for item in matched if item["recipient_side"] == group]
                if not subset:
                    continue
                stats = (stratified_effect_summary(subset, repeats=repeats,
                                                    seed=stable_seed(seed, "normalized", position, str(layer), metric, group))
                         if group == "all" else
                         bootstrap_values([float(item["effect"]) for item in subset], repeats=repeats,
                                          seed=stable_seed(seed, "normalized", position, str(layer), metric, group)))
                normalized_rows.append({"analysis": "normalized_answer_matched", "position": position,
                                        "layer": layer, "metric": metric, "group": group,
                                        "sample_count": len(subset), **stats})
    _atomic_csv(output / "normalized_answer_sensitivity.csv", normalized_rows)

    donor_sensitivity = []
    target_pairs_by_cell = {(position, layer): paired_rows(rows, METRICS["soft_sa"])
                            for (position, layer), rows in grouped.items() if position == "P1_PANL" and layer in TARGET_LAYERS}
    donor_ids = sorted({str(row["donor_case_id"]) for row in swap_rows})
    for (position, layer), pairs in sorted(target_pairs_by_cell.items()):
        for donor_id in donor_ids:
            keep = [row for row in pairs if row["same_case"]["donor_case_id"] != donor_id and row["cross_case"]["donor_case_id"] != donor_id]
            for group in ("all", "image_side", "text_side"):
                full = _lodo_effect(pairs, group)
                leave = _lodo_effect(keep, group)
                remaining = len(keep) if group == "all" else sum(row["recipient_side"] == group for row in keep)
                donor_sensitivity.append({"position": position, "layer": layer, "group": group,
                                          "donor_case_id": donor_id, "full_effect": full,
                                          "leave_one_donor_effect": leave,
                                          "direction_reversed": bool(full is not None and leave is not None and full * leave < 0),
                                          "remaining_recipient_count": remaining})
    _atomic_csv(output / "donor_sensitivity.csv", donor_sensitivity)

    diagnostics = load_jsonl(output / "swap_predictions.jsonl")
    def _diagnostic_warning(row: dict[str, Any]) -> bool:
        diag = row.get("activation_diagnostics", {})
        values = [diag.get("recipient_norm"), diag.get("donor_norm"), diag.get("norm_ratio"),
                  diag.get("abs_log_norm_ratio"), diag.get("cosine_distance")]
        if any(value is None or not np.isfinite(float(value)) for value in values):
            return True
        return bool(float(diag["recipient_norm"]) <= 0 or float(diag["donor_norm"]) <= 0 or
                    float(diag["norm_ratio"]) < 0.5 or float(diag["norm_ratio"]) > 2.0 or
                    float(diag["cosine_distance"]) > 0.1 or
                    not bool(diag.get("target_exact_after_cast")))

    anomaly_by_layer = {
        int(layer): sum(_diagnostic_warning(row) for row in diagnostics
                        if row["position"] == "P1_PANL" and int(row["layer"]) == int(layer))
        for layer in {int(row["layer"]) for row in diagnostics}
    }
    anomaly_count = int(sum(anomaly_by_layer.values()))
    drift_warning_layers = {int(row["layer"]) for row in activation_drift_checks
                            if row["cross_only_panl_drift_warning"]}
    lookup = {(row["position"], int(row["layer"]), row["group"], row["metric"]): row for row in metric_rows}
    contrast_lookup = {(row["contrast"], int(row["layer"]), row["group"], row["metric"]): row for row in contrast_rows}
    target_checks = []
    for layer in sorted(TARGET_LAYERS & {int(row["layer"]) for row in metric_rows}):
        soft = lookup.get(("P1_PANL", layer, "all", "soft_sa")); contrast = contrast_lookup.get(("PANL_minus_PANL_PLUS_1", layer, "all", "soft_sa")); margin = lookup.get(("P1_PANL", layer, "all", "fixed_clean_class_margin"))
        image = lookup.get(("P1_PANL", layer, "image_side", "soft_sa")); text = lookup.get(("P1_PANL", layer, "text_side", "soft_sa"))
        reversal = any(row["position"] == "P1_PANL" and int(row["layer"]) == layer and row["direction_reversed"] for row in donor_sensitivity)
        checks = {"layer": layer, "combined_soft_ci_above_zero": bool(soft and soft["ci_low"] > 0),
                  "image_direction_mean_positive": bool(image and image["mean"] > 0),
                  "text_direction_mean_positive": bool(text and text["mean"] > 0),
                  "panl_control_ci_above_zero": bool(contrast and contrast["ci_low"] > 0),
                  "margin_mean_positive": bool(margin and margin["mean"] > 0),
                  "donor_direction_reversal": reversal,
                  "activation_warning": bool(anomaly_by_layer.get(layer, 0) > 0 or layer in drift_warning_layers)}
        checks["supported"] = bool(checks["combined_soft_ci_above_zero"] and checks["image_direction_mean_positive"] and checks["text_direction_mean_positive"] and checks["panl_control_ci_above_zero"] and checks["margin_mean_positive"] and not checks["donor_direction_reversal"] and not checks["activation_warning"])
        target_checks.append(checks)
    supported = any(row["supported"] for row in target_checks)
    target_layers = sorted(TARGET_LAYERS & {int(row["layer"]) for row in metric_rows})
    panl_positive = any((lookup.get(("P1_PANL", layer, "all", "soft_sa")) or {}).get("ci_low", -math.inf) > 0
                        for layer in target_layers)
    sac_positive = any((lookup.get(("P1_SAC", layer, "all", "soft_sa")) or {}).get("ci_low", -math.inf) > 0
                       for layer in target_layers)
    panl_specific = any((contrast_lookup.get(("PANL_minus_PANL_PLUS_1", layer, "all", "soft_sa")) or {}).get("ci_low", -math.inf) > 0
                        for layer in target_layers)
    if supported:
        interpretation = "PANL transfer supported at a targeted layer under all prespecified gates."
    elif sac_positive and not panl_positive:
        interpretation = ("Only the downstream SAC control is supported; final output state may transfer, "
                          "but early PANL caching is not established.")
    elif panl_positive and not panl_specific:
        interpretation = ("PANL is not reliably stronger than PANL+1; the result is compatible with a general "
                          "local content-replacement effect, not PANL-specific transfer.")
    else:
        interpretation = "PANL transfer is not established under the prespecified success criteria."
    summary = {
        "status": "complete", "run_fingerprint": config["fingerprint"], "bootstrap_repeats": repeats,
        "sampling_unit": "recipient_item_level_paired", "recipient_count": config["recipient_count"],
        "donor_count": config["donor_count"], "swap_forward_count": len(swap_rows),
        "expected_clean_forward_count": config["expected_clean_forwards"],
        "expected_swap_forward_count": config["expected_swap_forwards"],
        "matching_diagnostics": matching,
        "normalized_answer_sensitivity_rows": len(normalized_rows),
        "paired_metric_row_count": len(paired_cell_rows),
        "target_checks": target_checks, "panl_transfer_supported": supported,
        "interpretation": interpretation,
        "activation_diagnostic_warning_count": int(anomaly_count),
        "activation_drift_checks": activation_drift_checks,
        "activation_drift_warning_layers": sorted(drift_warning_layers),
        "donor_direction_reversal_count": int(sum(row["direction_reversed"] for row in donor_sensitivity)),
        "outputs": ["run_config.json", "input_fingerprints.json", "recipient_manifest.jsonl", "donor_manifest.jsonl",
                    "swap_pair_manifest.jsonl", "matching_diagnostics.json", "clean_predictions.jsonl", "swap_predictions.jsonl",
                    "progress.json", "failures.jsonl", "swap_metrics.csv", "position_contrasts.csv", "bootstrap_results.csv",
                    "normalized_answer_sensitivity.csv", "donor_sensitivity.csv", "activation_diagnostics.csv",
                    "activation_diagnostic_comparisons.csv", "summary.json", "summary.md",
                    "soft_sa_swap_by_layer.png", "oriented_swap_effect_by_layer.png",
                    "panl_vs_control_contrast.png", "supporting_metrics.png",
                    "logit_change_diff_by_position.csv", "token_change_rate_by_position.csv",
                    "logit_change_diff_by_position.png", "token_change_rate_by_position.png"],
    }
    atomic_json(output / "summary.json", summary)
    lines = ["# Delayed-SA activation swap", "", f"Status: {summary['status']}",
             f"Recipients: {config['recipient_count']}; donors: {config['donor_count']}; swap forwards: {len(swap_rows)}.",
             f"Expected clean forwards: {config['expected_clean_forwards']}; expected swap forwards: {config['expected_swap_forwards']}.",
             f"Length bins: question={matching['effective_question_bins']}, answer={matching['effective_answer_bins']}; "
             f"donor reuse range={matching['min_donor_reuse']}–{matching['max_donor_reuse']}; "
             f"normalized-answer match={matching['normalized_answer_match_rate']:.3f}.",
             f"Normalized-answer matched sensitivity rows: {len(normalized_rows)}.", "",
             "## Targeted PANL checks", ""]
    for row in target_checks:
        lines.append(f"- L{row['layer']}: {'SUPPORTED' if row['supported'] else 'not established'}; "
                     f"soft CI>0={row['combined_soft_ci_above_zero']}, PANL-control CI>0={row['panl_control_ci_above_zero']}, "
                     f"image/text direction=({row['image_direction_mean_positive']}/{row['text_direction_mean_positive']}), "
                     f"activation warning={row['activation_warning']}.")
    if drift_warning_layers:
        lines.extend(["", f"Activation drift warning layers: {sorted(drift_warning_layers)}."])
    lines.extend(["", f"Interpretation: {interpretation}", "",
                  "Scientific null effects are valid outcomes and do not indicate execution failure.", ""])
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    _plot(output, condition_rows, [*metric_rows, *arm_summary_rows, *condition_logit_change_rows], contrast_rows)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze delayed-SA activation swap")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPEATS)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    analyze(args.output_root, repeats=int(args.bootstrap), seed=int(args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
