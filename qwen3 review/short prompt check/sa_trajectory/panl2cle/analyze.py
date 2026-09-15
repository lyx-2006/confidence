from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from attention_block.run import _atomic_csv

from .config import ALPHAS, BOOTSTRAP_REPEATS, GROUPS, PAIRS, SEED
from .contracts import atomic_json, atomic_jsonl, expected_logical_count, load_jsonl


def expand_logical(trials: Sequence[dict[str, Any]]):
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trials:
        by_case[str(row["case_id"])].append(row)
    output = []
    for case, rows in by_case.items():
        c0 = next(r for r in rows if r["condition"] == "C0")
        lookup = {
            (r["condition"], r.get("panl_layer"), r.get("cle_layer"), float(r["alpha"])): r
            for r in rows
        }
        for panl, cle in PAIRS:
            clean_probe = float(c0["cle_probe"][str(cle)]["predicted_soft_sa"])
            for alpha in ALPHAS:
                cells = (
                    ("C0", c0),
                    ("C1", lookup[("C1", panl, cle, float(alpha))]),
                    ("C2", lookup[("C2", panl, cle, float(alpha))]),
                    ("C3", lookup[("C3", panl, cle, float(alpha))]),
                )
                for condition, row in cells:
                    probe = float(row["cle_probe"][str(cle)]["predicted_soft_sa"])
                    output.append({
                        "case_id": case, "item_id": row["item_id"], "answer": row["answer"],
                        "test_side": row["test_side"], "condition": condition,
                        "panl_layer": panl, "cle_layer": cle, "alpha": float(alpha),
                        "final_soft_sa": float(row["final_soft_sa"]),
                        "delta_final_soft_sa": float(row["final_soft_sa"]) - float(c0["final_soft_sa"]),
                        "cle_probe_sa": probe, "delta_cle_probe_sa": probe - clean_probe,
                        "class_logits": row["class_logits"], "class_probabilities": row["class_probabilities"],
                        "hard_sa_class": row["hard_sa_class"],
                        "hard_change": int(row["hard_sa_class"] != c0["hard_sa_class"]),
                        "positions": row["positions"], "hook": row["hook"],
                    })
    return output


def effect_values(rows: Sequence[dict[str, Any]], field: str):
    values = {r["condition"]: float(r[field]) for r in rows}
    c0, c1, c2, c3 = (values[x] for x in ("C0", "C1", "C2", "C3"))
    return {
        "total": c1 - c0, "residual": c2 - c0, "attenuation": c1 - c2,
        "transfer": c3 - c0, "interaction": c1 - c2 - c3 + c0,
    }


def _select(rows: Sequence[dict[str, Any]], group: str):
    return [r for r in rows if group not in ("image_side", "text_side") or r["test_side"] == group]


def _aggregate(rows: Sequence[dict[str, Any]], value: Callable[[dict[str, Any]], float], group: str):
    selected = _select(rows, group)
    if not selected:
        return math.nan
    if group == "answer_equal_macro":
        answers = sorted({r["answer"] for r in selected})
        return float(np.mean([np.mean([value(r) for r in selected if r["answer"] == answer]) for answer in answers]))
    return float(np.mean([value(r) for r in selected]))


def _bootstrap(
    rows: Sequence[dict[str, Any]], value: Callable[[dict[str, Any]], float],
    group: str, repeats: int, seed: int,
):
    point = _aggregate(rows, value, group)
    rng = np.random.default_rng(seed)
    values = []
    selected = _select(rows, group)
    if group == "answer_equal_macro":
        grouped = {answer: [r for r in selected if r["answer"] == answer] for answer in sorted({r["answer"] for r in selected})}
        for _ in range(repeats):
            values.append(float(np.mean([
                np.mean([value(x) for x in rng.choice(pool, len(pool), replace=True)])
                for pool in grouped.values()
            ])))
    else:
        for _ in range(repeats):
            values.append(float(np.mean([value(x) for x in rng.choice(selected, len(selected), replace=True)])))
    low, high = np.quantile(values, [.025, .975])
    return point, float(low), float(high)


def analyze(*, output_root: Path, repeats: int = BOOTSTRAP_REPEATS):
    root = Path(output_root)
    trials = load_jsonl(root / "artifacts/trials.jsonl")
    logical = expand_logical(trials)
    case_count = len({r["case_id"] for r in trials})
    if len(logical) != expected_logical_count(case_count):
        raise RuntimeError(f"Logical four-cell grid incomplete: {len(logical)}")
    keys = {(r["case_id"], r["condition"], r["panl_layer"], r["cle_layer"], r["alpha"]) for r in logical}
    if len(keys) != len(logical):
        raise RuntimeError("Duplicate short logical four-cell rows")
    condition_rows = []
    effect_rows = []
    counter = 0
    endpoints = (("final_soft_sa", "delta_final_soft_sa"), ("cle_probe_sa", "delta_cle_probe_sa"))
    for panl, cle in PAIRS:
        for alpha in ALPHAS:
            cell = [r for r in logical if r["panl_layer"] == panl and r["cle_layer"] == cle and r["alpha"] == alpha]
            per_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in cell:
                per_case[row["case_id"]].append(row)
            for endpoint, delta_field in endpoints:
                for condition in ("C0", "C1", "C2", "C3"):
                    selected = [r for r in cell if r["condition"] == condition]
                    for group in GROUPS:
                        absolute = _bootstrap(selected, lambda r, f=endpoint: float(r[f]), group, repeats, SEED + counter)
                        delta = _bootstrap(selected, lambda r, f=delta_field: float(r[f]), group, repeats, SEED + 10000 + counter)
                        condition_rows.append({
                            "endpoint": endpoint, "condition": condition, "panl_layer": panl,
                            "cle_layer": cle, "alpha": alpha, "group": group,
                            "mean_value": absolute[0], "value_ci95_low": absolute[1], "value_ci95_high": absolute[2],
                            "mean_delta": delta[0], "delta_ci95_low": delta[1], "delta_ci95_high": delta[2],
                            "case_count": len(selected), "bootstrap_repeats": repeats,
                        })
                        counter += 1
                effects_by_case = []
                for records in per_case.values():
                    effects_by_case.append({**records[0], **effect_values(records, endpoint)})
                for effect in ("total", "residual", "attenuation", "transfer", "interaction"):
                    for group in GROUPS:
                        result = _bootstrap(
                            effects_by_case, lambda r, e=effect: float(r[e]),
                            group, repeats, SEED + 20000 + counter,
                        )
                        effect_rows.append({
                            "endpoint": endpoint, "effect": effect, "panl_layer": panl,
                            "cle_layer": cle, "alpha": alpha, "group": group,
                            "mean": result[0], "ci95_low": result[1], "ci95_high": result[2],
                            "case_count": len(effects_by_case), "bootstrap_repeats": repeats,
                        })
                        counter += 1
    atomic_jsonl(root / "artifacts/logical_four_cell.jsonl", logical)
    _atomic_csv(root / "tables/condition_summary.csv", condition_rows)
    _atomic_csv(root / "tables/effect_summary.csv", effect_rows)
    _plots(root, condition_rows, effect_rows)
    result = {
        "status": "complete", "physical_trial_count": len(trials),
        "logical_row_count": len(logical), "case_count": case_count,
        "bootstrap_repeats": repeats, "condition_cells": len(condition_rows),
        "effect_cells": len(effect_rows),
    }
    atomic_json(root / "analysis_summary.json", result)
    return result


def _plots(root: Path, condition_rows: Sequence[dict[str, Any]], effect_rows: Sequence[dict[str, Any]]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    # Human-readable condition names are used in figures; keep the compact
    # C0--C3 codes only in machine-readable tables/manifests.
    condition_labels = {
        "C0": "Clean run",
        "C1": "Steered PANL + natural CLE",
        "C2": "Steered PANL + clean-restored CLE",
        "C3": "Clean PANL + steered CLE",
    }
    condition_styles = {
        "C1": {"color": "#0072B2", "marker": "o"},
        "C2": {"color": "#E69F00", "marker": "s"},
        "C3": {"color": "#009E73", "marker": "D"},
    }
    # Slight horizontal dodging keeps coincident C1/C2/C3 estimates visible.
    condition_offsets = {"C1": -0.12, "C2": 0.0, "C3": 0.12}
    # The formal trajectory run contains -5, 0 (clean baseline), and +5.
    # Keep the requested dose axis explicit so future -2/+2 runs align with
    # the same figure without changing its semantics.
    dose_ticks = [-5.0, -2.0, 0.0, 2.0, 5.0]
    endpoint_labels = {
        "final_soft_sa": "Final canonical soft-SA",
        "cle_probe_sa": "CLE probe SA",
    }
    for endpoint in ("final_soft_sa", "cle_probe_sa"):
        selected = [r for r in condition_rows if r["endpoint"] == endpoint and r["group"] == "answer_equal_macro"]
        fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.0), sharex=True, sharey="row")
        for column, (panl, cle) in enumerate(PAIRS):
            absolute_axis, delta_axis = axes[0, column], axes[1, column]
            clean_rows = [r for r in selected if r["panl_layer"] == panl and r["condition"] == "C0"]
            clean_mean = float(np.mean([r["mean_value"] for r in clean_rows]))
            absolute_axis.axhline(
                clean_mean, color="#333333", linestyle="--", linewidth=1.5,
                label=condition_labels["C0"], zorder=1,
            )
            delta_axis.axhline(0, color="#333333", linestyle="--", linewidth=1.5,
                               label=condition_labels["C0"], zorder=1)
            for condition in ("C1", "C2", "C3"):
                rows = sorted(
                    [r for r in selected if r["panl_layer"] == panl and r["condition"] == condition],
                    key=lambda r: float(r["alpha"]),
                )
                style = condition_styles[condition]
                offset = condition_offsets[condition]
                x = np.asarray([float(r["alpha"]) + offset for r in rows])
                y = np.asarray([float(r["mean_value"]) for r in rows])
                yerr = np.asarray([
                    [float(r["mean_value"]) - float(r["value_ci95_low"]) for r in rows],
                    [float(r["value_ci95_high"]) - float(r["mean_value"]) for r in rows],
                ])
                dy = np.asarray([float(r["mean_delta"]) for r in rows])
                dyerr = np.asarray([
                    [float(r["mean_delta"]) - float(r["delta_ci95_low"]) for r in rows],
                    [float(r["delta_ci95_high"]) - float(r["mean_delta"]) for r in rows],
                ])
                for axis, values, errors in ((absolute_axis, y, yerr), (delta_axis, dy, dyerr)):
                    axis.errorbar(
                        x, values, yerr=errors, linestyle="none", capsize=3,
                        markersize=6.5, markeredgewidth=1.2, markeredgecolor=style["color"],
                        markerfacecolor="white", color=style["color"], marker=style["marker"],
                        label=condition_labels[condition], zorder=3,
                    )
            absolute_axis.set_title(f"PANL L{panl} → CLE L{cle}", fontsize=12)
            for axis in (absolute_axis, delta_axis):
                axis.set_xticks(dose_ticks)
                axis.set_xlim(-5.65, 5.65)
                axis.grid(axis="y", alpha=.22, linewidth=.7)
                axis.spines[["top", "right"]].set_visible(False)
            delta_axis.set_xlabel("Steering strength α")
        axes[0, 0].set_ylabel(endpoint_labels[endpoint])
        axes[1, 0].set_ylabel("Change from clean run (ΔSA)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
                   bbox_to_anchor=(.5, .015), fontsize=9)
        fig.suptitle(f"Short-prompt PANL→CLE trajectory: {endpoint_labels[endpoint]}", fontsize=14)
        fig.text(.5, .055, "Points show answer-equal macro means with 95% bootstrap CIs; α=−2 and +2 were not run.",
                 ha="center", fontsize=9, color="#555555")
        fig.tight_layout(rect=(0, .09, 1, .95))
        fig.savefig(output / f"four_cell_{endpoint}.png", dpi=220)
        plt.close(fig)
    selected = [r for r in effect_rows if r["group"] == "answer_equal_macro"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    effect_labels = {
        "total": "PANL steering total",
        "residual": "After clean-CLE restoration",
        "attenuation": "Removed by restoration",
        "transfer": "Clean PANL + steered CLE",
        "interaction": "Interaction",
    }
    for axis, endpoint in zip(axes, ("final_soft_sa", "cle_probe_sa")):
        rows = [r for r in selected if r["endpoint"] == endpoint]
        for effect in ("total", "residual", "attenuation", "transfer", "interaction"):
            points = sorted([r for r in rows if r["effect"] == effect and r["alpha"] == 5.0], key=lambda r: r["panl_layer"])
            y = np.asarray([float(r["mean"]) for r in points])
            yerr = np.asarray([
                [float(r["mean"]) - float(r["ci95_low"]) for r in points],
                [float(r["ci95_high"]) - float(r["mean"]) for r in points],
            ])
            axis.errorbar([r["panl_layer"] for r in points], y, yerr=yerr,
                          marker="o", capsize=3, linewidth=1.5, label=effect_labels[effect])
        axis.axhline(0, color="black", lw=.8)
        axis.set_xticks([14, 16, 18])
        axis.set_xlabel("PANL steering layer")
        axis.set_ylabel("Effect on SA")
        axis.set_title(endpoint_labels[endpoint] + " (α=+5)")
        axis.grid(axis="y", alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "effects_final_and_probe.png", dpi=220)
    plt.close(fig)
