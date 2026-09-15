from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .config import ALPHAS, BOOTSTRAP_REPEATS, CAUSAL_PAIRS, SEED
from .contracts import atomic_json, atomic_jsonl, expected_logical_count, load_jsonl


def expand_logical(trials: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trials: by_case[str(row["case_id"])].append(row)
    output = []
    for case, rows in by_case.items():
        c0 = next(r for r in rows if r["condition"] == "C0")
        lookup = {(r["condition"], r.get("panl_layer"), r.get("cle_layer"), float(r["alpha"])): r for r in rows}
        for panl, cle in CAUSAL_PAIRS:
            clean_probe = float(c0["cle_probe"][str(cle)]["predicted_soft_sa"])
            for alpha in ALPHAS:
                c1 = lookup[("C1", panl, None, float(alpha))]
                cells = (("C0", c0), ("C1", c1), ("C2", lookup[("C2", panl, cle, float(alpha))]), ("C3", lookup[("C3", panl, cle, float(alpha))]))
                for condition, row in cells:
                    probe = float(row["cle_probe"][str(cle)]["predicted_soft_sa"])
                    output.append({"case_id": case, "item_id": row["item_id"], "answer": row["answer"], "test_side": row["test_side"], "condition": condition, "panl_layer": panl, "cle_layer": cle, "alpha": float(alpha), "final_soft_sa": float(row["final_soft_sa"]), "delta_final_soft_sa": float(row["final_soft_sa"]) - float(c0["final_soft_sa"]), "cle_probe_sa": probe, "delta_cle_probe_sa": probe - clean_probe, "hard_sa_class": row["hard_sa_class"], "hard_change": int(row["hard_sa_class"] != c0["hard_sa_class"])})
    return output


def effect_values(rows: Sequence[dict[str, Any]], field: str) -> dict[str, float]:
    values = {r["condition"]: float(r[field]) for r in rows}; c0, c1, c2, c3 = (values[x] for x in ("C0", "C1", "C2", "C3"))
    return {"total": c1-c0, "residual": c2-c0, "attenuation": c1-c2, "transfer": c3-c0, "interaction": c1-c2-c3+c0}


def _aggregate(rows: Sequence[dict[str, Any]], value: Callable[[dict[str, Any]], float], group: str) -> float:
    selected = [r for r in rows if group not in ("image_side", "text_side") or r["test_side"] == group]
    if not selected: return math.nan
    if group == "answer_equal_macro":
        answers = sorted({r["answer"] for r in selected})
        return float(np.mean([np.mean([value(r) for r in selected if r["answer"] == answer]) for answer in answers]))
    return float(np.mean([value(r) for r in selected]))


def _bootstrap(rows: Sequence[dict[str, Any]], value: Callable[[dict[str, Any]], float], group: str, repeats: int) -> tuple[float, float, float]:
    point = _aggregate(rows, value, group); rng = np.random.default_rng(SEED); values = []
    if group == "answer_equal_macro":
        grouped = {a: [r for r in rows if r["answer"] == a] for a in sorted({r["answer"] for r in rows})}
        for _ in range(repeats): values.append(float(np.mean([np.mean([value(x) for x in rng.choice(v, len(v), replace=True)]) for v in grouped.values()])))
    else:
        pool = [r for r in rows if group == "overall_micro" or r["test_side"] == group]
        for _ in range(repeats): values.append(float(np.mean([value(x) for x in rng.choice(pool, len(pool), replace=True)])))
    low, high = np.quantile(values, [.025, .975]); return point, float(low), float(high)


def _csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def analyze(*, output_root: Path, repeats: int = BOOTSTRAP_REPEATS) -> dict[str, Any]:
    root = Path(output_root); trials = load_jsonl(root/"artifacts/trials.jsonl"); logical = expand_logical(trials); case_count = len({r["case_id"] for r in trials})
    if len(logical) != expected_logical_count(case_count): raise RuntimeError("Logical four-cell grid incomplete")
    keys = {(r["case_id"], r["condition"], r["panl_layer"], r["cle_layer"], r["alpha"]) for r in logical}
    if len(keys) != len(logical): raise RuntimeError("Duplicate logical rows")
    condition_rows = []; effect_rows = []
    for panl, cle in CAUSAL_PAIRS:
        for alpha in ALPHAS:
            cell = [r for r in logical if r["panl_layer"] == panl and r["cle_layer"] == cle and r["alpha"] == alpha]; per_case = defaultdict(list)
            for r in cell: per_case[r["case_id"]].append(r)
            for endpoint, field in (("final_soft_sa", "delta_final_soft_sa"), ("cle_probe_sa", "delta_cle_probe_sa")):
                for condition in ("C0", "C1", "C2", "C3"):
                    selected = [r for r in cell if r["condition"] == condition]
                    for group in ("answer_equal_macro", "overall_micro", "image_side", "text_side"):
                        point, low, high = _bootstrap(selected, lambda r: float(r[field]), group, repeats)
                        condition_rows.append({"endpoint": endpoint, "condition": condition, "panl_layer": panl, "cle_layer": cle, "alpha": alpha, "group": group, "mean_delta": point, "ci95_low": low, "ci95_high": high, "case_count": len(selected)})
                effects_by_case = []
                for records in per_case.values(): effects_by_case.append({**records[0], **effect_values(records, field)})
                for effect in ("total", "residual", "attenuation", "transfer", "interaction"):
                    for group in ("answer_equal_macro", "overall_micro", "image_side", "text_side"):
                        point, low, high = _bootstrap(effects_by_case, lambda r, e=effect: float(r[e]), group, repeats)
                        effect_rows.append({"endpoint": endpoint, "effect": effect, "panl_layer": panl, "cle_layer": cle, "alpha": alpha, "group": group, "mean": point, "ci95_low": low, "ci95_high": high, "case_count": len(effects_by_case)})
    atomic_jsonl(root/"artifacts/logical_four_cell.jsonl", logical); _csv(root/"tables/condition_summary.csv", condition_rows); _csv(root/"tables/effect_summary.csv", effect_rows); _plot(root, condition_rows)
    result = {"status": "complete", "physical_trial_count": len(trials), "logical_row_count": len(logical), "case_count": case_count, "bootstrap_repeats": repeats, "condition_cells": len(condition_rows), "effect_cells": len(effect_rows)}; atomic_json(root/"analysis_summary.json", result); return result


def _plot(root: Path, rows: Sequence[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    selected = [r for r in rows if r["endpoint"] == "cle_probe_sa" and r["group"] == "answer_equal_macro"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, condition in zip(axes.flat, ("C0", "C1", "C2", "C3")):
        for panl in (14, 16, 18):
            for alpha, style in ((-5.0, "--"), (5.0, "-")):
                points = sorted((r for r in selected if r["condition"] == condition and r["panl_layer"] == panl and r["alpha"] == alpha), key=lambda r: r["cle_layer"])
                if points: ax.plot([r["cle_layer"] for r in points], [r["mean_delta"] for r in points], marker="o", linestyle=style, label=f"PANL L{panl}, a={alpha:g}")
        ax.axhline(0, color="black", lw=.8); ax.set_title(condition); ax.grid(alpha=.2); ax.set_xticks((15, 17, 19, 21))
    axes[1, 0].set_xlabel("CLE layer"); axes[1, 1].set_xlabel("CLE layer"); axes[0, 0].set_ylabel("delta probe SA"); axes[1, 0].set_ylabel("delta probe SA")
    handles, labels = axes[0, 1].get_legend_handles_labels(); fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False); fig.tight_layout(rect=(0, 0, 1, .9)); destination = root/"figures/four_cell_cle_probe_trajectory.png"; destination.parent.mkdir(parents=True, exist_ok=True); fig.savefig(destination, dpi=220); plt.close(fig)
