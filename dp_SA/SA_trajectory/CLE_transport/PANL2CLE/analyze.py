from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .config import BOOTSTRAP_REPEATS, LAYERS, SMOKE_BOOTSTRAP_REPEATS, WINDOWS
from .io_utils import atomic_csv, atomic_json, atomic_text, load_jsonl
from .statistics import donor_contrast


def load_trials(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [json.loads(path.read_text()) for path in sorted((root / "artifacts/trials").glob("*.json"))]
    return [r for r in rows if r.get("condition") == "clean"], [r for r in rows if r.get("window") in WINDOWS]


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(r[field]) for r in rows if r.get(field) is not None]; return float(np.mean(values)) if values else None


def summarize(root: Path, *, smoke: bool) -> dict[str, Any]:
    clean, trials = load_trials(root); repeats = SMOKE_BOOTSTRAP_REPEATS if smoke else BOOTSTRAP_REPEATS
    conditions = []
    for (window, layer, condition), values in sorted(_groups(trials, lambda r: (r["window"], int(r["layer"]), r["condition"])).items()):
        conditions.append({"window": window, "layer": layer, "condition": condition, "n": len(values),
                           "mean_delta_soft_sa": _mean(values, "delta_soft_sa"), "mean_abs_delta_soft_sa": _mean(values, "abs_delta_soft_sa"),
                           "hard_change_rate": _mean(values, "hard_changed"), "mean_toward_score": _mean(values, "toward_score"),
                           "toward_rate": _mean([r for r in values if not r["zero_donor_gap"]], "toward")})
    contrasts = donor_contrast(trials, repeats=repeats, seed=42)
    toward = []
    for (window, layer), values in sorted(_groups(trials, lambda r: (r["window"], int(r["layer"]))).items()):
        nonzero = [r for r in values if not r["zero_donor_gap"]]
        toward.append({"window": window, "layer": layer, "n": len(values), "nonzero_gap_n": len(nonzero),
                       "zero_gap_n": len(values) - len(nonzero), "toward_rate": _mean(nonzero, "toward"), "mean_toward_score": _mean(values, "toward_score")})
    answer_rows = []
    for (answer, window, layer), values in sorted(_groups(trials, lambda r: (r["answer"], r["window"], int(r["layer"]))).items()):
        answer_rows.append({"answer": answer, "window": window, "layer": layer, "n": len(values), "mean_delta_soft_sa": _mean(values, "delta_soft_sa"), "toward_rate": _mean([r for r in values if not r["zero_donor_gap"]], "toward")})
    donor_rows = []
    donors = sorted({str(r["donor_case_id"]) for r in trials})
    for donor in donors:
        retained = [r for r in trials if str(r["donor_case_id"]) != donor]
        donor_rows.append({"left_out_donor": donor, "retained_n": len(retained), "mean_toward_score": _mean(retained, "toward_score"), "toward_rate": _mean([r for r in retained if not r["zero_donor_gap"]], "toward")})
    atomic_csv(root / "tables/condition_summary.csv", conditions); atomic_csv(root / "tables/donor_contrasts.csv", contrasts)
    atomic_csv(root / "tables/toward_donor.csv", toward); atomic_csv(root / "tables/answer_sensitivity.csv", answer_rows)
    atomic_csv(root / "tables/leave_one_donor_out.csv", donor_rows)
    _figures(root, trials, conditions, contrasts, toward)
    _readme(root, smoke=smoke)
    summary = {"status": "complete", "smoke_only": smoke, "clean_count": len(clean), "trial_count": len(trials),
               "bootstrap_repeats": repeats, "condition_rows": len(conditions), "contrast_rows": len(contrasts),
               "donor_reuse_counts": dict(Counter(r["donor_case_id"] for r in trials))}
    atomic_json(root / "summary.json", summary); return summary


def _groups(rows: list[dict[str, Any]], key):
    output = defaultdict(list)
    for row in rows: output[key(row)].append(row)
    return output


def _figures(root: Path, trials: list[dict[str, Any]], conditions: list[dict[str, Any]], contrasts: list[dict[str, Any]], toward: list[dict[str, Any]]) -> None:
    combined = [r for r in contrasts if r["stratum"] == "combined_equal_side"]
    _heatmap(combined, "estimate", root / "figures/fig1_donor_contrast_heatmap.png", "δ donor (high − low)", annotate=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for ax, window in zip(axes.flat, WINDOWS):
        subset = [row for row in conditions if row["window"] == window]
        for condition in sorted({row["condition"] for row in subset}):
            values = sorted((row for row in subset if row["condition"] == condition), key=lambda row: row["layer"])
            ax.plot([row["layer"] for row in values], [row["mean_delta_soft_sa"] for row in values], marker="o", label=condition)
        ax.axhline(0, color="black", lw=.7); ax.set_title(window)
        if subset: ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(root / "figures/fig2_conditions_by_layer.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"high_image": "tab:red", "high_text": "tab:blue"}; markers = {"high_image": "o", "high_text": "x"}
    for recipient_side in colors:
        for donor_side in markers:
            values = [row for row in trials if row["recipient_side"] == recipient_side and row["donor_side"] == donor_side]
            if values: ax.scatter([row["donor_gap"] for row in values], [row["delta_soft_sa"] for row in values], color=colors[recipient_side], marker=markers[donor_side], alpha=.65, label=f"{recipient_side}/{donor_side}")
    ax.axhline(0, color="black", lw=.7); ax.axvline(0, color="black", lw=.7); fig.tight_layout(); fig.savefig(root / "figures/fig3_donor_gap_scatter.png", dpi=180); plt.close(fig)
    _heatmap(toward, "toward_rate", root / "figures/fig4_toward_rate_heatmap.png", "Toward-donor rate", annotate=True, vmin=0, vmax=1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, field, title in zip(axes, ("hard_change_rate", "mean_abs_delta_soft_sa"), ("Hard-change rate", "Mean |ΔSA|")):
        groups = sorted({row["condition"] for row in conditions}); width = .8 / max(len(groups), 1); x = np.arange(len(WINDOWS))
        for index, condition in enumerate(groups):
            values = []
            for window in WINDOWS:
                cells = [row[field] for row in conditions if row["window"] == window and row["condition"] == condition and row[field] is not None]
                values.append(float(np.mean(cells)) if cells else np.nan)
            ax.bar(x + (index - (len(groups) - 1) / 2) * width, values, width=width, label=condition)
        ax.set_xticks(x, WINDOWS, rotation=30); ax.set_title(title); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(root / "figures/fig5_disruption_diagnostics.png", dpi=180); plt.close(fig)


def _heatmap(rows: list[dict[str, Any]], value: str, path: Path, title: str, *, annotate: bool, vmin=None, vmax=None) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    layers = sorted({int(row["layer"]) for row in rows})
    if rows and layers:
        lookup = {(row["window"], int(row["layer"])): row.get(value) for row in rows}
        table = np.asarray([[np.nan if lookup.get((window, layer)) is None else float(lookup[window, layer]) for layer in layers] for window in WINDOWS])
        if vmin is None:
            bound = max(abs(float(np.nanmin(table))), abs(float(np.nanmax(table))), 1e-12); vmin, vmax = -bound, bound; cmap = "coolwarm"
        else: cmap = "viridis"
        image = ax.imshow(table, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax); fig.colorbar(image, ax=ax)
        ax.set_xticks(range(len(layers)), layers); ax.set_yticks(range(len(WINDOWS)), WINDOWS)
        if annotate:
            for i in range(len(WINDOWS)):
                for j in range(len(layers)):
                    if np.isfinite(table[i, j]): ax.text(j, i, f"{table[i, j]:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title(title); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _readme(root: Path, *, smoke: bool) -> None:
    text = """# PANL→CLE answer-matched activation window swap

本目录是独立实验输出。当前结果为：%s。

## 六个窗口

- W1：开始来源归因；W2：定义归因对象；W3：比较文本、图像与平衡三种判断。
- W4：把判断组织成 0–8 整数；W5：绑定高分与图像贡献；W6：建立 class 4 的平衡规则。

## 如何读取 2×2

H←L 下降且 L←H 上升，同时 H←H、L←L 较弱，才符合双向 SA 状态转移。主统计量是同一 recipient 在 high donor 和 low donor 下的差，因此消除了 recipient 自身基线；不能把多个 trial 当作新增独立样本。

## 结论边界

方向一致、跨侧强于同侧、随相邻语义阶段或 layer 连续，并且 final SA 与合格的后期 CLE probe 同向，支持“该窗口包含对最终 SA 有功能作用的 SA 相关状态”。若只有单向变化、同侧同样大或全部节点效果相似，更像普通跨样本扰动或均值回归。本实验不能证明这是纯 SA 表征，也不能单独证明信息必然来自 PANL。
""" % ("smoke，仅验证流程，不作科学结论" if smoke else "正式探索性结果")
    atomic_text(root / "README_zh.md", text)
