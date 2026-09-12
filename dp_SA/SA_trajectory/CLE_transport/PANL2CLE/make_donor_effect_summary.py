from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import OUTPUT_PARENT
from .io_utils import atomic_csv, atomic_json, atomic_text


ROOT = OUTPUT_PARENT
SUMMARY_ROOT = ROOT / "donor_effect_summary"
STAGE1 = ROOT / "layer_trajectory_stage1/tables/layer_trajectory.csv"
BISECTION4 = ROOT / "window_bisection_stage2/tables/bisection_effects.csv"
ROUND2 = ROOT / "window_bisection_stage2_round2_cle_validation_v2/tables/round2_effects.csv"
PROBE = ROOT / "window_bisection_w2_singleton_stage3/tables/next_layer_cle_probe_effects.csv"
SINGLE = ROOT / "window_bisection_w2_singleton_stage3/tables/singleton_final_sa_effects.csv"


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _row(granularity: str, window: str, layer: int | str, segment: str, source: dict[str, str], *, retention: str = "") -> dict[str, str]:
    estimate = source.get("estimate", source.get("donor_contrast", ""))
    sem = source.get("sem", source.get("contrast_sem", ""))
    ci_low = source.get("ci_low", source.get("contrast_ci_low", ""))
    ci_high = source.get("ci_high", source.get("contrast_ci_high", ""))
    bootstrap_p = source.get("bootstrap_p_two_sided", source.get("contrast_p", ""))
    bh_q = source.get("bh_fdr_q", source.get("contrast_bh_fdr_q", ""))
    return {
        "granularity": granularity, "window": window, "layer": str(layer), "segment": segment,
        "metric": "final_soft_sa_paired_donor_effect", "donor_effect": estimate,
        "sem": sem, "ci_low": ci_low, "ci_high": ci_high,
        "bootstrap_p": bootstrap_p, "bh_fdr_q": bh_q,
        "retention_vs_parent": retention, "next_layer_cle_probe_effect": "",
    }


def collect() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source in _read(STAGE1):
        rows.append(_row("8-token", source["window"], source["layer"], "full8", source))
    first_split = _read(BISECTION4)
    round2 = _read(ROUND2)
    full_lookup = {(r["window"], r["layer"]): r for r in rows if r["granularity"] == "8-token"}
    four_lookup: dict[tuple[str, str, str], dict[str, str]] = {}
    for source in first_split:
        if source["component"] not in ("left4", "right4"):
            continue
        parent = full_lookup[source["window"], source["layer"]]
        retention = str(float(source["estimate"]) / float(parent["donor_effect"]))
        item = _row("4-token", source["window"], source["layer"], source["component"], source, retention=retention)
        rows.append(item); four_lookup[source["window"], source["layer"], source["component"]] = item
    for source in round2:
        if not source["component"].endswith("2"):
            continue
        parent_name = "left4" if source["component"].startswith("left4") else "right4"
        parent = four_lookup[source["window"], source["layer"], parent_name]
        retention = str(float(source["estimate"]) / float(parent["donor_effect"]))
        rows.append(_row("2-token", source["window"], source["layer"], source["component"], source, retention=retention))
    probe = {(r["segment"]): r for r in _read(PROBE)}
    parent2 = next(r for r in round2 if r["window"] == "W2" and r["layer"] == "17" and r["component"] == "right4_left2")
    for source in _read(SINGLE):
        if "token" not in source["component"]:
            continue
        retention = str(float(source["estimate"]) / float(parent2["estimate"]))
        item = _row("1-token", "W2", 17, source["component"], source, retention=retention)
        item["next_layer_cle_probe_effect"] = probe[source["component"]]["estimate"]
        rows.append(item)
    order = {"8-token": 0, "4-token": 1, "2-token": 2, "1-token": 3}
    return sorted(rows, key=lambda r: (r["window"], order[r["granularity"]], int(r["layer"]), r["segment"]))


def _label(row: dict[str, str]) -> str:
    if row["granularity"] == "8-token":
        return f"8-L{row['layer']}"
    return f"{row['granularity'].split('-')[0]}-{row['segment'].replace('right4_', 'R').replace('left4_', 'L').replace('right4', 'R4').replace('left4', 'L4').replace('token', 't')}"


def _plot(rows: list[dict[str, str]], window: str) -> None:
    subset = [row for row in rows if row["window"] == window]
    colors = {"8-token": "#4C78A8", "4-token": "#F58518", "2-token": "#54A24B", "1-token": "#E45756"}
    x = np.arange(len(subset), dtype=float)
    values = np.asarray([float(row["donor_effect"]) for row in subset])
    lows = np.asarray([float(row["ci_low"]) for row in subset])
    highs = np.asarray([float(row["ci_high"]) for row in subset])
    fig, ax = plt.subplots(figsize=(max(12, len(subset) * .65), 5.5))
    ax.bar(x, values, color=[colors[row["granularity"]] for row in subset], alpha=.88, width=.72)
    ax.errorbar(x, values, yerr=[values - lows, highs - values], fmt="none", ecolor="black", capsize=3, lw=1)
    ax.axhline(0, color="black", lw=.9)
    ax.set_xticks(x, [_label(row) for row in subset], rotation=45, ha="right")
    ax.set_ylabel("paired high-donor minus low-donor soft-SA effect")
    ax.set_title(f"{window}: donor effect across window granularity")
    represented = [name for name in colors if any(r["granularity"] == name for r in subset)]
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[name]) for name in represented]
    ax.legend(handles=handles, labels=represented, title="window granularity", frameon=False)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(SUMMARY_ROOT / "figures" / f"{window}_donor_effects.png", dpi=220)
    plt.close(fig)


def _plot_layer_trajectory(rows: list[dict[str, str]], window: str) -> None:
    subset = [row for row in rows if row["window"] == window and row["granularity"] == "8-token"]
    subset.sort(key=lambda row: int(row["layer"]))
    x = np.asarray([int(row["layer"]) for row in subset])
    values = np.asarray([float(row["donor_effect"]) for row in subset])
    lows = np.asarray([float(row["ci_low"]) for row in subset]); highs = np.asarray([float(row["ci_high"]) for row in subset])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(x, values, yerr=[values - lows, highs - values], fmt="o-", color="#4C78A8", capsize=4, lw=2)
    ax.axhline(0, color="black", lw=.9); ax.set_xticks(x)
    ax.set_xlabel("decoder layer"); ax.set_ylabel("paired high-donor minus low-donor soft-SA effect")
    ax.set_title(f"{window}: 8-token donor-effect layer trajectory")
    ax.grid(axis="y", alpha=.2); fig.tight_layout()
    fig.savefig(SUMMARY_ROOT / "figures" / f"{window}_layer_trajectory.png", dpi=220); plt.close(fig)


def _plot_bisection(rows: list[dict[str, str]], window: str) -> None:
    lookup = {(row["granularity"], row["segment"]): row for row in rows if row["window"] == window}
    if window == "W2":
        specs = [
            ("8-token", "full8", "8T\nfull window\nto the formation of the fixed answer ."),
            ("4-token", "left4", "4L\nto the formation of"),
            ("4-token", "right4", "4R\nthe fixed answer ."),
            ("2-token", "right4_left2", "2R-L\nthe fixed"),
            ("2-token", "right4_right2", "2R-R\nanswer ."),
            ("1-token", "right4_left2_token1", "1-RL-1\nthe"),
            ("1-token", "right4_left2_token2", "1-RL-2\nfixed"),
        ]
    else:
        specs = [
            ("8-token", "full8", "8T\nfull window"),
            ("4-token", "left4", "4L\nleft half"),
            ("4-token", "right4", "4R\nright half"),
            ("2-token", "left4_left2", "2L-L\nleft first 2"),
            ("2-token", "left4_right2", "2L-R\nleft last 2"),
            ("2-token", "right4_left2", "2R-L\nright first 2"),
            ("2-token", "right4_right2", "2R-R\nright last 2"),
        ]
    x = np.arange(len(specs), dtype=float)
    colors = {"8-token": "#4C78A8", "4-token": "#F58518", "2-token": "#54A24B", "1-token": "#E45756"}
    fig, ax = plt.subplots(figsize=(max(12, len(specs) * 1.35), 6.2))
    for position, (granularity, segment, label) in enumerate(specs):
        row = lookup.get((granularity, segment))
        if row is None:
            ax.text(position, 0.00012, "NOT\nMEASURED", ha="center", va="bottom", color="#666666", fontsize=9)
            continue
        value = float(row["donor_effect"]); low = float(row["ci_low"]); high = float(row["ci_high"])
        ax.bar(position, value, color=colors[granularity], alpha=.9, width=.7)
        ax.errorbar(position, value, yerr=[[value - low], [high - value]], fmt="none", ecolor="black", capsize=4, lw=1)
    ax.axhline(0, color="black", lw=.9); ax.set_xticks(x, [label for _, _, label in specs], rotation=0)
    ax.set_ylabel("paired high-donor minus low-donor soft-SA effect")
    ax.set_title(f"{window}: recursive window bisection")
    represented = [name for name in colors if any(granularity == name for granularity, _, _ in specs)]
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[name]) for name in represented]
    ax.legend(handles=handles, labels=represented, title="granularity", frameon=False)
    ax.grid(axis="y", alpha=.2); fig.tight_layout()
    fig.savefig(SUMMARY_ROOT / "figures" / f"{window}_bisection.png", dpi=220); plt.close(fig)


def main() -> None:
    (SUMMARY_ROOT / "figures").mkdir(parents=True, exist_ok=True)
    rows = collect()
    atomic_csv(SUMMARY_ROOT / "donor_effect_table.csv", rows)
    headers = ["granularity", "window", "layer", "segment", "donor_effect", "sem", "95% CI", "bootstrap_p", "bh_fdr_q", "retention_vs_parent", "next_layer_cle_probe_effect"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join([
            row["granularity"], row["window"], row["layer"], row["segment"], row["donor_effect"], row["sem"],
            f"[{row['ci_low']}, {row['ci_high']}]", row["bootstrap_p"], row["bh_fdr_q"], row["retention_vs_parent"], row["next_layer_cle_probe_effect"] or "—",
        ]) + " |")
    atomic_text(SUMMARY_ROOT / "donor_effect_table.md", "\n".join(lines) + "\n")
    for window in ("W2", "W5"):
        _plot(rows, window)
        _plot_layer_trajectory(rows, window)
        _plot_bisection(rows, window)
    atomic_json(SUMMARY_ROOT / "manifest.json", {"status": "complete", "rows": len(rows), "windows": ["W2", "W5"], "figures": [
        "W2_donor_effects.png", "W5_donor_effects.png", "W2_layer_trajectory.png", "W2_bisection.png",
        "W5_layer_trajectory.png", "W5_bisection.png",
    ]})


if __name__ == "__main__":
    main()
