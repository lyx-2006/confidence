from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COARSE = ROOT / "dp_SA" / "attention_block" / "outputs" / "formal_both_seed42_w12_20260823T093446Z"
DEFAULT_DELAYED_REFINE = ROOT / "dp_SA" / "attention_block" / "outputs" / "formal_delayed_refine_only_seed42_20260823T1238Z"

CONDITION_LABELS = {
    "panl_to_evidence": "PANL→EVIDENCE",
    "panl_to_answer": "PANL→ANSWER",
    "panl_to_evidence_answer": "PANL→E+A",
    "panl_plus_1_to_evidence_answer": "PANL+1→E+A",
    "sac_to_panl": "SAC→PANL",
    "sac_to_panl_plus_1": "SAC→PANL+1",
    "sac_to_evidence": "SAC→EVIDENCE",
    "sac_to_answer": "SAC→ANSWER",
    "sac_to_all_content": "SAC→ALL_CONTENT",
    "all_downstream_to_panl": "ALL_DOWNSTREAM→PANL",
    "all_downstream_to_panl_plus_1": "ALL_DOWNSTREAM→PANL+1",
    "all_later_to_evidence": "ALL_LATER→EVIDENCE",
    "all_later_to_evidence_keep_panl": "ALL_LATER→EVIDENCE (keep PANL)",
    "all_later_to_answer": "ALL_LATER→ANSWER",
    "all_later_to_answer_keep_panl": "ALL_LATER→ANSWER (keep PANL)",
    "all_later_to_evidence_answer": "ALL_LATER→E+A",
    "all_later_to_evidence_answer_keep_panl": "ALL_LATER→E+A (keep PANL)",
}

SAC_CONDITIONS = (
    "sac_to_panl",
    "sac_to_panl_plus_1",
    "sac_to_evidence",
    "sac_to_answer",
    "sac_to_all_content",
)
PANL_CONDITIONS = (
    "panl_to_evidence",
    "panl_to_answer",
    "panl_to_evidence_answer",
    "panl_plus_1_to_evidence_answer",
)

COLORS = {
    "sac_to_panl": "#7b61b3",
    "sac_to_panl_plus_1": "#ed8b2f",
    "sac_to_evidence": "#d95f8d",
    "sac_to_answer": "#4c9f70",
    "sac_to_all_content": "#c44e52",
    "panl_to_evidence": "#4c78a8",
    "panl_to_answer": "#8c8c8c",
    "panl_to_evidence_answer": "#6f4e7c",
    "panl_plus_1_to_evidence_answer": "#e07b39",
}

DELAYED_REFINE_COLORS = {
    "sac_to_all_content": "#d62728",
    "sac_to_panl_plus_1": "#1f77b4",
    "panl_to_evidence_answer": "#2ca02c",
    "panl_plus_1_to_evidence_answer": "#e377c2",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def logit_difference(logits: Sequence[float], target: int) -> float:
    values = np.asarray(logits, dtype=float)
    alternatives = np.delete(values, target)
    return float(values[target] - alternatives.mean())


def _sem(values: Sequence[float]) -> float:
    if len(values) < 2:
        return float("nan")
    return float(np.std(np.asarray(values, dtype=float), ddof=1) / math.sqrt(len(values)))


def build_item_rows(coarse_output: Path, delayed_refine_output: Path) -> list[dict[str, Any]]:
    coarse_clean = {(row["arm"], row["case_id"]): row for row in load_jsonl(coarse_output / "clean_baselines.jsonl")}
    refine_clean = {(row["arm"], row["case_id"]): row for row in load_jsonl(delayed_refine_output / "clean_baselines.jsonl")}
    sources = [
        ("formal_coarse", coarse_output, [row for row in load_jsonl(coarse_output / "blocked_results.jsonl") if row["phase"] == "coarse"], coarse_clean),
        ("formal_delayed_refine", delayed_refine_output, load_jsonl(delayed_refine_output / "blocked_results.jsonl"), refine_clean),
    ]
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, int, int]] = set()
    for source_name, source_path, rows, clean_map in sources:
        for row in rows:
            key = (
                str(row["arm"]), str(row["case_id"]), str(row["phase"]), str(row["condition"]),
                int(row["window_start"]), int(row["window_end"]),
            )
            if key in seen:
                raise ValueError(f"Duplicate final result: {key}")
            seen.add(key)
            clean = clean_map[(row["arm"], row["case_id"])]
            target = int(clean["clean_class"])
            clean_diff = logit_difference(clean["class_logits"], target)
            blocked_diff = logit_difference(row["class_logits"], target)
            output.append({
                "arm": row["arm"],
                "phase": row["phase"],
                "source_run": source_name,
                "source_path": str(source_path.resolve()),
                "case_id": row["case_id"],
                "item_id": str(row["item_id"]),
                "test_side": row["test_side"],
                "condition": row["condition"],
                "condition_label": CONDITION_LABELS[row["condition"]],
                "window_start": int(row["window_start"]),
                "window_end": int(row["window_end"]),
                "window_center": float(row["window_center"]),
                "blocked_layer_count": int(row["blocked_layer_count"]),
                "is_sliding_window": int(row["condition"] in SAC_CONDITIONS + PANL_CONDITIONS),
                "clean_class": target,
                "blocked_class": int(row["blocked_class"]),
                "token_changed": int(row["first_token_changed"]),
                "clean_logit_difference": clean_diff,
                "blocked_logit_difference": blocked_diff,
                "logit_difference_change": blocked_diff - clean_diff,
                "logit_margin_disruption": float(row["logit_margin_disruption"]),
                "clean_soft_sa": float(clean["soft_sa_image_score"]),
                "blocked_soft_sa": float(row["blocked_soft_sa"]),
                "delta_soft_sa": float(row["delta_soft_sa"]),
            })
    return output


def build_summary_rows(item_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in item_rows:
        base = (
            row["arm"], row["phase"], row["source_run"], row["condition"], row["condition_label"],
            row["window_start"], row["window_end"], row["window_center"], row["blocked_layer_count"],
            row["is_sliding_window"],
        )
        grouped[base + ("all_100",)].append(row)
        grouped[base + (row["test_side"],)].append(row)
    output: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        (
            arm, phase, source_run, condition, label, start, end, center, layer_count,
            is_sliding, group,
        ) = key
        changes = [float(row["token_changed"]) for row in rows]
        logit_changes = [float(row["logit_difference_change"]) for row in rows]
        disruptions = [float(row["logit_margin_disruption"]) for row in rows]
        output.append({
            "arm": arm,
            "phase": phase,
            "source_run": source_run,
            "condition": condition,
            "condition_label": label,
            "window_start": start,
            "window_end": end,
            "window_center": center,
            "blocked_layer_count": layer_count,
            "is_sliding_window": is_sliding,
            "group": group,
            "n": len(rows),
            "token_change_rate": float(np.mean(changes)),
            "token_change_rate_pct": 100.0 * float(np.mean(changes)),
            "token_change_rate_sem": _sem(changes),
            "token_change_rate_sem_pct": 100.0 * _sem(changes),
            "logit_difference_change_mean": float(np.mean(logit_changes)),
            "logit_difference_change_sem": _sem(logit_changes),
            "logit_margin_disruption_mean": float(np.mean(disruptions)),
            "logit_margin_disruption_sem": _sem(disruptions),
        })
    return sorted(output, key=lambda row: (
        row["arm"], row["group"], row["phase"], row["condition"], row["window_center"]
    ))


def validate_final_data(item_rows: Sequence[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str, str, int, int], set[str]] = defaultdict(set)
    for row in item_rows:
        key = (row["arm"], row["phase"], row["condition"], row["window_start"], row["window_end"])
        groups[key].add(row["case_id"])
    bad = {key: len(cases) for key, cases in groups.items() if len(cases) != 100}
    if bad:
        raise ValueError(f"Final plotted/data groups are incomplete: {bad}")
    coarse = [key for key in groups if key[1] == "coarse"]
    refine = [key for key in groups if key[1] == "refine"]
    if len(coarse) != 106 or len(refine) != 48:
        raise ValueError(f"Unexpected final condition-window groups: coarse={len(coarse)}, refine={len(refine)}")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_metric(
    summary_rows: Sequence[dict[str, Any]], arm: str, metric: str, output: Path
) -> None:
    all_rows = [
        row for row in summary_rows
        if row["arm"] == arm and row["group"] == "all_100" and row["is_sliding_window"]
    ]
    if metric == "token_change_rate":
        mean_key, sem_key = "token_change_rate_pct", "token_change_rate_sem_pct"
        ylabel, title_metric = "Token Change Rate (%)", "Token change rate"
    elif metric == "logit_difference_change":
        mean_key, sem_key = "logit_difference_change_mean", "logit_difference_change_sem"
        ylabel, title_metric = "Logit Difference Change", "Logit difference change"
    else:
        raise ValueError(metric)

    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(2, 1, figsize=(8.3, 7.2), sharex=True, constrained_layout=True)
    panels = [("A", "SAC reads from upstream positions", SAC_CONDITIONS),
              ("B", "PANL gathers from evidence and answer", PANL_CONDITIONS)]
    for ax, (panel, subtitle, conditions) in zip(axes, panels):
        for condition in conditions:
            condition_rows = [row for row in all_rows if row["condition"] == condition]
            for phase in ("coarse", "refine"):
                phase_rows = sorted((row for row in condition_rows if row["phase"] == phase), key=lambda row: row["window_center"])
                if not phase_rows:
                    continue
                width = int(phase_rows[0]["blocked_layer_count"])
                label = f"{CONDITION_LABELS[condition]} ({width}-layer)"
                ax.plot(
                    [row["window_center"] for row in phase_rows],
                    [row[mean_key] for row in phase_rows],
                    color=COLORS[condition],
                    linestyle="-" if phase == "refine" else "--",
                    marker="o" if phase == "refine" else "s",
                    markersize=3.5,
                    linewidth=1.35,
                    alpha=0.95,
                    label=label,
                )
        ax.axhline(0, color="#9a9a9a", linewidth=0.8, linestyle=(0, (2, 2)), zorder=0)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{panel}  {subtitle}", loc="left", fontweight="bold", fontsize=10)
        ax.grid(axis="y", color="#dddddd", linewidth=0.5, alpha=0.55)
        ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="best")
    centers = sorted({float(row["window_center"]) for row in all_rows})
    axes[-1].set_xticks(centers)
    axes[-1].set_xticklabels([f"{value:g}" for value in centers], rotation=45, ha="right")
    axes[-1].set_xlabel("Center layer of selectively blocked contiguous window")
    fig.suptitle(f"SA Attention Blocking — {arm.capitalize()} Arm\n{title_metric} (mean, n=100)", fontweight="bold", fontsize=12)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_delayed_refine_metric(
    summary_rows: Sequence[dict[str, Any]], metric: str, output: Path
) -> None:
    rows = [
        row for row in summary_rows
        if row["arm"] == "delayed" and row["phase"] == "refine"
        and row["group"] == "all_100" and row["is_sliding_window"]
    ]
    if metric == "token_change_rate":
        mean_key = "token_change_rate_pct"
        ylabel, title_metric = "Token Change Rate (%)", "Token change rate"
    elif metric == "logit_difference_change":
        mean_key = "logit_difference_change_mean"
        ylabel, title_metric = "Logit Difference Change", "Logit difference change"
    else:
        raise ValueError(metric)

    condition_order = (
        "sac_to_all_content",
        "sac_to_panl_plus_1",
        "panl_to_evidence_answer",
        "panl_plus_1_to_evidence_answer",
    )
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(8.3, 4.8), constrained_layout=True)
    for condition in condition_order:
        condition_rows = sorted(
            (row for row in rows if row["condition"] == condition),
            key=lambda row: row["window_center"],
        )
        if len(condition_rows) != 12:
            raise ValueError(f"Delayed refine curve is incomplete for {condition}: {len(condition_rows)}/12")
        ax.plot(
            [row["window_center"] for row in condition_rows],
            [row[mean_key] for row in condition_rows],
            color=DELAYED_REFINE_COLORS[condition],
            linestyle="-",
            marker="o",
            markersize=4,
            linewidth=1.45,
            alpha=0.95,
            label=CONDITION_LABELS[condition],
        )
    ax.axhline(0, color="#9a9a9a", linewidth=0.8, linestyle=(0, (2, 2)), zorder=0)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Center layer of selectively blocked 6-layer window")
    centers = sorted({float(row["window_center"]) for row in rows})
    ax.set_xticks(centers)
    ax.set_xticklabels([f"{value:g}" for value in centers], rotation=45, ha="right")
    ax.grid(axis="y", color="#dddddd", linewidth=0.5, alpha=0.55)
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="best")
    fig.suptitle(
        f"SA Attention Blocking — Delayed Arm, 6-layer Windows\n{title_metric} (mean, n=100)",
        fontweight="bold",
        fontsize=12,
    )
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def generate(
    coarse_output: Path,
    delayed_refine_output: Path,
    output: Path,
    arms: Sequence[str] = ("joint", "delayed"),
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    item_rows = build_item_rows(coarse_output, delayed_refine_output)
    validate_final_data(item_rows)
    summary_rows = build_summary_rows(item_rows)
    for arm in arms:
        if arm not in {"joint", "delayed"}:
            raise ValueError(f"Unknown arm: {arm}")
        write_csv(output / f"{arm}_item_level.csv", [row for row in item_rows if row["arm"] == arm])
        write_csv(output / f"{arm}_summary.csv", [row for row in summary_rows if row["arm"] == arm])
        if arm == "delayed":
            _plot_delayed_refine_metric(summary_rows, "token_change_rate", output / f"{arm}_figure1_token_change_rate")
            _plot_delayed_refine_metric(summary_rows, "logit_difference_change", output / f"{arm}_figure2_logit_difference_change")
        else:
            _plot_metric(summary_rows, arm, "token_change_rate", output / f"{arm}_figure1_token_change_rate")
            _plot_metric(summary_rows, arm, "logit_difference_change", output / f"{arm}_figure2_logit_difference_change")
    manifest = {
        "status": "complete",
        "coarse_source": str(coarse_output.resolve()),
        "delayed_refine_source": str(delayed_refine_output.resolve()),
        "item_rows": len(item_rows),
        "summary_rows": len(summary_rows),
        "joint_item_rows": sum(row["arm"] == "joint" for row in item_rows),
        "delayed_item_rows": sum(row["arm"] == "delayed" for row in item_rows),
        "definition": {
            "token_change_rate": "mean(blocked_class != fixed_clean_class)",
            "logit_difference": "fixed-clean-class logit minus mean of the other eight digit logits",
            "logit_difference_change": "blocked logit difference minus clean logit difference; negative means disruption",
            "error_bars": "item-level SEM; all-100 primary population",
        },
        "inclusion": {
            "joint": "complete coarse 12-layer windows and complete global 0-27 conditions; stopped partial refine excluded",
            "delayed": "complete coarse 12-layer windows, complete global 0-27 conditions, and complete selected-pair 6-layer refine",
            "plots": "joint remains coarse-only; delayed plots contain only completed 6-layer refine curves on one combined axis; global and delayed coarse conditions remain in CSV",
        },
    }
    (output / "README.md").write_text(
        "# SA attention-blocking Figure 7 style exports\n\n"
        "Each arm has separate item-level and aggregate CSV files, plus separate token-change-rate and "
        "logit-difference-change figures. Curves show all-100 mean points without error bars. Joint figures retain "
        "the completed 12-layer coarse windows. Delayed figures combine the four completed 6-layer refine curves "
        "on one axis and omit 12-layer curves. The stopped incomplete joint refine is excluded. Global 0–27 and "
        "delayed coarse conditions remain available in CSV.\n",
        encoding="utf-8",
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-output", type=Path, default=DEFAULT_COARSE)
    parser.add_argument("--delayed-refine-output", type=Path, default=DEFAULT_DELAYED_REFINE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=("joint", "delayed"), default=("joint", "delayed"))
    args = parser.parse_args(argv)
    generate(args.coarse_output.resolve(), args.delayed_refine_output.resolve(), args.output_dir.resolve(), args.arms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
