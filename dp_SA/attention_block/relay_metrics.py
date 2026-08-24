from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np


CONDITION_LABELS = {
    "full": "ALL_LATER↛A",
    "keep_panl": "ALL_LATER↛A, keep PANL→A",
    "keep_panl_plus_1": "ALL_LATER↛A, keep PANL+1→A",
}
COLORS = {"full": "#d62728", "keep_panl": "#2ca02c", "keep_panl_plus_1": "#e377c2"}


def recovery_proportion(d_full: float, d_keep: float) -> float:
    if d_full == 0:
        raise ZeroDivisionError("Mean full-block disruption is zero")
    return 100.0 * (float(d_full) - float(d_keep)) / float(d_full)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_table(item_rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    windows = sorted({(int(row["window_start"]), int(row["window_end"])) for row in item_rows})
    for start, end in windows:
        rows = [row for row in item_rows if int(row["window_start"]) == start]
        d_full = float(np.mean([float(row["D_full"]) for row in rows]))
        d_keep = float(np.mean([float(row["D_keepPANL"]) for row in rows]))
        d_control = float(np.mean([float(row["D_keepPANL_plus_1"]) for row in rows]))
        token_full = 100.0 * float(np.mean([float(row["change_rate_full"]) for row in rows]))
        token_keep = 100.0 * float(np.mean([float(row["change_rate_keepPANL"]) for row in rows]))
        token_control = 100.0 * float(np.mean([float(row["change_rate_keepPANL_plus_1"]) for row in rows]))
        recovery_panl = recovery_proportion(d_full, d_keep)
        recovery_control = recovery_proportion(d_full, d_control)
        output.append({
            "window_start": start,
            "window_end": end,
            "window_center": (start + end) / 2,
            "n": len(rows),
            "token_change_rate_full_pct": token_full,
            "token_change_rate_keepPANL_pct": token_keep,
            "token_change_rate_keepPANL_plus_1_pct": token_control,
            "logit_diff_change_full": -d_full,
            "logit_diff_change_keepPANL": -d_keep,
            "logit_diff_change_keepPANL_plus_1": -d_control,
            "D_full": d_full,
            "D_keepPANL": d_keep,
            "D_keepPANL_plus_1": d_control,
            "R_relay_PANL": d_full - d_keep,
            "R_relay_PANL_plus_1": d_full - d_control,
            "recovery_proportion_PANL_pct": recovery_panl,
            "recovery_proportion_PANL_plus_1_pct": recovery_control,
            "matched_recovery_advantage_pct_points": recovery_panl - recovery_control,
        })
    return output


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _base_axis(title: str, ylabel: str):
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    ax.set_title(title, fontweight="bold"); ax.set_xlabel("Center layer of selectively blocked 6-layer window"); ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#dddddd", linewidth=.5, alpha=.55)
    return fig, ax


def _save(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def generate(input_csv: Path, output: Path) -> list[dict[str, Any]]:
    output.mkdir(parents=True, exist_ok=True)
    table = build_table(_read(input_csv))
    _write_csv(output / "relay_three_metrics_table.csv", table)
    x = [row["window_center"] for row in table]

    fig, ax = _base_axis("Answer→PANL Relay Rescue — Token Change Rate (n=100)", "Token Change Rate (%)")
    for key, column in (
        ("full", "token_change_rate_full_pct"),
        ("keep_panl", "token_change_rate_keepPANL_pct"),
        ("keep_panl_plus_1", "token_change_rate_keepPANL_plus_1_pct"),
    ):
        ax.plot(x, [row[column] for row in table], marker="o", linewidth=1.7, color=COLORS[key], label=CONDITION_LABELS[key])
    ax.set_xticks(x); ax.axhline(0, color="#999999", linewidth=.8, linestyle=(0, (2, 2))); ax.legend(frameon=False)
    _save(fig, output / "figure1_token_change_rate")

    fig, ax = _base_axis("Answer→PANL Relay Rescue — Logit Difference Change (n=100)", "Logit Difference Change (blocked − clean)")
    for key, column in (
        ("full", "logit_diff_change_full"),
        ("keep_panl", "logit_diff_change_keepPANL"),
        ("keep_panl_plus_1", "logit_diff_change_keepPANL_plus_1"),
    ):
        ax.plot(x, [row[column] for row in table], marker="o", linewidth=1.7, color=COLORS[key], label=CONDITION_LABELS[key])
    ax.set_xticks(x); ax.axhline(0, color="#999999", linewidth=.8, linestyle=(0, (2, 2))); ax.legend(frameon=False)
    _save(fig, output / "figure2_logit_diff_change")

    fig, ax = _base_axis("Answer→PANL Relay Rescue — Recovery Proportion (n=100)", "Recovery Proportion (%)")
    ax.plot(x, [row["recovery_proportion_PANL_pct"] for row in table], marker="o", linewidth=1.7,
            color=COLORS["keep_panl"], label="Keep PANL→A")
    ax.plot(x, [row["recovery_proportion_PANL_plus_1_pct"] for row in table], marker="o", linewidth=1.7,
            color=COLORS["keep_panl_plus_1"], label="Keep PANL+1→A")
    ax.set_xticks(x); ax.axhline(0, color="#999999", linewidth=.8, linestyle=(0, (2, 2))); ax.legend(frameon=False)
    _save(fig, output / "figure3_recovery_proportion")

    lines = [
        "# Answer→PANL relay: three metrics", "",
        "Recovery proportion = (D_full - D_keep) / D_full × 100%.", "",
        "|Window|Token change: full / keep PANL / keep PANL+1|Logit diff change: full / keep PANL / keep PANL+1|Recovery: PANL / PANL+1|",
        "|---|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            f"|{row['window_start']}–{row['window_end']}|"
            f"{row['token_change_rate_full_pct']:.1f}% / {row['token_change_rate_keepPANL_pct']:.1f}% / {row['token_change_rate_keepPANL_plus_1_pct']:.1f}%|"
            f"{row['logit_diff_change_full']:+.4f} / {row['logit_diff_change_keepPANL']:+.4f} / {row['logit_diff_change_keepPANL_plus_1']:+.4f}|"
            f"{row['recovery_proportion_PANL_pct']:.1f}% / {row['recovery_proportion_PANL_plus_1_pct']:.1f}%|"
        )
    (output / "relay_three_metrics_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return table


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--input-csv", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv); generate(args.input_csv.resolve(), args.output_dir.resolve()); return 0


if __name__ == "__main__": raise SystemExit(main())
