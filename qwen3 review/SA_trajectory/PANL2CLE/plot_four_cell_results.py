from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def plot(*, output_root: Path, destination: Path | None = None) -> Path:
    output_root = Path(output_root)
    source = output_root / "tables" / "condition_summary.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row["group"] == "answer_equal_macro"]
    if not rows:
        raise ValueError(f"No answer_equal_macro rows found in {source}")

    destination = destination or output_root / "figures" / "four_cell_cleprobe_final.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    conditions = ("C0", "C1", "C2", "C3")
    labels = {
        "C0": "C0 clean PANL + clean CLE",
        "C1": "C1 steered PANL + steered CLE",
        "C2": "C2 steered PANL + clean CLE",
        "C3": "C3 clean PANL + steered CLE",
    }
    endpoints = (("cle_probe_sa", "CLE probe ΔSA"), ("final_soft_sa", "Final soft-SA ΔSA"))
    colors = {-5.0: "#2166ac", 5.0: "#b2182b"}
    styles = {14: "-", 16: "--", 18: ":"}
    fig, axes = plt.subplots(2, 4, figsize=(19, 8), sharex=True, sharey="row")
    for row_index, (endpoint, ylabel) in enumerate(endpoints):
        endpoint_rows = [row for row in rows if row["endpoint"] == endpoint]
        for col_index, condition in enumerate(conditions):
            axis = axes[row_index, col_index]
            for panl_layer in (14, 16, 18):
                for alpha in (-5.0, 5.0):
                    points = [
                        row for row in endpoint_rows
                        if row["condition"] == condition
                        and int(row["panl_layer"]) == panl_layer
                        and float(row["alpha"]) == alpha
                    ]
                    points.sort(key=lambda row: int(row["cle_layer"]))
                    if not points:
                        continue
                    x = [int(row["cle_layer"]) for row in points]
                    y = [float(row["mean_delta"]) for row in points]
                    low = [float(row["ci95_low"]) for row in points]
                    high = [float(row["ci95_high"]) for row in points]
                    color = colors[alpha]
                    style = styles[panl_layer]
                    label = f"PANL L{panl_layer}, α={alpha:g}"
                    axis.plot(x, y, color=color, linestyle=style, marker="o", linewidth=1.8, label=label)
                    axis.fill_between(x, low, high, color=color, alpha=0.07)
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.set_title(labels[condition], fontsize=10)
            axis.set_xticks((15, 17, 19, 21))
            axis.grid(axis="y", alpha=0.2)
            if col_index == 0:
                axis.set_ylabel(ylabel)
            if row_index == 1:
                axis.set_xlabel("CLE layer")

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Qwen3-VL PANL→CLE four-cell results (answer-equal macro; mean Δ from C0)", y=1.06)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(destination, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    print(plot(output_root=args.output_root, destination=args.destination))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
