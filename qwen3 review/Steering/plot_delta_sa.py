from __future__ import annotations

import argparse
import json
from pathlib import Path


def make_plot(statistics: Path, output_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = json.loads(statistics.read_text(encoding="utf-8"))
    rows = payload["cells"]
    positions = ["PANL", "LAT", "CLE"]
    layers = sorted({int(row["layer"]) for row in rows})
    alphas = [-5.0, -2.0, 0.0, 2.0, 5.0]
    colors = {a: c for a, c in zip(alphas, ("#d62728", "#ff7f0e", "#7f7f7f", "#2ca02c", "#1f77b4"))}
    lookup = {(row["position"], int(row["layer"]), float(row["alpha"])): row for row in rows}
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for position in positions:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for alpha in alphas:
            values = [lookup[(position, layer, alpha)]["delta_soft_sa_mean"] for layer in layers]
            ax.plot(layers, values, marker="o", linewidth=2, color=colors[alpha], label=f"α={alpha:g}")
        ax.axhline(0, color="black", linewidth=.8, alpha=.65)
        ax.set_xlabel("zero-based decoder layer")
        ax.set_ylabel("Δ soft-SA")
        ax.set_title(f"Qwen3-VL Steering: {position}")
        ax.set_xticks(layers)
        ax.grid(axis="y", alpha=.2)
        ax.legend(frameon=False, ncol=5)
        fig.tight_layout()
        path = output_dir / f"delta_sa_{position.lower()}.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, position in zip(axes, positions):
        for alpha in alphas:
            values = [lookup[(position, layer, alpha)]["delta_soft_sa_mean"] for layer in layers]
            ax.plot(layers, values, marker="o", linewidth=1.8, color=colors[alpha], label=f"α={alpha:g}")
        ax.axhline(0, color="black", linewidth=.8, alpha=.65)
        ax.set_title(position)
        ax.set_xlabel("layer")
        ax.set_xticks(layers)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Δ soft-SA")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Qwen3-VL Steering Δ soft-SA by layer", y=1.02)
    fig.tight_layout()
    combined = output_dir / "delta_sa_by_layer_all_positions.png"
    fig.savefig(combined, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(combined)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in make_plot(args.statistics, args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
