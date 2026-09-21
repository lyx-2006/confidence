from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from .config import STEERING_OUTPUT_ROOT, VARIANTS
from .layout import ensure_output_layout


def plot_position(
    summary_path: Path,
    output_path: Path,
    variant: str,
    position: str,
) -> None:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    cells = [
        row for row in payload["cells"]
        if row["group"] == "overall" and row["variant"] == variant
        and row["position"] == position
    ]
    alphas = sorted({float(row["alpha"]) for row in cells})
    palette = plt.get_cmap("coolwarm")
    alpha_colors = {
        alpha: palette(index / max(1, len(alphas) - 1))
        for index, alpha in enumerate(alphas)
    }
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for alpha in alphas:
        rows = sorted(
            (row for row in cells if float(row["alpha"]) == alpha),
            key=lambda row: int(row["layer"]),
        )
        axis.plot(
            [row["layer"] for row in rows],
            [row["delta_image_score_mean"] for row in rows],
            color=alpha_colors[alpha], marker="o", linewidth=2,
            label=f"α={alpha:g}",
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_title(f"{variant}: {position}")
    axis.set_xlabel("Layer")
    axis.set_ylabel("Mean ΔSA")
    axis.set_xticks(sorted({int(row["layer"]) for row in cells}))
    axis.grid(alpha=0.2)
    axis.legend(ncol=len(alphas), loc="upper center", fontsize=9)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_root(root: Path, variants: tuple[str, ...] = VARIANTS) -> None:
    ensure_output_layout(root, variants)
    for variant in variants:
        summary_path = root / variant / "tables" / "summary.json"
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        positions = [
            value for value in ("LAT", "PANL", "PANL+1", "CLE", "SAC")
            if any(row["position"] == value for row in payload["cells"])
        ]
        for position in positions:
            plot_position(
                summary_path,
                root / variant / "figures" / f"layer_curves_{position}.png",
                variant,
                position,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot Qwen3 chat-fiveway steering results")
    parser.add_argument("--steering-root", type=Path, default=STEERING_OUTPUT_ROOT)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = parser.parse_args()
    plot_root(args.steering_root, tuple(args.variants))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
