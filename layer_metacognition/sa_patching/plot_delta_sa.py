"""Plot direction-aligned soft-SA deltas from SA patching results."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt


CORRUPTIONS = {
    "image_only": "Image only",
    "text_only": "Text only",
    "image_text": "Image + Text",
    "answer_only": "Answer only",
    "all": "All",
}
POSITION_LAYERS = {
    "ac": (12, 16, 20),
    "panl": (16, 18, 20),
    "sac": (18, 20, 24),
}
POSITION_COLORS = {
    "ac": "#4C78A8",
    "panl": "#F58518",
    "sac": "#54A24B",
}
GROUP_STYLES = {
    "low": ("-", "o"),
    "high": ("--", "s"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_latest_records(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records[record["intervention_key"]] = record
    return records


def aligned_soft_delta(record: dict) -> float | None:
    clean = record.get("clean_soft_score")
    corrupt = record.get("corrupt_soft_score")
    patched = record.get("patched_soft_score")
    if not all(
        isinstance(value, (int, float)) and math.isfinite(value)
        for value in (clean, corrupt, patched)
    ):
        return None
    if clean == corrupt:
        return None
    direction = 1.0 if clean > corrupt else -1.0
    return direction * (patched - corrupt)


def aggregate(records: dict[str, dict]) -> dict[tuple[str, str, int, str], float]:
    values: defaultdict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    for record in records.values():
        if record.get("status") != "completed":
            continue
        key = (
            record.get("corruption_type"),
            record.get("position"),
            record.get("layer"),
            record.get("baseline_sa_group"),
        )
        if (
            key[0] not in CORRUPTIONS
            or key[1] not in POSITION_LAYERS
            or key[2] not in POSITION_LAYERS[key[1]]
            or key[3] not in GROUP_STYLES
        ):
            continue
        delta = aligned_soft_delta(record)
        if delta is not None:
            values[key].append(delta)
    return {key: median(group) for key, group in values.items()}


def plot_corruption(
    corruption: str,
    label: str,
    aggregated: dict[tuple[str, str, int, str], float],
    output_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    plotted_values: list[float] = []

    for position, layers in POSITION_LAYERS.items():
        for group, (linestyle, marker) in GROUP_STYLES.items():
            points = [
                (layer, aggregated[(corruption, position, layer, group)])
                for layer in layers
                if (corruption, position, layer, group) in aggregated
            ]
            if not points:
                continue
            xs, ys = zip(*points)
            plotted_values.extend(ys)
            ax.plot(
                xs,
                ys,
                color=POSITION_COLORS[position],
                linestyle=linestyle,
                marker=marker,
                linewidth=2.1,
                markersize=6.5,
                label=f"{position.upper()} - {group.capitalize()} SA",
            )

    max_abs = max((abs(value) for value in plotted_values), default=0.01)
    limit = max(0.01, max_abs * 1.18)
    ax.set_ylim(-limit, limit)
    ax.set_xticks((12, 16, 18, 20, 24))
    ax.axhline(0.0, color="#222222", linewidth=1.2, alpha=0.85)
    ax.grid(axis="both", color="#D9D9D9", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Direction-aligned Delta Soft SA (median)")
    ax.set_title(f"{label}: activation patching toward clean SA")
    ax.legend(ncol=2, frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    output_path = output_dir / f"delta_soft_sa_{corruption}.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_latest_records(args.results)
    aggregated = aggregate(records)
    for corruption, label in CORRUPTIONS.items():
        path = plot_corruption(corruption, label, aggregated, args.output_dir)
        print(path)


if __name__ == "__main__":
    main()
