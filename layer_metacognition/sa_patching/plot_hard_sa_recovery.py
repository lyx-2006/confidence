"""Plot midpoint-based hard-SA recovery with low/high cohorts pooled."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean

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
POSITION_STYLES = {
    "ac": ("#4C78A8", "o"),
    "panl": ("#F58518", "s"),
    "sac": ("#54A24B", "^"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_latest_records(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[record["intervention_key"]] = record
    return records


def load_midpoints(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    classes = config["source_attribution_classes"]
    midpoints = config["source_attribution_midpoints"]
    if len(classes) != len(midpoints) or len(set(classes)) != len(classes):
        raise ValueError("invalid source-attribution class/midpoint configuration")
    mapping = {str(label): float(value) for label, value in zip(classes, midpoints)}
    if not all(math.isfinite(value) for value in mapping.values()):
        raise ValueError("source-attribution midpoints must be finite")
    return mapping


def midpoint_recovery(record: dict, midpoints: dict[str, float]) -> float | None:
    try:
        clean = midpoints[str(record["clean_parsed_SA"])]
        corrupt = midpoints[str(record["corrupt_parsed_SA"])]
        patched = midpoints[str(record["patched_parsed_SA"])]
    except (KeyError, TypeError):
        return None
    denominator = clean - corrupt
    if denominator == 0.0:
        return None
    recovery = (patched - corrupt) / denominator
    return recovery if math.isfinite(recovery) else None


def aggregate(
    records: dict[str, dict], midpoints: dict[str, float]
) -> dict[tuple[str, str, int], float]:
    values: defaultdict[tuple[str, str, int], list[float]] = defaultdict(list)
    for record in records.values():
        if record.get("status") != "completed":
            continue
        corruption = record.get("corruption_type")
        position = record.get("position")
        layer = record.get("layer")
        if (
            corruption not in CORRUPTIONS
            or position not in POSITION_LAYERS
            or layer not in POSITION_LAYERS[position]
        ):
            continue
        recovery = midpoint_recovery(record, midpoints)
        if recovery is not None:
            values[(corruption, position, layer)].append(100.0 * recovery)
    # Hard labels are discrete and most patches leave the label unchanged, so a
    # median would collapse nearly every cell to zero.  The mean retains both
    # the frequency and midpoint-weighted magnitude of categorical movement.
    return {key: fmean(group) for key, group in values.items()}


def plot_corruption(
    corruption: str,
    label: str,
    aggregated: dict[tuple[str, str, int], float],
    output_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    plotted_values: list[float] = []
    for position, layers in POSITION_LAYERS.items():
        points = [
            (layer, aggregated[(corruption, position, layer)])
            for layer in layers
            if (corruption, position, layer) in aggregated
        ]
        if not points:
            continue
        xs, ys = zip(*points)
        plotted_values.extend(ys)
        color, marker = POSITION_STYLES[position]
        ax.plot(
            xs,
            ys,
            color=color,
            marker=marker,
            linewidth=2.3,
            markersize=7,
            label=position.upper(),
        )

    max_abs = max((abs(value) for value in plotted_values), default=1.0)
    limit = max(1.0, 1.18 * max_abs)
    ax.set_ylim(-limit, limit)
    ax.set_xticks((12, 16, 18, 20, 24))
    ax.axhline(0.0, color="#222222", linewidth=1.2, alpha=0.85)
    ax.grid(axis="both", color="#D9D9D9", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Midpoint-based Hard SA recovery (mean, %)")
    ax.set_title(f"{label}: Hard SA recovery toward clean (Low + High pooled)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    output_path = output_dir / f"hard_sa_recovery_{corruption}.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_latest_records(args.results)
    midpoints = load_midpoints(args.config)
    aggregated = aggregate(records, midpoints)
    for corruption, label in CORRUPTIONS.items():
        print(plot_corruption(corruption, label, aggregated, args.output_dir))


if __name__ == "__main__":
    main()
