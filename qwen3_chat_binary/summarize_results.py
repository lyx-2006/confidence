from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from dp_SA.io_utils import atomic_json, load_jsonl

from .config import STEERING_OUTPUT_ROOT, VARIANTS
from .layout import ensure_output_layout, steering_predictions_path


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _stats(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    deltas = [float(row["delta_image_attribution_score"]) for row in rows]
    signed_deltas = [float(row["delta_signed_attribution_score"]) for row in rows]
    mean = sum(deltas) / len(deltas)
    signed_mean = sum(signed_deltas) / len(signed_deltas)
    return {
        "n": len(rows),
        "delta_image_score_mean": mean,
        "delta_image_score_std": math.sqrt(sum((value - mean) ** 2 for value in deltas) / len(deltas)),
        "steered_image_score_mean": sum(float(row["steered_image_attribution_score"]) for row in rows) / len(rows),
        "delta_signed_score_mean": signed_mean,
        "delta_signed_score_std": math.sqrt(
            sum((value - signed_mean) ** 2 for value in signed_deltas) / len(signed_deltas)
        ),
        "label_change_rate": sum(bool(row["label_changed"]) for row in rows) / len(rows),
        "side_change_rate": sum(bool(row["side_changed"]) for row in rows) / len(rows),
        "label_probability_mass_mean": sum(float(row["steered_label_probability_mass"]) for row in rows) / len(rows),
        "hook_applied_counts": sorted({
            int(row["hook_diagnostics"]["steering_applied_count"]) for row in rows
        }),
        "alpha_zero_max_abs_delta": (
            max(abs(value) for value in deltas) if float(rows[0]["alpha"]) == 0.0 else None
        ),
    }


def summarize(predictions: Path | Sequence[Path]) -> dict[str, Any]:
    paths = [predictions] if isinstance(predictions, Path) else list(predictions)
    rows = [
        row for path in paths for row in load_jsonl(path)
        if row.get("status") == "completed"
    ]
    groups: dict[tuple[str, str, int, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        base = (row["variant"], row["position"], int(row["layer"]), float(row["alpha"]))
        groups[base + ("overall",)].append(row)
        groups[base + (str(row["test_side"]),)].append(row)
    cells = [
        {
            "variant": key[0], "position": key[1], "layer": key[2],
            "alpha": key[3], "group": key[4], **_stats(values),
        }
        for key, values in sorted(groups.items())
    ]

    by_pair: dict[tuple[str, str, int, float], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row["case_id"], row["position"], int(row["layer"]), float(row["alpha"]))
        by_pair[key][row["variant"]] = row
    paired_groups: dict[tuple[str, int, float], list[float]] = defaultdict(list)
    for (_case_id, position, layer, alpha), variants in by_pair.items():
        if set(variants) >= {"native_boundary", "explicit_newline"}:
            paired_groups[(position, layer, alpha)].append(
                float(variants["explicit_newline"]["delta_image_attribution_score"])
                - float(variants["native_boundary"]["delta_image_attribution_score"])
            )
    paired = []
    for (position, layer, alpha), values in sorted(paired_groups.items()):
        mean = sum(values) / len(values)
        paired.append({
            "position": position, "layer": layer, "alpha": alpha, "n": len(values),
            "explicit_minus_native_delta_mean": mean,
            "explicit_minus_native_delta_std": math.sqrt(
                sum((value - mean) ** 2 for value in values) / len(values)
            ),
        })
    return {
        "sources": [str(path) for path in paths], "completed_predictions": len(rows),
        "cell_count": len(cells), "cells": cells, "paired_cells": paired,
    }


def summarize_root(root: Path, variants: Sequence[str] = VARIANTS) -> dict[str, Any]:
    ensure_output_layout(root, variants)
    paths = [steering_predictions_path(root, variant) for variant in variants]
    result = summarize(paths)
    atomic_json(root / "tables" / "summary.json", result)
    _write_csv(root / "tables" / "cell_summary.csv", result["cells"])
    _write_csv(root / "tables" / "paired_variant_summary.csv", result["paired_cells"])
    for variant, path in zip(variants, paths):
        variant_result = summarize(path)
        atomic_json(root / variant / "tables" / "summary.json", variant_result)
        _write_csv(root / variant / "tables" / "cell_summary.csv", variant_result["cells"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Qwen3 chat-fiveway steering results")
    parser.add_argument("--steering-root", type=Path, default=STEERING_OUTPUT_ROOT)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = parser.parse_args()
    result = summarize_root(args.steering_root, args.variants)
    print(json.dumps({
        "completed_predictions": result["completed_predictions"],
        "cell_count": result["cell_count"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
