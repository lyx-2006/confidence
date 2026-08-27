#!/usr/bin/env python3
"""Create a compact answer/entropy summary from a generated dataset."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _compact_calibration(calibration: dict[str, Any]) -> dict[str, Any]:
    """Keep answer and entropy fields while omitting large class-probability maps."""
    runs = []
    for run in calibration.get("runs", []):
        runs.append(
            {
                "answer": run.get("top1_answer", run.get("normalized_answer", run.get("answer"))),
                "ground_truth_answer": run.get("ground_truth_answer"),
                "entropy": run.get("entropy"),
                "raw_entropy": run.get("raw_entropy", run.get("raw_answer_entropy")),
                "entropy_score": run.get("entropy_score"),
                "normalized_entropy": run.get("normalized_entropy"),
                "restricted_top1": run.get("restricted_top1"),
                "parse_success": run.get("parse_success"),
                "error": run.get("error"),
            }
        )
    result = {
        "answer": calibration.get("top1_answer", calibration.get("normalized_answer", calibration.get("answer"))),
        "ground_truth_answer": calibration.get("ground_truth_answer"),
        "entropy": calibration.get("entropy"),
        "raw_entropy": calibration.get("raw_entropy", calibration.get("raw_answer_entropy")),
        "entropy_score": calibration.get("entropy_score"),
        "normalized_entropy": calibration.get("normalized_entropy"),
        "restricted_top1": calibration.get("restricted_top1"),
        "correct_count": calibration.get("correct_count"),
        "all_correct": calibration.get("all_correct"),
        "parse_success": calibration.get("parse_success"),
        "runs": runs,
    }


def _compact_v2_variant(variant: dict[str, Any]) -> dict[str, Any]:
    calibration = variant.get("calibration")
    value = {
        "variant_index": variant.get("variant_index", 1),
        "image": variant.get("image"),
        "layout": variant.get("layout"),
        "target_mask": variant.get("target_mask"),
        "occluder_mask": variant.get("occluder_mask"),
        "seed": variant.get("seed"),
        "rotation": variant.get("rotation"),
        "artifact_sha256": variant.get("artifact_sha256"),
        "similarity_check": variant.get("similarity_check"),
    }
    if isinstance(calibration, dict):
        value.update(_compact_calibration(calibration))
    if isinstance(variant.get("entropy_check"), dict):
        value["entropy_check"] = variant["entropy_check"]
    return value


def build_summary(dataset: Any, source: str) -> dict[str, Any]:
    is_v2 = isinstance(dataset, dict) and dataset.get("schema_version") == "shape_color_dataset.v2"
    if is_v2:
        items = dataset.get("items")
        category = dataset.get("category", "colour")
    elif isinstance(dataset, list) and len(dataset) == 1 and isinstance(dataset[0], dict):
        items = dataset[0].get("items")
        category = dataset[0].get("category", "colour")
    else:
        raise ValueError("dataset must contain one category object or a shape_color_dataset.v2 object")
    if not isinstance(items, list):
        raise ValueError("dataset category has no items list")

    result_items = []
    for item in items:
        clues = item.get("image_clue")
        if not isinstance(clues, dict):
            raise ValueError(f"item {item.get('id')} has no image_clue object")
        groups: dict[str, Any] = {}
        branches = tuple(dataset.get("branches", ("consistent", "conflict"))) if is_v2 else ("consistent", "conflict")
        for branch in branches:
            branch_data = clues.get(branch)
            if not isinstance(branch_data, dict):
                continue
            for difficulty in ("easy", "hard"):
                raw = branch_data.get(difficulty)
                if is_v2 and isinstance(raw, list):
                    groups[f"{branch}_{difficulty}"] = [_compact_v2_variant(value) for value in raw if isinstance(value, dict)]
                    continue
                calibration = branch_data.get(f"{difficulty}_calibration")
                image = raw
                if not isinstance(calibration, dict) or not isinstance(image, str):
                    raise ValueError(f"item {item.get('id')} is missing {branch}/{difficulty} result")
                groups[f"{branch}_{difficulty}"] = {"image": image, **_compact_calibration(calibration)}
        question = item.get("question")
        if isinstance(question, dict):
            question = question.get("text", question)
        result_items.append(
            {
                "id": item.get("id"),
                "question": question,
                "answer": item.get("answer"),
                "conflict_answer": item.get("conflict_ans", item.get("conflict_answer")),
                "groups": groups,
            }
        )
    return {
        "source_dataset": source,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "item_count": len(result_items),
        "group_count": sum(
            len(value) if isinstance(value, list) else 1
            for item in result_items for value in item.get("groups", {}).values()
        ),
        "category": category,
        "items": result_items,
    }
    if is_v2:
        result["schema_version"] = "shape_color_dataset.summary.v2"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    dataset = json.loads(args.input.read_text(encoding="utf-8"))
    summary = build_summary(dataset, str(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {summary['item_count']} items / {summary['group_count']} groups to {args.output}")


if __name__ == "__main__":
    main()
