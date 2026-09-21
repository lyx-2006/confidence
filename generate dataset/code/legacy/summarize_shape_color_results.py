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
                "answer": run.get("top1_answer"),
                "ground_truth_answer": run.get("ground_truth_answer"),
                "entropy": run.get("entropy"),
                "normalized_entropy": run.get("normalized_entropy"),
                "parse_success": run.get("parse_success"),
                "error": run.get("error"),
            }
        )
    return {
        "answer": calibration.get("top1_answer"),
        "ground_truth_answer": calibration.get("ground_truth_answer"),
        "entropy": calibration.get("entropy"),
        "normalized_entropy": calibration.get("normalized_entropy"),
        "correct_count": calibration.get("correct_count"),
        "all_correct": calibration.get("all_correct"),
        "parse_success": calibration.get("parse_success"),
        "runs": runs,
    }


def build_summary(dataset: list[dict[str, Any]], source: str) -> dict[str, Any]:
    if len(dataset) != 1 or not isinstance(dataset[0], dict):
        raise ValueError("dataset must contain one category object")
    items = dataset[0].get("items")
    if not isinstance(items, list):
        raise ValueError("dataset category has no items list")

    result_items = []
    for item in items:
        clues = item.get("image_clue")
        if not isinstance(clues, dict):
            raise ValueError(f"item {item.get('id')} has no image_clue object")
        groups: dict[str, Any] = {}
        for branch in ("consistent", "conflict"):
            branch_data = clues.get(branch)
            if not isinstance(branch_data, dict):
                raise ValueError(f"item {item.get('id')} is missing {branch} branch")
            for difficulty in ("easy", "hard"):
                calibration = branch_data.get(f"{difficulty}_calibration")
                image = branch_data.get(difficulty)
                if not isinstance(calibration, dict) or not isinstance(image, str):
                    raise ValueError(
                        f"item {item.get('id')} is missing {branch}/{difficulty} result"
                    )
                groups[f"{branch}_{difficulty}"] = {
                    "image": image,
                    **_compact_calibration(calibration),
                }
        question = item.get("question")
        if isinstance(question, dict):
            question = question.get("text", question)
        result_items.append(
            {
                "id": item.get("id"),
                "question": question,
                "answer": item.get("answer"),
                "conflict_answer": item.get("conflict_ans"),
                "groups": groups,
            }
        )
    return {
        "source_dataset": source,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "item_count": len(result_items),
        "group_count": len(result_items) * 4,
        "items": result_items,
    }


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
