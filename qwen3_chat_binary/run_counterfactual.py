from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from dp_SA.io_utils import append_jsonl, atomic_json, canonical_hash, load_jsonl, sha256_file

from .capture import acquire_pid, model_fingerprint
from .config import COUNTERFACTUAL_OUTPUT_ROOT, DATASET_PATH, MODEL_PATH
from .contracts import ensure_fingerprinted_config
from .counterfactual_common import BehaviorRunner, cma_scores
from .dataset import load_conflict_cases


CELL_NAMES = ("original", "image_counterfactual", "text_counterfactual", "joint_counterfactual")


def run_config(root: Path, model_path: Path, case_ids: Sequence[str] | None) -> dict[str, Any]:
    manifest = root / "tables" / "construction_manifest.jsonl"
    prepare_config = root / "progress" / "prepare_config.json"
    return {
        "format_version": 1, "experiment": "qwen3_counterfactual_four_cell",
        "model": str(model_path.resolve()), "model_fingerprint": model_fingerprint(model_path),
        "dataset": str(DATASET_PATH.resolve()), "dataset_sha256": sha256_file(DATASET_PATH),
        "construction_manifest": str(manifest), "construction_manifest_sha256": sha256_file(manifest),
        "prepare_fingerprint": json.loads(prepare_config.read_text())["fingerprint"],
        "case_ids": list(case_ids) if case_ids else None,
        "cells": list(CELL_NAMES), "fixed_answer_source": "saved_phase0",
        "probability_definition": "12-color restricted softmax fixed-answer probability",
        "logit_definition": "raw fixed-answer vocabulary logit",
    }


def run_experiment(
    *, output_root: Path = COUNTERFACTUAL_OUTPUT_ROOT, model_path: Path = MODEL_PATH,
    case_ids: Sequence[str] | None = None, resume: bool = False,
) -> dict[str, Any]:
    root, model_path = output_root.resolve(), model_path.resolve()
    pid = root / "progress" / "run.pid"
    acquire_pid(pid, "Counterfactual four-cell run")
    try:
        config = ensure_fingerprinted_config(
            root / "progress" / "run_config.json", run_config(root, model_path, case_ids),
            resume=resume, label="Counterfactual run",
        )
        manifest = [row for row in load_jsonl(root / "tables" / "construction_manifest.jsonl")]
        if case_ids:
            requested = set(case_ids)
            manifest = [row for row in manifest if row["case_id"] in requested]
        cases = {case.case_id: case for case in load_conflict_cases(DATASET_PATH)}
        phase0_rows = {
            value["case_id"]: value for value in load_jsonl(
                root.parent / "Capture" / "tables" / "phase0_results.jsonl"
            )
        }
        results_path = root / "tables" / "four_cell_results.jsonl"
        existing = {row["case_id"]: row for row in load_jsonl(results_path)}
        runner = BehaviorRunner(model_path)
        started = time.time()
        for row in manifest:
            case_id = row["case_id"]
            if case_id in existing:
                continue
            if row.get("status") != "completed":
                result = {
                    "status": "skipped", "case_id": case_id,
                    "reason": "construction_not_completed",
                    "construction_error": row.get("error"),
                }
                append_jsonl(results_path, result)
                existing[case_id] = result
                continue
            case = cases[case_id]
            fixed = row["fixed_answer"]
            inputs = {
                "original": (row["original_image"], row["original_text"]),
                "image_counterfactual": (row["render_audit"]["counterfactual_image"], row["original_text"]),
                "text_counterfactual": (row["original_image"], row["counterfactual_text"]),
                "joint_counterfactual": (row["render_audit"]["counterfactual_image"], row["counterfactual_text"]),
            }
            try:
                cells = {
                    name: runner.multimodal(case, clue, image, fixed)
                    for name, (image, clue) in inputs.items()
                }
                phase0_raw = phase0_rows[case_id]["phase0_raw_output"]
                reproduction = cells["original"]["actual_output"] == phase0_raw
                probabilities = [cells[name]["target_probability"] for name in CELL_NAMES]
                logits = [cells[name]["target_logit"] for name in CELL_NAMES]
                result = {
                    "status": "completed" if reproduction else "failed",
                    "case_id": case_id, "pair_type": row["pair_type"],
                    "shape": row["shape"], "fixed_answer": fixed,
                    "answer_source": (
                        "text" if fixed == row["text_answer"] else
                        "image" if fixed == row["image_answer"] else "other"
                    ),
                    "third_color": row["third_color"],
                    "distractor_color_collision": row["render_audit"]["distractor_color_collision"],
                    "original_answer_reproduced": reproduction,
                    "saved_original_output": phase0_raw,
                    "cells": cells,
                    "cma_probability": cma_scores(*probabilities),
                    "cma_logit": cma_scores(*logits),
                    "image_only_probability_drop": probabilities[0] - probabilities[1],
                    "text_only_probability_drop": probabilities[0] - probabilities[2],
                    "cell_answer_categories": {
                        name: (
                            "fixed" if cells[name]["normalized_answer"] == fixed else
                            "third" if cells[name]["normalized_answer"] == row["third_color"] else "other"
                        ) for name in CELL_NAMES
                    },
                    "construction_fingerprint": canonical_hash(row),
                }
                if not reproduction:
                    result["error"] = {"type": "ReproductionMismatch", "message": "00 output differs from saved phase0"}
            except Exception as exc:
                result = {
                    "status": "failed", "case_id": case_id,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            append_jsonl(results_path, result)
            existing[case_id] = result
            atomic_json(root / "progress" / "run_progress.json", {
                "total": len(manifest), "processed": len(existing),
                "counts": dict(Counter(value["status"] for value in existing.values())),
                "elapsed_seconds": time.time() - started,
            })
        counts = Counter(value["status"] for value in existing.values())
        summary = {
            "status": "complete", "total": len(manifest), "counts": dict(counts),
            "config_fingerprint": config["fingerprint"], "elapsed_seconds": time.time() - started,
        }
        atomic_json(root / "progress" / "run_summary.json", summary)
        return summary
    finally:
        if pid.exists() and pid.read_text().strip() == str(__import__("os").getpid()):
            pid.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run four-cell counterfactual behavior measurement")
    parser.add_argument("--output-root", type=Path, default=COUNTERFACTUAL_OUTPUT_ROOT)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    print(run_experiment(output_root=args.output_root, model_path=args.model_path, case_ids=args.case_ids, resume=args.resume))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
