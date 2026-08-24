from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from confidence_test.runtime_imports import load_runtime
from dp_SA.attention_block.config import INFERENCE_PATH, MIDPOINTS, MODEL_PATH
from dp_SA.attention_block.run import _forward, _margin
from dp_SA.io_utils import append_jsonl, atomic_json, atomic_jsonl, canonical_hash, load_jsonl, sha256_file
from dp_SA.soft_score import class_token_ids
from layer_metacognition.model_adapter import resolve_language_modules

from .analyze import _spearman
from .core import MINIMAL_PROMPT_TEMPLATE, locate_minimal_positions, prepare_minimal_case


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = ROOT / "dp_SA" / "outputs" / "steering" / "test_manifest.jsonl"
DEFAULT_OUTPUT_PARENT = Path(__file__).resolve().parent / "outputs"


def select_remaining(rows: Iterable[dict[str, Any]], excluded_case_ids: set[str]) -> list[dict[str, Any]]:
    selected = [row for row in rows if str(row["case_id"]) not in excluded_case_ids]
    if len({str(row["case_id"]) for row in selected}) != len(selected):
        raise ValueError("Remaining clean-screen cases are not case-unique")
    if len({str(row["item_id"]) for row in selected}) != len(selected):
        raise ValueError("Remaining clean-screen cases are not item-unique")
    return sorted(
        selected,
        key=lambda row: (
            str(row["test_side"]),
            int(row.get("selection_rank", 10**9)),
            str(row["case_id"]),
        ),
    )


def select_clean_candidates(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible = [row for row in rows if int(row["clean_class"]) == int(row["full_clean_class"])]
    eligible.sort(key=lambda row: (str(row["test_side"]), float(row["abs_soft_sa_difference"]), str(row["case_id"])))
    by_side = {
        side: [row for row in eligible if row["test_side"] == side]
        for side in ("image_side", "text_side")
    }
    per_side = min(len(by_side["image_side"]), len(by_side["text_side"]))
    balanced = by_side["image_side"][:per_side] + by_side["text_side"][:per_side]
    balanced.sort(key=lambda row: (str(row["test_side"]), float(row["abs_soft_sa_difference"]), str(row["case_id"])))
    return eligible, balanced


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty clean screen")
    eligible, balanced = select_clean_candidates(rows)
    minimal_soft = [float(row["soft_sa_image_score"]) for row in rows]
    full_soft = [float(row["full_soft_sa_image_score"]) for row in rows]
    result = {
        "n": len(rows),
        "side_counts": dict(Counter(str(row["test_side"]) for row in rows)),
        "hard_agreement_count": len(eligible),
        "hard_agreement_rate": len(eligible) / len(rows),
        "soft_sa_spearman": _spearman(minimal_soft, full_soft),
        "minimal_class_distribution": {str(i): sum(int(row["clean_class"]) == i for row in rows) for i in range(9)},
        "full_class_distribution": {str(i): sum(int(row["full_clean_class"]) == i for row in rows) for i in range(9)},
        "mean_abs_soft_sa_difference": float(np.mean([float(row["abs_soft_sa_difference"]) for row in rows])),
        "eligible_by_side": dict(Counter(str(row["test_side"]) for row in eligible)),
        "balanced_candidate_count": len(balanced),
        "balanced_candidate_per_side": len(balanced) // 2,
        "selection_rule": (
            "hard minimal/full clean-class agreement; rank within side by absolute soft-SA difference; "
            "retain the maximum equal count from image_side and text_side"
        ),
    }
    return result


def _config(manifest: Path, discovery: Path, selection: Sequence[dict[str, Any]], smoke: bool) -> dict[str, Any]:
    implementation = [Path(__file__), Path(__file__).with_name("core.py")]
    values = {
        "format_version": 1,
        "diagnostic": "minimal_prompt_remaining_clean_screen",
        "model_path": str(MODEL_PATH.resolve()),
        "model_config_sha256": sha256_file(MODEL_PATH / "config.json"),
        "inference_path": str(INFERENCE_PATH.resolve()),
        "seed": 42,
        "smoke": smoke,
        "prompt": MINIMAL_PROMPT_TEMPLATE,
        "assistant_prefill": "**Source Attribution**:",
        "frozen_manifest_path": str(manifest.resolve()),
        "frozen_manifest_sha256": sha256_file(manifest),
        "discovery_output": str(discovery.resolve()),
        "selection_hash": canonical_hash(selection),
        "implementation_sha256": {str(path.resolve()): sha256_file(path) for path in implementation},
    }
    values["fingerprint"] = canonical_hash(values)
    return values


def run(output: Path, manifest: Path, discovery: Path, *, resume: bool, smoke: bool = False) -> dict[str, Any]:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    discovery_rows = load_jsonl(discovery / "selection_manifest.jsonl")
    excluded = {str(row["case_id"]) for row in discovery_rows}
    selection = select_remaining(load_jsonl(manifest), excluded)
    if len(selection) != 70 or Counter(row["test_side"] for row in selection) != {"image_side": 35, "text_side": 35}:
        raise ValueError(f"Expected untouched 35+35 cases, got {len(selection)}: {Counter(row['test_side'] for row in selection)}")
    if smoke:
        selection = [
            row
            for side in ("image_side", "text_side")
            for row in [candidate for candidate in selection if candidate["test_side"] == side][:2]
        ]

    config = _config(manifest, discovery, selection, smoke)
    config_path = output / "run_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous.get("fingerprint") != config["fingerprint"]:
            raise ValueError("Clean-screen fingerprint changed; refusing resume")
        if not resume:
            raise FileExistsError(f"Output already exists: {output}; use --resume")
    else:
        atomic_json(config_path, config)
        atomic_jsonl(output / "selection_manifest.jsonl", selection)

    clean_path = output / "clean_results.jsonl"
    spans_path = output / "token_spans.jsonl"
    failures_path = output / "failures.jsonl"
    for path in (clean_path, spans_path, failures_path):
        path.touch(exist_ok=True)
    clean = {row["case_id"]: row for row in load_jsonl(clean_path)}
    spans = {row["case_id"]: row for row in load_jsonl(spans_path)}

    runtime = load_runtime(INFERENCE_PATH)
    inference = runtime.QwenVLInference(str(MODEL_PATH))
    modules = resolve_language_modules(inference.model)
    if modules.num_hidden_layers != 28:
        raise RuntimeError(f"Expected 28 language layers, found {modules.num_hidden_layers}")
    tokenizer = getattr(inference.processor, "tokenizer", inference.processor)
    digit_ids = class_token_ids(tokenizer)
    decoded = [tokenizer.decode([token], skip_special_tokens=False, clean_up_tokenization_spaces=False) for token in digit_ids]
    if decoded != list("012345678"):
        raise RuntimeError(f"Classes are not single-token digits: {decoded}")

    started = time.time()
    try:
        for ordinal, row in enumerate(selection, start=1):
            if row["case_id"] in clean:
                continue
            try:
                rendered, inputs, prompt = prepare_minimal_case(inference, row)
                located = locate_minimal_positions(
                    tokenizer, rendered, inputs, row["phase0_raw_answer"], row["question"], row["text_clue"]
                )
                logits, score = _forward(inference.model, inputs, located["SAC"], digit_ids)
                target = int(score["argmax_hard_class"])
                full_class = int(row["argmax_hard_class"])
                full_soft = float(row["soft_sa_image_score"])
                result = {
                    "case_id": row["case_id"],
                    "item_id": str(row["item_id"]),
                    "test_side": row["test_side"],
                    "condition": row["condition"],
                    "selection_rank": int(row["selection_rank"]),
                    "prompt_hash": canonical_hash(prompt),
                    "raw_answer": row["phase0_raw_answer"],
                    "class_logits": logits,
                    **score,
                    "clean_class": target,
                    "clean_margin": _margin(logits, target),
                    "full_clean_class": full_class,
                    "full_soft_sa_image_score": full_soft,
                    "hard_agreement": target == full_class,
                    "abs_soft_sa_difference": abs(float(score["soft_sa_image_score"]) - full_soft),
                }
                span_row = {
                    "case_id": row["case_id"],
                    "item_id": str(row["item_id"]),
                    "test_side": row["test_side"],
                    **located,
                    "FULL_PANL_TO_SAC_DISTANCE": int(row["positions"]["P1_SAC"]["processed_index"])
                    - int(row["positions"]["P1_PANL"]["processed_index"]),
                }
                append_jsonl(clean_path, result)
                append_jsonl(spans_path, span_row)
                clean[row["case_id"]] = result
                spans[row["case_id"]] = span_row
                elapsed = time.time() - started
                remaining = len(selection) - len(clean)
                atomic_json(output / "progress.json", {
                    "status": "running",
                    "completed": len(clean),
                    "expected": len(selection),
                    "failed": len(load_jsonl(failures_path)),
                    "elapsed_seconds": elapsed,
                    "estimated_remaining_seconds": elapsed / max(1, len(clean)) * remaining,
                })
                del inputs
            except Exception as exc:
                append_jsonl(failures_path, {"case_id": row.get("case_id"), "error_type": type(exc).__name__, "error": str(exc)})
                raise

        ordered = [clean[row["case_id"]] for row in selection]
        eligible, balanced = select_clean_candidates(ordered)
        summary = summarize(ordered)
        atomic_json(output / "clean_summary.json", summary)
        atomic_jsonl(output / "eligible_hard_match_manifest.jsonl", eligible)
        atomic_jsonl(output / "selected_balanced_manifest.jsonl", balanced)
        completion = {
            "status": "complete",
            "clean": len(clean),
            "failures": len(load_jsonl(failures_path)),
            "elapsed_seconds": time.time() - started,
            "estimated_remaining_seconds": 0.0,
        }
        atomic_json(output / "completion.json", completion)
        atomic_json(output / "progress.json", completion)
        return completion
    finally:
        del inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--discovery-output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_dir or DEFAULT_OUTPUT_PARENT / time.strftime("clean_screen_seed42_%Y%m%dT%H%M%SZ", time.gmtime())
    run(output, args.manifest.resolve(), args.discovery_output.resolve(), resume=args.resume, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
