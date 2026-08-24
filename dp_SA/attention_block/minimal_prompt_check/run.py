from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Sequence

import torch

from confidence_test.runtime_imports import load_runtime
from dp_SA.attention_block.config import INFERENCE_PATH, MIDPOINTS, MODEL_PATH, ROW_SUM_TOLERANCE
from dp_SA.attention_block.masking import AttentionBlockContext, AttentionEdges
from dp_SA.attention_block.run import _forward, _margin
from dp_SA.io_utils import append_jsonl, atomic_json, atomic_jsonl, canonical_hash, load_jsonl, sha256_file
from dp_SA.soft_score import class_token_ids
from layer_metacognition.model_adapter import resolve_language_modules

from .core import CONDITIONS, MINIMAL_PROMPT_TEMPLATE, WINDOWS, load_selection, locate_minimal_positions, prepare_minimal_case

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = ROOT / "dp_SA" / "outputs" / "steering" / "test_manifest.jsonl"
DEFAULT_OUTPUT_PARENT = Path(__file__).resolve().parent / "outputs"


def _default_output(smoke: bool) -> Path:
    label = "smoke" if smoke else "formal"
    return DEFAULT_OUTPUT_PARENT / time.strftime(f"{label}_seed42_%Y%m%dT%H%M%SZ", time.gmtime())


def _config(manifest: Path, selection: list[dict[str, Any]], smoke: bool) -> dict[str, Any]:
    implementation = [Path(__file__), Path(__file__).with_name("core.py"), Path(__file__).with_name("analyze.py"),
                      Path(__file__).parents[1] / "masking.py"]
    values = {
        "format_version": 1,
        "diagnostic": "minimal_prompt_direct_sac_to_panl",
        "model_path": str(MODEL_PATH.resolve()),
        "model_config_sha256": sha256_file(MODEL_PATH / "config.json"),
        "inference_path": str(INFERENCE_PATH.resolve()),
        "attention_backend": "eager",
        "seed": 42,
        "bootstrap_repeats": 2000,
        "per_side": 2 if smoke else 15,
        "smoke": smoke,
        "prompt": MINIMAL_PROMPT_TEMPLATE,
        "assistant_prefill": "**Source Attribution**:",
        "windows": WINDOWS,
        "conditions": CONDITIONS,
        "selection_hash": canonical_hash(selection),
        "frozen_manifest_path": str(manifest.resolve()),
        "frozen_manifest_sha256": sha256_file(manifest),
        "implementation_sha256": {str(path.resolve()): sha256_file(path) for path in implementation},
    }
    values["fingerprint"] = canonical_hash(values)
    return values


def _ensure_config(path: Path, config: dict[str, Any], resume: bool) -> None:
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("fingerprint") != config["fingerprint"]:
            raise ValueError("Minimal-prompt run fingerprint changed; refusing resume")
        if not resume:
            raise FileExistsError(f"Output already exists: {path.parent}; use --resume")
    else:
        atomic_json(path, config)


def _edges(condition: str, positions: dict[str, Any]) -> AttentionEdges:
    if condition == "sac_to_panl":
        return AttentionEdges.from_sets([positions["SAC"]], [positions["PANL"]])
    if condition == "sac_to_panl_plus_1":
        return AttentionEdges.from_sets([positions["SAC"]], [positions["PANL_PLUS_1"]])
    if condition == "empty_block_parity":
        return AttentionEdges(())
    raise ValueError(f"Unknown minimal-prompt condition: {condition}")


def run(output: Path, manifest: Path, *, smoke: bool, resume: bool) -> dict[str, Any]:
    output = output.resolve(); output.mkdir(parents=True, exist_ok=True)
    selection = load_selection(manifest, per_side=2 if smoke else 15)
    config = _config(manifest, selection, smoke)
    _ensure_config(output / "run_config.json", config, resume)
    selection_path = output / "selection_manifest.jsonl"
    if selection_path.exists():
        if canonical_hash(load_jsonl(selection_path)) != canonical_hash(selection):
            raise ValueError("Stored minimal selection changed")
    else:
        atomic_jsonl(selection_path, selection)

    runtime = load_runtime(INFERENCE_PATH)
    inference = runtime.QwenVLInference(str(MODEL_PATH))
    if getattr(inference.model.config, "_attn_implementation", None) != "eager":
        raise RuntimeError("Minimal diagnostic requires eager attention")
    modules = resolve_language_modules(inference.model)
    if modules.num_hidden_layers != 28:
        raise RuntimeError(f"Expected 28 language layers, found {modules.num_hidden_layers}")
    tokenizer = getattr(inference.processor, "tokenizer", inference.processor)
    digit_ids = class_token_ids(tokenizer)
    decoded_digits = [tokenizer.decode([token], skip_special_tokens=False, clean_up_tokenization_spaces=False) for token in digit_ids]
    if decoded_digits != list("012345678") or len(set(digit_ids)) != 9:
        raise RuntimeError(f"Classes are not nine unique single-token digits: {decoded_digits}")

    clean_path = output / "clean_results.jsonl"; block_path = output / "block_results.jsonl"
    spans_path = output / "token_spans.jsonl"; failures_path = output / "failures.jsonl"
    for path in (clean_path, block_path, spans_path, failures_path): path.touch(exist_ok=True)
    clean = {row["case_id"]: row for row in load_jsonl(clean_path)}
    spans = {row["case_id"]: row for row in load_jsonl(spans_path)}
    completed = {(row["case_id"], row["condition"], row["window_start"]) for row in load_jsonl(block_path)}
    recent: deque[float] = deque(maxlen=200); started = time.time(); done = 0
    total = len(selection) * (1 + len(CONDITIONS) * len(WINDOWS))
    try:
        for row in selection:
            try:
                rendered, inputs, prompt = prepare_minimal_case(inference, row)
                located = locate_minimal_positions(
                    tokenizer, rendered, inputs, row["phase0_raw_answer"], row["question"], row["text_clue"]
                )
                full_panl = int(row["positions"]["P1_PANL"]["processed_index"])
                full_sac = int(row["positions"]["P1_SAC"]["processed_index"])
                span_row = {
                    "case_id": row["case_id"], "item_id": str(row["item_id"]), "test_side": row["test_side"],
                    **located,
                    "FULL_PANL": full_panl, "FULL_SAC": full_sac,
                    "FULL_PANL_TO_SAC_DISTANCE": full_sac - full_panl,
                    "distance_reduction": (full_sac - full_panl) - located["PANL_TO_SAC_DISTANCE"],
                }
                if row["case_id"] in spans:
                    if canonical_hash(spans[row["case_id"]]) != canonical_hash(span_row):
                        raise RuntimeError(f"Minimal token positions changed for {row['case_id']}")
                else:
                    append_jsonl(spans_path, span_row); spans[row["case_id"]] = span_row
                baseline = clean.get(row["case_id"])
                if baseline is None:
                    before = time.perf_counter(); logits, score = _forward(inference.model, inputs, located["SAC"], digit_ids)
                    recent.append(time.perf_counter() - before)
                    target = int(score["argmax_hard_class"])
                    generated = tokenizer.decode([digit_ids[target]], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    if generated != str(target) or len(tokenizer.encode(generated, add_special_tokens=False)) != 1:
                        raise RuntimeError(f"Constrained clean prediction is not one digit token: {generated!r}")
                    baseline = {
                        "case_id": row["case_id"], "item_id": str(row["item_id"]), "test_side": row["test_side"],
                        "prompt": prompt, "prompt_hash": canonical_hash(prompt), "raw_answer": row["phase0_raw_answer"],
                        "class_logits": logits, **score, "clean_class": target, "clean_margin": _margin(logits, target),
                        "generated_class": generated, "generated_token_id": int(digit_ids[target]),
                        "generation_constraint": "argmax_over_single_token_digits_0_8",
                    }
                    append_jsonl(clean_path, baseline); clean[row["case_id"]] = baseline; done += 1
                for condition in CONDITIONS:
                    for start, end in WINDOWS:
                        key = (row["case_id"], condition, start)
                        if key in completed: continue
                        edges = _edges(condition, located)
                        before = time.perf_counter()
                        with AttentionBlockContext(
                            modules.language_layers, layer_indices=range(start, end + 1), edges=edges,
                            sequence_length=located["sequence_length"], row_sum_tolerance=ROW_SUM_TOLERANCE,
                        ) as context:
                            logits, score = _forward(inference.model, inputs, located["SAC"], digit_ids)
                        elapsed = time.perf_counter() - before; recent.append(elapsed)
                        diagnostics = context.diagnostics(); target = int(baseline["clean_class"])
                        result = {
                            "case_id": row["case_id"], "item_id": str(row["item_id"]), "test_side": row["test_side"],
                            "condition": condition, "window_start": start, "window_end": end,
                            "class_logits": logits, "blocked_class": int(score["argmax_hard_class"]),
                            "blocked_soft_sa": float(score["soft_sa_image_score"]), "clean_class": target,
                            "clean_margin": float(baseline["clean_margin"]), "blocked_margin": _margin(logits, target),
                            "logit_margin_disruption": float(baseline["clean_margin"]) - _margin(logits, target),
                            "first_token_changed": int(score["argmax_hard_class"]) != target,
                            "delta_soft_sa": float(score["soft_sa_image_score"]) - float(baseline["soft_sa_image_score"]),
                            "original_token_logit_diff_change": float(baseline["class_logits"][target]) - float(logits[target]),
                            "elapsed_seconds": elapsed, "edge_count": len(edges.pairs), "attention_diagnostics": diagnostics,
                        }
                        if condition == "empty_block_parity":
                            result["empty_parity"] = {
                                "hard_equal": int(score["argmax_hard_class"]) == target,
                                "max_abs_logit_difference": max(abs(float(a) - float(b)) for a, b in zip(logits, baseline["class_logits"])),
                                "abs_soft_sa_difference": abs(float(score["soft_sa_image_score"]) - float(baseline["soft_sa_image_score"])),
                            }
                            if not result["empty_parity"]["hard_equal"] or result["empty_parity"]["max_abs_logit_difference"] > 1e-6 or result["empty_parity"]["abs_soft_sa_difference"] > 1e-9:
                                raise RuntimeError(f"Empty-block parity failed for {row['case_id']} at {start}-{end}")
                        append_jsonl(block_path, result); completed.add(key); done += 1
                        mean = sum(recent) / len(recent); remaining = total - len(clean) - len(completed)
                        atomic_json(output / "progress.json", {
                            "status": "running", "case_id": row["case_id"], "condition": condition,
                            "window": [start, end], "completed": len(clean) + len(completed), "expected": total,
                            "failed": len(load_jsonl(failures_path)), "elapsed_seconds": time.time() - started,
                            "recent_mean_forward_seconds": mean, "estimated_remaining_seconds": max(0, remaining) * mean,
                        })
                del inputs
            except Exception as exc:
                append_jsonl(failures_path, {"case_id": row.get("case_id"), "error_type": type(exc).__name__, "error": str(exc)})
                raise
        completion = {"status": "complete", "clean": len(clean), "blocked": len(completed),
                      "failures": len(load_jsonl(failures_path)), "elapsed_seconds": time.time() - started}
        atomic_json(output / "completion.json", completion)
        atomic_json(output / "progress.json", {**completion, "estimated_remaining_seconds": 0.0})
        return completion
    finally:
        del inference
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_dir or _default_output(args.smoke)
    run(output, args.manifest, smoke=args.smoke, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
