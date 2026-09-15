from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
SHORT_ROOT = HERE.parent
REVIEW_ROOT = SHORT_ROOT.parent
REPOSITORY_ROOT = REVIEW_ROOT.parent
for candidate in (REPOSITORY_ROOT, REVIEW_ROOT, SHORT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np
import torch
from transformers import AutoProcessor

from capture.positions import external_positions, locate_short_phase1_positions
from capture.run import POSITION_KEYS, POSITIONS, frozen_rows, processor_audit
from capture.short_prompt import SA_PREFILL
from capture_reverse.reverse_prompt import PHASE1_TEMPLATE_SHORT_REVERSE, phase1_prompt_short_reverse
from capture_reverse.scoring import REVERSE_MIDPOINTS, reverse_soft_sa_from_logits
from dp_SA.io_utils import append_jsonl, atomic_json, canonical_hash, load_jsonl, sha256_file
from dp_SA.soft_score import class_token_ids
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.model_adapter import model_input_device, resolve_language_modules, run_logits_forward
from Steering.capture import _acquire_pid, _atomic_npz, _generate, _messages, _model_fingerprint
from Steering.config import CAPTURE_LAYERS, EXPECTED_HIDDEN_SIZE, EXPECTED_NUM_HIDDEN_LAYERS, HIDDEN_DEFINITION, MODEL_PATH
from Steering.contracts import all_capture_keys, ensure_fingerprinted_config, hidden_key
from Steering.hooks import SelectedHiddenCapture
from Steering.runtime import load_qwen3_inference


OUTPUT_ROOT = SHORT_ROOT / "output" / "capture_reverse"
SHORT_CAPTURE_ROOT = SHORT_ROOT / "output" / "capture"


def _context(processor: Any, row: dict[str, Any], device: Any | None = None):
    answer = str(row["phase0_raw_answer"])
    prompt = phase1_prompt_short_reverse(row["question"], row["text_clue"], answer)
    messages = _messages(prompt, row["image_path"], SA_PREFILL)
    rendered = render_continued_assistant(processor, messages, SA_PREFILL)
    if rendered.startswith("<|im_start|>system"):
        raise RuntimeError("Qwen3 native rendering unexpectedly added a system message")
    inputs = prepare_multimodal_inputs(processor, messages, rendered, device=device)
    located = locate_short_phase1_positions(processor.tokenizer, rendered, inputs, answer)
    return prompt, rendered, messages, inputs, located


def run_preflight(*, output_root: Path = OUTPUT_ROOT, resume: bool = False) -> dict[str, Any]:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / "position_preflight.jsonl"
    if destination.exists() and not resume:
        raise FileExistsError(f"Preflight output exists: {destination}; use --resume")
    rows = frozen_rows()
    completed = {row["case_id"] for row in load_jsonl(destination) if row.get("status") == "completed"}
    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    audit = processor_audit(processor)
    atomic_json(output_root / "processor_audit.json", audit)
    if not audit["tokenizer_is_fast"] or not audit["image_processor_is_fast"]:
        raise RuntimeError(f"Expected actual Fast tokenizer and image processor: {audit}")
    for ordinal, row in enumerate(rows, 1):
        if row["case_id"] in completed:
            continue
        try:
            prompt, rendered, _messages_value, inputs, located = _context(processor, row)
            positions = external_positions(located)
            record = {
                "status": "completed", "ordinal": ordinal, "case_id": row["case_id"],
                "item_id": row["item_id"], "phase0_raw_answer": row["phase0_raw_answer"],
                "phase1_prompt_hash": canonical_hash(prompt), "rendered_hash": canonical_hash(rendered),
                "rendered_length": len(rendered), "processed_sequence_length": int(inputs.input_ids.shape[1]),
                "positions": positions, "causal_order_valid": True,
            }
            append_jsonl(destination, record)
            compact = " ".join(
                f"{name}:chars={value['rendered_token_span']} idx={value['processed_index']} "
                f"id={value['token_id']} tok={value['token_text']!r}"
                for name, value in positions.items() if name in {"LAT", "PANL", "CLE", "SAC"}
            )
            print(f"[{ordinal:03d}/500] {row['case_id']} {compact} order=LAT<PANL<CLE<SAC", flush=True)
        except Exception as exc:
            append_jsonl(destination, {
                "status": "failed", "ordinal": ordinal, "case_id": row["case_id"],
                "error": {"type": type(exc).__name__, "message": str(exc)},
            })
            atomic_json(output_root / "preflight_summary.json", {
                "status": "failed", "case_id": row["case_id"],
                "error": {"type": type(exc).__name__, "message": str(exc)},
            })
            raise
    records = [row for row in load_jsonl(destination) if row.get("status") == "completed"]
    if len({row["case_id"] for row in records}) != 500:
        raise RuntimeError("Reverse position preflight did not complete 500 unique cases")
    summary = {"status": "complete", "case_count": 500, "all_causal_orders_valid": True}
    atomic_json(output_root / "preflight_summary.json", summary)
    return summary


def capture_config(output_root: Path) -> dict[str, Any]:
    short_config = json.loads((SHORT_CAPTURE_ROOT / "config.json").read_text(encoding="utf-8"))
    return {
        "format_version": 1,
        "experiment": "qwen3_vl_short_reverse_prompt_capture",
        "model": str(MODEL_PATH.resolve()),
        "model_fingerprint": _model_fingerprint(MODEL_PATH),
        "source_short_capture": str(SHORT_CAPTURE_ROOT.resolve()),
        "source_short_capture_fingerprint": short_config["fingerprint"],
        "source_short_results_sha256": sha256_file(SHORT_CAPTURE_ROOT / "results.jsonl"),
        "fixed_answer_policy": "freeze_long_phase0_answer_via_short_case_set",
        "case_count": 500,
        "positions": list(POSITIONS), "position_keys": POSITION_KEYS,
        "layers": list(CAPTURE_LAYERS),
        "hidden_definition": HIDDEN_DEFINITION, "hidden_storage_dtype": "float16",
        "attention_implementation": "sdpa", "native_qwen3_system_message": None,
        "phase1_template_hash": canonical_hash(PHASE1_TEMPLATE_SHORT_REVERSE),
        "class_generation": {"max_new_tokens": 1, "do_sample": False, "use_cache": True},
        "score_orientation": {
            "raw_labels": "0=image_side, 8=text_side",
            "canonical_soft_sa": "higher=image_side",
            "raw_label_midpoints": list(REVERSE_MIDPOINTS),
            "canonical_hard_class": "8 - raw_argmax_class",
        },
        "cle_policy": "processed_token_containing_target_newline",
        "output_root": str(output_root),
    }


def run_capture(*, output_root: Path = OUTPUT_ROOT, resume: bool = False, limit: int | None = None) -> dict[str, Any]:
    output_root = output_root.resolve()
    if limit is None:
        run_preflight(output_root=output_root, resume=resume)
    config = ensure_fingerprinted_config(
        output_root / "config.json", capture_config(output_root), resume=resume, label="Reverse short capture"
    )
    pid = output_root / "active.pid"
    _acquire_pid(pid, "Reverse short capture")
    started = time.time()
    try:
        result_path = output_root / "results.jsonl"
        completed = {row["case_id"] for row in load_jsonl(result_path) if row.get("status") == "completed"}
        rows = frozen_rows()
        if limit is not None:
            rows = rows[: int(limit)]
        runtime = load_qwen3_inference(MODEL_PATH)
        audit = processor_audit(runtime.processor)
        stored_audit = output_root / "processor_audit.json"
        if stored_audit.is_file() and audit != json.loads(stored_audit.read_text(encoding="utf-8")):
            raise RuntimeError("Standalone and model-runtime processor configurations differ")
        atomic_json(stored_audit, audit)
        modules = resolve_language_modules(runtime.model)
        if (modules.num_hidden_layers, modules.hidden_size) != (EXPECTED_NUM_HIDDEN_LAYERS, EXPECTED_HIDDEN_SIZE):
            raise RuntimeError("Loaded model architecture differs from formal Qwen3 capture")
        tokenizer = runtime.processor.tokenizer
        class_ids = class_token_ids(tokenizer)
        device = model_input_device(runtime)
        failures = 0
        for ordinal, row in enumerate(rows, 1):
            if row["case_id"] in completed:
                continue
            try:
                prompt, _rendered, _message_values, inputs, located = _context(runtime.processor, row, device)
                positions = external_positions(located)
                indices = {name: int(value["processed_index"]) for name, value in positions.items()}
                sequence_length = int(inputs.input_ids.shape[1])
                sac = indices["SAC"]
                capture = SelectedHiddenCapture(
                    modules, positions=indices, layers=CAPTURE_LAYERS,
                    prefill_sequence_length=sequence_length,
                )
                with capture:
                    vocab = run_logits_forward(runtime.model, inputs, [sac], modules)[sac]
                hidden = capture.validate()
                score = reverse_soft_sa_from_logits(vocab, class_ids)
                generated_ids, generated_text, _ = _generate(runtime, inputs, 1, class_ids)
                if generated_text not in set(map(str, range(9))):
                    raise ValueError(f"Invalid constrained reverse Phase 1 output: {generated_text!r}")
                if int(generated_text) != int(score["raw_argmax_class"]):
                    raise ValueError("Reverse forward raw argmax differs from constrained greedy generation")
                arrays = {
                    hidden_key(position, layer): hidden[position][layer].detach().float().cpu().numpy().astype(np.float16)
                    for position in POSITIONS for layer in CAPTURE_LAYERS
                }
                if tuple(arrays) != all_capture_keys() or any(
                    value.shape != (EXPECTED_HIDDEN_SIZE,) or not np.isfinite(value).all()
                    for value in arrays.values()
                ):
                    raise ValueError("Reverse hidden-state key/shape/finiteness contract failed")
                relative = Path("hidden") / f"{row['case_id']}.npz"
                _atomic_npz(output_root / relative, arrays)
                append_jsonl(result_path, {
                    **{key: row[key] for key in (
                        "case_id", "item_id", "prior_index", "condition", "version", "question",
                        "text_clue", "image_path", "image_sha256", "phase0_raw_answer",
                        "phase0_normalized_answer", "phase0_answer_fingerprint", "phase0_correct",
                        "answer_matches_text", "answer_matches_image", "answer_length",
                    ) if key in row},
                    "status": "completed", "model_family": "qwen3_vl",
                    "fixed_answer_source": "long_capture_phase0",
                    "phase1_prompt": prompt, "phase1_prompt_hash": canonical_hash(prompt),
                    "phase1_answer_span": located["phase1_answer_span"],
                    "phase1_answer_token_ids": located["phase1_answer_token_ids"],
                    "positions": positions, "causal_order_valid": True,
                    **score, "raw_generated_class": generated_text, "valid_class": True,
                    "generated_token_ids": generated_ids,
                    "hidden_file": str(relative), "hidden_key_count": len(arrays),
                    "hidden_definition": HIDDEN_DEFINITION,
                    "capture_diagnostics": capture.diagnostics(), "elapsed_ordinal": ordinal,
                })
                completed.add(row["case_id"])
            except Exception as exc:
                failures += 1
                append_jsonl(result_path, {
                    "status": "failed", "case_id": row["case_id"],
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                })
                raise
            if ordinal % 10 == 0:
                atomic_json(output_root / "progress.json", {
                    "status": "running", "completed": len(completed),
                    "target": len(rows), "formal_total": 500,
                    "failed_this_run": failures, "elapsed_seconds": time.time() - started,
                })
        target = len(rows)
        completed_target = sum(row["case_id"] in completed for row in rows)
        summary = {
            "status": "complete" if completed_target == target else "incomplete",
            "target": target, "completed_target": completed_target,
            "formal": limit is None, "failed_this_run": failures,
            "hidden_keys_per_case": len(all_capture_keys()),
            "config_fingerprint": config["fingerprint"],
        }
        if completed_target != target:
            raise RuntimeError(f"Reverse capture grid incomplete: {completed_target}/{target}")
        atomic_json(output_root / ("summary.json" if limit is None else "smoke_summary.json"), summary)
        atomic_json(output_root / "progress.json", summary | {"elapsed_seconds": time.time() - started})
        return summary
    finally:
        if pid.exists() and pid.read_text(encoding="utf-8").strip() == str(os.getpid()):
            pid.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", type=int, default=None, metavar="N")
    args = parser.parse_args(argv)
    result = run_preflight(output_root=args.output_root, resume=args.resume) if args.preflight_only else run_capture(
        output_root=args.output_root, resume=args.resume, limit=args.smoke
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
