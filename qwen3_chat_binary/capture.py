from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from confidence_test.answer_metrics import parse_answer_output
from dp_SA.io_utils import append_jsonl, atomic_json, canonical_hash, load_jsonl, sha256_file
from layer_metacognition.model_adapter import (
    model_input_device,
    resolve_language_modules,
    run_logits_forward,
)

from .adapters import SelectedHiddenCapture
from .config import (
    CAPTURE_LAYERS,
    CAPTURE_ROOT,
    DATASET_PATH,
    ERROR_RATE_LIMIT,
    EXPECTED_HIDDEN_SIZE,
    EXPECTED_NUM_HIDDEN_LAYERS,
    HIDDEN_DEFINITION,
    INFERENCE_PATH,
    MODEL_PATH,
    POSITIONS,
    VARIANTS,
)
from .contracts import all_hidden_keys, ensure_fingerprinted_config, hidden_key, parse_variants
from .dataset import ConflictCase, QUESTION_TEMPLATE, load_conflict_cases
from .conversation import (
    prepare_multimodal_inputs,
    render_stage1,
    render_stage2,
    stage1_messages,
    stage2_messages,
)
from .positions import locate_positions
from .prompts import ATTRIBUTION_TEMPLATE, LABELS, LABEL_WEIGHTS, PHASE0_TEMPLATE, phase0_prompt
from .runtime import load_qwen3_inference
from .scoring import attribution_score, label_token_ids
from .layout import (
    capture_config_path,
    capture_phase0_path,
    capture_results_path,
    ensure_output_layout,
)


def acquire_pid(path: Path, label: str) -> None:
    if path.exists():
        try:
            pid = int(path.read_text(encoding="utf-8"))
            os.kill(pid, 0)
            raise RuntimeError(f"{label} already active: PID {pid}")
        except ProcessLookupError:
            path.unlink()
        except ValueError:
            raise RuntimeError(f"Invalid active PID file: {path}") from None
    path.write_text(str(os.getpid()), encoding="utf-8")


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        with open(temporary, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def model_fingerprint(model_path: Path) -> dict[str, str]:
    files = (
        "config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.json",
        "preprocessor_config.json", "model.safetensors.index.json",
    )
    missing = [name for name in files if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Qwen3 model directory is incomplete: {missing}")
    return {name: sha256_file(model_path / name) for name in files}


def case_rows(max_samples: int | None) -> list[ConflictCase]:
    return load_conflict_cases(DATASET_PATH, max_samples=max_samples)


def case_metadata(case: ConflictCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "source_index": case.source_index,
        "question": case.question,
        "text_clue": case.text_clue,
        "shape": case.shape,
        "pair_type": case.pair_type,
        "text_entropy": case.text_entropy,
        "image_entropy": case.image_entropy,
        "text_answer": case.text_answer,
        "image_answer": case.image_answer,
        "image_reference": case.image_reference,
    }


def generate_phase0(inference: Any, inputs: Any, max_new_tokens: int = 24) -> tuple[list[int], str, bool]:
    with torch.inference_mode():
        generated = inference.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
        )
    input_length = int(inputs.input_ids.shape[1])
    token_ids = [int(value) for value in generated[0, input_length:].tolist()]
    tokenizer = getattr(inference.processor, "tokenizer", inference.processor)
    continuation = tokenizer.decode(
        token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    eos_ids = getattr(inference.model.generation_config, "eos_token_id", [])
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    return token_ids, continuation, bool(token_ids and token_ids[-1] in set(eos_ids or []))


def capture_config(model_path: Path, variants: Sequence[str], max_samples: int | None) -> dict[str, Any]:
    return {
        "format_version": 3,
        "experiment": "qwen3_vl_chat_fiveway_capture",
        "model": str(model_path),
        "model_fingerprint": model_fingerprint(model_path),
        "inference_path": str(INFERENCE_PATH.resolve()),
        "dataset": str(DATASET_PATH.resolve()),
        "dataset_sha256": sha256_file(DATASET_PATH),
        "dataset_schema": "conflict_test_case_v1",
        "question_template": QUESTION_TEMPLATE,
        "question_template_hash": canonical_hash(QUESTION_TEMPLATE),
        "split_unit": "case_id",
        "variants": list(variants),
        "positions": list(POSITIONS),
        "layers": list(CAPTURE_LAYERS),
        "max_samples": max_samples,
        "phase0_template_hash": canonical_hash(PHASE0_TEMPLATE),
        "phase1_template_hash": canonical_hash(ATTRIBUTION_TEMPLATE),
        "phase0_generation": {"max_new_tokens": 24, "do_sample": False, "use_cache": True},
        "phase1_forward": {
            "use_cache": False, "labels": list(LABELS),
            "probability_weights": list(LABEL_WEIGHTS),
            "signed_mapping": "2*image_attribution_score-1",
        },
        "hidden_definition": HIDDEN_DEFINITION,
        "hidden_storage_dtype": "float16",
        "attention_implementation": "sdpa",
        "native_qwen3_system_message": None,
        "boundary_definitions": {
            "native_boundary": {"PANL": "newline_after_im_end", "PANL+1": "next_user_im_start"},
            "explicit_newline": {"PANL": "appended_answer_newline", "PANL+1": "assistant_im_end"},
        },
    }


def _validate_phase0(raw: str, continuation: str, eos: bool) -> tuple[str, str]:
    answer, normalized, parsed = parse_answer_output(raw)
    if not eos:
        raise ValueError("Phase-0 generation reached max_new_tokens without EOS")
    if not parsed or answer is None or normalized is None:
        raise ValueError("Phase-0 answer parser failed")
    if "\n" in continuation or "\r" in continuation:
        raise ValueError("Phase-0 output contains text after the required single-line answer")
    return answer, normalized


def run_capture(
    *,
    model_path: Path = MODEL_PATH,
    output_root: Path = CAPTURE_ROOT,
    variants: Sequence[str] = VARIANTS,
    max_samples: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    variants = parse_variants(variants)
    model_path, output_root = model_path.resolve(), output_root.resolve()
    ensure_output_layout(output_root, variants)
    pid_path = output_root / "progress" / "active.pid"
    acquire_pid(pid_path, "Qwen3 chat-fiveway capture")
    try:
        config = ensure_fingerprinted_config(
            capture_config_path(output_root),
            capture_config(model_path, variants, max_samples),
            resume=resume,
            label="Capture",
        )
        phase0_path = capture_phase0_path(output_root)
        phase0_rows = {row["case_id"]: row for row in load_jsonl(phase0_path)}
        completed = {
            variant: {
                row["case_id"] for row in load_jsonl(capture_results_path(output_root, variant))
                if row.get("status") == "completed"
            }
            for variant in variants
        }
        inference = load_qwen3_inference(model_path)
        modules = resolve_language_modules(inference.model)
        if (modules.num_hidden_layers, modules.hidden_size) != (
            EXPECTED_NUM_HIDDEN_LAYERS, EXPECTED_HIDDEN_SIZE,
        ):
            raise ValueError("Loaded model does not match Qwen3-VL-8B architecture")
        tokenizer = getattr(inference.processor, "tokenizer", inference.processor)
        token_ids = label_token_ids(tokenizer)
        device = model_input_device(inference)
        rows = case_rows(max_samples)
        failures = attempts = 0
        started = time.time()
        image_hashes: dict[str, str] = {}
        for ordinal, case in enumerate(rows, 1):
            case_id = case.case_id
            try:
                phase0 = phase0_rows.get(case_id)
                if phase0 is None or phase0.get("status") != "completed":
                    attempts += 1
                    prompt0 = phase0_prompt(case.question, case.text_clue)
                    messages0 = stage1_messages(prompt0, str(case.image_path))
                    rendered0 = render_stage1(inference.processor, messages0)
                    inputs0 = prepare_multimodal_inputs(inference.processor, messages0, rendered0, device=device)
                    generated_ids, continuation, eos = generate_phase0(inference, inputs0)
                    raw = "**Answer**:" + continuation
                    answer, normalized = _validate_phase0(raw, continuation, eos)
                    image_path = str(case.image_path)
                    image_hashes.setdefault(image_path, sha256_file(image_path))
                    phase0 = {
                        "status": "completed", **case_metadata(case),
                        "image_path": image_path, "image_sha256": image_hashes[image_path],
                        "phase0_prompt": prompt0, "phase0_prompt_hash": canonical_hash(prompt0),
                        "phase0_raw_output": raw, "phase0_raw_answer": answer,
                        "phase0_normalized_answer": normalized,
                        "phase0_generated_token_ids": generated_ids,
                        "phase0_eos_generated": eos,
                        "phase0_answer_fingerprint": canonical_hash(raw),
                    }
                    append_jsonl(phase0_path, phase0)
                    phase0_rows[case_id] = phase0
                for variant in variants:
                    if case_id in completed[variant]:
                        continue
                    attempts += 1
                    result_path = capture_results_path(output_root, variant)
                    try:
                        messages = stage2_messages(
                            phase0["phase0_prompt"], phase0["image_path"],
                            phase0["phase0_raw_output"], variant,
                        )
                        rendered = render_stage2(inference.processor, messages)
                        if rendered.startswith("<|im_start|>system"):
                            raise RuntimeError("Qwen3 native rendering unexpectedly added a system message")
                        inputs = prepare_multimodal_inputs(
                            inference.processor, messages, rendered, device=device
                        )
                        located = locate_positions(
                            tokenizer, rendered, inputs, phase0["phase0_raw_output"], variant
                        )
                        indices = located["indices"]
                        sequence_length = int(inputs.input_ids.shape[1])
                        capture = SelectedHiddenCapture(
                            modules, positions=indices, layers=CAPTURE_LAYERS,
                            prefill_sequence_length=sequence_length,
                        )
                        with capture:
                            logits = run_logits_forward(
                                inference.model, inputs, [indices["SAC"]], modules
                            )[indices["SAC"]]
                        hidden = capture.validate()
                        score = attribution_score(logits, token_ids)
                        arrays = {
                            hidden_key(position, layer): hidden[position][layer].detach().float().cpu().numpy().astype(np.float16)
                            for position in POSITIONS for layer in CAPTURE_LAYERS
                        }
                        if tuple(arrays) != all_hidden_keys() or any(
                            value.shape != (EXPECTED_HIDDEN_SIZE,) or not np.isfinite(value).all()
                            for value in arrays.values()
                        ):
                            raise ValueError("Captured hidden-state contract failed")
                        hidden_relative = Path("tables") / "hidden" / f"{case_id}.npz"
                        atomic_npz(output_root / variant / hidden_relative, arrays)
                        normalized = phase0["phase0_normalized_answer"]
                        result = {
                            "status": "completed", "variant": variant,
                            **case_metadata(case),
                            "image_path": phase0["image_path"], "image_sha256": phase0["image_sha256"],
                            "phase0_prompt": phase0["phase0_prompt"],
                            "phase0_raw_output": phase0["phase0_raw_output"],
                            "phase0_raw_answer": phase0["phase0_raw_answer"],
                            "phase0_normalized_answer": normalized,
                            "phase0_answer_fingerprint": phase0["phase0_answer_fingerprint"],
                            "explicit_newline_appended": variant == "explicit_newline",
                            "attribution_prompt": ATTRIBUTION_TEMPLATE,
                            "rendered_prompt_hash": canonical_hash(rendered),
                            "positions": located["positions"],
                            "causal_order_valid": located["causal_order_valid"],
                            **score,
                            "hidden_file": str(hidden_relative), "hidden_key_count": len(arrays),
                            "hidden_definition": HIDDEN_DEFINITION,
                            "capture_diagnostics": capture.diagnostics(),
                            "answer_matches_text": normalized == case.text_answer,
                            "answer_matches_image": normalized == case.image_answer,
                            "answer_length": len(phase0["phase0_raw_answer"]),
                            "elapsed_ordinal": ordinal,
                        }
                        append_jsonl(result_path, result)
                        completed[variant].add(case_id)
                    except Exception as exc:
                        failures += 1
                        append_jsonl(result_path, {
                            "status": "failed", "variant": variant, **case_metadata(case), "error": {
                                "type": type(exc).__name__, "message": str(exc),
                            },
                        })
                        if failures / max(1, attempts) > ERROR_RATE_LIMIT:
                            raise RuntimeError(
                                f"Capture failure rate exceeded {ERROR_RATE_LIMIT:.0%}: {failures}/{attempts}"
                            ) from exc
            except Exception as exc:
                failures += 1
                if case_id not in phase0_rows:
                    append_jsonl(phase0_path, {
                        "status": "failed", **case_metadata(case),
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    })
                if failures / max(1, attempts) > ERROR_RATE_LIMIT:
                    raise
            if ordinal % 10 == 0:
                progress = {
                    "total": len(rows), "completed": {key: len(value) for key, value in completed.items()},
                    "failures_this_run": failures, "elapsed_seconds": time.time() - started,
                }
                atomic_json(output_root / "progress" / "progress.json", progress)
                for variant in variants:
                    atomic_json(output_root / variant / "progress" / "progress.json", {
                        "variant": variant,
                        "total": len(rows),
                        "completed": len(completed[variant]),
                        "failures_this_run": failures,
                        "elapsed_seconds": progress["elapsed_seconds"],
                    })
        summary = {
            "status": "complete", "total_cases": len(rows),
            "completed": {key: len(value) for key, value in completed.items()},
            "shared_completed": len(set.intersection(*(completed[key] for key in variants))),
            "failures_this_run": failures, "hidden_keys_per_case": len(all_hidden_keys()),
            "config_fingerprint": config["fingerprint"],
        }
        atomic_json(output_root / "progress" / "summary.json", summary)
        atomic_json(output_root / "progress" / "progress.json", summary)
        for variant in variants:
            variant_summary = {
                "status": "complete",
                "variant": variant,
                "total_cases": len(rows),
                "completed": len(completed[variant]),
                "failures_this_run": failures,
                "config_fingerprint": config["fingerprint"],
            }
            atomic_json(output_root / variant / "progress" / "summary.json", variant_summary)
            atomic_json(output_root / variant / "progress" / "progress.json", variant_summary)
        return summary
    finally:
        if pid_path.exists() and pid_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
            pid_path.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture Qwen3 chat-fiveway attribution hidden states")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--output-root", type=Path, default=CAPTURE_ROOT)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    print(run_capture(
        model_path=args.model_path, output_root=args.output_root, variants=args.variants,
        max_samples=args.max_samples, resume=args.resume,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
