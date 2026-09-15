from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    _review = Path(__file__).resolve().parents[1]
    for _path in (_review.parent, _review):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))

import numpy as np
import torch

from confidence_test.answer_metrics import normalize_answer, parse_answer_output
from confidence_test.dataset_utils import load_evaluation_cases
from dp_SA.io_utils import append_jsonl, atomic_json, canonical_hash, load_jsonl, sha256_file
from dp_SA.positions import locate_phase1_positions
from dp_SA.prompts import ANSWER_PREFILL, SA_PREFILL, phase0_prompt, phase1_prompt
from dp_SA.soft_score import class_token_ids, soft_sa_from_logits
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.model_adapter import model_input_device, resolve_language_modules, run_logits_forward

from Steering.config import (
    CAPTURE_LAYERS, CAPTURE_ROOT, CONDITIONS, DATASET_PATH, ERROR_RATE_LIMIT,
    EXPECTED_HIDDEN_SIZE, EXPECTED_NUM_HIDDEN_LAYERS, HIDDEN_DEFINITION,
    INFERENCE_PATH, MODEL_PATH, POSITIONS, POSITION_KEYS,
)
from Steering.contracts import all_capture_keys, ensure_fingerprinted_config, hidden_key
from Steering.hooks import SelectedHiddenCapture
from Steering.runtime import load_qwen3_inference


def _messages(prompt: str, image_path: str, prefill: str) -> list[dict[str, Any]]:
    # Native Qwen3 rendering: deliberately do not add a system message.
    return [
        {"role": "user", "content": [
            {"type": "image", "image": str(Path(image_path).resolve())},
            {"type": "text", "text": prompt},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": prefill}]},
    ]


def _generate(
    inference: Any,
    inputs: Any,
    max_new_tokens: int,
    allowed_first_tokens: Sequence[int] | None = None,
) -> tuple[list[int], str, bool]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens), "do_sample": False, "use_cache": True,
    }
    if allowed_first_tokens is not None:
        allowed = tuple(int(value) for value in allowed_first_tokens)
        kwargs["prefix_allowed_tokens_fn"] = lambda _batch, _ids: list(allowed)
    with torch.inference_mode():
        generated = inference.model.generate(**inputs, **kwargs)
    input_length = int(inputs.input_ids.shape[1])
    tokens = [int(value) for value in generated[0, input_length:].tolist()]
    tokenizer = getattr(inference.processor, "tokenizer", inference.processor)
    text = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    eos_ids = getattr(inference.model.generation_config, "eos_token_id", [])
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    return tokens, text, bool(tokens and tokens[-1] in set(eos_ids or []))


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
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


def _case_rows(
    max_items: int | None,
    max_samples: int | None,
    *,
    unique_items: bool = False,
    preferred_case_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    cases, _ = load_evaluation_cases(DATASET_PATH, item_limit=max_items)
    rows: list[dict[str, Any]] = []
    for case in cases:
        for condition in CONDITIONS:
            image = case.conditions[condition]
            if image.error:
                continue
            rows.append({
                "case": case,
                "condition": condition,
                "image_path": image.resolved_image_path,
                "case_id": f"{case.item_id}__prior_{case.prior_index}__{condition}__v4__delayed_sa",
            })
    if not max_samples:
        return rows
    if not unique_items:
        if preferred_case_ids:
            preferred = [row for row in rows if row["case_id"] in preferred_case_ids]
            remainder = [row for row in rows if row["case_id"] not in preferred_case_ids]
            return (preferred + remainder)[:max_samples]
        return rows[:max_samples]
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for row in rows:
        item = str(row["case"].item_id)
        if item in used:
            continue
        selected.append(row)
        used.add(item)
        if len(selected) == max_samples:
            break
    return selected


def _model_fingerprint(model_path: Path) -> dict[str, str]:
    files = (
        "config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.json",
        "preprocessor_config.json", "model.safetensors.index.json",
    )
    missing = [name for name in files if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Qwen3 model directory is incomplete: {missing}")
    return {name: sha256_file(model_path / name) for name in files}


def _external_positions(located: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {name: dict(located[key]) for name, key in POSITION_KEYS.items()}


def _acquire_pid(path: Path, label: str) -> None:
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


def _capture_config(model_path: Path, max_items: int | None, max_samples: int | None) -> dict[str, Any]:
    return {
        "format_version": 1,
        "experiment": "qwen3_vl_delayed_sa_capture",
        "dataset": str(DATASET_PATH.resolve()),
        "dataset_sha256": sha256_file(DATASET_PATH),
        "model": str(model_path),
        "model_fingerprint": _model_fingerprint(model_path),
        "inference_path": str(INFERENCE_PATH.resolve()),
        "conditions": list(CONDITIONS),
        "positions": list(POSITIONS),
        "position_keys": dict(POSITION_KEYS),
        "layers": list(CAPTURE_LAYERS),
        "expected_num_hidden_layers": EXPECTED_NUM_HIDDEN_LAYERS,
        "expected_hidden_size": EXPECTED_HIDDEN_SIZE,
        "max_items": max_items,
        "max_samples": max_samples,
        "hidden_definition": HIDDEN_DEFINITION,
        "hidden_storage_dtype": "float16",
        "native_qwen3_system_message": None,
        "attention_implementation": "sdpa",
        "phase0_generation": {"max_new_tokens": 24, "do_sample": False, "use_cache": True},
        "phase1_generation": {
            "max_new_tokens": 1, "do_sample": False, "use_cache": True,
            "constraint": "validated_class_token_ids",
        },
        "phase0_template_hash": canonical_hash(phase0_prompt("{question}", "{text_clue}")),
        "phase1_template_hash": canonical_hash(phase1_prompt("{question}", "{text_clue}", "{answer}")),
    }


def run_capture(
    *,
    model_path: Path = MODEL_PATH,
    output_root: Path = CAPTURE_ROOT,
    max_items: int | None = None,
    max_samples: int | None = None,
    unique_items: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    model_path, output_root = model_path.resolve(), output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    pid_path = output_root / "active.pid"
    _acquire_pid(pid_path, "Qwen3 capture")
    try:
        config = ensure_fingerprinted_config(
            output_root / "config.json",
            _capture_config(model_path, max_items, max_samples),
            resume=resume,
            label="Capture",
        )
        phase0_path, results_path = output_root / "phase0_results.jsonl", output_root / "results.jsonl"
        phase0 = {row["case_id"]: row for row in load_jsonl(phase0_path)}
        completed = {
            row["case_id"] for row in load_jsonl(results_path)
            if row.get("status") == "completed"
        }
        inference = load_qwen3_inference(model_path)
        modules = resolve_language_modules(inference.model)
        if (modules.num_hidden_layers, modules.hidden_size) != (
            EXPECTED_NUM_HIDDEN_LAYERS, EXPECTED_HIDDEN_SIZE,
        ):
            raise ValueError(
                f"Unexpected Qwen3 language architecture: layers={modules.num_hidden_layers}, "
                f"hidden={modules.hidden_size}"
            )
        tokenizer = getattr(inference.processor, "tokenizer", inference.processor)
        class_ids, device = class_token_ids(tokenizer), model_input_device(inference)
        rows = _case_rows(
            max_items,
            max_samples,
            unique_items=unique_items,
            preferred_case_ids=completed,
        )
        failures = attempted = 0
        started = time.time()
        image_hashes: dict[str, str] = {}
        for ordinal, specification in enumerate(rows, 1):
            case, case_id = specification["case"], specification["case_id"]
            if case_id in completed:
                continue
            attempted += 1
            try:
                phase0_row = phase0.get(case_id)
                if phase0_row is None or phase0_row.get("status") != "completed":
                    prompt0 = phase0_prompt(case.question, case.text_clue)
                    messages0 = _messages(prompt0, specification["image_path"], ANSWER_PREFILL)
                    rendered0 = render_continued_assistant(inference.processor, messages0, ANSWER_PREFILL)
                    inputs0 = prepare_multimodal_inputs(inference.processor, messages0, rendered0, device=device)
                    token_ids, continuation, eos = _generate(inference, inputs0, 24)
                    raw = ANSWER_PREFILL + continuation
                    answer, normalized, parsed = parse_answer_output(raw)
                    image_path = str(specification["image_path"])
                    if image_path not in image_hashes:
                        image_hashes[image_path] = sha256_file(image_path)
                    phase0_row = {
                        "case_id": case_id, "item_id": case.item_id,
                        "prior_index": case.prior_index, "condition": specification["condition"],
                        "version": "v4", "model_family": "qwen3_vl",
                        "question": case.question, "text_clue": case.text_clue,
                        "image_path": image_path, "image_sha256": image_hashes[image_path],
                        "phase0_prompt": prompt0, "phase0_prompt_hash": canonical_hash(prompt0),
                        "phase0_raw_output": raw, "phase0_raw_answer": answer,
                        "phase0_normalized_answer": normalized,
                        "phase0_answer_token_ids": tokenizer.encode(str(answer), add_special_tokens=False) if answer else [],
                        "phase0_generated_token_ids": token_ids, "phase0_eos_generated": eos,
                        "phase0_generation_config": config["phase0_generation"],
                        "phase0_answer_fingerprint": canonical_hash(answer),
                        "parse_success": parsed, "status": "completed" if parsed else "failed",
                    }
                    append_jsonl(phase0_path, phase0_row)
                    phase0[case_id] = phase0_row
                if phase0_row.get("status") != "completed":
                    raise ValueError("Phase 0 answer parser failed")

                answer = str(phase0_row["phase0_raw_answer"])
                prompt1 = phase1_prompt(case.question, case.text_clue, answer)
                messages1 = _messages(prompt1, specification["image_path"], SA_PREFILL)
                rendered1 = render_continued_assistant(inference.processor, messages1, SA_PREFILL)
                if rendered1.startswith("<|im_start|>system"):
                    raise RuntimeError("Qwen3 native rendering unexpectedly added a system message")
                inputs1 = prepare_multimodal_inputs(inference.processor, messages1, rendered1, device=device)
                located = locate_phase1_positions(tokenizer, rendered1, inputs1, answer)
                external = _external_positions(located)
                indices = {name: int(record["processed_index"]) for name, record in external.items()}
                sequence_length, sac = int(inputs1.input_ids.shape[1]), indices["SAC"]
                capture = SelectedHiddenCapture(
                    modules, positions=indices, layers=CAPTURE_LAYERS,
                    prefill_sequence_length=sequence_length,
                )
                with capture:
                    logits = run_logits_forward(inference.model, inputs1, [sac], modules)[sac]
                hidden = capture.validate()
                score = soft_sa_from_logits(logits, class_ids)
                generated_ids, generated_text, _ = _generate(inference, inputs1, 1, class_ids)
                if generated_text not in set(map(str, range(9))):
                    raise ValueError(f"Invalid constrained Phase 1 output: {generated_text!r}")
                if int(generated_text) != score["argmax_hard_class"]:
                    raise ValueError("Forward argmax differs from greedy generation")
                arrays = {
                    hidden_key(position, layer): hidden[position][layer].detach().float().cpu().numpy().astype(np.float16)
                    for position in POSITIONS for layer in CAPTURE_LAYERS
                }
                if tuple(arrays) != all_capture_keys() or any(
                    value.shape != (EXPECTED_HIDDEN_SIZE,) or not np.isfinite(value).all()
                    for value in arrays.values()
                ):
                    raise ValueError("Captured hidden-state key, shape, or finiteness contract failed")
                hidden_relative = Path("hidden") / f"{case_id}.npz"
                _atomic_npz(output_root / hidden_relative, arrays)
                normalized = normalize_answer(answer)
                result = {
                    "status": "completed", "case_id": case_id, "item_id": case.item_id,
                    "prior_index": case.prior_index, "condition": specification["condition"],
                    "version": "v4", "model_family": "qwen3_vl",
                    "question": case.question, "text_clue": case.text_clue,
                    "image_path": specification["image_path"], "image_sha256": phase0_row["image_sha256"],
                    "phase0_raw_answer": answer, "phase0_normalized_answer": normalized,
                    "phase0_answer_fingerprint": phase0_row["phase0_answer_fingerprint"],
                    "phase1_inserted_raw_answer": answer,
                    "phase1_inserted_normalized_answer": normalize_answer(answer),
                    "phase1_prompt": prompt1, "phase1_prompt_hash": canonical_hash(prompt1),
                    "phase1_answer_span": located["phase1_answer_span"],
                    "phase1_answer_token_ids": located["phase1_answer_token_ids"],
                    "positions": external, "causal_order_valid": located["causal_order_valid"],
                    **score, "raw_generated_class": generated_text, "valid_class": True,
                    "generated_token_ids": generated_ids,
                    "phase1_generation_config": config["phase1_generation"],
                    "hidden_file": str(hidden_relative), "hidden_key_count": len(arrays),
                    "hidden_definition": HIDDEN_DEFINITION,
                    "capture_diagnostics": capture.diagnostics(),
                    "phase0_correct": normalized == case.ground_truth_answer,
                    "answer_matches_text": normalized == case.text_answer,
                    "answer_matches_image": normalized == case.conflict_answer,
                    "answer_length": len(answer), "elapsed_ordinal": ordinal,
                }
                append_jsonl(results_path, result)
                completed.add(case_id)
            except Exception as exc:
                failures += 1
                append_jsonl(results_path, {
                    "status": "failed", "case_id": case_id, "item_id": case.item_id,
                    "prior_index": case.prior_index, "condition": specification["condition"],
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                })
                if failures / max(1, attempted) > ERROR_RATE_LIMIT:
                    raise RuntimeError(
                        f"Capture failure rate exceeded {ERROR_RATE_LIMIT:.0%}: {failures}/{attempted}"
                    ) from exc
            if ordinal % 10 == 0:
                atomic_json(output_root / "progress.json", {
                    "completed": len(completed), "failed_this_run": failures,
                    "total": len(rows), "elapsed_seconds": time.time() - started,
                })
        summary = {
            "status": "complete", "total": len(rows), "completed": len(completed),
            "failed_this_run": failures, "hidden_keys_per_case": len(all_capture_keys()),
        }
        atomic_json(output_root / "summary.json", summary)
        atomic_json(output_root / "progress.json", summary | {"elapsed_seconds": time.time() - started})
        return summary
    finally:
        if pid_path.exists() and pid_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
            pid_path.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture Qwen3-VL hidden states at layers 8..35")
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--output-root", default=str(CAPTURE_ROOT))
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--unique-items", action="store_true",
        help="sample at most one case per item (useful for smoke Steering)",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    run_capture(
        model_path=Path(args.model_path), output_root=Path(args.output_root),
        max_items=args.max_items, max_samples=args.max_samples, resume=args.resume,
        unique_items=args.unique_items,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
