from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from confidence_test.answer_metrics import normalize_answer, parse_answer_output
from confidence_test.dataset_utils import load_evaluation_cases
from dp_SA.prompts import ANSWER_PREFILL, SA_PREFILL, phase0_prompt, phase1_prompt
from experiment_config import (
    CAPTURE_LAYERS,
    CONDITIONS,
    DATASET_PATH,
    ERROR_RATE_LIMIT,
    HIDDEN_DEFINITION,
    MODEL_PATH,
    POSITIONS,
    RESULTS_ROOT,
)
from gemma_runtime import GemmaRuntime, run_capture_forward
from io_utils import append_jsonl, atomic_json, canonical_hash, load_jsonl, sha256_file
from positions import locate_phase1_positions
from soft_score import class_token_ids, soft_sa_from_logits


def parse_layers(values: Sequence[int]) -> tuple[int, ...]:
    layers = tuple(map(int, values))
    if not layers or len(layers) != len(set(layers)):
        raise ValueError("Layers must be non-empty and unique")
    invalid = [layer for layer in layers if layer not in CAPTURE_LAYERS]
    if invalid:
        raise ValueError(f"Capture layers must be within L6-L33: {invalid}")
    return layers


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
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


def _case_rows(max_items: int | None, max_samples: int | None) -> list[dict[str, Any]]:
    cases, _ = load_evaluation_cases(DATASET_PATH, item_limit=max_items)
    rows = []
    for case in cases:
        for condition in CONDITIONS:
            image = case.conditions[condition]
            if image.error:
                continue
            rows.append(
                {
                    "case": case,
                    "condition": condition,
                    "image_path": image.resolved_image_path,
                    "case_id": f"{case.item_id}__prior_{case.prior_index}__{condition}__v4__gemma_delayed_sa",
                }
            )
    if max_samples is None:
        return rows
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    selected, used = [], set()
    for row in rows:
        item = str(row["case"].item_id)
        if item in used:
            continue
        selected.append(row)
        used.add(item)
        if len(selected) == max_samples:
            break
    return selected


def _model_fingerprint() -> dict[str, str]:
    names = (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "preprocessor_config.json",
        "model.safetensors.index.json",
    )
    return {name: sha256_file(MODEL_PATH / name) for name in names}


def run_capture(
    *,
    output_root: Path = RESULTS_ROOT,
    max_items: int | None = None,
    max_samples: int | None = None,
    resume: bool = False,
    layers: Sequence[int] = CAPTURE_LAYERS,
) -> dict[str, Any]:
    layers = parse_layers(layers)
    capture_dir = output_root / "capture"
    capture_dir.mkdir(parents=True, exist_ok=True)
    pid_path = capture_dir / "active.pid"
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text())
            os.kill(pid, 0)
            raise RuntimeError(f"Capture already active: PID {pid}")
        except ProcessLookupError:
            pid_path.unlink()
    pid_path.write_text(str(os.getpid()))
    try:
        config = {
            "format_version": 1,
            "model_family": "gemma3",
            "model": str(MODEL_PATH.resolve()),
            "model_fingerprint": _model_fingerprint(),
            "dataset": str(DATASET_PATH.resolve()),
            "dataset_sha256": sha256_file(DATASET_PATH),
            "conditions": list(CONDITIONS),
            "positions": list(POSITIONS),
            "layers": list(layers),
            "max_items": max_items,
            "max_samples": max_samples,
            "hidden_definition": HIDDEN_DEFINITION,
            "hidden_storage_dtype": "float32",
            "phase0_generation": {"max_new_tokens": 24, "do_sample": False, "use_cache": True},
            "phase1_generation": {
                "max_new_tokens": 1,
                "do_sample": False,
                "use_cache": True,
                "constraint": "validated_gemma_class_token_ids",
            },
            "phase0_template_hash": canonical_hash(phase0_prompt("{question}", "{text_clue}")),
            "phase1_template_hash": canonical_hash(phase1_prompt("{question}", "{text_clue}", "{answer}")),
        }
        config["fingerprint"] = canonical_hash(config)
        config_path = capture_dir / "config.json"
        if config_path.exists():
            previous = json.loads(config_path.read_text())
            if previous.get("fingerprint") != config["fingerprint"]:
                raise ValueError("Capture config changed; use a fresh output root")
            if not resume:
                raise FileExistsError("Capture output exists; pass --resume")
        else:
            atomic_json(config_path, config)

        phase0_path = capture_dir / "phase0_results.jsonl"
        results_path = capture_dir / "results.jsonl"
        phase0 = {row["case_id"]: row for row in load_jsonl(phase0_path)}
        completed = {
            row["case_id"] for row in load_jsonl(results_path) if row.get("status") == "completed"
        }
        runtime = GemmaRuntime(MODEL_PATH)
        class_ids = class_token_ids(runtime.processor.tokenizer)
        rows = _case_rows(max_items, max_samples)
        failures = 0
        started = time.time()
        image_hashes: dict[str, str] = {}
        for ordinal, spec in enumerate(rows, 1):
            case, case_id = spec["case"], spec["case_id"]
            if case_id in completed:
                continue
            try:
                phase0_row = phase0.get(case_id)
                if phase0_row is None:
                    prompt0 = phase0_prompt(case.question, case.text_clue)
                    messages0 = runtime.build_messages(prompt0, spec["image_path"], ANSWER_PREFILL)
                    _rendered0, inputs0 = runtime.prepare(messages0, ANSWER_PREFILL)
                    tokens, continuation, eos = runtime.generate(inputs0, 24)
                    raw_output = ANSWER_PREFILL + continuation
                    answer, normalized, parsed = parse_answer_output(raw_output)
                    answer_ids = (
                        runtime.processor.tokenizer.encode(str(answer), add_special_tokens=False)
                        if answer else []
                    )
                    image_path = str(spec["image_path"])
                    image_hashes.setdefault(image_path, sha256_file(image_path))
                    phase0_row = {
                        "status": "completed" if parsed else "failed",
                        "case_id": case_id,
                        "item_id": case.item_id,
                        "prior_index": case.prior_index,
                        "condition": spec["condition"],
                        "version": "v4_gemma",
                        "question": case.question,
                        "text_clue": case.text_clue,
                        "image_path": image_path,
                        "image_sha256": image_hashes[image_path],
                        "phase0_prompt": prompt0,
                        "phase0_prompt_hash": canonical_hash(prompt0),
                        "phase0_raw_output": raw_output,
                        "phase0_raw_answer": answer,
                        "phase0_normalized_answer": normalized,
                        "phase0_answer_token_ids": answer_ids,
                        "phase0_generated_token_ids": tokens,
                        "phase0_eos_generated": eos,
                        "phase0_generation_config": config["phase0_generation"],
                        "phase0_answer_fingerprint": canonical_hash(answer),
                        "parse_success": parsed,
                    }
                    append_jsonl(phase0_path, phase0_row)
                    phase0[case_id] = phase0_row
                if phase0_row.get("status") != "completed":
                    raise ValueError("Phase 0 answer parser failed")

                answer = str(phase0_row["phase0_raw_answer"])
                prompt1 = phase1_prompt(case.question, case.text_clue, answer)
                messages1 = runtime.build_messages(prompt1, spec["image_path"], SA_PREFILL)
                rendered1, inputs1 = runtime.prepare(messages1, SA_PREFILL)
                located = locate_phase1_positions(runtime.processor, rendered1, inputs1, answer)
                position_indices = {
                    name: int(located[name]["processed_index"]) for name in POSITIONS
                }
                sac = position_indices["P1_SAC"]
                captured = run_capture_forward(
                    runtime.model, inputs1, runtime.modules, position_indices, layers, [sac]
                )
                score = soft_sa_from_logits(captured.logits_by_position[sac], class_ids)
                generated_ids, generated_text, _ = runtime.generate(inputs1, 1, class_ids)
                valid = generated_text in set(map(str, range(9)))
                if not valid:
                    raise ValueError(f"Invalid constrained Phase 1 output: {generated_text!r}")
                if int(generated_text) != score["argmax_hard_class"]:
                    raise ValueError("Phase 1 generation differs from clean forward argmax")
                arrays = {
                    f"{position}__L{layer}": captured.hidden_by_name[position][layer]
                    .numpy().astype(np.float32)
                    for position in POSITIONS
                    for layer in layers
                }
                if len(arrays) != len(POSITIONS) * len(layers):
                    raise RuntimeError("Capture did not produce the complete position/layer grid")
                hidden_rel = Path("capture") / "hidden" / f"{case_id}.npz"
                _atomic_npz(output_root / hidden_rel, arrays)
                normalized = normalize_answer(answer)
                result = {
                    "status": "completed",
                    "case_id": case_id,
                    "item_id": case.item_id,
                    "prior_index": case.prior_index,
                    "condition": spec["condition"],
                    "version": "v4_gemma",
                    "question": case.question,
                    "text_clue": case.text_clue,
                    "image_path": spec["image_path"],
                    "image_sha256": phase0_row["image_sha256"],
                    "phase0_raw_answer": answer,
                    "phase0_normalized_answer": normalized,
                    "phase0_answer_fingerprint": phase0_row["phase0_answer_fingerprint"],
                    "phase1_inserted_raw_answer": answer,
                    "phase1_inserted_normalized_answer": normalize_answer(answer),
                    "phase1_prompt": prompt1,
                    "phase1_prompt_hash": canonical_hash(prompt1),
                    "phase1_answer_span": located["phase1_answer_span"],
                    "phase1_answer_token_ids": located["phase1_answer_token_ids"],
                    "positions": located,
                    **score,
                    "raw_generated_class": generated_text,
                    "valid_class": valid,
                    "generated_token_ids": generated_ids,
                    "phase1_generation_config": config["phase1_generation"],
                    "hidden_file": str(hidden_rel),
                    "hidden_key_count": len(arrays),
                    "phase0_correct": normalized == case.ground_truth_answer,
                    "answer_matches_text": normalized == case.text_answer,
                    "answer_matches_image": normalized == case.conflict_answer,
                    "answer_length": len(answer),
                    "elapsed_ordinal": ordinal,
                }
                append_jsonl(results_path, result)
                completed.add(case_id)
            except Exception as exc:
                failures += 1
                append_jsonl(
                    results_path,
                    {
                        "status": "failed",
                        "case_id": case_id,
                        "item_id": case.item_id,
                        "prior_index": case.prior_index,
                        "condition": spec["condition"],
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    },
                )
                if failures / max(1, ordinal) > ERROR_RATE_LIMIT:
                    raise RuntimeError(
                        f"Capture failure rate exceeded {ERROR_RATE_LIMIT:.0%} at item {ordinal}"
                    ) from exc
            if ordinal % 5 == 0:
                atomic_json(
                    capture_dir / "progress.json",
                    {
                        "completed": len(completed),
                        "failed_this_run": failures,
                        "total": len(rows),
                        "elapsed_seconds": time.time() - started,
                    },
                )
        all_results = load_jsonl(results_path)
        summary = {
            "status": "complete",
            "total": len(rows),
            "completed": len([row for row in all_results if row.get("status") == "completed"]),
            "failed_this_run": failures,
            "positions": list(POSITIONS),
            "layers": list(layers),
            "hidden_keys_per_sample": len(POSITIONS) * len(layers),
        }
        atomic_json(capture_dir / "progress.json", summary)
        atomic_json(capture_dir / "summary.json", summary)
        return summary
    finally:
        if pid_path.exists() and pid_path.read_text().strip() == str(os.getpid()):
            pid_path.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture Gemma delayed-SA activations")
    parser.add_argument("--output-root", default=str(RESULTS_ROOT))
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--layers", nargs="+", type=int, default=list(CAPTURE_LAYERS))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    run_capture(
        output_root=Path(args.output_root),
        max_items=args.max_items,
        max_samples=args.max_samples,
        resume=args.resume,
        layers=args.layers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
