from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from dp_SA.io_utils import append_jsonl, atomic_json, canonical_hash, load_jsonl, sha256_file
from layer_metacognition.model_adapter import model_input_device, resolve_language_modules, run_logits_forward

from .adapters import AdditiveActivationHook
from .capture import acquire_pid
from .config import (
    ANSWER_MATCHED_OUTPUT_ROOT,
    CAPTURE_ROOT,
    DEFAULT_ALPHAS,
    EXPECTED_HIDDEN_SIZE,
    EXPECTED_NUM_HIDDEN_LAYERS,
    LOGIT_PARITY_TOLERANCE,
    MODEL_PATH,
)
from .contracts import ensure_fingerprinted_config, parse_alphas, parse_layers, parse_positions
from .conversation import prepare_multimodal_inputs, render_stage2, stage2_messages
from .layout import capture_config_path, capture_results_path, ensure_output_layout
from .positions import locate_positions
from .prepare_answer_matched import ANSWER_MATCHED_LAYERS, ANSWER_MATCHED_POSITIONS, VARIANT, sa_group
from .prompts import ATTRIBUTION_TEMPLATE, LABELS
from .runtime import load_qwen3_inference
from .scoring import attribution_score, label_token_ids


def prediction_key(row: dict[str, Any]) -> str:
    return f'{row["case_id"]}|{row["position"]}|L{int(row["layer"])}|a{float(row["alpha"]):g}'


def _position_parity(saved: dict[str, Any], current: dict[str, Any], case_id: str) -> None:
    for position in ("LAT", "PANL", "CLE"):
        old, new = saved[position], current[position]
        if (int(old["processed_index"]), int(old["token_id"])) != (
            int(new["processed_index"]), int(new["token_id"]),
        ):
            raise RuntimeError(f"Position drift for {case_id} at {position}")


def _load_vectors(
    root: Path, positions: Sequence[str], layers: Sequence[int], answers: Sequence[str]
) -> tuple[dict[tuple[str, int, str], torch.Tensor], dict[tuple[str, int, str], dict[str, Any]]]:
    metadata_path = root / VARIANT / "tables" / "vector_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = {
        (str(row["position"]), int(row["layer"]), str(row["recipient_answer"])): row
        for row in metadata["vectors"]
    }
    vectors: dict[tuple[str, int, str], torch.Tensor] = {}
    selected: dict[tuple[str, int, str], dict[str, Any]] = {}
    for position in positions:
        for layer in layers:
            path = root / VARIANT / "tables" / "vectors" / f"{position}__L{layer}.npz"
            expected_rows = [rows[position, layer, answer] for answer in answers]
            if len({row["vector_file_sha256"] for row in expected_rows}) != 1:
                raise ValueError(f"Inconsistent vector file hashes for {position} L{layer}")
            if sha256_file(path) != expected_rows[0]["vector_file_sha256"]:
                raise ValueError(f"Vector file fingerprint mismatch: {path}")
            with np.load(path) as payload:
                for answer, row in zip(answers, expected_rows):
                    value = np.asarray(payload[row["scaled_key"]], dtype=np.float32)
                    if value.shape != (EXPECTED_HIDDEN_SIZE,) or not np.isfinite(value).all():
                        raise ValueError(f"Invalid vector {position} L{layer} {answer}")
                    vectors[position, layer, answer] = torch.from_numpy(value.copy())
                    selected[position, layer, answer] = row
    return vectors, selected


def _crossed_midpoint(clean: float, steered: float) -> bool:
    return bool((clean < 0.5 <= steered) or (clean > 0.5 >= steered))


def run(
    *, capture_root: Path = CAPTURE_ROOT, output_root: Path = ANSWER_MATCHED_OUTPUT_ROOT,
    model_path: Path = MODEL_PATH,
    positions: Sequence[str] = ANSWER_MATCHED_POSITIONS,
    layers: Sequence[int] = ANSWER_MATCHED_LAYERS,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    smoke: bool = False, resume: bool = False,
) -> dict[str, Any]:
    capture_root, output_root, model_path = capture_root.resolve(), output_root.resolve(), model_path.resolve()
    positions, layers, alphas = parse_positions(positions), parse_layers(layers), parse_alphas(alphas)
    prepare_config_path = output_root / "progress" / "prepare_config.json"
    if not prepare_config_path.is_file():
        raise FileNotFoundError("Run prepare_answer_matched before steering")
    prepare_config = json.loads(prepare_config_path.read_text(encoding="utf-8"))
    if not set(positions).issubset(prepare_config["positions"]) or not set(layers).issubset(prepare_config["layers"]):
        raise ValueError("Requested run grid is absent from prepared vectors")
    capture_configuration = json.loads(capture_config_path(capture_root).read_text(encoding="utf-8"))
    if Path(capture_configuration["model"]).resolve() != model_path:
        raise ValueError("Capture and answer-matched steering model paths differ")
    if capture_configuration.get("phase1_template_hash") != canonical_hash(ATTRIBUTION_TEMPLATE):
        raise ValueError("Capture prompt differs from the current five-class prompt")
    if capture_configuration.get("phase1_forward", {}).get("labels") != list(LABELS):
        raise ValueError("Capture labels differ from the current five-class labels")
    manifest_name = "smoke_manifest.jsonl" if smoke else "test_manifest.jsonl"
    test = load_jsonl(output_root / "tables" / manifest_name)
    capture_rows = {
        str(row["case_id"]): row for row in load_jsonl(capture_results_path(capture_root, VARIANT))
        if row.get("status") == "completed"
    }
    if any(str(row["case_id"]) not in capture_rows for row in test):
        raise ValueError("Test manifest references a missing capture row")
    answers = sorted({str(row["answer"]) for row in test})
    vectors, vector_rows = _load_vectors(output_root, positions, layers, answers)
    ensure_output_layout(output_root, (VARIANT,))
    predictions = output_root / VARIANT / "tables" / "predictions.jsonl"
    total = len(test) * len(positions) * len(layers) * len(alphas)
    config_payload = {
        "format_version": 1, "experiment": "native_answer_matched_steering_run",
        "prepare_fingerprint": prepare_config["fingerprint"], "model": str(model_path),
        "variant": VARIANT, "direction": "matched_loao", "smoke": bool(smoke),
        "positions": list(positions), "layers": list(layers), "alphas": list(alphas),
        "test_case_ids": [row["case_id"] for row in test], "expected_trials": total,
        "test_manifest_sha256": sha256_file(output_root / "tables" / manifest_name),
    }
    config = ensure_fingerprinted_config(
        output_root / "progress" / "run_config.json", config_payload,
        resume=resume, label="Answer-matched run",
    )
    existing_rows = [row for row in load_jsonl(predictions) if row.get("status") == "completed"]
    existing = {prediction_key(row) for row in existing_rows}
    if len(existing) != len(existing_rows):
        raise ValueError("Duplicate completed answer-matched predictions")
    pid_path = output_root / "progress" / "active.pid"
    acquire_pid(pid_path, "Native answer-matched steering")
    started = time.time()
    try:
        inference = load_qwen3_inference(model_path)
        modules = resolve_language_modules(inference.model)
        if (modules.num_hidden_layers, modules.hidden_size) != (EXPECTED_NUM_HIDDEN_LAYERS, EXPECTED_HIDDEN_SIZE):
            raise ValueError("Loaded model does not match Qwen3-VL-8B architecture")
        tokenizer = getattr(inference.processor, "tokenizer", inference.processor)
        attribution_ids = label_token_ids(tokenizer)
        device = model_input_device(inference)
        done = len(existing)
        for manifest_row in test:
            case_id = str(manifest_row["case_id"])
            answer = str(manifest_row["answer"])
            row = capture_rows[case_id]
            messages = stage2_messages(row["phase0_prompt"], row["image_path"], row["phase0_raw_output"], VARIANT)
            rendered = render_stage2(inference.processor, messages)
            inputs = prepare_multimodal_inputs(inference.processor, messages, rendered, device=device)
            located = locate_positions(tokenizer, rendered, inputs, row["phase0_raw_output"], VARIANT)
            _position_parity(row["positions"], located["positions"], case_id)
            sequence_length = int(inputs.input_ids.shape[1])
            sac = int(located["indices"]["SAC"])
            clean_score = float(row["image_attribution_score"])
            clean_logits = np.asarray([row["label_logits"][label] for label in LABELS], dtype=float)
            clean_probs = np.asarray([row["label_probabilities"][label] for label in LABELS], dtype=float)
            for position in positions:
                target = int(located["indices"][position])
                for layer in layers:
                    base = vectors[position, layer, answer]
                    vector_metadata = vector_rows[position, layer, answer]
                    for alpha in alphas:
                        prototype = {
                            "variant": VARIANT, "direction": "matched_loao", "case_id": case_id,
                            "test_answer": answer, "test_sa_group": str(manifest_row["sa_group"]),
                            "pair_type": str(manifest_row["pair_type"]),
                            "position": position, "layer": int(layer), "alpha": float(alpha),
                        }
                        key = prediction_key(prototype)
                        if key in existing:
                            continue
                        hook = AdditiveActivationHook(
                            modules, layer_index=layer, target_position=target,
                            steering_vector=base * float(alpha), prefill_sequence_length=sequence_length,
                        )
                        try:
                            with hook:
                                logits = run_logits_forward(inference.model, inputs, [sac], modules)[sac]
                            diagnostics = hook.diagnostics()
                            if int(diagnostics.get("steering_applied_count", -1)) != 1:
                                raise RuntimeError(f"Steering hook did not apply exactly once: {diagnostics}")
                            scored = attribution_score(logits, attribution_ids)
                            steered_logits = np.asarray([scored["label_logits"][label] for label in LABELS], dtype=float)
                            steered_probs = np.asarray([scored["label_probabilities"][label] for label in LABELS], dtype=float)
                            max_logit_delta = float(np.max(np.abs(steered_logits - clean_logits)))
                            max_probability_delta = float(np.max(np.abs(steered_probs - clean_probs)))
                            if alpha == 0.0 and max(max_logit_delta, max_probability_delta) > LOGIT_PARITY_TOLERANCE:
                                raise RuntimeError(f"Alpha-zero parity failed: {case_id} {position} L{layer}")
                            negative_control = layer == EXPECTED_NUM_HIDDEN_LAYERS - 1
                            if negative_control and max(max_logit_delta, max_probability_delta) > LOGIT_PARITY_TOLERANCE:
                                raise RuntimeError(f"Final-layer historical-position control failed: {case_id} {position}")
                            if hook.h_before is None or hook.h_after is None:
                                raise RuntimeError("Hook did not retain activation diagnostics")
                            before, after = hook.h_before.numpy(), hook.h_after.numpy()
                            result = {
                                "status": "completed", **prototype,
                                "included_answers": vector_metadata["included_answers"],
                                "vector_target_norm": vector_metadata["target_norm"],
                                "clean_image_attribution_score": clean_score,
                                "steered_image_attribution_score": scored["image_attribution_score"],
                                "delta_image_attribution_score": scored["image_attribution_score"] - clean_score,
                                "clean_predicted_label": row["predicted_label"],
                                "steered_predicted_label": scored["predicted_label"],
                                "label_changed": scored["predicted_label"] != row["predicted_label"],
                                "clean_sa_group": sa_group(row),
                                "steered_predicted_side": scored["predicted_side"],
                                "crossed_midpoint": _crossed_midpoint(clean_score, scored["image_attribution_score"]),
                                "clean_label_logits": row["label_logits"],
                                "clean_label_probabilities": row["label_probabilities"],
                                "steered_label_logits": scored["label_logits"],
                                "steered_label_probabilities": scored["label_probabilities"],
                                "clean_label_probability_mass": row["label_probability_mass"],
                                "steered_label_probability_mass": scored["label_probability_mass"],
                                "max_abs_logit_delta": max_logit_delta,
                                "max_abs_probability_delta": max_probability_delta,
                                "negative_control": negative_control,
                                "actual_perturbation_norm": float(np.linalg.norm(after - before)),
                                "expected_perturbation_norm": abs(float(alpha)) * float(vector_metadata["target_norm"]),
                                "hook_diagnostics": diagnostics,
                            }
                            append_jsonl(predictions, result)
                        except Exception as exc:
                            append_jsonl(output_root / VARIANT / "progress" / "failures.jsonl", {
                                "status": "failed", **prototype,
                                "error": {"type": type(exc).__name__, "message": str(exc)},
                            })
                            raise
                        existing.add(key)
                        done += 1
                        if done % 25 == 0 or done == total:
                            elapsed = time.time() - started
                            atomic_json(output_root / VARIANT / "progress" / "progress.json", {
                                "status": "running" if done < total else "complete",
                                "completed": done, "total": total, "elapsed_seconds": elapsed,
                                "forwards_per_second": (done - len(existing_rows)) / elapsed if elapsed else None,
                            })
        elapsed = time.time() - started
        summary = {
            "status": "complete", "completed": done, "total": total,
            "new_gpu_forwards": done - len(existing_rows), "elapsed_seconds": elapsed,
            "forwards_per_second": (done - len(existing_rows)) / elapsed if elapsed else None,
            "config_fingerprint": config["fingerprint"], "resumed_noop": done == len(existing_rows),
        }
        atomic_json(output_root / VARIANT / "progress" / "run_summary.json", summary)
        return summary
    finally:
        if pid_path.exists() and pid_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
            pid_path.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run native-boundary answer-matched LOAO steering")
    parser.add_argument("--capture-root", type=Path, default=CAPTURE_ROOT)
    parser.add_argument("--output-root", type=Path, default=ANSWER_MATCHED_OUTPUT_ROOT)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--positions", nargs="+", default=list(ANSWER_MATCHED_POSITIONS))
    parser.add_argument("--layers", nargs="+", type=int, default=list(ANSWER_MATCHED_LAYERS))
    parser.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    print(run(**vars(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
