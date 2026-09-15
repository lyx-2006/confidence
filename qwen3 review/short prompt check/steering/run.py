from __future__ import annotations

import argparse
import json
import os
from collections import Counter
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

from dp_SA.config import MIDPOINTS
from dp_SA.io_utils import append_jsonl, atomic_json, atomic_jsonl, canonical_hash, load_jsonl, sha256_file
from dp_SA.selection import record_key
from dp_SA.soft_score import class_token_ids, soft_sa_from_logits
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.model_adapter import AdditiveActivationHook, model_input_device, resolve_language_modules, run_logits_forward
from Steering.capture import _acquire_pid, _messages
from Steering.config import EXPECTED_HIDDEN_SIZE, EXPECTED_NUM_HIDDEN_LAYERS, MODEL_PATH
from Steering.contracts import ensure_fingerprinted_config, hidden_key
from Steering.runtime import load_qwen3_inference
from Steering.steering import _take_extreme, build_vectors

from capture.positions import external_positions, locate_short_phase1_positions
from capture.short_prompt import SA_PREFILL
from steering.analyze import analyze


CAPTURE_ROOT = SHORT_ROOT / "output" / "capture"
OUTPUT_ROOT = SHORT_ROOT / "output" / "steering"
LONG_VECTOR_PATH = REVIEW_ROOT / "Steering" / "output" / "panl_lat_cle_asym31" / "vectors.pt"
POSITIONS = ("PANL", "LAT", "CLE")
LAYERS = (10, 14, 16, 20, 22, 24, 28)
ALPHAS = (-5.0, -2.0, 0.0, 2.0, 5.0)
MODES = ("short_rebuilt", "long_vector_to_short")
KEYS = {"LAT": "P1_LAT", "PANL": "P1_PANL", "CLE": "P1_CLASS_LIST_END", "SAC": "P1_SAC"}


def _prediction_key(row: dict[str, Any]) -> str:
    return f'{row["case_id"]}|{row["direction_mode"]}|{row["position"]}|L{row["layer"]}|a{float(row["alpha"]):g}'


def select_short_manifests(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    eligible = [row for row in rows if row.get("status") == "completed" and row.get("valid_class", True)]
    used: set[str] = set()
    high_image = _take_extreme(eligible, 25, reverse=True, used=used)
    high_text = _take_extreme(eligible, 25, reverse=False, used=used)
    construction = []
    for side, group in (("high_image", high_image), ("high_text", high_text)):
        for rank, row in enumerate(group, 1):
            construction.append({**row, "construction_side": side, "selection_rank": rank})

    ordered = sorted(
        (row for row in eligible if str(row["item_id"]) not in used),
        key=lambda row: (abs(float(row["soft_sa_image_score"]) - 0.5), record_key(row)),
    )
    test = []
    test_items: set[str] = set()
    for row in ordered:
        item = str(row["item_id"])
        if item in test_items:
            continue
        score = float(row["soft_sa_image_score"])
        side = "text_side" if score < .5 else "image_side" if score > .5 else "midpoint"
        test.append({
            **row, "test_side": side, "selection_rank": len(test) + 1,
            "distance_to_midpoint": abs(score - .5),
            "test_selection": "closest_to_soft_sa_midpoint",
        })
        test_items.add(item)
        if len(test) == 80:
            break
    if len(test) != 80:
        raise ValueError(f"Could not select 80 item-disjoint midpoint-nearest short cases; found {len(test)}")
    if used & test_items:
        raise AssertionError("Short construction/test item leakage")
    summary = {
        "seed": 42,
        "construction_counts": dict(Counter(row["construction_side"] for row in construction)),
        "test_count": len(test), "test_item_count": len(test_items),
        "test_selection": "80_item_disjoint_cases_minimizing_abs_soft_sa_minus_0.5",
        "test_side_counts": dict(Counter(row["test_side"] for row in test)),
        "test_hard_class_counts": dict(Counter(str(row["argmax_hard_class"]) for row in test)),
        "distance_to_midpoint_min": min(row["distance_to_midpoint"] for row in test),
        "distance_to_midpoint_mean": sum(row["distance_to_midpoint"] for row in test) / len(test),
        "distance_to_midpoint_max": max(row["distance_to_midpoint"] for row in test),
        "selected_soft_sa_min": min(float(row["soft_sa_image_score"]) for row in test),
        "selected_soft_sa_mean": sum(float(row["soft_sa_image_score"]) for row in test) / len(test),
        "selected_soft_sa_max": max(float(row["soft_sa_image_score"]) for row in test),
    }
    return construction, test, summary


def _long_vectors() -> dict[tuple[str, int], torch.Tensor]:
    payload = torch.load(LONG_VECTOR_PATH, map_location="cpu", weights_only=True)
    result: dict[tuple[str, int], torch.Tensor] = {}
    for position in POSITIONS:
        for layer in LAYERS:
            key = hidden_key(position, layer)
            vector = payload[key]["scaled_vector"].detach().float().reshape(-1)
            if vector.shape != (EXPECTED_HIDDEN_SIZE,) or not bool(torch.isfinite(vector).all()):
                raise ValueError(f"Invalid historical long vector: {key}")
            result[position, layer] = vector
    return result


def run(*, resume: bool = False, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    output_root = output_root.resolve(); output_root.mkdir(parents=True, exist_ok=True)
    capture_config = json.loads((CAPTURE_ROOT / "config.json").read_text(encoding="utf-8"))
    capture_rows = [row for row in load_jsonl(CAPTURE_ROOT / "results.jsonl") if row.get("status") == "completed"]
    if len(capture_rows) != 500:
        raise ValueError(f"Expected 500 short capture rows, found {len(capture_rows)}")
    try:
        construction, test, selection = select_short_manifests(capture_rows)
    except Exception as exc:
        atomic_json(output_root / "selection_failure.json", {
            "status": "failed", "error": {"type": type(exc).__name__, "message": str(exc)},
            "hard_class_counts": {str(k): sum(int(row["argmax_hard_class"]) == k for row in capture_rows) for k in range(9)},
        })
        raise
    payload = {
        "format_version": 1, "experiment": "qwen3_short_prompt_steering",
        "capture_root": str(CAPTURE_ROOT.resolve()), "capture_fingerprint": capture_config["fingerprint"],
        "selection_source": "short_clean_results_midpoint_nearest_80", "selection": selection,
        "positions": list(POSITIONS), "layers": list(LAYERS), "alphas": list(ALPHAS),
        "direction_modes": list(MODES), "normalization_fraction": 0.03,
        "long_vector_path": str(LONG_VECTOR_PATH.resolve()),
        "long_vector_sha256": sha256_file(LONG_VECTOR_PATH),
        "model": str(MODEL_PATH.resolve()), "attention_implementation": "sdpa",
        "bootstrap_repeats": 2000, "seed": 42,
    }
    config = ensure_fingerprinted_config(output_root / "config.json", payload, resume=resume, label="Short steering")
    atomic_jsonl(output_root / "construction_manifest.jsonl", construction)
    atomic_jsonl(output_root / "test_manifest.jsonl", test)
    atomic_json(output_root / "selection_summary.json", selection)
    short_vectors, short_metadata, short_artifacts = build_vectors(
        CAPTURE_ROOT, construction, positions=POSITIONS, layers=LAYERS
    )
    torch.save(short_artifacts, output_root / "short_vectors.pt")
    atomic_json(output_root / "short_vector_metadata.json", short_metadata)
    vectors = {"short_rebuilt": short_vectors, "long_vector_to_short": _long_vectors()}

    prediction_path = output_root / "predictions.jsonl"
    existing = {_prediction_key(row) for row in load_jsonl(prediction_path) if row.get("status") == "completed"}
    total = len(test) * len(MODES) * len(POSITIONS) * len(LAYERS) * len(ALPHAS)
    pid = output_root / "active.pid"; _acquire_pid(pid, "Short steering")
    started = time.time()
    try:
        runtime = load_qwen3_inference(MODEL_PATH)
        modules = resolve_language_modules(runtime.model)
        if (modules.num_hidden_layers, modules.hidden_size) != (EXPECTED_NUM_HIDDEN_LAYERS, EXPECTED_HIDDEN_SIZE):
            raise RuntimeError("Unexpected Qwen3 architecture")
        tokenizer = runtime.processor.tokenizer; ids = class_token_ids(tokenizer); device = model_input_device(runtime)
        for row in test:
            messages = _messages(row["phase1_prompt"], row["image_path"], SA_PREFILL)
            rendered = render_continued_assistant(runtime.processor, messages, SA_PREFILL)
            inputs = prepare_multimodal_inputs(runtime.processor, messages, rendered, device=device)
            located = locate_short_phase1_positions(tokenizer, rendered, inputs, row["phase0_raw_answer"])
            current = external_positions(located)
            for name in ("LAT", "PANL", "CLE", "SAC"):
                saved = row["positions"][name]
                if (saved["processed_index"], saved["token_id"]) != (current[name]["processed_index"], current[name]["token_id"]):
                    raise RuntimeError(f"Short processed position drift: {row['case_id']} {name}")
            sequence_length = int(inputs.input_ids.shape[1]); sac = int(current["SAC"]["processed_index"])
            clean_logits = np.asarray(row["class_logits"], dtype=float)
            clean_probabilities = np.asarray(row["class_probabilities"], dtype=float)
            for mode in MODES:
                for position in POSITIONS:
                    target = int(current[position]["processed_index"])
                    for layer in LAYERS:
                        base = vectors[mode][position, layer]
                        for alpha in ALPHAS:
                            prototype = {
                                "case_id": row["case_id"], "direction_mode": mode,
                                "position": position, "layer": layer, "alpha": alpha,
                            }
                            if _prediction_key(prototype) in existing:
                                continue
                            hook = AdditiveActivationHook(
                                modules, layer_index=layer, target_position=target,
                                steering_vector=base * alpha, prefill_sequence_length=sequence_length,
                            )
                            with hook:
                                vocab = run_logits_forward(runtime.model, inputs, [sac], modules)[sac]
                            score = soft_sa_from_logits(vocab, ids); diagnostics = hook.diagnostics()
                            logit_error = float(np.max(np.abs(np.asarray(score["class_logits"]) - clean_logits)))
                            probability_error = float(np.max(np.abs(np.asarray(score["class_probabilities"]) - clean_probabilities)))
                            if alpha == 0.0 and (logit_error > 1e-6 or probability_error > 1e-6):
                                raise RuntimeError(f"Alpha-zero parity failed: {prototype}")
                            clean_class = int(row["argmax_hard_class"]); steered_class = int(score["argmax_hard_class"])
                            result = {
                                "status": "completed", **prototype, "item_id": row["item_id"],
                                "test_side": row["test_side"], "test_answer": row["phase0_normalized_answer"],
                                "clean_soft_sa": float(row["soft_sa_image_score"]),
                                "steered_soft_sa": float(score["soft_sa_image_score"]),
                                "delta_soft_sa": float(score["soft_sa_image_score"]) - float(row["soft_sa_image_score"]),
                                "clean_argmax_class": clean_class, "steered_argmax_class": steered_class,
                                "clean_hard_midpoint": float(MIDPOINTS[clean_class]),
                                "steered_hard_midpoint": float(MIDPOINTS[steered_class]),
                                "delta_hard_midpoint": float(MIDPOINTS[steered_class] - MIDPOINTS[clean_class]),
                                "hard_class_changed": clean_class != steered_class,
                                "class_logits": score["class_logits"], "class_probabilities": score["class_probabilities"],
                                "clean_class_logits": clean_logits.tolist(),
                                "alpha_zero_logit_max_abs_error": logit_error if alpha == 0 else 0.0,
                                "alpha_zero_probability_max_abs_error": probability_error if alpha == 0 else 0.0,
                                "hook_diagnostics": diagnostics,
                            }
                            append_jsonl(prediction_path, result); existing.add(_prediction_key(result))
                            if len(existing) % 25 == 0:
                                atomic_json(output_root / "progress.json", {
                                    "status": "running", "completed_cells": len(existing), "total_cells": total,
                                    "elapsed_seconds": time.time() - started, "last": prototype,
                                })
        if len(existing) != total:
            raise RuntimeError(f"Short steering grid incomplete: {len(existing)}/{total}")
        analysis = analyze(prediction_path, output_root)
        result = {"status": "complete", "completed_cells": len(existing), "expected_cells": total,
                  "selection": selection, "analysis": analysis, "config_fingerprint": config["fingerprint"]}
        atomic_json(output_root / "completion.json", result)
        return result
    finally:
        if pid.exists() and pid.read_text(encoding="utf-8").strip() == str(os.getpid()): pid.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT); args = parser.parse_args(argv)
    print(json.dumps(run(resume=args.resume, output_root=args.output_root), ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
