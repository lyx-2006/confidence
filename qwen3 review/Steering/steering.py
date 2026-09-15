from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    _review = Path(__file__).resolve().parents[1]
    for _path in (_review.parent, _review):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))

import numpy as np
import torch

from dp_SA.io_utils import append_jsonl, atomic_json, atomic_jsonl, canonical_hash, load_jsonl
from dp_SA.positions import locate_phase1_positions
from dp_SA.prompts import SA_PREFILL
from dp_SA.selection import record_key
from dp_SA.soft_score import class_token_ids, soft_sa_from_logits
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.model_adapter import AdditiveActivationHook, model_input_device, resolve_language_modules, run_logits_forward

from Steering.capture import _acquire_pid, _messages
from Steering.config import (
    CAPTURE_ROOT, CONSTRUCTION_PER_SIDE, EXPECTED_HIDDEN_SIZE, EXPECTED_NUM_HIDDEN_LAYERS,
    LOGIT_PARITY_TOLERANCE, MODEL_PATH, POSITION_KEYS, SEED,
    STEERING_OUTPUT_ROOT, IMAGE_TEST_COUNT, TEXT_TEST_COUNT, VECTOR_NORM_FRACTION,
)
from Steering.contracts import ensure_fingerprinted_config, hidden_key, parse_alphas, parse_layers, parse_positions
from Steering.runtime import load_qwen3_inference


def _load_hidden(capture_root: Path, row: dict[str, Any], position: str, layer: int) -> np.ndarray:
    key = hidden_key(position, layer)
    path = capture_root / row["hidden_file"]
    with np.load(path) as payload:
        if key not in payload:
            raise KeyError(f"{path} does not contain {key}")
        value = np.asarray(payload[key], dtype=np.float32)
    if value.shape != (EXPECTED_HIDDEN_SIZE,) or not np.isfinite(value).all():
        raise ValueError(f"Invalid hidden state {key} in {path}: shape={value.shape}")
    return value


def scaled_direction(high: np.ndarray, low: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    if high.ndim != 2 or low.ndim != 2 or high.shape[1:] != low.shape[1:]:
        raise ValueError("High/low hidden matrices must have matching [samples, hidden] shapes")
    if not len(high) or not len(low) or not np.isfinite(high).all() or not np.isfinite(low).all():
        raise ValueError("High/low hidden matrices must be non-empty and finite")
    combined = np.concatenate([high, low])
    raw = high.mean(0) - low.mean(0)
    raw_norm = float(np.linalg.norm(raw))
    mean_residual_norm = float(np.linalg.norm(combined, axis=1).mean())
    target_norm = VECTOR_NORM_FRACTION * mean_residual_norm
    if not all(math.isfinite(value) and value > 0 for value in (raw_norm, mean_residual_norm, target_norm)):
        raise ValueError("Direction or residual norm is zero/non-finite")
    scaled = (raw / raw_norm * target_norm).astype(np.float32)
    return scaled, {
        "raw_vector_norm": raw_norm,
        "mean_residual_norm": mean_residual_norm,
        "target_vector_norm": target_norm,
    }


def build_vectors(
    capture_root: Path,
    construction: Sequence[dict[str, Any]],
    *,
    positions: Sequence[str],
    layers: Sequence[int],
) -> tuple[dict[tuple[str, int], torch.Tensor], dict[str, Any], dict[str, Any]]:
    high_rows = [row for row in construction if row["construction_side"] == "high_image"]
    low_rows = [row for row in construction if row["construction_side"] == "high_text"]
    vectors: dict[tuple[str, int], torch.Tensor] = {}
    records: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}
    for position in positions:
        for layer in layers:
            high = np.stack([_load_hidden(capture_root, row, position, layer) for row in high_rows])
            low = np.stack([_load_hidden(capture_root, row, position, layer) for row in low_rows])
            scaled, norms = scaled_direction(high, low)
            raw = high.mean(0) - low.mean(0)
            vectors[(position, int(layer))] = torch.from_numpy(scaled)
            artifact_key = hidden_key(position, layer)
            artifacts[artifact_key] = {
                "raw_vector": torch.from_numpy(raw.astype(np.float32)),
                "scaled_vector": torch.from_numpy(scaled),
                "high_mean": torch.from_numpy(high.mean(0).astype(np.float32)),
                "low_mean": torch.from_numpy(low.mean(0).astype(np.float32)),
            }
            records.append({
                "position": position, "layer": int(layer), "direction_type": "true",
                **norms,
                "high_mean_norm": float(np.linalg.norm(high.mean(0))),
                "low_mean_norm": float(np.linalg.norm(low.mean(0))),
            })
    metadata = {
        "normalization_fraction": VECTOR_NORM_FRACTION,
        "construction_fingerprint": canonical_hash([row["case_id"] for row in construction]),
        "vectors": records,
    }
    return vectors, metadata, artifacts


def _save_torch_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def smoke_manifests(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (float(row["soft_sa_image_score"]), record_key(row)))
    used: set[str] = set()
    low: list[dict[str, Any]] = []
    high: list[dict[str, Any]] = []
    for row in ordered:
        if str(row["item_id"]) not in used:
            low.append({**row, "construction_side": "high_text"})
            used.add(str(row["item_id"]))
        if len(low) == 5:
            break
    for row in reversed(ordered):
        if str(row["item_id"]) not in used:
            high.append({**row, "construction_side": "high_image"})
            used.add(str(row["item_id"]))
        if len(high) == 5:
            break
    test: list[dict[str, Any]] = []
    candidates = list(rows)
    random.Random(SEED).shuffle(candidates)
    for row in candidates:
        if str(row["item_id"]) in used:
            continue
        test.append({**row, "test_side": "smoke"})
        used.add(str(row["item_id"]))
        if len(test) == 10:
            break
    if len(low) != 5 or len(high) != 5 or len(test) != 10:
        raise ValueError("Smoke capture needs at least 20 item-disjoint completed records")
    return high + low, sorted(test, key=record_key), {
        "smoke": True, "construction": 10, "test": 10, "seed": SEED,
    }


def _take_extreme(
    rows: Sequence[dict[str, Any]], count: int, *, reverse: bool, used: set[str]
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: ((-1 if reverse else 1) * float(row["soft_sa_image_score"]), record_key(row)),
    )
    selected: list[dict[str, Any]] = []
    for row in ordered:
        item_id = str(row["item_id"])
        if item_id in used:
            continue
        selected.append(dict(row))
        used.add(item_id)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Could not select {count} item-disjoint extreme records; found {len(selected)}")
    return selected


def asymmetric_manifests(
    rows: Sequence[dict[str, Any]], *,
    construction_per_side: int = CONSTRUCTION_PER_SIDE,
    image_test_count: int = IMAGE_TEST_COUNT,
    text_test_count: int = TEXT_TEST_COUNT,
    seed: int = SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Select two extreme construction groups and asymmetric item-disjoint tests."""
    eligible = [row for row in rows if row.get("status") == "completed" and row.get("valid_class", True)]
    used: set[str] = set()
    high_image = _take_extreme(eligible, construction_per_side, reverse=True, used=used)
    high_text = _take_extreme(eligible, construction_per_side, reverse=False, used=used)
    construction: list[dict[str, Any]] = []
    for side, group in (("high_image", high_image), ("high_text", high_text)):
        for rank, row in enumerate(group, 1):
            construction.append({**row, "construction_side": side, "selection_rank": rank})

    candidates: list[dict[str, Any]] = []
    for row in sorted(eligible, key=record_key):
        if str(row["item_id"]) in used:
            continue
        hard_class = int(row["argmax_hard_class"])
        side = "image_side" if hard_class in (5, 6, 7, 8) else "text_side" if hard_class in (0, 1, 2, 3) else None
        if side is not None:
            candidates.append({**row, "test_side": side})
    targets = {"image_side": int(image_test_count), "text_side": int(text_test_count)}
    counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    test_items: set[str] = set()
    # Text-side has the smaller candidate pool, so reserve it first; this
    # guarantees the requested 31 text examples before filling image-side.
    rng = random.Random(seed)
    permutation = list(range(len(candidates)))
    rng.shuffle(permutation)
    for side in ("text_side", "image_side"):
        for permutation_index in permutation:
            row = candidates[permutation_index]
            item_id = str(row["item_id"])
            if row["test_side"] != side or item_id in test_items:
                continue
            counts[side] += 1
            test_items.add(item_id)
            selected.append({
                **row, "random_permutation_index": permutation_index,
                "selection_rank": counts[side],
            })
            if counts[side] == targets[side]:
                break
    if any(counts[key] != value for key, value in targets.items()):
        unique_by_side = {
            side: len({str(row["item_id"]) for row in candidates if row["test_side"] == side})
            for side in targets
        }
        raise ValueError(f"Insufficient item-disjoint test records: selected={dict(counts)}, candidates={unique_by_side}")
    construction_items = {str(row["item_id"]) for row in construction}
    if construction_items & test_items:
        raise AssertionError("Construction/test item leakage")

    def distributions(records: Sequence[dict[str, Any]], group_field: str) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for group in sorted({str(row[group_field]) for row in records}):
            values = [row for row in records if str(row[group_field]) == group]
            output[group] = {
                "condition_counts": dict(Counter(str(row["condition"]) for row in values)),
                "argmax_class_counts": dict(Counter(str(row["argmax_hard_class"]) for row in values)),
                "correct_count": sum(bool(row.get("phase0_correct")) for row in values),
                "answer_length_mean": sum(int(row.get("answer_length", 0)) for row in values) / len(values),
                "soft_sa_min": min(float(row["soft_sa_image_score"]) for row in values),
                "soft_sa_mean": sum(float(row["soft_sa_image_score"]) for row in values) / len(values),
                "soft_sa_max": max(float(row["soft_sa_image_score"]) for row in values),
            }
        return output

    summary = {
        "seed": seed,
        "construction_counts": dict(Counter(row["construction_side"] for row in construction)),
        "test_counts": dict(counts),
        "construction_item_count": len(construction_items),
        "test_item_count": len(test_items),
        "test_selection": "seeded_random_permutation_asymmetric_without_soft_score_sorting",
        "test_class_counts": dict(Counter(str(row["argmax_hard_class"]) for row in selected)),
        "construction_distributions": distributions(construction, "construction_side"),
        "test_distributions": distributions(selected, "test_side"),
    }
    return construction, sorted(selected, key=record_key), summary


def prepare_manifests(capture_root: Path, *, smoke: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = [row for row in load_jsonl(capture_root / "results.jsonl") if row.get("status") == "completed"]
    if not rows:
        raise ValueError(f"No completed capture rows found in {capture_root}")
    return smoke_manifests(rows) if smoke else asymmetric_manifests(rows)


def prediction_key(row: dict[str, Any]) -> str:
    return f'{row["case_id"]}|{row["position"]}|L{row["layer"]}|a{float(row["alpha"]):g}'


def _assert_capture_matches_runtime(
    capture_config: dict[str, Any], model_path: Path, positions: Sequence[str], layers: Sequence[int]
) -> None:
    if Path(capture_config.get("model", "")).resolve() != model_path.resolve():
        raise ValueError("Steering model path differs from the capture model path")
    if capture_config.get("positions") != ["LAT", "PANL", "CLE", "PANL+1", "SAC"]:
        raise ValueError("Capture does not contain the required five-position contract")
    available_layers = set(map(int, capture_config.get("layers", [])))
    if not set(layers).issubset(available_layers):
        raise ValueError("Requested Steering layer is absent from capture")
    if not set(positions).issubset(set(capture_config["positions"])):
        raise ValueError("Requested Steering position is absent from capture")


def _assert_position_parity(row: dict[str, Any], located: dict[str, Any]) -> None:
    for name, internal in POSITION_KEYS.items():
        old, new = row["positions"][name], located[internal]
        if (int(old["processed_index"]), int(old["token_id"])) != (
            int(new["processed_index"]), int(new["token_id"]),
        ):
            raise RuntimeError(f"Processed position drift for {row['case_id']} at {name}")


def run_steering(
    *,
    positions: Sequence[str],
    layers: Sequence[int],
    alphas: Sequence[float],
    capture_root: Path = CAPTURE_ROOT,
    output_root: Path = STEERING_OUTPUT_ROOT,
    model_path: Path = MODEL_PATH,
    smoke: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    positions, layers, alphas = parse_positions(positions), parse_layers(layers), parse_alphas(alphas)
    capture_root, output_root, model_path = capture_root.resolve(), output_root.resolve(), model_path.resolve()
    capture_config_path = capture_root / "config.json"
    if not capture_config_path.is_file():
        raise FileNotFoundError(f"Capture config does not exist: {capture_config_path}")
    capture_config = json.loads(capture_config_path.read_text(encoding="utf-8"))
    _assert_capture_matches_runtime(capture_config, model_path, positions, layers)
    construction, test, selection = prepare_manifests(capture_root, smoke=smoke)
    config_payload = {
        "format_version": 1, "experiment": "qwen3_vl_delayed_sa_steering",
        "model": str(model_path), "capture_root": str(capture_root),
        "capture_fingerprint": capture_config["fingerprint"],
        "positions": list(positions), "layers": list(layers), "alphas": list(alphas),
        "grid_semantics": "cartesian_product", "direction_type": "true",
        "normalization_fraction": VECTOR_NORM_FRACTION, "smoke": bool(smoke),
        "seed": SEED, "construction_count": len(construction), "test_count": len(test),
        "construction_fingerprint": canonical_hash([row["case_id"] for row in construction]),
        "test_fingerprint": canonical_hash([row["case_id"] for row in test]),
        "attention_implementation": "sdpa", "native_qwen3_system_message": None,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    pid_path = output_root / "active.pid"
    _acquire_pid(pid_path, "Qwen3 Steering")
    try:
        config = ensure_fingerprinted_config(
            output_root / "config.json", config_payload, resume=resume, label="Steering"
        )
        atomic_jsonl(output_root / "construction_manifest.jsonl", construction)
        atomic_jsonl(output_root / "test_manifest.jsonl", test)
        atomic_json(output_root / "selection_summary.json", selection)
        vectors, metadata, artifacts = build_vectors(
            capture_root, construction, positions=positions, layers=layers
        )
        _save_torch_atomic(output_root / "vectors.pt", artifacts)
        atomic_json(output_root / "vector_metadata.json", metadata)

        predictions_path = output_root / "predictions.jsonl"
        existing = {
            prediction_key(row) for row in load_jsonl(predictions_path)
            if row.get("status") == "completed"
        }
        inference = load_qwen3_inference(model_path)
        modules = resolve_language_modules(inference.model)
        if (modules.num_hidden_layers, modules.hidden_size) != (
            EXPECTED_NUM_HIDDEN_LAYERS, EXPECTED_HIDDEN_SIZE,
        ):
            raise ValueError("Loaded model does not match the Qwen3-VL-8B architecture")
        tokenizer = getattr(inference.processor, "tokenizer", inference.processor)
        class_ids, device = class_token_ids(tokenizer), model_input_device(inference)
        total = len(test) * len(positions) * len(layers) * len(alphas)
        done, started = len(existing), time.time()
        for row in test:
            messages = _messages(row["phase1_prompt"], row["image_path"], SA_PREFILL)
            rendered = render_continued_assistant(inference.processor, messages, SA_PREFILL)
            if rendered.startswith("<|im_start|>system"):
                raise RuntimeError("Qwen3 native rendering unexpectedly added a system message")
            inputs = prepare_multimodal_inputs(inference.processor, messages, rendered, device=device)
            located = locate_phase1_positions(tokenizer, rendered, inputs, row["phase0_raw_answer"])
            _assert_position_parity(row, located)
            sequence_length = int(inputs.input_ids.shape[1])
            sac = int(located[POSITION_KEYS["SAC"]]["processed_index"])
            clean_logits = np.asarray(row["class_logits"], dtype=float)
            clean_probabilities = np.asarray(row["class_probabilities"], dtype=float)
            clean_score = float(row["soft_sa_image_score"])
            for position in positions:
                target = int(located[POSITION_KEYS[position]]["processed_index"])
                for layer in layers:
                    base = vectors[(position, layer)]
                    for alpha in alphas:
                        prototype = {
                            "case_id": row["case_id"], "item_id": row["item_id"],
                            "test_side": row["test_side"], "position": position,
                            "layer": int(layer), "direction_type": "true", "alpha": float(alpha),
                        }
                        if prediction_key(prototype) in existing:
                            continue
                        hook = AdditiveActivationHook(
                            modules, layer_index=layer, target_position=target,
                            steering_vector=base * float(alpha),
                            prefill_sequence_length=sequence_length,
                        )
                        with hook:
                            logits = run_logits_forward(inference.model, inputs, [sac], modules)[sac]
                        diagnostics = hook.diagnostics()
                        scored = soft_sa_from_logits(logits, class_ids)
                        if alpha == 0.0 and (
                            np.max(np.abs(np.asarray(scored["class_logits"]) - clean_logits)) > LOGIT_PARITY_TOLERANCE
                            or np.max(np.abs(np.asarray(scored["class_probabilities"]) - clean_probabilities)) > LOGIT_PARITY_TOLERANCE
                        ):
                            raise RuntimeError(f"Alpha-zero parity failed: {row['case_id']} {position} L{layer}")
                        assert hook.h_before is not None and hook.h_after is not None
                        before, after = hook.h_before.numpy(), hook.h_after.numpy()
                        cosine = float(np.dot(before, after) / (np.linalg.norm(before) * np.linalg.norm(after)))
                        norm_ratio = float(np.linalg.norm(after) / np.linalg.norm(before))
                        clean_hard = int(row["argmax_hard_class"])
                        steered_logits = np.asarray(scored["class_logits"], dtype=float)
                        clean_sorted, steered_sorted = np.sort(clean_logits), np.sort(steered_logits)
                        result = {
                            "status": "completed", **prototype,
                            "clean_soft_sa": clean_score,
                            "steered_soft_sa": scored["soft_sa_image_score"],
                            "delta_soft_sa": scored["soft_sa_image_score"] - clean_score,
                            "clean_argmax_class": clean_hard,
                            "steered_argmax_class": scored["argmax_hard_class"],
                            "class_logits": scored["class_logits"],
                            "class_probabilities": scored["class_probabilities"],
                            "clean_class_logits": clean_logits.tolist(),
                            "clean_argmax_logit_margin": float(clean_sorted[-1] - clean_sorted[-2]),
                            "steered_argmax_logit_margin": float(steered_sorted[-1] - steered_sorted[-2]),
                            "clean_class_logit_margin_after_steering": float(
                                steered_logits[clean_hard] - max(np.delete(steered_logits, clean_hard))
                            ),
                            "probability_sum": scored["probability_sum"],
                            "hard_class_changed": clean_hard != int(scored["argmax_hard_class"]),
                            "hard_class_delta": int(scored["argmax_hard_class"]) - clean_hard,
                            "ceiling_saturated": scored["soft_sa_image_score"] >= 0.95 - 1e-9,
                            "floor_saturated": scored["soft_sa_image_score"] <= 0.05 + 1e-9,
                            "hook_diagnostics": diagnostics,
                            "activation_cosine": cosine, "activation_norm_ratio": norm_ratio,
                        }
                        append_jsonl(predictions_path, result)
                        existing.add(prediction_key(result))
                        done += 1
                        if done % 25 == 0:
                            atomic_json(output_root / "progress.json", {
                                "completed_cells": done, "total_cells": total,
                                "fraction": done / total, "elapsed_seconds": time.time() - started,
                                "last": prototype,
                            })
        summary = {
            "status": "complete", "completed_cells": len(existing),
            "expected_cells": total, "grid": {
                "positions": list(positions), "layers": list(layers), "alphas": list(alphas),
            },
            "selection": selection,
        }
        if summary["completed_cells"] != total:
            raise RuntimeError(f"Steering grid incomplete: {summary['completed_cells']}/{total}")
        atomic_json(output_root / "summary.json", summary)
        atomic_json(output_root / "progress.json", summary | {"elapsed_seconds": time.time() - started})
        return summary
    finally:
        if pid_path.exists() and pid_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
            pid_path.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run parameterized Qwen3-VL activation Steering")
    parser.add_argument("--positions", nargs="+", required=True)
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--alphas", nargs="+", type=float, required=True)
    parser.add_argument("--capture-root", default=str(CAPTURE_ROOT))
    parser.add_argument("--output-root", default=str(STEERING_OUTPUT_ROOT))
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    run_steering(
        positions=args.positions, layers=args.layers, alphas=args.alphas,
        capture_root=Path(args.capture_root), output_root=Path(args.output_root),
        model_path=Path(args.model_path), smoke=args.smoke, resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
