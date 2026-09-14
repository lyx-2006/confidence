from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from dp_SA.prompts import SA_PREFILL
from experiment_config import (
    ALPHAS,
    CAPTURE_LAYERS,
    DEFAULT_SHUFFLED_LAYERS,
    MODEL_PATH,
    POSITIONS,
    RESULTS_ROOT,
    SEED,
    SMOKE_ALPHAS,
    SMOKE_LAYER,
    VECTOR_NORM_FRACTION,
)
from gemma_runtime import AdditiveActivationHook, GemmaRuntime, run_logits_forward
from io_utils import append_jsonl, atomic_json, atomic_jsonl, canonical_hash, load_jsonl
from positions import locate_phase1_positions
from selection import record_key, select_centered_manifests, select_manifests
from soft_score import class_token_ids, soft_sa_from_logits


def parse_steering_layers(values: Sequence[int] | None, smoke: bool = False) -> tuple[int, ...]:
    if smoke:
        return (SMOKE_LAYER,)
    if not values:
        raise ValueError("Formal steering requires --steering-layers")
    layers = tuple(map(int, values))
    if len(layers) != len(set(layers)):
        raise ValueError("Steering layers must be unique")
    invalid = [layer for layer in layers if layer not in CAPTURE_LAYERS]
    if invalid:
        raise ValueError(f"Steering layers must be within L6-L33: {invalid}")
    return layers


def parse_positions(values: Sequence[str] | None) -> tuple[str, ...]:
    positions = tuple(values or POSITIONS)
    if not positions or len(positions) != len(set(positions)):
        raise ValueError("Steering positions must be non-empty and unique")
    invalid = sorted(set(positions) - set(POSITIONS))
    if invalid:
        raise ValueError(f"Unsupported Gemma steering positions: {invalid}")
    return positions


def _load_hidden(root: Path, row: dict[str, Any], position: str, layer: int) -> np.ndarray:
    path = root / row["hidden_file"]
    if not path.is_file():
        raise FileNotFoundError(f"Captured hidden file not found: {path}")
    key = f"{position}__L{layer}"
    with np.load(path) as payload:
        if key not in payload:
            raise KeyError(f"Capture is missing required hidden key {key}: {path}")
        vector = np.asarray(payload[key], dtype=np.float32)
    return vector


def _save_torch_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def build_vectors(
    root: Path,
    construction: Sequence[dict[str, Any]],
    positions: Sequence[str],
    layers: Sequence[int],
    shuffled_layers: Sequence[int],
    seed: int = SEED,
) -> tuple[dict[tuple[str, int, str], torch.Tensor], dict[str, Any], dict[str, Any]]:
    high = [row for row in construction if row["construction_side"] == "high_image"]
    low = [row for row in construction if row["construction_side"] == "high_text"]
    if not high or not low:
        raise ValueError("Both high-image and high-text construction groups are required")
    vectors: dict[tuple[str, int, str], torch.Tensor] = {}
    metadata, artifacts = [], {}
    shuffled_set = set(map(int, shuffled_layers))
    for position in positions:
        for layer in layers:
            high_values = np.stack([_load_hidden(root, row, position, layer) for row in high])
            low_values = np.stack([_load_hidden(root, row, position, layer) for row in low])
            combined = np.concatenate([high_values, low_values])
            raw = high_values.mean(0) - low_values.mean(0)
            raw_norm = float(np.linalg.norm(raw))
            mean_residual_norm = float(np.linalg.norm(combined, axis=1).mean())
            target_norm = VECTOR_NORM_FRACTION * mean_residual_norm
            if not all(math.isfinite(value) and value > 0 for value in (raw_norm, mean_residual_norm, target_norm)):
                raise ValueError(f"Invalid direction norm at {position} L{layer}")
            scaled = (raw / raw_norm * target_norm).astype(np.float32)
            vectors[(position, layer, "true")] = torch.from_numpy(scaled)
            artifacts[f"{position}__L{layer}__true"] = {
                "raw_vector": torch.from_numpy(raw.astype(np.float32)),
                "scaled_vector": torch.from_numpy(scaled),
                "high_mean": torch.from_numpy(high_values.mean(0).astype(np.float32)),
                "low_mean": torch.from_numpy(low_values.mean(0).astype(np.float32)),
            }
            metadata.append(
                {
                    "position": position,
                    "layer": layer,
                    "direction_type": "true",
                    "raw_vector_norm": raw_norm,
                    "mean_residual_norm": mean_residual_norm,
                    "target_vector_norm": target_norm,
                }
            )
            if position == "P1_PANL" and layer in shuffled_set:
                labels = [1] * len(high_values) + [0] * len(low_values)
                random.Random(seed + layer).shuffle(labels)
                labels_array = np.asarray(labels)
                shuffled = combined[labels_array == 1].mean(0) - combined[labels_array == 0].mean(0)
                shuffled_norm = float(np.linalg.norm(shuffled))
                if not math.isfinite(shuffled_norm) or shuffled_norm <= 0:
                    raise ValueError(f"Invalid shuffled direction at L{layer}")
                scaled_shuffled = (shuffled / shuffled_norm * target_norm).astype(np.float32)
                vectors[(position, layer, "shuffled")] = torch.from_numpy(scaled_shuffled)
                artifacts[f"{position}__L{layer}__shuffled"] = {
                    "raw_vector": torch.from_numpy(shuffled.astype(np.float32)),
                    "scaled_vector": torch.from_numpy(scaled_shuffled),
                }
                metadata.append(
                    {
                        "position": position,
                        "layer": layer,
                        "direction_type": "shuffled",
                        "raw_vector_norm": shuffled_norm,
                        "mean_residual_norm": mean_residual_norm,
                        "target_vector_norm": target_norm,
                    }
                )
    return vectors, {
        "normalization_fraction": VECTOR_NORM_FRACTION,
        "vectors": metadata,
        "construction_fingerprint": canonical_hash([row["case_id"] for row in construction]),
    }, artifacts


def _smoke_manifests(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (float(row["soft_sa_image_score"]), record_key(row)))
    used: set[str] = set()
    low, high = [], []
    for row in ordered:
        item = str(row["item_id"])
        if item not in used:
            low.append({**row, "construction_side": "high_text"})
            used.add(item)
        if len(low) == 5:
            break
    for row in reversed(ordered):
        item = str(row["item_id"])
        if item not in used:
            high.append({**row, "construction_side": "high_image"})
            used.add(item)
        if len(high) == 5:
            break
    test = []
    for row in sorted(rows, key=record_key):
        item = str(row["item_id"])
        if item in used:
            continue
        test.append({**row, "test_side": "smoke"})
        used.add(item)
        if len(test) == 10:
            break
    if len(low) != 5 or len(high) != 5 or len(test) != 10:
        raise ValueError("Smoke requires 20 item-disjoint completed capture records")
    return high + low, test, {"smoke": True, "construction": 10, "test": 10}


def prepare_manifests(root: Path, smoke: bool, centered_test: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = [row for row in load_jsonl(root / "capture" / "results.jsonl") if row.get("status") == "completed"]
    if smoke:
        construction, test, summary = _smoke_manifests(rows)
    elif centered_test:
        construction, test, summary = select_centered_manifests(rows)
    else:
        construction, test, summary = select_manifests(rows)
    destination = root / "steering"
    atomic_jsonl(destination / "construction_manifest.jsonl", construction)
    atomic_jsonl(destination / "test_manifest.jsonl", test)
    atomic_json(destination / "selection_summary.json", summary)
    return construction, test, summary


def _key(row: dict[str, Any]) -> str:
    return (
        f'{row["case_id"]}|{row["position"]}|L{row["layer"]}|'
        f'{row["direction_type"]}|a{float(row["alpha"]):g}'
    )


def run_steering(
    *,
    output_root: Path = RESULTS_ROOT,
    steering_layers: Sequence[int] | None = None,
    shuffled_layers: Sequence[int] | None = None,
    positions: Sequence[str] | None = None,
    centered_test: bool = False,
    alphas: Sequence[float] | None = None,
    smoke: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    layers = parse_steering_layers(steering_layers, smoke)
    positions = parse_positions(positions)
    if smoke:
        controls: tuple[int, ...] = ()
        alphas = SMOKE_ALPHAS
    else:
        if shuffled_layers is None:
            controls = tuple(layer for layer in layers if layer in DEFAULT_SHUFFLED_LAYERS)
        else:
            requested_controls = tuple(map(int, shuffled_layers))
            invalid_controls = sorted(set(requested_controls) - set(layers))
            if invalid_controls:
                raise ValueError(f"Shuffled layers must also be steering layers: {invalid_controls}")
            controls = tuple(layer for layer in layers if layer in set(requested_controls))
        if alphas is None:
            alphas = ALPHAS
        else:
            alphas = tuple(float(value) for value in alphas)
            if not alphas or len(alphas) != len(set(alphas)):
                raise ValueError("Steering alphas must be non-empty and unique")
            if 0.0 not in alphas or not any(value < 0 for value in alphas) or not any(value > 0 for value in alphas):
                raise ValueError("Steering alphas must include negative, zero, and positive values")
    steering_dir = output_root / "steering"
    steering_dir.mkdir(parents=True, exist_ok=True)
    pid_path = steering_dir / "active.pid"
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text())
            os.kill(pid, 0)
            raise RuntimeError(f"Steering already active: PID {pid}")
        except ProcessLookupError:
            pid_path.unlink()
    pid_path.write_text(str(os.getpid()))
    try:
        construction, test, selection = prepare_manifests(output_root, smoke, centered_test)
        vectors, vector_metadata, artifacts = build_vectors(
            output_root, construction, positions, layers, controls
        )
        _save_torch_atomic(steering_dir / "vectors.pt", artifacts)
        atomic_json(steering_dir / "vector_metadata.json", vector_metadata)
        config = {
            "format_version": 1,
            "model_family": "gemma3",
            "model": str(MODEL_PATH.resolve()),
            "smoke": smoke,
            "positions": list(positions),
            "capture_positions": list(POSITIONS),
            "layers": list(layers),
            "shuffled_layers": list(controls),
            "alphas": list(alphas),
            "test_count": len(test),
            "centered_test": centered_test,
            "seed": SEED,
            "construction_fingerprint": vector_metadata["construction_fingerprint"],
            "test_fingerprint": canonical_hash([row["case_id"] for row in test]),
        }
        config["fingerprint"] = canonical_hash(config)
        config_path = steering_dir / "config.json"
        if config_path.exists():
            previous = json.loads(config_path.read_text())
            if previous.get("fingerprint") != config["fingerprint"]:
                raise ValueError("Steering config changed; use a fresh output root")
            if not resume:
                raise FileExistsError("Steering output exists; pass --resume")
        else:
            atomic_json(config_path, config)
        predictions_path = steering_dir / "predictions.jsonl"
        existing = {
            _key(row) for row in load_jsonl(predictions_path) if row.get("status") == "completed"
        }
        runtime = GemmaRuntime(MODEL_PATH)
        class_ids = class_token_ids(runtime.processor.tokenizer)
        true_directions = [(position, layer, "true") for position in positions for layer in layers]
        control_directions = [("P1_PANL", layer, "shuffled") for layer in controls]
        directions = true_directions + control_directions
        total = len(test) * len(directions) * len(alphas)
        done, started = len(existing), time.time()
        for row in test:
            messages = runtime.build_messages(row["phase1_prompt"], row["image_path"], SA_PREFILL)
            rendered, inputs = runtime.prepare(messages, SA_PREFILL)
            located = locate_phase1_positions(
                runtime.processor, rendered, inputs, row["phase0_raw_answer"]
            )
            sequence_length = int(inputs.input_ids.shape[1])
            clean_logits = np.asarray(row["class_logits"], dtype=float)
            clean_probabilities = np.asarray(row["class_probabilities"], dtype=float)
            clean_score = float(row["soft_sa_image_score"])
            sac = int(located["P1_SAC"]["processed_index"])
            for position, layer, direction_type in directions:
                target = int(located[position]["processed_index"])
                base = vectors[(position, layer, direction_type)]
                for alpha in alphas:
                    proto = {
                        "case_id": row["case_id"],
                        "item_id": row["item_id"],
                        "test_side": row["test_side"],
                        "position": position,
                        "layer": layer,
                        "direction_type": direction_type,
                        "alpha": float(alpha),
                    }
                    if _key(proto) in existing:
                        continue
                    hook = AdditiveActivationHook(
                        runtime.modules, layer, target, base * float(alpha), sequence_length
                    )
                    with hook:
                        logits = run_logits_forward(runtime.model, inputs, [sac])[sac]
                    diagnostics = hook.diagnostics()
                    scored = soft_sa_from_logits(logits, class_ids)
                    if alpha == 0 and (
                        np.max(np.abs(np.asarray(scored["class_logits"]) - clean_logits)) > 1e-6
                        or np.max(np.abs(np.asarray(scored["class_probabilities"]) - clean_probabilities)) > 1e-6
                    ):
                        raise RuntimeError(f"Alpha-zero parity failed: {row['case_id']} {position} L{layer}")
                    assert hook.before is not None and hook.after is not None
                    before, after = hook.before.numpy(), hook.after.numpy()
                    cosine = float(np.dot(before, after) / (np.linalg.norm(before) * np.linalg.norm(after)))
                    ratio = float(np.linalg.norm(after) / np.linalg.norm(before))
                    clean_hard = int(row["argmax_hard_class"])
                    clean_sorted = np.sort(clean_logits)
                    steered_logits = np.asarray(scored["class_logits"], dtype=float)
                    steered_sorted = np.sort(steered_logits)
                    result = {
                        "status": "completed",
                        **proto,
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
                        "activation_cosine": cosine,
                        "activation_norm_ratio": ratio,
                    }
                    append_jsonl(predictions_path, result)
                    existing.add(_key(result))
                    done += 1
                    if done % 10 == 0:
                        atomic_json(
                            steering_dir / "progress.json",
                            {
                                "completed_cells": done,
                                "total_cells": total,
                                "fraction": done / total,
                                "elapsed_seconds": time.time() - started,
                                "last": proto,
                            },
                        )
        summary = {
            "status": "complete",
            "completed_cells": len(
                [row for row in load_jsonl(predictions_path) if row.get("status") == "completed"]
            ),
            "expected_cells": total,
            "true_direction_cells": len(test) * len(true_directions) * len(alphas),
            "selection": selection,
        }
        atomic_json(steering_dir / "progress.json", summary)
        atomic_json(steering_dir / "summary.json", summary)
        return summary
    finally:
        if pid_path.exists() and pid_path.read_text().strip() == str(os.getpid()):
            pid_path.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Gemma activation steering")
    parser.add_argument("--output-root", default=str(RESULTS_ROOT))
    parser.add_argument("--steering-layers", nargs="+", type=int)
    parser.add_argument("--positions", nargs="+")
    parser.add_argument("--shuffled-layers", nargs="*", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--centered-test", action="store_true")
    parser.add_argument("--alphas", nargs="+", type=float)
    args = parser.parse_args(argv)
    run_steering(
        output_root=Path(args.output_root),
        steering_layers=args.steering_layers,
        shuffled_layers=args.shuffled_layers,
        positions=args.positions,
        centered_test=args.centered_test,
        alphas=args.alphas,
        smoke=args.smoke,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
