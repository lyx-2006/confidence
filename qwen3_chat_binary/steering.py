from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from dp_SA.io_utils import append_jsonl, atomic_json, atomic_jsonl, canonical_hash, load_jsonl
from layer_metacognition.model_adapter import model_input_device, resolve_language_modules, run_logits_forward

from .adapters import AdditiveActivationHook
from .capture import acquire_pid
from .config import (
    CAPTURE_ROOT,
    CONSTRUCTION_PER_SIDE,
    DEFAULT_ALPHAS,
    DEFAULT_STEERING_POSITIONS,
    DEFAULT_STEERING_LAYERS,
    EXPECTED_HIDDEN_SIZE,
    EXPECTED_NUM_HIDDEN_LAYERS,
    IMAGE_TEST_COUNT,
    LOGIT_PARITY_TOLERANCE,
    MODEL_PATH,
    POSITIONS,
    SEED,
    STEERING_OUTPUT_ROOT,
    TEXT_TEST_COUNT,
    VARIANTS,
    VECTOR_NORM_FRACTION,
)
from .contracts import (
    ensure_fingerprinted_config,
    hidden_key,
    parse_alphas,
    parse_layers,
    parse_positions,
    parse_variants,
)
from .conversation import prepare_multimodal_inputs, render_stage2, stage2_messages
from .positions import locate_positions
from .prompts import ATTRIBUTION_TEMPLATE, LABELS, LABEL_WEIGHTS
from .runtime import load_qwen3_inference
from .scoring import attribution_score, label_token_ids
from .layout import (
    capture_config_path,
    capture_results_path,
    ensure_output_layout,
    steering_predictions_path,
)


def record_key(row: dict[str, Any]) -> str:
    return str(row["case_id"])


def attribution_side(row: dict[str, Any]) -> str:
    stored = row.get("predicted_side")
    if stored in {"text", "image", "tie"}:
        return str(stored)
    score = float(row["image_attribution_score"])
    if score == 0.5:
        return "tie"
    return "image" if score > 0.5 else "text"


def _load_completed(path: Path) -> dict[str, dict[str, Any]]:
    return {
        row["case_id"]: row for row in load_jsonl(path)
        if row.get("status") == "completed" and attribution_side(row) != "tie"
    }


def _take_extreme(
    rows: Sequence[dict[str, Any]], count: int, *, reverse: bool, used: set[str]
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            (-1 if reverse else 1) * float(row["image_attribution_score"]),
            record_key(row),
        ),
    )
    selected: list[dict[str, Any]] = []
    for row in ordered:
        case_id = str(row["case_id"])
        if case_id in used:
            continue
        selected.append(dict(row))
        used.add(case_id)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Could not select {count} case-disjoint extreme records; found {len(selected)}")
    return selected


def _take_central(
    rows: Sequence[dict[str, Any]], count: int, *, side: str, used: set[str]
) -> list[dict[str, Any]]:
    candidates = [
        row for row in rows
        if attribution_side(row) == side and str(row["case_id"]) not in used
    ]
    candidates.sort(key=lambda row: (
        abs(float(row["image_attribution_score"]) - 0.5),
        record_key(row),
    ))
    selected = [dict(row) for row in candidates[:count]]
    if len(selected) != count:
        raise ValueError(f"Could not select {count} central {side} records; found {len(selected)}")
    used.update(str(row["case_id"]) for row in selected)
    return selected


def shared_manifests(
    capture_root: Path,
    *,
    variants: Sequence[str],
    smoke: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if "native_boundary" not in variants:
        raise ValueError("Shared sample selection requires native_boundary")
    by_variant = {
        variant: _load_completed(capture_results_path(capture_root, variant))
        for variant in variants
    }
    common = set.intersection(*(set(rows) for rows in by_variant.values()))
    native = [by_variant["native_boundary"][case_id] for case_id in common]
    native.sort(key=record_key)
    construction_count = 5 if smoke else CONSTRUCTION_PER_SIDE
    used: set[str] = set()
    high_image = _take_extreme(native, construction_count, reverse=True, used=used)
    high_text = _take_extreme(native, construction_count, reverse=False, used=used)
    construction = [
        {**row, "construction_side": side, "selection_rank": rank}
        for side, group in (("high_image", high_image), ("high_text", high_text))
        for rank, row in enumerate(group, 1)
    ]

    targets = {"text_side": 5 if smoke else TEXT_TEST_COUNT, "image_side": 5 if smoke else IMAGE_TEST_COUNT}
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for manifest_side, score_side in (("text_side", "text"), ("image_side", "image")):
        group = _take_central(native, targets[manifest_side], side=score_side, used=used)
        for rank, row in enumerate(group, 1):
            selected.append({
                **row, "test_side": manifest_side, "selection_rank": rank,
                "distance_from_midpoint": abs(float(row["image_attribution_score"]) - 0.5),
            })
        counts[manifest_side] = len(group)
    test_cases = {str(row["case_id"]) for row in selected}
    construction_cases = {str(row["case_id"]) for row in construction}
    if construction_cases & test_cases:
        raise AssertionError("Construction/test case leakage")
    summary = {
        "selection_reference_variant": "native_boundary",
        "construction_rule": "lowest/highest clean image_attribution_score",
        "test_rule": "closest clean image_attribution_score to 0.5 within each side",
        "common_completed_case_count": len(common),
        "construction_counts": dict(Counter(row["construction_side"] for row in construction)),
        "test_counts": dict(counts), "requested_test_counts": targets,
        "test_shortfall": {key: max(0, targets[key] - counts[key]) for key in targets},
        "split_unit": "case_id",
        "construction_case_count": len(construction_cases), "test_case_count": len(test_cases),
        "seed": SEED, "smoke": bool(smoke),
    }
    return construction, sorted(selected, key=record_key), summary


def scaled_direction(high: np.ndarray, low: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    if high.ndim != 2 or low.ndim != 2 or high.shape[1:] != low.shape[1:]:
        raise ValueError("High/low hidden matrices must have matching [samples, hidden] shapes")
    if not len(high) or not len(low) or not np.isfinite(high).all() or not np.isfinite(low).all():
        raise ValueError("High/low hidden matrices must be non-empty and finite")
    combined = np.concatenate([high, low])
    raw = high.mean(0) - low.mean(0)
    raw_norm = float(np.linalg.norm(raw))
    residual_norm = float(np.linalg.norm(combined, axis=1).mean())
    target_norm = VECTOR_NORM_FRACTION * residual_norm
    if not all(math.isfinite(value) and value > 0 for value in (raw_norm, residual_norm, target_norm)):
        raise ValueError("Direction or residual norm is zero/non-finite")
    return (raw / raw_norm * target_norm).astype(np.float32), {
        "raw_vector_norm": raw_norm,
        "mean_residual_norm": residual_norm,
        "target_vector_norm": target_norm,
    }


def _load_hidden(capture_root: Path, variant: str, row: dict[str, Any], position: str, layer: int) -> np.ndarray:
    path = capture_root / variant / row["hidden_file"]
    key = hidden_key(position, layer)
    with np.load(path) as payload:
        if key not in payload:
            raise KeyError(f"{path} does not contain {key}")
        value = np.asarray(payload[key], dtype=np.float32)
    if value.shape != (EXPECTED_HIDDEN_SIZE,) or not np.isfinite(value).all():
        raise ValueError(f"Invalid hidden state {key} in {path}: {value.shape}")
    return value


def build_vectors(
    capture_root: Path,
    variant_rows: dict[str, dict[str, Any]],
    construction: Sequence[dict[str, Any]],
    *,
    variant: str,
    positions: Sequence[str],
    layers: Sequence[int],
) -> tuple[dict[tuple[str, int], torch.Tensor], dict[str, Any], dict[str, Any]]:
    high_ids = [row["case_id"] for row in construction if row["construction_side"] == "high_image"]
    low_ids = [row["case_id"] for row in construction if row["construction_side"] == "high_text"]
    vectors: dict[tuple[str, int], torch.Tensor] = {}
    artifacts: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for position in positions:
        for layer in layers:
            high = np.stack([
                _load_hidden(capture_root, variant, variant_rows[case_id], position, layer)
                for case_id in high_ids
            ])
            low = np.stack([
                _load_hidden(capture_root, variant, variant_rows[case_id], position, layer)
                for case_id in low_ids
            ])
            scaled, norms = scaled_direction(high, low)
            raw = high.mean(0) - low.mean(0)
            key = hidden_key(position, layer)
            vectors[(position, layer)] = torch.from_numpy(scaled)
            artifacts[key] = {
                "raw_vector": torch.from_numpy(raw.astype(np.float32)),
                "scaled_vector": torch.from_numpy(scaled),
                "high_mean": torch.from_numpy(high.mean(0).astype(np.float32)),
                "low_mean": torch.from_numpy(low.mean(0).astype(np.float32)),
            }
            records.append({"position": position, "layer": layer, **norms})
    return vectors, {"variant": variant, "normalization_fraction": VECTOR_NORM_FRACTION, "vectors": records}, artifacts


def save_torch_atomic(path: Path, value: Any) -> None:
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


def prediction_key(row: dict[str, Any]) -> str:
    return (
        f'{row["variant"]}|{row["case_id"]}|{row["position"]}|'
        f'L{int(row["layer"])}|a{float(row["alpha"]):g}'
    )


def _position_parity(saved: dict[str, Any], current: dict[str, Any], case_id: str) -> None:
    for position in POSITIONS:
        old, new = saved[position], current[position]
        if (int(old["processed_index"]), int(old["token_id"])) != (
            int(new["processed_index"]), int(new["token_id"]),
        ):
            raise RuntimeError(f"Position drift for {case_id} at {position}")


def run_steering(
    *,
    positions: Sequence[str] = DEFAULT_STEERING_POSITIONS,
    layers: Sequence[int] = DEFAULT_STEERING_LAYERS,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    variants: Sequence[str] = VARIANTS,
    capture_root: Path = CAPTURE_ROOT,
    output_root: Path = STEERING_OUTPUT_ROOT,
    model_path: Path = MODEL_PATH,
    smoke: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    positions, layers = parse_positions(positions), parse_layers(layers)
    alphas, variants = parse_alphas(alphas), parse_variants(variants)
    if set(variants) != set(VARIANTS):
        raise ValueError("The paired experiment requires both conversation variants")
    capture_root, output_root, model_path = capture_root.resolve(), output_root.resolve(), model_path.resolve()
    capture_configuration_path = capture_config_path(capture_root)
    if not capture_configuration_path.is_file():
        raise FileNotFoundError(f"Capture config does not exist: {capture_configuration_path}")
    capture_config = json.loads(capture_configuration_path.read_text(encoding="utf-8"))
    if Path(capture_config["model"]).resolve() != model_path:
        raise ValueError("Capture and steering model paths differ")
    if capture_config.get("phase1_template_hash") != canonical_hash(ATTRIBUTION_TEMPLATE):
        raise ValueError("Capture prompt is not the current v28 five-class attribution prompt")
    if capture_config.get("phase1_forward", {}).get("labels") != list(LABELS):
        raise ValueError("Capture labels are not the current five-class attribution labels")
    if not set(layers).issubset(set(capture_config["layers"])):
        raise ValueError("Requested steering layer is absent from capture")
    if not set(variants).issubset(set(capture_config["variants"])):
        raise ValueError("Requested variant is absent from capture")
    construction, test, selection = shared_manifests(capture_root, variants=variants, smoke=smoke)
    config_payload = {
        "format_version": 2, "experiment": "qwen3_vl_chat_fiveway_steering",
        "model": str(model_path), "capture_root": str(capture_root),
        "capture_fingerprint": capture_config["fingerprint"],
        "positions": list(positions), "layers": list(layers), "alphas": list(alphas),
        "variants": list(variants), "smoke": bool(smoke), "seed": SEED,
        "split_unit": "case_id",
        "normalization_fraction": VECTOR_NORM_FRACTION,
        "attribution_labels": list(LABELS),
        "probability_weights": list(LABEL_WEIGHTS),
        "signed_mapping": "2*image_attribution_score-1",
        "construction_fingerprint": canonical_hash([row["case_id"] for row in construction]),
        "test_fingerprint": canonical_hash([row["case_id"] for row in test]),
        "selection_summary": selection,
    }
    ensure_output_layout(output_root, variants)
    pid_path = output_root / "progress" / "active.pid"
    acquire_pid(pid_path, "Qwen3 chat-fiveway steering")
    try:
        config = ensure_fingerprinted_config(
            output_root / "progress" / "config.json", config_payload, resume=resume, label="Steering"
        )
        atomic_jsonl(output_root / "tables" / "construction_manifest.jsonl", construction)
        atomic_jsonl(output_root / "tables" / "test_manifest.jsonl", test)
        atomic_json(output_root / "tables" / "selection_summary.json", selection)
        rows_by_variant = {
            variant: _load_completed(capture_results_path(capture_root, variant))
            for variant in variants
        }
        vector_sets: dict[str, dict[tuple[str, int], torch.Tensor]] = {}
        for variant in variants:
            vectors, metadata, artifacts = build_vectors(
                capture_root, rows_by_variant[variant], construction,
                variant=variant, positions=positions, layers=layers,
            )
            vector_sets[variant] = vectors
            save_torch_atomic(output_root / variant / "tables" / "vectors.pt", artifacts)
            atomic_json(output_root / variant / "progress" / "vector_metadata.json", metadata)

        existing_rows = {
            variant: [
                row for row in load_jsonl(steering_predictions_path(output_root, variant))
                if row.get("status") == "completed"
            ]
            for variant in variants
        }
        existing = {
            prediction_key(row) for rows in existing_rows.values() for row in rows
        }
        variant_done = Counter({variant: len(rows) for variant, rows in existing_rows.items()})
        inference = load_qwen3_inference(model_path)
        modules = resolve_language_modules(inference.model)
        if (modules.num_hidden_layers, modules.hidden_size) != (
            EXPECTED_NUM_HIDDEN_LAYERS, EXPECTED_HIDDEN_SIZE,
        ):
            raise ValueError("Loaded model does not match Qwen3-VL-8B architecture")
        tokenizer = getattr(inference.processor, "tokenizer", inference.processor)
        attribution_ids = label_token_ids(tokenizer)
        device = model_input_device(inference)
        done, started = len(existing), time.time()
        total = len(test) * len(variants) * len(positions) * len(layers) * len(alphas)
        for manifest_row in test:
            case_id = manifest_row["case_id"]
            for variant in variants:
                row = rows_by_variant[variant][case_id]
                messages = stage2_messages(
                    row["phase0_prompt"], row["image_path"], row["phase0_raw_output"], variant
                )
                rendered = render_stage2(inference.processor, messages)
                inputs = prepare_multimodal_inputs(
                    inference.processor, messages, rendered, device=device
                )
                located = locate_positions(
                    tokenizer, rendered, inputs, row["phase0_raw_output"], variant
                )
                _position_parity(row["positions"], located["positions"], case_id)
                sequence_length = int(inputs.input_ids.shape[1])
                sac = int(located["indices"]["SAC"])
                clean_score = float(row["image_attribution_score"])
                clean_logits = np.asarray([row["label_logits"][label] for label in LABELS], dtype=float)
                clean_probs = np.asarray([row["label_probabilities"][label] for label in LABELS], dtype=float)
                for position in positions:
                    target = int(located["indices"][position])
                    for layer in layers:
                        base = vector_sets[variant][(position, layer)]
                        for alpha in alphas:
                            prototype = {
                                "variant": variant, "case_id": case_id,
                                "test_side": manifest_row["test_side"],
                                "position": position, "layer": int(layer), "alpha": float(alpha),
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
                            if int(diagnostics.get("steering_applied_count", -1)) != 1:
                                raise RuntimeError(f"Steering hook did not apply exactly once: {diagnostics}")
                            scored = attribution_score(logits, attribution_ids)
                            steered_logits = np.asarray([
                                scored["label_logits"][label] for label in LABELS
                            ], dtype=float)
                            steered_probs = np.asarray([
                                scored["label_probabilities"][label] for label in LABELS
                            ], dtype=float)
                            max_logit_delta = float(np.max(np.abs(steered_logits - clean_logits)))
                            max_probability_delta = float(np.max(np.abs(steered_probs - clean_probs)))
                            if alpha == 0.0 and max(max_logit_delta, max_probability_delta) > LOGIT_PARITY_TOLERANCE:
                                raise RuntimeError(f"Alpha-zero parity failed: {case_id} {variant} {position} L{layer}")
                            negative_control = layer == 35 and position != "SAC"
                            if negative_control and max(max_logit_delta, max_probability_delta) > LOGIT_PARITY_TOLERANCE:
                                raise RuntimeError(
                                    f"Final-layer historical-position negative control failed: {case_id} {variant} {position}"
                                )
                            assert hook.h_before is not None and hook.h_after is not None
                            before, after = hook.h_before.numpy(), hook.h_after.numpy()
                            denominator = float(np.linalg.norm(before) * np.linalg.norm(after))
                            result = {
                                "status": "completed", **prototype,
                                "clean_image_attribution_score": clean_score,
                                "steered_image_attribution_score": scored["image_attribution_score"],
                                "delta_image_attribution_score": scored["image_attribution_score"] - clean_score,
                                "clean_predicted_label": row["predicted_label"],
                                "steered_predicted_label": scored["predicted_label"],
                                "label_changed": scored["predicted_label"] != row["predicted_label"],
                                "clean_predicted_side": attribution_side(row),
                                "steered_predicted_side": scored["predicted_side"],
                                "side_changed": scored["predicted_side"] != attribution_side(row),
                                "clean_signed_attribution_score": row["signed_attribution_score"],
                                "steered_signed_attribution_score": scored["signed_attribution_score"],
                                "delta_signed_attribution_score": (
                                    scored["signed_attribution_score"] - row["signed_attribution_score"]
                                ),
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
                                "activation_cosine": float(np.dot(before, after) / denominator) if denominator else None,
                                "activation_norm_ratio": float(np.linalg.norm(after) / np.linalg.norm(before)) if np.linalg.norm(before) else None,
                                "hook_diagnostics": diagnostics,
                            }
                            append_jsonl(steering_predictions_path(output_root, variant), result)
                            existing.add(prediction_key(result))
                            done += 1
                            variant_done[variant] += 1
                            if done % 100 == 0:
                                elapsed = time.time() - started
                                atomic_json(output_root / "progress" / "progress.json", {
                                    "completed": done, "total": total,
                                    "by_variant": dict(variant_done),
                                    "elapsed_seconds": elapsed,
                                })
                                for progress_variant in variants:
                                    atomic_json(
                                        output_root / progress_variant / "progress" / "progress.json",
                                        {
                                            "variant": progress_variant,
                                            "completed": variant_done[progress_variant],
                                            "expected": total // len(variants),
                                            "elapsed_seconds": elapsed,
                                        },
                                    )
        summary = {
            "status": "complete", "completed": done, "total": total,
            "config_fingerprint": config["fingerprint"],
            "elapsed_seconds": time.time() - started,
        }
        atomic_json(output_root / "progress" / "summary.json", summary)
        atomic_json(output_root / "progress" / "progress.json", summary)
        for variant in variants:
            variant_summary = {
                "status": "complete",
                "variant": variant,
                "completed": variant_done[variant],
                "expected": total // len(variants),
                "config_fingerprint": config["fingerprint"],
            }
            atomic_json(output_root / variant / "progress" / "summary.json", variant_summary)
            atomic_json(output_root / variant / "progress" / "progress.json", variant_summary)
        return summary
    finally:
        if pid_path.exists() and pid_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
            pid_path.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run paired Qwen3 chat-fiveway activation steering")
    parser.add_argument("--capture-root", type=Path, default=CAPTURE_ROOT)
    parser.add_argument("--output-root", type=Path, default=STEERING_OUTPUT_ROOT)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--positions", nargs="+", default=list(DEFAULT_STEERING_POSITIONS))
    parser.add_argument("--layers", nargs="+", type=int, default=list(DEFAULT_STEERING_LAYERS))
    parser.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    print(run_steering(
        positions=args.positions, layers=args.layers, alphas=args.alphas, variants=args.variants,
        capture_root=args.capture_root, output_root=args.output_root, model_path=args.model_path,
        smoke=args.smoke, resume=args.resume,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
