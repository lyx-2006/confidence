from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import torch
import transformers

from confidence_test.runtime_imports import load_runtime
from dp_SA.checkpoint_steering.run import class_margin
from dp_SA.confidence_steering.processor import enforce_fast_image_processor, processor_identity
from dp_SA.confidence_steering.run import _messages, _predict, _score
from dp_SA.positions import locate_phase1_positions
from dp_SA.prompts import SA_PREFILL
from dp_SA.soft_score import class_token_ids
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.model_adapter import model_input_device, resolve_language_modules, run_hooked_forward

from .config import (
    DIRECTIONS, FROZEN_LAT_PROBE, FROZEN_PANL_SA_PROBE, HIDDEN_SIZE,
    INFERENCE_PATH, LAYER, MODEL_PATH, NONZERO_ALPHAS, PANL_LAYER,
    PANL_POSITION, POSITION, SAC_POSITION,
)
from .io_utils import (
    append_jsonl, array_hash, atomic_json, atomic_jsonl, atomic_npz,
    canonical_jsonable, load_jsonl, stable_shard,
)


def trial_key(row: dict[str, Any]) -> str:
    return f'{int(row["seed"])}|{row["case_id"]}|{row["direction"]}|a{float(row["alpha"]):g}'


def canonical_merge(rows: Sequence[dict[str, Any]], expected_keys: set[str] | None = None) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = trial_key(row)
        if key in unique and row != unique[key]:
            raise RuntimeError(f"Conflicting duplicate trial: {key}")
        unique[key] = row
    if expected_keys is not None and set(unique) != expected_keys:
        missing = sorted(expected_keys - set(unique))
        extra = sorted(set(unique) - expected_keys)
        raise RuntimeError(f"Trial coverage mismatch missing={missing[:3]} extra={extra[:3]}")
    return [unique[key] for key in sorted(unique)]


class StrictTargetHook:
    """One additive block-output intervention with an off-target bitwise proof."""

    def __init__(self, modules: Any, target_position: int, vector: torch.Tensor, sequence_length: int):
        if vector.numel() != modules.hidden_size or vector.numel() != HIDDEN_SIZE:
            raise ValueError("Steering vector hidden size mismatch")
        self.layer = modules.language_layers[LAYER]
        self.target_position = int(target_position)
        self.vector = vector.detach().reshape(-1)
        self.sequence_length = int(sequence_length)
        self.hook_call_count = 0
        self.applied_count = 0
        self.off_target_bitwise = True
        self.h_before: torch.Tensor | None = None
        self.h_after: torch.Tensor | None = None
        self._handle = None

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        self.hook_call_count += 1
        tensor = output if isinstance(output, torch.Tensor) else output[0]
        trailing = None if isinstance(output, torch.Tensor) else output[1:]
        if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[2] != HIDDEN_SIZE:
            raise RuntimeError(f"Unexpected decoder activation shape: {tuple(tensor.shape)}")
        if self.applied_count or int(tensor.shape[1]) != self.sequence_length:
            return output
        before_full = tensor
        patched = tensor.clone()
        self.h_before = tensor[0, self.target_position].detach().float().cpu()
        patched[0, self.target_position] += self.vector.to(device=patched.device, dtype=patched.dtype)
        self.h_after = patched[0, self.target_position].detach().float().cpu()
        left_equal = torch.equal(patched[:, :self.target_position], before_full[:, :self.target_position])
        right_equal = torch.equal(patched[:, self.target_position + 1:], before_full[:, self.target_position + 1:])
        other_batch_equal = torch.equal(patched[1:], before_full[1:])
        self.off_target_bitwise = bool(left_equal and right_equal and other_batch_equal)
        if not self.off_target_bitwise:
            raise RuntimeError("Hook modified a non-target token")
        self.applied_count += 1
        return patched if trailing is None else (patched, *trailing)

    def __enter__(self):
        self._handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self._handle is not None:
            self._handle.remove()
        return False


def _load_model():
    runtime = load_runtime(INFERENCE_PATH)
    inference = runtime.QwenVLInference(str(MODEL_PATH))
    enforce_fast_image_processor(inference.processor)
    identity = processor_identity(inference.processor)
    if not identity["is_fast"] or not identity["image_processor_class"].endswith("Qwen2VLImageProcessorFast"):
        raise RuntimeError(f"Explicit Fast processor gate failed: {identity}")
    image_config = canonical_jsonable(inference.processor.image_processor.to_dict())
    modules = resolve_language_modules(inference.model)
    tokenizer = getattr(inference.processor, "tokenizer", inference.processor)
    return inference, modules, tokenizer, class_token_ids(tokenizer), model_input_device(inference), {
        **identity, "image_processor_config": image_config,
        "transformers_version": transformers.__version__,
    }


def _locate(inference: Any, tokenizer: Any, row: dict[str, Any], device: torch.device):
    messages = _messages(row)
    rendered = render_continued_assistant(inference.processor, messages, SA_PREFILL)
    inputs = prepare_multimodal_inputs(inference.processor, messages, rendered, device=device)
    located = locate_phase1_positions(tokenizer, rendered, inputs, str(row["phase0_raw_answer"]))
    for name in (POSITION, PANL_POSITION, SAC_POSITION):
        expected = row["positions"][name]
        if int(located[name]["processed_index"]) != int(expected["processed_index"]):
            raise RuntimeError(f"Position index mismatch: {row['case_id']} {name}")
        if int(located[name]["token_id"]) != int(expected["token_id"]):
            raise RuntimeError(f"Position token mismatch: {row['case_id']} {name}")
    return inputs, located, rendered


def _probe(root: Path, seed: int, name: str):
    return joblib.load(root / f"artifacts/probes/seed_{seed}/{name}__full.joblib")["model"]


def _historical_hidden(case: str) -> tuple[np.ndarray, np.ndarray]:
    from dp_SA.confidence_steering.core import HiddenResolver
    resolver = HiddenResolver()
    return resolver.load(case, f"P1_LAT__L{LAYER}"), resolver.load(case, f"P1_PANL__L{PANL_LAYER}")


def _completed(root: Path) -> dict[str, dict[str, Any]]:
    rows = []
    for path in sorted((root / "artifacts/trials").glob("trials.shard_*.jsonl")):
        rows.extend(load_jsonl(path, repair_trailing=True))
    return {trial_key(row): row for row in canonical_merge(rows)}


def _endpoint_values(lat: np.ndarray, panl: np.ndarray, score: dict[str, Any], anchors: dict[str, Any], seed_probes: dict[str, Any]) -> dict[str, float]:
    return {
        "final_soft_sa": float(score["soft_sa_image_score"]),
        "anchor_panl_sa": _predict(anchors["panl_sa"], panl),
        "anchor_lat_confidence": _predict(anchors["lat_confidence"], lat),
        "seed_panl_sa": _predict(seed_probes["panl_sa"], panl),
        "seed_lat_confidence": _predict(seed_probes["lat_confidence"], lat),
    }


def _base_row(seed: int, row: dict[str, Any], direction: str, alpha: float, score: dict[str, Any], endpoints: dict[str, float], identity: dict[str, Any]) -> dict[str, Any]:
    logits = np.asarray(score["class_logits"], np.float64)
    hard = int(score["argmax_hard_class"])
    return {
        "status": "completed", "seed": seed, "case_id": str(row["case_id"]),
        "family_id": str(row["family_id"]), "item_id": str(row["item_id"]),
        "condition": str(row["condition"]), "fixed_answer": str(row["phase0_normalized_answer"]),
        "direction": direction, "alpha": float(alpha), "layer": LAYER, "position": POSITION,
        **endpoints,
        "sa_logits": score["class_logits"], "sa_probabilities": score["class_probabilities"],
        "hard_sa": hard, "class_margin": class_margin(logits, hard),
        "processor_identity": {key: identity[key] for key in ("image_processor_class", "is_fast", "min_pixels", "max_pixels", "transformers_version")},
    }


def run_worker(root: Path, worker: int, num_gpus: int, seeds: Sequence[int], fingerprint: str) -> dict[str, Any]:
    manifest = load_jsonl(root / "artifacts/trials/runtime_manifest.jsonl")
    selected = [row for row in manifest if stable_shard(str(row["case_id"]), num_gpus) == worker]
    completed = _completed(root)
    expected = {
        trial_key({"seed": seed, "case_id": row["case_id"], "direction": direction, "alpha": alpha})
        for seed in seeds for row in selected
        for direction, alpha in [("shared_alpha_zero", 0.0), *[(direction, alpha) for direction in DIRECTIONS for alpha in NONZERO_ALPHAS]]
    }
    if expected <= set(completed):
        return {"worker": worker, "new_gpu_forwards": 0, "resumed_noop": True}
    inference, modules, tokenizer, class_ids, device, identity = _load_model()
    anchors = {
        "lat_confidence": joblib.load(FROZEN_LAT_PROBE)["model"],
        "panl_sa": joblib.load(FROZEN_PANL_SA_PROBE)["model"],
    }
    seed_probes = {seed: {
        "lat_confidence": _probe(root, seed, "confidence_gap__P1_LAT__L14"),
        "panl_sa": _probe(root, seed, "final_sa__P1_PANL__L18"),
    } for seed in seeds}
    vector_files = {seed: np.load(root / f"artifacts/directions/seed_{seed}/P1_LAT__L14.npz") for seed in seeds}
    vector_meta = {}
    for seed in seeds:
        payload = json.loads((root / f"artifacts/directions/seed_{seed}/vector_metadata.json").read_text())
        vector_meta[seed] = {(row["recipient_answer"], row["direction"]): row for row in payload["vectors"]}
    output = root / f"artifacts/trials/trials.shard_{worker}.jsonl"
    forwards = 0
    parity_rows = []
    for manifest_row in selected:
        case = str(manifest_row["case_id"])
        inputs, located, rendered = _locate(inference, tokenizer, manifest_row, device)
        lat_position = int(located[POSITION]["processed_index"])
        panl_position = int(located[PANL_POSITION]["processed_index"])
        sac_position = int(located[SAC_POSITION]["processed_index"])
        sequence_length = int(inputs.input_ids.shape[1])
        historical_lat, historical_panl = _historical_hidden(case)
        for seed in seeds:
            zero_key = trial_key({"seed": seed, "case_id": case, "direction": "shared_alpha_zero", "alpha": 0.0})
            baseline_path = root / f"artifacts/trials/seed_{seed}/baseline/{case}.npz"
            if zero_key not in completed:
                hook = StrictTargetHook(modules, lat_position, torch.zeros(HIDDEN_SIZE), sequence_length)
                with hook:
                    forward = run_hooked_forward(inference.model, inputs, modules, {PANL_POSITION: panl_position}, logits_positions=[sac_position])
                forwards += 1
                if hook.hook_call_count != 1 or hook.applied_count != 1 or not hook.off_target_bitwise:
                    raise RuntimeError(f"Zero hook gate failed: seed={seed} case={case}")
                lat = hook.h_after.numpy().astype(np.float32)
                panl = forward.hidden_by_name[PANL_POSITION][PANL_LAYER].detach().float().cpu().numpy().astype(np.float32)
                score = _score(forward, sac_position, class_ids)
                parity = {
                    "seed": seed, "case_id": case,
                    "lat_before_after_bitwise": bool(np.array_equal(hook.h_before.numpy(), hook.h_after.numpy())),
                    "lat_historical_float16_bitwise": bool(np.array_equal(lat.astype(np.float16), historical_lat.astype(np.float16))),
                    "panl_historical_float16_bitwise": bool(np.array_equal(panl.astype(np.float16), historical_panl.astype(np.float16))),
                    "sac_logits_bitwise": bool(np.array_equal(np.asarray(score["class_logits"]), np.asarray(manifest_row["class_logits"]))),
                    "sac_probabilities_bitwise": bool(np.array_equal(np.asarray(score["class_probabilities"]), np.asarray(manifest_row["class_probabilities"]))),
                    "soft_sa_bitwise": float(score["soft_sa_image_score"]) == float(manifest_row["soft_sa_image_score"]),
                    "hard_sa_equal": int(score["argmax_hard_class"]) == int(manifest_row["argmax_hard_class"]),
                    "position_and_token_identity": True, "off_target_tokens_bitwise": hook.off_target_bitwise,
                    "processor_is_explicit_fast": identity["is_fast"] and identity["image_processor_class"].endswith("Qwen2VLImageProcessorFast"),
                    "rendered_prompt_sha256": __import__("hashlib").sha256(rendered.encode()).hexdigest(),
                }
                parity["passed"] = all(value for key, value in parity.items() if key not in {"seed", "case_id", "rendered_prompt_sha256"})
                if not parity["passed"]:
                    raise RuntimeError(f"Fast clean parity failed: {parity}")
                parity_rows.append(parity)
                atomic_npz(baseline_path, {"lat": lat, "panl": panl})
                endpoints = _endpoint_values(lat, panl, score, anchors, seed_probes[seed])
                result = _base_row(seed, manifest_row, "shared_alpha_zero", 0.0, score, endpoints, identity)
                result.update({
                    "hook_hit_count": 1, "off_target_tokens_bitwise": True,
                    "alpha_zero_shared": True, "parity": parity, "fingerprint": fingerprint,
                    "clean_sa_logits": score["class_logits"], "steered_sa_logits": score["class_logits"],
                    "clean_sa_probabilities": score["class_probabilities"], "steered_sa_probabilities": score["class_probabilities"],
                    "clean_final_soft_sa": endpoints["final_soft_sa"], "steered_final_soft_sa": endpoints["final_soft_sa"],
                    "delta_final_soft_sa": 0.0, "clean_hard_sa": int(score["argmax_hard_class"]),
                    "steered_hard_sa": int(score["argmax_hard_class"]), "hard_sa_changed": False,
                    "clean_class_margin": class_margin(np.asarray(score["class_logits"], np.float64), int(score["argmax_hard_class"])),
                    "class_margin_change": 0.0, "lat_hidden_sha256": array_hash(lat), "panl_hidden_sha256": array_hash(panl),
                })
                append_jsonl(output, result); completed[zero_key] = result
            baseline = completed[zero_key]
            answer = str(manifest_row["phase0_normalized_answer"])
            for direction in DIRECTIONS:
                info = vector_meta[seed][answer, direction]
                vector = np.asarray(vector_files[seed][info["scaled_key"]], np.float32)
                if array_hash(vector) != info["scaled_hash"]:
                    raise RuntimeError("Vector hash mismatch")
                for alpha in NONZERO_ALPHAS:
                    key = trial_key({"seed": seed, "case_id": case, "direction": direction, "alpha": alpha})
                    if key in completed:
                        continue
                    hook = StrictTargetHook(modules, lat_position, torch.from_numpy(vector) * alpha, sequence_length)
                    with hook:
                        forward = run_hooked_forward(inference.model, inputs, modules, {PANL_POSITION: panl_position}, logits_positions=[sac_position])
                    forwards += 1
                    if hook.hook_call_count != 1 or hook.applied_count != 1 or not hook.off_target_bitwise:
                        raise RuntimeError(f"Hook gate failed: {key}")
                    lat = hook.h_after.numpy().astype(np.float32)
                    panl = forward.hidden_by_name[PANL_POSITION][PANL_LAYER].detach().float().cpu().numpy().astype(np.float32)
                    score = _score(forward, sac_position, class_ids)
                    endpoints = _endpoint_values(lat, panl, score, anchors, seed_probes[seed])
                    result = _base_row(seed, manifest_row, direction, alpha, score, endpoints, identity)
                    result.update({
                        "hook_hit_count": 1, "off_target_tokens_bitwise": True,
                        "alpha_zero_shared": False, "vector_sha256": info["scaled_hash"],
                        "vector_fingerprint": info["vector_fingerprint"],
                        "activation_before_sha256": array_hash(hook.h_before.numpy()),
                        "activation_after_sha256": array_hash(hook.h_after.numpy()),
                        "actual_injection_norm": float(np.linalg.norm(hook.h_after.numpy().astype(np.float64) - hook.h_before.numpy().astype(np.float64))),
                        "clean_sa_logits": baseline["sa_logits"], "steered_sa_logits": score["class_logits"],
                        "clean_sa_probabilities": baseline["sa_probabilities"], "steered_sa_probabilities": score["class_probabilities"],
                        "clean_final_soft_sa": baseline["final_soft_sa"], "steered_final_soft_sa": endpoints["final_soft_sa"],
                        "delta_final_soft_sa": endpoints["final_soft_sa"] - baseline["final_soft_sa"],
                        "clean_hard_sa": baseline["hard_sa"], "steered_hard_sa": int(score["argmax_hard_class"]),
                        "hard_sa_changed": int(score["argmax_hard_class"]) != int(baseline["hard_sa"]),
                        "clean_class_margin": baseline["class_margin"], "class_margin_change": class_margin(np.asarray(score["class_logits"], np.float64), int(baseline["hard_sa"])) - float(baseline["class_margin"]),
                        "delta_anchor_panl_sa": endpoints["anchor_panl_sa"] - baseline["anchor_panl_sa"],
                        "delta_anchor_lat_confidence": endpoints["anchor_lat_confidence"] - baseline["anchor_lat_confidence"],
                        "delta_seed_panl_sa": endpoints["seed_panl_sa"] - baseline["seed_panl_sa"],
                        "delta_seed_lat_confidence": endpoints["seed_lat_confidence"] - baseline["seed_lat_confidence"],
                        "lat_hidden_sha256": array_hash(lat), "panl_hidden_sha256": array_hash(panl),
                        "fingerprint": fingerprint,
                    })
                    append_jsonl(output, result); completed[key] = result
    for payload in vector_files.values():
        payload.close()
    atomic_jsonl(root / f"artifacts/diagnostics/parity.shard_{worker}.jsonl", parity_rows)
    return {"worker": worker, "new_gpu_forwards": forwards, "resumed_noop": forwards == 0, "processor_identity": identity}


def run_steering(root: Path, seeds: Sequence[int], num_gpus: int, fingerprint: str) -> dict[str, Any]:
    if num_gpus not in (1, 2):
        raise ValueError("--num-gpus must be 1 or 2")
    if not torch.cuda.is_available() or torch.cuda.device_count() < num_gpus:
        raise RuntimeError(f"Requested {num_gpus} GPUs; visible={torch.cuda.device_count()}")
    started = time.time()
    processes = []
    repo = Path(__file__).resolve().parents[3]
    for worker in range(num_gpus):
        env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(worker)
        command = [sys.executable, "-m", "dp_SA.confidence_steering.robust_check.run", "--worker", str(worker), "--num-gpus", str(num_gpus), "--output-root", str(root), "--seeds", *map(str, seeds), "--fingerprint", fingerprint]
        processes.append(subprocess.Popen(command, cwd=repo, env=env))
    codes = [process.wait() for process in processes]
    if any(codes):
        raise RuntimeError(f"Robust steering worker failed: {codes}")
    manifest = load_jsonl(root / "artifacts/trials/runtime_manifest.jsonl")
    expected = {
        trial_key({"seed": seed, "case_id": row["case_id"], "direction": direction, "alpha": alpha})
        for seed in seeds for row in manifest
        for direction, alpha in [("shared_alpha_zero", 0.0), *[(direction, alpha) for direction in DIRECTIONS for alpha in NONZERO_ALPHAS]]
    }
    rows = []
    for path in sorted((root / "artifacts/trials").glob("trials.shard_*.jsonl")):
        rows.extend(load_jsonl(path, repair_trailing=True))
    merged = canonical_merge(rows, expected)
    atomic_jsonl(root / "artifacts/trials/trials.jsonl", merged)
    worker_reports = [json.loads((root / f"progress/worker_{worker}.json").read_text()) for worker in range(num_gpus)]
    result = {
        "status": "complete", "seeds": list(seeds), "cases": len(manifest),
        "trial_rows": len(merged), "expected_physical_forwards": len(expected),
        "new_gpu_forwards": sum(row["new_gpu_forwards"] for row in worker_reports),
        "resumed_noop": all(row["resumed_noop"] for row in worker_reports),
        "num_gpus": num_gpus, "fingerprint": fingerprint,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(root / "progress/run.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--num-gpus", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--fingerprint", required=True)
    args = parser.parse_args(argv)
    result = run_worker(args.output_root.resolve(), args.worker, args.num_gpus, args.seeds, args.fingerprint)
    atomic_json(args.output_root.resolve() / f"progress/worker_{args.worker}.json", result)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
