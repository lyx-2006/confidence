from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import joblib
import numpy as np
import torch

from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.model_adapter import model_input_device, resolve_language_modules, run_logits_forward
from dp_SA.positions import locate_phase1_positions
from dp_SA.prompts import SA_PREFILL
from dp_SA.soft_score import class_token_ids, soft_sa_from_logits

from .config import (
    ALPHAS, ANSWER_MATCHED_ROOT, HISTORICAL_CLEAN_PATH, LAT_LAYER, MANIFEST_PATH,
    MAX_PIXELS, MIN_PIXELS, MODEL_PATH, NONZERO_ALPHAS, PANL_MEDIATOR_LAYERS,
    PROBE_ROOT, RESULTS_ROOT, SMOKE_ALPHAS, SMOKE_LAYERS, VECTOR_METADATA_PATH,
)
from .hooks import LATPANLMediationHook
from .io_utils import (
    atomic_bf16_npz, atomic_json, atomic_jsonl, bits_hash, canonical_hash,
    load_bf16_npz, load_jsonl, sha256_file,
)


def _messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "image", "image": str(Path(row["image_path"]).resolve())},
                                             {"type": "text", "text": str(row["phase1_prompt"])}]},
            {"role": "assistant", "content": [{"type": "text", "text": SA_PREFILL}]}]


def load_explicit_fast_runtime() -> Any:
    import transformers
    from transformers import (
        AutoTokenizer, Qwen2VLImageProcessorFast, Qwen2VLVideoProcessor,
        Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor,
    )
    image_processor = Qwen2VLImageProcessorFast.from_pretrained(
        MODEL_PATH, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    video_processor = Qwen2VLVideoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    processor = Qwen2_5_VLProcessor(image_processor=image_processor, tokenizer=tokenizer,
                                    video_processor=video_processor, chat_template=tokenizer.chat_template)
    if processor.image_processor.__class__.__name__ != "Qwen2VLImageProcessorFast" or not getattr(processor.image_processor, "is_fast", False):
        raise RuntimeError("Explicit Fast processor identity gate failed")
    if int(processor.image_processor.min_pixels) != MIN_PIXELS or int(processor.image_processor.max_pixels) != MAX_PIXELS:
        raise RuntimeError("Explicit Fast processor pixel configuration changed")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager", local_files_only=True).eval()
    if next(model.parameters()).dtype != torch.bfloat16:
        raise RuntimeError("Model did not load in bfloat16")
    runtime = SimpleNamespace(model=model, processor=processor, dtype=torch.bfloat16,
                              dtype_name="bfloat16", transformers_version=transformers.__version__)
    runtime._get_inputs_device = lambda: next(model.get_input_embeddings().parameters()).device
    return runtime


def _class_margin(logits: Sequence[float], clean_class: int) -> float:
    values = np.asarray(logits, dtype=np.float64); selected = float(values[clean_class])
    return selected - float(np.max(np.delete(values, clean_class)))


def _trial_name(case_id: str, condition: str, alpha: float, layer: int | None) -> str:
    layer_part = "none" if layer is None else f"L{layer}"
    alpha_part = f"m{abs(alpha):g}" if alpha < 0 else f"p{alpha:g}"
    return f"{case_id}__{condition}__{layer_part}__a{alpha_part}.json"


def _source_path(root: Path, case_id: str, condition: str, alpha: float = 0) -> Path:
    suffix = "C0" if condition == "C0" else f"C1__a{alpha:+g}".replace("+", "p").replace("-", "m")
    return root / "artifacts" / "hidden" / f"{case_id}__{suffix}.npz"


def _probe_payload(fingerprint: dict[str, Any]) -> dict[str, Any]:
    row = fingerprint["probe_index_row"]; path = PROBE_ROOT / row["probe_file"]
    if sha256_file(path) != row["probe_sha256"]: raise ValueError("Frozen probe changed after prepare")
    payload = joblib.load(path)
    if payload.get("target") != "final_soft_sa" or payload.get("position") != "P1_CLASS_LIST_END" or int(payload.get("layer")) != 20:
        raise ValueError("Frozen probe payload identity mismatch")
    return payload


def _probe_value(hidden: torch.Tensor, payload: dict[str, Any]) -> tuple[float, float]:
    array = hidden.float().numpy().reshape(1, -1)
    model_value = float(payload["model"].predict(array)[0])
    raw_value = float(array.reshape(-1) @ np.asarray(payload["raw_weight"]) + float(payload["raw_intercept"]))
    if abs(model_value - raw_value) > 1e-6: raise RuntimeError("Frozen probe/raw expression mismatch")
    return model_value, abs(model_value - raw_value)


def _load_vector(case: dict[str, Any], selection: dict[str, dict[str, Any]]) -> tuple[torch.Tensor, dict[str, Any]]:
    row = selection[str(case["case_id"])]
    if int(row["fold"]) != int(case["fold"]) or row["recipient_answer"] != case["test_answer"]:
        raise ValueError("Case-specific vector selection mismatch")
    path = ANSWER_MATCHED_ROOT / row["vector_file"]
    if sha256_file(path) != row["vector_file_sha256"]: raise ValueError("Vector file changed after prepare")
    with np.load(path, allow_pickle=False) as archive: vector = np.asarray(archive[row["scaled_key"]], dtype=np.float32).copy()
    return torch.from_numpy(vector), row


def _score(logits: torch.Tensor, token_ids: Sequence[int]) -> dict[str, Any]:
    scored = soft_sa_from_logits(logits, token_ids)
    if not math.isclose(float(scored["probability_sum"]), 1.0, abs_tol=1e-9): raise RuntimeError("Invalid class probabilities")
    return scored


def _forward(runtime: Any, modules: Any, inputs: Any, positions: dict[str, int], token_ids: Sequence[int],
             *, vector: torch.Tensor | None, patch_layer: int | None, patch_source: torch.Tensor | None,
             capture_layers: Sequence[int]) -> tuple[dict[str, Any], LATPANLMediationHook]:
    hook = LATPANLMediationHook(modules, prefill_sequence_length=int(inputs.input_ids.shape[1]),
                                lat_position=positions["P1_LAT"], panl_position=positions["P1_PANL"],
                                cle_position=positions["P1_CLASS_LIST_END"], steering_vector=vector,
                                patch_layer=patch_layer, patch_source=patch_source,
                                capture_panl_layers=capture_layers)
    with hook: logits = run_logits_forward(runtime.model, inputs, [positions["P1_SAC"]], modules)[positions["P1_SAC"]]
    hook.validate(); return _score(logits, token_ids), hook


def _base_trial(case: dict[str, Any], condition: str, alpha: float, layer: int | None,
                score: dict[str, Any], hook: LATPANLMediationHook, probe: dict[str, Any],
                fingerprint: dict[str, Any], vector_row: dict[str, Any] | None,
                source_path: Path | None = None, source_key: str | None = None) -> dict[str, Any]:
    eligible = bool(case["cle_probe_eligible"])
    cle_value = cle_error = None
    if eligible:
        assert hook.cle_hidden is not None
        cle_value, cle_error = _probe_value(hook.cle_hidden, probe)
    hidden_hash = None
    if source_path is not None and source_key is not None:
        _, meta = load_bf16_npz(source_path, source_key); hidden_hash = meta["bits_sha256"]
    logits = score["class_logits"]
    return {
        "status": "completed", "case_id": str(case["case_id"]), "family_id": str(case["family_id"]),
        "item_id": str(case["item_id"]), "image_sha256": str(case["image_sha256"]),
        "answer": str(case["test_answer"]), "test_side": str(case["test_side"]), "fold": int(case["fold"]),
        "condition": condition, "alpha": float(alpha), "panl_mediator_layer": layer,
        "final_soft_sa": float(score["soft_sa_image_score"]), "hard_sa_class": int(score["argmax_hard_class"]),
        "class_logits": logits, "class_probabilities": score["class_probabilities"],
        "cle_probe_sa": cle_value, "cle_probe_expression_error": cle_error,
        "cle_probe_eligible": eligible, "cle_probe_exclusion_reasons": case["cle_probe_exclusion_reasons"],
        "hook": hook.diagnostics(), "vector": vector_row,
        "source_hidden_file": None if source_path is None else str(source_path.relative_to(Path(fingerprint["output_root"]))),
        "source_hidden_key": source_key, "source_hidden_bits_sha256": hidden_hash,
        "semantic_fingerprint": fingerprint["semantic_fingerprint"],
    }


def _case_context(runtime: Any, case: dict[str, Any], historical: dict[str, dict[str, Any]]) -> tuple[Any, dict[str, int], str]:
    messages = _messages(case); rendered = render_continued_assistant(runtime.processor, messages, SA_PREFILL)
    device = model_input_device(runtime); inputs = prepare_multimodal_inputs(runtime.processor, messages, rendered, device=device)
    tokenizer = runtime.processor.tokenizer
    located = locate_phase1_positions(tokenizer, rendered, inputs, str(case["phase0_raw_answer"]))
    names = ("P1_LAT", "P1_PANL", "P1_CLASS_LIST_END", "P1_SAC")
    positions = {name: int(located[name]["processed_index"]) for name in names}
    source = historical[str(case["case_id"])]
    if canonical_hash(rendered) != source["rendered_prompt_hash"]: raise RuntimeError("Rendered prompt parity failed")
    if any(positions[name] != int(source["positions"][name]["processed_index"]) for name in names): raise RuntimeError("Position parity failed")
    return inputs, positions, rendered


def _run_shard(root: Path, *, smoke: bool, resume: bool, stage: str, rank: int, world_size: int) -> dict[str, Any]:
    fingerprint = json.loads((root / "fingerprint.json").read_text()); fingerprint["output_root"] = str(root.resolve())
    cases = sorted(load_jsonl(root / "artifacts" / "manifests" / "test_manifest.jsonl"), key=lambda row: str(row["case_id"]))
    cases = [row for index, row in enumerate(cases) if index % world_size == rank]
    selections = {row["case_id"]: row for row in load_jsonl(root / "artifacts" / "manifests" / "vector_selection.jsonl")}
    historical = {str(row["case_id"]): row for row in load_jsonl(HISTORICAL_CLEAN_PATH)}
    probe = _probe_payload(fingerprint); runtime = load_explicit_fast_runtime(); modules = resolve_language_modules(runtime.model)
    token_ids = class_token_ids(runtime.processor.tokenizer)
    runtime_identity = {"processor_class": runtime.processor.__class__.__module__ + "." + runtime.processor.__class__.__name__,
                        "image_processor_class": runtime.processor.image_processor.__class__.__module__ + "." + runtime.processor.image_processor.__class__.__name__,
                        "is_fast": bool(runtime.processor.image_processor.is_fast), "transformers_version": runtime.transformers_version,
                        "model_dtype": runtime.dtype_name}
    atomic_json(root / "progress" / f"runtime_rank{rank}.json", runtime_identity)
    if rank == 0:
        recorded = json.loads((root / "fingerprint.json").read_text())
        recorded["runtime_processor_identity"] = runtime_identity
        atomic_json(root / "fingerprint.json", recorded)
    layers = SMOKE_LAYERS if smoke else PANL_MEDIATOR_LAYERS
    alphas = tuple(value for value in (SMOKE_ALPHAS if smoke else ALPHAS) if value != 0)
    new_forwards = 0; started = time.time()
    for case in cases:
        inputs, positions, _ = _case_context(runtime, case, historical); case_id = str(case["case_id"])
        vector, vector_row = _load_vector(case, selections)
        if stage == "c0":
            destination = root / "artifacts" / "trials" / _trial_name(case_id, "C0", 0, None)
            source_path = _source_path(root, case_id, "C0")
            if destination.exists() and source_path.exists():
                if not resume: raise FileExistsError(destination)
                continue
            score, hook = _forward(runtime, modules, inputs, positions, token_ids, vector=None, patch_layer=None,
                                   patch_source=None, capture_layers=layers)
            source = historical[case_id]
            logit_error = float(np.max(np.abs(np.asarray(score["class_logits"]) - np.asarray(source["class_logits"]))))
            prob_error = float(np.max(np.abs(np.asarray(score["class_probabilities"]) - np.asarray(source["class_probabilities"]))))
            soft_error = abs(float(score["soft_sa_image_score"]) - float(source["soft_sa_image_score"]))
            if max(logit_error, prob_error, soft_error) > 1e-6 or int(score["argmax_hard_class"]) != int(source["argmax_hard_class"]):
                raise RuntimeError(f"C0 score parity failed for {case_id}")
            arrays = {f"PANL_L{layer}": hook.captured_panl[layer] for layer in layers}
            atomic_bf16_npz(source_path, arrays)
            historical_hidden_equal = True; checked = []
            old_path = ANSWER_MATCHED_ROOT / source["hidden_file"]
            if old_path.is_file():
                with np.load(old_path, allow_pickle=False) as archive:
                    for layer in layers:
                        key = f"P1_PANL__L{layer}"
                        if key in archive:
                            checked.append(layer)
                            historical_hidden_equal &= np.array_equal(hook.captured_panl[layer].float().numpy().astype(np.float16), archive[key])
            if checked and not historical_hidden_equal: raise RuntimeError(f"Historical hidden parity failed for {case_id}")
            trial = _base_trial(case, "C0", 0, None, score, hook, probe, fingerprint, vector_row)
            trial["parity"] = {"passed": True, "logit_max_abs_error": logit_error, "probability_max_abs_error": prob_error,
                               "soft_sa_abs_error": soft_error, "historical_hidden_layers_checked": checked,
                               "historical_hidden_equal_after_fp16_cast": historical_hidden_equal}
            trial["captured_hidden_file"] = str(source_path.relative_to(root)); trial["captured_hidden_sha256"] = sha256_file(source_path)
            atomic_json(destination, trial); new_forwards += 1
            continue
        if stage != "interventions": raise ValueError(stage)
        for alpha in alphas:
            c1_dest = root / "artifacts" / "trials" / _trial_name(case_id, "C1", alpha, None)
            c1_source = _source_path(root, case_id, "C1", alpha)
            if not (c1_dest.exists() and c1_source.exists()):
                score, hook = _forward(runtime, modules, inputs, positions, token_ids, vector=vector * alpha,
                                       patch_layer=None, patch_source=None, capture_layers=layers)
                atomic_bf16_npz(c1_source, {f"PANL_L{layer}": hook.captured_panl[layer] for layer in layers})
                trial = _base_trial(case, "C1", alpha, None, score, hook, probe, fingerprint, vector_row)
                trial["captured_hidden_file"] = str(c1_source.relative_to(root)); trial["captured_hidden_sha256"] = sha256_file(c1_source)
                atomic_json(c1_dest, trial); new_forwards += 1
            elif not resume: raise FileExistsError(c1_dest)
            for layer in layers:
                for condition, source_file, use_vector in (("C2", _source_path(root, case_id, "C0"), True), ("C3", c1_source, False)):
                    destination = root / "artifacts" / "trials" / _trial_name(case_id, condition, alpha, layer)
                    if destination.exists():
                        if not resume: raise FileExistsError(destination)
                        continue
                    source_key = f"PANL_L{layer}"; patch_source, _ = load_bf16_npz(source_file, source_key)
                    score, hook = _forward(runtime, modules, inputs, positions, token_ids,
                                           vector=vector * alpha if use_vector else None,
                                           patch_layer=layer, patch_source=patch_source, capture_layers=(layer,))
                    trial = _base_trial(case, condition, alpha, layer, score, hook, probe, fingerprint,
                                        vector_row, source_file, source_key)
                    before = hook.patch_before.float().numpy(); after = hook.patch_after.float().numpy()
                    source_float = patch_source.float().numpy(); delta = source_float - before
                    before_norm = float(np.linalg.norm(before)); delta_norm = float(np.linalg.norm(delta))
                    cosine = float(np.dot(before, source_float) / max(np.linalg.norm(before) * np.linalg.norm(source_float), 1e-30))
                    trial["panl_manipulation"] = {"delta_norm": delta_norm, "relative_norm": delta_norm / max(before_norm, 1e-30),
                                                  "cosine_before_source": cosine, "replacement_bitwise_equal": True}
                    atomic_json(destination, trial); new_forwards += 1
        atomic_json(root / "progress" / f"run_{stage}_rank{rank}.json",
                    {"status": "running", "new_gpu_forwards": new_forwards, "last_case_id": case_id,
                     "elapsed_seconds": time.time() - started})
    result = {"status": "complete", "stage": stage, "rank": rank, "world_size": world_size,
              "new_gpu_forwards": new_forwards, "elapsed_seconds": time.time() - started}
    atomic_json(root / "progress" / f"run_{stage}_rank{rank}.json", result); return result


def _merge_and_validate(root: Path, *, smoke: bool, require_complete: bool = True) -> list[dict[str, Any]]:
    rows = [json.loads(path.read_text()) for path in sorted((root / "artifacts" / "trials").glob("*.json"))]
    if any(row.get("semantic_fingerprint") != json.loads((root / "fingerprint.json").read_text())["semantic_fingerprint"] for row in rows):
        raise RuntimeError("Trial fingerprint mismatch")
    cases = load_jsonl(root / "artifacts" / "manifests" / "test_manifest.jsonl")
    layers = SMOKE_LAYERS if smoke else PANL_MEDIATOR_LAYERS
    nonzero = 2 if smoke else 4; expected = len(cases) * (1 + nonzero + 2 * len(layers) * nonzero)
    keys = {(row["case_id"], row["condition"], row["panl_mediator_layer"], row["alpha"]) for row in rows}
    if len(keys) != len(rows): raise RuntimeError("Duplicate canonical trial keys")
    if require_complete and len(rows) != expected: raise RuntimeError(f"Trial grid incomplete: {len(rows)}/{expected}")
    atomic_jsonl(root / "artifacts" / "trials.jsonl", sorted(rows, key=lambda row: (row["case_id"], row["condition"], row["panl_mediator_layer"] or -1, row["alpha"])))
    return rows


def _validate_c0_gate(root: Path, *, smoke: bool) -> None:
    rows = _merge_and_validate(root, smoke=smoke, require_complete=False)
    cases = load_jsonl(root / "artifacts" / "manifests" / "test_manifest.jsonl")
    c0 = [row for row in rows if row["condition"] == "C0"]
    if len(c0) != len(cases) or any(not row.get("parity", {}).get("passed") for row in c0):
        atomic_json(root / "progress" / "c0_gate.json", {"status": "failed", "completed": len(c0), "expected": len(cases)})
        raise RuntimeError("Global C0 parity gate failed")
    atomic_json(root / "progress" / "c0_gate.json", {"status": "passed", "completed": len(c0), "expected": len(cases)})


def _spawn_stage(root: Path, *, smoke: bool, resume: bool, stage: str, num_gpus: int) -> list[dict[str, Any]]:
    processes = []
    for rank in range(num_gpus):
        command = [sys.executable, "-m", "dp_SA.SA_trajectory.LAT2PANL.run", "--output-root", str(root),
                   "--stage", stage, "--worker-rank", str(rank), "--world-size", str(num_gpus)]
        if smoke: command.append("--smoke")
        if resume: command.append("--resume")
        environment = dict(os.environ); environment["CUDA_VISIBLE_DEVICES"] = str(rank)
        log_path = root / "progress" / f"{stage}_gpu{rank}.log"; log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("w", encoding="utf-8")
        processes.append((rank, subprocess.Popen(command, cwd=Path(__file__).resolve().parents[3], env=environment,
                                                  stdout=handle, stderr=subprocess.STDOUT, text=True), handle, log_path))
    results = []
    for rank, process, handle, log_path in processes:
        code = process.wait(); handle.close()
        if code: raise RuntimeError(f"GPU worker {rank} failed; see {log_path}")
        results.append(json.loads((root / "progress" / f"run_{stage}_rank{rank}.json").read_text()))
    return results


def run(*, output_root: Path = RESULTS_ROOT, smoke: bool = False, resume: bool = False,
        num_gpus: int = 1) -> dict[str, Any]:
    root = Path(output_root)
    if num_gpus not in (1, 2): raise ValueError("--num-gpus must be 1 or 2")
    if not (root / "fingerprint.json").is_file(): raise RuntimeError("Run prepare first")
    if num_gpus == 1:
        c0 = [_run_shard(root, smoke=smoke, resume=resume, stage="c0", rank=0, world_size=1)]
    else: c0 = _spawn_stage(root, smoke=smoke, resume=resume, stage="c0", num_gpus=num_gpus)
    _validate_c0_gate(root, smoke=smoke)
    if num_gpus == 1:
        interventions = [_run_shard(root, smoke=smoke, resume=resume, stage="interventions", rank=0, world_size=1)]
    else: interventions = _spawn_stage(root, smoke=smoke, resume=resume, stage="interventions", num_gpus=num_gpus)
    rows = _merge_and_validate(root, smoke=smoke)
    result = {"status": "complete", "trial_count": len(rows),
              "new_gpu_forwards": sum(row["new_gpu_forwards"] for row in c0 + interventions),
              "c0_gate": "passed", "num_gpus": num_gpus,
              "resumed_noop": sum(row["new_gpu_forwards"] for row in c0 + interventions) == 0}
    atomic_json(root / "progress" / "run.json", result); return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--smoke", action="store_true"); parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-gpus", type=int, choices=(1, 2), default=1)
    parser.add_argument("--stage", choices=("c0", "interventions")); parser.add_argument("--worker-rank", type=int)
    parser.add_argument("--world-size", type=int)
    args = parser.parse_args(argv)
    if args.worker_rank is not None:
        result = _run_shard(args.output_root, smoke=args.smoke, resume=args.resume, stage=args.stage,
                            rank=args.worker_rank, world_size=args.world_size)
    else: result = run(output_root=args.output_root, smoke=args.smoke, resume=args.resume, num_gpus=args.num_gpus)
    print(json.dumps(result, ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
