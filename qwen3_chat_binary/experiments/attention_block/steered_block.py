from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from dp_SA.attention_block.masking import AttentionBlockContext, AttentionEdges
from dp_SA.io_utils import atomic_json, atomic_jsonl, canonical_hash, load_jsonl, sha256_file
from layer_metacognition.model_adapter import AdditiveActivationHook, model_input_device, resolve_language_modules, run_logits_forward
from qwen3_chat_binary.config import CAPTURE_ROOT, MODEL_PATH, OUTPUT_ROOT, VECTOR_NORM_FRACTION
from qwen3_chat_binary.contracts import ensure_fingerprinted_config
from qwen3_chat_binary.conversation import prepare_multimodal_inputs, render_stage2, stage2_messages
from qwen3_chat_binary.layout import capture_results_path
from qwen3_chat_binary.positions import locate_positions
from qwen3_chat_binary.prompts import LABELS
from qwen3_chat_binary.runtime import load_qwen3_inference
from qwen3_chat_binary.scoring import attribution_score, label_token_ids
from qwen3_chat_binary.steering import _load_completed, build_vectors, save_torch_atomic

from .config import CLEAN_LOGIT_TOLERANCE, CLEAN_SCORE_TOLERANCE, ROW_SUM_TOLERANCE, SEED, VARIANT
from .core import class_margin, deterministic_argmax


STEERING_ROOT = OUTPUT_ROOT / "Steering"
MODE_ROOT = OUTPUT_ROOT / "AttentionBlockFiveway" / "steered_block"
STEERING_LAYER = 16
ALPHAS = (-5.0, 5.0)
WINDOWS = ((17, 24), (22, 30), (24, 32), (26, 34))
PARITY_TOLERANCE = 1e-5


def default_output(smoke: bool = False) -> Path:
    return MODE_ROOT / ("smoke" if smoke else "formal")


def _implementation_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = ("steered_block.py", "steered_block_analysis.py")
    return {name: sha256_file(root / name) for name in names}


def _manifest_paths(steering_root: Path) -> tuple[Path, Path, Path, Path]:
    return (
        steering_root / "tables/construction_manifest.jsonl",
        steering_root / "tables/test_manifest.jsonl",
        steering_root / "native_boundary/tables/vectors.pt",
        steering_root / "progress/config.json",
    )


def prepare(*, output_root: Path, capture_root: Path = CAPTURE_ROOT,
            steering_root: Path = STEERING_ROOT, model_path: Path = MODEL_PATH,
            smoke: bool = False, resume: bool = False,
            validate_tokens: bool = True) -> dict[str, Any]:
    root, capture_root, steering_root, model_path = (
        Path(value).resolve() for value in (output_root, capture_root, steering_root, model_path)
    )
    construction_path, test_path, vector_path, steering_config_path = _manifest_paths(steering_root)
    steering_config = json.loads(steering_config_path.read_text(encoding="utf-8"))
    if Path(steering_config["model"]).resolve() != model_path:
        raise ValueError("Steering and enhanced-block model paths differ")
    if Path(steering_config["capture_root"]).resolve() != capture_root:
        raise ValueError("Steering and enhanced-block capture roots differ")
    if float(steering_config["normalization_fraction"]) != VECTOR_NORM_FRACTION:
        raise ValueError("Steering vector normalization is not 3%")
    construction, formal = load_jsonl(construction_path), load_jsonl(test_path)
    if Counter(r["construction_side"] for r in construction) != Counter({"high_image": 25, "high_text": 25}):
        raise ValueError("Expected 25+25 steering construction cases")
    if Counter(r["test_side"] for r in formal) != Counter({"text_side": 50, "image_side": 50}):
        raise ValueError("Expected 50+50 intermediate steering test cases")
    construction_ids, test_ids = ({str(r["case_id"]) for r in rows} for rows in (construction, formal))
    if construction_ids & test_ids:
        raise ValueError("Steering construction/test leakage")
    selected = ([next(r for r in formal if r["test_side"] == side)
                 for side in ("text_side", "image_side")] if smoke else formal)
    stored = torch.load(vector_path, map_location="cpu", weights_only=False)["PANL__L16"]["scaled_vector"].float()
    capture_rows = _load_completed(capture_results_path(capture_root, VARIANT))
    _, vector_meta, artifacts = build_vectors(
        capture_root, capture_rows, construction, variant=VARIANT,
        positions=("PANL",), layers=(STEERING_LAYER,),
    )
    rebuilt = artifacts["PANL__L16"]["scaled_vector"].float()
    vector_error = float((stored - rebuilt).abs().max().item())
    if vector_error != 0.0:
        raise ValueError(f"Stored PANL L16 direction differs from rebuilt direction: {vector_error}")
    if validate_tokens:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        label_token_ids(processor.tokenizer)
        for row in selected:
            messages = stage2_messages(row["phase0_prompt"], row["image_path"], row["phase0_raw_output"], VARIANT)
            rendered = render_stage2(processor, messages)
            inputs = prepare_multimodal_inputs(processor, messages, rendered)
            current = locate_positions(processor.tokenizer, rendered, inputs, row["phase0_raw_output"], VARIANT)
            for name in ("LAT", "PANL", "PANL+1", "CLE", "SAC"):
                old, new = row["positions"][name], current["positions"][name]
                if (int(old["processed_index"]), int(old["token_id"])) != (int(new["processed_index"]), int(new["token_id"])):
                    raise RuntimeError(f"Position drift for {row['case_id']} at {name}")
        processor_audit = {"checked_cases": len(selected), "labels_are_single_tokens": True}
    else:
        processor_audit = {"skipped": True}
    payload = {
        "format_version": 1, "experiment": "qwen3_chat_fiveway_panl_steered_sac_block",
        "mode": "steered_block", "variant": VARIANT, "model": str(model_path),
        "capture_root": str(capture_root), "steering_root": str(steering_root),
        "steering_config_fingerprint": steering_config["fingerprint"],
        "steering_layer": STEERING_LAYER, "steering_position": "PANL",
        "alphas": list(ALPHAS), "windows": [list(x) for x in WINDOWS],
        "window_semantics": "inclusive_zero_based", "query": "SAC", "source": "PANL",
        "conditions": ["C0", "S", "SB"], "smoke": bool(smoke), "seed": SEED,
        "normalization_fraction": VECTOR_NORM_FRACTION,
        "construction_fingerprint": canonical_hash(construction),
        "test_fingerprint": canonical_hash(selected),
        "vector_source_sha256": sha256_file(vector_path),
        "implementation_hashes": _implementation_hashes(),
    }
    config = ensure_fingerprinted_config(root / "run_config.json", payload, resume=resume,
                                         label="Steered attention block")
    atomic_jsonl(root / "artifacts/manifests/construction.jsonl", construction)
    atomic_jsonl(root / "artifacts/manifests/test.jsonl", selected)
    save_torch_atomic(root / "artifacts/panl_l16_vector.pt", {"scaled_vector": stored})
    summary = {
        "case_count": len(selected), "formal_case_count": len(formal),
        "construction_counts": dict(Counter(r["construction_side"] for r in construction)),
        "test_counts": dict(Counter(r["test_side"] for r in selected)),
        "construction_test_overlap": 0, "vector_max_abs_error": vector_error,
        "vector_metadata": vector_meta, "processor_audit": processor_audit,
    }
    atomic_json(root / "artifacts/manifests/selection_summary.json", summary)
    return {"status": "complete", "fingerprint": config["fingerprint"], **summary}


def _context(runtime: Any, row: dict[str, Any]):
    messages = stage2_messages(row["phase0_prompt"], row["image_path"], row["phase0_raw_output"], VARIANT)
    rendered = render_stage2(runtime.processor, messages)
    inputs = prepare_multimodal_inputs(runtime.processor, messages, rendered, device=model_input_device(runtime))
    located = locate_positions(runtime.processor.tokenizer, rendered, inputs, row["phase0_raw_output"], VARIANT)
    for name in ("LAT", "PANL", "PANL+1", "CLE", "SAC"):
        old, new = row["positions"][name], located["positions"][name]
        if (int(old["processed_index"]), int(old["token_id"])) != (int(new["processed_index"]), int(new["token_id"])):
            raise RuntimeError(f"Position drift for {row['case_id']} at {name}")
    return inputs, located, {k: int(v) for k, v in located["indices"].items()}


def _score(runtime: Any, modules: Any, inputs: Any, sac: int, ids: tuple[int, ...]):
    logits = run_logits_forward(runtime.model, inputs, [sac], modules)[sac]
    scored = attribution_score(logits, ids)
    values = [float(scored["label_logits"][label]) for label in LABELS]
    predicted, tie = deterministic_argmax(values)
    return values, scored, predicted, tie


def _trial_path(root: Path, case: str, condition: str, alpha: float | None = None,
                window: tuple[int, int] | None = None) -> Path:
    parts = [case, condition]
    if alpha is not None: parts.append(f"a{'m' if alpha < 0 else 'p'}{abs(alpha):g}")
    if window is not None: parts.append(f"L{window[0]}-{window[1]}")
    return root / "artifacts/trials" / ("__".join(parts) + ".json")


def _complete_trials(root: Path, config: dict[str, Any], case_count: int) -> list[dict[str, Any]] | None:
    files = sorted((root / "artifacts/trials").glob("*.json"))
    expected = case_count * (1 + len(ALPHAS) + len(ALPHAS) * len(WINDOWS))
    if len(files) != expected: return None
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    keys = {(r["case_id"], r["condition"], r.get("alpha"), r.get("window_start"), r.get("window_end")) for r in rows}
    if len(keys) != expected or any(r.get("fingerprint") != config["fingerprint"] for r in rows):
        raise RuntimeError("Completed steered-block trials are duplicated or have a mixed fingerprint")
    return rows


def _same_logits(left: Sequence[float], right: Sequence[float]) -> float:
    return float(np.max(np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))))


def comparison_metrics(reference_logits: Sequence[float], target_logits: Sequence[float]) -> dict[str, Any]:
    reference_class, reference_tie = deterministic_argmax(reference_logits)
    target_class, target_tie = deterministic_argmax(target_logits)
    return {
        "reference_class": reference_class, "target_class": target_class,
        "reference_argmax_tie": reference_tie, "target_argmax_tie": target_tie,
        "token_changed": target_class != reference_class,
        "token_change_rate": float(target_class != reference_class),
        "logit_change_diff": class_margin(reference_logits, reference_class)-class_margin(target_logits, reference_class),
    }


def run(*, output_root: Path, model_path: Path = MODEL_PATH, resume: bool = False) -> dict[str, Any]:
    root = Path(output_root).resolve(); config = json.loads((root / "run_config.json").read_text())
    if config.get("mode") != "steered_block": raise ValueError("Output root is not steered-block mode")
    if Path(config["model"]).resolve() != Path(model_path).resolve(): raise ValueError("Runtime model differs from config")
    cases = load_jsonl(root / "artifacts/manifests/test.jsonl")
    completed = _complete_trials(root, config, len(cases))
    if resume and completed is not None:
        atomic_jsonl(root / "artifacts/trials.jsonl", completed)
        return {"status": "complete", "case_count": len(cases), "trial_count": len(completed),
                "new_gpu_forwards": 0, "resumed_noop": True}
    vector = torch.load(root / "artifacts/panl_l16_vector.pt", map_location="cpu", weights_only=False)["scaled_vector"].float()
    runtime = load_qwen3_inference(model_path, attn_implementation="eager")
    modules = resolve_language_modules(runtime.model)
    if (modules.num_hidden_layers, modules.hidden_size) != (36, 4096): raise RuntimeError("Unexpected Qwen3 architecture")
    ids = label_token_ids(runtime.processor.tokenizer); new_forwards = 0; started = time.time(); smoke_audits = []
    try:
        for ordinal, row in enumerate(cases, 1):
            inputs, located, positions = _context(runtime, row); case = row["case_id"]
            clean_path = _trial_path(root, case, "C0")
            if resume and clean_path.is_file(): clean = json.loads(clean_path.read_text())
            else:
                logits, scored, predicted, tie = _score(runtime, modules, inputs, positions["SAC"], ids)
                saved = [float(row["label_logits"][label]) for label in LABELS]
                clean = {"status": "completed", "case_id": case, "test_side": row["test_side"], "condition": "C0",
                         "alpha": None, "window_start": None, "window_end": None, "label_logits": logits,
                         "image_attribution_score": float(scored["image_attribution_score"]), "hard_class": predicted,
                         "argmax_tie": tie, "clean_margin": class_margin(logits, predicted), "positions": located,
                         "capture_audit": {"max_logit_error": _same_logits(logits, saved),
                                           "score_error": abs(float(scored["image_attribution_score"])-float(row["image_attribution_score"]))},
                         "fingerprint": config["fingerprint"]}
                atomic_json(clean_path, clean); new_forwards += 1
            if clean["fingerprint"] != config["fingerprint"]: raise RuntimeError("C0 fingerprint mismatch")
            if config["smoke"]:
                repeat, _, _, _ = _score(runtime, modules, inputs, positions["SAC"], ids); new_forwards += 1
                zero_hook = AdditiveActivationHook(modules, layer_index=STEERING_LAYER, target_position=positions["PANL"],
                    steering_vector=vector*0, prefill_sequence_length=int(inputs.input_ids.shape[1]))
                with zero_hook: zero, _, _, _ = _score(runtime, modules, inputs, positions["SAC"], ids)
                zero_diag = zero_hook.diagnostics(); new_forwards += 1
                empty = AttentionBlockContext(modules.language_layers, layer_indices=range(17, 25),
                    edges=AttentionEdges(tuple()), sequence_length=int(inputs.input_ids.shape[1]))
                with empty: empty_logits, _, _, _ = _score(runtime, modules, inputs, positions["SAC"], ids)
                empty_diag = empty.diagnostics(); new_forwards += 1
                errors = {"repeat_clean": _same_logits(clean["label_logits"], repeat),
                          "alpha_zero": _same_logits(clean["label_logits"], zero),
                          "empty_block": _same_logits(clean["label_logits"], empty_logits)}
                if max(errors.values()) > PARITY_TOLERANCE: raise RuntimeError(f"Smoke parity failed: {case}: {errors}")
                smoke_audits.append({"case_id": case, "test_side": row["test_side"], "errors": errors,
                                     "alpha_zero_hook": zero_diag, "empty_block": empty_diag, "passed": True})
            for alpha in ALPHAS:
                s_path = _trial_path(root, case, "S", alpha)
                if resume and s_path.is_file(): steered = json.loads(s_path.read_text())
                else:
                    hook = AdditiveActivationHook(modules, layer_index=STEERING_LAYER, target_position=positions["PANL"],
                        steering_vector=vector*alpha, prefill_sequence_length=int(inputs.input_ids.shape[1]))
                    with hook: logits, scored, predicted, tie = _score(runtime, modules, inputs, positions["SAC"], ids)
                    diagnostics = hook.diagnostics(); clean_class = int(clean["hard_class"])
                    steered = {"status": "completed", "case_id": case, "test_side": row["test_side"], "condition": "S",
                        "alpha": alpha, "window_start": None, "window_end": None, "label_logits": logits,
                        "image_attribution_score": float(scored["image_attribution_score"]), "hard_class": predicted,
                        "argmax_tie": tie, "vs_clean_token_changed": predicted != clean_class,
                        "vs_clean_token_change_rate": float(predicted != clean_class),
                        "vs_clean_logit_change_diff": float(clean["clean_margin"])-class_margin(logits, clean_class),
                        "steering_diagnostics": diagnostics, "fingerprint": config["fingerprint"]}
                    atomic_json(s_path, steered); new_forwards += 1
                if steered["fingerprint"] != config["fingerprint"]: raise RuntimeError("S fingerprint mismatch")
                for window in WINDOWS:
                    destination = _trial_path(root, case, "SB", alpha, window)
                    if resume and destination.is_file(): continue
                    steering_hook = AdditiveActivationHook(modules, layer_index=STEERING_LAYER,
                        target_position=positions["PANL"], steering_vector=vector*alpha,
                        prefill_sequence_length=int(inputs.input_ids.shape[1]))
                    block = AttentionBlockContext(modules.language_layers,
                        layer_indices=range(window[0], window[1]+1),
                        edges=AttentionEdges(((positions["SAC"], positions["PANL"]),)),
                        sequence_length=int(inputs.input_ids.shape[1]), row_sum_tolerance=ROW_SUM_TOLERANCE)
                    with steering_hook, block:
                        logits, scored, predicted, tie = _score(runtime, modules, inputs, positions["SAC"], ids)
                    steering_diag, block_diag = steering_hook.diagnostics(), block.diagnostics()
                    clean_class, s_class = int(clean["hard_class"]), int(steered["hard_class"])
                    result = {"status": "completed", "case_id": case, "test_side": row["test_side"], "condition": "SB",
                        "alpha": alpha, "window_start": window[0], "window_end": window[1], "label_logits": logits,
                        "image_attribution_score": float(scored["image_attribution_score"]), "hard_class": predicted,
                        "argmax_tie": tie, "steered_baseline_trial": str(s_path.relative_to(root)),
                        "vs_steered_token_changed": predicted != s_class,
                        "vs_steered_token_change_rate": float(predicted != s_class),
                        "vs_steered_logit_change_diff": class_margin(steered["label_logits"], s_class)-class_margin(logits, s_class),
                        "vs_clean_token_changed": predicted != clean_class,
                        "vs_clean_token_change_rate": float(predicted != clean_class),
                        "vs_clean_logit_change_diff": float(clean["clean_margin"])-class_margin(logits, clean_class),
                        "steering_flipped": s_class != clean_class,
                        "restored_clean_label": s_class != clean_class and predicted == clean_class,
                        "steering_diagnostics": steering_diag, "attention_diagnostics": block_diag,
                        "fingerprint": config["fingerprint"]}
                    if not all(math.isfinite(float(result[k])) for k in ("vs_steered_logit_change_diff", "vs_clean_logit_change_diff")):
                        raise RuntimeError("Non-finite steered-block metric")
                    atomic_json(destination, result); new_forwards += 1
            atomic_json(root / "progress/run.json", {"status": "running", "completed_cases": ordinal,
                "total_cases": len(cases), "new_gpu_forwards": new_forwards})
        rows = _complete_trials(root, config, len(cases))
        if rows is None: raise RuntimeError("Steered-block trial grid incomplete")
        atomic_jsonl(root / "artifacts/trials.jsonl", rows)
        if config["smoke"]: atomic_json(root / "progress/smoke_gates.json", {"status": "passed", "audits": smoke_audits})
        result = {"status": "complete", "case_count": len(cases), "trial_count": len(rows),
                  "new_gpu_forwards": new_forwards, "resumed_noop": False,
                  "elapsed_seconds": time.time()-started}
        atomic_json(root / "progress/run.json", result); return result
    finally:
        del runtime
        if torch.cuda.is_available(): torch.cuda.empty_cache()
