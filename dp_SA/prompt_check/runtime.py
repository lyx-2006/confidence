from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import transformers

from confidence_test.runtime_imports import load_runtime
from dp_SA.confidence_steering.processor import load_fast_processor, processor_identity
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.model_adapter import AdditiveActivationHook, model_input_device, resolve_language_modules, run_hooked_forward, run_logits_forward

from .config import INFERENCE_PATH, MODEL_PATH, POSITIONS, T3_LABELS
from .io_utils import canonical_hash, sha256_file
from .positions import locate_template_positions
from .scoring import conditional_sequence_log_likelihood, numeric_score, parse_t3_greedy, t3_score
from .templates import TemplateSpec


def processor_audit(processor: Any) -> dict[str, Any]:
    identity = processor_identity(processor); tokenizer = getattr(processor, "tokenizer", processor)
    if not identity["is_fast"] or not identity["image_processor_class"].endswith("Qwen2VLImageProcessorFast"): raise RuntimeError(f"Explicit Fast processor failed: {identity}")
    if identity["min_pixels"] != 200704 or identity["max_pixels"] != 1003520: raise RuntimeError(f"Pixel policy changed: {identity}")
    chat = getattr(tokenizer, "chat_template", None) or getattr(processor, "chat_template", None) or ""
    return {**identity, "transformers_version": transformers.__version__, "tokenizer_json_sha256": sha256_file(MODEL_PATH / "tokenizer.json"), "chat_template_sha256": hashlib.sha256(str(chat).encode()).hexdigest(), "model_config_sha256": sha256_file(MODEL_PATH / "config.json")}


def load_inference() -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    runtime = load_runtime(INFERENCE_PATH); inference = runtime.QwenVLInference(str(MODEL_PATH))
    # Replace the complete runtime processor with the repository's verified full loader.
    inference.processor = load_fast_processor()
    modules = resolve_language_modules(inference.model); device = model_input_device(inference)
    return inference, modules, getattr(inference.processor, "tokenizer", inference.processor), device, processor_audit(inference.processor)


def messages(record: dict[str, Any], spec: TemplateSpec) -> tuple[list[dict[str, Any]], str]:
    answer = str(record.get("phase0_raw_answer") or record.get("phase1_inserted_raw_answer"))
    prompt = spec.render(question=str(record["question"]), text_clue=str(record["text_clue"]), answer=answer)
    wire = [{"role": "user", "content": [{"type": "image", "image": str(Path(record["image_path"]).resolve())}, {"type": "text", "text": prompt}]}, {"role": "assistant", "content": [{"type": "text", "text": "**Source Attribution**:"}]}]
    return wire, answer


def prepare_case(processor: Any, tokenizer: Any, device: Any, record: dict[str, Any], spec: TemplateSpec) -> tuple[Any, str, dict[str, Any]]:
    wire, answer = messages(record, spec); rendered = render_continued_assistant(processor, wire, "**Source Attribution**:")
    inputs = prepare_multimodal_inputs(processor, wire, rendered, device=device)
    located = locate_template_positions(tokenizer, rendered, inputs, answer, spec)
    return inputs, rendered, located


def class_token_ids(tokenizer: Any) -> list[int]:
    output = []
    for label in map(str, range(9)):
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1: raise ValueError(f"Numeric label is not one token: {label}={ids}")
        output.append(int(ids[0]))
    return output


def _append_candidate(inputs: Any, candidate_ids: Sequence[int]) -> dict[str, torch.Tensor]:
    result = {key: value for key, value in inputs.items()}
    ids = torch.tensor([list(map(int, candidate_ids))], device=inputs.input_ids.device, dtype=inputs.input_ids.dtype)
    result["input_ids"] = torch.cat([inputs.input_ids, ids], dim=1)
    if "attention_mask" in result:
        ones = torch.ones((1, len(candidate_ids)), device=result["attention_mask"].device, dtype=result["attention_mask"].dtype)
        result["attention_mask"] = torch.cat([result["attention_mask"], ones], dim=1)
    return result


@torch.inference_mode()
def greedy_t3_audit(model: Any, tokenizer: Any, inputs: Any) -> dict[str, Any]:
    """Small, unconstrained formatting audit; never used to define the T3 score."""
    prefix_length = int(inputs.input_ids.shape[1])
    generated = model.generate(**inputs, do_sample=False, max_new_tokens=5)
    new_ids = generated[0, prefix_length:].detach().cpu().tolist()
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return {**parse_t3_greedy(text), "greedy_token_ids": list(map(int, new_ids)), "greedy_audit_scope": "smoke_only"}


@torch.inference_mode()
def score_prepared(model: Any, modules: Any, tokenizer: Any, inputs: Any, located: dict[str, Any], spec: TemplateSpec, *, hook: AdditiveActivationHook | None = None) -> dict[str, Any]:
    sac = int(located["P1_SAC"]["processed_index"])
    context = hook if hook is not None else __import__("contextlib").nullcontext()
    if spec.kind != "labels":
        with context: logits = run_logits_forward(model, inputs, [sac], modules)[sac]
        ids = class_token_ids(tokenizer); selected = [float(logits[i]) for i in ids]
        return numeric_score(selected, reversed_scale=spec.kind == "numeric_reversed", token_ids=ids)
    token_ids = [list(map(int, tokenizer.encode(label, add_special_tokens=False))) for label in T3_LABELS]
    prefix_length = int(inputs.input_ids.shape[1]); likelihoods = []
    # A full teacher-forcing forward is intentionally used for each fixed candidate.
    # This is slower than cache branching but exact and keeps multimodal position state simple.
    for index, ids in enumerate(token_ids):
        candidate_inputs = _append_candidate(inputs, ids)
        candidate_context = hook if index == 0 and hook is not None else __import__("contextlib").nullcontext()
        with candidate_context: output = model(**candidate_inputs, use_cache=False, return_dict=True)
        likelihoods.append(conditional_sequence_log_likelihood(output.logits, prefix_length, ids))
    return t3_score(likelihoods, token_ids)


@torch.inference_mode()
def capture_and_score(model: Any, modules: Any, tokenizer: Any, inputs: Any, located: dict[str, Any], spec: TemplateSpec, requested: dict[str, Sequence[int]], *, score_required: bool = True) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    positions = {name: int(located[name]["processed_index"]) for name in requested}
    layers = sorted({int(layer) for values in requested.values() for layer in values})
    forward = run_hooked_forward(model, inputs, modules, positions, logits_positions=[int(located["P1_SAC"]["processed_index"])])
    hidden = {f"{position}__L{layer}": forward.hidden_by_name[position][layer].detach().float().cpu().numpy().astype(np.float16) for position, wanted in requested.items() for layer in wanted}
    if not score_required:
        score = {"scoring_status": "not_required_for_geometry_only"}
    elif spec.kind == "labels":
        score = score_prepared(model, modules, tokenizer, inputs, located, spec)
    else:
        ids = class_token_ids(tokenizer); logits = forward.logits_by_position[int(located["P1_SAC"]["processed_index"])]
        score = numeric_score([float(logits[i]) for i in ids], reversed_scale=spec.kind == "numeric_reversed", token_ids=ids)
    return hidden, score
