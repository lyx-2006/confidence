from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch

from confidence_test.inference_extension import ASSISTANT_ANSWER_PREFILL, _user_content
from confidence_test.prompt_utils import STAGE1_TEXT_ANSWER_PROMPT
from confidence_test.runtime_imports import load_runtime
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.model_adapter import model_input_device, resolve_language_modules, run_logits_forward
from layer_metacognition.probe.prompts import IMAGE_ONLY_ANSWER_PROMPT

from .build_manifest import build_manifest, join_scores
from .config import ARTIFACT_NAMES, INFERENCE_PATH, MODEL_PATH, RESULTS_ROOT
from .io_utils import append_jsonl, canonical_hash, ensure_output_layout, load_jsonl, stage_update
from .metrics import candidate_metrics


def unique_specs(manifest: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    text: dict[tuple[str, int], dict[str, Any]] = {}
    image: dict[tuple[str, str], dict[str, Any]] = {}
    for row in manifest:
        text_key = (str(row["item_id"]), int(row["prior_index"]))
        candidate = {
            "modality": "text", "item_id": text_key[0], "prior_index": text_key[1],
            "condition": None, "question": row["question"], "text_clue": row["text_clue"],
            "image_path": None, "answer_classes": list(row["answer_classes"]), "target_answer": row["text_answer"],
        }
        old = text.setdefault(text_key, candidate)
        if old != candidate:
            raise ValueError(f"Inconsistent text unique key: {text_key}")
        image_key = (str(row["item_id"]), str(row["condition"]))
        candidate = {
            "modality": "image", "item_id": image_key[0], "prior_index": None,
            "condition": image_key[1], "question": row["question"], "text_clue": None,
            "image_path": row["image_path"], "answer_classes": list(row["answer_classes"]), "target_answer": row["image_answer"],
        }
        old = image.setdefault(image_key, candidate)
        if old != candidate:
            raise ValueError(f"Inconsistent image unique key: {image_key}")
    return list(text.values()) + list(image.values())


def _wire(spec: dict[str, Any]) -> tuple[str, list[dict[str, Any]], str]:
    if spec["modality"] == "text":
        prompt = STAGE1_TEXT_ANSWER_PROMPT.format(question=spec["question"], text_clue=spec["text_clue"])
        content = _user_content(prompt, None)
        if any(part.get("type") != "text" for part in content):
            raise AssertionError("Text-only wire contains a non-text modality")
    else:
        prompt = IMAGE_ONLY_ANSWER_PROMPT.format(question=spec["question"])
        content = _user_content(prompt, spec["image_path"])
        if [part.get("type") for part in content] != ["image", "text"] or "Text clue:" in prompt:
            raise AssertionError("Image-only wire is not isolated from the text clue")
    messages = [{"role": "user", "content": content}, {"role": "assistant", "content": [{"type": "text", "text": ASSISTANT_ANSWER_PREFILL}]}]
    return prompt, messages, ASSISTANT_ANSWER_PREFILL


def candidate_suffix_ids(tokenizer: Any, rendered: str, candidate: str) -> list[int]:
    base = list(map(int, tokenizer.encode(rendered, add_special_tokens=False)))
    full = list(map(int, tokenizer.encode(rendered + candidate, add_special_tokens=False)))
    if full[: len(base)] != base or len(full) <= len(base):
        raise ValueError(f"Candidate does not append cleanly after assistant prefill: {candidate!r}")
    return full[len(base) :]


def tokenizer_preflight(processor: Any, specs: Sequence[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    tokenizer = getattr(processor, "tokenizer", processor)
    audits: dict[str, Any] = {}
    any_multi = False
    for spec in specs:
        _prompt, messages, prefill = _wire(spec)
        rendered = render_continued_assistant(processor, messages, prefill)
        ids = {candidate: candidate_suffix_ids(tokenizer, rendered, candidate) for candidate in spec["answer_classes"]}
        if len(ids) != 12 or len({tuple(value) for value in ids.values()}) != 12:
            raise ValueError("Candidate token sequences are not twelve distinct values")
        any_multi = any_multi or any(len(value) != 1 for value in ids.values())
        key = f"{spec['modality']}:{spec['item_id']}:{spec.get('prior_index')}:{spec.get('condition')}"
        audits[key] = {"rendered_hash": canonical_hash(rendered), "candidate_token_ids": ids}
    return ("teacher_forced_sequence" if any_multi else "single_token"), audits


def _score_single(inference: Any, modules: Any, spec: dict[str, Any], rendered: str, messages: list[dict[str, Any]], token_ids: dict[str, list[int]]) -> list[float]:
    inputs = prepare_multimodal_inputs(inference.processor, messages, rendered, device=model_input_device(inference))
    position = int(inputs.input_ids.shape[1]) - 1
    logits = run_logits_forward(inference.model, inputs, [position], modules)[position]
    log_probs = torch.log_softmax(logits.double(), dim=-1)
    return [float(log_probs[token_ids[name][0]]) for name in spec["answer_classes"]]


def _score_sequences(inference: Any, modules: Any, spec: dict[str, Any], rendered: str, messages: list[dict[str, Any]], token_ids: dict[str, list[int]]) -> list[float]:
    base_inputs = prepare_multimodal_inputs(inference.processor, messages, rendered, device=model_input_device(inference))
    base_length = int(base_inputs.input_ids.shape[1])
    values: list[float] = []
    for candidate in spec["answer_classes"]:
        full_inputs = prepare_multimodal_inputs(inference.processor, messages, rendered + candidate, device=model_input_device(inference))
        full_length = int(full_inputs.input_ids.shape[1])
        suffix = [int(value) for value in full_inputs.input_ids[0, base_length:].tolist()]
        if suffix != token_ids[candidate] or full_length != base_length + len(suffix):
            raise ValueError(f"Processed candidate suffix mismatch: {candidate}")
        positions = list(range(base_length - 1, full_length - 1))
        logits = run_logits_forward(inference.model, full_inputs, positions, modules)
        total = 0.0
        for offset, token_id in enumerate(suffix):
            total += float(torch.log_softmax(logits[positions[offset]].double(), dim=-1)[token_id])
        values.append(total)
    return values


def score_unimodal(root: Path, *, resume: bool, max_specs: int | None = None) -> dict[str, Any]:
    manifest_path = root / "artifacts" / ARTIFACT_NAMES["manifest"]
    if not manifest_path.is_file():
        build_manifest(root)
    specs = unique_specs(load_jsonl(manifest_path))
    if max_specs is not None:
        specs = specs[:max_specs]
    output = root / "artifacts" / ARTIFACT_NAMES["unimodal"]
    existing_rows = load_jsonl(output, repair_trailing=resume)
    completed = {(row["modality"], str(row["item_id"]), row.get("prior_index"), row.get("condition")) for row in existing_rows}
    requested = {(spec["modality"], str(spec["item_id"]), spec.get("prior_index"), spec.get("condition")) for spec in specs}
    if completed == requested and existing_rows:
        join = join_scores(root)
        policy = str(existing_rows[0]["scoring_policy"])
        if any(str(row["scoring_policy"]) != policy for row in existing_rows):
            raise ValueError("Resumed unimodal scores mix scoring policies")
        summary = {"status": "complete", "resumed_noop": True, "scoring_policy": policy, "unique_key_count": len(specs), **join}
        stage_update(root, "score_unimodal", "complete", **{key: value for key, value in summary.items() if key != "status"})
        return summary
    runtime = load_runtime(INFERENCE_PATH)
    inference = runtime.QwenVLInference(str(MODEL_PATH))
    modules = resolve_language_modules(inference.model)
    policy, audits = tokenizer_preflight(inference.processor, specs)
    stage_update(root, "score_unimodal", "running", scoring_policy=policy, total=len(specs), completed=len(completed))
    for ordinal, spec in enumerate(specs, 1):
        key = (spec["modality"], str(spec["item_id"]), spec.get("prior_index"), spec.get("condition"))
        if key in completed:
            continue
        prompt, messages, prefill = _wire(spec)
        rendered = render_continued_assistant(inference.processor, messages, prefill)
        audit_key = f"{spec['modality']}:{spec['item_id']}:{spec.get('prior_index')}:{spec.get('condition')}"
        token_ids = audits[audit_key]["candidate_token_ids"]
        scores = _score_single(inference, modules, spec, rendered, messages, token_ids) if policy == "single_token" else _score_sequences(inference, modules, spec, rendered, messages, token_ids)
        metrics = candidate_metrics(spec["answer_classes"], scores, str(spec["target_answer"]))
        prefix = f"{spec['modality']}_"
        row = {
            "modality": spec["modality"], "item_id": spec["item_id"], "prior_index": spec.get("prior_index"), "condition": spec.get("condition"),
            "unique_key": list(key[1:]), "question": spec["question"], "text_clue": spec.get("text_clue"), "image_path": spec.get("image_path"),
            "prompt": prompt, "prompt_hash": canonical_hash(prompt), "rendered_hash": canonical_hash(rendered), "assistant_prefill": prefill,
            "input_modalities": [part["type"] for part in messages[0]["content"]], "scoring_policy": policy,
            f"{prefix}candidate_token_ids": token_ids,
            **{prefix + name: value for name, value in metrics.items()},
        }
        append_jsonl(output, row)
        completed.add(key)
        if ordinal % 25 == 0:
            stage_update(root, "score_unimodal", "running", scoring_policy=policy, total=len(specs), completed=len(completed))
    if completed != requested:
        raise RuntimeError("Unimodal scoring did not complete every unique key")
    join = join_scores(root)
    summary = {"status": "complete", "scoring_policy": policy, "unique_key_count": len(specs), **join}
    stage_update(root, "score_unimodal", "complete", **{key: value for key, value in summary.items() if key != "status"})
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    root = ensure_output_layout(RESULTS_ROOT, resume=args.resume)
    print(json.dumps(score_unimodal(root, resume=args.resume), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
