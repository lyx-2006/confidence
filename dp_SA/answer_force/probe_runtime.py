from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from dp_SA.io_utils import canonical_hash, load_jsonl, sha256_file
from dp_SA.positions import locate_phase1_positions
from dp_SA.prompts import SA_INSTRUCTION_START, SA_PREFILL, phase1_prompt
from dp_SA.patching.protocol import resolved_image
from dp_SA.soft_score import soft_sa_from_logits
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.model_adapter import LanguageModules, model_input_device, run_logits_forward

from .config import FLOAT_TOLERANCE, PRIMARY_LAYER, PRIMARY_POSITION


@dataclass(frozen=True)
class FoldProbe:
    models: dict[int, Pipeline]
    item_to_fold: dict[str, int]
    records: list[dict[str, Any]]

    def fold_for_item(self, item_id: Any) -> int:
        key = str(item_id)
        if key not in self.item_to_fold:
            raise KeyError(f"item is absent from frozen split assignments: {key}")
        return int(self.item_to_fold[key])

    def predict(self, item_id: Any, hidden: Any) -> float:
        fold = self.fold_for_item(item_id)
        array = np.asarray(hidden, dtype=np.float32).reshape(1, -1)
        # Historical hidden files are float16.  Perform the same round-trip on
        # every fresh activation before it enters the reconstructed model.
        array = array.astype(np.float16).astype(np.float32)
        value = float(self.models[fold].predict(array)[0])
        if not np.isfinite(value):
            raise ValueError(f"non-finite probe prediction for item {item_id}")
        return value


def _hidden_array(source_root: Path, row: Mapping[str, Any]) -> np.ndarray:
    path = source_root / str(row["hidden_file"])
    if not path.is_file():
        raise FileNotFoundError(f"historical hidden file missing: {path}")
    with np.load(path) as payload:
        key = f"{PRIMARY_POSITION}__L{PRIMARY_LAYER}"
        if key not in payload:
            raise ValueError(f"historical hidden key missing: {row['case_id']} {key}")
        value = np.asarray(payload[key], dtype=np.float32)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError(f"historical hidden is invalid: {row['case_id']}")
    return value


def reconstruct_probe(source_root: Path, split_path: Path, historical_oof_path: Path) -> tuple[FoldProbe, dict[str, Any]]:
    records = [row for row in load_jsonl(source_root / "capture" / "results.jsonl") if row.get("status") == "completed"]
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    item_to_fold = {str(key): int(value) for key, value in split_payload["item_to_fold"].items()}
    if int(split_payload.get("n_splits", 5)) != 5:
        raise ValueError("frozen probe split must have exactly five folds")
    if {str(row["item_id"]) for row in records} != set(item_to_fold):
        raise ValueError("clean capture records do not exactly cover frozen split items")
    case_ids = [str(row["case_id"]) for row in records]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate clean capture case IDs")
    X = np.stack([_hidden_array(source_root, row) for row in records])
    y = np.asarray([float(row["soft_sa_image_score"]) for row in records], dtype=np.float64)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("historical probe inputs are non-finite")
    models: dict[int, Pipeline] = {}
    prediction = np.empty(len(records), dtype=np.float64)
    prediction_fold = np.empty(len(records), dtype=np.int64)
    # LSQR can otherwise vary at the last few bits with a multithreaded BLAS
    # backend, which is enough to fail the deliberately strict OOF parity
    # gate.  Keep the reconstruction numerically reproducible regardless of
    # the caller's process environment.
    with threadpool_limits(limits=1):
        for fold in range(5):
            train = np.asarray([item_to_fold[str(row["item_id"])] != fold for row in records], dtype=bool)
            test = ~train
            if not test.any() or not train.any():
                raise ValueError(f"invalid empty probe fold {fold}")
            model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0, solver="lsqr"))])
            model.fit(X[train].astype(np.float16).astype(np.float32), y[train])
            prediction[test] = model.predict(X[test].astype(np.float16).astype(np.float32))
            prediction_fold[test] = fold
            models[fold] = model
    historical = {
        str(row["case_id"]): row
        for row in load_jsonl(historical_oof_path)
        if str(row.get("position")) == PRIMARY_POSITION and int(row.get("layer", -1)) == PRIMARY_LAYER
    }
    if set(historical) != set(case_ids):
        raise ValueError("historical OOF predictions do not exactly cover clean capture")
    differences = np.asarray([abs(float(prediction[index]) - float(historical[case_id]["prediction"])) for index, case_id in enumerate(case_ids)])
    max_difference = float(differences.max(initial=0.0))
    audit = {
        "position": PRIMARY_POSITION, "layer": PRIMARY_LAYER, "alpha": 1.0, "solver": "lsqr",
        "scaler": "StandardScaler", "fold_count": 5, "record_count": len(records),
        "item_count": len(item_to_fold), "prediction_count": len(prediction),
        "max_abs_difference": max_difference, "mean_abs_difference": float(differences.mean()),
        "over_tolerance_count": int((differences > FLOAT_TOLERANCE).sum()),
        "historical_oof_sha256": sha256_file(historical_oof_path),
        "split_sha256": sha256_file(split_path), "status": "passed" if max_difference <= FLOAT_TOLERANCE else "failed",
        "fold_counts": {str(fold): int((prediction_fold == fold).sum()) for fold in range(5)},
    }
    if max_difference > FLOAT_TOLERANCE:
        raise ValueError(f"probe parity failed: max abs difference {max_difference} > {FLOAT_TOLERANCE}")
    return FoldProbe(models=models, item_to_fold=item_to_fold, records=records), audit


def _input_ids(inputs: Any) -> torch.Tensor:
    return inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids


def prepare_phase1_condition(inference: Any, row: Mapping[str, Any], answer: str) -> tuple[str, Any, dict[str, Any]]:
    if not isinstance(answer, str) or not answer or "\n" in answer or "\r" in answer:
        raise ValueError("Phase 1 fixed answer must be non-empty and single-line")
    prompt = phase1_prompt(str(row["question"]), str(row["text_clue"]), answer)
    image = resolved_image(dict(row), None)
    messages = [
        {"role": "user", "content": [{"type": "image", "image": str(image)}, {"type": "text", "text": prompt}]},
        {"role": "assistant", "content": [{"type": "text", "text": SA_PREFILL}]},
    ]
    rendered = render_continued_assistant(inference.processor, messages, SA_PREFILL)
    inputs = prepare_multimodal_inputs(inference.processor, messages, rendered, device=model_input_device(inference))
    tokenizer = getattr(inference.processor, "tokenizer", inference.processor)
    located = locate_phase1_positions(tokenizer, rendered, inputs, answer)
    details = {
        "messages": messages, "messages_hash": canonical_hash(messages), "rendered": rendered,
        "prompt": prompt, "rendered_hash": canonical_hash(rendered), "located": located, "image_path": str(image),
        "image_sha256": sha256_file(image), "answer": answer,
    }
    return rendered, inputs, details


def _output_tensor(output: Any) -> torch.Tensor:
    tensor = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
        raise TypeError(f"decoder block output must be rank-3 tensor, got {type(tensor)!r}")
    return tensor


def run_primary_forward(model: torch.nn.Module, inputs: Any, modules: LanguageModules, *, panl_position: int, sac_position: int, class_token_ids: Sequence[int]) -> tuple[torch.Tensor, dict[str, Any]]:
    if PRIMARY_LAYER >= modules.num_hidden_layers:
        raise ValueError(f"primary layer {PRIMARY_LAYER} is outside model")
    captured: list[torch.Tensor] = []

    def capture(_module: Any, _args: Any, output: Any) -> None:
        tensor = _output_tensor(output)
        if int(tensor.shape[1]) <= panl_position:
            raise ValueError("PANL position is outside decoder output")
        captured.append(tensor[0, panl_position, :].detach().clone())

    handle = modules.language_layers[PRIMARY_LAYER].register_forward_hook(capture)
    try:
        logits = run_logits_forward(model, inputs, [int(sac_position)], modules)[int(sac_position)]
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"primary layer hook did not run exactly once: {len(captured)}")
    score = soft_sa_from_logits(logits, class_token_ids)
    if not np.isfinite(np.asarray(score["class_logits"], dtype=float)).all():
        raise ValueError("SAC logits are non-finite")
    return captured[0].detach().float().cpu(), score


def prompt_only_answer_audit(clean: Mapping[str, Any], counterfactual: Mapping[str, Any]) -> dict[str, Any]:
    clean_answer, cf_answer = str(clean["answer"]), str(counterfactual["answer"])
    clean_rendered, cf_rendered = str(clean["rendered"]), str(counterfactual["rendered"])
    clean_field = f"**Answer**: {clean_answer}"
    cf_field = f"**Answer**: {cf_answer}"
    instruction_clean = clean_rendered.find(SA_INSTRUCTION_START)
    instruction_cf = cf_rendered.find(SA_INSTRUCTION_START)
    clean_start = clean_rendered.rfind(clean_field, 0, instruction_clean if instruction_clean >= 0 else len(clean_rendered))
    cf_start = cf_rendered.rfind(cf_field, 0, instruction_cf if instruction_cf >= 0 else len(cf_rendered))
    if clean_start < 0 or cf_start < 0:
        raise ValueError("answer field missing from rendered prompt")
    clean_masked = clean_rendered[:clean_start] + "**Answer**: <ANSWER>" + clean_rendered[clean_start + len(clean_field) :]
    cf_masked = cf_rendered[:cf_start] + "**Answer**: <ANSWER>" + cf_rendered[cf_start + len(cf_field) :]
    clean_ids = _input_ids(clean["inputs"])[0].detach().cpu().tolist()
    cf_ids = _input_ids(counterfactual["inputs"])[0].detach().cpu().tolist()
    clean_span = list(map(int, clean["details"]["located"]["phase1_answer_span"]))
    cf_span = list(map(int, counterfactual["details"]["located"]["phase1_answer_span"]))
    prefix_equal = clean_ids[: clean_span[0]] == cf_ids[: cf_span[0]]
    suffix_equal = clean_ids[clean_span[1] :] == cf_ids[cf_span[1] :]
    passed = clean_masked == cf_masked and prefix_equal and suffix_equal and clean["details"]["image_sha256"] == counterfactual["details"]["image_sha256"]
    audit = {
        "masked_rendered_equal": clean_masked == cf_masked, "prefix_token_ids_equal": prefix_equal,
        "suffix_token_ids_equal": suffix_equal, "image_sha256_equal": clean["details"]["image_sha256"] == counterfactual["details"]["image_sha256"],
        "passed": passed,
    }
    if not passed:
        raise ValueError(f"prompt-only-answer-span audit failed: {audit}")
    return audit


__all__ = ["FoldProbe", "prepare_phase1_condition", "prompt_only_answer_audit", "reconstruct_probe", "run_primary_forward"]
