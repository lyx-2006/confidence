from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from confidence_test.runtime_imports import load_runtime
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.model_adapter import model_input_device, resolve_language_modules, run_hooked_forward

from dp_SA.positions import locate_phase1_positions
from dp_SA.prompts import SA_PREFILL, phase1_prompt
from dp_SA.soft_score import class_token_ids, soft_sa_from_logits

from .build_manifest import build_manifest
from .config import (
    ARTIFACT_NAMES, HISTORICAL_LAYERS, HISTORICAL_POSITIONS, INFERENCE_PATH, LAYERS,
    LOGIT_TOLERANCE, MODEL_PATH, POSITIONS, RESULTS_ROOT, SOFT_SA_TOLERANCE,
    SOURCE_CAPTURE_ROOT,
)
from .io_utils import append_jsonl, atomic_json, canonical_hash, ensure_output_layout, load_jsonl, stage_update


def _messages(prompt: str, image_path: str) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": [{"type": "image", "image": str(Path(image_path).resolve())}, {"type": "text", "text": prompt}]},
        {"role": "assistant", "content": [{"type": "text", "text": SA_PREFILL}]},
    ]


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        with open(temporary, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _position_parity(historical: dict[str, Any], current: dict[str, Any]) -> None:
    for position in POSITIONS:
        old = historical["positions"][position]
        new = current[position]
        for field in ("processed_index", "rendered_index", "token_id", "token_text"):
            if old[field] != new[field]:
                raise ValueError(f"Position parity failed for {historical['case_id']} {position} {field}")
    if list(historical["phase1_answer_span"]) != list(current["phase1_answer_span"]):
        raise ValueError(f"Answer span parity failed: {historical['case_id']}")
    if list(historical["phase1_answer_token_ids"]) != list(current["phase1_answer_token_ids"]):
        raise ValueError(f"Answer token parity failed: {historical['case_id']}")


def _hidden_parity(source_root: Path, historical: dict[str, Any], current: dict[str, np.ndarray]) -> int:
    path = source_root / str(historical["historical_hidden_file"])
    checked = 0
    with np.load(path) as payload:
        for position in HISTORICAL_POSITIONS:
            for layer in HISTORICAL_LAYERS:
                key = f"{position}__L{layer}"
                if key not in current:
                    continue
                if key not in payload:
                    raise ValueError(f"Historical hidden overlap missing: {historical['case_id']} {key}")
                if not np.array_equal(np.asarray(payload[key]), current[key]):
                    raise ValueError(f"Float16 hidden bitwise parity failed: {historical['case_id']} {key}")
                checked += 1
    return checked


def run_capture(
    root: Path,
    *,
    resume: bool,
    max_records: int | None = None,
    layers: Sequence[int] = LAYERS,
) -> dict[str, Any]:
    manifest_path = root / "artifacts" / ARTIFACT_NAMES["manifest"]
    if not manifest_path.is_file():
        build_manifest(root)
    manifest = load_jsonl(manifest_path)
    if max_records is not None:
        manifest = manifest[:max_records]
    selected_layers = tuple(sorted(set(int(value) for value in layers)))
    if not selected_layers or any(layer not in LAYERS for layer in selected_layers):
        raise ValueError("Capture layers must be a non-empty subset of configured layers")
    output = root / "artifacts" / ARTIFACT_NAMES["capture"]
    existing = load_jsonl(output, repair_trailing=resume)
    completed = {str(row["case_id"]) for row in existing if row.get("status") == "completed"}
    requested = {str(row["case_id"]) for row in manifest}
    if completed == requested and existing:
        checked_cells = 0
        for row in existing:
            path = root / str(row["hidden_file"])
            if not path.is_file():
                raise FileNotFoundError(f"Resumed capture hidden file is missing: {path}")
            checked_cells += int(row.get("parity", {}).get("hidden_cells_bitwise_checked", 0))
        audit_path = root / "artifacts" / "capture_parity_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {"status": "passed", "record_count": len(manifest), "overlap_cells_checked": checked_cells}
        summary = {"status": "complete", "resumed_noop": True, "record_count": len(manifest), "hidden_vector_count": sum(len(row.get("hidden_keys", [])) for row in existing), "parity_audit": audit}
        stage_update(root, "capture", "complete", **{key: value for key, value in summary.items() if key != "status"})
        return summary
    runtime = load_runtime(INFERENCE_PATH)
    inference = runtime.QwenVLInference(str(MODEL_PATH))
    modules = resolve_language_modules(inference.model)
    tokenizer = getattr(inference.processor, "tokenizer", inference.processor)
    sa_ids = class_token_ids(tokenizer)
    checked_cells = 0
    stage_update(root, "capture", "running", total=len(manifest), completed=len(completed), layers=list(selected_layers))
    for ordinal, row in enumerate(manifest, 1):
        case_id = str(row["case_id"])
        if case_id in completed:
            continue
        answer = str(row["phase0_raw_answer"])
        prompt = phase1_prompt(str(row["question"]), str(row["text_clue"]), answer)
        if prompt != row["phase1_prompt"] or canonical_hash(prompt) != row["phase1_prompt_hash"]:
            raise ValueError(f"Frozen Phase 1 prompt parity failed: {case_id}")
        messages = _messages(prompt, str(row["image_path"]))
        rendered = render_continued_assistant(inference.processor, messages, SA_PREFILL)
        inputs = prepare_multimodal_inputs(inference.processor, messages, rendered, device=model_input_device(inference))
        located = locate_phase1_positions(tokenizer, rendered, inputs, answer)
        _position_parity(row, located)
        positions = {name: int(located[name]["processed_index"]) for name in POSITIONS}
        forward = run_hooked_forward(inference.model, inputs, modules, positions, logits_positions=[positions["P1_SAC"]])
        score = soft_sa_from_logits(forward.logits_by_position[positions["P1_SAC"]], sa_ids)
        arrays = {
            f"{position}__L{layer}": forward.hidden_by_name[position][layer].detach().float().cpu().numpy().astype(np.float16)
            for position in POSITIONS for layer in selected_layers
        }
        checked_cells += _hidden_parity(SOURCE_CAPTURE_ROOT, row, arrays)
        old_logits = np.asarray(row["class_logits"], dtype=float)
        new_logits = np.asarray(score["class_logits"], dtype=float)
        max_logit_difference = float(np.max(np.abs(old_logits - new_logits)))
        soft_difference = abs(float(row["soft_sa_image_score"]) - float(score["soft_sa_image_score"]))
        if max_logit_difference > LOGIT_TOLERANCE:
            raise ValueError(f"Historical logits parity failed: {case_id} {max_logit_difference}")
        if int(row["argmax_hard_class"]) != int(score["argmax_hard_class"]):
            raise ValueError(f"SAC hard class parity failed: {case_id}")
        if soft_difference > SOFT_SA_TOLERANCE:
            raise ValueError(f"Soft SA parity failed: {case_id} {soft_difference}")
        hidden_rel = Path("artifacts") / "hidden" / f"{case_id}.npz"
        _atomic_npz(root / hidden_rel, arrays)
        append_jsonl(output, {
            "status": "completed", "case_id": case_id, "item_id": row["item_id"], "prior_index": row["prior_index"], "condition": row["condition"],
            "outer_fold": row["outer_fold"], "phase0_raw_answer": answer, "phase1_prompt_hash": canonical_hash(prompt),
            "positions": located, "phase1_answer_span": located["phase1_answer_span"], "phase1_answer_token_ids": located["phase1_answer_token_ids"],
            "hidden_file": str(hidden_rel), "hidden_keys": sorted(arrays), **score,
            "parity": {"hidden_cells_bitwise_checked": len(HISTORICAL_POSITIONS) * len(set(selected_layers).intersection(HISTORICAL_LAYERS)), "max_abs_logit_difference": max_logit_difference, "soft_sa_abs_difference": soft_difference, "status": "passed"},
        })
        completed.add(case_id)
        if ordinal % 10 == 0:
            stage_update(root, "capture", "running", total=len(manifest), completed=len(completed), checked_overlap_cells=checked_cells)
    if completed != requested:
        raise RuntimeError("Capture did not complete every requested record")
    audit = {"status": "passed", "record_count": len(manifest), "overlap_cells_checked": checked_cells, "hidden_parity": "float16_bitwise", "position_and_span_parity": "exact", "soft_sa_tolerance": SOFT_SA_TOLERANCE, "logit_tolerance": LOGIT_TOLERANCE}
    atomic_json(root / "artifacts" / "capture_parity_audit.json", audit)
    summary = {"status": "complete", "record_count": len(manifest), "hidden_vector_count": len(manifest) * len(POSITIONS) * len(selected_layers), "parity_audit": audit}
    stage_update(root, "capture", "complete", **{key: value for key, value in summary.items() if key != "status"})
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    root = ensure_output_layout(RESULTS_ROOT, resume=args.resume)
    print(json.dumps(run_capture(root, resume=args.resume), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
