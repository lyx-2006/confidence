from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from confidence_test.runtime_imports import load_runtime
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.model_adapter import model_input_device, resolve_language_modules, run_hooked_forward

from dp_SA.positions import locate_phase1_positions
from dp_SA.prompts import SA_PREFILL
from dp_SA.soft_score import class_token_ids, soft_sa_from_logits

from .config import (
    CHECKPOINT_CLEAN, CHECKPOINT_COMPLETION, CHECKPOINT_EXTENSION_CLEAN,
    CHECKPOINT_ROOT, EXPECTED_SOURCE_SHA256, FLOAT_TOLERANCE,
    HISTORICAL_CAPTURE, HISTORICAL_CAPTURE_CONFIG, HISTORICAL_CONSTRUCTION,
    HISTORICAL_TEST, INFERENCE_PATH, LAYERS, MODEL_PATH, POSITION,
    RESULTS_ROOT, SMOKE_LAYERS,
)
from .fingerprint import check_or_write, experiment_config
from .io_utils import append_jsonl, atomic_json, atomic_jsonl, atomic_npz, atomic_text, canonical_hash, load_jsonl, sha256_file
from .manifests import build_formal_design, build_smoke_design, leakage_audit


MANIFEST_NAMES = (
    "candidate_manifest.jsonl", "family_manifest.jsonl", "fold_assignments.jsonl",
    "test_manifest.jsonl", "construction_family_cells.jsonl", "construction_distribution.jsonl",
)


def _manifest_fingerprints(root: Path) -> dict[str, str]:
    directory = root / "artifacts" / "manifests"
    return {name: sha256_file(directory / name) for name in MANIFEST_NAMES}


def _split_markdown(design: dict[str, Any], audit: dict[str, Any], *, smoke: bool) -> str:
    lines = ["# Answer-matched LAT steering split design", "", f"- smoke_only: `{str(smoke).lower()}`", f"- candidate records: {len(design['candidates'])}", f"- families: {len(design['families'])}", f"- test families: {len(design['test'])}", f"- leakage gate: {audit['status']}", f"- minimum LOAO eligible colors: {audit['loao_minimum']}", ""]
    if not smoke:
        lines += ["- selected scheme: grouped 15-fold cross-fitting", "- Blue has 9 test families and is exploratory_sparse; every other answer has 15.", "- Split selection used clean metadata only; intervention outcomes were unavailable.", ""]
    return "\n".join(lines)


def freeze_design(root: Path, *, smoke: bool, resume: bool) -> dict[str, Any]:
    directory = root / "artifacts" / "manifests"
    existing = any((directory / name).exists() for name in MANIFEST_NAMES)
    if existing:
        if not resume: raise FileExistsError("Frozen split artifacts exist; use --resume")
        rows = {name: load_jsonl(directory / name) for name in MANIFEST_NAMES}
        source_path = directory / "source_and_fingerprints.json"
        if not source_path.is_file() or any(not rows[name] for name in MANIFEST_NAMES): raise ValueError("Frozen split is incomplete")
        source = json.loads(source_path.read_text())
        if source.get("manifest_fingerprints") != _manifest_fingerprints(root): raise ValueError("Frozen split fingerprint mismatch")
        return {"candidates": rows["candidate_manifest.jsonl"], "families": rows["family_manifest.jsonl"], "fold_assignments": rows["fold_assignments.jsonl"], "test": rows["test_manifest.jsonl"], "construction_cells": rows["construction_family_cells.jsonl"], "construction_distribution": rows["construction_distribution.jsonl"], "smoke_only": smoke, "resumed": True}

    design = build_smoke_design() if smoke else build_formal_design()
    audit = leakage_audit(design, smoke=smoke)
    if audit["status"] != "passed": raise RuntimeError(f"Split gate failed: {audit}")
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "candidate_manifest.jsonl": design["candidates"], "family_manifest.jsonl": design["families"],
        "fold_assignments.jsonl": design["fold_assignments"], "test_manifest.jsonl": design["test"],
        "construction_family_cells.jsonl": design["construction_cells"], "construction_distribution.jsonl": design["construction_distribution"],
    }
    for name, rows in artifacts.items(): atomic_jsonl(directory / name, rows)
    fingerprints = _manifest_fingerprints(root)
    source = {
        "smoke_only": smoke, "historical_capture": str(HISTORICAL_CAPTURE.resolve()),
        "historical_capture_sha256": sha256_file(HISTORICAL_CAPTURE),
        "historical_construction_sha256": sha256_file(HISTORICAL_CONSTRUCTION),
        "historical_test_sha256": sha256_file(HISTORICAL_TEST),
        "manifest_fingerprints": fingerprints,
        "design_fingerprint": canonical_hash(fingerprints),
    }
    atomic_json(directory / "source_and_fingerprints.json", source)
    atomic_json(root / "artifacts" / "diagnostics" / "family_overlap_audit.json", design["family_audit"])
    atomic_json(root / "artifacts" / "diagnostics" / "split_candidate_comparison.json", design["comparison"])
    atomic_json(root / "progress" / "split_gate.json", audit)
    atomic_text(root / "progress" / "split_design.md", _split_markdown(design, audit, smoke=smoke))
    return {**design, "smoke_only": smoke, "resumed": False}


def _messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "image", "image": str(Path(row["image_path"]).resolve())}, {"type": "text", "text": str(row["phase1_prompt"])}]}, {"role": "assistant", "content": [{"type": "text", "text": SA_PREFILL}]}]


def _validate_source_hashes() -> None:
    actual = {
        "historical_capture": sha256_file(HISTORICAL_CAPTURE),
        "historical_capture_config": sha256_file(HISTORICAL_CAPTURE_CONFIG),
        "checkpoint_clean": sha256_file(CHECKPOINT_CLEAN),
        "checkpoint_extension_clean": sha256_file(CHECKPOINT_EXTENSION_CLEAN),
        "checkpoint_completion": sha256_file(CHECKPOINT_COMPLETION),
    }
    if actual != EXPECTED_SOURCE_SHA256: raise ValueError(f"Historical reuse source changed: {actual}")


def _reuse_rows(candidates: Sequence[dict[str, Any]]) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    _validate_source_hashes()
    base = {str(row["case_id"]): row for row in load_jsonl(CHECKPOINT_CLEAN) if row.get("status") == "completed"}
    extension = {str(row["case_id"]): row for row in load_jsonl(CHECKPOINT_EXTENSION_CLEAN) if row.get("status") == "completed"}
    candidate_ids = {str(row["case_id"]) for row in candidates}
    common = sorted(candidate_ids & base.keys() & extension.keys())
    if len(common) != 150: raise ValueError(f"Expected 150 reusable hidden cases, found {len(common)}")
    return {case_id: (base[case_id], extension[case_id]) for case_id in common}


def _parity(historical: dict[str, Any], located: dict[str, Any], score: dict[str, Any]) -> dict[str, Any]:
    if canonical_hash(historical["phase1_prompt"]) != historical["phase1_prompt_hash"]: raise ValueError("Historical prompt hash is inconsistent")
    if list(located["phase1_answer_token_ids"]) != list(historical["phase1_answer_token_ids"]): raise ValueError("Answer token IDs changed")
    for name in ("P1_LAT", "P1_SAC"):
        if int(located[name]["processed_index"]) != int(historical["positions"][name]["processed_index"]): raise ValueError(f"Position parity failed at {name}")
    logit_error = float(np.max(np.abs(np.asarray(score["class_logits"]) - np.asarray(historical["class_logits"]))))
    probability_error = float(np.max(np.abs(np.asarray(score["class_probabilities"]) - np.asarray(historical["class_probabilities"]))))
    soft_error = abs(float(score["soft_sa_image_score"]) - float(historical["soft_sa_image_score"]))
    if max(logit_error, probability_error, soft_error) > FLOAT_TOLERANCE or int(score["argmax_hard_class"]) != int(historical["argmax_hard_class"]): raise ValueError("Clean score parity failed")
    return {"passed": True, "logit_max_abs_error": logit_error, "probability_max_abs_error": probability_error, "soft_sa_abs_error": soft_error, "hard_class_equal": True}


def _write_reused(root: Path, candidate: dict[str, Any], base: dict[str, Any], extension: dict[str, Any], config_fingerprint: str) -> dict[str, Any]:
    arrays = {}
    for source_row, layers in ((base, (10, 14)), (extension, (12, 16))):
        source_path = CHECKPOINT_ROOT / source_row["hidden_file"]
        if sha256_file(source_path) != source_row["hidden_sha256"]: raise ValueError(f"Reusable hidden changed: {candidate['case_id']}")
        with np.load(source_path) as payload:
            for layer in layers: arrays[f"{POSITION}__L{layer}"] = np.asarray(payload[f"{POSITION}__L{layer}"], dtype=np.float16)
    relative = Path("artifacts") / "hidden" / f"{candidate['case_id']}.npz"; destination = root / relative
    atomic_npz(destination, arrays)
    historical = candidate
    return {
        "status": "completed", "case_id": candidate["case_id"], "family_id": candidate["family_id"], "item_id": str(candidate["item_id"]),
        "canonical_answer": candidate["canonical_answer"], "sa_side": candidate["sa_side"], "phase0_raw_answer": candidate["phase0_raw_answer"],
        "phase1_prompt_hash": candidate["phase1_prompt_hash"], "rendered_prompt_hash": base["rendered_prompt_hash"],
        "phase1_answer_token_ids": candidate["phase1_answer_token_ids"], "positions": base["positions"],
        "class_token_ids": candidate["class_token_ids"], "class_logits": candidate["class_logits"], "class_probabilities": candidate["class_probabilities"],
        "soft_sa_image_score": candidate["soft_sa_image_score"], "argmax_hard_class": candidate["argmax_hard_class"],
        "hidden_file": str(relative), "hidden_sha256": sha256_file(destination), "hidden_keys": sorted(arrays), "hidden_dtype": "float16",
        "source": "checkpoint_steering_reuse", "source_hidden_sha256": {"base": base["hidden_sha256"], "extension": extension["hidden_sha256"]},
        "parity": {"passed": True, "source_checkpoint_parity": True}, "config_fingerprint": config_fingerprint,
    }


def capture_hidden(root: Path, candidates: Sequence[dict[str, Any]], *, smoke: bool, resume: bool, config: dict[str, Any]) -> dict[str, Any]:
    clean_path = root / "artifacts" / "diagnostics" / "clean_capture.jsonl"
    existing_rows = [row for row in load_jsonl(clean_path) if row.get("status") == "completed"]
    existing = {str(row["case_id"]): row for row in existing_rows}
    expected = {str(row["case_id"]) for row in candidates}
    for row in existing.values():
        if row.get("config_fingerprint") != config["fingerprint"] or sha256_file(root / row["hidden_file"]) != row["hidden_sha256"]: raise ValueError("Existing clean capture fingerprint mismatch")
    if set(existing) == expected:
        summary = {"status": "complete", "smoke_only": smoke, "completed": len(existing), "expected": len(expected), "reused_hidden": sum(row.get("source") == "checkpoint_steering_reuse" for row in existing.values()), "new_gpu_forwards": 0, "resumed_noop": True}
        atomic_json(root / "progress" / "prepare_progress.json", summary); return summary
    historical = {str(row["case_id"]): row for row in load_jsonl(HISTORICAL_CAPTURE) if row.get("status") == "completed"}
    reusable = {} if smoke else _reuse_rows(candidates)
    reused_count = 0
    for candidate in candidates:
        case_id = str(candidate["case_id"])
        if case_id in existing or case_id not in reusable: continue
        row = _write_reused(root, candidate, *reusable[case_id], config["fingerprint"])
        append_jsonl(clean_path, row); existing[case_id] = row; reused_count += 1
    missing = [row for row in candidates if str(row["case_id"]) not in existing]
    if not missing:
        summary = {"status": "complete", "smoke_only": smoke, "completed": len(existing), "expected": len(expected), "reused_hidden": reused_count, "new_gpu_forwards": 0, "resumed_noop": False}
        atomic_json(root / "progress" / "prepare_progress.json", summary); return summary
    runtime = load_runtime(INFERENCE_PATH); inference = runtime.QwenVLInference(str(MODEL_PATH)); modules = resolve_language_modules(inference.model)
    tokenizer = getattr(inference.processor, "tokenizer", inference.processor); token_ids = class_token_ids(tokenizer); device = model_input_device(inference)
    layers = SMOKE_LAYERS if smoke else LAYERS; started = time.time(); new_forwards = 0
    for candidate in missing:
        case_id = str(candidate["case_id"])
        try:
            source = historical[case_id]; messages = _messages(candidate); rendered = render_continued_assistant(inference.processor, messages, SA_PREFILL)
            inputs = prepare_multimodal_inputs(inference.processor, messages, rendered, device=device); located = locate_phase1_positions(tokenizer, rendered, inputs, str(candidate["phase0_raw_answer"]))
            lat = int(located[POSITION]["processed_index"]); sac = int(located["P1_SAC"]["processed_index"])
            forward = run_hooked_forward(inference.model, inputs, modules, {POSITION: lat}, logits_positions=[sac]); new_forwards += 1
            score = soft_sa_from_logits(forward.logits_by_position[sac], token_ids); parity = _parity(source, located, score)
            arrays = {f"{POSITION}__L{layer}": forward.hidden_by_name[POSITION][layer].detach().float().cpu().numpy().astype(np.float16) for layer in layers}
            relative = Path("artifacts") / "hidden" / f"{case_id}.npz"; destination = root / relative; atomic_npz(destination, arrays)
            result = {"status": "completed", "case_id": case_id, "family_id": candidate["family_id"], "item_id": str(candidate["item_id"]), "canonical_answer": candidate["canonical_answer"], "sa_side": candidate["sa_side"], "phase0_raw_answer": candidate["phase0_raw_answer"], "phase1_prompt_hash": candidate["phase1_prompt_hash"], "rendered_prompt_hash": canonical_hash(rendered), "phase1_answer_token_ids": located["phase1_answer_token_ids"], "positions": located, **score, "hidden_file": str(relative), "hidden_sha256": sha256_file(destination), "hidden_keys": sorted(arrays), "hidden_dtype": "float16", "source": "new_clean_forward", "parity": parity, "config_fingerprint": config["fingerprint"]}
            append_jsonl(clean_path, result); existing[case_id] = result
        except Exception as exc:
            failure = {"stage": "prepare_capture", "case_id": case_id, "type": type(exc).__name__, "message": str(exc), "timestamp": time.time()}; append_jsonl(root / "progress" / "failures.jsonl", failure); atomic_json(root / "progress" / "prepare_progress.json", {"status": "failed", "completed": len(existing), "expected": len(expected), "failure": failure}); raise
        atomic_json(root / "progress" / "prepare_progress.json", {"status": "running", "completed": len(existing), "expected": len(expected), "reused_hidden": reused_count, "new_gpu_forwards": new_forwards, "elapsed_seconds": time.time() - started, "last_case_id": case_id})
    summary = {"status": "complete", "smoke_only": smoke, "completed": len(existing), "expected": len(expected), "reused_hidden": reused_count, "new_gpu_forwards": new_forwards, "resumed_noop": new_forwards == 0 and reused_count == 0, "elapsed_seconds": time.time() - started}
    atomic_json(root / "progress" / "prepare_progress.json", summary); return summary


def run_prepare(*, output_root: Path = RESULTS_ROOT, smoke: bool = False, resume: bool = False) -> dict[str, Any]:
    root = Path(output_root); root.mkdir(parents=True, exist_ok=True)
    failure_path = root / "progress" / "failures.jsonl"
    if not failure_path.exists(): atomic_jsonl(failure_path, [])
    design = freeze_design(root, smoke=smoke, resume=resume)
    fingerprints = _manifest_fingerprints(root); config = experiment_config(smoke=smoke, manifest_fingerprints=fingerprints)
    check_or_write(root / "progress" / "prepare_config.json", config, resume=resume)
    capture = capture_hidden(root, design["candidates"], smoke=smoke, resume=resume, config=config)
    return {"status": "complete", "smoke_only": smoke, "split_gate": "passed", "candidate_count": len(design["candidates"]), "family_count": len(design["families"]), "test_count": len(design["test"]), "capture": capture, "config_fingerprint": config["fingerprint"]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root"); parser.add_argument("--resume", action="store_true"); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv); root = Path(args.output_root) if args.output_root else RESULTS_ROOT
    if args.smoke and not args.output_root: parser.error("--smoke requires an explicit output root outside formal results")
    print(json.dumps(run_prepare(output_root=root, smoke=args.smoke, resume=args.resume), ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
