from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from dp_SA.attention_block.masking import AttentionBlockContext
from dp_SA.io_utils import atomic_json, atomic_jsonl, load_jsonl
from layer_metacognition.model_adapter import model_input_device, resolve_language_modules, run_logits_forward
from qwen3_chat_binary.config import MODEL_PATH
from qwen3_chat_binary.conversation import prepare_multimodal_inputs, render_stage2, stage2_messages
from qwen3_chat_binary.positions import locate_positions
from qwen3_chat_binary.prompts import LABELS
from qwen3_chat_binary.runtime import load_qwen3_inference
from qwen3_chat_binary.scoring import attribution_score, label_token_ids

from .config import (CLEAN_LOGIT_TOLERANCE, CLEAN_SCORE_TOLERANCE, CONDITIONS,
                     ROW_SUM_TOLERANCE, VARIANT, default_output)
from .core import add_cle_plus_1, class_margin, deterministic_argmax, edge_for_condition


def _forward(runtime: Any, modules: Any, inputs: Any, sac: int, ids: tuple[int, ...]):
    logits = run_logits_forward(runtime.model, inputs, [sac], modules)[sac]
    score = attribution_score(logits, ids)
    values = [float(score["label_logits"][label]) for label in LABELS]
    predicted, tie = deterministic_argmax(values)
    return values, score, predicted, tie


def _context(runtime: Any, row: dict[str, Any]):
    messages = stage2_messages(row["phase0_prompt"], row["image_path"],
                              row["phase0_raw_output"], VARIANT)
    rendered = render_stage2(runtime.processor, messages)
    inputs = prepare_multimodal_inputs(runtime.processor, messages, rendered,
                                      device=model_input_device(runtime))
    tokenizer = runtime.processor.tokenizer
    located = add_cle_plus_1(
        locate_positions(tokenizer, rendered, inputs, row["phase0_raw_output"], VARIANT),
        tokenizer, inputs.input_ids,
    )
    for name in ("LAT", "PANL", "PANL+1", "CLE", "SAC"):
        old, new = row["positions"][name], located["positions"][name]
        if (int(old["processed_index"]), int(old["token_id"])) != (
            int(new["processed_index"]), int(new["token_id"])):
            raise RuntimeError(f"Position drift for {row['case_id']} at {name}")
    return inputs, located, {k: int(v) for k, v in located["indices"].items()}


def trial_path(root: Path, case: str, condition: str, window: tuple[int, int] | None = None) -> Path:
    suffix = condition if window is None else f"{condition}__L{window[0]}-{window[1]}"
    return root / "artifacts/trials" / f"{case}__{suffix}.json"


def run(*, output_root: Path, model_path: Path = MODEL_PATH, resume: bool = False) -> dict[str, Any]:
    root = Path(output_root).resolve()
    config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    if Path(config["model"]).resolve() != Path(model_path).resolve():
        raise ValueError("Runtime model path differs from the prepared experiment config")
    fingerprint = config["fingerprint"]
    windows = tuple(tuple(map(int, x)) for x in config["windows"])
    cases = load_jsonl(root / "artifacts/manifests/test_manifest.jsonl")
    runtime = load_qwen3_inference(model_path, attn_implementation="eager")
    modules = resolve_language_modules(runtime.model)
    if modules.num_hidden_layers != 36:
        raise RuntimeError("Expected 36 Qwen3 language layers")
    if getattr(runtime.model.config, "_attn_implementation", None) != "eager":
        raise RuntimeError("Attention blocking requires eager attention")
    ids = label_token_ids(runtime.processor.tokenizer)
    new_forwards, started = 0, time.time()
    try:
        for ordinal, row in enumerate(cases, 1):
            inputs, located, positions = _context(runtime, row)
            clean_file = trial_path(root, row["case_id"], "C0_clean")
            if clean_file.exists() and resume:
                clean = json.loads(clean_file.read_text(encoding="utf-8"))
            else:
                logits, score, predicted, tie = _forward(runtime, modules, inputs, positions["SAC"], ids)
                saved = [float(row["label_logits"][label]) for label in LABELS]
                logit_error = float(np.max(np.abs(np.asarray(logits) - np.asarray(saved))))
                score_error = abs(float(score["image_attribution_score"]) - float(row["image_attribution_score"]))
                saved_class, _ = deterministic_argmax(saved)
                if not np.isfinite(np.asarray(logits)).all() or not math.isfinite(score_error):
                    raise RuntimeError(f"Non-finite eager clean output for {row['case_id']}")
                hard_logit_gate = predicted == saved_class and logit_error <= CLEAN_LOGIT_TOLERANCE
                clean = {
                    "status": "completed", "condition": "C0_clean", "case_id": row["case_id"],
                    "test_side": row["test_side"], "answer": row["phase0_normalized_answer"],
                    "clean_class_logits": logits, "clean_image_attribution_score": float(score["image_attribution_score"]),
                    "clean_hard_class": predicted, "clean_argmax_tie": tie,
                    "clean_margin": class_margin(logits, predicted), "positions": located,
                    "parity": {"passed": hard_logit_gate and score_error <= CLEAN_SCORE_TOLERANCE,
                               "hard_logit_gate": hard_logit_gate,
                               "max_logit_error": logit_error, "score_error": score_error,
                               "score_warning": score_error > CLEAN_SCORE_TOLERANCE,
                               "baseline": "same_eager_clean_forward"},
                    "fingerprint": fingerprint,
                }
                atomic_json(clean_file, clean); new_forwards += 1
            if clean.get("fingerprint") != fingerprint:
                raise RuntimeError("Clean trial fingerprint mismatch")
            for window in windows:
                for condition in CONDITIONS:
                    destination = trial_path(root, row["case_id"], condition, window)
                    if destination.exists() and resume:
                        continue
                    edges, source = edge_for_condition(condition, positions)
                    with AttentionBlockContext(
                        modules.language_layers, layer_indices=range(window[0], window[1] + 1),
                        edges=edges, sequence_length=int(inputs.input_ids.shape[1]),
                        row_sum_tolerance=ROW_SUM_TOLERANCE,
                    ) as context:
                        logits, score, predicted, tie = _forward(runtime, modules, inputs, positions["SAC"], ids)
                    blocked_margin = class_margin(logits, int(clean["clean_hard_class"]))
                    trial = {
                        "status": "completed", "case_id": row["case_id"], "test_side": row["test_side"],
                        "answer": row["phase0_normalized_answer"], "condition": condition,
                        "window_start": window[0], "window_end": window[1],
                        "query_name": "SAC", "source_name": source,
                        "query_index": positions["SAC"], "source_index": positions[source],
                        "clean_class_logits": clean["clean_class_logits"], "blocked_class_logits": logits,
                        "clean_image_attribution_score": clean["clean_image_attribution_score"],
                        "blocked_image_attribution_score": float(score["image_attribution_score"]),
                        "clean_hard_class": clean["clean_hard_class"], "blocked_hard_class": predicted,
                        "clean_argmax_tie": clean["clean_argmax_tie"], "blocked_argmax_tie": tie,
                        "clean_margin": clean["clean_margin"], "blocked_margin": blocked_margin,
                        "token_changed": predicted != int(clean["clean_hard_class"]),
                        "token_change_rate": float(predicted != int(clean["clean_hard_class"])),
                        "logit_change_diff": float(clean["clean_margin"]) - blocked_margin,
                        "positions": located, "attention_diagnostics": context.diagnostics(),
                        "fingerprint": fingerprint,
                    }
                    if not all(math.isfinite(float(trial[k])) for k in ("token_change_rate", "logit_change_diff")):
                        raise RuntimeError("Non-finite attention metric")
                    atomic_json(destination, trial); new_forwards += 1
            atomic_json(root / "progress/run.json", {"status": "running", "completed_cases": ordinal,
                        "total_cases": len(cases), "new_gpu_forwards": new_forwards})
        trials = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((root / "artifacts/trials").glob("*.json"))]
        expected = len(cases) * (1 + len(windows) * len(CONDITIONS))
        if len(trials) != expected or any(r.get("fingerprint") != fingerprint for r in trials):
            raise RuntimeError(f"Incomplete or mixed trial grid: {len(trials)}/{expected}")
        atomic_jsonl(root / "artifacts/trials.jsonl", trials)
        result = {"status": "complete", "case_count": len(cases), "trial_count": len(trials),
                  "new_gpu_forwards": new_forwards, "elapsed_seconds": time.time() - started}
        atomic_json(root / "progress/run.json", result); return result
    finally:
        del runtime
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run five-class attention blocking (GPU)")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--steered-block", action="store_true")
    args = parser.parse_args(argv)
    if args.steered_block:
        from .steered_block import default_output as enhanced_output, run as enhanced_run
        result = enhanced_run(output_root=args.output_root or enhanced_output(args.smoke),
                              model_path=args.model_path, resume=args.resume)
    else:
        result = run(output_root=args.output_root or default_output(args.smoke),
                     model_path=args.model_path, resume=args.resume)
    print(json.dumps(result, ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
