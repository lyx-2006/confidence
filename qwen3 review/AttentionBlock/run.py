from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from dp_SA.attention_block.masking import AttentionBlockContext, AttentionEdges
from dp_SA.attention_block.run import _forward
from dp_SA.io_utils import atomic_json, atomic_jsonl, load_jsonl
from dp_SA.positions import locate_phase1_positions
from dp_SA.prompts import SA_PREFILL
from dp_SA.soft_score import class_token_ids
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.model_adapter import model_input_device, resolve_language_modules
from Steering.capture import _messages
from Steering.config import POSITION_KEYS
from Steering.runtime import load_qwen3_inference

from .config import (
    CLEAN_LOGIT_TOLERANCE, CLEAN_SOFT_SA_TOLERANCE, CONDITIONS, EXPERIMENTS,
    MODEL_PATH, ROW_SUM_TOLERANCE,
)
from .contracts import add_cle_plus_1, load_trials, trial_path


def class_margin(logits: Sequence[float], selected: int) -> float:
    values = np.asarray(logits, dtype=np.float64)
    if values.shape != (9,) or not 0 <= int(selected) < 9:
        raise ValueError("Margin requires nine logits and a class in 0..8")
    return float(values[int(selected)] - np.delete(values, int(selected)).mean())


def edge_for_condition(experiment: str, condition: str,
                       positions: dict[str, int]) -> tuple[AttentionEdges, str, str]:
    spec = EXPERIMENTS[experiment]
    source = spec.main_source if condition == CONDITIONS[0] else spec.control_source
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    return AttentionEdges(((positions[spec.query], positions[source]),)), spec.query, source


def _case_context(runtime: Any, row: dict[str, Any]):
    messages = _messages(row["phase1_prompt"], row["image_path"], SA_PREFILL)
    rendered = render_continued_assistant(runtime.processor, messages, SA_PREFILL)
    inputs = prepare_multimodal_inputs(
        runtime.processor, messages, rendered, device=model_input_device(runtime)
    )
    tokenizer = runtime.processor.tokenizer
    located = add_cle_plus_1(
        locate_phase1_positions(tokenizer, rendered, inputs, row["phase0_raw_answer"]),
        tokenizer, inputs.input_ids,
    )
    for name, internal in POSITION_KEYS.items():
        old, new = row["positions"][name], located[internal]
        if (int(old["processed_index"]), int(old["token_id"])) != (
            int(new["processed_index"]), int(new["token_id"]),
        ):
            raise RuntimeError(f"Position parity failed for {row['case_id']} at {name}")
    positions = {name: int(value["processed_index"]) for name, value in located.items()
                 if isinstance(value, dict) and "processed_index" in value}
    return inputs, located, positions


def _clean_trial(row: dict[str, Any], logits: list[float], score: dict[str, Any],
                 located: dict[str, Any], fingerprint: str, elapsed: float) -> dict[str, Any]:
    logit_error = max(abs(float(a) - float(b)) for a, b in zip(logits, row["class_logits"]))
    soft_error = abs(float(score["soft_sa_image_score"]) - float(row["soft_sa_image_score"]))
    hard_equal = int(score["argmax_hard_class"]) == int(row["argmax_hard_class"])
    passed = hard_equal and logit_error <= CLEAN_LOGIT_TOLERANCE and soft_error <= CLEAN_SOFT_SA_TOLERANCE
    if not passed:
        raise RuntimeError(
            f"Eager clean gate failed for {row['case_id']}: logit={logit_error}, "
            f"soft={soft_error}, hard_equal={hard_equal}"
        )
    clean_class = int(score["argmax_hard_class"])
    margin = class_margin(logits, clean_class)
    return {
        "status": "completed", "experiment": None, "case_id": row["case_id"],
        "family_id": row["family_id"], "item_id": row["item_id"],
        "answer": row["test_answer"], "test_side": row["test_side"],
        "condition": "C0_clean", "window_start": None, "window_end": None,
        "query_name": None, "source_name": None, "query_index": None, "source_index": None,
        "clean_class_logits": logits, "blocked_class_logits": logits,
        "clean_soft_sa": float(score["soft_sa_image_score"]),
        "blocked_soft_sa": float(score["soft_sa_image_score"]),
        "clean_hard_sa_class": clean_class, "blocked_hard_sa_class": clean_class,
        "clean_margin": margin, "blocked_margin": margin,
        "delta_soft_sa": 0.0, "token_changed": False,
        "token_change_rate": 0.0, "logit_change_diff": 0.0,
        "positions": located, "attention_diagnostics": None,
        "parity": {"passed": True, "logit_max_abs_error": logit_error,
                   "soft_sa_abs_error": soft_error, "hard_class_equal": hard_equal},
        "elapsed_seconds": elapsed, "fingerprint": fingerprint,
    }


def run(*, experiment: str, output_root: Path, resume: bool = False) -> dict[str, Any]:
    root = output_root.resolve()
    config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    fingerprint = config["fingerprint"]
    spec = EXPERIMENTS[experiment]
    windows = tuple(tuple(map(int, value)) for value in config["spec"]["windows"])
    cases = load_jsonl(root / "artifacts" / "manifests" / "test_manifest.jsonl")
    pid_path = root / "active.pid"
    if pid_path.exists():
        pid = int(pid_path.read_text(encoding="utf-8"))
        try:
            os.kill(pid, 0)
            raise RuntimeError(f"AttentionBlock already active: PID {pid}")
        except ProcessLookupError:
            pid_path.unlink()
    pid_path.write_text(str(os.getpid()), encoding="utf-8")
    runtime = None
    started = time.time()
    new_forwards = 0
    try:
        runtime = load_qwen3_inference(MODEL_PATH, attn_implementation="eager")
        modules = resolve_language_modules(runtime.model)
        if modules.num_hidden_layers != 36 or runtime.model.config._attn_implementation != "eager":
            raise RuntimeError("Expected Qwen3 eager attention with 36 language layers")
        token_ids = class_token_ids(runtime.processor.tokenizer)
        for ordinal, row in enumerate(cases, 1):
            inputs, located, positions = _case_context(runtime, row)
            clean_path = trial_path(root, row["case_id"], "C0_clean")
            if clean_path.exists():
                clean = json.loads(clean_path.read_text(encoding="utf-8"))
                if not resume and clean.get("fingerprint") != fingerprint:
                    raise ValueError(f"Invalid existing clean trial: {clean_path}")
            else:
                eager = row.get("eager_clean")
                if eager and eager.get("passed"):
                    score = {"soft_sa_image_score": eager["soft_sa_image_score"],
                             "argmax_hard_class": eager["argmax_hard_class"]}
                    clean = _clean_trial(row, eager["class_logits"], score, located, fingerprint, 0.0)
                else:
                    before = time.perf_counter()
                    logits, score = _forward(runtime.model, inputs, positions["P1_SAC"], token_ids)
                    clean = _clean_trial(row, logits, score, located, fingerprint, time.perf_counter() - before)
                clean["experiment"] = experiment
                atomic_json(clean_path, clean)
                new_forwards += 1
            if not clean.get("parity", {}).get("passed") or clean.get("fingerprint") != fingerprint:
                raise RuntimeError(f"Invalid clean baseline: {row['case_id']}")
            for window in windows:
                for condition in CONDITIONS:
                    destination = trial_path(root, row["case_id"], condition, window)
                    if destination.exists():
                        if resume:
                            continue
                        raise FileExistsError(destination)
                    edges, query_name, source_name = edge_for_condition(experiment, condition, positions)
                    before = time.perf_counter()
                    with AttentionBlockContext(
                        modules.language_layers,
                        layer_indices=range(window[0], window[1] + 1),
                        edges=edges,
                        sequence_length=int(inputs.input_ids.shape[1]),
                        row_sum_tolerance=ROW_SUM_TOLERANCE,
                    ) as context:
                        logits, score = _forward(runtime.model, inputs, positions["P1_SAC"], token_ids)
                    diagnostics = context.diagnostics()
                    blocked_class = int(score["argmax_hard_class"])
                    clean_class = int(clean["clean_hard_sa_class"])
                    blocked_margin = class_margin(logits, clean_class)
                    delta = float(score["soft_sa_image_score"]) - float(clean["clean_soft_sa"])
                    trial = {
                        **{key: clean[key] for key in ("case_id", "family_id", "item_id", "answer", "test_side")},
                        "status": "completed", "experiment": experiment, "condition": condition,
                        "window_start": window[0], "window_end": window[1],
                        "query_name": query_name, "source_name": source_name,
                        "query_index": positions[query_name], "source_index": positions[source_name],
                        "clean_class_logits": clean["clean_class_logits"], "blocked_class_logits": logits,
                        "clean_soft_sa": clean["clean_soft_sa"],
                        "blocked_soft_sa": float(score["soft_sa_image_score"]),
                        "clean_hard_sa_class": clean_class, "blocked_hard_sa_class": blocked_class,
                        "clean_margin": clean["clean_margin"], "blocked_margin": blocked_margin,
                        "delta_soft_sa": delta,
                        "token_changed": blocked_class != clean_class,
                        "token_change_rate": float(blocked_class != clean_class),
                        "logit_change_diff": float(clean["clean_margin"]) - blocked_margin,
                        "positions": located, "attention_diagnostics": diagnostics,
                        "parity": None, "elapsed_seconds": time.perf_counter() - before,
                        "fingerprint": fingerprint,
                    }
                    if not all(math.isfinite(float(trial[key])) for key in (
                        "delta_soft_sa", "token_change_rate", "logit_change_diff"
                    )):
                        raise RuntimeError("Non-finite attention-block metric")
                    atomic_json(destination, trial)
                    new_forwards += 1
            atomic_json(root / "progress" / "run.json", {
                "status": "running", "completed_cases": ordinal, "total_cases": len(cases),
                "new_gpu_forwards": new_forwards, "last_case_id": row["case_id"],
                "elapsed_seconds": time.time() - started,
            })
            del inputs
        trials = load_trials(root)
        expected = len(cases) * (1 + len(windows) * len(CONDITIONS))
        if len(trials) != expected:
            raise RuntimeError(f"Trial grid incomplete: {len(trials)}/{expected}")
        atomic_jsonl(root / "artifacts" / "trials.jsonl", sorted(
            trials, key=lambda row: (str(row["case_id"]), str(row["condition"]),
                                     int(row.get("window_start") or -1))
        ))
        result = {"status": "complete", "experiment": experiment, "case_count": len(cases),
                  "trial_count": len(trials), "expected_trials": expected,
                  "new_gpu_forwards": new_forwards, "elapsed_seconds": time.time() - started}
        atomic_json(root / "progress" / "run.json", result)
        return result
    finally:
        if runtime is not None:
            del runtime
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if pid_path.exists() and pid_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
            pid_path.unlink()
