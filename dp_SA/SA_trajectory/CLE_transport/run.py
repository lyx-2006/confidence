from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from dp_SA.attention_block.masking import AttentionBlockContext, AttentionEdges
from dp_SA.io_utils import atomic_json, atomic_jsonl, canonical_hash, load_jsonl
from dp_SA.positions import locate_phase1_positions
from dp_SA.soft_score import class_token_ids
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.model_adapter import model_input_device, resolve_language_modules
from dp_SA.SA_trajectory.LAT2PANL.run import _messages, load_explicit_fast_runtime
from dp_SA.attention_block.run import _forward

from .config import (
    CONDITIONS, EXPERIMENTS, HISTORICAL_CLEAN_PATH, LOGIT_PARITY_TOLERANCE,
    ROW_SUM_TOLERANCE, SOFT_PARITY_TOLERANCE, WINDOWS, WINDOW_NAMES,
    default_output, parse_windows,
)


def class_margin(logits: Sequence[float], selected: int) -> float:
    values = np.asarray(logits, dtype=np.float64)
    if values.shape != (9,) or not 0 <= int(selected) < 9:
        raise ValueError("Margin requires nine logits and a class in 0..8")
    return float(values[int(selected)] - np.delete(values, int(selected)).mean())


def add_class_list_end_plus_1(located: dict[str, Any], tokenizer: Any,
                              input_ids: torch.Tensor) -> dict[str, Any]:
    result = dict(located)
    index = int(located["P1_CLASS_LIST_END"]["processed_index"]) + 1
    ids = input_ids.detach().cpu().reshape(-1).tolist()
    if index >= len(ids):
        raise ValueError("P1_CLASS_LIST_END_PLUS_1 is outside the processed prompt")
    token_id = int(ids[index])
    result["P1_CLASS_LIST_END_PLUS_1"] = {
        "processed_index": index, "token_id": token_id,
        "token_text": tokenizer.decode([token_id], skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False),
        "definition": "P1_CLASS_LIST_END processed index + 1",
    }
    order = [int(result[name]["processed_index"]) for name in (
        "P1_LAT", "P1_PANL", "P1_PANL_PLUS_1", "P1_CLASS_LIST_END",
        "P1_CLASS_LIST_END_PLUS_1", "P1_SAC",
    )]
    if not all(left < right for left, right in zip(order, order[1:])):
        raise ValueError(f"Required causal token order failed: {order}")
    result["transport_causal_order_valid"] = True
    return result


def edge_for_condition(experiment: str, condition: str,
                       positions: dict[str, int]) -> tuple[AttentionEdges, str, str]:
    spec = EXPERIMENTS[experiment]
    if condition == CONDITIONS[0]:
        source_name = spec.main_source
    elif condition == CONDITIONS[1]:
        source_name = spec.control_source
    else:
        raise ValueError(f"Unknown condition: {condition}")
    query_name = spec.query
    return AttentionEdges(((positions[query_name], positions[source_name]),)), query_name, source_name


def _trial_path(root: Path, case_id: str, condition: str,
                window: tuple[int, int] | None = None) -> Path:
    suffix = condition if window is None else f"{condition}__L{window[0]}-{window[1]}"
    return root / "artifacts" / "trials" / f"{case_id}__{suffix}.json"


def _running_summary(root: Path, *, experiment: str, stage: str, rank: int,
                     world_size: int, completed: int, expected: int,
                     elapsed: float, last_case_id: str) -> None:
    if rank != 0:
        return
    rate = elapsed / max(completed, 1)
    remaining = max(0, expected - completed) * rate
    text = (
        f"# {experiment} 运行摘要\n\n"
        f"- 状态：running\n- stage：{stage}\n"
        f"- rank：{rank}/{world_size}\n- 本 rank 已完成：{completed}/{expected}\n"
        f"- last case：{last_case_id}\n- 已耗时：{elapsed:.2f} 秒\n"
        f"- 本 rank 预计剩余：{remaining:.2f} 秒\n"
    )
    (root / "progress").mkdir(parents=True, exist_ok=True)
    (root / "progress" / "summary.md").write_text(text, encoding="utf-8")


def _case_context(runtime: Any, case: dict[str, Any], historical: dict[str, Any]):
    messages = _messages(case)
    from dp_SA.prompts import SA_PREFILL
    rendered = render_continued_assistant(runtime.processor, messages, SA_PREFILL)
    inputs = prepare_multimodal_inputs(
        runtime.processor, messages, rendered, device=model_input_device(runtime)
    )
    tokenizer = runtime.processor.tokenizer
    located = locate_phase1_positions(
        tokenizer, rendered, inputs, str(case["phase0_raw_answer"])
    )
    located = add_class_list_end_plus_1(located, tokenizer, inputs.input_ids)
    old = historical[str(case["case_id"])]
    if canonical_hash(rendered) != old["rendered_prompt_hash"]:
        raise RuntimeError(f"Rendered prompt parity failed: {case['case_id']}")
    for name in ("P1_LAT", "P1_PANL", "P1_PANL_PLUS_1", "P1_CLASS_LIST_END", "P1_SAC"):
        actual = int(located[name]["processed_index"])
        expected = int(old["positions"][name]["processed_index"])
        if actual != expected:
            raise RuntimeError(f"Position parity failed for {case['case_id']} {name}: {actual} != {expected}")
    positions = {name: int(record["processed_index"]) for name, record in located.items()
                 if isinstance(record, dict) and "processed_index" in record}
    return inputs, located, positions, old


def _clean_trial(case: dict[str, Any], score: dict[str, Any], logits: list[float],
                 located: dict[str, Any], historical: dict[str, Any], fingerprint: str,
                 elapsed: float) -> dict[str, Any]:
    logit_error = max(abs(float(a) - float(b)) for a, b in zip(logits, historical["class_logits"]))
    soft_error = abs(float(score["soft_sa_image_score"]) - float(historical["soft_sa_image_score"]))
    hard_equal = int(score["argmax_hard_class"]) == int(historical["argmax_hard_class"])
    if logit_error > LOGIT_PARITY_TOLERANCE or soft_error > SOFT_PARITY_TOLERANCE or not hard_equal:
        raise RuntimeError(
            f"Clean parity failed for {case['case_id']}: logit={logit_error}, soft={soft_error}, hard={hard_equal}"
        )
    clean_class = int(score["argmax_hard_class"])
    margin = class_margin(logits, clean_class)
    return {
        "status": "completed", "experiment": None, "case_id": str(case["case_id"]),
        "family_id": str(case["family_id"]), "item_id": str(case["item_id"]),
        "answer": str(case["test_answer"]), "test_side": str(case["test_side"]),
        "condition": "C0_clean", "window_name": None, "window_start": None, "window_end": None,
        "query_name": None, "source_name": None, "query_index": None, "source_index": None,
        "clean_class_logits": logits, "blocked_class_logits": logits,
        "clean_soft_sa": float(score["soft_sa_image_score"]),
        "blocked_soft_sa": float(score["soft_sa_image_score"]),
        "clean_hard_sa_class": clean_class, "blocked_hard_sa_class": clean_class,
        "clean_margin": margin, "blocked_margin": margin,
        "delta_soft_sa": 0.0, "abs_delta_soft_sa": 0.0,
        "token_changed": False, "token_change_rate": 0.0, "logit_change_diff": 0.0,
        "positions": located, "attention_diagnostics": None,
        "parity": {"passed": True, "logit_max_abs_error": logit_error,
                   "soft_sa_abs_error": soft_error, "hard_class_equal": hard_equal},
        "elapsed_seconds": elapsed, "fingerprint": fingerprint,
    }


def _worker(*, root: Path, experiment: str, windows: tuple[tuple[int, int], ...],
            stage: str, rank: int, world_size: int, resume: bool) -> dict[str, Any]:
    config = json.loads((root / "run_config.json").read_text())
    fingerprint = str(config["fingerprint"])
    cases = sorted(load_jsonl(root / "artifacts" / "manifests" / "test_manifest.jsonl"),
                   key=lambda row: str(row["case_id"]))
    cases = [row for index, row in enumerate(cases) if index % world_size == rank]
    historical = {str(row["case_id"]): row for row in load_jsonl(HISTORICAL_CLEAN_PATH)}
    runtime = load_explicit_fast_runtime()
    modules = resolve_language_modules(runtime.model)
    if modules.num_hidden_layers != 28 or getattr(runtime.model.config, "_attn_implementation", None) != "eager":
        raise RuntimeError("Expected eager attention and 28 language layers")
    token_ids = class_token_ids(runtime.processor.tokenizer)
    new_forwards = 0
    started = time.time()
    expected_forwards = len(cases) * (1 if stage == "clean" else 2 * len(windows))
    try:
        for case in cases:
            inputs, located, positions, old = _case_context(runtime, case, historical)
            case_id = str(case["case_id"])
            clean_path = _trial_path(root, case_id, "C0_clean")
            if stage == "clean":
                if clean_path.exists():
                    if resume:
                        continue
                    raise FileExistsError(clean_path)
                before = time.perf_counter()
                logits, score = _forward(runtime.model, inputs, positions["P1_SAC"], token_ids)
                trial = _clean_trial(case, score, logits, located, old, fingerprint,
                                     time.perf_counter() - before)
                trial["experiment"] = experiment
                atomic_json(clean_path, trial)
                new_forwards += 1
            elif stage == "blocked":
                if not clean_path.is_file():
                    raise RuntimeError(f"Missing clean baseline: {case_id}")
                clean = json.loads(clean_path.read_text())
                if clean.get("fingerprint") != fingerprint or not clean.get("parity", {}).get("passed"):
                    raise RuntimeError(f"Invalid clean baseline: {case_id}")
                for window in windows:
                    for condition in CONDITIONS:
                        destination = _trial_path(root, case_id, condition, window)
                        if destination.exists():
                            if resume:
                                continue
                            raise FileExistsError(destination)
                        edges, query_name, source_name = edge_for_condition(experiment, condition, positions)
                        before = time.perf_counter()
                        with AttentionBlockContext(
                            modules.language_layers, layer_indices=range(window[0], window[1] + 1),
                            edges=edges, sequence_length=int(inputs.input_ids.shape[1]),
                            row_sum_tolerance=ROW_SUM_TOLERANCE,
                        ) as context:
                            logits, score = _forward(runtime.model, inputs, positions["P1_SAC"], token_ids)
                        elapsed = time.perf_counter() - before
                        diagnostics = context.diagnostics()
                        clean_class = int(clean["clean_hard_sa_class"])
                        blocked_margin = class_margin(logits, clean_class)
                        delta = float(score["soft_sa_image_score"]) - float(clean["clean_soft_sa"])
                        trial = {
                            **{key: clean[key] for key in ("case_id", "family_id", "item_id", "answer", "test_side")},
                            "status": "completed", "experiment": experiment, "condition": condition,
                            "window_name": WINDOW_NAMES[window], "window_start": window[0], "window_end": window[1],
                            "query_name": query_name, "source_name": source_name,
                            "query_index": positions[query_name], "source_index": positions[source_name],
                            "clean_class_logits": clean["clean_class_logits"], "blocked_class_logits": logits,
                            "clean_soft_sa": clean["clean_soft_sa"],
                            "blocked_soft_sa": float(score["soft_sa_image_score"]),
                            "clean_hard_sa_class": clean_class,
                            "blocked_hard_sa_class": int(score["argmax_hard_class"]),
                            "clean_margin": float(clean["clean_margin"]), "blocked_margin": blocked_margin,
                            "delta_soft_sa": delta, "abs_delta_soft_sa": abs(delta),
                            "token_changed": int(score["argmax_hard_class"]) != clean_class,
                            "token_change_rate": float(int(score["argmax_hard_class"]) != clean_class),
                            "logit_change_diff": float(clean["clean_margin"]) - blocked_margin,
                            "positions": located, "attention_diagnostics": diagnostics,
                            "parity": None, "elapsed_seconds": elapsed, "fingerprint": fingerprint,
                        }
                        atomic_json(destination, trial)
                        new_forwards += 1
            else:
                raise ValueError(stage)
            atomic_json(root / "progress" / f"{stage}_rank{rank}.json", {
                "status": "running", "rank": rank, "world_size": world_size,
                "new_gpu_forwards": new_forwards, "last_case_id": case_id,
                "elapsed_seconds": time.time() - started,
            })
            _running_summary(
                root, experiment=experiment, stage=stage, rank=rank,
                world_size=world_size, completed=new_forwards,
                expected=expected_forwards, elapsed=time.time() - started,
                last_case_id=case_id,
            )
            del inputs
        result = {"status": "complete", "stage": stage, "rank": rank,
                  "world_size": world_size, "new_gpu_forwards": new_forwards,
                  "elapsed_seconds": time.time() - started}
        atomic_json(root / "progress" / f"{stage}_rank{rank}.json", result)
        return result
    finally:
        del runtime
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_trials(root: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in sorted((root / "artifacts" / "trials").glob("*.json"))]


def _validate_clean_gate(root: Path) -> dict[str, Any]:
    cases = load_jsonl(root / "artifacts" / "manifests" / "test_manifest.jsonl")
    clean = [row for row in load_trials(root) if row["condition"] == "C0_clean"]
    passed = len(clean) == len(cases) and all(row.get("parity", {}).get("passed") for row in clean)
    result = {"status": "passed" if passed else "failed", "completed": len(clean), "expected": len(cases)}
    atomic_json(root / "progress" / "clean_gate.json", result)
    if not passed:
        raise RuntimeError(f"Global clean parity gate failed: {len(clean)}/{len(cases)}")
    return result


def _visible_gpu_tokens() -> list[str]:
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured:
        return [cell.strip() for cell in configured.split(",") if cell.strip()]
    return [str(index) for index in range(torch.cuda.device_count())]


def _spawn_stage(root: Path, *, experiment: str, windows: tuple[tuple[int, int], ...],
                 stage: str, num_gpus: int, resume: bool) -> list[dict[str, Any]]:
    tokens = _visible_gpu_tokens()
    if len(tokens) < num_gpus:
        raise RuntimeError(f"Requested {num_gpus} GPUs but only {len(tokens)} are visible")
    processes = []
    window_arg = ",".join(f"{a}-{b}" for a, b in windows)
    for rank in range(num_gpus):
        command = [sys.executable, "-m", "dp_SA.SA_trajectory.CLE_transport.run",
                   "--experiment", experiment, "--output-root", str(root), "--windows", window_arg,
                   "--stage", stage, "--worker-rank", str(rank), "--world-size", str(num_gpus)]
        if resume:
            command.append("--resume")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = tokens[rank]
        log_path = root / "progress" / f"{stage}_gpu{rank}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("a", encoding="utf-8")
        processes.append((subprocess.Popen(command, cwd=Path(__file__).resolve().parents[3],
                                           env=environment, stdout=handle, stderr=subprocess.STDOUT), handle, log_path))
    failures = []
    for process, handle, log_path in processes:
        code = process.wait(); handle.close()
        if code:
            failures.append(f"{log_path}: exit {code}")
    if failures:
        raise RuntimeError("GPU worker failure: " + "; ".join(failures))
    return [json.loads((root / "progress" / f"{stage}_rank{rank}.json").read_text())
            for rank in range(num_gpus)]


def run(*, experiment: str, output_root: Path | None = None,
        windows: tuple[tuple[int, int], ...] = WINDOWS, num_gpus: int = 1,
        resume: bool = False) -> dict[str, Any]:
    root = Path(output_root or default_output(experiment)).resolve()
    clean_results = _spawn_stage(root, experiment=experiment, windows=windows,
                                 stage="clean", num_gpus=num_gpus, resume=resume)
    gate = _validate_clean_gate(root)
    blocked_results = _spawn_stage(root, experiment=experiment, windows=windows,
                                   stage="blocked", num_gpus=num_gpus, resume=resume)
    trials = load_trials(root)
    atomic_jsonl(root / "artifacts" / "trials.jsonl", sorted(
        trials, key=lambda row: (row["case_id"], row["condition"], row.get("window_start") or -1)
    ))
    result = {
        "status": "complete", "experiment": experiment, "windows": [list(x) for x in windows],
        "clean_gate": gate,
        "new_gpu_forwards": sum(x["new_gpu_forwards"] for x in clean_results + blocked_results),
        "clean_new_gpu_forwards": sum(x["new_gpu_forwards"] for x in clean_results),
        "blocked_new_gpu_forwards": sum(x["new_gpu_forwards"] for x in blocked_results),
        "persisted_trials": len(trials),
        "elapsed_seconds": max((x["elapsed_seconds"] for x in clean_results), default=0)
                           + max((x["elapsed_seconds"] for x in blocked_results), default=0),
    }
    atomic_json(root / "progress" / "run.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS), required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--windows", type=parse_windows, default=WINDOWS)
    parser.add_argument("--num-gpus", type=int, choices=(1, 2), default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", choices=("clean", "blocked"))
    parser.add_argument("--worker-rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args(argv)
    if args.stage:
        result = _worker(root=Path(args.output_root).resolve(), experiment=args.experiment,
                         windows=args.windows, stage=args.stage, rank=args.worker_rank,
                         world_size=args.world_size, resume=args.resume)
    else:
        result = run(experiment=args.experiment, output_root=args.output_root,
                     windows=args.windows, num_gpus=args.num_gpus, resume=args.resume)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
