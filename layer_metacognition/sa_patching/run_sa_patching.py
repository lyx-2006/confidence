#!/usr/bin/env python3
"""Run Source Attribution teacher-forced embedding corruption and activation patching."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from confidence_test.joint_answer_source_extension import JointAnswerSourceGenerator
from confidence_test.runtime_imports import DEFAULT_INFERENCE_PATH
from confidence_test.source_attribution_variants import get_source_prompt_variant
from layer_metacognition.hidden_state_store import (
    append_jsonl,
    atomic_write_json,
    load_jsonl,
)
from layer_metacognition.model_adapter import (
    load_qwen_inference,
    resolve_language_modules,
)
from layer_metacognition.sa_steering.artifacts import (
    configuration_fingerprint,
    load_baseline_records,
    load_item_folds,
    read_json,
    sha256_file,
)
from layer_metacognition.sa_steering.runner import build_runtime_cases
from layer_metacognition.steering.decision_side_steering import assert_cuda_only_model
from layer_metacognition.v3_v4_source_runner import reconstruction_tolerance

from . import (
    CORRUPTIONS,
    DEFAULT_DATASET,
    DEFAULT_EXPERIMENT_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_DIR,
    EVALUATION_CONDITIONS,
    FORMAT_VERSION,
    POSITIONS,
    POSITION_LAYERS,
)
from .artifacts import build_or_load_embedding_artifacts, select_cohorts
from .runner import SAPatchingRunner, build_summary, intervention_key, position_layer_grid
from .sa_patching_hook import PatchingInvariantError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--image-root")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--inference-path", default=str(DEFAULT_INFERENCE_PATH))
    parser.add_argument("--positions", nargs="+", choices=POSITIONS, default=list(POSITIONS))
    parser.add_argument(
        "--corruptions",
        nargs="+",
        choices=CORRUPTIONS,
        default=list(CORRUPTIONS),
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=sorted({layer for values in POSITION_LAYERS.values() for layer in values}),
    )
    parser.add_argument("--eval-cases", type=int, default=100)
    parser.add_argument("--test-fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-source-tokens", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser


def _distinct(values: Sequence[Any], name: str) -> list[Any]:
    output = list(values)
    if not output or len(output) != len(set(output)):
        raise ValueError(f"{name} must contain distinct values")
    return output


def _validate_args(args: argparse.Namespace) -> tuple[list[str], list[str], list[int]]:
    positions = _distinct([str(value) for value in args.positions], "--positions")
    corruptions = _distinct(
        [str(value) for value in args.corruptions], "--corruptions"
    )
    layers = _distinct([int(value) for value in args.layers], "--layers")
    if any(layer < 0 for layer in layers):
        raise ValueError("--layers must be non-negative")
    if args.eval_cases < 2 or args.eval_cases % 2:
        raise ValueError("--eval-cases must be a positive even number")
    if args.test_fold < 0:
        raise ValueError("--test-fold must be non-negative")
    if args.max_source_tokens < 1:
        raise ValueError("--max-source-tokens must be positive")
    position_layer_grid(positions, layers, POSITION_LAYERS)
    return positions, corruptions, layers


def _progress(
    records: Sequence[dict[str, Any]],
    *,
    expected_count: int,
    status: str,
) -> dict[str, Any]:
    completed = sum(record.get("status") == "completed" for record in records)
    failed = sum(record.get("status") == "failed" for record in records)
    return {
        "status": status,
        "expected_count": int(expected_count),
        "record_count": len(records),
        "completed_count": completed,
        "failed_count": failed,
        "remaining_count": max(0, int(expected_count) - len(records)),
        "last_intervention_key": (
            records[-1].get("intervention_key") if records else None
        ),
        "updated_at_unix": time.time(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    positions, corruptions, layers = _validate_args(args)
    grid = position_layer_grid(positions, layers, POSITION_LAYERS)
    experiment_dir = Path(args.experiment_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    dataset = Path(args.dataset).resolve()
    model_path = Path(args.model_path).resolve()
    inference_path = Path(args.inference_path).resolve()
    image_root = Path(args.image_root).resolve() if args.image_root else None
    probe_dir = experiment_dir / "stage_sa_prediction_probe"
    required = (
        experiment_dir / "config.json",
        experiment_dir / "results.jsonl",
        experiment_dir / "hidden_states" / "index.json",
        probe_dir / "split_assignments.json",
        dataset,
        inference_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required SA patching inputs are missing: " + ", ".join(missing))
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory is missing: {model_path}")
    if image_root is not None and not image_root.is_dir():
        raise FileNotFoundError(f"Image root is missing: {image_root}")
    source_config = read_json(experiment_dir / "config.json")
    if source_config.get("source_prompt_variant") != "answer_basis_9":
        raise ValueError("SA patching requires source_prompt_variant=answer_basis_9")
    if source_config.get("versions") != ["v4"] or source_config.get("attribution_mode") != "joint":
        raise ValueError("SA patching requires the V4 joint source experiment")

    records, provenance = load_baseline_records(
        experiment_dir,
        layers=sorted({layer for _position, layer in grid}),
        positions=positions,
    )
    item_to_fold = load_item_folds(probe_dir, records)
    folds = sorted(set(item_to_fold.values()))
    if args.test_fold not in folds:
        raise ValueError(f"--test-fold {args.test_fold} is unavailable; folds={folds}")
    evaluation, sources, cohorts = select_cohorts(
        records,
        item_to_fold,
        test_fold=args.test_fold,
        eval_cases=args.eval_cases,
        seed=args.seed,
        evaluation_conditions=EVALUATION_CONDITIONS,
    )
    source_variant = get_source_prompt_variant("answer_basis_9")
    runtime_cases, dataset_metadata = build_runtime_cases(
        evaluation,
        dataset=dataset,
        output_dir=output_dir,
        source_variant=source_variant,
        image_root=image_root,
    )
    expected_keys = {
        intervention_key(record["case_id"], corruption, position, layer)
        for record in evaluation
        for corruption in corruptions
        for position, layer in grid
    }
    expected_count = len(expected_keys)
    immutable = {
        "format_version": FORMAT_VERSION,
        "experiment": "source_attribution_teacher_forced_activation_patching",
        "protocol": "fresh_joint_answer_then_teacher_forced_sa",
        "answer_fixed": True,
        "experiment_dir": str(experiment_dir),
        "output_dir": str(output_dir),
        "dataset": str(dataset),
        "image_root": str(image_root) if image_root else None,
        "model_path": str(model_path),
        "inference_path": str(inference_path),
        "positions": positions,
        "layers": layers,
        "position_layer_grid": [
            {"position": position, "layer": layer} for position, layer in grid
        ],
        "corruptions": corruptions,
        "eval_cases": int(args.eval_cases),
        "test_fold": int(args.test_fold),
        "evaluation_conditions": list(EVALUATION_CONDITIONS),
        "seed": int(args.seed),
        "max_source_tokens": int(args.max_source_tokens),
        "source_prompt_variant": "answer_basis_9",
        "source_classes": list(source_variant.classes),
        "embedding_corruption_site": "language_model_inputs_embeds_after_vision_replacement",
        "activation_patch_site": "decoder_block_output_post_mlp_residual",
        "text_mean_policy": "ragged_position_mean",
        "use_cache": True,
        "cuda_only": True,
        "expected_result_count": expected_count,
        "cohorts": cohorts,
        "dataset_metadata": dataset_metadata,
        "provenance": {
            **provenance,
            "split_assignments_sha256": sha256_file(
                probe_dir / "split_assignments.json"
            ),
        },
    }
    fingerprint = configuration_fingerprint(immutable)
    configuration = {**immutable, "config_fingerprint": fingerprint}
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    results_path = output_dir / "results.jsonl"
    progress_path = output_dir / "progress.json"
    summary_path = output_dir / "summary.json"
    existing = load_jsonl(results_path, repair_trailing=args.resume)
    if existing and not args.resume:
        raise ValueError("Output contains results; pass --resume or choose a new directory")
    existing_keys: set[str] = set()
    for record in existing:
        key = str(record.get("intervention_key", ""))
        if not key or key in existing_keys:
            raise ValueError(f"Duplicate or empty intervention key in results: {key!r}")
        if key not in expected_keys:
            raise ValueError(f"Existing result is outside the requested grid: {key}")
        existing_keys.add(key)
    if config_path.is_file():
        saved = read_json(config_path)
        if saved.get("config_fingerprint") != fingerprint:
            raise ValueError("Resume configuration differs from existing run_config.json")
    atomic_write_json(
        config_path,
        {
            **configuration,
            "status": "initializing",
            "created_at_unix": (
                read_json(config_path).get("created_at_unix", time.time())
                if config_path.is_file()
                else time.time()
            ),
        },
    )
    atomic_write_json(
        progress_path,
        _progress(existing, expected_count=expected_count, status="initializing"),
    )
    if expected_keys.issubset(existing_keys):
        summary = build_summary(existing, expected_count=expected_count)
        atomic_write_json(summary_path, summary)
        final_status = "complete_with_failures" if summary["failed_count"] else "complete"
        atomic_write_json(config_path, {**read_json(config_path), "status": final_status})
        atomic_write_json(
            progress_path,
            _progress(existing, expected_count=expected_count, status=final_status),
        )
        return {
            "status": final_status,
            "output_dir": str(output_dir),
            "record_count": len(existing),
            "completed_count": summary["completed_count"],
            "failed_count": summary["failed_count"],
        }

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU SA patching is forbidden")
    inference = load_qwen_inference(str(model_path), inference_path)
    modules = resolve_language_modules(inference.model)
    assert_cuda_only_model(inference.model, modules)
    if max(layer for _position, layer in grid) >= modules.num_hidden_layers:
        raise ValueError("Requested patch layer exceeds the model")
    joint = JointAnswerSourceGenerator(inference)
    embedding_artifacts = build_or_load_embedding_artifacts(
        output_dir=output_dir,
        sources=sources,
        metadata=cohorts,
        dataset=dataset,
        image_root=image_root,
        source_variant=source_variant,
        inference=inference,
        joint_generator=joint,
        hidden_size=modules.hidden_size,
    )
    runtime_config = {
        **read_json(config_path),
        "status": "running",
        "started_at_unix": time.time(),
        "model_runtime": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "device_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "dtype": inference.dtype_name,
            "device_map": getattr(inference.model, "hf_device_map", None),
            "num_hidden_layers": modules.num_hidden_layers,
            "hidden_size": modules.hidden_size,
        },
        "embedding_artifacts": embedding_artifacts.metadata,
        "clean_parity_tolerance": reconstruction_tolerance(inference.dtype_name),
        "no_patch_validation": {},
    }
    atomic_write_json(config_path, runtime_config)
    runner = SAPatchingRunner(
        joint_generator=joint,
        modules=modules,
        source_variant=source_variant,
        artifacts=embedding_artifacts,
        grid=grid,
        corruptions=corruptions,
        max_source_tokens=args.max_source_tokens,
        parity_tolerance=reconstruction_tolerance(inference.dtype_name),
    )

    def commit(record: dict[str, Any]) -> None:
        key = str(record["intervention_key"])
        if key in existing_keys:
            raise RuntimeError(f"Attempted to append duplicate intervention: {key}")
        append_jsonl(results_path, record, fsync=True)
        existing.append(record)
        existing_keys.add(key)
        atomic_write_json(
            progress_path,
            _progress(existing, expected_count=expected_count, status="running"),
        )

    try:
        for runtime_case in runtime_cases:
            pending = expected_keys.difference(existing_keys)
            case_pending = {
                key
                for key in pending
                if key.startswith(f"{runtime_case.record['case_id']}|")
            }
            if not case_pending:
                continue
            runner.process_case(
                runtime_case,
                pending_keys=case_pending,
                commit=commit,
                validate_no_patch=not all(
                    corruption in runner.no_patch_validation
                    for corruption in corruptions
                ),
            )
            atomic_write_json(summary_path, build_summary(existing, expected_count=expected_count))
            atomic_write_json(
                config_path,
                {
                    **runtime_config,
                    "no_patch_validation": runner.no_patch_validation,
                },
            )
    except Exception as exc:
        partial = build_summary(existing, expected_count=expected_count)
        atomic_write_json(summary_path, partial)
        atomic_write_json(
            config_path,
            {
                **runtime_config,
                "status": "failed",
                "no_patch_validation": runner.no_patch_validation,
                "fatal_error": {"type": type(exc).__name__, "message": str(exc)},
                "failed_at_unix": time.time(),
            },
        )
        atomic_write_json(
            progress_path,
            _progress(existing, expected_count=expected_count, status="failed"),
        )
        raise

    summary = build_summary(existing, expected_count=expected_count)
    atomic_write_json(summary_path, summary)
    final_status = "complete_with_failures" if summary["failed_count"] else "complete"
    atomic_write_json(
        config_path,
        {
            **runtime_config,
            "status": final_status,
            "no_patch_validation": runner.no_patch_validation,
            "finished_at_unix": time.time(),
        },
    )
    atomic_write_json(
        progress_path,
        _progress(existing, expected_count=expected_count, status=final_status),
    )
    return {
        "status": final_status,
        "output_dir": str(output_dir),
        "record_count": len(existing),
        "completed_count": summary["completed_count"],
        "failed_count": summary["failed_count"],
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
