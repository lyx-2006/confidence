#!/usr/bin/env python3
"""Run answer-fixed Source Attribution hidden-state steering."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from confidence_test.joint_answer_source_extension import JointAnswerSourceGenerator
from confidence_test.runtime_imports import DEFAULT_INFERENCE_PATH
from confidence_test.source_attribution_variants import get_source_prompt_variant
from layer_metacognition.hidden_state_store import atomic_write_json
from layer_metacognition.model_adapter import load_qwen_inference, resolve_language_modules
from layer_metacognition.steering.decision_side_steering import assert_cuda_only_model

from . import (
    ALPHAS,
    DEFAULT_DATASET,
    DEFAULT_EXPERIMENT_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROBE_DIR,
    DIRECTIONS,
    LAYERS,
    METHODS,
    POSITIONS,
)
from .artifacts import (
    build_or_load_vectors,
    cohort_manifest,
    configuration_fingerprint,
    load_baseline_records,
    load_item_folds,
    read_json,
    select_evaluation_cases,
    select_extreme_sources,
    sha256_file,
)
from .runner import (
    build_runtime_cases,
    build_summary,
    completed_alpha_zero_by_case,
    execute_alpha_zero_baselines,
    execute_run,
    initialize_output,
    migrate_results_to_corrected_delta,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    parser.add_argument("--probe-dir", default=str(DEFAULT_PROBE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--image-root")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--inference-path", default=str(DEFAULT_INFERENCE_PATH))
    parser.add_argument("--positions", nargs="+", choices=POSITIONS, default=list(POSITIONS))
    parser.add_argument("--layers", nargs="+", type=int, default=list(LAYERS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--directions", nargs="+", choices=DIRECTIONS, default=list(DIRECTIONS))
    parser.add_argument("--alphas", nargs="+", type=float, default=list(ALPHAS))
    parser.add_argument("--test-fold", type=int, default=0)
    parser.add_argument("--eval-cases", type=int, default=100)
    parser.add_argument("--source-cases-per-side", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-source-tokens", type=int, default=4)
    parser.add_argument(
        "--alpha-zero-parity",
        action="store_true",
        help="Run a dedicated alpha=0 parity check in a separate output directory.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def _distinct(values: Sequence[Any], name: str) -> list[Any]:
    output = list(values)
    if not output or len(output) != len(set(output)):
        raise ValueError(f"{name} must contain distinct values")
    return output


def _validate_args(args: argparse.Namespace) -> tuple[list[str], list[int], list[str], list[str], list[float]]:
    positions = _distinct([str(value) for value in args.positions], "--positions")
    layers = _distinct([int(value) for value in args.layers], "--layers")
    methods = _distinct([str(value) for value in args.methods], "--methods")
    directions = _distinct([str(value) for value in args.directions], "--directions")
    alphas = _distinct([float(value) for value in args.alphas], "--alphas")
    if any(layer < 0 for layer in layers):
        raise ValueError("--layers must be non-negative")
    if any(not math.isfinite(alpha) or alpha < 0 for alpha in alphas):
        raise ValueError("--alphas must contain non-negative finite values")
    if args.alpha_zero_parity:
        if alphas != [0.0]:
            raise ValueError("--alpha-zero-parity requires exactly --alphas 0")
    elif any(alpha == 0 for alpha in alphas):
        raise ValueError("alpha 0 requires the explicit --alpha-zero-parity mode")
    if args.test_fold < 0:
        raise ValueError("--test-fold must be non-negative")
    if args.eval_cases < 2 or args.eval_cases % 2:
        raise ValueError("--eval-cases must be a positive even number")
    if args.source_cases_per_side < 1:
        raise ValueError("--source-cases-per-side must be positive")
    if args.max_source_tokens < 1:
        raise ValueError("--max-source-tokens must be positive")
    return positions, layers, methods, directions, alphas


def run(args: argparse.Namespace) -> dict[str, Any]:
    positions, layers, methods, directions, alphas = _validate_args(args)
    experiment_dir = Path(args.experiment_dir).resolve()
    probe_dir = Path(args.probe_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    dataset = Path(args.dataset).resolve()
    model_path = Path(args.model_path).resolve()
    inference_path = Path(args.inference_path).resolve()
    image_root = Path(args.image_root).resolve() if args.image_root else None
    required = [
        experiment_dir / "config.json",
        experiment_dir / "results.jsonl",
        experiment_dir / "hidden_states" / "index.json",
        probe_dir / "run_config.json",
        probe_dir / "split_assignments.json",
        probe_dir / "predictions" / "oof_predictions.jsonl",
        dataset,
        inference_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required SA steering inputs are missing: " + ", ".join(missing))
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory is missing: {model_path}")
    if image_root is not None and not image_root.is_dir():
        raise FileNotFoundError(f"Image root is missing: {image_root}")

    source_config = read_json(experiment_dir / "config.json")
    probe_config = read_json(probe_dir / "run_config.json")
    if source_config.get("source_prompt_variant") != "answer_basis_9":
        raise ValueError("This experiment requires source_prompt_variant=answer_basis_9")
    if probe_config.get("status") != "complete":
        raise ValueError("SA prediction probe run is not complete")
    if probe_config.get("split_mode") not in (None, "item"):
        raise ValueError("SA prediction probe does not use an item split")
    unavailable_probe_layers = sorted(set(layers) - set(map(int, probe_config.get("layers", []))))
    unavailable_probe_positions = sorted(
        set(positions) - set(map(str, probe_config.get("positions", [])))
    )
    if unavailable_probe_layers or unavailable_probe_positions:
        raise ValueError(
            "Probe artifacts do not cover requested grid: "
            f"layers={unavailable_probe_layers}, positions={unavailable_probe_positions}"
        )

    records, provenance = load_baseline_records(
        experiment_dir,
        layers=layers,
        positions=positions,
    )
    item_to_fold = load_item_folds(probe_dir, records)
    folds = sorted(set(item_to_fold.values()))
    if args.test_fold not in folds:
        raise ValueError(f"--test-fold {args.test_fold} is unavailable; folds={folds}")
    evaluation = select_evaluation_cases(
        records,
        item_to_fold,
        test_fold=args.test_fold,
        eval_cases=args.eval_cases,
        seed=args.seed,
    )
    sources = select_extreme_sources(
        records,
        item_to_fold,
        test_fold=args.test_fold,
        cases_per_side=args.source_cases_per_side,
    )
    cohorts = cohort_manifest(
        evaluation,
        sources,
        item_to_fold,
        test_fold=args.test_fold,
    )
    expected_count = (
        len(evaluation)
        * len(positions)
        * len(layers)
        * len(methods)
        * len(directions)
        * len(alphas)
    )
    immutable = {
        "format_version": 1,
        "experiment": (
            "answer_fixed_sa_alpha_zero_parity"
            if args.alpha_zero_parity
            else "answer_fixed_sa_hidden_state_steering"
        ),
        "protocol": "reuse_answer_then_teacher_force_answer_and_sa_prefix",
        "answer_fixed": True,
        "baseline_source": "existing_generated.source_attribution",
        "alpha_zero_parity": bool(args.alpha_zero_parity),
        "experiment_dir": str(experiment_dir),
        "probe_dir": str(probe_dir),
        "output_dir": str(output_dir),
        "dataset": str(dataset),
        "image_root": str(image_root) if image_root else None,
        "model_path": str(model_path),
        "inference_path": str(inference_path),
        "positions": positions,
        "layers": layers,
        "methods": methods,
        "directions": directions,
        "alphas": alphas,
        "test_fold": int(args.test_fold),
        "eval_cases": int(args.eval_cases),
        "source_cases_per_side": int(args.source_cases_per_side),
        "seed": int(args.seed),
        "max_source_tokens": int(args.max_source_tokens),
        "source_prompt_variant": "answer_basis_9",
        "source_classes": list(source_config["source_attribution_classes"]),
        "normalization": "0.03 * training_mean_hidden_l2",
        "injection_site": "decoder_block_output",
        "use_cache": True,
        "cuda_only": True,
        "per_case_failure_policy": "record_and_continue",
        "systemic_invariant_failure_policy": "abort",
        "expected_intervention_count": expected_count,
        "cohorts": cohorts,
        "provenance": {
            **provenance,
            "probe_run_config_sha256": sha256_file(probe_dir / "run_config.json"),
            "split_assignments_sha256": sha256_file(probe_dir / "split_assignments.json"),
            "oof_predictions_sha256": sha256_file(
                probe_dir / "predictions" / "oof_predictions.jsonl"
            ),
        },
    }
    fingerprint = configuration_fingerprint(immutable)
    configuration = {**immutable, "config_fingerprint": fingerprint}
    existing, existing_keys, config_path, results_path, progress_path, summary_path = (
        initialize_output(output_dir, configuration, resume=args.resume)
    )

    source_fingerprint = configuration_fingerprint(
        {
            "source_results_sha256": provenance["source_results_sha256"],
            "split_assignments_sha256": immutable["provenance"]["split_assignments_sha256"],
            "layers": layers,
            "positions": positions,
            "test_fold": args.test_fold,
            "source_case_ids": {
                name: [record["case_id"] for record in group]
                for name, group in sources.items()
            },
        }
    )
    repository = build_or_load_vectors(
        vector_dir=output_dir / "steering_vectors",
        records=records,
        item_to_fold=item_to_fold,
        sources=sources,
        experiment_dir=experiment_dir,
        probe_dir=probe_dir,
        layers=layers,
        positions=positions,
        test_fold=args.test_fold,
        source_fingerprint=source_fingerprint,
    )
    source_variant = get_source_prompt_variant("answer_basis_9")
    runtime_cases, dataset_metadata = build_runtime_cases(
        evaluation,
        dataset=dataset,
        output_dir=output_dir,
        source_variant=source_variant,
        image_root=image_root,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU SA steering is forbidden")
    inference = load_qwen_inference(str(model_path), inference_path)
    modules = resolve_language_modules(inference.model)
    assert_cuda_only_model(inference.model, modules)
    if max(layers) >= modules.num_hidden_layers:
        raise ValueError(
            f"Requested layer exceeds model range [0, {modules.num_hidden_layers - 1}]"
        )
    sample = repository.get(methods[0], positions[0], layers[0])
    if sample.vector.shape != (modules.hidden_size,):
        raise ValueError(
            f"Steering vector hidden size {sample.vector.shape} differs from model "
            f"hidden size {modules.hidden_size}"
        )
    runtime = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "device_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
        "dtype": inference.dtype_name,
        "device_map": getattr(inference.model, "hf_device_map", None),
        "num_hidden_layers": modules.num_hidden_layers,
        "hidden_size": modules.hidden_size,
    }
    running_config = {
        **configuration,
        "dataset_metadata": dataset_metadata,
        "vector_index": str(repository.index_path),
        "vector_count": repository.index["vector_count"],
        "model_runtime": runtime,
        "status": "running",
        "started_at_unix": time.time(),
        "alpha_zero_baseline": {
            "enabled": not bool(args.alpha_zero_parity),
            "storage": "alpha0_baselines.jsonl",
            "one_per_evaluation_case": True,
            "corrected_delta_definition": "SA_after - case_alpha0_SA_after",
        },
    }
    atomic_write_json(config_path, running_config)
    joint_generator = JointAnswerSourceGenerator(inference)
    try:
        alpha_zero_baselines: list[dict[str, Any]] = []
        if not args.alpha_zero_parity:
            baseline_position = "sac" if "sac" in positions else positions[0]
            baseline_layer = 18 if 18 in layers else layers[0]
            alpha_zero_baselines = execute_alpha_zero_baselines(
                cases=runtime_cases,
                repository=repository,
                joint_generator=joint_generator,
                modules=modules,
                source_variant=source_variant,
                position=baseline_position,
                layer=baseline_layer,
                method=methods[0],
                max_source_tokens=args.max_source_tokens,
                results_path=output_dir / "alpha0_baselines.jsonl",
                progress_path=output_dir / "alpha0_progress.json",
            )
            completed_alpha_zero = completed_alpha_zero_by_case(alpha_zero_baselines)
            if len(completed_alpha_zero) != len(runtime_cases):
                raise RuntimeError("Alpha-zero baseline backfill did not cover all cases")
            existing = migrate_results_to_corrected_delta(
                existing,
                alpha_zero_baselines,
                results_path=results_path,
            )
        final_records = execute_run(
            cases=runtime_cases,
            repository=repository,
            joint_generator=joint_generator,
            modules=modules,
            source_variant=source_variant,
            positions=positions,
            layers=layers,
            methods=methods,
            directions=directions,
            alphas=alphas,
            max_source_tokens=args.max_source_tokens,
            existing=existing,
            existing_keys=existing_keys,
            results_path=results_path,
            progress_path=progress_path,
            summary_path=summary_path,
            alpha_zero_baselines=alpha_zero_baselines,
        )
    except Exception:
        atomic_write_json(config_path, {**running_config, "status": "failed"})
        raise
    final_summary = build_summary(
        final_records,
        repository,
        expected_count=expected_count,
        alpha_zero_baselines=alpha_zero_baselines,
    )
    atomic_write_json(summary_path, final_summary)
    final_status = (
        "complete_with_failures" if final_summary["failed_count"] else "complete"
    )
    atomic_write_json(config_path, {**running_config, "status": final_status})
    return {
        "status": final_status,
        "output_dir": str(output_dir),
        "record_count": len(final_records),
        "completed_count": final_summary["completed_count"],
        "failed_count": final_summary["failed_count"],
        "alpha_zero_baseline_count": final_summary[
            "alpha_zero_baseline_completed_case_count"
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        config_path = Path(args.output_dir).resolve() / "run_config.json"
        if config_path.is_file():
            try:
                failed = read_json(config_path)
                failed.update(
                    {
                        "status": "failed",
                        "fatal_error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                        "failed_at_unix": time.time(),
                    }
                )
                atomic_write_json(config_path, failed)
            except Exception:
                pass
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
