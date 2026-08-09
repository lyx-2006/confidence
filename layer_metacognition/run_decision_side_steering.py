#!/usr/bin/env python3
"""Run OOF Decision-Side steering on V4 joint answer/source attribution."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from confidence_test.dataset_utils import CONDITIONS  # noqa: E402
from confidence_test.joint_answer_source_extension import (  # noqa: E402
    JointAnswerSourceGenerator,
)
from confidence_test.runtime_imports import DEFAULT_INFERENCE_PATH  # noqa: E402
from confidence_test.source_attribution_analyzer import (  # noqa: E402
    SourceAttributionAnalyzer,
)
from confidence_test.source_attribution_variants import (  # noqa: E402
    get_source_prompt_variant,
)
from layer_metacognition.hidden_state_store import atomic_write_json  # noqa: E402
from layer_metacognition.model_adapter import (  # noqa: E402
    load_qwen_inference,
    resolve_language_modules,
)
from layer_metacognition.steering.decision_side_steering import (  # noqa: E402
    CONFLICT_CONDITIONS,
    DEFAULT_CASES_PER_DECISION_SIDE,
    INJECTION_SITES,
    INTERVENTION_MODES,
    STEERING_POSITIONS,
    STEERING_SCALES,
    BaselineHiddenStateRepository,
    DirectionRepository,
    assert_cuda_only_model,
    configuration_fingerprint,
    execute_run,
    file_sha256,
    initialize_output,
    prepare_cases,
    validate_grid,
)


DEFAULT_EXPERIMENT_DIR = (
    ROOT / "layer_metacognition" / "output" / "Final_v4_run" / "answer_basis_9"
)
DEFAULT_MODEL_PATH = ROOT / "qwen-2.5-vl" / "models" / "Qwen2.5-VL-7B-Instruct"


def _string_set(values: list[str] | None, label: str) -> set[str] | None:
    if values is None:
        return None
    output = {
        part.strip()
        for value in values
        for part in str(value).split(",")
        if part.strip()
    }
    if not output:
        raise ValueError(f"--{label} requires at least one value")
    return output


def _canonical_conditions(values: list[str]) -> list[str]:
    expanded = [
        part.strip()
        for value in values
        for part in str(value).split(",")
        if part.strip()
    ]
    if not expanded or len(expanded) != len(set(expanded)):
        raise ValueError("--conditions must contain distinct values")
    invalid = [value for value in expanded if value not in CONFLICT_CONDITIONS]
    if invalid:
        raise ValueError(
            "Decision-Side Steering only supports conflict_easy/conflict_hard; "
            f"invalid: {invalid}"
        )
    return [value for value in CONDITIONS if value in set(expanded)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    parser.add_argument("--probe-run-dir")
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument(
        "--positions",
        nargs="+",
        choices=list(STEERING_POSITIONS),
        required=True,
    )
    parser.add_argument("--alphas", nargs="+", type=float, required=True)
    parser.add_argument(
        "--steering-scale",
        choices=list(STEERING_SCALES),
        default="probe_logit",
    )
    parser.add_argument(
        "--injection-site",
        choices=list(INJECTION_SITES),
        default="block_output",
        help=(
            "block_output preserves the original experiment; block_input makes "
            "the perturbation visible to the target block's own attention/KV"
        ),
    )
    parser.add_argument(
        "--intervention-mode",
        choices=list(INTERVENTION_MODES),
        default="single",
        help=(
            "single injects once; reinject adds the same requested probe-logit "
            "shift again at every downstream layer with an OOF direction"
        ),
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(CONFLICT_CONDITIONS),
    )
    parser.add_argument("--item-ids", nargs="+")
    parser.add_argument("--prior-indices", nargs="+", type=int)
    parser.add_argument(
        "--cases-per-decision-side",
        type=int,
        default=DEFAULT_CASES_PER_DECISION_SIDE,
        help="Stably select this many follows_text and follows_image cases.",
    )
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--max-baseline-abs-answer-margin",
        type=float,
        help=(
            "Keep only cases with abs(pre-Steering AnswerMargin) below this "
            "threshold before balanced selection"
        ),
    )
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--dataset", default=str(ROOT / "datasets" / "datasets.json"))
    parser.add_argument("--image-root")
    parser.add_argument("--inference-path", default=str(DEFAULT_INFERENCE_PATH))
    parser.add_argument("--output-dir")
    parser.add_argument("--max-answer-tokens", type=int, default=24)
    parser.add_argument("--resume", action="store_true")
    return parser


def _validate_output_before_model(
    output_dir: Path,
    configuration: dict[str, Any],
    *,
    resume: bool,
) -> None:
    artifacts = [
        output_dir / name
        for name in ("run_config.json", "results.jsonl", "progress.json", "summary.json")
    ]
    existing = [path for path in artifacts if path.exists()]
    if existing and not resume:
        raise ValueError(
            "Steering output already exists; pass --resume or choose a new directory: "
            + ", ".join(str(path) for path in existing)
        )
    if resume:
        config_path = output_dir / "run_config.json"
        if not config_path.is_file():
            raise ValueError("--resume requires an existing run_config.json")
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        if saved.get("config_fingerprint") != configuration_fingerprint(configuration):
            raise ValueError("Resume configuration differs from saved run_config.json")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        layers, positions, alphas = validate_grid(
            args.layers,
            args.positions,
            args.alphas,
        )
        conditions = _canonical_conditions(args.conditions)
        item_ids = _string_set(args.item_ids, "item-ids")
        prior_indices = set(args.prior_indices) if args.prior_indices else None
        if prior_indices is not None and (
            len(prior_indices) != len(args.prior_indices)
            or any(value < 0 for value in prior_indices)
        ):
            raise ValueError("--prior-indices must contain distinct non-negative values")
        if args.max_cases is not None and args.max_cases < 1:
            raise ValueError("--max-cases must be positive")
        if args.max_baseline_abs_answer_margin is not None and (
            not math.isfinite(args.max_baseline_abs_answer_margin)
            or args.max_baseline_abs_answer_margin <= 0.0
        ):
            raise ValueError("--max-baseline-abs-answer-margin must be positive")
        if args.max_answer_tokens < 1:
            raise ValueError("--max-answer-tokens must be positive")
        if args.intervention_mode == "reinject" and args.injection_site != "block_output":
            raise ValueError(
                "--intervention-mode reinject only supports "
                "--injection-site block_output"
            )

        experiment_dir = Path(args.experiment_dir).resolve()
        probe_run_dir = (
            Path(args.probe_run_dir).resolve()
            if args.probe_run_dir
            else experiment_dir / "stage1_metacognition" / "item_split"
        )
        output_dir = (
            Path(args.output_dir).resolve()
            if args.output_dir
            else experiment_dir / "stage2_decision_steering"
        )
        dataset = Path(args.dataset).resolve()
        model_path = Path(args.model_path).resolve()
        inference_path = Path(args.inference_path).resolve()
        image_root = Path(args.image_root).resolve() if args.image_root else None
        if not experiment_dir.is_dir():
            raise FileNotFoundError(f"Experiment directory does not exist: {experiment_dir}")
        if not (experiment_dir / "results.jsonl").is_file():
            raise FileNotFoundError(f"Baseline results do not exist under {experiment_dir}")
        if not dataset.is_file():
            raise FileNotFoundError(f"Dataset does not exist: {dataset}")
        if not model_path.is_dir():
            raise FileNotFoundError(f"Model does not exist: {model_path}")
        if not inference_path.is_file():
            raise FileNotFoundError(f"Inference source does not exist: {inference_path}")
        if image_root is not None and not image_root.is_dir():
            raise FileNotFoundError(f"Image root does not exist: {image_root}")

        # This check deliberately precedes dataset fallback creation, output
        # initialization, and Qwen loading.  There is no CPU execution path.
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; CPU Steering is forbidden")

        if not args.resume:
            occupied = [
                output_dir / name
                for name in (
                    "run_config.json",
                    "results.jsonl",
                    "progress.json",
                    "summary.json",
                )
                if (output_dir / name).exists()
            ]
            if occupied:
                raise ValueError(
                    "Steering output already exists; pass --resume or choose a new "
                    "directory: " + ", ".join(str(path) for path in occupied)
                )

        repository = DirectionRepository(probe_run_dir)
        repository.validate_requested_grid(layers, positions)
        cases, dataset_metadata = prepare_cases(
            repository=repository,
            experiment_dir=experiment_dir,
            dataset=dataset,
            image_root=image_root,
            conditions=conditions,
            item_ids=item_ids,
            prior_indices=prior_indices,
            cases_per_decision_side=args.cases_per_decision_side,
            max_cases=args.max_cases,
            max_baseline_abs_answer_margin=(
                args.max_baseline_abs_answer_margin
            ),
            fallback_null_path=output_dir / ".runtime" / "null.png",
        )
        baseline_hidden_states = BaselineHiddenStateRepository(experiment_dir)
        trajectory_layers_by_position = baseline_hidden_states.validate_cases(
            cases,
            repository,
            layers,
            positions,
        )
        expected_count = len(cases) * len(layers) * len(positions) * len(alphas)
        configuration = {
            "format_version": 3,
            "experiment_dir": str(experiment_dir),
            "probe_run_dir": str(probe_run_dir),
            "manifest_path": str(repository.manifest_path),
            "dataset": str(dataset),
            "image_root": str(image_root) if image_root else None,
            "model_path": str(model_path),
            "inference_path": str(inference_path),
            "output_dir": str(output_dir),
            "layers": list(layers),
            "positions": list(positions),
            "alphas": list(alphas),
            "steering_scale": args.steering_scale,
            "injection_site": args.injection_site,
            "intervention_mode": args.intervention_mode,
            "injection_site_semantics": (
                "post_block_residual_original"
                if args.injection_site == "block_output"
                else "pre_block_residual_kv_visible_direction_coordinates_are_post_block"
            ),
            "reinject_semantics": (
                "add_requested_shift_at_each_available_downstream_oof_layer"
                if args.intervention_mode == "reinject"
                else None
            ),
            "conditions": list(conditions),
            "item_ids": sorted(item_ids) if item_ids else None,
            "prior_indices": sorted(prior_indices) if prior_indices else None,
            "cases_per_decision_side": args.cases_per_decision_side,
            "balanced_selection_order": "stable_case_order_within_decision_side",
            "max_cases": args.max_cases,
            "max_baseline_abs_answer_margin": (
                args.max_baseline_abs_answer_margin
            ),
            "max_answer_tokens": args.max_answer_tokens,
            "case_ids": [str(case.manifest["case_id"]) for case in cases],
            "case_count": len(cases),
            "decision_side_counts": {
                label: sum(
                    case.manifest["decision_side"] == label for case in cases
                )
                for label in ("follows_text", "follows_image")
            },
            "condition_counts": {
                condition: sum(
                    case.manifest["condition"] == condition for case in cases
                )
                for condition in conditions
            },
            "expected_intervention_count": expected_count,
            "version": "v4",
            "attribution_mode": "joint",
            "source_prompt_variant": "answer_basis_9",
            "direction_version_setting": "v4_to_v4",
            "direction_positive_class": "follows_image",
            "trajectory_enabled": True,
            "trajectory_hidden_state_definition": (
                "decoder_block_output_pre_final_norm_at_same_token"
            ),
            "trajectory_layers_by_position": trajectory_layers_by_position,
            "trajectory_baseline_source": "original_experiment_hidden_states",
            "trajectory_readout": "own_layer_item_split_oof_decision_probe",
            "trajectory_retention_denominator": "injection_layer_delta_logit",
            "trajectory_vector_diagnostics": [
                "delta_hidden_l2",
                "delta_hidden_projection_on_d_K",
                "delta_hidden_cosine_with_d_K",
                "delta_hidden_orthogonal_l2",
                "directional_energy_fraction",
            ],
            "panl_answer_effect": "not_applicable_post_answer",
            "baseline_definition": "pre_steering_results_reused_for_alpha_zero",
            "alpha_zero_execution": "reuse_without_model_forward",
            "per_intervention_failure_policy": "record_and_continue",
            "cuda_only": True,
            "manifest_sha256": file_sha256(repository.manifest_path),
            "split_assignments_sha256": file_sha256(repository.split_path),
            "direction_index_sha256": file_sha256(repository.index_path),
            "hidden_state_index_sha256": file_sha256(
                baseline_hidden_states.index_path
            ),
            "stage1_config_fingerprint": repository.run_config.get("config_fingerprint"),
            "dataset_metadata": dataset_metadata,
        }
        _validate_output_before_model(output_dir, configuration, resume=args.resume)

        inference = load_qwen_inference(str(model_path), inference_path)
        modules = resolve_language_modules(inference.model)
        assert_cuda_only_model(inference.model, modules)
        if any(layer >= modules.num_hidden_layers for layer in layers):
            raise ValueError(
                f"Requested layer exceeds model range [0, {modules.num_hidden_layers - 1}]"
            )
        sample_direction = repository.get(cases[0].fold, layers[0], positions[0])
        if sample_direction.d_raw.size != modules.hidden_size:
            raise ValueError(
                f"Direction hidden size {sample_direction.d_raw.size} does not match "
                f"model hidden size {modules.hidden_size}"
            )

        source_variant = get_source_prompt_variant("answer_basis_9")
        joint_generator = JointAnswerSourceGenerator(inference)
        source_analyzer = SourceAttributionAnalyzer(
            inference,
            source_classes=source_variant.classes,
            source_midpoints=source_variant.midpoints,
        )
        existing, config_path, results_path, progress_path, summary_path = initialize_output(
            output_dir,
            configuration,
            resume=args.resume,
        )
        runtime = {
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
        }
        atomic_write_json(
            config_path,
            {
                **configuration,
                "config_fingerprint": configuration_fingerprint(configuration),
                "model_runtime": runtime,
                "status": "running",
            },
        )
        try:
            records = execute_run(
                cases=cases,
                repository=repository,
                joint_generator=joint_generator,
                source_analyzer=source_analyzer,
                modules=modules,
                baseline_hidden_states=baseline_hidden_states,
                layers=layers,
                positions=positions,
                alphas=alphas,
                steering_scale=args.steering_scale,
                injection_site=args.injection_site,
                intervention_mode=args.intervention_mode,
                max_answer_tokens=args.max_answer_tokens,
                existing=existing,
                results_path=results_path,
                progress_path=progress_path,
                summary_path=summary_path,
            )
        except Exception:
            atomic_write_json(
                config_path,
                {
                    **configuration,
                    "config_fingerprint": configuration_fingerprint(configuration),
                    "model_runtime": runtime,
                    "status": "failed",
                },
            )
            raise
        failed_count = sum(record.get("status") == "failed" for record in records)
        if len(records) == expected_count:
            final_status = "complete_with_failures" if failed_count else "complete"
        else:
            final_status = "incomplete"
        atomic_write_json(
            config_path,
            {
                **configuration,
                "config_fingerprint": configuration_fingerprint(configuration),
                "model_runtime": runtime,
                "status": final_status,
            },
        )
        print(
            f"[INFO] Decision-Side Steering {final_status}: "
            f"records={len(records)} output={output_dir}"
        )
        return 0
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
