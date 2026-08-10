#!/usr/bin/env python3
"""Run strong-SA mean-difference steering at AC and PANL."""

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
    MEAN_DIFFERENCE_STEERING_SCALE,
    BaselineHiddenStateRepository,
    assert_cuda_only_model,
    build_paired_summary,
    configuration_fingerprint,
    execute_run,
    file_sha256,
    initialize_output,
    prepare_cases,
    validate_grid,
)
from layer_metacognition.steering.source_attribution_mean_steering import (  # noqa: E402
    SOURCE_POSITIONS,
    build_mean_directions,
    load_sa_candidates,
    MeanSADirectionRepository,
    persist_direction_artifacts,
    select_heldout_evaluation_cases,
    select_strong_sa_sources,
    source_manifest_payload,
)


DEFAULT_EXPERIMENT_DIR = (
    ROOT / "layer_metacognition" / "output" / "Final_v4_run" / "answer_basis_9"
)
DEFAULT_MODEL_PATH = ROOT / "qwen-2.5-vl" / "models" / "Qwen2.5-VL-7B-Instruct"
DEFAULT_LAYERS = (20, 24)
DEFAULT_ALPHAS = (-1.0, -0.5, 0.0, 0.5, 1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    parser.add_argument("--layers", nargs="+", type=int, default=list(DEFAULT_LAYERS))
    parser.add_argument(
        "--positions",
        nargs="+",
        choices=list(SOURCE_POSITIONS),
        default=list(SOURCE_POSITIONS),
    )
    parser.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    parser.add_argument("--source-cases-per-side", type=int, default=25)
    parser.add_argument("--eval-cases-per-side", type=int, default=25)
    parser.add_argument(
        "--max-cases",
        type=int,
        help="Limit the held-out evaluation cohort after balanced selection (smoke use).",
    )
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--dataset", default=str(ROOT / "datasets" / "datasets.json"))
    parser.add_argument("--image-root")
    parser.add_argument("--inference-path", default=str(DEFAULT_INFERENCE_PATH))
    parser.add_argument("--output-dir")
    parser.add_argument("--max-answer-tokens", type=int, default=24)
    parser.add_argument("--resume", action="store_true")
    return parser


def _public_direction_metadata(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in entry.items() if not hasattr(value, "shape")}
        for entry in entries
    ]


def _validate_output_before_model(
    output_dir: Path,
    configuration: dict[str, Any],
    *,
    resume: bool,
) -> None:
    artifacts = [
        output_dir / name
        for name in (
            "run_config.json",
            "results.jsonl",
            "progress.json",
            "summary.json",
            "source_cohort_manifest.json",
            "evaluation_manifest.json",
        )
    ]
    if (output_dir / "directions").exists():
        artifacts.append(output_dir / "directions")
    existing = [path for path in artifacts if path.exists()]
    if existing and not resume:
        raise ValueError(
            "SA mean Steering output already exists; pass --resume or choose a new "
            "directory: " + ", ".join(str(path) for path in existing)
        )
    if resume:
        config_path = output_dir / "run_config.json"
        if not config_path.is_file():
            raise ValueError("--resume requires an existing run_config.json")
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        if saved.get("config_fingerprint") != configuration_fingerprint(configuration):
            raise ValueError("Resume configuration differs from saved run_config.json")


def _evaluation_manifest(cases: list[Any], excluded_items: set[str]) -> dict[str, Any]:
    return {
        "format_version": 1,
        "source_item_overlap": False,
        "excluded_source_item_ids": sorted(excluded_items),
        "case_count": len(cases),
        "decision_side_counts": {
            side: sum(case.manifest["decision_side"] == side for case in cases)
            for side in ("follows_text", "follows_image")
        },
        "condition_counts": {
            condition: sum(case.manifest["condition"] == condition for case in cases)
            for condition in CONFLICT_CONDITIONS
        },
        "cases": [
            {
                "case_id": str(case.manifest["case_id"]),
                "item_id": str(case.manifest["item_id"]),
                "prior_index": int(case.manifest["prior_index"]),
                "condition": str(case.manifest["condition"]),
                "decision_side": str(case.manifest["decision_side"]),
            }
            for case in cases
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        layers, positions, alphas = validate_grid(args.layers, args.positions, args.alphas)
        if any(position not in SOURCE_POSITIONS for position in positions):
            raise ValueError("This experiment only supports AC and PANL")
        if args.source_cases_per_side < 1 or args.eval_cases_per_side < 1:
            raise ValueError("Source/evaluation cases per side must be positive")
        if args.max_cases is not None and args.max_cases < 1:
            raise ValueError("--max-cases must be positive")
        if args.max_answer_tokens < 1:
            raise ValueError("--max-answer-tokens must be positive")
        if not all(math.isfinite(value) for value in alphas):
            raise ValueError("--alphas must be finite")

        experiment_dir = Path(args.experiment_dir).resolve()
        manifest_path = experiment_dir / "extended_probe" / "probe_manifest.jsonl"
        output_dir = (
            Path(args.output_dir).resolve()
            if args.output_dir
            else experiment_dir / "stage2_sa_mean_steering"
        )
        dataset = Path(args.dataset).resolve()
        model_path = Path(args.model_path).resolve()
        inference_path = Path(args.inference_path).resolve()
        image_root = Path(args.image_root).resolve() if args.image_root else None
        required_files = [
            experiment_dir / "results.jsonl",
            experiment_dir / "hidden_states" / "index.json",
            manifest_path,
            dataset,
            inference_path,
        ]
        missing = [str(path) for path in required_files if not path.is_file()]
        if missing:
            raise FileNotFoundError("Required inputs are missing: " + ", ".join(missing))
        if not model_path.is_dir():
            raise FileNotFoundError(f"Model does not exist: {model_path}")
        if image_root is not None and not image_root.is_dir():
            raise FileNotFoundError(f"Image root does not exist: {image_root}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; CPU Steering is forbidden")

        hidden_states = BaselineHiddenStateRepository(experiment_dir)
        stored_layers = [int(value) for value in hidden_states.index.get("layer_indices", [])]
        if not stored_layers:
            raise ValueError("Baseline hidden-state index has no layers")
        unavailable = [layer for layer in layers if layer not in stored_layers]
        if unavailable:
            raise ValueError(
                f"Requested layers lack saved baseline states: {unavailable}; "
                f"available={stored_layers}"
            )

        candidates = load_sa_candidates(experiment_dir, manifest_path)
        source_groups = select_strong_sa_sources(
            candidates,
            cases_per_side=args.source_cases_per_side,
        )
        source_manifest = source_manifest_payload(source_groups)
        source_manifest["candidate_count"] = len(candidates)
        directions, direction_entries = build_mean_directions(
            groups=source_groups,
            hidden_states=hidden_states,
            layers=stored_layers,
            positions=positions,
        )
        repository = MeanSADirectionRepository(
            directions=directions,
            manifest_path=manifest_path,
        )
        repository.validate_requested_grid(layers, positions)

        all_cases, dataset_metadata = prepare_cases(
            repository=repository,
            experiment_dir=experiment_dir,
            dataset=dataset,
            image_root=image_root,
            conditions=CONFLICT_CONDITIONS,
            item_ids=None,
            prior_indices=None,
            cases_per_decision_side=None,
            max_cases=None,
            fallback_null_path=output_dir / ".runtime" / "null.png",
        )
        source_item_ids = {
            str(record["item_id"])
            for group in source_groups.values()
            for record in group
        }
        cases = select_heldout_evaluation_cases(
            all_cases,
            excluded_item_ids=source_item_ids,
            cases_per_side=args.eval_cases_per_side,
            max_cases=args.max_cases,
        )
        if source_item_ids.intersection(
            str(case.manifest["item_id"]) for case in cases
        ):
            raise ValueError("Source/evaluation item leakage detected")
        trajectory_layers_by_position = hidden_states.validate_cases(
            cases,
            repository,
            layers,
            positions,
        )
        evaluation_manifest = _evaluation_manifest(cases, source_item_ids)
        expected_count = len(cases) * len(layers) * len(positions) * len(alphas)
        public_directions = _public_direction_metadata(direction_entries)
        configuration = {
            "format_version": 1,
            "experiment": "strong_sa_mean_difference_steering",
            "experiment_dir": str(experiment_dir),
            "manifest_path": str(manifest_path),
            "dataset": str(dataset),
            "image_root": str(image_root) if image_root else None,
            "model_path": str(model_path),
            "inference_path": str(inference_path),
            "output_dir": str(output_dir),
            "layers": list(layers),
            "positions": list(positions),
            "alphas": list(alphas),
            "steering_scale": MEAN_DIFFERENCE_STEERING_SCALE,
            "vector_definition": "mean_strong_image_minus_mean_strong_text",
            "injection_vector": "alpha_times_raw_mean_difference",
            "positive_direction": "+alpha -> imageward",
            "negative_direction": "-alpha -> textward",
            "source_score": "SA_soft_image_score",
            "source_cases_per_side": args.source_cases_per_side,
            "source_case_ids": {
                side: [str(record["case_id"]) for record in source_groups[side]]
                for side in ("follows_text", "follows_image")
            },
            "source_item_disjoint": True,
            "source_evaluation_item_disjoint": True,
            "eval_cases_per_side": args.eval_cases_per_side,
            "max_cases": args.max_cases,
            "case_ids": [str(case.manifest["case_id"]) for case in cases],
            "case_count": len(cases),
            "decision_side_counts": evaluation_manifest["decision_side_counts"],
            "condition_counts": evaluation_manifest["condition_counts"],
            "expected_intervention_count": expected_count,
            "injection_site": "block_output",
            "intervention_mode": "single",
            "ac_outcomes": ["answer_margin", "SA_soft_image_score"],
            "panl_outcomes": ["SA_soft_image_score"],
            "panl_answer_effect": "not_applicable_post_answer",
            "alpha_zero_execution": "reuse_pre_steering_baseline_without_forward",
            "trajectory_layers_by_position": trajectory_layers_by_position,
            "trajectory_readout": "same_layer_mean_SA_direction_projection",
            "stored_direction_layers": stored_layers,
            "direction_metadata": public_directions,
            "max_answer_tokens": args.max_answer_tokens,
            "source_manifest_sha256": file_sha256(manifest_path),
            "baseline_results_sha256": file_sha256(experiment_dir / "results.jsonl"),
            "hidden_state_index_sha256": file_sha256(hidden_states.index_path),
            "dataset_metadata": dataset_metadata,
            "per_intervention_failure_policy": "record_and_continue",
            "cuda_only": True,
        }
        _validate_output_before_model(output_dir, configuration, resume=args.resume)
        existing, config_path, results_path, progress_path, summary_path = initialize_output(
            output_dir,
            configuration,
            resume=args.resume,
        )
        persist_direction_artifacts(
            output_dir,
            source_manifest=source_manifest,
            direction_metadata=direction_entries,
        )
        evaluation_path = output_dir / "evaluation_manifest.json"
        if evaluation_path.exists():
            saved_evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            if saved_evaluation != evaluation_manifest:
                raise ValueError("Existing evaluation manifest differs")
        else:
            atomic_write_json(evaluation_path, evaluation_manifest)

        inference = load_qwen_inference(str(model_path), inference_path)
        modules = resolve_language_modules(inference.model)
        assert_cuda_only_model(inference.model, modules)
        if any(layer >= modules.num_hidden_layers for layer in layers):
            raise ValueError(
                f"Requested layer exceeds model range [0, {modules.num_hidden_layers - 1}]"
            )
        sample = repository.get(0, layers[0], positions[0])
        if sample.d_raw.size != modules.hidden_size:
            raise ValueError(
                f"Direction hidden size {sample.d_raw.size} does not match model "
                f"hidden size {modules.hidden_size}"
            )
        source_variant = get_source_prompt_variant("answer_basis_9")
        joint_generator = JointAnswerSourceGenerator(inference)
        source_analyzer = SourceAttributionAnalyzer(
            inference,
            source_classes=source_variant.classes,
            source_midpoints=source_variant.midpoints,
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
                baseline_hidden_states=hidden_states,
                layers=layers,
                positions=positions,
                alphas=alphas,
                steering_scale=MEAN_DIFFERENCE_STEERING_SCALE,
                injection_site="block_output",
                intervention_mode="single",
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
        final_status = (
            "complete_with_failures"
            if len(records) == expected_count and failed_count
            else "complete"
            if len(records) == expected_count
            else "incomplete"
        )
        summary = build_paired_summary(records)
        summary.update(
            {
                "experiment": "strong_sa_mean_difference_steering",
                "source_groups": source_manifest["groups"],
                "source_evaluation_item_overlap": False,
                "direction_metadata": public_directions,
                "interpretation": {
                    "positive_alpha": "imageward",
                    "negative_alpha": "textward",
                    "ac": "answer_and_source_report_steering",
                    "panl": "post_answer_source_report_steering_only",
                },
            }
        )
        atomic_write_json(summary_path, summary)
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
            f"[INFO] Strong-SA mean Steering {final_status}: "
            f"records={len(records)} output={output_dir}"
        )
        return 0
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

