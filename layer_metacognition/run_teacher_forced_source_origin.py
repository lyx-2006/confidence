#!/usr/bin/env python3
"""Run Teacher-Forced Source Attribution causal experiments."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

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
from layer_metacognition.model_adapter import (  # noqa: E402
    load_qwen_inference,
    resolve_language_modules,
)
from layer_metacognition.steering.decision_side_steering import (  # noqa: E402
    assert_cuda_only_model,
)
from layer_metacognition.teacher_forced_source_origin import (  # noqa: E402
    CONFLICT_CONDITIONS,
    DECISION_SIDES,
    FORMAT_VERSION,
    INTERVENTION_LAYERS,
    TeacherForcedSourceOriginRunner,
    cohort_manifest_payload,
    load_causal_candidates,
    select_balanced_cohort,
)


DEFAULT_EXPERIMENT_DIR = (
    ROOT / "layer_metacognition" / "output" / "Final_v4_run" / "answer_basis_9"
)
DEFAULT_MODEL_PATH = ROOT / "qwen-2.5-vl" / "models" / "Qwen2.5-VL-7B-Instruct"
DEFAULT_DATASET = ROOT / "datasets" / "datasets.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    parser.add_argument("--output-dir")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--inference-path", default=str(DEFAULT_INFERENCE_PATH))
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=list(INTERVENTION_LAYERS),
    )
    parser.add_argument("--cases-per-cell", type=int, default=25)
    parser.add_argument("--state-pair-min-gap", type=float, default=0.15)
    parser.add_argument("--max-state-pairs", type=int, default=30)
    parser.add_argument("--self-swap-tolerance", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run a small path-complete profile at L20 in a separate smoke output; "
            "never expands to the formal cohort"
        ),
    )
    return parser


def _validate_layers(values: Sequence[int]) -> list[int]:
    layers = [int(value) for value in values]
    if not layers or len(layers) != len(set(layers)):
        raise ValueError("--layers must contain distinct values")
    if any(layer < 0 for layer in layers):
        raise ValueError("--layers cannot contain negative values")
    return sorted(layers)


def select_smoke_cohort(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Choose one same-answer pair plus side/difficulty coverage without SA filtering."""

    ordered = sorted(
        (dict(row) for row in candidates),
        key=lambda row: (
            int(row["item_order"]),
            int(row["prior_index"]),
            str(row["condition"]),
        ),
    )
    pair: tuple[dict[str, Any], dict[str, Any]] | None = None
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if (
                left["normalized_answer"] == right["normalized_answer"]
                and left["decision_side"] == right["decision_side"]
                and left["case_id"] != right["case_id"]
            ):
                pair = (left, right)
                break
        if pair is not None:
            break
    if pair is None:
        raise ValueError("No same-answer/same-side pair exists for smoke")
    selected = [dict(pair[0]), dict(pair[1])]
    selected_ids = {str(row["case_id"]) for row in selected}
    for side in DECISION_SIDES:
        for condition in CONFLICT_CONDITIONS:
            if any(
                row["decision_side"] == side and row["condition"] == condition
                for row in selected
            ):
                continue
            candidate = next(
                row
                for row in ordered
                if row["decision_side"] == side
                and row["condition"] == condition
                and row["case_id"] not in selected_ids
            )
            selected.append(dict(candidate))
            selected_ids.add(str(candidate["case_id"]))
    return sorted(
        selected,
        key=lambda row: (
            int(row["item_order"]),
            int(row["prior_index"]),
            str(row["condition"]),
        ),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        experiment_dir = Path(args.experiment_dir).resolve()
        dataset = Path(args.dataset).resolve()
        model_path = Path(args.model_path).resolve()
        inference_path = Path(args.inference_path).resolve()
        if not experiment_dir.is_dir():
            raise FileNotFoundError(f"Experiment directory does not exist: {experiment_dir}")
        if not dataset.is_file():
            raise FileNotFoundError(f"Dataset does not exist: {dataset}")
        if not model_path.is_dir():
            raise FileNotFoundError(f"Model does not exist: {model_path}")
        if not inference_path.is_file():
            raise FileNotFoundError(f"Inference source does not exist: {inference_path}")
        if args.cases_per_cell < 1:
            raise ValueError("--cases-per-cell must be positive")
        if args.max_state_pairs < 0:
            raise ValueError("--max-state-pairs cannot be negative")
        if not math.isfinite(args.state_pair_min_gap) or args.state_pair_min_gap < 0:
            raise ValueError("--state-pair-min-gap must be finite and non-negative")
        if not math.isfinite(args.self_swap_tolerance) or args.self_swap_tolerance < 0:
            raise ValueError("--self-swap-tolerance must be finite and non-negative")
        layers = _validate_layers(args.layers)
        if args.smoke:
            layers = [20]

        default_name = (
            "stage2_teacher_forced_source_origin_smoke"
            if args.smoke
            else "stage2_teacher_forced_source_origin"
        )
        output_dir = (
            Path(args.output_dir).resolve()
            if args.output_dir
            else experiment_dir / default_name
        )
        occupied = [
            output_dir / name
            for name in (
                "run_config.json",
                "cohort_manifest.json",
                "results.jsonl",
                "summary.json",
            )
            if (output_dir / name).exists()
        ]
        if occupied and not args.resume:
            raise ValueError(
                "Output already exists; pass --resume or choose another directory: "
                + ", ".join(str(path) for path in occupied)
            )

        candidates, dataset_metadata = load_causal_candidates(
            experiment_dir=experiment_dir,
            dataset=dataset,
            fallback_null_path=output_dir / ".runtime" / "null.png",
        )
        if args.smoke:
            cohort = select_smoke_cohort(candidates)
            cases_per_cell = 0
            selection_profile = "smoke_path_complete"
            min_gap = 0.0
            max_pairs = 1
        else:
            cohort = select_balanced_cohort(
                candidates,
                cases_per_cell=args.cases_per_cell,
            )
            cases_per_cell = args.cases_per_cell
            selection_profile = "formal_balanced"
            min_gap = float(args.state_pair_min_gap)
            max_pairs = int(args.max_state_pairs)
        manifest = cohort_manifest_payload(
            cohort,
            source_candidate_count=len(candidates),
            cases_per_cell=cases_per_cell,
            selection_profile=selection_profile,
        )
        configuration = {
            "format_version": FORMAT_VERSION,
            "experiment_dir": str(experiment_dir),
            "dataset": str(dataset),
            "model_path": str(model_path),
            "inference_path": str(inference_path),
            "output_dir": str(output_dir),
            "source_prompt_variant": "answer_basis_9",
            "prompt_mutated": False,
            "intervention_layers": layers,
            "prestored_layers": "all_decoder_layers",
            "cases_per_cell": cases_per_cell,
            "state_pair_min_gap": min_gap,
            "max_state_pairs": max_pairs,
            "self_swap_tolerance": float(args.self_swap_tolerance),
            "do_sample": False,
            "use_cache": True,
            "state_retention_policy": "stream_delete",
            "selection_profile": selection_profile,
            "dataset_metadata": dataset_metadata,
            "causal_baseline": "same_input_same_forced_answer_clean_teacher_forced",
            "decision_side_K_interpretation": "predictive_representation_only",
        }

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; Qwen causal smoke cannot run on CPU")
        inference = load_qwen_inference(
            model_path=str(model_path),
            inference_path=inference_path,
        )
        modules = resolve_language_modules(inference.model)
        if any(layer >= modules.num_hidden_layers for layer in layers):
            raise ValueError(
                f"Requested layers must be below {modules.num_hidden_layers}: {layers}"
            )
        assert_cuda_only_model(inference.model, modules)
        variant = get_source_prompt_variant("answer_basis_9")
        joint_generator = JointAnswerSourceGenerator(inference)
        source_analyzer = SourceAttributionAnalyzer(
            inference,
            source_classes=variant.classes,
            source_midpoints=variant.midpoints,
        )
        configuration["source_classes"] = list(variant.classes)
        configuration["source_midpoints"] = list(variant.midpoints)
        configuration["num_hidden_layers"] = modules.num_hidden_layers
        configuration["hidden_size"] = modules.hidden_size
        configuration["model_dtype"] = str(inference.dtype_name)
        runner = TeacherForcedSourceOriginRunner(
            inference=inference,
            modules=modules,
            joint_generator=joint_generator,
            source_analyzer=source_analyzer,
            source_variant=variant,
            output_dir=output_dir,
            configuration=configuration,
            cohort_manifest=manifest,
            layers=layers,
            state_pair_min_gap=min_gap,
            max_state_pairs=max_pairs,
            self_swap_tolerance=float(args.self_swap_tolerance),
            resume=args.resume,
        )
        summary = runner.execute()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

