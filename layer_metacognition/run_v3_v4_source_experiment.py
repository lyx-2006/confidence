#!/usr/bin/env python3
"""Run V3/V4 source attribution, layer readout, and per-head sinks."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from confidence_test.confidence_extension import MultimodalConfidenceAnalyzer  # noqa: E402
from confidence_test.dataset_utils import (  # noqa: E402
    CONDITIONS,
    ConditionInput,
    load_evaluation_cases,
)
from confidence_test.inference_extension import build_extended_inference_class  # noqa: E402
from confidence_test.joint_answer_source_extension import (  # noqa: E402
    JointAnswerSourceGenerator,
)
from confidence_test.runtime_imports import DEFAULT_INFERENCE_PATH, load_runtime  # noqa: E402
from confidence_test.source_attribution_analyzer import SourceAttributionAnalyzer  # noqa: E402
from confidence_test.source_attribution_schema import (  # noqa: E402
    SOURCE_ATTRIBUTION_CLASSES,
    SOURCE_ATTRIBUTION_CLASS_TEXT,
    SOURCE_ATTRIBUTION_MIDPOINTS,
)
from layer_metacognition.analyze_main_results import (  # noqa: E402
    build_minimal_analysis,
    write_minimal_analysis,
)
from layer_metacognition.analyze_source_sink_results import (  # noqa: E402
    build_source_sink_minimal,
    build_source_sink_summary,
    group_records_by_version,
    load_case_metadata,
    split_analysis_by_version,
    write_layer_readout_minimal,
    write_source_sink_minimal,
)
from layer_metacognition.hidden_state_store import (  # noqa: E402
    append_jsonl,
    atomic_write_json,
    load_jsonl,
)
from layer_metacognition.model_adapter import resolve_language_modules  # noqa: E402
from layer_metacognition.source_patchscope import SourcePatchscopeDecoder  # noqa: E402
from layer_metacognition.v3_v4_source_runner import (  # noqa: E402
    V3V4SourceRunner,
    reconstruction_tolerance,
)


DEFAULT_MODEL_PATH = ROOT / "qwen-2.5-vl" / "models" / "Qwen2.5-VL-7B-Instruct"
DEFAULT_DATASET_PATH = ROOT / "datasets" / "dataset_with_images.json"
DEFAULT_OUTPUT_DIR = ROOT / "layer_metacognition" / "output" / "v3_v4_source"
ATTRIBUTION_MODES = ("none", "parallel", "joint")
ANALYSIS_MODE_ORDER = ("LMhead", "Identity", "Semantic")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def expand_attribution_modes(value: str) -> tuple[str, ...]:
    """Expand the CLI's aggregate mode while keeping a stable execution order."""
    if value == "all":
        return ATTRIBUTION_MODES
    if value not in ATTRIBUTION_MODES:
        raise ValueError(f"Unknown attribution mode: {value}")
    return (value,)


def normalize_analysis_modes(values: Iterable[str]) -> tuple[str, ...]:
    """Reject duplicates and return the stable SAC readout execution order."""
    raw = [str(value) for value in values]
    duplicates = sorted({value for value in raw if raw.count(value) > 1})
    if duplicates:
        raise ValueError(
            "--analysis_mode contains duplicate value(s): "
            + ", ".join(duplicates)
        )
    invalid = [value for value in raw if value not in ANALYSIS_MODE_ORDER]
    if invalid:
        raise ValueError(f"Unknown analysis mode(s): {', '.join(invalid)}")
    if not raw:
        raise ValueError("--analysis_mode requires at least one mode")
    selected = set(raw)
    return tuple(mode for mode in ANALYSIS_MODE_ORDER if mode in selected)


def saved_configuration_for_comparison(
    saved: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Interpret pre-analysis-mode configs as the old LMhead-only default."""
    normalized = dict(saved)
    missing = "analysis_modes" not in normalized
    if missing:
        normalized["analysis_modes"] = ["LMhead"]
    return normalized, missing


def share_initial_cache(
    runners: list[V3V4SourceRunner],
    existing: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Attach one V3 initial-stage cache to every mode runner."""
    shared: dict[str, dict[str, Any]] = {}
    for runner in runners:
        runner.shared_initial = shared
    if runners:
        runners[0].seed_shared_initial(existing)
    return shared


def run_case_groups(
    runners: list[V3V4SourceRunner],
    cases: list[Any],
    *,
    versions: list[str],
    conditions: list[str],
    existing_ids: set[str],
    commit: Callable[[dict[str, Any]], None],
    after_group: Callable[[], None],
) -> dict[str, dict[str, int]]:
    """Run every mode for one base case before advancing to the next one."""
    for case in cases:
        for condition in conditions:
            for version in versions:
                group_changed = False
                for runner in runners:
                    case_id = (
                        f"{case.item_id}__prior_{case.prior_index}__"
                        f"{condition}__{version}__{runner.mode}"
                    )
                    if case_id in existing_ids:
                        runner._log("resume_skip", case_id=case_id)
                        continue
                    record = runner.process_case(
                        case=case,
                        condition=condition,
                        version=version,
                    )
                    commit(record)
                    existing_ids.add(case_id)
                    group_changed = True
                if group_changed:
                    after_group()
    return {runner.mode: dict(runner.call_counts) for runner in runners}


def _canonical_values(
    raw_values: Iterable[str],
    allowed: tuple[str, ...],
    label: str,
) -> list[str]:
    parts = [
        part.strip()
        for raw in raw_values
        for part in str(raw).split(",")
        if part.strip()
    ]
    if parts == ["all"]:
        return list(allowed)
    invalid = [part for part in parts if part not in allowed]
    if invalid:
        raise ValueError(f"Unknown {label}: {', '.join(invalid)}")
    if not parts:
        raise ValueError(f"At least one {label} is required")
    return [value for value in allowed if value in set(parts)]


def _parse_string_selection(values: list[str] | None) -> set[str] | None:
    if values is None:
        return None
    selected = {
        part.strip()
        for raw in values
        for part in raw.split(",")
        if part.strip()
    }
    if not selected:
        raise ValueError("--item-ids requires at least one ID")
    return selected


def _rebase_images(cases: list[Any], image_root: Path | None) -> list[Any]:
    if image_root is None:
        return cases
    output: list[Any] = []
    for case in cases:
        conditions: dict[str, ConditionInput] = {}
        for name, condition in case.conditions.items():
            raw = condition.relative_image_path
            if raw is None:
                conditions[name] = condition
                continue
            raw_path = Path(raw)
            resolved = raw_path.resolve() if raw_path.is_absolute() else (image_root / raw_path).resolve()
            error = None
            if not resolved.is_file():
                error = {
                    "type": "FileNotFoundError",
                    "message": f"Image does not exist under --image-root: {resolved}",
                }
            conditions[name] = replace(
                condition,
                resolved_image_path=str(resolved),
                error=error,
            )
        output.append(replace(case, conditions=conditions))
    return output


def configure_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("layer_metacognition.v3_v4_source")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _progress(records: list[dict[str, Any]], *, status: str) -> dict[str, Any]:
    completed = [
        str(record["case_id"])
        for record in records
        if record.get("status") == "completed"
    ]
    failed = [
        str(record["case_id"])
        for record in records
        if record.get("status") == "failed"
    ]
    return {
        "status": status,
        "updated_at": utc_now(),
        "record_count": len(records),
        "completed_count": len(completed),
        "failed_count": len(failed),
        "completed_case_ids": completed,
        "failed_case_ids": failed,
        "last_case_id": records[-1].get("case_id") if records else None,
    }


def _write_analyses(output_dir: Path, records: list[dict[str, Any]]) -> None:
    ordered_records = group_records_by_version(records)
    minimal, _skipped = build_minimal_analysis(ordered_records)
    write_minimal_analysis(output_dir / "analysis_minimal.json", minimal)
    metadata = None
    config_path = output_dir / "config.json"
    if config_path.is_file():
        try:
            dataset = json.loads(config_path.read_text(encoding="utf-8")).get("dataset")
            if isinstance(dataset, str) and Path(dataset).is_file():
                metadata = load_case_metadata(dataset)
        except Exception:
            logging.getLogger("layer_metacognition.v3_v4_source").exception(
                "analysis_metadata_load_failed"
            )
    source_analysis, statistics = build_source_sink_minimal(
        ordered_records,
        metadata,
    )
    write_source_sink_minimal(
        output_dir / "analysis_source_sink_minimal.json",
        source_analysis,
    )
    split_analysis = split_analysis_by_version(source_analysis)
    write_layer_readout_minimal(
        output_dir / "analysis_layer_readout_minimal_v3.json",
        split_analysis["v3"],
    )
    write_layer_readout_minimal(
        output_dir / "analysis_layer_readout_minimal_v4.json",
        split_analysis["v4"],
    )
    atomic_write_json(
        output_dir / "summary.json",
        build_source_sink_summary(ordered_records, source_analysis, statistics),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--image-root")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--inference-path", default=str(DEFAULT_INFERENCE_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--versions", nargs="+", default=["v3", "v4"])
    parser.add_argument(
        "--attribution-mode",
        choices=[*ATTRIBUTION_MODES, "all"],
        default="none",
    )
    parser.add_argument(
        "--analysis_mode",
        nargs="+",
        choices=list(ANALYSIS_MODE_ORDER),
        default=["LMhead"],
        help=(
            "One or more SAC hidden-state readouts. Execution and output order "
            "is always LMhead, Identity, Semantic; duplicate values are invalid. "
            "(default: LMhead)"
        ),
    )
    parser.add_argument("--conditions", nargs="+", default=["all"])
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--item-ids", nargs="+")
    parser.add_argument("--prior-indices", nargs="+", type=int)
    parser.add_argument("--skip-attention", action="store_true")
    parser.add_argument("--skip-layer-readout", action="store_true")
    parser.add_argument("--max-answer-tokens", type=int, default=24)
    parser.add_argument("--max-confidence-tokens", type=int, default=12)
    parser.add_argument("--max-source-tokens", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        versions = _canonical_values(args.versions, ("v3", "v4"), "version")
        analysis_modes = normalize_analysis_modes(args.analysis_mode)
        conditions = _canonical_values(args.conditions, CONDITIONS, "condition")
        item_ids = _parse_string_selection(args.item_ids)
        prior_indices = set(args.prior_indices) if args.prior_indices else None
        if prior_indices is not None and any(index < 0 for index in prior_indices):
            raise ValueError("--prior-indices must be non-negative")
        for name in (
            "max_items",
            "max_answer_tokens",
            "max_confidence_tokens",
            "max_source_tokens",
        ):
            value = getattr(args, name)
            if value is not None and value < 1:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
        dataset = Path(args.dataset).resolve()
        model_path = Path(args.model_path).resolve()
        inference_path = Path(args.inference_path).resolve()
        output_dir = Path(args.output_dir).resolve()
        image_root = Path(args.image_root).resolve() if args.image_root else None
        if not dataset.is_file():
            raise FileNotFoundError(f"Dataset does not exist: {dataset}")
        if not model_path.is_dir():
            raise FileNotFoundError(f"Model does not exist: {model_path}")
        if not inference_path.is_file():
            raise FileNotFoundError(f"Inference source does not exist: {inference_path}")
        if image_root is not None and not image_root.is_dir():
            raise FileNotFoundError(f"Image root does not exist: {image_root}")
    except Exception as exc:
        parser.error(str(exc))

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    config_path = output_dir / "config.json"
    progress_path = output_dir / "progress.json"
    logger = configure_logger(output_dir / "run.log")
    try:
        modes = expand_attribution_modes(args.attribution_mode)
        existing = load_jsonl(results_path, repair_trailing=args.resume)
        if existing and not args.resume:
            raise ValueError(
                f"Output already contains results; pass --resume or choose a new directory: {results_path}"
            )
        cases, dataset_metadata = load_evaluation_cases(
            dataset,
            item_limit=args.max_items,
            fallback_null_path=output_dir / ".runtime" / "null.png",
        )
        cases = _rebase_images(cases, image_root)
        if item_ids is not None:
            cases = [case for case in cases if case.item_id in item_ids]
            missing = item_ids.difference(case.item_id for case in cases)
            if missing:
                raise ValueError(f"Unknown --item-ids after filtering: {sorted(missing)}")
        if prior_indices is not None:
            missing_priors = prior_indices.difference(
                case.prior_index for case in cases
            )
            if missing_priors:
                raise ValueError(
                    f"Unknown --prior-indices after filtering: {sorted(missing_priors)}"
                )
            cases = [case for case in cases if case.prior_index in prior_indices]
        if not cases:
            raise ValueError("No cases remain after item/prior filtering")
        configuration = {
            "format_version": 1,
            "model_path": str(model_path),
            "dataset": str(dataset),
            "image_root": str(image_root) if image_root else None,
            "inference_path": str(inference_path),
            "output_dir": str(output_dir),
            "versions": versions,
            "attribution_mode": args.attribution_mode,
            "attribution_modes": list(modes),
            "analysis_modes": list(analysis_modes),
            "conditions": conditions,
            "max_items": args.max_items,
            "item_ids": sorted(item_ids) if item_ids else None,
            "prior_indices": sorted(prior_indices) if prior_indices else None,
            "skip_attention": args.skip_attention,
            "skip_layer_readout": args.skip_layer_readout,
            "max_answer_tokens": args.max_answer_tokens,
            "max_confidence_tokens": args.max_confidence_tokens,
            "max_source_tokens": args.max_source_tokens,
            "source_wire_format": "**Source Attribution**:<CLASS>",
            "source_attribution_classes": SOURCE_ATTRIBUTION_CLASSES,
            "source_attribution_midpoints": SOURCE_ATTRIBUTION_MIDPOINTS,
            "source_attribution_class_text": SOURCE_ATTRIBUTION_CLASS_TEXT,
            "reconstruction_tolerances": {
                "bfloat16": reconstruction_tolerance("bfloat16"),
                "float16": reconstruction_tolerance("float16"),
            },
            "dataset_metadata": dataset_metadata,
        }
        if config_path.exists():
            saved = json.loads(config_path.read_text(encoding="utf-8"))
            comparable_saved, missing_analysis_modes = (
                saved_configuration_for_comparison(saved)
            )
            comparable = {
                key: comparable_saved.get(key) for key in configuration
            }
            if comparable != configuration:
                raise ValueError("Resume configuration differs from saved config.json")
            if missing_analysis_modes:
                saved["analysis_modes"] = ["LMhead"]
                atomic_write_json(config_path, saved)
        else:
            atomic_write_json(
                config_path,
                {**configuration, "created_at": utc_now()},
            )
        atomic_write_json(progress_path, _progress(existing, status="initializing"))
        existing_ids = {str(record["case_id"]) for record in existing}
        expected_ids = {
            f"{case.item_id}__prior_{case.prior_index}__{condition}__{version}__{mode}"
            for case in cases
            for condition in conditions
            for version in versions
            for mode in modes
        }
        if expected_ids.issubset(existing_ids):
            _write_analyses(output_dir, existing)
            atomic_write_json(progress_path, _progress(existing, status="complete"))
            print("[INFO] No pending cases; analyses refreshed.")
            return 0

        runtime = load_runtime(inference_path)
        extended_class = build_extended_inference_class(runtime.QwenVLInference)
        inference = extended_class(model_path=str(model_path))
        modules = resolve_language_modules(inference.model)
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        text_config = getattr(
            inference.model.config,
            "text_config",
            inference.model.config,
        )
        saved_config["model_runtime"] = {
            "dtype": inference.dtype_name,
            "device_map": getattr(inference.model, "hf_device_map", None),
            "num_hidden_layers": modules.num_hidden_layers,
            "hidden_size": modules.hidden_size,
            "num_attention_heads": int(text_config.num_attention_heads),
        }
        atomic_write_json(config_path, saved_config)
        base_confidence = runtime.ConfidenceAnalyzer(
            inference,
            max_new_tokens=args.max_confidence_tokens,
        )
        confidence = MultimodalConfidenceAnalyzer(base_confidence, inference)
        source = SourceAttributionAnalyzer(
            inference,
            max_new_tokens=args.max_source_tokens,
        )
        joint = JointAnswerSourceGenerator(inference)
        patchscope_decoder = None
        if (
            not args.skip_layer_readout
            and any(mode in ("Identity", "Semantic") for mode in analysis_modes)
        ):
            patchscope_decoder = SourcePatchscopeDecoder(
                inference=inference,
                modules=modules,
                class_token_ids=source.token_specification.class_token_ids,
                analysis_modes=analysis_modes,
            )
        runners: list[V3V4SourceRunner] = []
        for mode in modes:
            runner = V3V4SourceRunner(
                inference=inference,
                modules=modules,
                confidence_analyzer=confidence,
                base_confidence_analyzer=base_confidence,
                source_analyzer=source,
                joint_generator=joint,
                confidence_classes=runtime.CONFIDENCE_CLASSES,
                confidence_midpoints=runtime.CLASS_MIDPOINTS,
                confidence_class_text=runtime.CONFIDENCE_CLASS_TEXT,
                versions=versions,
                attribution_mode=mode,
                analysis_modes=list(analysis_modes),
                patchscope_decoder=patchscope_decoder,
                conditions=conditions,
                skip_attention=args.skip_attention,
                skip_layer_readout=args.skip_layer_readout,
                max_answer_tokens=args.max_answer_tokens,
                logger=logger,
            )
            runners.append(runner)
        share_initial_cache(runners, existing)
        atomic_write_json(progress_path, _progress(existing, status="running"))

        def commit(record: dict[str, Any]) -> None:
            append_jsonl(results_path, record, fsync=True)
            existing.append(record)
            atomic_write_json(progress_path, _progress(existing, status="running"))

        counts = run_case_groups(
            runners,
            cases,
            versions=versions,
            conditions=conditions,
            existing_ids=existing_ids,
            commit=commit,
            after_group=lambda: _write_analyses(output_dir, existing),
        )
        _write_analyses(output_dir, existing)
        atomic_write_json(progress_path, _progress(existing, status="complete"))
        print(
            json.dumps(
                {
                    "status": "complete",
                    "records": len(existing),
                    "call_counts": counts,
                    "patchscope_call_counts": (
                        dict(patchscope_decoder.call_counts)
                        if patchscope_decoder is not None
                        else {}
                    ),
                    "output_dir": str(output_dir),
                },
                ensure_ascii=False,
            )
        )
        return 0
    except KeyboardInterrupt:
        try:
            records = load_jsonl(results_path, repair_trailing=True)
            _write_analyses(output_dir, records)
            atomic_write_json(progress_path, _progress(records, status="interrupted"))
        except Exception:
            logger.exception("interruption_cleanup_failed")
        print("[WARN] Interrupted after committing completed terminal cases.", file=sys.stderr)
        return 130
    except Exception as exc:
        logger.exception("experiment_failed")
        try:
            records = load_jsonl(results_path, repair_trailing=True)
            atomic_write_json(progress_path, _progress(records, status="failed"))
        except Exception:
            logger.exception("failure_progress_update_failed")
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
