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
from confidence_test.source_attribution_variants import (  # noqa: E402
    SOURCE_PROMPT_VARIANT_ORDER,
    SourcePromptVariant,
    get_source_prompt_variant,
)
from layer_metacognition.analyze_main_results import (  # noqa: E402
    build_minimal_analysis,
    write_minimal_analysis,
)
from layer_metacognition.analyze_source_sink_results import (  # noqa: E402
    COMPACT_LAYER_COLUMNS,
    COMPACT_LAYER_COLUMNS_WITH_ANSWER_VAL,
    build_source_sink_minimal,
    build_source_sink_summary,
    group_records_by_version,
    load_case_metadata,
    split_analysis_by_version,
    write_layer_readout_minimal,
    write_source_sink_minimal,
)
from layer_metacognition.hidden_state_store import (  # noqa: E402
    TargetLayerHiddenStateStore,
    append_jsonl,
    atomic_write_json,
    load_jsonl,
)
from layer_metacognition.model_adapter import resolve_language_modules  # noqa: E402
from layer_metacognition.probability_tables import write_probability_tables  # noqa: E402
from layer_metacognition.source_patchscope import (  # noqa: E402
    ANSWER_PATCHSCOPE_SHUFFLE_SEEDS,
    ANSWER_PATCHSCOPE_VARIANTS,
    AnswerPatchscopeDecoder,
    SourcePatchscopeDecoder,
)
from layer_metacognition.v3_v4_source_runner import (  # noqa: E402
    V3V4SourceRunner,
    reconstruction_tolerance,
)


DEFAULT_MODEL_PATH = ROOT / "qwen-2.5-vl" / "models" / "Qwen2.5-VL-7B-Instruct"
DEFAULT_DATASET_PATH = ROOT / "datasets" / "datasets.json"
DEFAULT_OUTPUT_DIR = ROOT / "layer_metacognition" / "output" / "v3_v4_source"
ATTRIBUTION_MODES = ("none", "parallel", "joint")
ANALYSIS_MODE_ORDER = ("LMhead", "Identity", "Semantic")
HIDDEN_STATE_POSITION_ORDER = (
    "ac",
    "lat",
    "panl",
    "ltt",
    "ptnl",
    "pit",
    "sac",
)
FORMAT_VERSION = 2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def expand_attribution_modes(value: str) -> tuple[str, ...]:
    """Expand the CLI's aggregate mode while keeping a stable execution order."""
    if value == "all":
        return ATTRIBUTION_MODES
    if value not in ATTRIBUTION_MODES:
        raise ValueError(f"Unknown attribution mode: {value}")
    return (value,)


def normalize_source_prompt_variants(values: Iterable[str]) -> tuple[str, ...]:
    """Return selected prompt variants in stable experiment order."""
    raw = [str(value) for value in values]
    invalid = [value for value in raw if value not in SOURCE_PROMPT_VARIANT_ORDER]
    if invalid:
        raise ValueError(
            "Unknown --source-prompt-variant value(s): " + ", ".join(invalid)
        )
    if not raw:
        raise ValueError("--source-prompt-variant requires at least one value")
    selected = set(raw)
    return tuple(
        value for value in SOURCE_PROMPT_VARIANT_ORDER if value in selected
    )


def validate_source_prompt_variant_modes(
    source_prompt_variants: Iterable[str],
    attribution_mode: str,
) -> None:
    joint_only = [
        value for value in source_prompt_variants if value != "baseline"
    ]
    if joint_only and attribution_mode != "joint":
        raise ValueError(
            "Source prompt variant(s) "
            f"{', '.join(joint_only)} require --attribution-mode joint; "
            f"got {attribution_mode!r}"
        )


def source_variant_output_dir(output_root: Path, variant: str) -> Path:
    if variant not in SOURCE_PROMPT_VARIANT_ORDER:
        raise ValueError(f"Unknown source prompt variant: {variant}")
    return output_root / variant


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


def parse_save_hidden_state(value: str) -> int | None:
    """Parse ``none`` or one non-negative, zero-based decoder layer."""
    normalized = str(value).strip().lower()
    if normalized == "none":
        return None
    try:
        layer_index = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--save_hidden_state must be 'none' or a non-negative integer"
        ) from exc
    if layer_index < 0:
        raise argparse.ArgumentTypeError(
            "--save_hidden_state must be 'none' or a non-negative integer"
        )
    return layer_index


def normalize_save_hidden_states(
    values: Iterable[int | None],
) -> tuple[int, ...]:
    raw = list(values)
    if any(value is None for value in raw):
        if len(raw) != 1:
            raise ValueError(
                "--save_hidden_state cannot combine 'none' with layer indices"
            )
        return ()
    layers = [int(value) for value in raw if value is not None]
    duplicates = sorted({value for value in layers if layers.count(value) > 1})
    if duplicates:
        raise ValueError(
            "--save_hidden_state contains duplicate layer(s): "
            + ", ".join(str(value) for value in duplicates)
        )
    return tuple(sorted(layers))


def serialize_save_hidden_states(layer_indices: tuple[int, ...]) -> str | int | list[int]:
    if not layer_indices:
        return "none"
    if len(layer_indices) == 1:
        return layer_indices[0]
    return list(layer_indices)


def normalize_hidden_state_positions(values: Iterable[str]) -> tuple[str, ...]:
    raw = [str(value).strip().lower() for value in values]
    invalid = [value for value in raw if value not in HIDDEN_STATE_POSITION_ORDER]
    if invalid:
        raise ValueError(
            "Unknown --save_hidden_state_positions value(s): " + ", ".join(invalid)
        )
    selected = set(raw)
    return tuple(
        name for name in HIDDEN_STATE_POSITION_ORDER if name in selected
    )


def validate_save_hidden_state(
    layer_indices: tuple[int, ...],
    *,
    skip_layer_readout: bool,
    num_hidden_layers: int | None = None,
) -> None:
    if not layer_indices:
        return
    del skip_layer_readout
    invalid = (
        []
        if num_hidden_layers is None
        else [value for value in layer_indices if value >= num_hidden_layers]
    )
    if invalid:
        raise ValueError(
            "--save_hidden_state layer(s) "
            f"{invalid} outside [0, {num_hidden_layers - 1}]"
        )


def saved_configuration_for_comparison(
    saved: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Interpret pre-analysis-mode configs as the old LMhead-only default."""
    normalized = dict(saved)
    missing = "analysis_modes" not in normalized
    if missing:
        normalized["analysis_modes"] = ["LMhead"]
    normalized.setdefault("save_hidden_state", "none")
    normalized.setdefault("save_hidden_state_positions", ["ac", "panl"])
    normalized.setdefault("save_probtable", False)
    normalized.setdefault("skip_confidence", False)
    return normalized, missing


def resume_configuration_differences(
    saved: dict[str, Any],
    current: dict[str, Any],
) -> tuple[list[str], bool]:
    """Return differing persisted fields and whether legacy analysis mode was inferred."""
    comparable_saved, missing_analysis_modes = saved_configuration_for_comparison(
        saved
    )
    differing = [
        key for key, value in current.items() if comparable_saved.get(key) != value
    ]
    return differing, missing_analysis_modes


def validate_resume_format(saved: dict[str, Any]) -> None:
    """Reject outputs that cannot contain Semantic Answer Patchscope rows."""
    if saved.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            "Output directory uses the legacy layer-readout format and cannot "
            f"be resumed with format version {FORMAT_VERSION}; choose a new "
            "--output-dir and rerun"
        )


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


def run_prompt_variant_case_groups(
    variant_runs: list[dict[str, Any]],
    cases: list[Any],
    *,
    versions: list[str],
    conditions: list[str],
) -> dict[str, dict[str, dict[str, int]]]:
    """Finish each prompt variant for one terminal case before advancing."""
    for case in cases:
        for condition in conditions:
            for version in versions:
                for variant_run in variant_runs:
                    variant_run["active"] = True
                    group_changed = False
                    for runner in variant_run["runners"]:
                        case_id = (
                            f"{case.item_id}__prior_{case.prior_index}__"
                            f"{condition}__{version}__{runner.mode}"
                        )
                        if case_id in variant_run["existing_ids"]:
                            runner._log("resume_skip", case_id=case_id)
                            continue
                        record = runner.process_case(
                            case=case,
                            condition=condition,
                            version=version,
                        )
                        variant_run["commit"](record)
                        variant_run["existing_ids"].add(case_id)
                        group_changed = True
                    if group_changed:
                        variant_run["after_group"]()
                    variant_run["active"] = False
    return {
        str(variant_run["name"]): {
            runner.mode: dict(runner.call_counts)
            for runner in variant_run["runners"]
        }
        for variant_run in variant_runs
    }


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


def configure_logger(path: Path, *, source_prompt_variant: str | None = None) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger_name = "layer_metacognition.v3_v4_source"
    if source_prompt_variant is not None:
        logger_name = f"{logger_name}.{source_prompt_variant}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    for existing_handler in logger.handlers:
        existing_handler.close()
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


def _write_variant_outputs(
    state: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    _write_analyses(state["output_dir"], records)
    if state.get("save_probtable"):
        write_probability_tables(
            state["output_dir"] / "probability_tables.json",
            records,
            source_classes=state["source_variant"].classes,
            confidence_classes=state.get("confidence_classes"),
        )


def _checkpoint_variant_runs_after_stop(
    variant_runs: list[dict[str, Any]],
    *,
    status: str,
) -> set[str]:
    """Flush every interleaved variant so earlier completed cases are durable."""
    handled: set[str] = set()
    for variant_run in variant_runs:
        state = variant_run["state"]
        store = state.get("hidden_state_store")
        if store is not None:
            store.flush(state["results_path"])
        records = load_jsonl(state["results_path"], repair_trailing=True)
        _write_variant_outputs(state, records)
        atomic_write_json(
            state["progress_path"],
            _progress(records, status=status),
        )
        handled.add(str(state["name"]))
    return handled


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
        "--source-prompt-variant",
        nargs="+",
        choices=list(SOURCE_PROMPT_VARIANT_ORDER),
        default=["baseline"],
        help=(
            "One or more joint Source Attribution prompt variants. Each variant "
            "writes to its own child directory under --output-dir."
        ),
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
    parser.add_argument(
        "--save_probtable",
        action="store_true",
        help="Write per-layer restricted class probabilities to probability_tables.json.",
    )
    parser.add_argument(
        "--skip_confidence",
        action="store_true",
        help=(
            "Skip final confidence generation and CC readout while retaining "
            "the V3 initial confidence required by its answer prompt."
        ),
    )
    parser.add_argument(
        "--answer_val",
        action="store_true",
        help=(
            "Run original plus three deterministic shuffled Semantic Answer "
            "Patchscope target orders and summarize pairwise robustness."
        ),
    )
    parser.add_argument(
        "--save_hidden_state",
        type=parse_save_hidden_state,
        nargs="+",
        default=[None],
        metavar="none|LAYER",
        help=(
            "Save one or more zero-based decoder layers' selected-position "
            "hidden states as CPU FP16 shards; default: none."
        ),
    )
    parser.add_argument(
        "--save_hidden_state_positions",
        nargs="+",
        choices=list(HIDDEN_STATE_POSITION_ORDER),
        default=["ac", "panl"],
        help=(
            "Hidden-state positions to save. Duplicates are removed and output "
            "order is always ac, lat, panl, ltt, ptnl, pit, sac. "
            "(default: ac panl)"
        ),
    )
    parser.add_argument("--max-answer-tokens", type=int, default=24)
    parser.add_argument("--max-confidence-tokens", type=int, default=12)
    parser.add_argument("--max-source-tokens", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    positions_explicit = any(
        value == "--save_hidden_state_positions"
        or value.startswith("--save_hidden_state_positions=")
        for value in raw_argv
    )
    args = parser.parse_args(raw_argv)
    try:
        versions = _canonical_values(args.versions, ("v3", "v4"), "version")
        source_prompt_variants = normalize_source_prompt_variants(
            args.source_prompt_variant
        )
        validate_source_prompt_variant_modes(
            source_prompt_variants,
            args.attribution_mode,
        )
        modes = expand_attribution_modes(args.attribution_mode)
        analysis_modes = normalize_analysis_modes(args.analysis_mode)
        save_hidden_states = normalize_save_hidden_states(args.save_hidden_state)
        save_hidden_state_positions = normalize_hidden_state_positions(
            args.save_hidden_state_positions
        )
        if positions_explicit and not save_hidden_states:
            raise ValueError(
                "--save_hidden_state_positions was specified but "
                "--save_hidden_state selected no layer"
            )
        if "sac" in save_hidden_state_positions and "none" in modes:
            raise ValueError(
                "--save_hidden_state_positions sac requires attribution mode "
                "parallel or joint; mode 'none' has no SAC token"
            )
        conditions = _canonical_values(args.conditions, CONDITIONS, "condition")
        item_ids = _parse_string_selection(args.item_ids)
        prior_indices = set(args.prior_indices) if args.prior_indices else None
        if prior_indices is not None and any(index < 0 for index in prior_indices):
            raise ValueError("--prior-indices must be non-negative")
        validate_save_hidden_state(
            save_hidden_states,
            skip_layer_readout=args.skip_layer_readout,
        )
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
        output_root = Path(args.output_dir).resolve()
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

    output_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("layer_metacognition.v3_v4_source")
    variant_states: list[dict[str, Any]] = []
    active_state: dict[str, Any] | None = None
    states_initialized = False
    try:
        cases, dataset_metadata = load_evaluation_cases(
            dataset,
            item_limit=args.max_items,
            fallback_null_path=output_root / ".runtime" / "null.png",
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

        # Preflight every variant before creating or updating any variant files.
        for variant_name in source_prompt_variants:
            source_variant = get_source_prompt_variant(variant_name)
            output_dir = source_variant_output_dir(output_root, variant_name)
            results_path = output_dir / "results.jsonl"
            config_path = output_dir / "config.json"
            progress_path = output_dir / "progress.json"
            existing = load_jsonl(results_path, repair_trailing=args.resume)
            if existing and not args.resume:
                raise ValueError(
                    "Output already contains results; pass --resume or choose a "
                    f"new directory: {results_path}"
                )
            configuration = {
                "format_version": FORMAT_VERSION,
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
                "save_probtable": args.save_probtable,
                "skip_confidence": args.skip_confidence,
                "answer_val": args.answer_val,
                "save_hidden_state": serialize_save_hidden_states(save_hidden_states),
                "save_hidden_state_positions": list(save_hidden_state_positions),
                "hidden_state_position_collision_policy": (
                    "shift_ltt_to_previous_token_before_ptnl"
                    if {"ltt", "ptnl"}.issubset(save_hidden_state_positions)
                    else None
                ),
                "max_answer_tokens": args.max_answer_tokens,
                "max_confidence_tokens": args.max_confidence_tokens,
                "max_source_tokens": args.max_source_tokens,
                "source_prompt_variant": source_variant.name,
                "source_wire_format": "**Source Attribution**:<CLASS>",
                "source_attribution_classes": list(source_variant.classes),
                "source_attribution_midpoints": list(source_variant.midpoints),
                "source_attribution_class_text": source_variant.class_text,
                "compact_layer_columns": list(
                    COMPACT_LAYER_COLUMNS_WITH_ANSWER_VAL
                    if args.answer_val
                    else COMPACT_LAYER_COLUMNS
                ),
                "answer_patchscope": {
                    "analysis_mode": "Semantic",
                    "target_context": "candidate_answers_only",
                    "wire_format": "**Answer**:<ANSWER>",
                    "probability_definition": "restricted_candidate_softmax_top1",
                    "validation_enabled": args.answer_val,
                    "validation_variants": (
                        list(ANSWER_PATCHSCOPE_VARIANTS)
                        if args.answer_val
                        else ["original"]
                    ),
                    "shuffle_seeds": (
                        dict(ANSWER_PATCHSCOPE_SHUFFLE_SEEDS)
                        if args.answer_val
                        else {}
                    ),
                },
                "reconstruction_tolerances": {
                    "bfloat16": reconstruction_tolerance("bfloat16"),
                    "float16": reconstruction_tolerance("float16"),
                },
                "dataset_metadata": dataset_metadata,
            }
            config_to_write: dict[str, Any]
            if config_path.exists():
                saved = json.loads(config_path.read_text(encoding="utf-8"))
                validate_resume_format(saved)
                differing, missing_analysis_modes = resume_configuration_differences(
                    saved,
                    configuration,
                )
                if differing:
                    raise ValueError(
                        f"{variant_name}: resume configuration differs from "
                        "saved config.json for: " + ", ".join(differing)
                    )
                config_to_write = dict(saved)
                if missing_analysis_modes:
                    config_to_write["analysis_modes"] = ["LMhead"]
                config_to_write.setdefault(
                    "save_hidden_state_positions", ["ac", "panl"]
                )
                config_to_write.setdefault("save_probtable", False)
                config_to_write.setdefault("skip_confidence", False)
            else:
                config_to_write = {**configuration, "created_at": utc_now()}
            existing_ids = {str(record["case_id"]) for record in existing}
            expected_ids = {
                f"{case.item_id}__prior_{case.prior_index}__{condition}__{version}__{mode}"
                for case in cases
                for condition in conditions
                for version in versions
                for mode in modes
            }
            variant_states.append(
                {
                    "name": variant_name,
                    "source_variant": source_variant,
                    "output_dir": output_dir,
                    "results_path": results_path,
                    "config_path": config_path,
                    "progress_path": progress_path,
                    "configuration": configuration,
                    "config_to_write": config_to_write,
                    "existing": existing,
                    "existing_ids": existing_ids,
                    "expected_ids": expected_ids,
                    "complete": expected_ids.issubset(existing_ids),
                    "call_counts": {},
                    "patchscope_call_counts": {},
                    "save_probtable": args.save_probtable,
                    "confidence_classes": None,
                }
            )

        for state in variant_states:
            state["output_dir"].mkdir(parents=True, exist_ok=True)
            atomic_write_json(state["config_path"], state["config_to_write"])
            atomic_write_json(
                state["progress_path"],
                _progress(state["existing"], status="initializing"),
            )
        states_initialized = True

        pending_states = [state for state in variant_states if not state["complete"]]
        if not pending_states:
            for state in variant_states:
                _write_variant_outputs(state, state["existing"])
                atomic_write_json(
                    state["progress_path"],
                    _progress(state["existing"], status="complete"),
                )
            print("[INFO] No pending cases; analyses refreshed for all prompt variants.")
            return 0

        runtime = load_runtime(inference_path)
        extended_class = build_extended_inference_class(runtime.QwenVLInference)
        inference = extended_class(model_path=str(model_path))
        modules = resolve_language_modules(inference.model)
        validate_save_hidden_state(
            save_hidden_states,
            skip_layer_readout=args.skip_layer_readout,
            num_hidden_layers=modules.num_hidden_layers,
        )
        text_config = getattr(
            inference.model.config,
            "text_config",
            inference.model.config,
        )
        model_runtime = {
            "dtype": inference.dtype_name,
            "device_map": getattr(inference.model, "hf_device_map", None),
            "num_hidden_layers": modules.num_hidden_layers,
            "hidden_size": modules.hidden_size,
            "num_attention_heads": int(text_config.num_attention_heads),
        }
        for state in variant_states:
            saved_config = json.loads(
                state["config_path"].read_text(encoding="utf-8")
            )
            saved_config["model_runtime"] = model_runtime
            atomic_write_json(state["config_path"], saved_config)
        base_confidence = runtime.ConfidenceAnalyzer(
            inference,
            max_new_tokens=args.max_confidence_tokens,
        )
        confidence = MultimodalConfidenceAnalyzer(base_confidence, inference)
        for state in variant_states:
            state["confidence_classes"] = list(runtime.CONFIDENCE_CLASSES)
        joint = JointAnswerSourceGenerator(inference)
        answer_patchscope_decoder = (
            None
            if args.skip_layer_readout
            else AnswerPatchscopeDecoder(
                inference=inference,
                modules=modules,
            )
        )

        variant_runs: list[dict[str, Any]] = []
        for state in variant_states:
            if state["complete"]:
                _write_variant_outputs(state, state["existing"])
                atomic_write_json(
                    state["progress_path"],
                    _progress(state["existing"], status="complete"),
                )
                continue
            source_variant: SourcePromptVariant = state["source_variant"]
            output_dir: Path = state["output_dir"]
            existing: list[dict[str, Any]] = state["existing"]
            variant_logger = configure_logger(
                output_dir / "run.log",
                source_prompt_variant=source_variant.name,
            )
            variant_logger.info(
                "model_loaded model_instances=1 processor_instances=1 source_prompt_variant=%s",
                source_variant.name,
            )
            source = SourceAttributionAnalyzer(
                inference,
                max_new_tokens=args.max_source_tokens,
                source_classes=source_variant.classes,
                source_midpoints=source_variant.midpoints,
            )
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
                    source_classes=source_variant.classes,
                    source_midpoints=source_variant.midpoints,
                    source_class_text=source_variant.class_text,
                )
            runners: list[V3V4SourceRunner] = []
            for mode in modes:
                runners.append(
                    V3V4SourceRunner(
                        inference=inference,
                        modules=modules,
                        confidence_analyzer=confidence,
                        base_confidence_analyzer=base_confidence,
                        source_analyzer=source,
                        joint_generator=joint,
                        source_prompt_variant=source_variant,
                        confidence_classes=runtime.CONFIDENCE_CLASSES,
                        confidence_midpoints=runtime.CLASS_MIDPOINTS,
                        confidence_class_text=runtime.CONFIDENCE_CLASS_TEXT,
                        versions=versions,
                        attribution_mode=mode,
                        analysis_modes=list(analysis_modes),
                        patchscope_decoder=patchscope_decoder,
                        answer_patchscope_decoder=answer_patchscope_decoder,
                        answer_val=args.answer_val,
                        save_hidden_state=list(save_hidden_states),
                        conditions=conditions,
                        save_hidden_state_positions=list(save_hidden_state_positions),
                        skip_attention=args.skip_attention,
                        skip_layer_readout=args.skip_layer_readout,
                        skip_confidence=args.skip_confidence,
                        max_answer_tokens=args.max_answer_tokens,
                        logger=variant_logger,
                    )
                )
            share_initial_cache(runners, existing)
            atomic_write_json(
                state["progress_path"],
                _progress(existing, status="running"),
            )
            state["hidden_state_store"] = (
                None
                if not save_hidden_states
                else TargetLayerHiddenStateStore(
                    output_dir=output_dir,
                    layer_index=list(save_hidden_states),
                    position_names=list(save_hidden_state_positions),
                )
            )
            state["patchscope_decoder"] = patchscope_decoder
            state["logger"] = variant_logger
            state["active"] = False

            def build_commit(target_state: dict[str, Any]) -> Callable[[dict[str, Any]], None]:
                def commit(record: dict[str, Any]) -> None:
                    target_hidden_state = record.pop("_target_hidden_state", None)
                    target_state["existing"].append(record)
                    store = target_state["hidden_state_store"]
                    if store is not None and record.get("status") == "completed":
                        if target_hidden_state is None:
                            raise RuntimeError(
                                "Completed case is missing requested hidden state"
                            )
                        if store.add(
                            case_id=str(record["case_id"]),
                            hidden_states=target_hidden_state,
                            positions=record["token_positions"],
                            stages=record["token_position_stages"],
                            result=record,
                        ):
                            store.flush(target_state["results_path"])
                    else:
                        if target_hidden_state is not None:
                            raise RuntimeError(
                                "Unexpected target hidden state without an active store"
                            )
                        append_jsonl(
                            target_state["results_path"],
                            record,
                            fsync=True,
                        )
                    atomic_write_json(
                        target_state["progress_path"],
                        _progress(target_state["existing"], status="running"),
                    )

                return commit

            variant_runs.append(
                {
                    "name": state["name"],
                    "runners": runners,
                    "existing_ids": state["existing_ids"],
                    "commit": build_commit(state),
                    "after_group": lambda target_state=state: _write_variant_outputs(
                        target_state,
                        target_state["existing"],
                    ),
                    "active": False,
                    "state": state,
                }
            )

        counts_by_variant = run_prompt_variant_case_groups(
            variant_runs,
            cases,
            versions=versions,
            conditions=conditions,
        )
        for variant_run in variant_runs:
            state = variant_run["state"]
            store = state["hidden_state_store"]
            if store is not None:
                store.flush(state["results_path"])
            _write_variant_outputs(state, state["existing"])
            atomic_write_json(
                state["progress_path"],
                _progress(state["existing"], status="complete"),
            )
            state["call_counts"] = counts_by_variant[state["name"]]
            patchscope_decoder = state["patchscope_decoder"]
            state["patchscope_call_counts"] = (
                dict(patchscope_decoder.call_counts)
                if patchscope_decoder is not None
                else {}
            )
            state["complete"] = True
            state["hidden_state_store"] = None

        print(
            json.dumps(
                {
                    "status": "complete",
                    "records": {
                        state["name"]: len(state["existing"])
                        for state in variant_states
                    },
                    "call_counts": {
                        state["name"]: state["call_counts"]
                        for state in variant_states
                    },
                    "answer_patchscope_call_counts": (
                        dict(answer_patchscope_decoder.call_counts)
                        if answer_patchscope_decoder is not None
                        else {}
                    ),
                    "patchscope_call_counts": (
                        {
                            state["name"]: state["patchscope_call_counts"]
                            for state in variant_states
                        }
                    ),
                    "output_dirs": {
                        state["name"]: str(state["output_dir"])
                        for state in variant_states
                    },
                },
                ensure_ascii=False,
            )
        )
        return 0
    except KeyboardInterrupt:
        try:
            active_run = next(
                (
                    variant_run
                    for variant_run in locals().get("variant_runs", [])
                    if variant_run.get("active")
                ),
                None,
            )
            if active_run is not None:
                active_state = active_run["state"]
            handled = _checkpoint_variant_runs_after_stop(
                locals().get("variant_runs", []),
                status="interrupted",
            )
            if states_initialized:
                for state in variant_states:
                    if state["name"] not in handled and not state.get("complete"):
                        atomic_write_json(
                            state["progress_path"],
                            _progress(state["existing"], status="interrupted"),
                        )
        except Exception:
            cleanup_logger = (
                active_state.get("logger", logger)
                if active_state is not None
                else logger
            )
            cleanup_logger.exception("interruption_cleanup_failed")
        print("[WARN] Interrupted after committing completed terminal cases.", file=sys.stderr)
        return 130
    except Exception as exc:
        active_run = next(
            (
                variant_run
                for variant_run in locals().get("variant_runs", [])
                if variant_run.get("active")
            ),
            None,
        )
        if active_run is not None:
            active_state = active_run["state"]
        failure_logger = (
            active_state.get("logger", logger)
            if active_state is not None
            else logger
        )
        failure_logger.exception("experiment_failed")
        try:
            handled = _checkpoint_variant_runs_after_stop(
                locals().get("variant_runs", []),
                status="failed",
            )
            if states_initialized:
                for state in variant_states:
                    if (
                        state["name"] not in handled
                        and not state.get("complete")
                    ):
                        atomic_write_json(
                            state["progress_path"],
                            _progress(state["existing"], status="failed"),
                        )
        except Exception:
            failure_logger.exception("failure_progress_update_failed")
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
