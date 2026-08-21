"""Answer-fixed SA steering execution, resume, and summaries."""

from __future__ import annotations

import json
import math
import statistics
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from confidence_test.answer_metrics import normalize_answer
from confidence_test.dataset_utils import EvaluationCase, load_evaluation_cases
from confidence_test.joint_answer_source_extension import JointAnswerSourceGenerator
from confidence_test.source_attribution_variants import SourcePromptVariant
from layer_metacognition.hidden_state_store import (
    append_jsonl,
    atomic_write_json,
    atomic_write_text,
    load_jsonl,
)
from layer_metacognition.model_adapter import LanguageModules

from .artifacts import SteeringVectorRepository, direction_cosines
from .sa_steering_hook import (
    SAActivationAdditionHook,
    fixed_answer_assistant_prefix,
    locate_sa_steering_position,
)


class SteeringInvariantError(RuntimeError):
    """A systemic intervention error that must stop the full run."""


@dataclass(frozen=True)
class RuntimeCase:
    record: dict[str, Any]
    evaluation: EvaluationCase
    image_path: str
    prompt: str


def intervention_key(
    case_id: str,
    position: str,
    layer: int,
    method: str,
    direction: str,
    alpha: float,
) -> str:
    return "|".join(
        (
            str(case_id),
            f"position={position}",
            f"layer={int(layer)}",
            f"method={method}",
            f"direction={direction}",
            f"alpha={format(float(alpha), '.17g')}",
        )
    )


def alpha_zero_baseline_key(case_id: str) -> str:
    return f"{case_id}|answer_fixed_alpha_zero_baseline"


def completed_alpha_zero_by_case(
    records: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("status") != "completed":
            continue
        case_id = str(record.get("case_id", ""))
        if not case_id:
            raise ValueError("Alpha-zero baseline contains an empty case_id")
        completed[case_id] = record
    return completed


def build_runtime_cases(
    selected: Sequence[dict[str, Any]],
    *,
    dataset: str | Path,
    output_dir: str | Path,
    source_variant: SourcePromptVariant,
    image_root: str | Path | None = None,
) -> tuple[list[RuntimeCase], dict[str, Any]]:
    cases, metadata = load_evaluation_cases(
        dataset,
        fallback_null_path=Path(output_dir) / ".runtime" / "null.png",
    )
    by_key = {(case.item_id, case.prior_index): case for case in cases}
    image_root_path = Path(image_root).resolve() if image_root else None
    output: list[RuntimeCase] = []
    for record in selected:
        key = (record["item_id"], int(record["prior_index"]))
        if key not in by_key:
            raise ValueError(f"Dataset has no evaluation case for {record['case_id']}")
        case = by_key[key]
        condition = case.conditions.get(record["condition"])
        if condition is None or condition.error or not condition.resolved_image_path:
            raise ValueError(
                f"Selected case has no usable image: {record['case_id']}: "
                f"{None if condition is None else condition.error}"
            )
        image_path = Path(condition.resolved_image_path)
        if image_root_path is not None and condition.relative_image_path:
            raw = Path(condition.relative_image_path)
            image_path = raw.resolve() if raw.is_absolute() else (image_root_path / raw).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image is missing for {record['case_id']}: {image_path}")
        fixed_answer = normalize_answer(record["fixed_answer"])
        if fixed_answer not in [normalize_answer(value) for value in case.answer_classes]:
            raise ValueError(
                f"Fixed answer is outside candidate classes: {record['case_id']}: "
                f"{record['fixed_answer']!r}"
            )
        prompt = source_variant.v4_joint_prompt.format(
            question=case.question,
            text_clue=case.text_clue,
            source_classes=source_variant.class_text,
        )
        output.append(
            RuntimeCase(
                record=record,
                evaluation=case,
                image_path=str(image_path),
                prompt=prompt,
            )
        )
    return output, metadata


def _messages(prompt: str, image_path: str, assistant_text: str) -> list[dict[str, Any]]:
    resolved = Path(image_path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Image not found: {resolved}")
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(resolved)},
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_text}],
        },
    ]


def _base_result(
    runtime_case: RuntimeCase,
    *,
    position: str,
    layer: int,
    method: str,
    direction: str,
    alpha: float,
    vector_norm: float,
    normalization_hidden_norm: float,
) -> dict[str, Any]:
    source = runtime_case.record["baseline"]
    return {
        "intervention_key": intervention_key(
            runtime_case.record["case_id"], position, layer, method, direction, alpha
        ),
        "case_id": runtime_case.record["case_id"],
        "item_id": runtime_case.record["item_id"],
        "prior_index": runtime_case.record["prior_index"],
        "condition": runtime_case.record["condition"],
        "baseline_sa_group": runtime_case.record["baseline_sa_group"],
        "position": position,
        "layer": int(layer),
        "steering_type": method,
        "direction": direction,
        "alpha": float(alpha),
        "answer_fixed": True,
        "fixed_answer": runtime_case.record["fixed_answer"],
        "SA_before": float(source["soft_score"]),
        "hard_label_before": str(source["hard_label"]),
        "generated_label_before": str(source["generated_label"]),
        "SA_logits_before": list(source["class_logits"]),
        "SA_probabilities_before": list(source["class_probabilities"]),
        "entropy_before": float(source["entropy"]),
        "vector_norm": float(vector_norm),
        "normalization_hidden_norm": float(normalization_hidden_norm),
        "status": "running",
    }


def run_intervention(
    *,
    runtime_case: RuntimeCase,
    repository: SteeringVectorRepository,
    joint_generator: JointAnswerSourceGenerator,
    modules: LanguageModules,
    source_variant: SourcePromptVariant,
    position: str,
    layer: int,
    method: str,
    direction: str,
    alpha: float,
    max_source_tokens: int,
    alpha_zero_score: float | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    vector_artifact = repository.get(method, position, layer)
    sign = 1.0 if direction == "high" else -1.0
    injection = np.asarray(
        sign * float(alpha) * vector_artifact.vector,
        dtype=np.float32,
    )
    record = _base_result(
        runtime_case,
        position=position,
        layer=layer,
        method=method,
        direction=direction,
        alpha=alpha,
        vector_norm=vector_artifact.vector_norm,
        normalization_hidden_norm=vector_artifact.hidden_norm_mean,
    )
    assistant_text = fixed_answer_assistant_prefix(runtime_case.record["fixed_answer"])
    holder: dict[str, Any] = {}

    def generation_context_factory(inputs: Any, rendered: str) -> SAActivationAdditionHook:
        target_position, detail = locate_sa_steering_position(
            tokenizer=joint_generator.tokenizer,
            rendered=rendered,
            inputs=inputs,
            assistant_text=assistant_text,
            answer=runtime_case.record["fixed_answer"],
            position=position,
        )
        input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
        hook = SAActivationAdditionHook(
            modules,
            layer_index=int(layer),
            target_position=target_position,
            steering_vector=torch.from_numpy(injection),
            prefill_sequence_length=int(input_ids.shape[1]),
        )
        holder.update(
            {
                "hook": hook,
                "target_position": target_position,
                "position_detail": detail,
                "prefill_sequence_length": int(input_ids.shape[1]),
            }
        )
        return hook

    try:
        result_object = joint_generator.generate_messages(
            _messages(runtime_case.prompt, runtime_case.image_path, assistant_text),
            list(runtime_case.evaluation.answer_classes),
            assistant_text=assistant_text,
            max_new_tokens=max_source_tokens,
            use_cache=True,
            source_classes=source_variant.classes,
            source_midpoints=source_variant.midpoints,
            generation_context_factory=generation_context_factory,
        )
        result = result_object.to_dict()
        hook = holder.get("hook")
        if not isinstance(hook, SAActivationAdditionHook):
            raise SteeringInvariantError(
                f"Generation never constructed the steering hook: {result.get('error')}"
            )
        try:
            diagnostics = hook.diagnostics()
        except Exception as exc:
            raise SteeringInvariantError(
                f"Steering hook did not apply exactly once: {exc}"
            ) from exc
        if hook.h_before is None:
            raise SteeringInvariantError("Steering hook omitted its runtime hidden state")
        source = result.get("source_attribution")
        if not (
            result.get("parse_success")
            and result.get("source_metric_status") == "completed"
            and isinstance(source, dict)
            and result.get("source_label") is not None
        ):
            raise RuntimeError(f"Answer-fixed SA generation failed: {result.get('error')}")
        if result.get("answer") != runtime_case.record["fixed_answer"]:
            raise RuntimeError(
                "Teacher-forced answer changed during parsing: "
                f"expected={runtime_case.record['fixed_answer']!r}, "
                f"found={result.get('answer')!r}"
            )
        after_score = float(source["soft_image_score"])
        after_hard = str(source["hard_label"])
        after_generated = str(source["parsed_label"])
        record.update(
            {
                "SA_after": after_score,
                "hard_label_after": after_hard,
                "generated_label_after": after_generated,
                "changed": after_generated != record["generated_label_before"],
                "scored_hard_label_changed": after_hard != record["hard_label_before"],
                "SA_logits_after": [float(value) for value in source["class_logits"]],
                "SA_probabilities_after": [
                    float(value) for value in source["class_probabilities"]
                ],
                "entropy_after": float(source["source_entropy"]),
                "entropy_delta": float(source["source_entropy"])
                - float(record["entropy_before"]),
                "hidden_norm": float(torch.linalg.vector_norm(hook.h_before).item()),
                "injection_norm": float(np.linalg.norm(injection.astype(np.float64))),
                "target_position": int(holder["target_position"]),
                "position_detail": holder["position_detail"],
                "prefill_sequence_length": holder["prefill_sequence_length"],
                "hook_diagnostics": diagnostics,
                "raw_output_after": result.get("raw_output"),
                "answer_after": result.get("answer"),
                "answer_changed": False,
                "source_token_step": result.get("source_token_step"),
                "status": "completed",
            }
        )
        if alpha_zero_score is not None:
            record.update(
                {
                    "alpha_zero_SA": float(alpha_zero_score),
                    "corrected_delta_SA": after_score - float(alpha_zero_score),
                }
            )
    except SteeringInvariantError:
        raise
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "traceback": traceback.format_exc(),
            }
        )
    record["elapsed_seconds"] = float(time.perf_counter() - started)
    return record


def execute_alpha_zero_baselines(
    *,
    cases: Sequence[RuntimeCase],
    repository: SteeringVectorRepository,
    joint_generator: JointAnswerSourceGenerator,
    modules: LanguageModules,
    source_variant: SourcePromptVariant,
    position: str,
    layer: int,
    method: str,
    max_source_tokens: int,
    results_path: Path,
    progress_path: Path,
) -> list[dict[str, Any]]:
    records = load_jsonl(results_path, repair_trailing=True) if results_path.exists() else []
    completed = completed_alpha_zero_by_case(records)
    expected_case_ids = {str(runtime_case.record["case_id"]) for runtime_case in cases}
    unexpected = sorted(set(completed) - expected_case_ids)
    if unexpected:
        raise ValueError(f"Alpha-zero baselines contain unexpected cases: {unexpected[:10]}")

    def checkpoint(status: str) -> None:
        current = completed_alpha_zero_by_case(records)
        atomic_write_json(
            progress_path,
            {
                "status": status,
                "expected_case_count": len(cases),
                "attempt_count": len(records),
                "completed_case_count": len(current),
                "failed_attempt_count": sum(
                    record.get("status") == "failed" for record in records
                ),
                "remaining_case_count": len(cases) - len(current),
                "updated_at_unix": time.time(),
            },
        )

    checkpoint("running")
    for runtime_case in cases:
        case_id = str(runtime_case.record["case_id"])
        if case_id in completed:
            continue
        result = run_intervention(
            runtime_case=runtime_case,
            repository=repository,
            joint_generator=joint_generator,
            modules=modules,
            source_variant=source_variant,
            position=position,
            layer=layer,
            method=method,
            direction="high",
            alpha=0.0,
            max_source_tokens=max_source_tokens,
        )
        result.update(
            {
                "record_type": "answer_fixed_alpha_zero_baseline",
                "alpha_zero_baseline_key": alpha_zero_baseline_key(case_id),
                "canonical_position": position,
                "canonical_layer": int(layer),
                "canonical_method": method,
            }
        )
        append_jsonl(results_path, result, fsync=True)
        records.append(result)
        if result.get("status") == "completed":
            completed[case_id] = result
        checkpoint("running")

    completed = completed_alpha_zero_by_case(records)
    missing = sorted(expected_case_ids - set(completed))
    if missing:
        checkpoint("failed")
        raise RuntimeError(f"Alpha-zero baseline generation failed for cases: {missing[:10]}")
    checkpoint("complete")
    return records


def migrate_results_to_corrected_delta(
    records: Sequence[dict[str, Any]],
    alpha_zero_baselines: Sequence[dict[str, Any]],
    *,
    results_path: Path,
) -> list[dict[str, Any]]:
    """Atomically replace legacy cached-before deltas with case alpha-zero deltas."""

    alpha_zero_by_case = completed_alpha_zero_by_case(alpha_zero_baselines)
    migrated: list[dict[str, Any]] = []
    missing: set[str] = set()
    for source in records:
        record = dict(source)
        record.pop("delta_SA", None)
        case_id = str(record.get("case_id", ""))
        baseline = alpha_zero_by_case.get(case_id)
        if baseline is None:
            missing.add(case_id)
        else:
            record["alpha_zero_SA"] = float(baseline["SA_after"])
            if record.get("status") == "completed":
                record["corrected_delta_SA"] = (
                    float(record["SA_after"]) - float(baseline["SA_after"])
                )
            else:
                record.pop("corrected_delta_SA", None)
        migrated.append(record)
    if missing:
        raise ValueError(
            "Cannot migrate interventions without alpha-zero baselines: "
            f"{sorted(missing)[:10]}"
        )
    payload = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in migrated
    )
    atomic_write_text(results_path, payload)
    return migrated


def _stats(values: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in values]
    if not values:
        return {"n": 0, "mean": None, "median": None, "std": None, "sem": None}
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "std": float(std),
        "sem": float(std / math.sqrt(len(values))),
    }


def _alpha_zero_parity(records: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    selected = [
        record
        for record in records
        if record.get("status") == "completed" and float(record.get("alpha", -1)) == 0.0
    ]
    if not selected:
        return None

    def summarize(group: Sequence[dict[str, Any]]) -> dict[str, Any]:
        offsets = [
            float(record["SA_after"]) - float(record["SA_before"]) for record in group
        ]
        injection_norms = [float(record["injection_norm"]) for record in group]
        return {
            "n": len(group),
            "cached_before_offset_SA": _stats(offsets),
            "absolute_cached_before_offset_SA": _stats(
                [abs(value) for value in offsets]
            ),
            "generated_label_match_rate": (
                sum(not bool(record["changed"]) for record in group) / len(group)
            ),
            "scored_hard_label_match_rate": (
                sum(not bool(record["scored_hard_label_changed"]) for record in group)
                / len(group)
            ),
            "answer_unchanged_rate": (
                sum(not bool(record["answer_changed"]) for record in group) / len(group)
            ),
            "max_injection_norm": max(injection_norms),
        }

    case_scores: dict[str, list[float]] = {}
    for record in selected:
        case_scores.setdefault(str(record["case_id"]), []).append(float(record["SA_after"]))
    within_case_spans = [max(values) - min(values) for values in case_scores.values()]
    return {
        **summarize(selected),
        "by_position": [
            {
                "position": position,
                **summarize(
                    [record for record in selected if record["position"] == position]
                ),
            }
            for position in sorted({str(record["position"]) for record in selected})
        ],
        "case_count": len(case_scores),
        "max_within_case_SA_after_span": max(within_case_spans),
    }


def build_summary(
    records: Sequence[dict[str, Any]],
    repository: SteeringVectorRepository,
    *,
    expected_count: int,
    alpha_zero_baselines: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    completed = [record for record in records if record.get("status") == "completed"]
    failed = [record for record in records if record.get("status") == "failed"]
    steering_completed = [
        record for record in completed if "corrected_delta_SA" in record
    ]
    alpha_zero_by_case = completed_alpha_zero_by_case(alpha_zero_baselines)
    cells: list[dict[str, Any]] = []
    keys = sorted(
        {
            (
                str(record["position"]),
                int(record["layer"]),
                str(record["steering_type"]),
                str(record["direction"]),
                float(record["alpha"]),
            )
            for record in steering_completed
        }
    )
    for position, layer, method, direction, alpha in keys:
        selected = [
            record
            for record in steering_completed
            if record["position"] == position
            and int(record["layer"]) == layer
            and record["steering_type"] == method
            and record["direction"] == direction
            and float(record["alpha"]) == alpha
        ]
        corrected_deltas = [float(record["corrected_delta_SA"]) for record in selected]
        corrected_directional = sum(
            (delta > 0 if direction == "high" else delta < 0)
            for delta in corrected_deltas
        )
        self_selected = [
            record for record in selected if record["baseline_sa_group"] == direction
        ]
        self_deltas = [float(record["corrected_delta_SA"]) for record in self_selected]
        self_directional = sum(
            (delta > 0 if direction == "high" else delta < 0)
            for delta in self_deltas
        )
        cells.append(
            {
                "position": position,
                "layer": layer,
                "method": method,
                "direction": direction,
                "alpha": alpha,
                "corrected_delta_SA": _stats(corrected_deltas),
                "corrected_direction_consistency": (
                    corrected_directional / len(corrected_deltas)
                    if corrected_deltas
                    else None
                ),
                "generated_label_flip_rate": (
                    sum(bool(record["changed"]) for record in selected) / len(selected)
                    if selected
                    else None
                ),
                "scored_hard_label_flip_rate": (
                    sum(bool(record["scored_hard_label_changed"]) for record in selected)
                    / len(selected)
                    if selected
                    else None
                ),
                "self_direction": {
                    "baseline_group": direction,
                    "corrected_delta_SA": _stats(self_deltas),
                    "corrected_direction_consistency": (
                        self_directional / len(self_deltas) if self_deltas else None
                    ),
                },
            }
        )
    positions: list[dict[str, Any]] = []
    for position in repository.index["positions"]:
        selected = [
            record for record in steering_completed if record["position"] == position
        ]
        corrected = [float(record["corrected_delta_SA"]) for record in selected]
        positions.append(
            {
                "position": position,
                "record_count": len(selected),
                "absolute_corrected_delta_SA": _stats(
                    [abs(value) for value in corrected]
                ),
                "generated_label_flip_rate": (
                    sum(bool(record["changed"]) for record in selected) / len(selected)
                    if selected
                    else None
                ),
            }
        )
    summary = {
        "format_version": 1,
        "experiment": "answer_fixed_sa_hidden_state_steering",
        "answer_fixed": True,
        "expected_record_count": int(expected_count),
        "record_count": len(records),
        "completed_count": len(completed),
        "failed_count": len(failed),
        "completion_fraction": len(records) / expected_count if expected_count else 0.0,
        "steering_effectiveness": cells,
        "position_comparison": positions,
        "method_direction_cosines": direction_cosines(repository),
        "alpha_zero_baseline_completed_case_count": len(alpha_zero_by_case),
    }
    parity = _alpha_zero_parity(alpha_zero_baselines or completed)
    if parity is not None:
        summary["alpha_zero_parity"] = parity
    return summary


def initialize_output(
    output_dir: str | Path,
    configuration: dict[str, Any],
    *,
    resume: bool,
) -> tuple[list[dict[str, Any]], set[str], Path, Path, Path, Path]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"
    progress_path = output_dir / "progress.json"
    existing: list[dict[str, Any]] = []
    if config_path.exists():
        if not resume:
            raise FileExistsError(
                f"SA steering output exists; pass --resume or choose another directory: {output_dir}"
            )
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        if saved.get("config_fingerprint") != configuration["config_fingerprint"]:
            raise ValueError("Resume configuration differs from existing run_config.json")
        existing = load_jsonl(results_path, repair_trailing=True)
    elif any(path.exists() for path in (results_path, summary_path, progress_path)):
        raise FileExistsError("Protected SA steering artifacts exist without run_config.json")
    keys: set[str] = set()
    for record in existing:
        key = str(record.get("intervention_key", ""))
        if not key or key in keys:
            raise ValueError(f"Existing results contain duplicate/empty intervention key: {key!r}")
        keys.add(key)
    atomic_write_json(config_path, {**configuration, "status": "preparing"})
    return existing, keys, config_path, results_path, progress_path, summary_path


def execute_run(
    *,
    cases: Sequence[RuntimeCase],
    repository: SteeringVectorRepository,
    joint_generator: JointAnswerSourceGenerator,
    modules: LanguageModules,
    source_variant: SourcePromptVariant,
    positions: Sequence[str],
    layers: Sequence[int],
    methods: Sequence[str],
    directions: Sequence[str],
    alphas: Sequence[float],
    max_source_tokens: int,
    existing: list[dict[str, Any]],
    existing_keys: set[str],
    results_path: Path,
    progress_path: Path,
    summary_path: Path,
    alpha_zero_baselines: Sequence[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    expected = (
        len(cases)
        * len(positions)
        * len(layers)
        * len(methods)
        * len(directions)
        * len(alphas)
    )
    records = list(existing)
    alpha_zero_by_case = completed_alpha_zero_by_case(alpha_zero_baselines)

    def checkpoint(status: str) -> None:
        completed_count = sum(record.get("status") == "completed" for record in records)
        failed_count = sum(record.get("status") == "failed" for record in records)
        atomic_write_json(
            progress_path,
            {
                "status": status,
                "expected_count": expected,
                "record_count": len(records),
                "completed_count": completed_count,
                "failed_count": failed_count,
                "remaining_count": expected - len(records),
                "updated_at_unix": time.time(),
            },
        )
        atomic_write_json(
            summary_path,
            build_summary(
                records,
                repository,
                expected_count=expected,
                alpha_zero_baselines=alpha_zero_baselines,
            ),
        )

    checkpoint("running")
    since_summary = 0
    for runtime_case in cases:
        for position in positions:
            for layer in layers:
                for method in methods:
                    for direction in directions:
                        for alpha in alphas:
                            key = intervention_key(
                                runtime_case.record["case_id"],
                                position,
                                layer,
                                method,
                                direction,
                                alpha,
                            )
                            if key in existing_keys:
                                continue
                            result = run_intervention(
                                runtime_case=runtime_case,
                                repository=repository,
                                joint_generator=joint_generator,
                                modules=modules,
                                source_variant=source_variant,
                                position=position,
                                layer=layer,
                                method=method,
                                direction=direction,
                                alpha=alpha,
                                max_source_tokens=max_source_tokens,
                                alpha_zero_score=(
                                    float(
                                        alpha_zero_by_case[
                                            str(runtime_case.record["case_id"])
                                        ]["SA_after"]
                                    )
                                    if str(runtime_case.record["case_id"])
                                    in alpha_zero_by_case
                                    else None
                                ),
                            )
                            append_jsonl(results_path, result, fsync=True)
                            records.append(result)
                            existing_keys.add(key)
                            since_summary += 1
                            if since_summary >= 25 or result.get("status") == "failed":
                                checkpoint("running")
                                since_summary = 0
    final_status = (
        "complete_with_failures"
        if any(record.get("status") == "failed" for record in records)
        else "complete"
    )
    checkpoint(final_status)
    return records
