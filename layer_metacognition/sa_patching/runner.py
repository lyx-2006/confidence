"""Execution, metrics, failure isolation, and summary for SA activation patching."""

from __future__ import annotations

import math
import statistics
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch

from confidence_test.joint_answer_source_extension import JointAnswerSourceGenerator
from confidence_test.source_attribution_variants import SourcePromptVariant
from layer_metacognition.model_adapter import LanguageModules
from layer_metacognition.sa_steering.runner import RuntimeCase
from layer_metacognition.sa_steering.sa_steering_hook import fixed_answer_assistant_prefix

from .artifacts import EmbeddingArtifacts
from .protocol import fixed_messages, locate_fixed_inputs
from .sa_patching_hook import (
    ActivationReplacementHook,
    CombinedHooks,
    EmbeddingReplacement,
    EmbeddingReplacementHook,
    PatchingInvariantError,
    ResidualActivationCacheHook,
    resolve_language_model,
)


CORRUPTION_COMPONENTS = {
    "image_only": ("image",),
    "text_only": ("text",),
    "image_text": ("image", "text"),
    "answer_only": ("answer",),
    "all": ("image", "text", "answer"),
}


def intervention_key(case_id: str, corruption: str, position: str, layer: int) -> str:
    return (
        f"{case_id}|corruption={corruption}|position={position}|layer={int(layer)}"
    )


def position_layer_grid(
    positions: Sequence[str],
    layers: Sequence[int],
    allowed: Mapping[str, Sequence[int]],
) -> list[tuple[str, int]]:
    selected_layers = {int(value) for value in layers}
    output = [
        (position, int(layer))
        for position in positions
        for layer in allowed[position]
        if int(layer) in selected_layers
    ]
    if not output:
        raise ValueError("Requested positions/layers produce an empty patching grid")
    return output


def _source_payload(result: dict[str, Any]) -> dict[str, Any]:
    source = result.get("source_attribution")
    if not (
        result.get("parse_success")
        and result.get("source_metric_status") == "completed"
        and isinstance(source, dict)
        and result.get("source_label") is not None
    ):
        raise RuntimeError(f"SA generation failed: {result.get('error')}")
    required = ("hard_label", "soft_image_score", "class_logits", "class_probabilities")
    if any(source.get(key) is None for key in required):
        raise RuntimeError("SA scorer returned an incomplete payload")
    logits = [float(value) for value in source["class_logits"]]
    probabilities = [float(value) for value in source["class_probabilities"]]
    if len(logits) != 9 or len(probabilities) != 9:
        raise RuntimeError("SA scorer did not return nine classes")
    return {
        "parsed_label": str(result["source_label"]),
        "hard_label": str(source["hard_label"]),
        "soft_score": float(source["soft_image_score"]),
        "logits": logits,
        "probabilities": probabilities,
        "entropy": float(source["source_entropy"]),
        "raw_output": result.get("raw_output"),
    }


def logit_difference(logits: Sequence[float], clean_class: str) -> float:
    index = int(clean_class)
    if index < 0 or index >= len(logits) or len(logits) < 2:
        raise ValueError("Invalid clean SA class for logit difference")
    others = [float(value) for local, value in enumerate(logits) if local != index]
    return float(logits[index]) - sum(others) / len(others)


def recovery_value(clean: float, corrupt: float, patched: float) -> float | None:
    denominator = float(clean) - float(corrupt)
    if not all(math.isfinite(value) for value in (clean, corrupt, patched)):
        return None
    if abs(denominator) <= 1e-8:
        return None
    return (float(patched) - float(corrupt)) / denominator


def metric_bundle(
    clean: dict[str, Any],
    corrupt: dict[str, Any],
    patched: dict[str, Any],
) -> dict[str, Any]:
    clean_logit = logit_difference(clean["logits"], clean["hard_label"])
    corrupt_logit = logit_difference(corrupt["logits"], clean["hard_label"])
    patched_logit = logit_difference(patched["logits"], clean["hard_label"])
    clean_hard = float(int(clean["hard_label"]))
    corrupt_hard = float(int(corrupt["hard_label"]))
    patched_hard = float(int(patched["hard_label"]))
    return {
        "soft_delta": float(patched["soft_score"] - corrupt["soft_score"]),
        "logit_difference": {
            "clean": clean_logit,
            "corrupt": corrupt_logit,
            "patched": patched_logit,
        },
        "recovery": {
            "soft": recovery_value(
                clean["soft_score"], corrupt["soft_score"], patched["soft_score"]
            ),
            "logit": recovery_value(clean_logit, corrupt_logit, patched_logit),
            "hard_formula": recovery_value(clean_hard, corrupt_hard, patched_hard),
            "hard_match_recovered": patched["hard_label"] == clean["hard_label"],
            "corrupt_hard_differs": corrupt["hard_label"] != clean["hard_label"],
        },
    }


@dataclass
class FixedRun:
    result: dict[str, Any]
    source: dict[str, Any]
    prepared: Any
    embedding_hook: EmbeddingReplacementHook
    cache_hook: ResidualActivationCacheHook | None
    patch_hook: ActivationReplacementHook | None


class SAPatchingRunner:
    def __init__(
        self,
        *,
        joint_generator: JointAnswerSourceGenerator,
        modules: LanguageModules,
        source_variant: SourcePromptVariant,
        artifacts: EmbeddingArtifacts,
        grid: Sequence[tuple[str, int]],
        corruptions: Sequence[str],
        max_source_tokens: int,
        parity_tolerance: float,
    ) -> None:
        self.joint = joint_generator
        self.modules = modules
        self.source_variant = source_variant
        self.artifacts = artifacts
        self.grid = list(grid)
        self.corruptions = list(corruptions)
        self.max_source_tokens = int(max_source_tokens)
        self.parity_tolerance = float(parity_tolerance)
        self.language_model = resolve_language_model(self.joint.model)
        self.positions = tuple(dict.fromkeys(position for position, _layer in self.grid))
        self.no_patch_validation: dict[str, Any] = {}

    def _replacements(self, prepared: Any, corruption: str) -> list[EmbeddingReplacement]:
        if corruption not in CORRUPTION_COMPONENTS:
            raise ValueError(f"Unknown corruption: {corruption}")
        output: list[EmbeddingReplacement] = []
        for name in CORRUPTION_COMPONENTS[corruption]:
            span = prepared.spans[name]
            if name == "image":
                source = self.artifacts.image
            elif name == "text":
                length = len(span)
                if length > int(self.artifacts.text.shape[0]):
                    raise PatchingInvariantError(
                        f"Text token span {length} exceeds ragged mean length "
                        f"{self.artifacts.text.shape[0]}"
                    )
                source = self.artifacts.text[:length]
            else:
                source = self.artifacts.answer
            if tuple(source.shape) != (len(span), self.modules.hidden_size):
                raise PatchingInvariantError(
                    f"{name} mean embedding shape mismatch: {tuple(source.shape)} "
                    f"vs {(len(span), self.modules.hidden_size)}"
                )
            output.append(EmbeddingReplacement(name, span, source))
        return output

    def _cache_targets(self, prepared: Any) -> dict[int, dict[str, int]]:
        targets: dict[int, dict[str, int]] = {}
        for position, layer in self.grid:
            targets.setdefault(layer, {})[position] = prepared.positions[position]
        return targets

    def _run_fixed(
        self,
        runtime_case: RuntimeCase,
        answer: str,
        *,
        corruption: str | None,
        capture_cache: bool,
        patch: tuple[str, int, torch.Tensor] | None = None,
    ) -> FixedRun:
        assistant_text = fixed_answer_assistant_prefix(answer)
        messages = fixed_messages(runtime_case.prompt, runtime_case.image_path, assistant_text)
        holder: dict[str, Any] = {}

        def context_factory(inputs: Any, rendered: str) -> CombinedHooks:
            prepared = locate_fixed_inputs(
                self.joint,
                runtime_case,
                answer,
                messages=messages,
                assistant_text=assistant_text,
                rendered=rendered,
                inputs=inputs,
                positions=self.positions,
            )
            input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
            sequence_length = int(input_ids.shape[1])
            replacements = [] if corruption is None else self._replacements(prepared, corruption)
            embedding_hook = EmbeddingReplacementHook(
                self.language_model,
                replacements=replacements,
                prefill_sequence_length=sequence_length,
                hidden_size=self.modules.hidden_size,
                capture_clean=corruption is None,
            )
            hooks: list[Any] = [embedding_hook]
            cache_hook = None
            if capture_cache:
                cache_hook = ResidualActivationCacheHook(
                    self.modules,
                    targets=self._cache_targets(prepared),
                    prefill_sequence_length=sequence_length,
                )
                hooks.append(cache_hook)
            patch_hook = None
            if patch is not None:
                patch_position, patch_layer, source_hidden = patch
                patch_hook = ActivationReplacementHook(
                    self.modules,
                    layer=patch_layer,
                    position=prepared.positions[patch_position],
                    source_hidden=source_hidden,
                    prefill_sequence_length=sequence_length,
                )
                hooks.append(patch_hook)
            holder.update(
                {
                    "prepared": prepared,
                    "embedding_hook": embedding_hook,
                    "cache_hook": cache_hook,
                    "patch_hook": patch_hook,
                }
            )
            return CombinedHooks(*hooks)

        result = self.joint.generate_messages(
            messages,
            list(runtime_case.evaluation.answer_classes),
            assistant_text=assistant_text,
            max_new_tokens=self.max_source_tokens,
            use_cache=True,
            source_classes=self.source_variant.classes,
            source_midpoints=self.source_variant.midpoints,
            generation_context_factory=context_factory,
        ).to_dict()
        embedding_hook = holder.get("embedding_hook")
        if not isinstance(embedding_hook, EmbeddingReplacementHook):
            raise PatchingInvariantError(
                f"Generation did not construct the embedding hook: {result.get('error')}"
            )
        embedding_hook.validate_applied_once()
        cache_hook = holder.get("cache_hook")
        if cache_hook is not None:
            cache_hook.validate()
        patch_hook = holder.get("patch_hook")
        if patch is not None:
            if not isinstance(patch_hook, ActivationReplacementHook):
                raise PatchingInvariantError("Generation did not construct the patch hook")
            patch_hook.diagnostics()
        source = _source_payload(result)
        if result.get("answer") != answer:
            raise RuntimeError(
                f"Teacher-forced answer changed: expected={answer!r} got={result.get('answer')!r}"
            )
        return FixedRun(
            result=result,
            source=source,
            prepared=holder["prepared"],
            embedding_hook=embedding_hook,
            cache_hook=cache_hook,
            patch_hook=patch_hook,
        )

    def _fresh_joint(self, runtime_case: RuntimeCase) -> tuple[str, dict[str, Any]]:
        result = self.joint.generate(
            runtime_case.prompt,
            list(runtime_case.evaluation.answer_classes),
            runtime_case.image_path,
            max_new_tokens=max(32, self.max_source_tokens + 24),
            source_classes=self.source_variant.classes,
        ).to_dict()
        source = _source_payload(result)
        answer = result.get("answer")
        if not isinstance(answer, str) or not answer.strip() or "\n" in answer:
            raise RuntimeError(f"Fresh joint generation produced no valid answer: {answer!r}")
        return answer.strip(), source

    def _validate_clean_parity(
        self,
        original: dict[str, Any],
        clean: dict[str, Any],
    ) -> dict[str, Any]:
        maximum = max(
            abs(left - right)
            for left, right in zip(original["logits"], clean["logits"], strict=True)
        )
        soft_difference = abs(float(original["soft_score"]) - float(clean["soft_score"]))
        numeric_close = maximum <= self.parity_tolerance
        passed = (
            original["parsed_label"] == clean["parsed_label"]
            and original["hard_label"] == clean["hard_label"]
        )
        diagnostics = {
            "passed": passed,
            "parsed_label_equal": original["parsed_label"] == clean["parsed_label"],
            "hard_label_equal": original["hard_label"] == clean["hard_label"],
            "max_abs_logit_difference": maximum,
            "soft_score_abs_difference": soft_difference,
            "numeric_close_at_reconstruction_tolerance": numeric_close,
            "tolerance": self.parity_tolerance,
            "pass_definition": "parsed_label_equal and hard_label_equal",
        }
        if not passed:
            raise RuntimeError(f"Clean teacher-forced parity failed: {diagnostics}")
        return diagnostics

    def _validate_no_patch(
        self,
        runtime_case: RuntimeCase,
        answer: str,
        corruption: str,
        baseline: dict[str, Any],
    ) -> dict[str, Any]:
        repeated = self._run_fixed(
            runtime_case,
            answer,
            corruption=corruption,
            capture_cache=False,
        ).source
        maximum = max(
            abs(left - right)
            for left, right in zip(baseline["logits"], repeated["logits"], strict=True)
        )
        passed = (
            baseline["parsed_label"] == repeated["parsed_label"]
            and baseline["hard_label"] == repeated["hard_label"]
            and maximum <= self.parity_tolerance
        )
        diagnostics = {
            "passed": passed,
            "parsed_label_equal": baseline["parsed_label"] == repeated["parsed_label"],
            "hard_label_equal": baseline["hard_label"] == repeated["hard_label"],
            "max_abs_logit_difference": maximum,
            "tolerance": self.parity_tolerance,
        }
        if not passed:
            raise PatchingInvariantError(
                f"No-patch corruption parity failed for {corruption}: {diagnostics}"
            )
        return diagnostics

    def _base_row(
        self,
        runtime_case: RuntimeCase,
        *,
        corruption: str,
        position: str,
        layer: int,
        answer: str | None,
    ) -> dict[str, Any]:
        return {
            "intervention_key": intervention_key(
                runtime_case.record["case_id"], corruption, position, layer
            ),
            "case_id": runtime_case.record["case_id"],
            "item_id": runtime_case.record["item_id"],
            "condition": runtime_case.record["condition"],
            "baseline_sa_group": runtime_case.record["baseline_sa_group"],
            "answer_fixed": True,
            "answer": answer,
            "corruption_type": corruption,
            "position": position,
            "layer": int(layer),
            "status": "running",
            "error": None,
        }

    def _failure_rows(
        self,
        runtime_case: RuntimeCase,
        *,
        answer: str | None,
        corruptions: Sequence[str],
        error: Exception,
        pending_keys: set[str],
    ) -> list[dict[str, Any]]:
        detail = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        rows = []
        for corruption in corruptions:
            for position, layer in self.grid:
                key = intervention_key(
                    runtime_case.record["case_id"], corruption, position, layer
                )
                if key not in pending_keys:
                    continue
                row = self._base_row(
                    runtime_case,
                    corruption=corruption,
                    position=position,
                    layer=layer,
                    answer=answer,
                )
                row.update({"status": "failed", "error": detail})
                rows.append(row)
        return rows

    def process_case(
        self,
        runtime_case: RuntimeCase,
        *,
        pending_keys: set[str],
        commit: Callable[[dict[str, Any]], None],
        validate_no_patch: bool,
    ) -> None:
        case_started = time.perf_counter()
        answer: str | None = None
        try:
            answer, original = self._fresh_joint(runtime_case)
            clean_run = self._run_fixed(
                runtime_case,
                answer,
                corruption=None,
                capture_cache=True,
            )
            parity = self._validate_clean_parity(original, clean_run.source)
            assert clean_run.cache_hook is not None
            clean_cache = clean_run.cache_hook.cache
        except PatchingInvariantError:
            raise
        except Exception as exc:
            for row in self._failure_rows(
                runtime_case,
                answer=answer,
                corruptions=self.corruptions,
                error=exc,
                pending_keys=pending_keys,
            ):
                row["elapsed_seconds"] = float(time.perf_counter() - case_started)
                commit(row)
            return

        for corruption in self.corruptions:
            corruption_keys = {
                intervention_key(
                    runtime_case.record["case_id"], corruption, position, layer
                )
                for position, layer in self.grid
            }
            if not corruption_keys.intersection(pending_keys):
                continue
            try:
                corrupt_run = self._run_fixed(
                    runtime_case,
                    answer,
                    corruption=corruption,
                    capture_cache=True,
                )
                if validate_no_patch:
                    self.no_patch_validation[corruption] = self._validate_no_patch(
                        runtime_case,
                        answer,
                        corruption,
                        corrupt_run.source,
                    )
                assert corrupt_run.cache_hook is not None
            except PatchingInvariantError:
                raise
            except Exception as exc:
                for row in self._failure_rows(
                    runtime_case,
                    answer=answer,
                    corruptions=(corruption,),
                    error=exc,
                    pending_keys=pending_keys,
                ):
                    row["elapsed_seconds"] = float(time.perf_counter() - case_started)
                    commit(row)
                continue

            for position, layer in self.grid:
                key = intervention_key(
                    runtime_case.record["case_id"], corruption, position, layer
                )
                if key not in pending_keys:
                    continue
                row_started = time.perf_counter()
                row = self._base_row(
                    runtime_case,
                    corruption=corruption,
                    position=position,
                    layer=layer,
                    answer=answer,
                )
                try:
                    patched_run = self._run_fixed(
                        runtime_case,
                        answer,
                        corruption=corruption,
                        capture_cache=False,
                        patch=(position, layer, clean_cache[layer][position]),
                    )
                    assert patched_run.patch_hook is not None
                    patch_diagnostics = patched_run.patch_hook.diagnostics()
                    metrics = metric_bundle(
                        clean_run.source,
                        corrupt_run.source,
                        patched_run.source,
                    )
                    corrupt_hidden = corrupt_run.cache_hook.cache[layer][position]
                    row.update(
                        {
                            "clean_SA": clean_run.source["hard_label"],
                            "corrupt_SA": corrupt_run.source["hard_label"],
                            "patched_SA": patched_run.source["hard_label"],
                            "clean_parsed_SA": clean_run.source["parsed_label"],
                            "corrupt_parsed_SA": corrupt_run.source["parsed_label"],
                            "patched_parsed_SA": patched_run.source["parsed_label"],
                            "clean_soft_score": clean_run.source["soft_score"],
                            "corrupt_soft_score": corrupt_run.source["soft_score"],
                            "patched_soft_score": patched_run.source["soft_score"],
                            "clean_logits": clean_run.source["logits"],
                            "corrupt_logits": corrupt_run.source["logits"],
                            "patched_logits": patched_run.source["logits"],
                            **metrics,
                            "patch_applied": patch_diagnostics["applied_count"] == 1,
                            "hook_count": patch_diagnostics["hook_count"],
                            "clean_hidden_norm": float(
                                torch.linalg.vector_norm(
                                    clean_cache[layer][position]
                                ).item()
                            ),
                            "corrupt_hidden_norm": float(
                                torch.linalg.vector_norm(corrupt_hidden).item()
                            ),
                            "patch_diagnostics": patch_diagnostics,
                            "embedding_diagnostics": patched_run.embedding_hook.diagnostics(),
                            "token_position": patched_run.prepared.positions[position],
                            "token_position_detail": patched_run.prepared.position_details[
                                position
                            ],
                            "embedding_span_lengths": {
                                name: len(span)
                                for name, span in patched_run.prepared.spans.items()
                            },
                            "clean_parity": parity,
                            "status": "completed",
                        }
                    )
                except PatchingInvariantError:
                    raise
                except Exception as exc:
                    row.update(
                        {
                            "status": "failed",
                            "error": {
                                "type": type(exc).__name__,
                                "message": str(exc),
                                "traceback": traceback.format_exc(),
                            },
                        }
                    )
                row["elapsed_seconds"] = float(time.perf_counter() - row_started)
                commit(row)


def _finite_values(records: Sequence[dict[str, Any]], path: Sequence[str]) -> list[float]:
    output: list[float] = []
    for record in records:
        value: Any = record
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            output.append(float(value))
    return output


def _stats(values: Sequence[float]) -> dict[str, Any]:
    return {
        "valid_count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
    }


def build_summary(
    records: Sequence[dict[str, Any]],
    *,
    expected_count: int,
) -> dict[str, Any]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for record in records:
        key = (
            str(record.get("corruption_type")),
            str(record.get("position")),
            int(record.get("layer", -1)),
        )
        groups.setdefault(key, []).append(record)
    cells = []
    for (corruption, position, layer), group in sorted(groups.items()):
        completed = [record for record in group if record.get("status") == "completed"]
        corrupt_changed = [
            record
            for record in completed
            if record.get("recovery", {}).get("corrupt_hard_differs") is True
        ]
        cells.append(
            {
                "corruption_type": corruption,
                "position": position,
                "layer": layer,
                "record_count": len(group),
                "completed_count": len(completed),
                "failed_count": len(group) - len(completed),
                "soft_SA_recovery": _stats(
                    _finite_values(completed, ("recovery", "soft"))
                ),
                "logit_recovery": _stats(
                    _finite_values(completed, ("recovery", "logit"))
                ),
                "hard_formula_recovery": _stats(
                    _finite_values(completed, ("recovery", "hard_formula"))
                ),
                "hard_SA_recovery": {
                    "eligible_count": len(corrupt_changed),
                    "recovered_count": sum(
                        record["recovery"]["hard_match_recovered"]
                        for record in corrupt_changed
                    ),
                    "rate": (
                        statistics.fmean(
                            float(record["recovery"]["hard_match_recovered"])
                            for record in corrupt_changed
                        )
                        if corrupt_changed
                        else None
                    ),
                },
                "soft_delta": _stats(_finite_values(completed, ("soft_delta",))),
            }
        )
    completed_count = sum(record.get("status") == "completed" for record in records)
    return {
        "expected_count": int(expected_count),
        "record_count": len(records),
        "completed_count": completed_count,
        "failed_count": len(records) - completed_count,
        "remaining_count": max(0, int(expected_count) - len(records)),
        "cells": cells,
        "recovery_definition": "(patched - corrupt) / (clean - corrupt); null when denominator is zero",
        "hard_recovery_definition": (
            "hard_formula uses numeric class indices; hard_SA_recovery is the "
            "clean-label match rate among cases whose corruption changed the hard label"
        ),
    }
