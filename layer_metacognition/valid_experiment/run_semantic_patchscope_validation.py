#!/usr/bin/env python3
"""Run Semantic Patchscope prompt-robustness validation for joint V3/V4 cases."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from confidence_test.answer_metrics import (  # noqa: E402
    normalize_answer,
    parse_answer_output,
)
from confidence_test.dataset_utils import (  # noqa: E402
    CONDITIONS,
    ConditionInput,
    load_evaluation_cases,
)
from confidence_test.inference_extension import ASSISTANT_ANSWER_PREFILL  # noqa: E402
from confidence_test.runtime_imports import (  # noqa: E402
    DEFAULT_INFERENCE_PATH,
    load_runtime,
)
from confidence_test.source_attribution_analyzer import (  # noqa: E402
    parse_joint_answer_source_output,
)
from confidence_test.source_attribution_prompt_utils import (  # noqa: E402
    V3_STAGE3_REANSWER_WITH_SOURCE_ATTRIBUTION_PROMPT,
    V4_STAGE1_FULL_EVIDENCE_ANSWER_WITH_SOURCE_ATTRIBUTION_PROMPT,
)
from confidence_test.source_attribution_schema import (  # noqa: E402
    ASSISTANT_SOURCE_ATTRIBUTION_PREFILL,
    SOURCE_ATTRIBUTION_CLASS_TEXT,
    SOURCE_ATTRIBUTION_CLASSES,
    build_source_token_specification,
    gather_source_class_logits,
)
from confidence_test.prompt_utils import STAGE1_TEXT_ANSWER_PROMPT  # noqa: E402
from layer_metacognition.conversation_builder import (  # noqa: E402
    prepare_multimodal_inputs,
    render_continued_assistant,
)
from layer_metacognition.direct_readout import (  # noqa: E402
    _restricted_logits,
    build_first_token_collision_report,
    project_hidden_to_vocab,
)
from layer_metacognition.hidden_state_store import (  # noqa: E402
    append_jsonl,
    atomic_write_json,
    atomic_write_text,
    load_jsonl,
)
from layer_metacognition.model_adapter import (  # noqa: E402
    LanguageModules,
    resolve_language_modules,
    run_hooked_forward,
    run_logits_forward,
    run_patched_logits_forward,
)
from layer_metacognition.token_positions import locate_marker_in_assistant  # noqa: E402
from layer_metacognition.token_spans import build_rendered_alignment  # noqa: E402
from layer_metacognition.valid_experiment.semantic_variants import (  # noqa: E402
    RESULT_COLUMNS,
    SemanticVariant,
    build_semantic_prompt,
    select_semantic_variants,
    variant_to_dict,
)


DEFAULT_MODEL_PATH = (
    ROOT / "qwen-2.5-vl" / "models" / "Qwen2.5-VL-7B-Instruct"
)
DEFAULT_DATASET_PATH = ROOT / "datasets" / "dataset_with_images.json"
DEFAULT_OUTPUT_DIR = (
    ROOT / "layer_metacognition" / "valid_experiment" / "output" / "validation"
)
VERSIONS = ("v3", "v4")
INTERNAL_CONFIDENCE_MAX_TOKENS = 12


class SkippableCaseError(RuntimeError):
    """A sample-local generation or parsing failure that may be recorded and skipped."""

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    if "all" in parts:
        raise ValueError(f"'all' cannot be combined with individual {label}s")
    duplicates = sorted({value for value in parts if parts.count(value) > 1})
    if duplicates:
        raise ValueError(f"Duplicate {label}(s): {', '.join(duplicates)}")
    invalid = [value for value in parts if value not in allowed]
    if invalid:
        raise ValueError(f"Unknown {label}(s): {', '.join(invalid)}")
    if not parts:
        raise ValueError(f"At least one {label} is required")
    selected = set(parts)
    return [value for value in allowed if value in selected]


def _parse_string_selection(values: list[str] | None) -> set[str] | None:
    if values is None:
        return None
    selected = {
        part.strip()
        for raw in values
        for part in str(raw).split(",")
        if part.strip()
    }
    if not selected:
        raise ValueError("--item-ids requires at least one ID")
    return selected


def _rebase_images(cases: list[Any], image_root: Path | None) -> list[Any]:
    if image_root is None:
        return cases
    rebased: list[Any] = []
    for case in cases:
        conditions: dict[str, ConditionInput] = {}
        for name, condition in case.conditions.items():
            raw = condition.relative_image_path
            if raw is None:
                conditions[name] = condition
                continue
            raw_path = Path(raw)
            resolved = (
                raw_path.resolve()
                if raw_path.is_absolute()
                else (image_root / raw_path).resolve()
            )
            error = None
            if not resolved.is_file():
                error = {
                    "type": "FileNotFoundError",
                    "message": (
                        f"Image does not exist under --image-root: {resolved}"
                    ),
                }
            conditions[name] = replace(
                condition,
                resolved_image_path=str(resolved),
                error=error,
            )
        rebased.append(replace(case, conditions=conditions))
    return rebased


def _image_content(prompt: str, image_path: str | None) -> list[dict[str, str]]:
    if image_path is None:
        return [{"type": "text", "text": prompt}]
    resolved = Path(image_path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Image not found: {resolved}")
    return [
        {"type": "image", "image": str(resolved)},
        {"type": "text", "text": prompt},
    ]


def _generate_continuation(
    inference: Any,
    *,
    prompt: str,
    image_path: str | None,
    assistant_prefill: str,
    max_new_tokens: int,
) -> str:
    """Greedy generation without requesting scores or entropy-related metrics."""
    messages = [
        {"role": "user", "content": _image_content(prompt, image_path)},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_prefill}],
        },
    ]
    rendered = render_continued_assistant(
        inference.processor,
        messages,
        assistant_prefill,
    )
    inputs = prepare_multimodal_inputs(
        inference.processor,
        messages,
        rendered,
        device=inference._get_inputs_device(),
    )
    with torch.inference_mode():
        generated = inference.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    input_length = int(inputs.input_ids.shape[1])
    new_tokens = generated[0, input_length:]
    tokenizer = getattr(inference.processor, "tokenizer", inference.processor)
    continuation = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    del generated, inputs
    return assistant_prefill + continuation


def _parse_joint_candidate_output(
    raw_output: str,
    answer_candidates: Sequence[str],
) -> tuple[str, str]:
    raw_answer, source_class, parsed = parse_joint_answer_source_output(
        raw_output
    )
    generated_answer = normalize_answer(raw_answer)
    if (
        not parsed
        or generated_answer not in answer_candidates
        or source_class not in SOURCE_ATTRIBUTION_CLASSES
    ):
        raise SkippableCaseError(
            "joint_output_parse",
            f"Could not parse exact joint candidate output: {raw_output!r}",
        )
    return generated_answer, source_class


def _score_distribution(
    class_logits: torch.Tensor,
    midpoints: Sequence[float],
) -> dict[str, Any]:
    logits = class_logits.detach().float().cpu()
    if logits.ndim != 1 or int(logits.numel()) != 9:
        raise ValueError(f"Expected nine class logits, got {tuple(logits.shape)}")
    probabilities = torch.softmax(logits, dim=-1)
    midpoint_tensor = torch.tensor(midpoints, dtype=torch.float32)
    score = float(torch.sum(probabilities * midpoint_tensor).item())
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise RuntimeError(f"Invalid soft image score: {score}")
    hard_index = int(torch.argmax(probabilities).item())
    return {
        "class_logits": [float(value) for value in logits.tolist()],
        "class_probabilities": [
            float(value) for value in probabilities.tolist()
        ],
        "hard_source_class": SOURCE_ATTRIBUTION_CLASSES[hard_index],
        "soft_image_score": score,
    }


@dataclass
class PreparedValidationTarget:
    variant: SemanticVariant
    inputs: Any
    target_position: int
    user_prompt: str
    rendered_prompt: str
    baseline: dict[str, Any]


class ValidationSemanticPatchscopeDecoder:
    """Prepare semantic targets once and decode every SAC layer consistently."""

    def __init__(
        self,
        *,
        inference: Any,
        modules: LanguageModules,
        class_token_ids: dict[str, Sequence[int]],
        variants: Sequence[SemanticVariant],
    ):
        self.inference = inference
        self.model = inference.model
        self.modules = modules
        self.processor = inference.processor
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.class_token_ids = {
            label: [int(token_id) for token_id in class_token_ids[label]]
            for label in SOURCE_ATTRIBUTION_CLASSES
        }
        self.targets: dict[str, PreparedValidationTarget] = {}
        self.call_counts = {"baseline": 0, "patched": 0}
        for variant in variants:
            self.targets[variant.variant_id] = self._prepare_target(variant)

    def _prepare_target(
        self,
        variant: SemanticVariant,
    ) -> PreparedValidationTarget:
        user_prompt = build_semantic_prompt(variant)
        prefill = ASSISTANT_SOURCE_ATTRIBUTION_PREFILL
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": prefill}],
            },
        ]
        rendered = render_continued_assistant(
            self.processor,
            messages,
            prefill,
        )
        inputs = prepare_multimodal_inputs(
            self.processor,
            messages,
            rendered,
            device=self.inference._get_inputs_device(),
        )
        alignment = build_rendered_alignment(
            self.tokenizer,
            rendered,
            inputs.input_ids,
            inputs.attention_mask,
        )
        location = locate_marker_in_assistant(
            self.tokenizer,
            alignment.rendered_ids,
            prefill,
            prefill,
            name=f"{variant.variant_id} semantic target sac",
            position_map=alignment.rendered_to_processed,
            processed_ids=alignment.processed_ids,
        )
        target_position = int(location["position"])
        if target_position != len(alignment.processed_ids) - 1:
            raise ValueError(
                f"{variant.variant_id} target position {target_position} is not "
                f"the final valid input token {len(alignment.processed_ids) - 1}"
            )
        vocab_logits = run_logits_forward(
            self.model,
            inputs,
            [target_position],
            self.modules,
        )[target_position]
        self.call_counts["baseline"] += 1
        class_logits = gather_source_class_logits(
            vocab_logits,
            self.class_token_ids,
        )
        baseline = _score_distribution(
            class_logits,
            variant.image_midpoints,
        )
        del vocab_logits, class_logits
        return PreparedValidationTarget(
            variant=variant,
            inputs=inputs,
            target_position=target_position,
            user_prompt=user_prompt,
            rendered_prompt=rendered,
            baseline=baseline,
        )

    def baselines(self) -> dict[str, dict[str, Any]]:
        return {
            variant_id: deepcopy(target.baseline)
            for variant_id, target in self.targets.items()
        }

    def target_metadata(self) -> dict[str, dict[str, Any]]:
        return {
            variant_id: {
                "user_prompt": target.user_prompt,
                "rendered_target_prompt": target.rendered_prompt,
                "class_order": list(target.variant.class_order),
                "image_midpoints": list(target.variant.image_midpoints),
                "target_position": target.target_position,
                "unpatched_baseline": deepcopy(target.baseline),
            }
            for variant_id, target in self.targets.items()
        }

    def run_patched_source_readout(
        self,
        *,
        variant_id: str,
        layer_index: int,
        source_hidden: torch.Tensor,
    ) -> dict[str, Any]:
        try:
            target = self.targets[variant_id]
        except KeyError as exc:
            raise ValueError(
                f"Semantic target {variant_id!r} was not prepared"
            ) from exc
        vocab_logits = run_patched_logits_forward(
            self.model,
            target.inputs,
            self.modules,
            layer_index=layer_index,
            target_position=target.target_position,
            source_hidden=source_hidden,
        )
        self.call_counts["patched"] += 1
        class_logits = gather_source_class_logits(
            vocab_logits,
            self.class_token_ids,
        )
        raw = _score_distribution(
            class_logits,
            target.variant.image_midpoints,
        )
        baseline_logits = torch.tensor(
            target.baseline["class_logits"],
            dtype=torch.float32,
        )
        delta_logits = class_logits.detach().float().cpu() - baseline_logits
        corrected = _score_distribution(
            delta_logits,
            target.variant.image_midpoints,
        )
        result = {
            **raw,
            "baseline_corrected_class_logits": corrected["class_logits"],
            "baseline_corrected_class_probabilities": corrected[
                "class_probabilities"
            ],
            "baseline_corrected_hard_source_class": corrected[
                "hard_source_class"
            ],
            "baseline_corrected_soft_image_score": corrected[
                "soft_image_score"
            ],
        }
        del vocab_logits, class_logits, delta_logits
        return result


def _answer_readout(
    hidden: torch.Tensor,
    *,
    modules: LanguageModules,
    candidates: list[str],
    collision_report: dict[str, Any],
) -> dict[str, Any]:
    vocab_logits = project_hidden_to_vocab(
        hidden,
        modules.final_norm,
        modules.lm_head,
    )
    class_logits = _restricted_logits(
        vocab_logits,
        candidates,
        collision_report,
    )
    probabilities = torch.softmax(class_logits, dim=-1)
    predicted_index = int(torch.argmax(probabilities).item())
    result = {
        "predicted_answer": candidates[predicted_index],
        "predicted_answer_probability": float(
            probabilities[predicted_index].item()
        ),
        "answer_class_logits": [
            float(value) for value in class_logits.detach().cpu().tolist()
        ],
        "answer_class_probabilities": [
            float(value) for value in probabilities.detach().cpu().tolist()
        ],
    }
    del vocab_logits, class_logits, probabilities
    return result


class SemanticValidationRunner:
    """Run only answer readout and semantic SAC Patchscope for one joint case."""

    def __init__(
        self,
        *,
        inference: Any,
        modules: LanguageModules,
        confidence_analyzer: Any,
        decoder: ValidationSemanticPatchscopeDecoder,
        variants: Sequence[SemanticVariant],
        max_answer_tokens: int,
        max_source_tokens: int,
    ):
        self.inference = inference
        self.modules = modules
        self.confidence_analyzer = confidence_analyzer
        self.decoder = decoder
        self.variants = tuple(variants)
        self.max_answer_tokens = max_answer_tokens
        self.max_source_tokens = max_source_tokens
        self.processor = inference.processor
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.initial_cache: dict[str, tuple[str, str]] = {}
        self.call_counts = {
            "initial_answer_generation": 0,
            "previous_confidence_generation": 0,
            "joint_answer_source_generation": 0,
            "source_hooked_forward": 0,
        }

    def _initial_v3(self, case: Any) -> tuple[str, str]:
        key = f"{case.item_id}::{case.prior_index}"
        if key in self.initial_cache:
            return self.initial_cache[key]
        prompt = STAGE1_TEXT_ANSWER_PROMPT.format(
            question=case.question,
            text_clue=case.text_clue,
        )
        self.call_counts["initial_answer_generation"] += 1
        raw = _generate_continuation(
            self.inference,
            prompt=prompt,
            image_path=None,
            assistant_prefill=ASSISTANT_ANSWER_PREFILL,
            max_new_tokens=self.max_answer_tokens,
        )
        _answer, normalized, parsed = parse_answer_output(raw)
        if not parsed or normalized not in case.answer_classes:
            raise SkippableCaseError(
                "v3_initial_answer_parse",
                f"Could not parse a candidate V3 initial answer: {raw!r}"
            )
        self.call_counts["previous_confidence_generation"] += 1
        confidence = self.confidence_analyzer.analyze(
            case.question,
            case.text_clue,
            normalized,
        )
        label = getattr(confidence, "confidence_label", None)
        if not isinstance(label, str) or not label:
            raise SkippableCaseError(
                "v3_previous_confidence_parse",
                "V3 previous confidence produced no label",
            )
        value = (normalized, label)
        self.initial_cache[key] = value
        return value

    def _joint_prompt(self, case: Any, version: str) -> str:
        if version == "v3":
            previous_answer, previous_confidence = self._initial_v3(case)
            return V3_STAGE3_REANSWER_WITH_SOURCE_ATTRIBUTION_PROMPT.format(
                question=case.question,
                text_clue=case.text_clue,
                previous_answer=previous_answer,
                previous_confidence=previous_confidence,
                source_classes=SOURCE_ATTRIBUTION_CLASS_TEXT,
            )
        if version == "v4":
            return V4_STAGE1_FULL_EVIDENCE_ANSWER_WITH_SOURCE_ATTRIBUTION_PROMPT.format(
                question=case.question,
                text_clue=case.text_clue,
                source_classes=SOURCE_ATTRIBUTION_CLASS_TEXT,
            )
        raise ValueError(f"Unsupported version: {version}")

    def process_case(
        self,
        *,
        case: Any,
        condition: str,
        version: str,
    ) -> dict[str, Any]:
        case_id = (
            f"{case.item_id}__prior_{case.prior_index}__"
            f"{condition}__{version}__joint"
        )
        condition_input = case.conditions[condition]
        if condition_input.error:
            raise SkippableCaseError(
                "image_resolution",
                f"{case_id} image resolution failed: {condition_input.error}"
            )
        image_path = condition_input.resolved_image_path
        if not image_path:
            raise SkippableCaseError(
                "image_resolution",
                f"{case_id} condition has no resolved image",
            )
        answer_candidates = list(case.answer_classes)
        if not answer_candidates:
            raise SkippableCaseError(
                "answer_candidates",
                f"{case_id} has no answer candidates: {case.answer_class_error}"
            )
        answer_report = build_first_token_collision_report(
            self.tokenizer,
            answer_candidates,
        )
        if answer_report["collisions"]:
            raise SkippableCaseError(
                "answer_token_collision",
                f"Answer first-token collisions: {answer_report['collisions']}"
            )

        prompt = self._joint_prompt(case, version)
        self.call_counts["joint_answer_source_generation"] += 1
        raw_output = _generate_continuation(
            self.inference,
            prompt=prompt,
            image_path=image_path,
            assistant_prefill=ASSISTANT_ANSWER_PREFILL,
            max_new_tokens=max(
                32,
                self.max_answer_tokens + self.max_source_tokens + 8,
            ),
        )
        generated_answer, source_class = _parse_joint_candidate_output(
            raw_output,
            answer_candidates,
        )

        assistant_text = (
            f"{ASSISTANT_ANSWER_PREFILL} {generated_answer}\n"
            f"{ASSISTANT_SOURCE_ATTRIBUTION_PREFILL}{source_class}"
        )
        messages = [
            {"role": "user", "content": _image_content(prompt, image_path)},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            },
        ]
        rendered = render_continued_assistant(
            self.processor,
            messages,
            assistant_text,
        )
        inputs = prepare_multimodal_inputs(
            self.processor,
            messages,
            rendered,
            device=self.inference._get_inputs_device(),
        )
        alignment = build_rendered_alignment(
            self.tokenizer,
            rendered,
            inputs.input_ids,
            inputs.attention_mask,
        )
        processed_ids = alignment.processed_ids
        ac = locate_marker_in_assistant(
            self.tokenizer,
            alignment.rendered_ids,
            assistant_text,
            ASSISTANT_ANSWER_PREFILL,
            name="ac",
            position_map=alignment.rendered_to_processed,
            processed_ids=processed_ids,
        )
        sac = locate_marker_in_assistant(
            self.tokenizer,
            alignment.rendered_ids,
            assistant_text,
            ASSISTANT_SOURCE_ATTRIBUTION_PREFILL,
            name="sac",
            position_map=alignment.rendered_to_processed,
            processed_ids=processed_ids,
        )
        positions = {"ac": int(ac["position"]), "sac": int(sac["position"])}
        self.call_counts["source_hooked_forward"] += 1
        forward = run_hooked_forward(
            self.inference.model,
            inputs,
            self.modules,
            positions,
            logits_positions=list(positions.values()),
        )

        reference_answer_logits = _restricted_logits(
            forward.logits_by_position[positions["ac"]],
            answer_candidates,
            answer_report,
        )
        reference_source_logits = gather_source_class_logits(
            forward.logits_by_position[positions["sac"]],
            self.decoder.class_token_ids,
        )
        layers: dict[str, dict[str, Any]] = {}
        for layer_index in range(self.modules.num_hidden_layers):
            answer = _answer_readout(
                forward.hidden_by_name["ac"][layer_index],
                modules=self.modules,
                candidates=answer_candidates,
                collision_report=answer_report,
            )
            semantic: dict[str, Any] = {}
            for variant in self.variants:
                semantic[variant.variant_id] = (
                    self.decoder.run_patched_source_readout(
                        variant_id=variant.variant_id,
                        layer_index=layer_index,
                        source_hidden=forward.hidden_by_name["sac"][layer_index],
                    )
                )
            layers[str(layer_index)] = {
                "answer": answer,
                "semantic_variants": semantic,
            }

        record = {
            "case_id": case_id,
            "item_id": case.item_id,
            "prior_index": case.prior_index,
            "condition": condition,
            "version": version,
            "attribution_mode": "joint",
            "ground_truths": {
                "answer": case.ground_truth_answer,
                "conflict_answer": case.conflict_answer,
            },
            "text_answer": case.text_answer,
            "generated_answer": generated_answer,
            "generated_source_class": source_class,
            "answer_candidates": answer_candidates,
            "answer_token_ids": {
                label: list(
                    answer_report["labels"][label]["first_token_variants"]
                )
                for label in answer_candidates
            },
            "source_token_ids": deepcopy(self.decoder.class_token_ids),
            "token_positions": {"ac": ac, "sac": sac},
            "reference_restricted_logits": {
                "answer": [
                    float(value)
                    for value in reference_answer_logits.detach().cpu().tolist()
                ],
                "source": [
                    float(value)
                    for value in reference_source_logits.detach().cpu().tolist()
                ],
            },
            "variant_targets": self.decoder.target_metadata(),
            "layers": layers,
            "call_counts": {
                "source_hooked_forward": 1,
                "target_baseline_total": self.decoder.call_counts["baseline"],
                "target_baseline_by_variant": {
                    variant.variant_id: 1 for variant in self.variants
                },
            },
            "status": "completed",
        }
        del (
            reference_answer_logits,
            reference_source_logits,
            forward,
            inputs,
        )
        return record


def result_columns(variants: Sequence[SemanticVariant]) -> list[str]:
    selected = {variant.variant_id for variant in variants}
    return [
        column
        for column in RESULT_COLUMNS
        if column in ("answer", "answer_probability") or column in selected
    ]


def build_main_results(
    records: list[dict[str, Any]],
    columns: list[str],
    *,
    corrected: bool,
) -> dict[str, Any]:
    variant_columns = columns[2:]
    output_records: list[dict[str, Any]] = []
    for record in records:
        if record.get("status") != "completed":
            continue
        layers: dict[str, list[Any]] = {}
        raw_layers = record.get("layers")
        if not isinstance(raw_layers, dict):
            raise ValueError(f"{record.get('case_id')} has no layer mapping")
        for layer in sorted(raw_layers, key=lambda value: int(value)):
            layer_value = raw_layers[layer]
            answer = layer_value["answer"]
            semantic = layer_value["semantic_variants"]
            row: list[Any] = [
                answer["predicted_answer"],
                float(answer["predicted_answer_probability"]),
            ]
            score_key = (
                "baseline_corrected_soft_image_score"
                if corrected
                else "soft_image_score"
            )
            row.extend(float(semantic[name][score_key]) for name in variant_columns)
            if len(row) != len(columns):
                raise RuntimeError(
                    f"{record.get('case_id')} layer {layer} row length mismatch"
                )
            layers[str(layer)] = row
        output_records.append(
            {
                "case_id": record["case_id"],
                "ground_truths": deepcopy(record.get("ground_truths")),
                "text_answer": record.get("text_answer"),
                "generated_answer": record.get("generated_answer"),
                "generated_source_class": record.get(
                    "generated_source_class"
                ),
                "layers": layers,
            }
        )
    return {
        "columns": list(columns),
        "source_value_definition": (
            "baseline_corrected_soft_image_score"
            if corrected
            else "soft_image_score"
        ),
        "records": output_records,
    }


def write_rebuilt_outputs(
    output_dir: Path,
    records: list[dict[str, Any]],
    columns: list[str],
) -> None:
    _atomic_write_compact_layer_rows(
        output_dir / "validation_results.json",
        build_main_results(records, columns, corrected=False),
    )
    _atomic_write_compact_layer_rows(
        output_dir / "validation_results_corrected.json",
        build_main_results(records, columns, corrected=True),
    )


def _atomic_write_compact_layer_rows(path: Path, payload: dict[str, Any]) -> None:
    """Pretty-print result metadata while keeping each per-layer row on one line."""
    def compact_row(values: list[Any]) -> str:
        encoded: list[str] = []
        for index, value in enumerate(values):
            if index == 0:
                encoded.append(json.dumps(value, ensure_ascii=False))
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                encoded.append(f"{float(value):.3f}")
            else:
                encoded.append(json.dumps(value, ensure_ascii=False))
        return "[" + ",".join(encoded) + "]"

    lines = ["{"]
    lines.append('  "columns": ' + json.dumps(payload["columns"], ensure_ascii=False) + ",")
    lines.append(
        '  "source_value_definition": '
        + json.dumps(payload["source_value_definition"], ensure_ascii=False)
        + ","
    )
    lines.append('  "records": [')
    records = payload["records"]
    for record_index, record in enumerate(records):
        lines.append("    {")
        lines.append(
            '      "case_id": '
            + json.dumps(record["case_id"], ensure_ascii=False)
            + ","
        )
        lines.append(
            '      "ground_truths": '
            + json.dumps(record["ground_truths"], ensure_ascii=False)
            + ","
        )
        for key in (
            "text_answer",
            "generated_answer",
            "generated_source_class",
        ):
            lines.append(
                f'      "{key}": '
                + json.dumps(record[key], ensure_ascii=False)
                + ","
            )
        lines.append('      "layers": {')
        layer_items = list(record["layers"].items())
        for layer_index, (layer, values) in enumerate(layer_items):
            suffix = "," if layer_index + 1 < len(layer_items) else ""
            lines.append(
                "        "
                + json.dumps(str(layer))
                + ": "
                + compact_row(values)
                + suffix
            )
        lines.append("      }")
        suffix = "," if record_index + 1 < len(records) else ""
        lines.append("    }" + suffix)
    lines.append("  ]")
    lines.append("}")
    atomic_write_text(path, "\n".join(lines) + "\n")


def _progress(
    records: list[dict[str, Any]],
    *,
    expected_count: int,
    status: str,
    skipped_records: Sequence[dict[str, Any]] = (),
    error: dict[str, str] | None = None,
) -> dict[str, Any]:
    completed = [str(record["case_id"]) for record in records]
    skipped = [str(record["case_id"]) for record in skipped_records]
    value: dict[str, Any] = {
        "status": status,
        "updated_at": utc_now(),
        "expected_count": expected_count,
        "completed_count": len(completed),
        "skipped_count": len(skipped),
        "attempted_count": len(completed) + len(skipped),
        "pending_count": max(
            0,
            expected_count - len(completed) - len(skipped),
        ),
        "completed_case_ids": completed,
        "skipped_case_ids": skipped,
        "last_case_id": completed[-1] if completed else None,
        "last_skipped_case_id": skipped[-1] if skipped else None,
    }
    if error is not None:
        value["error"] = error
    return value


def _filter_cases(
    cases: list[Any],
    *,
    item_ids: set[str] | None,
    prior_indices: set[int] | None,
) -> list[Any]:
    if item_ids is not None:
        available = {case.item_id for case in cases}
        missing = item_ids.difference(available)
        if missing:
            raise ValueError(f"Unknown --item-ids: {sorted(missing)}")
        cases = [case for case in cases if case.item_id in item_ids]
    if prior_indices is not None:
        available_priors = {case.prior_index for case in cases}
        missing_priors = prior_indices.difference(available_priors)
        if missing_priors:
            raise ValueError(
                f"Unknown --prior-indices: {sorted(missing_priors)}"
            )
        cases = [
            case for case in cases if case.prior_index in prior_indices
        ]
    if not cases:
        raise ValueError("No cases remain after item/prior filtering")
    return cases


def build_resume_signature(
    *,
    dataset: Path,
    image_root: Path | None,
    model_path: Path,
    inference_path: Path,
    versions: list[str],
    conditions: list[str],
    item_ids: set[str] | None,
    prior_indices: set[int] | None,
    max_items: int | None,
    variants: Sequence[SemanticVariant],
    columns: list[str],
    max_answer_tokens: int,
    max_source_tokens: int,
) -> dict[str, Any]:
    return {
        "dataset": str(dataset),
        "image_root": str(image_root) if image_root else None,
        "model_path": str(model_path),
        "inference_path": str(inference_path),
        "versions": list(versions),
        "attribution_mode": "joint",
        "conditions": list(conditions),
        "item_ids": sorted(item_ids) if item_ids else None,
        "prior_indices": sorted(prior_indices) if prior_indices else None,
        "max_items": max_items,
        "selected_variants": [
            variant.variant_id for variant in variants
        ],
        "result_columns": list(columns),
        "semantic_variants": [
            variant_to_dict(variant) for variant in variants
        ],
        "max_answer_tokens": max_answer_tokens,
        "max_source_tokens": max_source_tokens,
        "internal_confidence_max_tokens": INTERNAL_CONFIDENCE_MAX_TOKENS,
    }


def _expected_case_ids(
    cases: list[Any],
    conditions: list[str],
    versions: list[str],
) -> list[str]:
    return [
        (
            f"{case.item_id}__prior_{case.prior_index}__"
            f"{condition}__{version}__joint"
        )
        for case in cases
        for condition in conditions
        for version in versions
    ]


def _validate_existing_records(
    records: list[dict[str, Any]],
    expected_ids: set[str],
) -> set[str]:
    ids = [str(record.get("case_id")) for record in records]
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicates:
        raise ValueError(
            f"Duplicate case_id(s) in validation_details.jsonl: {duplicates}"
        )
    unexpected = sorted(set(ids).difference(expected_ids))
    if unexpected:
        raise ValueError(
            f"JSONL contains cases outside the saved configuration: {unexpected}"
        )
    if any(record.get("status") != "completed" for record in records):
        raise ValueError("validation_details.jsonl contains a non-completed record")
    return set(ids)


def _validate_skipped_records(
    records: list[dict[str, Any]],
    expected_ids: set[str],
) -> set[str]:
    ids = [str(record.get("case_id")) for record in records]
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicates:
        raise ValueError(
            f"Duplicate case_id(s) in validation_failures.jsonl: {duplicates}"
        )
    unexpected = sorted(set(ids).difference(expected_ids))
    if unexpected:
        raise ValueError(
            "Failure JSONL contains cases outside the saved configuration: "
            f"{unexpected}"
        )
    if any(record.get("status") != "skipped" for record in records):
        raise ValueError(
            "validation_failures.jsonl contains a non-skipped record"
        )
    return set(ids)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--image-root")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--inference-path", default=str(DEFAULT_INFERENCE_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--versions", nargs="+", default=["v3"])
    parser.add_argument(
        "--attribution-mode",
        choices=["joint"],
        default="joint",
    )
    parser.add_argument("--conditions", nargs="+", default=["all"])
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--item-ids", nargs="+")
    parser.add_argument("--prior-indices", nargs="+", type=int)
    parser.add_argument(
        "--semantic-variants",
        nargs="+",
        default=["all"],
    )
    parser.add_argument("--max-answer-tokens", type=int, default=24)
    parser.add_argument("--max-source-tokens", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser


def _configure_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("semantic_patchscope_validation")
    logger.setLevel(logging.INFO)
    for existing_handler in logger.handlers:
        existing_handler.close()
    logger.handlers.clear()
    handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        versions = _canonical_values(args.versions, VERSIONS, "version")
        conditions = _canonical_values(
            args.conditions,
            CONDITIONS,
            "condition",
        )
        variants = select_semantic_variants(args.semantic_variants)
        columns = result_columns(variants)
        item_ids = _parse_string_selection(args.item_ids)
        prior_indices = (
            set(args.prior_indices) if args.prior_indices is not None else None
        )
        if prior_indices is not None and any(
            value < 0 for value in prior_indices
        ):
            raise ValueError("--prior-indices must be non-negative")
        for name in ("max_items", "max_answer_tokens", "max_source_tokens"):
            value = getattr(args, name)
            if value is not None and value < 1:
                raise ValueError(
                    f"--{name.replace('_', '-')} must be positive"
                )
        dataset = Path(args.dataset).resolve()
        image_root = (
            Path(args.image_root).resolve() if args.image_root else None
        )
        model_path = Path(args.model_path).resolve()
        inference_path = Path(args.inference_path).resolve()
        output_dir = Path(args.output_dir).resolve()
        if not dataset.is_file():
            raise FileNotFoundError(f"Dataset does not exist: {dataset}")
        if image_root is not None and not image_root.is_dir():
            raise FileNotFoundError(f"Image root does not exist: {image_root}")
        if not model_path.is_dir():
            raise FileNotFoundError(f"Model does not exist: {model_path}")
        if not inference_path.is_file():
            raise FileNotFoundError(
                f"Inference source does not exist: {inference_path}"
            )
    except Exception as exc:
        parser.error(str(exc))

    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "validation_details.jsonl"
    failures_path = output_dir / "validation_failures.jsonl"
    config_path = output_dir / "config.json"
    progress_path = output_dir / "progress.json"
    logger = _configure_logger(output_dir)
    try:
        if config_path.exists() and not args.resume:
            raise ValueError(
                "Output already has config.json; pass --resume or use a new directory"
            )
        if (
            details_path.exists() or failures_path.exists()
        ) and not config_path.exists():
            raise ValueError(
                "Validation JSONL exists without config.json"
            )
        cases, dataset_metadata = load_evaluation_cases(
            dataset,
            item_limit=args.max_items,
            fallback_null_path=output_dir / ".runtime" / "null.png",
        )
        cases = _filter_cases(
            _rebase_images(cases, image_root),
            item_ids=item_ids,
            prior_indices=prior_indices,
        )
        signature = build_resume_signature(
            dataset=dataset,
            image_root=image_root,
            model_path=model_path,
            inference_path=inference_path,
            versions=versions,
            conditions=conditions,
            item_ids=item_ids,
            prior_indices=prior_indices,
            max_items=args.max_items,
            variants=variants,
            columns=columns,
            max_answer_tokens=args.max_answer_tokens,
            max_source_tokens=args.max_source_tokens,
        )
        if config_path.exists():
            saved_config = json.loads(config_path.read_text(encoding="utf-8"))
            if saved_config.get("resume_signature") != signature:
                raise ValueError(
                    "Resume configuration differs from saved config.json"
                )
        else:
            saved_config = {
                "format_version": 1,
                **signature,
                "output_dir": str(output_dir),
                "dataset_metadata": dataset_metadata,
                "source_value_definition": "soft_image_score",
                "collected_layer_targets": ["ac", "sac"],
                "collected_metrics": [
                    "predicted_answer",
                    "predicted_answer_probability",
                    "semantic_variant_soft_image_scores",
                ],
                "confidence_layer_readout": False,
                "attention_analysis": False,
                "identity_patchscope": False,
                "source_lmhead": False,
                "model_dtype": None,
                "hidden_layer_count": None,
                "comparison_layers": None,
                "created_at": utc_now(),
                "resume_signature": signature,
            }
            atomic_write_json(config_path, saved_config)

        existing = load_jsonl(
            details_path,
            repair_trailing=args.resume,
        )
        skipped_records = load_jsonl(
            failures_path,
            repair_trailing=args.resume,
        )
        expected_id_list = _expected_case_ids(cases, conditions, versions)
        expected_ids = set(expected_id_list)
        completed_ids = _validate_existing_records(existing, expected_ids)
        skipped_ids = _validate_skipped_records(
            skipped_records,
            expected_ids,
        )
        overlap = sorted(completed_ids.intersection(skipped_ids))
        if overlap:
            raise ValueError(
                "Cases cannot be both completed and skipped: "
                f"{overlap}"
            )
        attempted_ids = completed_ids | skipped_ids
        write_rebuilt_outputs(output_dir, existing, columns)
        if attempted_ids == expected_ids:
            terminal_status = (
                "complete_with_skips" if skipped_records else "complete"
            )
            atomic_write_json(
                progress_path,
                _progress(
                    existing,
                    expected_count=len(expected_ids),
                    status=terminal_status,
                    skipped_records=skipped_records,
                ),
            )
            print(
                "[INFO] No pending cases; main results rebuilt from JSONL "
                f"(skipped={len(skipped_records)})."
            )
            return 0

        atomic_write_json(
            progress_path,
            _progress(
                existing,
                expected_count=len(expected_ids),
                status="initializing",
                skipped_records=skipped_records,
            ),
        )
        runtime = load_runtime(inference_path)
        inference = runtime.QwenVLInference(model_path=str(model_path))
        modules = resolve_language_modules(inference.model)
        confidence_analyzer = runtime.ConfidenceAnalyzer(
            inference,
            max_new_tokens=INTERNAL_CONFIDENCE_MAX_TOKENS,
        )
        tokenizer = getattr(inference.processor, "tokenizer", inference.processor)
        token_specification = build_source_token_specification(tokenizer)
        decoder = ValidationSemanticPatchscopeDecoder(
            inference=inference,
            modules=modules,
            class_token_ids=token_specification.class_token_ids,
            variants=variants,
        )
        atomic_write_json(
            output_dir / "target_baselines.json",
            decoder.baselines(),
        )
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        saved_config["model_dtype"] = inference.dtype_name
        saved_config["hidden_layer_count"] = modules.num_hidden_layers
        saved_config["comparison_layers"] = {
            "start": 0,
            "end": modules.num_hidden_layers - 2,
            "inclusive": True,
            "excluded_layers": [modules.num_hidden_layers - 1],
        }
        saved_config["model_runtime"] = {
            "dtype": inference.dtype_name,
            "device_map": getattr(inference.model, "hf_device_map", None),
            "num_hidden_layers": modules.num_hidden_layers,
            "hidden_size": modules.hidden_size,
        }
        atomic_write_json(config_path, saved_config)
        runner = SemanticValidationRunner(
            inference=inference,
            modules=modules,
            confidence_analyzer=confidence_analyzer,
            decoder=decoder,
            variants=variants,
            max_answer_tokens=args.max_answer_tokens,
            max_source_tokens=args.max_source_tokens,
        )
        atomic_write_json(
            progress_path,
            _progress(
                existing,
                expected_count=len(expected_ids),
                status="running",
                skipped_records=skipped_records,
            ),
        )

        for case in cases:
            for condition in conditions:
                for version in versions:
                    case_id = (
                        f"{case.item_id}__prior_{case.prior_index}__"
                        f"{condition}__{version}__joint"
                    )
                    if case_id in attempted_ids:
                        prior_status = (
                            "completed"
                            if case_id in completed_ids
                            else "skipped"
                        )
                        logger.info(
                            "resume_skip case_id=%s prior_status=%s",
                            case_id,
                            prior_status,
                        )
                        continue
                    started = time.perf_counter()
                    try:
                        record = runner.process_case(
                            case=case,
                            condition=condition,
                            version=version,
                        )
                    except SkippableCaseError as exc:
                        skipped_record = {
                            "case_id": case_id,
                            "item_id": case.item_id,
                            "prior_index": case.prior_index,
                            "condition": condition,
                            "version": version,
                            "attribution_mode": "joint",
                            "status": "skipped",
                            "skipped_at": utc_now(),
                            "elapsed_seconds": round(
                                time.perf_counter() - started,
                                6,
                            ),
                            "error": {
                                "stage": exc.stage,
                                "type": type(exc).__name__,
                                "message": str(exc),
                            },
                        }
                        append_jsonl(
                            failures_path,
                            skipped_record,
                            fsync=True,
                        )
                        skipped_records = load_jsonl(
                            failures_path,
                            repair_trailing=False,
                        )
                        skipped_ids.add(case_id)
                        attempted_ids.add(case_id)
                        atomic_write_json(
                            progress_path,
                            _progress(
                                existing,
                                expected_count=len(expected_ids),
                                status="running_with_skips",
                                skipped_records=skipped_records,
                            ),
                        )
                        logger.warning(
                            "case_skipped case_id=%s stage=%s "
                            "elapsed_seconds=%.6f message=%s",
                            case_id,
                            exc.stage,
                            skipped_record["elapsed_seconds"],
                            exc,
                        )
                        continue
                    append_jsonl(details_path, record, fsync=True)
                    existing = load_jsonl(details_path, repair_trailing=False)
                    completed_ids.add(case_id)
                    attempted_ids.add(case_id)
                    write_rebuilt_outputs(output_dir, existing, columns)
                    atomic_write_json(
                        progress_path,
                        _progress(
                            existing,
                            expected_count=len(expected_ids),
                            status=(
                                "running_with_skips"
                                if skipped_records
                                else "running"
                            ),
                            skipped_records=skipped_records,
                        ),
                    )
                    logger.info(
                        "case_completed case_id=%s elapsed_seconds=%.6f",
                        case_id,
                        time.perf_counter() - started,
                    )

        existing = load_jsonl(details_path, repair_trailing=False)
        skipped_records = load_jsonl(
            failures_path,
            repair_trailing=False,
        )
        write_rebuilt_outputs(output_dir, existing, columns)
        terminal_status = (
            "complete_with_skips" if skipped_records else "complete"
        )
        atomic_write_json(
            progress_path,
            _progress(
                existing,
                expected_count=len(expected_ids),
                status=terminal_status,
                skipped_records=skipped_records,
            ),
        )
        print(
            json.dumps(
                {
                    "status": terminal_status,
                    "completed_cases": len(existing),
                    "skipped_cases": len(skipped_records),
                    "runner_call_counts": runner.call_counts,
                    "patchscope_call_counts": decoder.call_counts,
                    "output_dir": str(output_dir),
                },
                ensure_ascii=False,
            )
        )
        return 0
    except KeyboardInterrupt:
        try:
            records = load_jsonl(details_path, repair_trailing=True)
            failures = load_jsonl(failures_path, repair_trailing=True)
            write_rebuilt_outputs(output_dir, records, columns)
            atomic_write_json(
                progress_path,
                _progress(
                    records,
                    expected_count=len(expected_ids),
                    status="interrupted",
                    skipped_records=failures,
                ),
            )
        except Exception:
            logger.exception("interruption_cleanup_failed")
        print("[WARN] Interrupted after preserving completed cases.", file=sys.stderr)
        return 130
    except Exception as exc:
        logger.exception("validation_failed")
        try:
            records = load_jsonl(details_path, repair_trailing=True)
            failures = load_jsonl(failures_path, repair_trailing=True)
            write_rebuilt_outputs(output_dir, records, columns)
            atomic_write_json(
                progress_path,
                _progress(
                    records,
                    expected_count=len(locals().get("expected_ids", set())),
                    status="failed",
                    skipped_records=failures,
                    error={
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                ),
            )
        except Exception:
            logger.exception("failure_progress_update_failed")
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
