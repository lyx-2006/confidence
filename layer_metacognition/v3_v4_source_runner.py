"""V3/V4 source-attribution generation and metacognition orchestration."""

from __future__ import annotations

import logging
import time
import traceback
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from confidence_test.answer_metrics import parse_answer_classes
from confidence_test.inference_extension import ASSISTANT_ANSWER_PREFILL
from confidence_test.prompt_utils import (
    STAGE1_TEXT_ANSWER_PROMPT,
    STAGE2_TEXT_CONFIDENCE_PROMPT,
    V3_STAGE3_REANSWER_PROMPT,
    V3_STAGE4_META_CONFIDENCE_PROMPT,
    V4_STAGE1_FULL_EVIDENCE_ANSWER_PROMPT,
    V4_STAGE2_FULL_EVIDENCE_CONFIDENCE_PROMPT,
)
from confidence_test.source_attribution_prompt_utils import (
    V3_STAGE3_REANSWER_WITH_SOURCE_ATTRIBUTION_PROMPT,
    V3_STAGE4_META_SOURCE_ATTRIBUTION_PROMPT,
    V4_STAGE1_FULL_EVIDENCE_ANSWER_WITH_SOURCE_ATTRIBUTION_PROMPT,
    V4_STAGE2_FULL_EVIDENCE_SOURCE_ATTRIBUTION_PROMPT,
)
from confidence_test.source_attribution_schema import (
    ASSISTANT_SOURCE_ATTRIBUTION_PREFILL,
    SOURCE_ATTRIBUTION_CLASS_TEXT,
    SOURCE_ATTRIBUTION_CLASSES,
    SOURCE_ATTRIBUTION_MIDPOINTS,
)

from .attention_sinks import collect_attention_sinks
from .conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from .direct_readout import (
    answer_layer_readout,
    build_first_token_collision_report,
    project_hidden_to_vocab,
)
from .layer_stage_analyzer import (
    confidence_layer_readout_runtime,
    source_layer_readout,
    validate_restricted_reconstruction,
)
from .model_adapter import (
    LanguageModules,
    run_hooked_forward,
    run_logits_forward,
)
from .stage_specs import stage_spec
from .token_positions import (
    locate_field_value_span,
    locate_image_pad_span,
    locate_marker_in_assistant,
    locate_token_after_field,
)
from .token_spans import build_rendered_alignment


def _to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return deepcopy(value)
    if hasattr(value, "to_dict"):
        return deepcopy(value.to_dict())
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Unsupported result type: {type(value)!r}")


def _image_content(prompt: str, image_path: str | None) -> list[dict[str, str]]:
    if image_path is None:
        return [{"type": "text", "text": prompt}]
    resolved = Path(image_path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return [
        {"type": "image", "image": str(resolved)},
        {"type": "text", "text": prompt},
    ]


def _answer_ready(result: dict[str, Any]) -> bool:
    return bool(
        result.get("parse_success")
        and result.get("answer")
        and result.get("answer_metric_status") == "completed"
    )


def _confidence_label(result: dict[str, Any]) -> str:
    label = result.get("confidence_label")
    if not isinstance(label, str) or not label:
        raise RuntimeError("Confidence generation produced no usable label")
    return label


def reconstruction_tolerance(dtype_name: str) -> float:
    """Allow one BF16-scale logit step while retaining restricted checks."""
    return 0.1 if "bfloat16" in str(dtype_name).lower() else 1e-3


class CaseStageError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


class V3V4SourceRunner:
    """Serial runner sharing one model and processor across every case."""

    def __init__(
        self,
        *,
        inference: Any,
        modules: LanguageModules,
        confidence_analyzer: Any,
        base_confidence_analyzer: Any,
        source_analyzer: Any,
        joint_generator: Any,
        confidence_classes: list[str],
        confidence_midpoints: list[float],
        confidence_class_text: str,
        versions: list[str],
        attribution_mode: str,
        conditions: list[str],
        skip_attention: bool = False,
        skip_layer_readout: bool = False,
        max_answer_tokens: int = 24,
        logger: logging.Logger | None = None,
    ):
        self.inference = inference
        self.modules = modules
        self.confidence_analyzer = confidence_analyzer
        self.base_confidence = base_confidence_analyzer
        self.source_analyzer = source_analyzer
        self.joint_generator = joint_generator
        self.confidence_classes = list(confidence_classes)
        self.confidence_midpoints = [float(value) for value in confidence_midpoints]
        self.confidence_class_text = confidence_class_text
        self.versions = versions
        self.mode = attribution_mode
        self.conditions = conditions
        self.skip_attention = skip_attention
        self.skip_layer_readout = skip_layer_readout
        self.max_answer_tokens = max_answer_tokens
        self.logger = logger or logging.getLogger("v3_v4_source")
        self.processor = inference.processor
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.selected_layers = list(range(modules.num_hidden_layers))
        self.final_layer = modules.num_hidden_layers - 1
        self.tolerance = reconstruction_tolerance(inference.dtype_name)
        self.confidence_token_ids = {
            label: [int(token_id) for token_id in ids]
            for label, ids in self.base_confidence._class_token_variants.items()
        }
        for index, left in enumerate(self.confidence_classes):
            for right in self.confidence_classes[index + 1 :]:
                overlap = set(self.confidence_token_ids[left]).intersection(
                    self.confidence_token_ids[right]
                )
                if overlap:
                    raise RuntimeError(
                        f"Confidence first-token collision: {left!r} vs {right!r}: "
                        f"{sorted(overlap)}"
                    )
        self.source_token_ids = self.source_analyzer.token_specification.class_token_ids
        self.shared_initial: dict[str, dict[str, Any]] = {}
        self.call_counts: dict[str, int] = {
            "initial_answer": 0,
            "initial_confidence": 0,
            "answer": 0,
            "joint_answer_source": 0,
            "source_attribution": 0,
            "current_confidence": 0,
        }

    def seed_shared_initial(self, records: list[dict[str, Any]]) -> None:
        for record in records:
            if record.get("version") != "v3":
                continue
            generated = record.get("generated") or {}
            initial_answer = generated.get("initial_answer")
            initial_confidence = generated.get("initial_confidence")
            if isinstance(initial_answer, str) and isinstance(initial_confidence, dict):
                key = f"{record.get('item_id')}::{record.get('prior_index')}"
                self.shared_initial.setdefault(
                    key,
                    {
                        "answer": initial_answer,
                        "answer_result": deepcopy(generated.get("initial_answer_result")),
                        "confidence_result": deepcopy(initial_confidence),
                    },
                )

    def _log(self, event: str, **fields: Any) -> None:
        suffix = " ".join(f"{key}={value}" for key, value in fields.items())
        self.logger.info("%s%s", event, f" {suffix}" if suffix else "")

    def _run_answer(
        self,
        prompt: str,
        answer_classes: list[str],
        image_path: str | None,
    ) -> dict[str, Any]:
        self.call_counts["answer"] += 1
        return _to_dict(
            self.inference.generate_answer_with_metrics(
                prompt=prompt,
                answer_classes=answer_classes,
                image_path=image_path,
                max_new_tokens=self.max_answer_tokens,
            )
        )

    def _run_confidence(self, prompt: str, image_path: str | None) -> dict[str, Any]:
        started = time.perf_counter()
        result = _to_dict(self.confidence_analyzer.analyze_prompt(prompt, image_path))
        result.pop("rendered_prompt", None)
        result.pop("class_token_variants", None)
        result.pop("hidden_state_collected", None)
        result["status"] = "completed"
        result["error"] = None
        result["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        return result

    def _initial_v3(self, case: Any) -> dict[str, Any]:
        key = f"{case.item_id}::{case.prior_index}"
        if key in self.shared_initial:
            return deepcopy(self.shared_initial[key])
        answer_prompt = STAGE1_TEXT_ANSWER_PROMPT.format(
            question=case.question,
            text_clue=case.text_clue,
        )
        self.call_counts["initial_answer"] += 1
        answer_result = _to_dict(
            self.inference.generate_answer_with_metrics(
                prompt=answer_prompt,
                answer_classes=case.answer_classes,
                image_path=None,
                max_new_tokens=self.max_answer_tokens,
            )
        )
        if not _answer_ready(answer_result):
            raise CaseStageError(
                "v3_initial_answer",
                f"Initial answer failed: {answer_result.get('error')}",
            )
        confidence_prompt = STAGE2_TEXT_CONFIDENCE_PROMPT.format(
            question=case.question,
            text_clue=case.text_clue,
            answer=answer_result["answer"],
            classes=self.confidence_class_text,
        )
        self.call_counts["initial_confidence"] += 1
        confidence_result = self._run_confidence(confidence_prompt, None)
        _confidence_label(confidence_result)
        value = {
            "answer": answer_result["answer"],
            "answer_result": answer_result,
            "confidence_result": confidence_result,
        }
        self.shared_initial[key] = deepcopy(value)
        return value

    def _field_specifications(
        self,
        version: str,
        target: str,
        values: dict[str, str],
    ) -> dict[str, tuple[str, str, str]]:
        specification = stage_spec(version, self.mode, target)
        fields: dict[str, tuple[str, str, str]] = {}
        for source in specification.compared_sources:
            if source == "image":
                continue
            if source == "text_clue":
                fields[source] = ("Text clue:", values[source], "\n")
            elif source == "previous_answer":
                fields[source] = ("**Previous Answer**:", values[source], " ")
            elif source == "previous_confidence":
                fields[source] = ("**Previous Confidence**:", values[source], " ")
            elif source == "initial_answer":
                fields[source] = ("**Initial Answer**:", values[source], " ")
            elif source == "initial_confidence":
                fields[source] = ("**Initial Confidence**:", values[source], " ")
            elif source == "current_answer":
                if self.mode == "joint" and target == "sac":
                    prefix = "**Answer**:"
                elif version == "v3":
                    prefix = "**Current Answer**:"
                else:
                    prefix = "**Answer**:"
                fields[source] = (prefix, values[source], " ")
            else:
                raise ValueError(f"Unsupported source field {source!r}")
        return fields

    def _prepare_teacher_stage(
        self,
        *,
        name: str,
        prompt: str,
        assistant_text: str,
        image_path: str,
        version: str,
        targets: list[str],
        values: dict[str, str],
        panl_field: tuple[str, str] | None,
    ) -> dict[str, Any]:
        messages = [
            {"role": "user", "content": _image_content(prompt, image_path)},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
        ]
        rendered = render_continued_assistant(self.processor, messages, assistant_text)
        inputs = prepare_multimodal_inputs(
            self.processor,
            messages,
            rendered,
            device=self.inference._get_inputs_device(),
        )
        processed_ids = [int(value) for value in inputs.input_ids[0].tolist()]
        alignment = build_rendered_alignment(
            self.tokenizer,
            rendered,
            inputs.input_ids,
            inputs.attention_mask,
        )
        markers = {
            "ac": "**Answer**:",
            "cc": "**Confidence**:",
            "sac": ASSISTANT_SOURCE_ATTRIBUTION_PREFILL,
        }
        positions = {
            target: locate_marker_in_assistant(
                self.tokenizer,
                alignment.rendered_ids,
                assistant_text,
                markers[target],
                name=target,
                position_map=alignment.rendered_to_processed,
                processed_ids=processed_ids,
            )
            for target in targets
        }
        field_locations: dict[str, dict[str, Any]] = {}
        target_sources: dict[str, Any] = {}
        image_location = locate_image_pad_span(self.tokenizer, processed_ids)
        for target in targets:
            source_spans: dict[str, list[int]] = {}
            fields = self._field_specifications(version, target, values)
            for source in stage_spec(version, self.mode, target).compared_sources:
                if source == "image":
                    source_spans[source] = list(image_location["span"])
                    continue
                prefix, value, separator = fields[source]
                location_key = f"{source}|{prefix}|{value}"
                if location_key not in field_locations:
                    field_locations[location_key] = locate_field_value_span(
                        self.tokenizer,
                        alignment.rendered_ids,
                        prefix,
                        value,
                        separator=separator,
                        name=source,
                        position_map=alignment.rendered_to_processed,
                        processed_ids=processed_ids,
                    )
                source_spans[source] = list(field_locations[location_key]["span"])
            target_sources[target] = {
                "target_position": positions[target]["position"],
                "source_spans": source_spans,
            }
        panl = None
        if panl_field is not None:
            panl = locate_token_after_field(
                self.tokenizer,
                alignment.rendered_ids,
                panl_field[0],
                panl_field[1],
                name="panl",
                position_map=alignment.rendered_to_processed,
                processed_ids=processed_ids,
            )
        return {
            "name": name,
            "prompt": prompt,
            "assistant_text": assistant_text,
            "inputs": inputs,
            "positions": positions,
            "panl": panl,
            "target_sources": target_sources,
        }

    def _analyze_stages(
        self,
        *,
        stages: list[dict[str, Any]],
        answer_classes: list[str],
        case: Any,
        current_answer: str,
        generated_source: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
        direct = {"ac_layers": [], "cc_layers": [], "sac_layers": []}
        validation = {
            "ac_last_layer": None,
            "cc_last_layer": None,
            "sac_last_layer": None,
        }
        attention: dict[str, Any] = {}
        token_positions = {"ac": None, "panl": None, "cc": None, "sac": None}
        token_position_stages: dict[str, str] = {}
        answer_report = build_first_token_collision_report(self.tokenizer, answer_classes)
        if answer_report["collisions"]:
            raise RuntimeError(f"Answer first-token collisions: {answer_report['collisions']}")
        answer_token_ids = {
            label: answer_report["labels"][label]["first_token_variants"]
            for label in answer_classes
        }

        for stage in stages:
            inputs = stage["inputs"]
            target_positions = {
                target: int(record["position"])
                for target, record in stage["positions"].items()
            }
            for target, position in target_positions.items():
                token_positions[target] = position
                token_position_stages[target] = stage["name"]
            if stage["panl"] is not None:
                token_positions["panl"] = int(stage["panl"]["position"])
                token_position_stages["panl"] = stage["name"]

            forward = None
            reference_logits: dict[int, torch.Tensor] = {}
            needs_joint_source_score = (
                "sac" in target_positions
                and generated_source is not None
                and "class_probabilities" not in generated_source
            )
            if not self.skip_layer_readout:
                forward = run_hooked_forward(
                    self.inference.model,
                    inputs,
                    self.modules,
                    target_positions,
                    logits_positions=list(target_positions.values()),
                )
                reference_logits = forward.logits_by_position
                for target, position in target_positions.items():
                    hidden_by_layer = forward.hidden_by_name[target]
                    if target == "ac":
                        layers = [
                            answer_layer_readout(
                                layer,
                                hidden_by_layer[layer],
                                self.modules.final_norm,
                                self.modules.lm_head,
                                answer_classes,
                                answer_report,
                                case.ground_truth_answer,
                                current_answer,
                                case.conflict_answer,
                                case.text_answer,
                            )
                            for layer in self.selected_layers
                        ]
                        direct["ac_layers"].extend(layers)
                        midpoints = None
                        labels = answer_classes
                        token_ids = answer_token_ids
                    elif target == "cc":
                        layers = [
                            confidence_layer_readout_runtime(
                                layer,
                                hidden_by_layer[layer],
                                self.modules.final_norm,
                                self.modules.lm_head,
                                self.confidence_classes,
                                self.confidence_midpoints,
                                self.confidence_token_ids,
                            )
                            for layer in self.selected_layers
                        ]
                        direct["cc_layers"].extend(layers)
                        midpoints = self.confidence_midpoints
                        labels = self.confidence_classes
                        token_ids = self.confidence_token_ids
                    else:
                        layers = [
                            source_layer_readout(
                                layer,
                                hidden_by_layer[layer],
                                self.modules.final_norm,
                                self.modules.lm_head,
                                self.source_token_ids,
                            )
                            for layer in self.selected_layers
                        ]
                        direct["sac_layers"].extend(layers)
                        midpoints = SOURCE_ATTRIBUTION_MIDPOINTS
                        labels = SOURCE_ATTRIBUTION_CLASSES
                        token_ids = self.source_token_ids
                    reconstructed = project_hidden_to_vocab(
                        hidden_by_layer[self.final_layer],
                        self.modules.final_norm,
                        self.modules.lm_head,
                    )
                    check = validate_restricted_reconstruction(
                        reconstructed,
                        reference_logits[position],
                        labels=labels,
                        class_token_ids=token_ids,
                        midpoints=midpoints,
                        tolerance=self.tolerance,
                    )
                    validation[f"{target}_last_layer"] = check
                    if not check["passed"]:
                        raise RuntimeError(
                            f"{target.upper()} final-layer reconstruction failed: {check}"
                        )
                    del reconstructed
            elif needs_joint_source_score:
                reference_logits = run_logits_forward(
                    self.inference.model,
                    inputs,
                    [target_positions["sac"]],
                    self.modules,
                )

            if needs_joint_source_score:
                position = target_positions["sac"]
                scored = self.source_analyzer.score_vocab_logits(
                    reference_logits[position],
                    raw_output=str(generated_source["raw_output"]),
                    parsed_label=str(generated_source["parsed_label"]),
                )
                generated_source.clear()
                generated_source.update(scored.to_dict())

            if not self.skip_attention:
                stage_attention = collect_attention_sinks(
                    self.inference.model,
                    inputs,
                    stage["target_sources"],
                )
                overlap = set(attention).intersection(stage_attention)
                if overlap:
                    raise RuntimeError(f"Duplicate attention target(s): {sorted(overlap)}")
                attention.update(stage_attention)
            if forward is not None:
                del forward
            del inputs
            stage["inputs"] = None
        return direct, validation, attention, token_positions, token_position_stages

    def process_case(
        self,
        *,
        case: Any,
        condition: str,
        version: str,
    ) -> dict[str, Any]:
        case_id = (
            f"{case.item_id}__prior_{case.prior_index}__"
            f"{condition}__{version}__{self.mode}"
        )
        started = time.perf_counter()
        record: dict[str, Any] = {
            "case_id": case_id,
            "item_id": case.item_id,
            "prior_index": case.prior_index,
            "condition": condition,
            "version": version,
            "attribution_mode": self.mode,
            "ground_truths": {
                "answer": case.ground_truth_answer,
                "conflict_answer": case.conflict_answer,
            },
            "status": "running",
            "generated": {
                "initial_answer": None,
                "initial_answer_result": None,
                "initial_confidence": None,
                "current_answer": None,
                "current_answer_result": None,
                "current_confidence": None,
                "source_attribution": None,
            },
            "direct_readout": {"ac_layers": [], "cc_layers": [], "sac_layers": []},
            "token_positions": {"ac": None, "panl": None, "cc": None, "sac": None},
            "token_position_stages": {},
            "attention_sinks": {},
            "validation": {
                "ac_last_layer": None,
                "cc_last_layer": None,
                "sac_last_layer": None,
            },
            "model_structure": {
                "dtype": self.inference.dtype_name,
                "num_hidden_layers": self.modules.num_hidden_layers,
                "num_attention_heads": int(
                    getattr(
                        getattr(
                            self.inference.model.config,
                            "text_config",
                            self.inference.model.config,
                        ),
                        "num_attention_heads",
                    )
                ),
                "layer_definition": "decoder_block_output_pre_final_norm",
            },
            "parse_status": {},
            "error": None,
        }
        if version == "v3":
            record["text_answer"] = case.text_answer
        stage_name = "initialization"
        stages: list[dict[str, Any]] = []
        try:
            condition_input = case.conditions[condition]
            if condition_input.error:
                raise CaseStageError("image_resolution", str(condition_input.error))
            image_path = condition_input.resolved_image_path
            if not image_path:
                raise CaseStageError("image_resolution", "Condition has no resolved image")
            answer_classes = list(case.answer_classes or parse_answer_classes(case.question))
            values = {"text_clue": case.text_clue}
            initial = None
            if version == "v3":
                stage_name = "v3_initial"
                initial = self._initial_v3(case)
                initial_answer = str(initial["answer"])
                initial_confidence_label = _confidence_label(initial["confidence_result"])
                record["generated"]["initial_answer"] = initial_answer
                record["generated"]["initial_answer_result"] = deepcopy(initial["answer_result"])
                record["generated"]["initial_confidence"] = deepcopy(
                    initial["confidence_result"]
                )
                values.update(
                    {
                        "previous_answer": initial_answer,
                        "previous_confidence": initial_confidence_label,
                        "initial_answer": initial_answer,
                        "initial_confidence": initial_confidence_label,
                    }
                )

            stage_name = "answer_generation"
            if version == "v3":
                if self.mode == "joint":
                    answer_prompt = V3_STAGE3_REANSWER_WITH_SOURCE_ATTRIBUTION_PROMPT.format(
                        question=case.question,
                        text_clue=case.text_clue,
                        previous_answer=values["previous_answer"],
                        previous_confidence=values["previous_confidence"],
                        source_classes=SOURCE_ATTRIBUTION_CLASS_TEXT,
                    )
                else:
                    answer_prompt = V3_STAGE3_REANSWER_PROMPT.format(
                        question=case.question,
                        text_clue=case.text_clue,
                        previous_answer=values["previous_answer"],
                        previous_confidence=values["previous_confidence"],
                    )
            elif self.mode == "joint":
                answer_prompt = V4_STAGE1_FULL_EVIDENCE_ANSWER_WITH_SOURCE_ATTRIBUTION_PROMPT.format(
                    question=case.question,
                    text_clue=case.text_clue,
                    source_classes=SOURCE_ATTRIBUTION_CLASS_TEXT,
                )
            else:
                answer_prompt = V4_STAGE1_FULL_EVIDENCE_ANSWER_PROMPT.format(
                    question=case.question,
                    text_clue=case.text_clue,
                )

            generated_source: dict[str, Any] | None = None
            if self.mode == "joint":
                self.call_counts["joint_answer_source"] += 1
                answer_result = _to_dict(
                    self.joint_generator.generate(
                        answer_prompt,
                        answer_classes,
                        image_path,
                        max_new_tokens=max(self.max_answer_tokens + 8, 32),
                    )
                )
                generated_source = {
                    "raw_output": answer_result.get("raw_output", ""),
                    "hard_label_parsed": bool(answer_result.get("parse_success")),
                    "parsed_label": answer_result.get("source_label"),
                }
                record["generated"]["source_attribution"] = deepcopy(generated_source)
            else:
                answer_result = self._run_answer(
                    answer_prompt,
                    answer_classes,
                    image_path,
                )
            record["generated"]["current_answer_result"] = deepcopy(answer_result)
            if not _answer_ready(answer_result):
                raise CaseStageError(
                    "answer_generation",
                    f"Current answer failed: {answer_result.get('error')}",
                )
            current_answer = str(answer_result["answer"])
            values["current_answer"] = current_answer
            record["generated"]["current_answer"] = current_answer

            source_prompt = None
            if self.mode == "parallel":
                stage_name = "source_attribution_generation"
                if version == "v3":
                    source_prompt = V3_STAGE4_META_SOURCE_ATTRIBUTION_PROMPT.format(
                        question=case.question,
                        text_clue=case.text_clue,
                        initial_answer=values["initial_answer"],
                        initial_confidence=values["initial_confidence"],
                        stage3_answer=current_answer,
                        source_classes=SOURCE_ATTRIBUTION_CLASS_TEXT,
                    )
                else:
                    source_prompt = V4_STAGE2_FULL_EVIDENCE_SOURCE_ATTRIBUTION_PROMPT.format(
                        question=case.question,
                        text_clue=case.text_clue,
                        answer=current_answer,
                        source_classes=SOURCE_ATTRIBUTION_CLASS_TEXT,
                    )
                self.call_counts["source_attribution"] += 1
                generated_source = _to_dict(
                    self.source_analyzer.analyze_prompt(source_prompt, image_path)
                )
                if not generated_source.get("hard_label_parsed"):
                    record["generated"]["source_attribution"] = deepcopy(generated_source)
                    raise CaseStageError(
                        "source_attribution_parse",
                        f"Could not parse exact SA output: {generated_source.get('raw_output')!r}",
                    )

            stage_name = "current_confidence_generation"
            if version == "v3":
                confidence_prompt = V3_STAGE4_META_CONFIDENCE_PROMPT.format(
                    question=case.question,
                    text_clue=case.text_clue,
                    initial_answer=values["initial_answer"],
                    initial_confidence=values["initial_confidence"],
                    stage3_answer=current_answer,
                    classes=self.confidence_class_text,
                )
            else:
                confidence_prompt = V4_STAGE2_FULL_EVIDENCE_CONFIDENCE_PROMPT.format(
                    question=case.question,
                    text_clue=case.text_clue,
                    answer=current_answer,
                    classes=self.confidence_class_text,
                )
            self.call_counts["current_confidence"] += 1
            current_confidence = self._run_confidence(confidence_prompt, image_path)
            current_confidence_label = _confidence_label(current_confidence)
            record["generated"]["current_confidence"] = deepcopy(current_confidence)
            if generated_source is not None:
                record["generated"]["source_attribution"] = generated_source

            stage_name = "teacher_forced_analysis"
            if self.mode == "joint":
                joint_assistant = (
                    f"{ASSISTANT_ANSWER_PREFILL} {current_answer}\n"
                    f"{ASSISTANT_SOURCE_ATTRIBUTION_PREFILL}{generated_source['parsed_label']}"
                )
                stages.append(
                    self._prepare_teacher_stage(
                        name="joint_answer_source",
                        prompt=answer_prompt,
                        assistant_text=joint_assistant,
                        image_path=image_path,
                        version=version,
                        targets=["ac", "sac"],
                        values=values,
                        panl_field=("**Answer**:", current_answer),
                    )
                )
            else:
                stages.append(
                    self._prepare_teacher_stage(
                        name="answer",
                        prompt=answer_prompt,
                        assistant_text=f"{ASSISTANT_ANSWER_PREFILL} {current_answer}",
                        image_path=image_path,
                        version=version,
                        targets=["ac"],
                        values=values,
                        panl_field=None,
                    )
                )
                if self.mode == "parallel":
                    assert source_prompt is not None and generated_source is not None
                    stages.append(
                        self._prepare_teacher_stage(
                            name="source_attribution",
                            prompt=source_prompt,
                            assistant_text=(
                                f"{ASSISTANT_SOURCE_ATTRIBUTION_PREFILL}"
                                f"{generated_source['parsed_label']}"
                            ),
                            image_path=image_path,
                            version=version,
                            targets=["sac"],
                            values=values,
                            panl_field=None,
                        )
                    )
            confidence_panl = None
            if self.mode != "joint":
                confidence_panl = (
                    "**Current Answer**:" if version == "v3" else "**Answer**:",
                    current_answer,
                )
            stages.append(
                self._prepare_teacher_stage(
                    name="confidence",
                    prompt=confidence_prompt,
                    assistant_text=f"**Confidence**: {current_confidence_label}",
                    image_path=image_path,
                    version=version,
                    targets=["cc"],
                    values=values,
                    panl_field=confidence_panl,
                )
            )

            direct, validation, attention, positions, position_stages = self._analyze_stages(
                stages=stages,
                answer_classes=answer_classes,
                case=case,
                current_answer=current_answer,
                generated_source=generated_source,
            )
            record["direct_readout"] = direct
            record["validation"] = validation
            record["attention_sinks"] = attention
            record["token_positions"] = positions
            record["token_position_stages"] = position_stages
            if generated_source is not None:
                record["generated"]["source_attribution"] = generated_source
            record["parse_status"] = {
                "current_answer": bool(answer_result.get("parse_success")),
                "current_confidence": bool(current_confidence.get("hard_label_parsed")),
                "source_attribution": (
                    None
                    if generated_source is None
                    else bool(generated_source.get("hard_label_parsed"))
                ),
            }
            record["status"] = "completed"
        except Exception as exc:
            for teacher_stage in stages:
                inputs = teacher_stage.get("inputs")
                if inputs is not None:
                    del inputs
                    teacher_stage["inputs"] = None
            cause = exc.__cause__ or exc
            record["status"] = "failed"
            record["error"] = {
                "stage": getattr(exc, "stage", stage_name),
                "type": type(cause).__name__,
                "message": str(exc),
                "traceback": "".join(traceback.format_exception(exc)),
            }
            source = record["generated"].get("source_attribution")
            record["parse_status"] = {
                "current_answer": bool(
                    (record["generated"].get("current_answer_result") or {}).get(
                        "parse_success"
                    )
                ),
                "current_confidence": bool(
                    (record["generated"].get("current_confidence") or {}).get(
                        "hard_label_parsed"
                    )
                ),
                "source_attribution": (
                    None if source is None else bool(source.get("hard_label_parsed"))
                ),
            }
        record["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        self._log(
            "case_finished",
            case_id=case_id,
            status=record["status"],
            elapsed_seconds=record["elapsed_seconds"],
        )
        return record

    def run(
        self,
        cases: list[Any],
        *,
        existing_ids: set[str],
        commit: Callable[[dict[str, Any]], None],
    ) -> dict[str, int]:
        for case in cases:
            for condition in self.conditions:
                for version in self.versions:
                    case_id = (
                        f"{case.item_id}__prior_{case.prior_index}__"
                        f"{condition}__{version}__{self.mode}"
                    )
                    if case_id in existing_ids:
                        self._log("resume_skip", case_id=case_id)
                        continue
                    record = self.process_case(
                        case=case,
                        condition=condition,
                        version=version,
                    )
                    commit(record)
                    existing_ids.add(case_id)
        return dict(self.call_counts)
