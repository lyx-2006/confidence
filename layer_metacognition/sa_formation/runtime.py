"""GPU runtime and exact conversational reconstructions for Stage 3."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from confidence_test.dataset_utils import EvaluationCase, load_evaluation_cases
from confidence_test.inference_extension import ASSISTANT_ANSWER_PREFILL
from confidence_test.joint_answer_source_extension import JointAnswerSourceGenerator
from confidence_test.joint_answer_source_extension import JointAnswerSourceGenerationResult
from confidence_test.prompt_utils import STAGE1_TEXT_ANSWER_PROMPT
from confidence_test.source_attribution_analyzer import SourceAttributionAnalyzer
from confidence_test.source_attribution_schema import ASSISTANT_SOURCE_ATTRIBUTION_PREFILL
from confidence_test.source_attribution_variants import get_source_prompt_variant
from layer_metacognition.model_adapter import (
    AdditiveActivationHook,
    LanguageModules,
    load_qwen_inference,
    resolve_language_modules,
    run_logits_forward,
)
from layer_metacognition.probe.prompts import IMAGE_ONLY_ANSWER_PROMPT
from layer_metacognition.token_spans import build_rendered_alignment

from .core import FoldDirection, SAFormationArtifacts, canonical_message_hash


SOURCE_CHOICE_PROMPT = """Choose which source the fixed answer relied on more.

0=Text, 1=Image

Output exactly:

**Source Choice**:<CLASS>

CLASS must be 0 or 1. Do not include any additional text."""


def text_content(text: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text}]


def image_content(image_path: str, text: str) -> list[dict[str, str]]:
    path = Path(image_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return [{"type": "image", "image": str(path)}, {"type": "text", "text": text}]


def assistant_message(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text_content(text)}


def full_prompt(case: EvaluationCase) -> str:
    variant = get_source_prompt_variant("answer_basis_9")
    return variant.v4_joint_prompt.format(
        question=case.question,
        text_clue=case.text_clue,
        source_classes=variant.class_text,
    )


def build_history_messages(
    case: EvaluationCase,
    condition: str,
    history_side: str,
    initial_answer: str,
    *,
    assistant_text: str = ASSISTANT_ANSWER_PREFILL,
) -> list[dict[str, Any]]:
    image_path = str(case.conditions[condition].resolved_image_path)
    if history_side == "text_first":
        initial_prompt = STAGE1_TEXT_ANSWER_PROMPT.format(
            question=case.question,
            text_clue=case.text_clue,
        )
        first_user = {"role": "user", "content": text_content(initial_prompt)}
    elif history_side == "image_first":
        initial_prompt = IMAGE_ONLY_ANSWER_PROMPT.format(question=case.question)
        first_user = {"role": "user", "content": image_content(image_path, initial_prompt)}
    else:
        raise ValueError(f"Unknown history side: {history_side}")
    return [
        first_user,
        assistant_message(f"**Answer**: {initial_answer}"),
        {"role": "user", "content": image_content(image_path, full_prompt(case))},
        assistant_message(assistant_text),
    ]


def build_factorial_history_messages(
    case: EvaluationCase,
    condition: str,
    prior_modality: str,
    prior_answer: str,
    *,
    assistant_text: str = ASSISTANT_ANSWER_PREFILL,
) -> list[dict[str, Any]]:
    """Build a history branch with independently controlled modality/answer."""
    image_path = str(case.conditions[condition].resolved_image_path)
    if prior_modality == "text":
        initial_prompt = STAGE1_TEXT_ANSWER_PROMPT.format(
            question=case.question,
            text_clue=case.text_clue,
        )
        first_user = {"role": "user", "content": text_content(initial_prompt)}
    elif prior_modality == "image":
        initial_prompt = IMAGE_ONLY_ANSWER_PROMPT.format(question=case.question)
        first_user = {
            "role": "user",
            "content": image_content(image_path, initial_prompt),
        }
    else:
        raise ValueError(f"Unknown prior modality: {prior_modality}")
    return [
        first_user,
        assistant_message(f"**Answer**: {prior_answer}"),
        {"role": "user", "content": image_content(image_path, full_prompt(case))},
        assistant_message(assistant_text),
    ]


def build_no_history_messages(
    case: EvaluationCase,
    condition: str,
    *,
    assistant_text: str = ASSISTANT_ANSWER_PREFILL,
) -> list[dict[str, Any]]:
    image_path = str(case.conditions[condition].resolved_image_path)
    return [
        {"role": "user", "content": image_content(image_path, full_prompt(case))},
        assistant_message(assistant_text),
    ]


def source_prefix_from_generation(raw_output: str, source_label: str) -> str:
    match = re.search(
        r"(?s)^(.*\*\*Source Attribution\*\*:[ \t]*(?:<)?)"
        + re.escape(source_label)
        + r"(?:>)?\s*$",
        raw_output,
    )
    if match is None:
        raise ValueError(f"Cannot recover class-bearing SA prefix from {raw_output!r}")
    return match.group(1)


@dataclass
class PreparedMeasurement:
    messages: list[dict[str, Any]]
    rendered: str
    inputs: Any
    assistant_text: str
    answer: str
    panl_position: int
    target_position: int
    prefix_hash: str


def right_pad_measurement_inputs(
    prepared: PreparedMeasurement,
    target_length: int,
    *,
    pad_token_id: int,
) -> None:
    """Right-pad only language tokens so paired forwards have the same shape."""
    ids = prepared.inputs["input_ids"]
    mask = prepared.inputs["attention_mask"]
    current = int(ids.shape[1])
    target = int(target_length)
    if target < current:
        raise ValueError(f"Cannot pad length {current} down to {target}")
    if target == current:
        return
    width = target - current
    prepared.inputs["input_ids"] = torch.cat(
        [ids, torch.full((int(ids.shape[0]), width), int(pad_token_id), dtype=ids.dtype, device=ids.device)],
        dim=1,
    )
    prepared.inputs["attention_mask"] = torch.cat(
        [mask, torch.zeros((int(mask.shape[0]), width), dtype=mask.dtype, device=mask.device)],
        dim=1,
    )


def append_exact_token_ids(inputs: Any, token_ids: Sequence[int]) -> None:
    """Append exact generated language ids without touching multimodal tensors."""
    if not token_ids:
        raise ValueError("Exact generated token prefix is empty")
    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_tensor = torch.tensor([list(token_ids)], dtype=ids.dtype, device=ids.device)
    inputs["input_ids"] = torch.cat([ids, token_tensor], dim=1)
    inputs["attention_mask"] = torch.cat(
        [
            mask,
            torch.ones(
                (int(mask.shape[0]), len(token_ids)),
                dtype=mask.dtype,
                device=mask.device,
            ),
        ],
        dim=1,
    )
def _position_for_char(alignment: Any, char_index: int) -> int:
    positions = alignment.processed_tokens_for_char_span(char_index, char_index + 1)
    if len(set(positions)) != 1:
        raise ValueError(f"Character at {char_index} maps to multiple tokens: {positions}")
    return int(positions[0])


def prepare_measurement(
    generator: JointAnswerSourceGenerator,
    messages: list[dict[str, Any]],
    *,
    assistant_text: str,
    answer: str,
) -> PreparedMeasurement:
    rendered, inputs = generator.prepare_messages(messages, assistant_text=assistant_text)
    alignment = build_rendered_alignment(
        generator.tokenizer,
        rendered,
        inputs.input_ids,
        inputs.attention_mask,
    )
    answer_prefix = f"**Answer**: {answer}"
    if not assistant_text.startswith(answer_prefix):
        raise ValueError("Measurement assistant text does not start with fixed Answer")
    newline = len(rendered) - len(assistant_text) + len(answer_prefix)
    if newline >= len(rendered) or rendered[newline] != "\n":
        raise ValueError("PANL reconstruction has no exact newline after Answer")
    panl_position = _position_for_char(alignment, newline)
    target_position = len(alignment.processed_ids) - 1
    return PreparedMeasurement(
        messages=messages,
        rendered=rendered,
        inputs=inputs,
        assistant_text=assistant_text,
        answer=answer,
        panl_position=panl_position,
        target_position=target_position,
        prefix_hash=canonical_message_hash(messages),
    )


def prepare_exact_generated_measurement(
    generator: JointAnswerSourceGenerator,
    messages: list[dict[str, Any]],
    generated: JointAnswerSourceGenerationResult,
    *,
    assistant_text: str = ASSISTANT_ANSWER_PREFILL,
) -> PreparedMeasurement:
    """Replay the exact Pass-1 token prefix preceding the generated SA class.

    This deliberately avoids decoding a parsed Answer and re-tokenizing a
    canonical assistant prefix.  Only the original prompt is prepared again;
    the generated token ids are appended verbatim through the step immediately
    before the SA class token.
    """
    if generated.source_token_step is None or not generated.generated_token_ids:
        raise ValueError("Generation does not retain an SA token step and token ids")
    if generated.normalized_answer is None or generated.answer is None:
        raise ValueError("Generation does not contain a parsed Answer")
    source_step = int(generated.source_token_step)
    if source_step <= 0 or source_step >= len(generated.generated_token_ids):
        raise ValueError(f"Invalid generated SA token step: {source_step}")
    rendered_base, inputs = generator.prepare_messages(
        messages,
        assistant_text=assistant_text,
    )
    prefix_token_ids = generated.generated_token_ids[:source_step]
    append_exact_token_ids(inputs, prefix_token_ids)
    decoded_prefix = generator.tokenizer.decode(
        prefix_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    assistant_prefix = assistant_text + decoded_prefix
    rendered = rendered_base + decoded_prefix
    alignment = build_rendered_alignment(
        generator.tokenizer,
        rendered,
        inputs.input_ids,
        inputs.attention_mask,
    )
    answer_match = re.search(r"\*\*Answer\*\*:[^\r\n]*\r?\n", assistant_prefix)
    if answer_match is None:
        raise ValueError("Exact generated prefix has no newline after Answer")
    newline_in_assistant = answer_match.end() - 1
    newline_in_rendered = len(rendered_base) - len(assistant_text) + newline_in_assistant
    panl_position = _position_for_char(alignment, newline_in_rendered)
    target_position = len(alignment.processed_ids) - 1
    return PreparedMeasurement(
        messages=messages,
        rendered=rendered,
        inputs=inputs,
        assistant_text=assistant_prefix,
        answer=str(generated.normalized_answer),
        panl_position=panl_position,
        target_position=target_position,
        prefix_hash=canonical_message_hash(
            [
                *messages,
                {
                    "role": "exact_generated_token_prefix",
                    "token_ids": prefix_token_ids,
                },
            ]
        ),
    )
def prepare_policy_measurement(
    generator: JointAnswerSourceGenerator,
    messages: list[dict[str, Any]],
    *,
    assistant_text: str,
    fixed_answer: str,
) -> PreparedMeasurement:
    """Locate the post-Answer newline in an earlier assistant turn.

    The target logits remain at the final Source Choice continuation.  Causal
    masking makes the earlier PANL state identical to the SA continuation when
    the canonical prefix through Answer is identical.
    """
    rendered, inputs = generator.prepare_messages(messages, assistant_text=assistant_text)
    alignment = build_rendered_alignment(
        generator.tokenizer,
        rendered,
        inputs.input_ids,
        inputs.attention_mask,
    )
    needle = f"**Answer**: {fixed_answer}\n"
    occurrence = rendered.rfind(needle)
    if occurrence < 0:
        raise ValueError("Policy reconstruction has no fixed Answer newline")
    newline = occurrence + len(needle) - 1
    panl_position = _position_for_char(alignment, newline)
    return PreparedMeasurement(
        messages=messages,
        rendered=rendered,
        inputs=inputs,
        assistant_text=assistant_text,
        answer=fixed_answer,
        panl_position=panl_position,
        target_position=len(alignment.processed_ids) - 1,
        prefix_hash=canonical_message_hash(messages),
    )


@dataclass
class MeasuredState:
    source: dict[str, Any]
    hidden: np.ndarray
    z_sa: float
    applied_delta_z: float
    expected_delta_z: float
    injection_l2: float
    hook_call_count: int
    applied_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "z_sa": self.z_sa,
            "applied_delta_z": self.applied_delta_z,
            "expected_delta_z": self.expected_delta_z,
            "injection_l2": self.injection_l2,
            "hook_call_count": self.hook_call_count,
            "applied_count": self.applied_count,
        }


class Stage3Runtime:
    def __init__(self, artifacts: SAFormationArtifacts) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Stage 3 formal experiments")
        self.inference = load_qwen_inference(
            str(artifacts.model_path),
            str(artifacts.inference_path),
        )
        self.model = self.inference.model
        self.modules: LanguageModules = resolve_language_modules(self.model)
        self.generator = JointAnswerSourceGenerator(self.inference)
        self.source_analyzer = SourceAttributionAnalyzer(self.inference)
        self.policy_analyzer = SourceAttributionAnalyzer(
            self.inference,
            source_classes=["0", "1"],
            source_midpoints=[0.0, 1.0],
        )
        cases, self.dataset_metadata = load_evaluation_cases(artifacts.dataset)
        self.cases = {(case.item_id, case.prior_index): case for case in cases}

    def case(self, item_id: str, prior_index: int) -> EvaluationCase:
        return self.cases[(str(item_id), int(prior_index))]

    def measure(
        self,
        prepared: PreparedMeasurement,
        direction: FoldDirection,
        *,
        steering_vector: np.ndarray | None = None,
        policy: bool = False,
        analyzer: Any | None = None,
    ) -> MeasuredState:
        vector = np.zeros(self.modules.hidden_size, dtype=np.float64) if steering_vector is None else np.asarray(steering_vector, dtype=np.float64)
        if vector.shape != (self.modules.hidden_size,):
            raise ValueError(f"Steering vector has wrong shape: {vector.shape}")
        input_length = int(prepared.inputs.input_ids.shape[1])
        hook = AdditiveActivationHook(
            self.modules,
            layer_index=18,
            target_position=prepared.panl_position,
            steering_vector=torch.from_numpy(vector),
            prefill_sequence_length=input_length,
        )
        with hook:
            logits = run_logits_forward(
                self.model,
                prepared.inputs,
                [prepared.target_position],
                self.modules,
            )[prepared.target_position]
        hook.validate_applied_once()
        assert hook.h_before is not None and hook.h_after is not None
        before = hook.h_before.numpy().astype(np.float64, copy=False)
        after = hook.h_after.numpy().astype(np.float64, copy=False)
        selected_analyzer = (
            analyzer
            if analyzer is not None
            else (self.policy_analyzer if policy else self.source_analyzer)
        )
        source = selected_analyzer.score_vocab_logits(
            logits,
            raw_output="",
            parsed_label=None,
        ).to_dict()
        applied_delta_z = float((after - before) @ direction.d_unit)
        expected_delta_z = float(vector @ direction.d_unit)
        tolerance = max(0.125, abs(expected_delta_z) * 0.05)
        if abs(applied_delta_z - expected_delta_z) > tolerance:
            raise RuntimeError(
                f"Coordinate manipulation failed: actual={applied_delta_z}, expected={expected_delta_z}, tolerance={tolerance}"
            )
        return MeasuredState(
            source=source,
            hidden=after,
            z_sa=float(after @ direction.d_unit),
            applied_delta_z=applied_delta_z,
            expected_delta_z=expected_delta_z,
            injection_l2=float(np.linalg.norm(after - before)),
            hook_call_count=hook.hook_call_count,
            applied_count=hook.applied_count,
        )

    def release_inputs(self, *prepared: PreparedMeasurement) -> None:
        for context in prepared:
            del context.inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def clone_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return copy.deepcopy(list(messages))
