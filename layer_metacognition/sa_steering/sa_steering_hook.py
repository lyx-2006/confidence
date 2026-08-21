"""Validated answer-fixed SA token location and additive activation hook."""

from __future__ import annotations

from typing import Any

import torch

from confidence_test.inference_extension import ASSISTANT_ANSWER_PREFILL
from confidence_test.source_attribution_schema import (
    ASSISTANT_SOURCE_ATTRIBUTION_PREFILL,
)
from layer_metacognition.model_adapter import AdditiveActivationHook, LanguageModules
from layer_metacognition.token_positions import (
    locate_last_answer_token,
    locate_marker_in_assistant,
    locate_token_after_field,
)
from layer_metacognition.token_spans import build_rendered_alignment

from . import POSITIONS


def fixed_answer_assistant_prefix(answer: str) -> str:
    answer = str(answer).strip()
    if not answer or "\n" in answer or "\r" in answer:
        raise ValueError("Fixed answer must be non-empty and single-line")
    return (
        f"{ASSISTANT_ANSWER_PREFILL} {answer}\n"
        f"{ASSISTANT_SOURCE_ATTRIBUTION_PREFILL}"
    )


def locate_sa_steering_position(
    *,
    tokenizer: Any,
    rendered: str,
    inputs: Any,
    assistant_text: str,
    answer: str,
    position: str,
) -> tuple[int, dict[str, Any]]:
    """Locate AC/LAT/PANL/SAC in one shared answer-fixed prefill."""

    if position not in POSITIONS:
        raise ValueError(f"Unsupported SA steering position: {position!r}")
    input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
    if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
        raise ValueError(
            "Answer-fixed SA steering requires batch size 1; got "
            f"input_ids shape={tuple(input_ids.shape)}"
        )
    processed_ids = [int(value) for value in input_ids[0].tolist()]
    alignment = build_rendered_alignment(
        tokenizer,
        rendered,
        input_ids,
        inputs["attention_mask"] if isinstance(inputs, dict) else inputs.attention_mask,
    )
    if position == "ac":
        detail = locate_marker_in_assistant(
            tokenizer,
            alignment.rendered_ids,
            assistant_text,
            ASSISTANT_ANSWER_PREFILL,
            name="ac",
            position_map=alignment.rendered_to_processed,
            processed_ids=processed_ids,
        )
    elif position == "panl":
        detail = locate_token_after_field(
            tokenizer,
            alignment.rendered_ids,
            ASSISTANT_ANSWER_PREFILL,
            answer,
            name="panl",
            position_map=alignment.rendered_to_processed,
            processed_ids=processed_ids,
        )
    elif position == "lat":
        panl = locate_token_after_field(
            tokenizer,
            alignment.rendered_ids,
            ASSISTANT_ANSWER_PREFILL,
            answer,
            name="lat_panl",
            position_map=alignment.rendered_to_processed,
            processed_ids=processed_ids,
        )
        detail = locate_last_answer_token(
            tokenizer,
            alignment.rendered_ids,
            ASSISTANT_ANSWER_PREFILL,
            answer,
            panl_position=int(panl["position"]),
            name="lat",
            position_map=alignment.rendered_to_processed,
            processed_ids=processed_ids,
        )
    else:
        detail = locate_marker_in_assistant(
            tokenizer,
            alignment.rendered_ids,
            assistant_text,
            ASSISTANT_SOURCE_ATTRIBUTION_PREFILL,
            name="sac",
            position_map=alignment.rendered_to_processed,
            processed_ids=processed_ids,
        )
    return int(detail["position"]), detail


class SAActivationAdditionHook(AdditiveActivationHook):
    """The existing additive hook with strict single-case batch validation."""

    def __init__(
        self,
        modules: LanguageModules,
        *,
        layer_index: int,
        target_position: int,
        steering_vector: torch.Tensor,
        prefill_sequence_length: int,
    ) -> None:
        vector = steering_vector.detach()
        if vector.ndim != 1:
            raise ValueError(
                f"SA steering vector must be rank 1, got {tuple(vector.shape)}"
            )
        super().__init__(
            modules,
            layer_index=layer_index,
            target_position=target_position,
            steering_vector=vector,
            prefill_sequence_length=prefill_sequence_length,
            injection_site="block_output",
        )

    def _patch_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            return super()._patch_tensor(tensor)
        if int(tensor.shape[0]) != 1:
            raise ValueError(
                "Answer-fixed SA steering requires decoder batch size 1; got "
                f"shape={tuple(tensor.shape)}"
            )
        return super()._patch_tensor(tensor)

