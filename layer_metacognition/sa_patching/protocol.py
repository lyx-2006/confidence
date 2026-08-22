"""Shared answer-fixed prompt preparation and exact token-span location."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from layer_metacognition.sa_steering.runner import RuntimeCase
from layer_metacognition.sa_steering.sa_steering_hook import (
    fixed_answer_assistant_prefix,
    locate_sa_steering_position,
)
from layer_metacognition.token_positions import (
    locate_field_value_span,
    locate_image_pad_span,
)
from layer_metacognition.token_spans import build_rendered_alignment


@dataclass
class PreparedFixedPrefix:
    messages: list[dict[str, Any]]
    assistant_text: str
    rendered: str
    inputs: Any
    spans: dict[str, tuple[int, ...]]
    positions: dict[str, int]
    position_details: dict[str, dict[str, Any]]


def fixed_messages(
    prompt: str,
    image_path: str,
    assistant_text: str,
) -> list[dict[str, Any]]:
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


def prepare_fixed_prefix(
    joint_generator: Any,
    runtime_case: RuntimeCase,
    answer: str,
    *,
    positions: Sequence[str],
) -> PreparedFixedPrefix:
    assistant_text = fixed_answer_assistant_prefix(answer)
    messages = fixed_messages(runtime_case.prompt, runtime_case.image_path, assistant_text)
    rendered, inputs = joint_generator.prepare_messages(
        messages,
        assistant_text=assistant_text,
    )
    return locate_fixed_inputs(
        joint_generator,
        runtime_case,
        answer,
        messages=messages,
        assistant_text=assistant_text,
        rendered=rendered,
        inputs=inputs,
        positions=positions,
    )


def locate_fixed_inputs(
    joint_generator: Any,
    runtime_case: RuntimeCase,
    answer: str,
    *,
    messages: list[dict[str, Any]],
    assistant_text: str,
    rendered: str,
    inputs: Any,
    positions: Sequence[str],
) -> PreparedFixedPrefix:
    """Locate all intervention spans in inputs prepared by the shared generator."""

    input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
    attention_mask = (
        inputs["attention_mask"] if isinstance(inputs, dict) else inputs.attention_mask
    )
    if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
        raise ValueError("SA patching requires a single prepared case")
    processed_ids = [int(value) for value in input_ids[0].tolist()]
    alignment = build_rendered_alignment(
        joint_generator.tokenizer,
        rendered,
        input_ids,
        attention_mask,
    )
    image = locate_image_pad_span(joint_generator.tokenizer, processed_ids)
    text = locate_field_value_span(
        joint_generator.tokenizer,
        alignment.rendered_ids,
        "Text clue:",
        runtime_case.evaluation.text_clue,
        separator="\n",
        name="text clue embedding",
        position_map=alignment.rendered_to_processed,
        processed_ids=processed_ids,
    )
    answer_span = locate_field_value_span(
        joint_generator.tokenizer,
        alignment.rendered_ids,
        "**Answer**:",
        answer,
        separator=" ",
        name="fixed answer embedding",
        position_map=alignment.rendered_to_processed,
        processed_ids=processed_ids,
    )
    located: dict[str, int] = {}
    details: dict[str, dict[str, Any]] = {}
    for position in positions:
        index, detail = locate_sa_steering_position(
            tokenizer=joint_generator.tokenizer,
            rendered=rendered,
            inputs=inputs,
            assistant_text=assistant_text,
            answer=answer,
            position=position,
        )
        located[position] = index
        details[position] = detail
    spans = {
        "image": tuple(range(int(image["span"][0]), int(image["span"][1]))),
        "text": tuple(range(int(text["span"][0]), int(text["span"][1]))),
        "answer": tuple(
            range(int(answer_span["span"][0]), int(answer_span["span"][1]))
        ),
    }
    if any(not span for span in spans.values()):
        raise ValueError("Located an empty corruption embedding span")
    return PreparedFixedPrefix(
        messages=messages,
        assistant_text=assistant_text,
        rendered=rendered,
        inputs=inputs,
        spans=spans,
        positions=located,
        position_details=details,
    )
