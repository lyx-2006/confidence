from __future__ import annotations

from pathlib import Path
from typing import Any

from layer_metacognition.conversation_builder import (
    prepare_multimodal_inputs,
    render_continued_assistant,
)

from .config import VARIANTS
from .prompts import ANSWER_PREFILL, SA_PREFILL, attribution_prompt


def _text(text: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text}]


def stage1_messages(prompt: str, image_path: str) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": [
            {"type": "image", "image": str(Path(image_path).resolve())},
            {"type": "text", "text": prompt},
        ]},
        {"role": "assistant", "content": _text(ANSWER_PREFILL)},
    ]


def history_content(raw_output: str, variant: str) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown conversation variant: {variant!r}")
    if not raw_output.startswith(ANSWER_PREFILL):
        raise ValueError("Phase-0 raw output does not start with the answer prefill")
    return raw_output if variant == "native_boundary" else raw_output + "\n"


def stage2_messages(
    phase0_prompt_text: str,
    image_path: str,
    phase0_raw_output: str,
    variant: str,
) -> list[dict[str, Any]]:
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": str(Path(image_path).resolve())},
            {"type": "text", "text": phase0_prompt_text},
        ]},
        {"role": "assistant", "content": _text(history_content(phase0_raw_output, variant))},
        {"role": "user", "content": _text(attribution_prompt())},
        {"role": "assistant", "content": _text(SA_PREFILL)},
    ]
    image_count = sum(
        item.get("type") == "image"
        for message in messages
        for item in message.get("content", [])
        if isinstance(item, dict)
    )
    if image_count != 1:
        raise AssertionError(f"Stage-2 conversation must contain exactly one image, got {image_count}")
    return messages


def render_stage1(processor: Any, messages: list[dict[str, Any]]) -> str:
    return render_continued_assistant(processor, messages, ANSWER_PREFILL)


def render_stage2(processor: Any, messages: list[dict[str, Any]]) -> str:
    return render_continued_assistant(processor, messages, SA_PREFILL)


__all__ = [
    "history_content", "prepare_multimodal_inputs", "render_stage1", "render_stage2",
    "stage1_messages", "stage2_messages",
]

