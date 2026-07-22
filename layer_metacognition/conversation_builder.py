"""Qwen chat messages, assistant prefills, and multimodal input preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .prompts import (
    ASSISTANT_ANSWER_PREFILL,
    ASSISTANT_CONFIDENCE_PREFILL,
    STAGE1_MULTIMODAL_ANSWER_PROMPT,
    STAGE2_CONFIDENCE_PROMPT,
)


def _content(text: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text}]


def _image_user_content(image_path: str, text: str) -> list[dict[str, str]]:
    resolved = Path(image_path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return [
        {"type": "image", "image": str(resolved)},
        {"type": "text", "text": text},
    ]


def build_stage1_messages(
    question: str,
    text_clue: str,
    image_path: str,
    answer: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    prompt = STAGE1_MULTIMODAL_ANSWER_PROMPT.format(question=question, text_clue=text_clue)
    assistant = ASSISTANT_ANSWER_PREFILL if answer is None else f"{ASSISTANT_ANSWER_PREFILL} {answer}"
    return [
        {"role": "user", "content": _image_user_content(image_path, prompt)},
        {"role": "assistant", "content": _content(assistant)},
    ], assistant


def build_stage2_messages(
    question: str,
    text_clue: str,
    answer: str,
    image_path: str,
    classes: str,
) -> tuple[list[dict[str, Any]], str]:
    prompt = STAGE2_CONFIDENCE_PROMPT.format(
        question=question,
        text_clue=text_clue,
        answer=answer,
        classes=classes,
    )
    return [
        {"role": "user", "content": _image_user_content(image_path, prompt)},
        {"role": "assistant", "content": _content(ASSISTANT_CONFIDENCE_PREFILL)},
    ], ASSISTANT_CONFIDENCE_PREFILL


def render_continued_assistant(processor: Any, messages: list[dict[str, Any]], expected_suffix: str) -> str:
    rendered: str | None = None
    try:
        rendered = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
    except (TypeError, ValueError):
        rendered = None
    if not rendered or not rendered.endswith(expected_suffix):
        rendered = processor.apply_chat_template(
            messages[:1],
            tokenize=False,
            add_generation_prompt=True,
        ) + expected_suffix
    if not rendered.endswith(expected_suffix):
        raise RuntimeError(f"Rendered prompt does not end with exact assistant content {expected_suffix!r}")
    trailing = rendered[len(rendered) - len(expected_suffix) :]
    if trailing != expected_suffix:
        raise RuntimeError("Assistant prefill was normalized by the chat template")
    return rendered


def prepare_multimodal_inputs(
    processor: Any,
    messages: list[dict[str, Any]],
    rendered_prompt: str,
    device: Any | None = None,
) -> Any:
    from qwen_vl_utils import process_vision_info

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[rendered_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs.to(device) if device is not None else inputs
