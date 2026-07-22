"""Reliable AC, PANL, CC, clue, and image token location."""

from __future__ import annotations

import math
from typing import Any

from .token_spans import RenderedTokenAlignment, unique_text_span, verify_decoded_span


def _token_context(tokenizer: Any, ids: list[int], position: int, radius: int = 8) -> str:
    start = max(0, position - radius)
    end = min(len(ids), position + radius + 1)
    return tokenizer.decode(ids[start:end], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def token_position_record(tokenizer: Any, alignment: RenderedTokenAlignment, position: int) -> dict[str, Any]:
    token_id = int(alignment.processed_ids[position])
    return {
        "position": int(position),
        "token_id": token_id,
        "token_text": tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False),
        "validation_context": _token_context(tokenizer, alignment.processed_ids, position),
    }


def locate_suffix_colon(
    tokenizer: Any,
    alignment: RenderedTokenAlignment,
    assistant_content: str,
) -> dict[str, Any]:
    if not alignment.rendered.endswith(assistant_content):
        raise ValueError(f"Rendered prompt does not end with {assistant_content!r}")
    colon_relative = assistant_content.find(":")
    if colon_relative < 0 or assistant_content.find(":", colon_relative + 1) >= 0:
        raise ValueError(f"Assistant content must contain exactly one colon: {assistant_content!r}")
    char_position = len(alignment.rendered) - len(assistant_content) + colon_relative
    token_positions = alignment.processed_tokens_for_char_span(char_position, char_position + 1)
    if len(set(token_positions)) != 1:
        raise ValueError("Assistant prefill colon maps to multiple tokens")
    position = token_positions[0]
    record = token_position_record(tokenizer, alignment, position)
    if ":" not in record["token_text"]:
        raise ValueError(f"Located prefill token does not contain a colon: {record}")
    return record


def locate_cc(tokenizer: Any, alignment: RenderedTokenAlignment, assistant_prefill: str) -> dict[str, Any]:
    record = locate_suffix_colon(tokenizer, alignment, assistant_prefill)
    if record["position"] != len(alignment.processed_ids) - 1:
        raise ValueError("CC is not the final valid input token")
    return record


def locate_panl(tokenizer: Any, alignment: RenderedTokenAlignment, answer: str) -> dict[str, Any]:
    left = f"**Answer**: {answer}"
    bounded = f"{left}\n\nClassify"
    bounded_start, _ = unique_text_span(alignment.rendered, bounded)
    newline_start = bounded_start + len(left)
    positions = alignment.processed_tokens_for_char_span(newline_start, newline_start + 2)
    position = positions[-1]
    record = token_position_record(tokenizer, alignment, position)
    record["span_start_position"] = positions[0]
    record["span_end_position"] = positions[-1]
    record["decoded_span"] = verify_decoded_span(
        tokenizer,
        alignment.processed_ids[positions[0] : positions[-1] + 1],
        "\n\n",
    )
    return record


def locate_text_clue(
    tokenizer: Any,
    alignment: RenderedTokenAlignment,
    text_clue: str,
    right_boundary: str,
) -> dict[str, Any]:
    bounded = f"Text clue:\n{text_clue}\n\n{right_boundary}"
    bounded_start, _ = unique_text_span(alignment.rendered, bounded)
    clue_start = bounded_start + len("Text clue:\n")
    positions = alignment.processed_tokens_for_char_span(clue_start, clue_start + len(text_clue))
    decoded = verify_decoded_span(
        tokenizer,
        alignment.processed_ids[positions[0] : positions[-1] + 1],
        text_clue,
    )
    return {
        "start_position": positions[0],
        "end_position": positions[-1],
        "token_ids": alignment.processed_ids[positions[0] : positions[-1] + 1],
        "token_text": decoded,
        "validation_context": _token_context(tokenizer, alignment.processed_ids, positions[0]),
    }


def locate_image_span(
    tokenizer: Any,
    processor: Any,
    alignment: RenderedTokenAlignment,
    image_grid_thw: Any,
) -> dict[str, Any]:
    start_id = int(tokenizer.convert_tokens_to_ids("<|vision_start|>"))
    image_id = int(tokenizer.convert_tokens_to_ids("<|image_pad|>"))
    end_id = int(tokenizer.convert_tokens_to_ids("<|vision_end|>"))
    starts = [index for index, value in enumerate(alignment.processed_ids) if value == start_id]
    ends = [index for index, value in enumerate(alignment.processed_ids) if value == end_id]
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError(f"Expected one image span, found starts={starts}, ends={ends}")
    start, end = starts[0], ends[0]
    pad_positions = [index for index in range(start + 1, end) if alignment.processed_ids[index] == image_id]
    if pad_positions != list(range(start + 1, end)):
        raise ValueError("Image span is not a contiguous run of processor image-pad tokens")
    grid = image_grid_thw[0].tolist() if hasattr(image_grid_thw[0], "tolist") else list(image_grid_thw[0])
    merge_size = int(getattr(processor.image_processor, "merge_size", 0))
    if merge_size < 1:
        raise ValueError("Processor image merge_size is unavailable")
    expected = math.prod(int(value) for value in grid) // (merge_size**2)
    if len(pad_positions) != expected:
        raise ValueError(f"Image-pad count mismatch: observed={len(pad_positions)}, expected={expected}")
    return {
        "start_position": start,
        "end_position": end,
        "image_token_start": pad_positions[0],
        "image_token_end": pad_positions[-1],
        "image_token_id": image_id,
        "image_token_count": len(pad_positions),
        "image_grid_thw": [int(value) for value in grid],
        "token_text": "<|vision_start|>...<|vision_end|>",
        "validation_context": _token_context(tokenizer, alignment.processed_ids, start, radius=2),
    }
