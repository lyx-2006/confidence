"""Character-to-token alignment after Qwen image placeholder expansion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RenderedTokenAlignment:
    rendered: str
    rendered_ids: list[int]
    processed_ids: list[int]
    offsets: list[tuple[int, int]]
    rendered_to_processed: dict[int, int]

    def processed_tokens_for_char_span(self, start: int, end: int) -> list[int]:
        if not (0 <= start < end <= len(self.rendered)):
            raise ValueError(f"Invalid character span [{start}, {end})")
        rendered_positions = [
            index
            for index, (token_start, token_end) in enumerate(self.offsets)
            if token_end > token_start and token_start < end and token_end > start
        ]
        if not rendered_positions:
            raise ValueError(f"Character span [{start}, {end}) maps to no tokens")
        try:
            return [self.rendered_to_processed[index] for index in rendered_positions]
        except KeyError as exc:
            raise ValueError(f"Character span maps through an unaligned token: {exc}") from exc


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return list(value)


def build_rendered_alignment(
    tokenizer: Any,
    rendered: str,
    input_ids: Any,
    attention_mask: Any | None = None,
) -> RenderedTokenAlignment:
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    rendered_ids = [int(value) for value in _as_list(encoded["input_ids"])]
    offsets = [tuple(map(int, pair)) for pair in _as_list(encoded["offset_mapping"])]
    processed_ids = [int(value) for value in _as_list(input_ids)]
    if attention_mask is not None:
        mask = [int(value) for value in _as_list(attention_mask)]
        processed_ids = [token_id for token_id, keep in zip(processed_ids, mask) if keep]

    image_token_id = int(tokenizer.convert_tokens_to_ids("<|image_pad|>"))
    mapping: dict[int, int] = {}
    source_index = 0
    processed_index = 0
    while source_index < len(rendered_ids) and processed_index < len(processed_ids):
        source_id = rendered_ids[source_index]
        if source_id != processed_ids[processed_index]:
            raise ValueError(
                "Rendered-token alignment failed at "
                f"rendered[{source_index}]={source_id}, processed[{processed_index}]={processed_ids[processed_index]}"
            )
        mapping[source_index] = processed_index
        source_index += 1
        processed_index += 1
        if source_id == image_token_id:
            while processed_index < len(processed_ids) and processed_ids[processed_index] == image_token_id:
                processed_index += 1
    if source_index != len(rendered_ids) or processed_index != len(processed_ids):
        raise ValueError(
            f"Rendered-token alignment did not consume both sequences: "
            f"{source_index}/{len(rendered_ids)}, {processed_index}/{len(processed_ids)}"
        )
    return RenderedTokenAlignment(rendered, rendered_ids, processed_ids, offsets, mapping)


def unique_text_span(rendered: str, exact_text: str) -> tuple[int, int]:
    starts: list[int] = []
    cursor = 0
    while True:
        position = rendered.find(exact_text, cursor)
        if position < 0:
            break
        starts.append(position)
        cursor = position + 1
    if len(starts) != 1:
        raise ValueError(f"Expected one occurrence of {exact_text!r}, found {len(starts)}")
    return starts[0], starts[0] + len(exact_text)


def normalize_tokenizer_whitespace(text: str) -> str:
    return " ".join(text.split())


def verify_decoded_span(tokenizer: Any, ids: list[int], expected: str) -> str:
    decoded = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if normalize_tokenizer_whitespace(decoded) != normalize_tokenizer_whitespace(expected):
        raise ValueError(f"Decoded token span {decoded!r} does not match expected text {expected!r}")
    return decoded
