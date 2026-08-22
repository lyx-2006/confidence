"""Reliable cognition-token and half-open source-span location."""

from __future__ import annotations

import math
from typing import Any

from .token_spans import RenderedTokenAlignment, unique_text_span, verify_decoded_span


def encode_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    return [int(value) for value in encoded]


def find_subsequence_positions(sequence: list[int], pattern: list[int]) -> list[int]:
    if not pattern:
        raise ValueError("Cannot locate an empty token subsequence")
    width = len(pattern)
    return [
        index
        for index in range(0, len(sequence) - width + 1)
        if sequence[index : index + width] == pattern
    ]


def unique_subsequence(
    sequence: list[int],
    pattern: list[int],
    *,
    name: str,
) -> tuple[int, int]:
    starts = find_subsequence_positions(sequence, pattern)
    if len(starts) != 1:
        raise ValueError(
            f"Expected exactly one {name} token subsequence, found {len(starts)}; "
            f"pattern={pattern}, starts={starts}, sequence_length={len(sequence)}"
        )
    return starts[0], starts[0] + len(pattern)


def locate_marker_in_assistant(
    tokenizer: Any,
    token_ids: list[int],
    assistant_text: str,
    marker: str,
    *,
    name: str,
    position_map: dict[int, int] | None = None,
    processed_ids: list[int] | None = None,
    assistant_occurrence: str = "unique",
) -> dict[str, Any]:
    """Locate a marker in a unique assistant output or final generation suffix."""

    assistant_ids = encode_without_special_tokens(tokenizer, assistant_text)
    if assistant_occurrence == "unique":
        assistant_start, assistant_end = unique_subsequence(
            token_ids,
            assistant_ids,
            name=f"{name} assistant output",
        )
    elif assistant_occurrence == "final_suffix":
        assistant_end = len(token_ids)
        assistant_start = assistant_end - len(assistant_ids)
        if assistant_start < 0 or token_ids[assistant_start:assistant_end] != assistant_ids:
            raise ValueError(
                f"Expected {name} assistant output to be the final token suffix; "
                f"pattern={assistant_ids}, sequence_length={len(token_ids)}"
            )
    else:
        raise ValueError(
            "assistant_occurrence must be 'unique' or 'final_suffix', got "
            f"{assistant_occurrence!r}"
        )
    marker_ids = encode_without_special_tokens(tokenizer, marker)
    local_start, local_end = unique_subsequence(
        assistant_ids,
        marker_ids,
        name=f"{name} marker within assistant output",
    )
    raw_position = assistant_start + local_end - 1
    mapping = position_map or {index: index for index in range(len(token_ids))}
    output_ids = processed_ids or token_ids
    try:
        position = mapping[raw_position]
        mapped_assistant_start = mapping[assistant_start]
        mapped_assistant_end = mapping[assistant_end - 1] + 1
        mapped_marker_start = mapping[assistant_start + local_start]
        mapped_marker_end = mapping[assistant_start + local_end - 1] + 1
    except KeyError as exc:
        raise ValueError(f"{name} marker maps through an unavailable token: {exc}") from exc
    record = {
        "position": position,
        "token_id": int(output_ids[position]),
        "token_text": tokenizer.decode(
            [output_ids[position]],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "assistant_span": [mapped_assistant_start, mapped_assistant_end],
        "marker_span": [mapped_marker_start, mapped_marker_end],
        "marker_token_ids": marker_ids,
        "validation_context": _token_context(tokenizer, output_ids, position),
    }
    if ":" not in record["token_text"]:
        raise ValueError(f"{name} cognition token is not the marker colon: {record}")
    return record


def locate_field_value_span(
    tokenizer: Any,
    token_ids: list[int],
    field_prefix: str,
    value: str,
    *,
    separator: str = " ",
    name: str,
    position_map: dict[int, int] | None = None,
    processed_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Match a complete labelled field before deriving its value span."""

    field_text = f"{field_prefix}{separator}{value}"
    # Qwen BPE can fuse both surrounding newlines into the field boundary
    # tokens (for example ".\n\n"). Match a bounded complete field, then use
    # that bounded tokenization's offsets to derive only value-overlapping
    # tokens.
    matches: dict[
        tuple[int, int],
        tuple[int, int, list[int], int, int, int, int | None],
    ] = {}
    for leading in ("\n\n", "\n", ""):
        for trailing in ("\n\n", "\n", ""):
            bounded_text = leading + field_text + trailing
            encoded = tokenizer(
                bounded_text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            bounded_ids_value = encoded["input_ids"]
            offsets_value = encoded["offset_mapping"]
            if hasattr(bounded_ids_value, "tolist"):
                bounded_ids_value = bounded_ids_value.tolist()
            if hasattr(offsets_value, "tolist"):
                offsets_value = offsets_value.tolist()
            if bounded_ids_value and isinstance(bounded_ids_value[0], list):
                bounded_ids_value = bounded_ids_value[0]
            if offsets_value and isinstance(offsets_value[0][0], list):
                offsets_value = offsets_value[0]
            bounded_ids = [int(token_id) for token_id in bounded_ids_value]
            offsets = [tuple(map(int, pair)) for pair in offsets_value]
            value_char_start = len(leading) + len(field_prefix) + len(separator)
            value_char_end = value_char_start + len(value)
            field_char_start = len(leading)
            field_char_end = field_char_start + len(field_text)
            local_value_positions = [
                index
                for index, (start, end) in enumerate(offsets)
                if end > start and start < value_char_end and end > value_char_start
            ]
            if not local_value_positions:
                continue
            local_field_positions = [
                index
                for index, (start, end) in enumerate(offsets)
                if end > start and start < field_char_end and end > field_char_start
            ]
            following_positions = [
                index
                for index, (start, end) in enumerate(offsets)
                if trailing
                and end > start
                and start < field_char_end + 1
                and end > field_char_end
            ]
            for start in find_subsequence_positions(token_ids, bounded_ids):
                value_start = start + local_value_positions[0]
                value_end = start + local_value_positions[-1] + 1
                key = (value_start, value_end)
                previous = matches.get(key)
                if previous is None or len(bounded_ids) > previous[3]:
                    matches[key] = (
                        start,
                        start + len(bounded_ids),
                        bounded_ids,
                        len(bounded_ids),
                        start + local_field_positions[0],
                        start + local_field_positions[-1] + 1,
                        start + following_positions[0] if following_positions else None,
                    )
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {name} complete-field token subsequence, "
            f"found {len(matches)}; value_spans={sorted(matches)}, "
            f"sequence_length={len(token_ids)}"
        )
    (raw_value_start, raw_value_end), (
        field_start,
        field_end,
        field_ids,
        _bounded_length,
        raw_field_start,
        raw_field_end,
        raw_following_position,
    ) = next(iter(matches.items()))
    mapping = position_map or {index: index for index in range(len(token_ids))}
    output_ids = processed_ids or token_ids
    try:
        value_start = mapping[raw_value_start]
        value_end = mapping[raw_value_end - 1] + 1
        mapped_field_start = mapping[raw_field_start]
        mapped_field_end = mapping[raw_field_end - 1] + 1
        mapped_following_position = (
            mapping[raw_following_position]
            if raw_following_position is not None
            else None
        )
    except KeyError as exc:
        raise ValueError(f"{name} field maps through an unavailable token: {exc}") from exc
    return {
        "span": [value_start, value_end],
        "field_span": [mapped_field_start, mapped_field_end],
        "following_token_position": mapped_following_position,
        "token_ids": output_ids[value_start:value_end],
        "field_token_ids": field_ids,
        "token_text": tokenizer.decode(
            output_ids[value_start:value_end],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "validation_context": _token_context(tokenizer, output_ids, value_start),
    }


def locate_panl_subsequence(
    tokenizer: Any,
    processed_ids: list[int],
    left_field: str,
    right_boundary: str,
    *,
    name: str = "panl",
) -> dict[str, Any]:
    """Locate the newline token between an exact field and its right boundary."""

    bounded_ids = encode_without_special_tokens(
        tokenizer,
        f"{left_field}\n{right_boundary}",
    )
    bounded_start, _ = unique_subsequence(
        processed_ids,
        bounded_ids,
        name=f"{name} bounded field",
    )
    right_ids = encode_without_special_tokens(tokenizer, right_boundary)
    right_start, _ = unique_subsequence(
        bounded_ids,
        right_ids,
        name=f"{name} right boundary",
    )
    if right_start < 1:
        raise ValueError(f"{name} has no token before the right boundary")
    position = bounded_start + right_start - 1
    record = token_position_record(
        tokenizer,
        RenderedTokenAlignment("", [], processed_ids, [], {}),
        position,
    )
    if "\n" not in record["token_text"]:
        raise ValueError(f"{name} token is not a newline token: {record}")
    return record


def locate_token_after_field(
    tokenizer: Any,
    token_ids: list[int],
    field_prefix: str,
    value: str,
    *,
    separator: str = " ",
    name: str = "panl",
    position_map: dict[int, int] | None = None,
    processed_ids: list[int] | None = None,
) -> dict[str, Any]:
    field = locate_field_value_span(
        tokenizer,
        token_ids,
        field_prefix,
        value,
        separator=separator,
        name=f"{name} preceding field",
        position_map=position_map,
        processed_ids=processed_ids,
    )
    following = field.get("following_token_position")
    if following is None:
        raise ValueError(f"{name} field has no bounded following token")
    position = int(following)
    output_ids = processed_ids or token_ids
    if position >= len(output_ids):
        raise ValueError(f"{name} has no token after the answer field")
    record = {
        "position": position,
        "token_id": int(output_ids[position]),
        "token_text": tokenizer.decode(
            [output_ids[position]],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "preceding_field_span": field["field_span"],
        "validation_context": _token_context(tokenizer, output_ids, position),
    }
    if "\n" not in record["token_text"]:
        raise ValueError(f"{name} token after answer is not a newline: {record}")
    return record


def locate_answer_panl_position(
    tokenizer: Any,
    token_ids: list[int],
    field_prefix: str,
    value: str,
    *,
    separator: str = " ",
    name: str = "panl",
    position_map: dict[int, int] | None = None,
    processed_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Locate the answer-following newline, including answer/newline fusion.

    Some tokenizers encode the final answer characters and the following newline
    as one token.  In that case the fused token is the only truthful PANL
    representation; LAT is shifted before it by ``locate_last_answer_token``.
    """

    field = locate_field_value_span(
        tokenizer,
        token_ids,
        field_prefix,
        value,
        separator=separator,
        name=f"{name} preceding field",
        position_map=position_map,
        processed_ids=processed_ids,
    )
    output_ids = processed_ids or token_ids
    mapping = position_map or {index: index for index in range(len(token_ids))}
    candidates: list[int] = []
    following = field.get("following_token_position")
    if following is not None:
        candidates.append(int(following))
    # The final value token may itself contain the newline (BPE fusion).
    if field.get("span"):
        candidates.append(int(field["span"][-1]) - 1)
    for position in candidates:
        if position < 0 or position >= len(output_ids):
            continue
        token_text = tokenizer.decode(
            [output_ids[position]],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if "\n" not in token_text:
            continue
        record = {
            "position": position,
            "token_id": int(output_ids[position]),
            "token_text": token_text,
            "preceding_field_span": list(field["field_span"]),
            "validation_context": _token_context(tokenizer, output_ids, position),
            "validation_status": "fused" if position == int(field["span"][-1]) - 1 and following != position else "passed",
        }
        if record["validation_status"] == "fused":
            record["position_adjustment"] = {
                "type": "PANLFusedWithAnswer",
                "reason": "answer-final token contains the following newline",
            }
        return record
    raise ValueError(f"{name} token after answer is not a newline, including fused-token fallback: {field}")


def locate_last_answer_token(
    tokenizer: Any,
    token_ids: list[int],
    field_prefix: str,
    value: str,
    *,
    panl_position: int,
    separator: str = " ",
    name: str = "lat",
    position_map: dict[int, int] | None = None,
    processed_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Locate the last answer token, shifting before PANL on token fusion."""

    field = locate_field_value_span(
        tokenizer,
        token_ids,
        field_prefix,
        value,
        separator=separator,
        name=f"{name} answer field",
        position_map=position_map,
        processed_ids=processed_ids,
    )
    raw_position = int(field["span"][1]) - 1
    position = raw_position
    adjustment = None
    if position == int(panl_position):
        position = int(panl_position) - 1
        adjustment = {
            "type": "LATShiftedBeforePANL",
            "reason": "last answer token is fused with the PANL token",
            "original_position": raw_position,
            "adjusted_position": position,
            "panl_position": int(panl_position),
        }
    output_ids = processed_ids or token_ids
    if position < 0 or position >= len(output_ids) or position >= int(panl_position):
        raise ValueError(
            f"{name} position {position} is invalid relative to PANL "
            f"{panl_position}"
        )
    record = {
        "position": position,
        "token_id": int(output_ids[position]),
        "token_text": tokenizer.decode(
            [output_ids[position]],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "answer_span": list(field["span"]),
        "preceding_field_span": list(field["field_span"]),
        "panl_position": int(panl_position),
        "validation_context": _token_context(tokenizer, output_ids, position),
        "validation_status": "adjusted" if adjustment is not None else "passed",
    }
    if adjustment is not None:
        record["position_adjustment"] = adjustment
    return record


def locate_image_pad_span(
    tokenizer: Any,
    processed_ids: list[int],
) -> dict[str, Any]:
    """Return only the contiguous image-pad run as a half-open span."""

    image_id = int(tokenizer.convert_tokens_to_ids("<|image_pad|>"))
    positions = [index for index, token_id in enumerate(processed_ids) if token_id == image_id]
    if not positions:
        raise ValueError("No <|image_pad|> tokens found")
    expected = list(range(positions[0], positions[-1] + 1))
    if positions != expected:
        raise ValueError(f"Image-pad tokens are not contiguous: {positions}")
    return {
        "span": [positions[0], positions[-1] + 1],
        "image_token_id": image_id,
        "image_token_count": len(positions),
        "token_text": "<|image_pad|>",
        "validation_context": _token_context(tokenizer, processed_ids, positions[0], radius=2),
    }


def locate_post_image_token(
    tokenizer: Any,
    processor: Any,
    alignment: RenderedTokenAlignment,
    image_grid_thw: Any,
    *,
    name: str = "pit",
) -> dict[str, Any]:
    """Return the first processed token after the complete vision span."""

    image = locate_image_span(tokenizer, processor, alignment, image_grid_thw)
    position = int(image["end_position"]) + 1
    if position >= len(alignment.processed_ids):
        raise ValueError(f"{name} has no token after the image span")
    record = token_position_record(tokenizer, alignment, position)
    record.update(
        {
            "image_span": [
                int(image["start_position"]),
                int(image["end_position"]),
            ],
            "image_token_span": [
                int(image["image_token_start"]),
                int(image["image_token_end"]),
            ],
            "image_token_count": int(image["image_token_count"]),
            "definition": "complete_image_span_end_plus_one",
            "validation_status": "passed",
        }
    )
    return record


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


def locate_text_clue_save_positions(
    tokenizer: Any,
    alignment: RenderedTokenAlignment,
    text_clue: str,
) -> dict[str, dict[str, Any]]:
    """Locate LTT and the following double-newline without token-text search."""

    bounded_prefix = f"Text clue:\n{text_clue}\n\n"
    bounded_start, prefix_end = unique_text_span(
        alignment.rendered,
        bounded_prefix,
    )
    next_line_end = alignment.rendered.find("\n", prefix_end)
    if next_line_end < 0:
        next_line_end = len(alignment.rendered)
    right_boundary = alignment.rendered[prefix_end:next_line_end]
    if not right_boundary.strip():
        raise ValueError("Text clue has no following non-empty section boundary")
    unique_text_span(alignment.rendered, bounded_prefix + right_boundary)
    stripped = text_clue.rstrip()
    if not stripped:
        raise ValueError("Text clue has no non-whitespace character for LTT")

    clue_start = bounded_start + len("Text clue:\n")
    last_char = clue_start + len(stripped) - 1
    ltt_positions = alignment.processed_tokens_for_char_span(last_char, last_char + 1)
    if len(set(ltt_positions)) != 1:
        raise ValueError("LTT character maps to multiple processed tokens")
    ltt = token_position_record(tokenizer, alignment, ltt_positions[0])

    separator_start = clue_start + len(text_clue)
    separator_positions = alignment.processed_tokens_for_char_span(
        separator_start,
        separator_start + 2,
    )
    ptnl = token_position_record(tokenizer, alignment, separator_positions[-1])
    ptnl["span_start_position"] = separator_positions[0]
    ptnl["span_end_position"] = separator_positions[-1]
    decoded_separator = tokenizer.decode(
        alignment.processed_ids[
            separator_positions[0] : separator_positions[-1] + 1
        ],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    ptnl["decoded_span"] = decoded_separator
    if "\n" not in ptnl["token_text"] or "\n" not in decoded_separator:
        raise ValueError(f"PTNL token does not contain a newline: {ptnl}")
    if ltt["position"] == ptnl["position"]:
        fused = dict(ltt)
        adjusted_position = int(ptnl["position"]) - 1
        if adjusted_position < 0:
            raise ValueError("Cannot shift LTT before PTNL at position zero")
        ltt = token_position_record(tokenizer, alignment, adjusted_position)
        if "\n" in ltt["token_text"]:
            raise ValueError(
                "Adjusted LTT token before PTNL still contains a newline: "
                f"{ltt}"
            )
        ltt["position_adjustment"] = {
            "type": "LTTShiftedBeforePTNL",
            "reason": "semantic-final character is fused with the PTNL separator",
            "original_position": int(fused["position"]),
            "original_token_id": int(fused["token_id"]),
            "original_token_text": str(fused["token_text"]),
            "adjusted_position": adjusted_position,
            "ptnl_position": int(ptnl["position"]),
        }
        ltt["validation_status"] = "adjusted"
    else:
        ltt["validation_status"] = "passed"
    ptnl["validation_status"] = "passed"
    return {"ltt": ltt, "ptnl": ptnl}


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
