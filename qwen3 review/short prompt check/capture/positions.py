from __future__ import annotations

from typing import Any

from layer_metacognition.token_spans import build_rendered_alignment, unique_text_span

from .short_prompt import CLASS_LIST_ANCHOR, SA_PREFILL, SOURCE_PARAGRAPH_START


def _token_span(alignment: Any, processed_index: int) -> list[int]:
    rendered_indices = [
        source for source, target in alignment.rendered_to_processed.items()
        if int(target) == int(processed_index)
    ]
    if len(rendered_indices) != 1:
        raise ValueError(
            f"Processed token {processed_index} has {len(rendered_indices)} rendered mappings"
        )
    start, end = alignment.offsets[rendered_indices[0]]
    if end <= start:
        raise ValueError(f"Processed token {processed_index} has an empty rendered span")
    return [int(start), int(end)]


def _record(
    tokenizer: Any,
    alignment: Any,
    processed_index: int,
    target_start: int,
    target_end: int,
) -> dict[str, Any]:
    token_id = int(alignment.processed_ids[processed_index])
    return {
        "processed_index": int(processed_index),
        "rendered_target_span": [int(target_start), int(target_end)],
        "rendered_token_span": _token_span(alignment, processed_index),
        "token_id": token_id,
        "token_text": tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
    }


def locate_short_phase1_positions(
    tokenizer: Any,
    rendered: str,
    inputs: Any,
    answer: str,
) -> dict[str, Any]:
    ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
    mask = inputs.get("attention_mask") if isinstance(inputs, dict) else getattr(inputs, "attention_mask", None)
    alignment = build_rendered_alignment(tokenizer, rendered, ids, mask)

    source_start = unique_text_span(rendered, SOURCE_PARAGRAPH_START)[0]
    answer_field = f"**Answer**: {answer}"
    field_start = rendered.rfind(answer_field, 0, source_start)
    if field_start < 0:
        raise ValueError(f"Fixed answer field is absent before source paragraph: {answer_field!r}")
    field_end = field_start + len(answer_field)
    if rendered[field_end:source_start] != "\n\n":
        raise ValueError("Fixed answer field is not immediately followed by PANL and source paragraph")
    answer_start = field_start + len("**Answer**: ")
    answer_end = answer_start + len(answer)
    if rendered[answer_end] != "\n":
        raise ValueError("Fixed answer is not immediately followed by PANL")

    answer_tokens = alignment.processed_tokens_for_char_span(answer_start, answer_end)
    panl_tokens = alignment.processed_tokens_for_char_span(answer_end, answer_end + 1)
    panl = int(panl_tokens[0])
    lat_candidates = [int(index) for index in answer_tokens if int(index) < panl]
    if not lat_candidates:
        raise ValueError("No true fixed-answer token occurs before PANL")
    lat = lat_candidates[-1]
    panl_plus_1 = panl + 1
    if panl_plus_1 >= len(alignment.processed_ids):
        raise ValueError("PANL+1 lies outside the processed prompt")

    anchor_start, anchor_end = unique_text_span(rendered, CLASS_LIST_ANCHOR)
    if anchor_end >= len(rendered) or rendered[anchor_end] != "\n":
        raise ValueError("Complete class-list anchor is not immediately followed by a newline")
    cle_tokens = alignment.processed_tokens_for_char_span(anchor_end, anchor_end + 1)
    if not cle_tokens:
        raise ValueError("CLE newline maps to no processed token")
    cle = int(cle_tokens[0])

    if not rendered.endswith(SA_PREFILL):
        raise ValueError("Rendered prompt does not end with the exact SA assistant prefill")
    sac_char = len(rendered) - 1
    sac = int(alignment.processed_tokens_for_char_span(sac_char, sac_char + 1)[-1])

    records = {
        "P1_LAT": _record(tokenizer, alignment, lat, answer_end - 1, answer_end),
        "P1_PANL": _record(tokenizer, alignment, panl, answer_end, answer_end + 1),
        "P1_PANL_PLUS_1": _record(
            tokenizer,
            alignment,
            panl_plus_1,
            _token_span(alignment, panl_plus_1)[0],
            _token_span(alignment, panl_plus_1)[1],
        ),
        "P1_CLASS_LIST_END": _record(tokenizer, alignment, cle, anchor_end, anchor_end + 1),
        "P1_SAC": _record(tokenizer, alignment, sac, sac_char, sac_char + 1),
    }
    records["P1_CLASS_LIST_END"].update({
        "anchor_text": CLASS_LIST_ANCHOR,
        "anchor_span": [anchor_start, anchor_end],
        "anchor_occurrence_count": 1,
        "newline_merge_policy": "processed_token_containing_target_newline",
        "newline_merged": records["P1_CLASS_LIST_END"]["rendered_token_span"] != [anchor_end, anchor_end + 1],
    })
    if "\n" not in records["P1_PANL"]["token_text"]:
        raise ValueError("PANL processed token does not contain a newline")
    if "\n" not in records["P1_CLASS_LIST_END"]["token_text"]:
        raise ValueError("CLE processed token does not contain its target newline")
    order = [lat, panl, cle, sac]
    if not all(left < right for left, right in zip(order, order[1:])):
        raise ValueError(f"Required causal order LAT < PANL < CLE < SAC failed: {order}")
    records["phase1_answer_span"] = [int(answer_tokens[0]), int(answer_tokens[-1]) + 1]
    records["phase1_answer_token_ids"] = [
        int(alignment.processed_ids[index]) for index in answer_tokens
    ]
    records["causal_order_valid"] = True
    records["processed_sequence_length"] = len(alignment.processed_ids)
    return records


def external_positions(located: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapping = {
        "LAT": "P1_LAT",
        "PANL": "P1_PANL",
        "CLE": "P1_CLASS_LIST_END",
        "PANL+1": "P1_PANL_PLUS_1",
        "SAC": "P1_SAC",
    }
    return {name: dict(located[key]) for name, key in mapping.items()}

