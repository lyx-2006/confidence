from __future__ import annotations

from typing import Any

from layer_metacognition.token_spans import build_rendered_alignment

from dp_SA.positions import locate_phase1_positions

from .config import ANCHORS, POSITION_ORDER, TOKEN_WINDOW_RADIUS


def _occurrences(text: str, needle: str) -> list[int]:
    output: list[int] = []
    cursor = 0
    while True:
        index = text.find(needle, cursor)
        if index < 0:
            return output
        output.append(index)
        cursor = index + 1


def _token_text(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _window(tokenizer: Any, processed_ids: list[int], position: int, radius: int) -> list[dict[str, Any]]:
    start = max(0, position - radius)
    end = min(len(processed_ids), position + radius + 1)
    return [
        {
            "processed_index": index,
            "token_id": int(processed_ids[index]),
            "token_text": _token_text(tokenizer, processed_ids[index]),
            "is_target": index == position,
        }
        for index in range(start, end)
    ]


def _enrich(
    tokenizer: Any,
    alignment: Any,
    record: dict[str, Any],
    *,
    rendered_index: int,
    anchor_text: str,
    anchor_count: int,
    anchor_start: int,
) -> dict[str, Any]:
    position = int(record["processed_index"])
    return {
        **record,
        "rendered_index": int(rendered_index),
        "anchor_text": anchor_text,
        "anchor_occurrence_count": int(anchor_count),
        "anchor_start_index": int(anchor_start),
        "token_window": _window(tokenizer, alignment.processed_ids, position, TOKEN_WINDOW_RADIUS),
    }


def locate_checkpoint_positions(tokenizer: Any, rendered: str, inputs: Any, answer: str) -> dict[str, Any]:
    ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
    mask = inputs.get("attention_mask") if isinstance(inputs, dict) else getattr(inputs, "attention_mask", None)
    alignment = build_rendered_alignment(tokenizer, rendered, ids, mask)
    base = locate_phase1_positions(tokenizer, rendered, inputs, answer)

    answer_anchor = f"**Answer**: {answer}"
    answer_starts = _occurrences(rendered, answer_anchor)
    instruction_index = int(base["SA_INSTRUCTION_START"]["rendered_index"])
    bounded = [value for value in answer_starts if value < instruction_index]
    if not bounded:
        raise ValueError("Fixed answer anchor is absent before the SA instruction")
    selected_answer_start = bounded[-1]

    output: dict[str, Any] = {}
    for name in ("P1_LAT", "P1_PANL"):
        output[name] = _enrich(
            tokenizer,
            alignment,
            base[name],
            rendered_index=int(base[name]["rendered_index"]),
            anchor_text=answer_anchor,
            anchor_count=len(answer_starts),
            anchor_start=selected_answer_start,
        )

    for name, anchor in ANCHORS.items():
        if name == "P1_CLASS_LIST_END":
            common = base[name]
            output[name] = _enrich(
                tokenizer,
                alignment,
                common,
                rendered_index=int(common["rendered_index"]),
                anchor_text=str(common["anchor_text"]),
                anchor_count=int(common["anchor_occurrence_count"]),
                anchor_start=int(common["anchor_start_index"]),
            )
            continue
        starts = _occurrences(rendered, anchor)
        if len(starts) != 1:
            raise ValueError(f"{name}: expected one anchor occurrence, found {len(starts)}")
        anchor_start = starts[0]
        newline_char = anchor_start + len(anchor)
        if newline_char >= len(rendered) or rendered[newline_char] != "\n":
            raise ValueError(f"{name}: anchor is not immediately followed by a newline")
        mapped = alignment.processed_tokens_for_char_span(newline_char, newline_char + 1)
        if not mapped:
            raise ValueError(f"{name}: newline maps to no processed token")
        position = int(mapped[0])
        token_id = int(alignment.processed_ids[position])
        token_text = _token_text(tokenizer, token_id)
        if "\n" not in token_text:
            raise ValueError(f"{name}: mapped token is not a newline token: {token_text!r}")
        output[name] = _enrich(
            tokenizer,
            alignment,
            {"processed_index": position, "rendered_index": newline_char, "token_id": token_id, "token_text": token_text},
            rendered_index=newline_char,
            anchor_text=anchor,
            anchor_count=1,
            anchor_start=anchor_start,
        )

    sac = base["P1_SAC"]
    output["P1_SAC"] = _enrich(
        tokenizer,
        alignment,
        sac,
        rendered_index=int(sac["rendered_index"]),
        anchor_text="**Source Attribution**:",
        anchor_count=len(_occurrences(rendered, "**Source Attribution**:")),
        anchor_start=rendered.rfind("**Source Attribution**:"),
    )
    indices = [int(output[name]["processed_index"]) for name in POSITION_ORDER]
    if any(left >= right for left, right in zip(indices, indices[1:])):
        raise ValueError(f"Checkpoint causal order failed: {dict(zip(POSITION_ORDER, indices))}")
    output["phase1_answer_span"] = base["phase1_answer_span"]
    output["phase1_answer_token_ids"] = base["phase1_answer_token_ids"]
    output["causal_order"] = {name: index for name, index in zip(POSITION_ORDER, indices)}
    output["causal_order_valid"] = True
    return output
