from __future__ import annotations

from typing import Any

from layer_metacognition.token_spans import build_rendered_alignment

from .config import POSITIONS, VARIANTS
from .prompts import ATTRIBUTION_INSTRUCTION_START, CLE_ANCHOR, SA_PREFILL


ASSISTANT_HEADER = "<|im_start|>assistant\n"
USER_HEADER = "<|im_start|>user\n"
MESSAGE_END = "<|im_end|>"


def _record(tokenizer: Any, alignment: Any, processed_index: int, char_index: int) -> dict[str, Any]:
    token_id = int(alignment.processed_ids[processed_index])
    left = max(0, processed_index - 2)
    right = min(len(alignment.processed_ids), processed_index + 3)
    neighbor_ids = [int(value) for value in alignment.processed_ids[left:right]]
    return {
        "processed_index": int(processed_index),
        "rendered_index": int(char_index),
        "token_id": token_id,
        "token_text": tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False),
        "neighbor_start": left,
        "neighbor_token_ids": neighbor_ids,
        "neighbor_token_text": tokenizer.decode(
            neighbor_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        ),
    }


def _single_token(alignment: Any, start: int, end: int, label: str) -> int:
    tokens = alignment.processed_tokens_for_char_span(start, end)
    if len(tokens) != 1:
        raise ValueError(f"{label} character span maps to {len(tokens)} tokens: {tokens}")
    return int(tokens[0])


def locate_positions(
    tokenizer: Any,
    rendered: str,
    inputs: Any,
    raw_output: str,
    variant: str,
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant!r}")
    ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
    mask = inputs.get("attention_mask") if isinstance(inputs, dict) else getattr(inputs, "attention_mask", None)
    alignment = build_rendered_alignment(tokenizer, rendered, ids, mask)

    assistant_starts: list[int] = []
    cursor = 0
    while True:
        index = rendered.find(ASSISTANT_HEADER, cursor)
        if index < 0:
            break
        assistant_starts.append(index)
        cursor = index + 1
    if len(assistant_starts) != 2:
        raise ValueError(f"Expected two assistant messages, found {len(assistant_starts)}")
    history_start = assistant_starts[0] + len(ASSISTANT_HEADER)
    expected_history = raw_output if variant == "native_boundary" else raw_output + "\n"
    if not rendered.startswith(expected_history, history_start):
        raise ValueError("Rendered historical assistant content differs from the saved Phase-0 output")
    history_end = history_start + len(expected_history)
    expected_boundary = MESSAGE_END + "\n" + USER_HEADER
    if rendered[history_end:history_end + len(expected_boundary)] != expected_boundary:
        raise ValueError("Historical assistant is not followed by the native assistant/user boundary")

    answer_body_start = history_start + len("**Answer**:")
    answer_body = raw_output[len("**Answer**:"):]
    stripped = answer_body.rstrip()
    if not stripped:
        raise ValueError("Phase-0 answer has no non-whitespace content")
    lat_char = answer_body_start + len(stripped) - 1
    lat = alignment.processed_tokens_for_char_span(lat_char, lat_char + 1)[-1]

    if variant == "native_boundary":
        panl_char = history_end + len(MESSAGE_END)
        panl_plus_1_char = panl_char + 1
        if rendered[panl_char] != "\n" or not rendered.startswith("<|im_start|>", panl_plus_1_char):
            raise ValueError("Native boundary PANL definition does not match rendered chat template")
    else:
        panl_char = history_end - 1
        panl_plus_1_char = history_end
        if rendered[panl_char] != "\n" or not rendered.startswith(MESSAGE_END, panl_plus_1_char):
            raise ValueError("Explicit-newline PANL definition does not match rendered chat template")

    instruction = rendered.find(ATTRIBUTION_INSTRUCTION_START, history_end)
    if instruction < 0:
        raise ValueError("Attribution instruction is absent after the historical answer")
    if rendered.find(ATTRIBUTION_INSTRUCTION_START, 0, history_end) >= 0:
        raise ValueError("Attribution instruction leaked before the historical answer boundary")
    cle_anchor = rendered.find(CLE_ANCHOR, instruction)
    if cle_anchor < 0 or rendered.find(CLE_ANCHOR, cle_anchor + 1) >= 0:
        raise ValueError("CLE label-definition anchor must occur exactly once after the answer")
    cle_char = cle_anchor + len(CLE_ANCHOR)
    if cle_char >= len(rendered) or rendered[cle_char] != "\n":
        raise ValueError("CLE anchor is not followed by a newline")
    if not rendered.endswith(SA_PREFILL):
        raise ValueError("Rendered prompt does not end at the source-attribution prefill")
    sac_char = len(rendered) - 1
    if rendered[sac_char] != ":":
        raise ValueError("SAC is not the final prefill colon")

    indices = {
        "LAT": int(lat),
        "PANL": _single_token(alignment, panl_char, panl_char + 1, "PANL"),
        "PANL+1": _single_token(
            alignment,
            panl_plus_1_char,
            panl_plus_1_char + (len(MESSAGE_END) if variant == "explicit_newline" else len("<|im_start|>")),
            "PANL+1",
        ),
        "CLE": _single_token(alignment, cle_char, cle_char + 1, "CLE"),
        "SAC": _single_token(alignment, sac_char, sac_char + 1, "SAC"),
    }
    chars = {
        "LAT": lat_char, "PANL": panl_char, "PANL+1": panl_plus_1_char,
        "CLE": cle_char, "SAC": sac_char,
    }
    if not (indices["LAT"] < indices["PANL"] < indices["PANL+1"] < indices["CLE"] < indices["SAC"]):
        raise ValueError(f"Position causal order failed: {indices}")
    records = {name: _record(tokenizer, alignment, indices[name], chars[name]) for name in POSITIONS}
    records["PANL"]["definition"] = (
        "newline_after_historical_assistant_im_end"
        if variant == "native_boundary" else "explicit_newline_appended_to_historical_assistant"
    )
    records["PANL+1"]["definition"] = (
        "next_user_im_start" if variant == "native_boundary" else "historical_assistant_im_end"
    )
    return {
        "variant": variant,
        "positions": records,
        "indices": indices,
        "history_rendered_span": [history_start, history_end],
        "answer_body_rendered_span": [answer_body_start, answer_body_start + len(answer_body)],
        "attribution_instruction_rendered_index": instruction,
        "causal_order_valid": True,
    }

