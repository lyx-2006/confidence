from __future__ import annotations

from typing import Any

from layer_metacognition.token_spans import build_rendered_alignment, unique_text_span
from .prompts import FORBIDDEN_BEFORE_PANL, SA_INSTRUCTION_START, SA_PREFILL

P1_CLASS_LIST_END_ANCHOR = "8: The answer was based almost entirely on the image."
P1_CLASS_LIST_END_DEFINITION = {
    "anchor_text": P1_CLASS_LIST_END_ANCHOR,
    "anchor_occurrence": "exactly_once_in_full_rendered_prompt",
    "target": "first_newline_character_immediately_after_anchor",
    "alignment": "rendered_to_processed",
    "decoded_token_must_contain": "\n",
}


def _occurrences(text: str, needle: str) -> list[int]:
    output: list[int] = []
    cursor = 0
    while True:
        index = text.find(needle, cursor)
        if index < 0:
            return output
        output.append(index)
        cursor = index + 1

def _record(tokenizer: Any, alignment: Any, position: int, rendered_position: int) -> dict[str, Any]:
    token_id = int(alignment.processed_ids[position])
    return {"processed_index": int(position), "rendered_index": int(rendered_position), "token_id": token_id,
            "token_text": tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)}

def locate_phase1_positions(tokenizer: Any, rendered: str, inputs: Any, answer: str) -> dict[str, Any]:
    ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
    mask = inputs.get("attention_mask") if isinstance(inputs, dict) else getattr(inputs, "attention_mask", None)
    alignment = build_rendered_alignment(tokenizer, rendered, ids, mask)
    answer_field = f"**Answer**: {answer}"
    instruction_char = unique_text_span(rendered, SA_INSTRUCTION_START)[0]
    # Dataset clues may themselves contain literal ``**Answer**: value`` text.
    # The delayed-SA fixed field is the final bounded occurrence before the
    # first SA instruction, not an unscoped globally-unique string.
    field_start = rendered.rfind(answer_field, 0, instruction_char)
    if field_start < 0:
        raise ValueError(f"Fixed answer field is absent before SA instruction: {answer_field!r}")
    field_end = field_start + len(answer_field)
    if rendered[field_end:instruction_char] != "\n\n":
        raise ValueError("Fixed answer field is not immediately followed by PANL and SA instruction")
    colon_char = field_start + len("**Answer**")
    answer_start = field_start + len("**Answer**: ")
    answer_end = answer_start + len(answer)
    if rendered[answer_end] != "\n": raise ValueError("Fixed answer is not followed by PANL")
    ac_tokens = alignment.processed_tokens_for_char_span(colon_char, colon_char + 1)
    answer_tokens = alignment.processed_tokens_for_char_span(answer_start, answer_end)
    panl_tokens = alignment.processed_tokens_for_char_span(answer_end, answer_end + 1)
    ac, panl = ac_tokens[-1], panl_tokens[0]
    lat_candidates = [p for p in answer_tokens if p < panl]
    if not lat_candidates: raise ValueError("No true answer token before PANL")
    lat = lat_candidates[-1]
    plus1 = panl + 1
    if plus1 >= len(alignment.processed_ids): raise ValueError("PANL+1 is outside prompt")
    if not rendered.endswith(SA_PREFILL): raise ValueError("P1_SAC is not the final prompt suffix")
    sac_colon_char = len(rendered) - 1
    sac = alignment.processed_tokens_for_char_span(sac_colon_char, sac_colon_char + 1)[-1]
    instruction = alignment.processed_tokens_for_char_span(instruction_char, instruction_char + len("State"))[0]
    class_list_starts = _occurrences(rendered, P1_CLASS_LIST_END_ANCHOR)
    if len(class_list_starts) != 1:
        raise ValueError(
            "P1_CLASS_LIST_END: expected one anchor occurrence, "
            f"found {len(class_list_starts)}"
        )
    class_list_anchor_start = class_list_starts[0]
    class_list_newline_char = class_list_anchor_start + len(P1_CLASS_LIST_END_ANCHOR)
    if class_list_newline_char >= len(rendered) or rendered[class_list_newline_char] != "\n":
        raise ValueError("P1_CLASS_LIST_END: anchor is not immediately followed by a newline")
    class_list_tokens = alignment.processed_tokens_for_char_span(
        class_list_newline_char, class_list_newline_char + 1
    )
    if not class_list_tokens:
        raise ValueError("P1_CLASS_LIST_END: newline maps to no processed token")
    class_list_end = int(class_list_tokens[0])
    class_list_token_id = int(alignment.processed_ids[class_list_end])
    class_list_token_text = tokenizer.decode(
        [class_list_token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    if "\n" not in class_list_token_text:
        raise ValueError(
            "P1_CLASS_LIST_END: mapped token is not a newline token: "
            f"{class_list_token_text!r}"
        )
    if not (lat < panl < instruction < class_list_end < sac):
        raise ValueError(
            "Causal order failed: "
            f"LAT={lat}, PANL={panl}, instruction={instruction}, "
            f"CLASS_LIST_END={class_list_end}, SAC={sac}"
        )
    before = rendered[:answer_end + 1].casefold()
    found = [term for term in FORBIDDEN_BEFORE_PANL if term in before]
    if found: raise ValueError(f"SA text before PANL: {found}")
    if "\n" not in tokenizer.decode([alignment.processed_ids[panl]], skip_special_tokens=False, clean_up_tokenization_spaces=False):
        raise ValueError("P1_PANL token does not contain newline")
    output = {
        "P1_AC": _record(tokenizer, alignment, ac, colon_char),
        "P1_LAT": _record(tokenizer, alignment, lat, answer_end - 1),
        "P1_PANL": _record(tokenizer, alignment, panl, answer_end),
        "P1_PANL_PLUS_1": _record(tokenizer, alignment, plus1, answer_end + 1),
        "P1_CLASS_LIST_END": {
            "processed_index": class_list_end,
            "rendered_index": class_list_newline_char,
            "token_id": class_list_token_id,
            "token_text": class_list_token_text,
            "anchor_text": P1_CLASS_LIST_END_ANCHOR,
            "anchor_occurrence_count": len(class_list_starts),
            "anchor_start_index": class_list_anchor_start,
        },
        "P1_SAC": _record(tokenizer, alignment, sac, sac_colon_char),
        "SA_INSTRUCTION_START": _record(tokenizer, alignment, instruction, instruction_char),
    }
    output["phase1_answer_span"] = [answer_tokens[0], answer_tokens[-1] + 1]
    output["phase1_answer_token_ids"] = [int(alignment.processed_ids[p]) for p in answer_tokens]
    output["causal_order_valid"] = True
    return output
