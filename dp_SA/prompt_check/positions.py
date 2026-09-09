from __future__ import annotations

from typing import Any

from layer_metacognition.token_spans import build_rendered_alignment, unique_text_span

from .templates import TemplateSpec


def _occurrences(text: str, needle: str) -> list[int]:
    result, cursor = [], 0
    while True:
        index = text.find(needle, cursor)
        if index < 0: return result
        result.append(index); cursor = index + 1


def _record(tokenizer: Any, alignment: Any, token_index: int, rendered_index: int) -> dict[str, Any]:
    token_id = int(alignment.processed_ids[token_index])
    return {
        "processed_index": int(token_index), "rendered_index": int(rendered_index), "token_id": token_id,
        "token_text": tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False),
    }


def locate_template_positions(tokenizer: Any, rendered: str, inputs: Any, answer: str, spec: TemplateSpec) -> dict[str, Any]:
    ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
    mask = inputs.get("attention_mask") if isinstance(inputs, dict) else getattr(inputs, "attention_mask", None)
    alignment = build_rendered_alignment(tokenizer, rendered, ids, mask)
    instruction_char = unique_text_span(rendered, spec.post_answer_anchor)[0]
    answer_field = f"**Answer**: {answer}"
    field_start = rendered.rfind(answer_field, 0, instruction_char)
    if field_start < 0: raise ValueError(f"Fixed answer field absent for {spec.name}")
    field_end = field_start + len(answer_field)
    if rendered[field_end:instruction_char] != "\n\n": raise ValueError("Fixed answer is not followed by PANL and instruction")
    answer_start = field_start + len("**Answer**: "); answer_end = answer_start + len(answer)
    if rendered[answer_end] != "\n": raise ValueError("Fixed answer is not followed by newline")
    answer_tokens = alignment.processed_tokens_for_char_span(answer_start, answer_end)
    panl_tokens = alignment.processed_tokens_for_char_span(answer_end, answer_end + 1)
    if not answer_tokens or not panl_tokens: raise ValueError("Answer/PANL maps to no processed tokens")
    panl = int(panl_tokens[0]); lat_candidates = [int(p) for p in answer_tokens if int(p) < panl]
    if not lat_candidates: raise ValueError("No true answer token before PANL")
    lat = lat_candidates[-1]
    anchors = _occurrences(rendered, spec.last_class_description)
    if len(anchors) != 1: raise ValueError(f"{spec.name} CLE anchor count is {len(anchors)}, expected 1")
    cle_char = anchors[0] + len(spec.last_class_description)
    if cle_char >= len(rendered) or rendered[cle_char] != "\n": raise ValueError("CLE anchor is not immediately followed by newline")
    cle_tokens = alignment.processed_tokens_for_char_span(cle_char, cle_char + 1)
    if not cle_tokens: raise ValueError("CLE newline maps to no processed token")
    cle = int(cle_tokens[0])
    if not rendered.endswith("**Source Attribution**:"): raise ValueError("SAC prefill is not the exact suffix")
    sac_char = len(rendered) - 1
    sac_tokens = alignment.processed_tokens_for_char_span(sac_char, sac_char + 1)
    if not sac_tokens: raise ValueError("SAC colon maps to no processed token")
    sac = int(sac_tokens[-1])
    instruction = int(alignment.processed_tokens_for_char_span(instruction_char, instruction_char + 1)[0])
    if not (lat < panl < instruction < cle < sac):
        raise ValueError(f"Causal order failed: LAT={lat}, PANL={panl}, instruction={instruction}, CLE={cle}, SAC={sac}")
    panl_text = tokenizer.decode([int(alignment.processed_ids[panl])], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    cle_text = tokenizer.decode([int(alignment.processed_ids[cle])], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if "\n" not in panl_text or "\n" not in cle_text: raise ValueError("PANL/CLE token must contain newline")
    output = {
        "P1_LAT": _record(tokenizer, alignment, lat, answer_end - 1),
        "P1_PANL": _record(tokenizer, alignment, panl, answer_end),
        "P1_CLASS_LIST_END": {**_record(tokenizer, alignment, cle, cle_char), "anchor_text": spec.last_class_description},
        "P1_SAC": _record(tokenizer, alignment, sac, sac_char),
        "SA_INSTRUCTION_START": _record(tokenizer, alignment, instruction, instruction_char),
        "phase1_answer_span": [int(answer_tokens[0]), int(answer_tokens[-1]) + 1],
        "phase1_answer_token_ids": [int(alignment.processed_ids[p]) for p in answer_tokens],
        "causal_order_valid": True,
    }
    if lat != output["phase1_answer_span"][1] - 1: raise ValueError("LAT is not final true answer token")
    return output
