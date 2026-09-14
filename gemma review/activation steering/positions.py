from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dp_SA.prompts import FORBIDDEN_BEFORE_PANL, SA_INSTRUCTION_START, SA_PREFILL

P1_CLE_ANCHOR = "8: The answer was based almost entirely on the image."


@dataclass(frozen=True)
class ExpandedAlignment:
    rendered: str
    expanded: str
    input_ids: list[int]
    offsets: list[tuple[int, int]]
    image_starts: tuple[int, ...]
    expansion_delta: int

    def translate(self, index: int) -> int:
        prior = sum(start < index for start in self.image_starts)
        return index + prior * self.expansion_delta

    def tokens_for_span(self, start: int, end: int) -> list[int]:
        if not 0 <= start < end <= len(self.rendered):
            raise ValueError(f"Invalid rendered character span [{start}, {end})")
        expanded_start, expanded_end = self.translate(start), self.translate(end)
        tokens = [
            index
            for index, (token_start, token_end) in enumerate(self.offsets)
            if token_end > token_start and token_start < expanded_end and token_end > expanded_start
        ]
        if not tokens:
            raise ValueError(f"Character span [{start}, {end}) maps to no Gemma tokens")
        return tokens


def _occurrences(text: str, needle: str) -> list[int]:
    starts: list[int] = []
    cursor = 0
    while True:
        position = text.find(needle, cursor)
        if position < 0:
            return starts
        starts.append(position)
        cursor = position + 1


def build_expanded_alignment(processor: Any, rendered: str, inputs: Any) -> ExpandedAlignment:
    boi = str(processor.boi_token)
    full = str(processor.full_image_sequence)
    starts = tuple(_occurrences(rendered, boi))
    if len(starts) != 1:
        raise ValueError(f"Expected exactly one Gemma image placeholder, found {len(starts)}")
    expanded = rendered.replace(boi, full)
    encoded = processor.tokenizer(
        expanded, add_special_tokens=False, return_offsets_mapping=True
    )
    encoded_ids = [int(value) for value in encoded["input_ids"]]
    actual_ids = [int(value) for value in inputs.input_ids[0].tolist()]
    mask = inputs.get("attention_mask")
    if mask is not None:
        actual_ids = [
            token for token, keep in zip(actual_ids, mask[0].tolist()) if int(keep)
        ]
    if encoded_ids != actual_ids:
        mismatch = next(
            (index for index, pair in enumerate(zip(encoded_ids, actual_ids)) if pair[0] != pair[1]),
            min(len(encoded_ids), len(actual_ids)),
        )
        raise ValueError(
            "Gemma expanded prompt does not match processed input_ids: "
            f"mismatch={mismatch}, expanded={len(encoded_ids)}, actual={len(actual_ids)}"
        )
    return ExpandedAlignment(
        rendered=rendered,
        expanded=expanded,
        input_ids=actual_ids,
        offsets=[tuple(map(int, pair)) for pair in encoded["offset_mapping"]],
        image_starts=starts,
        expansion_delta=len(full) - len(boi),
    )


def _record(tokenizer: Any, alignment: ExpandedAlignment, position: int, rendered_index: int) -> dict[str, Any]:
    token_id = alignment.input_ids[position]
    return {
        "processed_index": int(position),
        "rendered_index": int(rendered_index),
        "token_id": int(token_id),
        "token_text": tokenizer.decode(
            [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
        ),
    }


def locate_phase1_positions(
    processor: Any, rendered: str, inputs: Any, answer: str
) -> dict[str, Any]:
    tokenizer = processor.tokenizer
    alignment = build_expanded_alignment(processor, rendered, inputs)
    instruction_starts = _occurrences(rendered, SA_INSTRUCTION_START)
    if len(instruction_starts) != 1:
        raise ValueError(f"Expected one SA instruction, found {len(instruction_starts)}")
    instruction_char = instruction_starts[0]
    answer_field = f"**Answer**: {answer}"
    field_start = rendered.rfind(answer_field, 0, instruction_char)
    if field_start < 0:
        raise ValueError("Fixed Phase 1 answer is absent before the SA instruction")
    field_end = field_start + len(answer_field)
    if rendered[field_end:instruction_char] != "\n\n":
        raise ValueError("Fixed answer is not immediately followed by PANL and SA instruction")
    colon_char = field_start + len("**Answer**")
    answer_start = field_start + len("**Answer**: ")
    answer_end = answer_start + len(answer)
    answer_tokens = alignment.tokens_for_span(answer_start, answer_end)
    ac = alignment.tokens_for_span(colon_char, colon_char + 1)[-1]
    panl = alignment.tokens_for_span(answer_end, answer_end + 1)[0]
    lat_candidates = [position for position in answer_tokens if position < panl]
    if not lat_candidates:
        raise ValueError("No answer token exists before PANL")
    lat = lat_candidates[-1]
    plus_one = panl + 1
    if plus_one >= len(alignment.input_ids):
        raise ValueError("PANL+1 is outside the prompt")
    if not rendered.endswith(SA_PREFILL):
        raise ValueError("P1_SAC is not the exact final prompt suffix")
    sac_char = len(rendered) - 1
    sac = alignment.tokens_for_span(sac_char, sac_char + 1)[-1]
    instruction = alignment.tokens_for_span(instruction_char, instruction_char + len("State"))[0]
    cle_starts = _occurrences(rendered, P1_CLE_ANCHOR)
    if len(cle_starts) != 1:
        raise ValueError(f"P1_CLE expected one anchor, found {len(cle_starts)}")
    cle_char = cle_starts[0] + len(P1_CLE_ANCHOR)
    if cle_char >= len(rendered) or rendered[cle_char] != "\n":
        raise ValueError("P1_CLE anchor is not immediately followed by a newline")
    cle = alignment.tokens_for_span(cle_char, cle_char + 1)[0]
    if not (lat < panl < instruction < cle < sac):
        raise ValueError(
            f"Causal order failed: LAT={lat}, PANL={panl}, instruction={instruction}, CLE={cle}, SAC={sac}"
        )
    before_panl = rendered[: answer_end + 1].casefold()
    forbidden = [term for term in FORBIDDEN_BEFORE_PANL if term in before_panl]
    if forbidden:
        raise ValueError(f"SA text occurs before PANL: {forbidden}")
    for name, position in (("P1_PANL", panl), ("P1_CLE", cle)):
        decoded = tokenizer.decode(
            [alignment.input_ids[position]],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if "\n" not in decoded:
            raise ValueError(f"{name} token is not a newline token: {decoded!r}")
    output = {
        "P1_AC": _record(tokenizer, alignment, ac, colon_char),
        "P1_LAT": _record(tokenizer, alignment, lat, answer_end - 1),
        "P1_PANL": _record(tokenizer, alignment, panl, answer_end),
        "P1_PANL_PLUS_1": _record(tokenizer, alignment, plus_one, answer_end + 1),
        "P1_CLE": _record(tokenizer, alignment, cle, cle_char),
        "P1_SAC": _record(tokenizer, alignment, sac, sac_char),
        "SA_INSTRUCTION_START": _record(tokenizer, alignment, instruction, instruction_char),
        "phase1_answer_span": [answer_tokens[0], answer_tokens[-1] + 1],
        "phase1_answer_token_ids": [alignment.input_ids[position] for position in answer_tokens],
        "causal_order_valid": True,
    }
    output["P1_CLE"].update(
        {"anchor_text": P1_CLE_ANCHOR, "anchor_start_index": cle_starts[0]}
    )
    return output

