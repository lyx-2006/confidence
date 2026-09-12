from __future__ import annotations

from typing import Any

from layer_metacognition.token_spans import build_rendered_alignment

from .config import WINDOW_SIZE

WINDOW_SPECS = {
    "W1": ("State what you believe the fixed answer above was mainly based on.", "newline", "Attribution query"),
    "W2": ("Source attribution refers to the relative contribution of the text clue and the image to the formation of the fixed answer.", "period", "Attribution definition"),
    "W3": ("Report whether the fixed answer was based more on the text clue, more on the image, or on both sources to a similar extent.", "newline", "Modality comparison"),
    "W4": ("State your source attribution as exactly one integer between 0 and 8 using the classes below.", "newline", "Integer instruction"),
    "W5": ("A higher class indicates stronger image contribution.", "period", "Image polarity"),
    "W6": ("Do not choose class 4 merely because both sources were shown. Choose class 4 only if you believe the text clue and the image contributed to the fixed answer to a similar extent.", "newline", "Class-4 rule"),
}


def _occurrences(text: str, needle: str) -> list[int]:
    starts: list[int] = []; cursor = 0
    while True:
        found = text.find(needle, cursor)
        if found < 0: return starts
        starts.append(found); cursor = found + 1


def locate_swap_windows(tokenizer: Any, rendered: str, inputs: Any) -> dict[str, dict[str, Any]]:
    ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
    mask = inputs.get("attention_mask") if isinstance(inputs, dict) else getattr(inputs, "attention_mask", None)
    alignment = build_rendered_alignment(tokenizer, rendered, ids, mask)
    output: dict[str, dict[str, Any]] = {}
    previous = -1
    for name, (anchor, target_kind, meaning) in WINDOW_SPECS.items():
        starts = _occurrences(rendered, anchor)
        if len(starts) != 1: raise ValueError(f"{name}: expected one anchor, found {len(starts)}")
        anchor_start = starts[0]
        if target_kind == "newline":
            rendered_target = anchor_start + len(anchor)
            if rendered_target >= len(rendered) or rendered[rendered_target] != "\n":
                raise ValueError(f"{name}: anchor is not immediately followed by newline")
            mapped = alignment.processed_tokens_for_char_span(rendered_target, rendered_target + 1)
            endpoint = int(mapped[0])
            token_text = tokenizer.decode([alignment.processed_ids[endpoint]], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if "\n" not in token_text: raise ValueError(f"{name}: endpoint is not a newline token: {token_text!r}")
        else:
            rendered_target = anchor_start + len(anchor) - 1
            if rendered[rendered_target] != ".": raise ValueError(f"{name}: anchor does not end at period")
            mapped = alignment.processed_tokens_for_char_span(rendered_target, rendered_target + 1)
            endpoint = int(mapped[-1])
        start = endpoint - WINDOW_SIZE + 1
        if start < 0: raise ValueError(f"{name}: insufficient tokens for window")
        indices = list(range(start, endpoint + 1)); token_ids = [int(alignment.processed_ids[index]) for index in indices]
        decoded_tokens = [tokenizer.decode([token], skip_special_tokens=False, clean_up_tokenization_spaces=False) for token in token_ids]
        if endpoint <= previous: raise ValueError(f"Window causal order failed at {name}")
        previous = endpoint
        reverse = {processed: rendered_index for rendered_index, processed in alignment.rendered_to_processed.items()}
        if start not in reverse or endpoint not in reverse:
            raise ValueError(f"{name}: textual window contains an expanded image-only token")
        rendered_span = (alignment.offsets[reverse[start]][0], alignment.offsets[reverse[endpoint]][1])
        output[name] = {
            "name": name, "meaning": meaning, "anchor": anchor, "target_kind": target_kind,
            "anchor_rendered_span": [anchor_start, anchor_start + len(anchor)],
            "endpoint_rendered_index": rendered_target, "rendered_span": list(rendered_span),
            "processed_start": start, "processed_end": endpoint,
            "processed_indices": indices, "token_ids": token_ids,
            "decoded_tokens": decoded_tokens, "decoded_window": tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
            "window_length": len(indices),
        }
    return output
