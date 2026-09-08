from __future__ import annotations

from pathlib import Path
from typing import Any

from dp_SA.io_utils import canonical_hash, sha256_file
from dp_SA.prompts import ANSWER_PREFILL, phase0_prompt
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.token_positions import locate_image_pad_span
from layer_metacognition.token_spans import build_rendered_alignment, unique_text_span, verify_decoded_span


def resolved_image(row: dict[str, Any]) -> Path:
    path = Path(str(row["image_path"])).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    expected = row.get("image_sha256") or row.get("image_hash")
    actual = sha256_file(path)
    if expected and str(expected) != actual:
        raise ValueError(f"Image SHA256 changed: {row.get('case_id', row.get('unique_key'))}")
    return path


def phase0_messages(row: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    prompt = phase0_prompt(str(row["question"]), str(row["text_clue"]))
    image = resolved_image(row)
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": str(image)},
            {"type": "text", "text": prompt},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": ANSWER_PREFILL}]},
    ]
    return prompt, messages


def locate_evidence(processor: Any, row: dict[str, Any], rendered: str, inputs: Any) -> dict[str, Any]:
    ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
    mask = inputs.get("attention_mask") if isinstance(inputs, dict) else inputs.attention_mask
    tokenizer = getattr(processor, "tokenizer", processor)
    alignment = build_rendered_alignment(tokenizer, rendered, ids, mask)
    text_chars = unique_text_span(rendered, str(row["text_clue"]))
    text_positions = alignment.processed_tokens_for_char_span(*text_chars)
    if text_positions != list(range(text_positions[0], text_positions[-1] + 1)):
        raise ValueError("Text clue tokens are not contiguous")
    text_ids = [alignment.processed_ids[index] for index in text_positions]
    verify_decoded_span(tokenizer, text_ids, str(row["text_clue"]))
    image_span = locate_image_pad_span(tokenizer, alignment.processed_ids)["span"]
    image_positions = list(range(int(image_span[0]), int(image_span[1])))
    if not text_positions or not image_positions or set(text_positions) & set(image_positions):
        raise ValueError("Phase 0 evidence spans are empty or overlapping")
    return {
        "sequence_length": len(alignment.processed_ids), "image_positions": image_positions,
        "text_positions": text_positions, "text_token_ids": text_ids,
    }


def prepare_phase0(processor: Any, row: dict[str, Any], *, device: Any | None = None) -> tuple[str, Any, dict[str, Any]]:
    prompt, messages = phase0_messages(row)
    historical = row.get("phase0_prompt")
    if historical is not None and prompt != historical:
        raise ValueError(f"Frozen Phase 0 prompt changed for {row.get('case_id')}")
    expected_hash = row.get("phase0_prompt_hash")
    if expected_hash and canonical_hash(prompt) != expected_hash:
        raise ValueError(f"Frozen Phase 0 prompt hash changed for {row.get('case_id')}")
    rendered = render_continued_assistant(processor, messages, ANSWER_PREFILL)
    inputs = prepare_multimodal_inputs(processor, messages, rendered, device=device)
    spans = locate_evidence(processor, row, rendered, inputs)
    details = {
        "prompt": prompt,
        "prompt_hash": canonical_hash(prompt),
        "rendered": rendered,
        "rendered_hash": canonical_hash(rendered),
        "messages_hash": canonical_hash(messages),
        "image_path": str(resolved_image(row)),
        "image_sha256": sha256_file(resolved_image(row)),
        **spans,
    }
    return rendered, inputs, details


__all__ = ["locate_evidence", "phase0_messages", "prepare_phase0", "resolved_image"]
