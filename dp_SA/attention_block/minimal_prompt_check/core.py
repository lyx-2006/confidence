from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from dp_SA.io_utils import load_jsonl
from dp_SA.prompts import SA_PREFILL
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.token_positions import locate_image_pad_span
from layer_metacognition.token_spans import build_rendered_alignment

MINIMAL_PROMPT_TEMPLATE = """You will be shown a question, a text clue, an image, and an answer you previously provided.

Question:
{question}

Text clue:
{text_clue}

**Answer**: {answer}

State source attribution from 0 to 8: 0=text only, 4=both equally, 8=image only; intermediate integers indicate degree.
Do not choose class 4 merely because both sources were shown. Choose class 4 only if you believe the text clue and the image contributed to the fixed answer to a similar extent.
"""

WINDOWS = ((0, 11), (4, 15), (8, 19), (12, 23), (16, 27))
CONDITIONS = ("sac_to_panl", "sac_to_panl_plus_1", "empty_block_parity")


def minimal_prompt(question: str, text_clue: str, answer: str) -> str:
    if not answer or "\n" in answer or "\r" in answer:
        raise ValueError("Fixed answer must be non-empty and single-line")
    return MINIMAL_PROMPT_TEMPLATE.format(question=question, text_clue=text_clue, answer=answer)


def select_frozen_items(rows: Iterable[dict[str, Any]], per_side: int = 15) -> list[dict[str, Any]]:
    rows = list(rows)
    if any(row["phase0_raw_answer"] != row["phase1_inserted_raw_answer"] for row in rows):
        raise ValueError("Frozen delayed manifest contains a raw-answer byte mismatch")
    selected: list[dict[str, Any]] = []
    for side in ("image_side", "text_side"):
        candidates = sorted(
            (row for row in rows if row.get("test_side") == side),
            key=lambda row: (
                int(row.get("selection_rank", 10**9)),
                int(row.get("random_permutation_index", 10**9)),
                str(row["case_id"]),
            ),
        )
        if len(candidates) < per_side:
            raise ValueError(f"Insufficient frozen {side} items: {len(candidates)}/{per_side}")
        selected.extend(candidates[:per_side])
    if len({str(row["item_id"]) for row in selected}) != 2 * per_side:
        raise ValueError("Minimal-prompt selection is not item-unique")
    return sorted(selected, key=lambda row: (row["test_side"], int(row["selection_rank"]), str(row["case_id"])))


def load_selection(manifest_path: Path, per_side: int = 15) -> list[dict[str, Any]]:
    return select_frozen_items(load_jsonl(manifest_path), per_side=per_side)


def prepare_minimal_case(inference: Any, row: dict[str, Any]) -> tuple[str, Any, str]:
    prompt = minimal_prompt(row["question"], row["text_clue"], row["phase0_raw_answer"])
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": str(Path(row["image_path"]).resolve())},
            {"type": "text", "text": prompt},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": SA_PREFILL}]},
    ]
    rendered = render_continued_assistant(inference.processor, messages, SA_PREFILL)
    inputs = prepare_multimodal_inputs(
        inference.processor, messages, rendered, device=inference._get_inputs_device()
    )
    return rendered, inputs, prompt


def _token_span(alignment: Any, start: int, end: int) -> list[int]:
    tokens = alignment.processed_tokens_for_char_span(start, end)
    return [int(tokens[0]), int(tokens[-1]) + 1]


def locate_minimal_positions(
    tokenizer: Any,
    rendered: str,
    inputs: Any,
    answer: str,
    question: str | None = None,
    text_clue: str | None = None,
) -> dict[str, Any]:
    ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
    mask = inputs.get("attention_mask") if isinstance(inputs, dict) else inputs.attention_mask
    alignment = build_rendered_alignment(tokenizer, rendered, ids, mask)
    marker = f"**Answer**: {answer}"
    marker_start = rendered.find(marker)
    if marker_start < 0 or rendered.find(marker, marker_start + 1) >= 0:
        raise ValueError("Expected exactly one fixed-answer marker in rendered prompt")
    answer_start = marker_start + len("**Answer**: ")
    answer_end = answer_start + len(answer)
    answer_span = _token_span(alignment, answer_start, answer_end)
    if rendered[answer_end:answer_end + 2] != "\n\n":
        raise ValueError("Fixed answer is not followed by the registered PANL blank line")
    panl = alignment.processed_tokens_for_char_span(answer_end, answer_end + 1)[0]
    instruction_start = answer_end + 2
    if not rendered.startswith("State source attribution", instruction_start):
        raise ValueError("PANL+1 instruction start changed")
    plus1 = alignment.processed_tokens_for_char_span(instruction_start, instruction_start + len("State"))[0]
    sac_chars = rendered.rfind(SA_PREFILL)
    if sac_chars < 0 or sac_chars + len(SA_PREFILL) != len(rendered):
        raise ValueError("Rendered prompt does not end at the SAC prefill")
    sac_span = _token_span(alignment, sac_chars, len(rendered))
    sac = sac_span[1] - 1
    sequence_length = len(alignment.processed_ids)
    sa_pred = sequence_length
    if not (answer_span[0] < answer_span[1] <= panl < plus1 < sac < sa_pred):
        raise ValueError(
            f"Invalid minimal causal order: answer={answer_span}, PANL={panl}, "
            f"PANL+1={plus1}, SAC={sac}, SA_PRED={sa_pred}"
        )
    if sac != sequence_length - 1:
        raise ValueError(f"SAC must be the final processed input token: {sac}/{sequence_length}")
    result = {
        "sequence_length": sequence_length,
        "ANSWER": answer_span,
        "PANL": int(panl),
        "PANL_PLUS_1": int(plus1),
        "SAC": int(sac),
        "SA_PRED": int(sa_pred),
        "PANL_TO_SAC_DISTANCE": int(sac - panl),
        "PANL_TO_SA_PRED_DISTANCE": int(sa_pred - panl),
        "token_ids": alignment.processed_ids,
    }
    if question is not None and text_clue is not None:
        question_start = rendered.find(question)
        text_start = rendered.find(text_clue)
        if question_start < 0 or rendered.find(question, question_start + 1) >= 0:
            raise ValueError("Expected exactly one minimal question span")
        if text_start < 0 or rendered.find(text_clue, text_start + 1) >= 0:
            raise ValueError("Expected exactly one minimal text-clue span")
        question_span = _token_span(alignment, question_start, question_start + len(question))
        text_span = _token_span(alignment, text_start, text_start + len(text_clue))
        image_span = list(locate_image_pad_span(tokenizer, alignment.processed_ids)["span"])
        evidence = sorted(
            set(range(text_span[0], text_span[1])) | set(range(image_span[0], image_span[1]))
        )
        answer_positions = list(range(answer_span[0], answer_span[1]))
        if not evidence or not answer_positions:
            raise ValueError("Minimal E+A source positions must be non-empty")
        if set(evidence) & set(answer_positions):
            raise ValueError("Minimal EVIDENCE and ANSWER spans overlap")
        result.update({
            "QUESTION": question_span,
            "TEXT_CLUE": text_span,
            "IMAGE": image_span,
            "EVIDENCE": evidence,
            "EVIDENCE_ANSWER": sorted(set(evidence) | set(answer_positions)),
        })
    return result
