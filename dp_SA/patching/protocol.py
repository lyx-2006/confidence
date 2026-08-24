from __future__ import annotations

from pathlib import Path
from typing import Any

from dp_SA.attention_block.spans import locate_spans
from dp_SA.io_utils import canonical_hash, sha256_file
from dp_SA.positions import locate_phase1_positions
from dp_SA.prompts import SA_PREFILL, phase1_prompt
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant


def resolved_image(row: dict[str, Any], image_root: Path | None) -> Path:
    source = Path(str(row["image_path"]))
    path = (image_root / source.name) if image_root is not None else source
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    expected = row.get("image_sha256")
    if expected and sha256_file(path) != expected:
        raise ValueError(f"Image fingerprint changed for {row['case_id']}")
    return path


def prepare_delayed_case(inference: Any, row: dict[str, Any], *, image_root: Path | None = None) -> tuple[str, Any, dict[str, Any]]:
    answer = str(row["phase0_raw_answer"])
    if answer != row.get("phase1_inserted_raw_answer"):
        raise ValueError(f"Fixed answer byte parity failed for {row['case_id']}")
    prompt = phase1_prompt(str(row["question"]), str(row["text_clue"]), answer)
    if prompt != row.get("phase1_prompt"):
        raise ValueError(f"Frozen Phase 1 prompt changed for {row['case_id']}")
    if row.get("phase1_prompt_hash") and canonical_hash(prompt) != row["phase1_prompt_hash"]:
        raise ValueError(f"Phase 1 prompt hash changed for {row['case_id']}")
    image = resolved_image(row, image_root)
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": str(image)},
            {"type": "text", "text": prompt},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": SA_PREFILL}]},
    ]
    rendered = render_continued_assistant(inference.processor, messages, SA_PREFILL)
    inputs = prepare_multimodal_inputs(
        inference.processor, messages, rendered, device=inference._get_inputs_device()
    )
    tokenizer = getattr(inference.processor, "tokenizer", inference.processor)
    located = locate_phase1_positions(tokenizer, rendered, inputs, answer)
    for name in ("P1_PANL", "P1_PANL_PLUS_1", "P1_SAC"):
        stored = int(row["positions"][name]["processed_index"])
        current = int(located[name]["processed_index"])
        if stored != current:
            raise ValueError(f"Position changed for {row['case_id']} {name}: {current} != {stored}")
    if list(located["phase1_answer_span"]) != list(row["phase1_answer_span"]):
        raise ValueError(f"Fixed answer span changed for {row['case_id']}")
    if row.get("phase1_answer_token_ids") and list(located["phase1_answer_token_ids"]) != list(row["phase1_answer_token_ids"]):
        raise ValueError(f"Fixed answer tokens changed for {row['case_id']}")
    span_row = {**row, "arm": "delayed", "image_path": str(image)}
    spans = locate_spans(tokenizer, rendered, inputs, span_row)
    sets = {name: set(range(*spans[name])) for name in ("IMAGE", "TEXT_CLUE", "ANSWER")}
    if any(not value for value in sets.values()):
        raise ValueError("A corruption span is empty")
    if sets["IMAGE"] & sets["TEXT_CLUE"] or sets["IMAGE"] & sets["ANSWER"] or sets["TEXT_CLUE"] & sets["ANSWER"]:
        raise ValueError("Corruption spans overlap")
    length = int(inputs.input_ids.shape[1])
    if any(min(value) < 0 or max(value) >= length for value in sets.values()):
        raise ValueError("A corruption span is outside the prepared input")
    details = {
        "messages_hash": canonical_hash(messages), "rendered_hash": canonical_hash(rendered),
        "rendered": rendered, "spans": spans, "located": located,
        "image_path": str(image), "image_sha256": sha256_file(image),
        "question_hash": canonical_hash(row["question"]),
        "text_clue_hash": canonical_hash(row["text_clue"]),
        "answer_hash": canonical_hash(answer),
    }
    return rendered, inputs, details


def span_positions(spans: dict[str, Any], name: str) -> tuple[int, ...]:
    start, end = spans[name]
    output = tuple(range(int(start), int(end)))
    if not output:
        raise ValueError(f"Empty span: {name}")
    return output
