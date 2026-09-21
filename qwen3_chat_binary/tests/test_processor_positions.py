from __future__ import annotations

from pathlib import Path

import pytest

from qwen3_chat_binary.config import DATASET_PATH, MODEL_PATH, POSITIONS
from qwen3_chat_binary.conversation import (
    prepare_multimodal_inputs,
    render_stage2,
    stage2_messages,
)
from qwen3_chat_binary.positions import locate_positions
from qwen3_chat_binary.prompts import phase0_prompt
from qwen3_chat_binary.scoring import label_token_ids
from qwen3_chat_binary.dataset import load_conflict_cases


def test_qwen3_processor_locates_both_boundary_variants() -> None:
    if not (MODEL_PATH / "tokenizer.json").is_file():
        pytest.skip("local Qwen3 tokenizer is unavailable")
    transformers = pytest.importorskip("transformers")
    processor = transformers.AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    case = load_conflict_cases(DATASET_PATH, max_samples=1)[0]
    image_path = case.image_path
    raw = "**Answer**: blue\n"
    located = {}
    for variant in ("native_boundary", "explicit_newline"):
        messages = stage2_messages(
            phase0_prompt(case.question, case.text_clue), image_path, raw, variant
        )
        assert sum(
            part.get("type") == "image"
            for message in messages for part in message["content"]
        ) == 1
        rendered = render_stage2(processor, messages)
        prefix_ids = processor.tokenizer.encode(rendered, add_special_tokens=False)
        for label, token_id in zip(("0", "1", "2", "3", "4"), label_token_ids(processor.tokenizer)):
            combined = processor.tokenizer.encode(rendered + label, add_special_tokens=False)
            assert combined[:-1] == prefix_ids
            assert combined[-1] == token_id
        inputs = prepare_multimodal_inputs(processor, messages, rendered)
        located[variant] = locate_positions(processor.tokenizer, rendered, inputs, raw, variant)
        indices = located[variant]["indices"]
        assert list(indices) == list(POSITIONS)
        assert indices["LAT"] < indices["PANL"] < indices["PANL+1"] < indices["CLE"] < indices["SAC"]
    assert located["native_boundary"]["positions"]["PANL+1"]["token_text"] == "<|im_start|>"
    assert located["explicit_newline"]["positions"]["PANL+1"]["token_text"] == "<|im_end|>"
    assert "newline" in located["native_boundary"]["positions"]["PANL"]["definition"]
