from __future__ import annotations

import json

import pytest
from transformers import AutoProcessor

from SA_trajectory.PANL2CLE.config import CAPTURE_ROOT, MODEL_PATH
from Steering.capture import _messages
from dp_SA.positions import locate_phase1_positions
from dp_SA.prompts import SA_PREFILL
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant


@pytest.mark.processor
def test_qwen3_native_template_and_position_parity():
    row = json.loads(next((CAPTURE_ROOT/"results.jsonl").open()))
    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    messages = _messages(row["phase1_prompt"], row["image_path"], SA_PREFILL); rendered = render_continued_assistant(processor, messages, SA_PREFILL)
    assert not rendered.startswith("<|im_start|>system")
    inputs = prepare_multimodal_inputs(processor, messages, rendered)
    located = locate_phase1_positions(processor.tokenizer, rendered, inputs, row["phase0_raw_answer"])
    names = (("PANL", "P1_PANL"), ("CLE", "P1_CLASS_LIST_END"), ("SAC", "P1_SAC")); indices = []
    for external, internal in names:
        old, new = row["positions"][external], located[internal]
        assert (old["processed_index"], old["token_id"]) == (new["processed_index"], new["token_id"]); indices.append(new["processed_index"])
    assert indices == sorted(indices) and len(set(indices)) == 3
