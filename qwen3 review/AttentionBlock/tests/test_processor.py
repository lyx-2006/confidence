from __future__ import annotations

import json

import pytest
from transformers import AutoProcessor

from AttentionBlock.contracts import add_cle_plus_1
from Steering.capture import _messages
from Steering.config import MODEL_PATH
from dp_SA.positions import locate_phase1_positions
from dp_SA.prompts import SA_PREFILL
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant


@pytest.mark.processor
def test_qwen3_processor_reconstructs_transport_positions():
    row = json.loads(next(open(MODEL_PATH.parents[1] / "qwen3 review/capture/results.jsonl")))
    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    messages = _messages(row["phase1_prompt"], row["image_path"], SA_PREFILL)
    rendered = render_continued_assistant(processor, messages, SA_PREFILL)
    inputs = prepare_multimodal_inputs(processor, messages, rendered)
    located = add_cle_plus_1(
        locate_phase1_positions(processor.tokenizer, rendered, inputs, row["phase0_raw_answer"]),
        processor.tokenizer, inputs.input_ids,
    )
    order = [located[name]["processed_index"] for name in (
        "P1_PANL", "P1_PANL_PLUS_1", "P1_CLASS_LIST_END",
        "P1_CLASS_LIST_END_PLUS_1", "P1_SAC",
    )]
    assert order == sorted(order) and len(set(order)) == 5

