from __future__ import annotations

from pathlib import Path

import pytest
from transformers import AutoProcessor

from confidence_test.dataset_utils import load_evaluation_cases
from dp_SA.positions import locate_phase1_positions
from dp_SA.prompts import SA_PREFILL, phase1_prompt
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from Steering.capture import _external_positions, _messages
from Steering.config import DATASET_PATH, MODEL_PATH, POSITIONS


@pytest.mark.processor
def test_native_qwen3_template_and_five_positions():
    state = MODEL_PATH / ".download_state"
    if not any((state / f"tokenizer.json.{suffix}").is_file() for suffix in ("done", "complete")):
        pytest.skip("local Qwen3 tokenizer download is incomplete")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    cases, _ = load_evaluation_cases(DATASET_PATH, item_limit=1)
    case = cases[0]
    answer = "blue"
    prompt = phase1_prompt(case.question, case.text_clue, answer)
    messages = _messages(prompt, case.conditions["conflict_easy"].resolved_image_path, SA_PREFILL)
    rendered = render_continued_assistant(processor, messages, SA_PREFILL)
    assert not rendered.startswith("<|im_start|>system")
    inputs = prepare_multimodal_inputs(processor, messages, rendered)
    located = locate_phase1_positions(processor.tokenizer, rendered, inputs, answer)
    external = _external_positions(located)
    assert tuple(external) == POSITIONS
    indices = [external[name]["processed_index"] for name in ("LAT", "PANL", "PANL+1", "CLE", "SAC")]
    assert indices == sorted(indices)
    assert len(set(indices)) == 5
    assert external["PANL"]["processed_index"] + 1 == external["PANL+1"]["processed_index"]
