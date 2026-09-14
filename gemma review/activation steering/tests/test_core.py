from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from analysis import bh_fdr, directional_gate
from capture import parse_layers
from experiment_config import CAPTURE_LAYERS, HIDDEN_SIZE, MODEL_PATH, POSITIONS
from gemma_runtime import AdditiveActivationHook, LanguageModules
from selection import select_manifests
from soft_score import soft_sa_from_logits
from steering import parse_steering_layers


def _row(item: int, score: float, hard: int) -> dict:
    return {
        "status": "completed",
        "valid_class": True,
        "case_id": f"case-{item}",
        "item_id": str(item),
        "prior_index": 0,
        "condition": "conflict_easy",
        "version": "v4_gemma",
        "soft_sa_image_score": score,
        "argmax_hard_class": hard,
    }


def test_capture_grid_and_steering_layer_validation():
    assert CAPTURE_LAYERS == tuple(range(6, 34))
    assert len(POSITIONS) * len(CAPTURE_LAYERS) == 168
    assert parse_layers(CAPTURE_LAYERS) == CAPTURE_LAYERS
    assert parse_steering_layers(None, smoke=True) == (22,)
    with pytest.raises(ValueError, match="requires"):
        parse_steering_layers(None)
    with pytest.raises(ValueError, match="L6-L33"):
        parse_layers([5])


def test_soft_sa_and_item_disjoint_selection():
    score = soft_sa_from_logits(np.zeros(9), list(range(9)))
    assert score["probability_sum"] == pytest.approx(1.0)
    rows = [_row(index, index / 180, 8 if index % 2 == 0 else 0) for index in range(180)]
    construction, test, _ = select_manifests(rows)
    assert len(construction) == 50
    assert len(test) == 100
    assert not ({row["item_id"] for row in construction} & {row["item_id"] for row in test})


def test_dynamic_directional_gate():
    rows = []
    for item in range(20):
        for alpha in (-10, -2, 0, 2, 10):
            rows.append(
                {
                    "status": "completed",
                    "item_id": str(item),
                    "position": "P1_LAT",
                    "layer": 22,
                    "direction_type": "true",
                    "alpha": alpha,
                    "delta_soft_sa": alpha * 0.01,
                }
            )
    selected, metrics = directional_gate(rows, repeats=100, seed=1)
    assert len(metrics) == 1
    assert selected[0]["position"] == "P1_LAT"
    assert bh_fdr([0.01, 0.02, 0.5]) == pytest.approx([0.03, 0.03, 0.5])


class _Block(torch.nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


def test_additive_hook_applies_once_to_target():
    layers = [_Block() for _ in range(34)]
    modules = LanguageModules(layers, torch.nn.Identity(), torch.nn.Linear(HIDDEN_SIZE, 10), HIDDEN_SIZE, 34)
    hidden = torch.zeros(1, 4, HIDDEN_SIZE)
    hook = AdditiveActivationHook(modules, 22, 2, torch.ones(HIDDEN_SIZE), 4)
    with hook:
        output = layers[22](hidden)
    assert torch.equal(output[0, 2], torch.ones(HIDDEN_SIZE))
    assert hook.diagnostics()["steering_applied_count"] == 1


def test_local_gemma_processor_template_and_positions():
    from PIL import Image
    from transformers import AutoProcessor, AutoTokenizer
    from dp_SA.prompts import SA_PREFILL, phase1_prompt
    from positions import locate_phase1_positions

    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True, use_fast=False)
    processor.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, use_fast=True)
    image = Image.new("RGB", (16, 16), "white")
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": phase1_prompt("q", "c", "blue")}]},
        {"role": "assistant", "content": [{"type": "text", "text": SA_PREFILL}]},
    ]
    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, continue_final_message=True
    )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
        do_pan_and_scan=False,
    )
    assert rendered.endswith(SA_PREFILL)
    assert inputs.input_ids[0, :2].tolist() != [2, 2]
    class_ids = [processor.tokenizer.encode(str(value), add_special_tokens=False) for value in range(9)]
    assert all(len(value) == 1 for value in class_ids)
    located = locate_phase1_positions(processor, rendered, inputs, "blue")
    assert tuple(name for name in POSITIONS if name in located) == POSITIONS
    order = [located[name]["processed_index"] for name in ("P1_LAT", "P1_PANL", "P1_CLE", "P1_SAC")]
    assert order == sorted(order)
    assert "\n" in located["P1_CLE"]["token_text"]
