from __future__ import annotations

import numpy as np
import torch

from layer_metacognition.model_adapter import LanguageModules
from dp_SA.SA_trajectory.LAT2PANL.hooks import LATPANLMediationHook
from dp_SA.SA_trajectory.LAT2PANL.io_utils import atomic_bf16_npz, bf16_to_uint16, load_bf16_npz, uint16_to_bf16


def test_bf16_bit_pattern_round_trip(tmp_path):
    source = torch.tensor([1.0, -2.5, float("inf"), -0.0], dtype=torch.bfloat16)
    bits = bf16_to_uint16(source)
    restored = uint16_to_bf16(bits)
    assert torch.equal(source.view(torch.uint16), restored.view(torch.uint16))
    path = tmp_path / "bits.npz"
    metadata = atomic_bf16_npz(path, {"x": source})
    loaded, item = load_bf16_npz(path, "x")
    assert metadata["format"] == "raw_bfloat16_bits_v1"
    assert item["storage_dtype"] == "uint16"
    assert torch.equal(source.view(torch.uint16), loaded.view(torch.uint16))


def test_joint_l14_hook_changes_only_lat_and_panl():
    layers = [torch.nn.Identity() for _ in range(22)]
    modules = LanguageModules(layers, torch.nn.Identity(), torch.nn.Linear(4, 2), 4, len(layers))
    source = torch.tensor([7, 8, 9, 10], dtype=torch.bfloat16)
    vector = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    hidden = torch.arange(20, dtype=torch.float32).reshape(1, 5, 4).to(torch.bfloat16)
    original = hidden.clone()
    hook = LATPANLMediationHook(modules, prefill_sequence_length=5, lat_position=1, panl_position=2,
                                cle_position=3, steering_vector=vector, patch_layer=14,
                                patch_source=source, capture_panl_layers=(14,))
    with hook:
        hidden = layers[14](hidden)
        hidden = layers[20](hidden)
    hook.validate(); diagnostics = hook.diagnostics()
    assert diagnostics["non_target_unchanged"]
    assert diagnostics["replacement_bitwise_equal"]
    assert torch.equal(hidden[0, 0], original[0, 0]) and torch.equal(hidden[0, 4], original[0, 4])
    assert torch.equal(hidden[0, 2].view(torch.uint16), source.view(torch.uint16))
    assert torch.equal(hidden[0, 1], original[0, 1] + vector.to(torch.bfloat16))
    assert torch.equal(hook.captured_panl[14].view(torch.uint16), source.view(torch.uint16))
