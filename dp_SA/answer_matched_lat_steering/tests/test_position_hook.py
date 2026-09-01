from __future__ import annotations

import torch

from layer_metacognition.model_adapter import AdditiveActivationHook, LanguageModules


def test_panl_hook_changes_only_the_requested_token():
    modules = LanguageModules(language_layers=[torch.nn.Identity()], final_norm=torch.nn.Identity(), lm_head=torch.nn.Identity(), hidden_size=2, num_hidden_layers=1)
    hook = AdditiveActivationHook(modules, layer_index=0, target_position=2, steering_vector=torch.tensor([1.0, -1.0]), prefill_sequence_length=4)
    original = torch.zeros((1, 4, 2)); patched = hook._patch_tensor(original)
    assert torch.equal(patched[0, 0], original[0, 0]) and torch.equal(patched[0, 1], original[0, 1]) and torch.equal(patched[0, 3], original[0, 3])
    assert torch.equal(patched[0, 2], torch.tensor([1.0, -1.0]))
