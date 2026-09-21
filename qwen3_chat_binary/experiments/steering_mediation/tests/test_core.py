from __future__ import annotations

import torch

from layer_metacognition.model_adapter import LanguageModules
from qwen3_chat_binary.experiments.steering_mediation.analyze import effect_values
from qwen3_chat_binary.experiments.steering_mediation.hooks import MediationHook


class Identity(torch.nn.Module):
    def forward(self, value): return value


def modules() -> LanguageModules:
    layers = [Identity() for _ in range(4)]
    return LanguageModules(layers, Identity(), Identity(), hidden_size=3, num_hidden_layers=4)


def run_layers(hook: MediationHook, value: torch.Tensor, layers: list[torch.nn.Module]) -> torch.Tensor:
    with hook:
        for layer in layers: value = layer(value)
    hook.validate(); return value


def test_steering_capture_restore_and_hook_cleanup() -> None:
    model = modules(); original = torch.zeros(1, 4, 3); vector = torch.tensor([1., 2., 3.])
    clean = MediationHook(model, prefill_sequence_length=4, upstream_position=1, downstream_position=2,
                          capture_targets={"donor": (2, 2)})
    run_layers(clean, original.clone(), model.language_layers); assert torch.equal(clean.captured["donor"], torch.zeros(3))
    c1 = MediationHook(model, prefill_sequence_length=4, upstream_position=1, downstream_position=2,
                       steering_layer=1, steering_vector=vector,
                       capture_targets={"corrupted": (2, 2)})
    result = run_layers(c1, original.clone(), model.language_layers)
    assert torch.equal(result[0, 1], vector)
    c2 = MediationHook(model, prefill_sequence_length=4, upstream_position=1, downstream_position=2,
                       steering_layer=1, steering_vector=vector, patch_layer=2,
                       patch_source=clean.captured["donor"], capture_targets={"restored": (2, 2)})
    result = run_layers(c2, original.clone(), model.language_layers)
    assert torch.equal(result[0, 1], vector) and torch.equal(result[0, 2], torch.zeros(3))
    assert all(len(layer._forward_hooks) == 0 for layer in model.language_layers)


def test_alpha_zero_and_effect_definitions() -> None:
    model = modules(); original = torch.randn(1, 4, 3)
    hook = MediationHook(model, prefill_sequence_length=4, upstream_position=1, downstream_position=2,
                         steering_layer=1, steering_vector=torch.zeros(3), capture_targets={"x": (2, 2)})
    result = run_layers(hook, original.clone(), model.language_layers)
    assert torch.equal(result, original)
    assert effect_values({"C0": .4, "C1": .7, "C2": .5, "C3": .6}) == {
        "total_effect": .7 - .4, "restore_residual": .5 - .4,
        "restored_amount": .7 - .5, "transplant_effect": .6 - .4,
    }


def test_hooks_are_removed_after_exception() -> None:
    model = modules()
    hook = MediationHook(model, prefill_sequence_length=4, upstream_position=1, downstream_position=2,
                         steering_layer=1, steering_vector=torch.ones(3))
    try:
        with hook:
            raise RuntimeError("synthetic failure")
    except RuntimeError:
        pass
    assert all(len(layer._forward_hooks) == 0 for layer in model.language_layers)
