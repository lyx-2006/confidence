from __future__ import annotations

import torch

from dp_SA.attention_block.masking import AttentionBlockContext
from qwen3_chat_binary.experiments.attention_block.core import class_margin, deterministic_argmax, edge_for_condition


class ToyAttention(torch.nn.Module):
    def forward(self, hidden, *, attention_mask, cache_position):
        weights = torch.softmax(attention_mask.expand(1, 2, -1, -1).float(), dim=-1)
        return hidden, weights


class ToyLayer(torch.nn.Module):
    def __init__(self): super().__init__(); self.self_attn = ToyAttention()


def test_metrics_and_tie_policy() -> None:
    assert class_margin([5, 1, 1, 1, 1], 0) == 4
    assert deterministic_argmax([3, 3, 2, 1, 0]) == (0, True)
    assert deterministic_argmax([0, 1, 4, 2, 3]) == (2, False)


def test_exact_attention_edge_is_blocked_and_renormalized() -> None:
    layers = [ToyLayer() for _ in range(3)]; positions = {"SAC": 4, "PANL": 1}
    edges, source = edge_for_condition("SAC_to_PANL", positions)
    assert source == "PANL" and edges.pairs == ((4, 1),)
    mask = torch.zeros(1, 1, 5, 5); hidden = torch.zeros(1, 5, 2)
    with AttentionBlockContext(layers, layer_indices=(1, 2), edges=edges,
                               sequence_length=5, row_sum_tolerance=1e-6) as context:
        for layer in layers[1:]: layer.self_attn(hidden, attention_mask=mask, cache_position=torch.arange(5))
    diagnostics = context.diagnostics()
    assert diagnostics["layers"] == [1, 2]
    assert all(diagnostics["by_layer"][str(i)]["max_blocked_weight"] == 0 for i in (1, 2))
    assert all(diagnostics["by_layer"][str(i)]["max_row_sum_error"] <= 1e-6 for i in (1, 2))

