from __future__ import annotations

import torch

from dp_SA.attention_block.masking import AttentionBlockContext, AttentionEdges


class RecordingAttention(torch.nn.Module):
    def __init__(self, heads=3):
        super().__init__(); self.heads = heads; self.seen_mask = None

    def forward(self, hidden_states, attention_mask=None):
        self.seen_mask = attention_mask.detach().clone()
        batch, length, _ = hidden_states.shape
        logits = torch.zeros(batch, self.heads, length, length) + attention_mask
        weights = torch.softmax(logits, dim=-1)
        return hidden_states, weights


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.self_attn = RecordingAttention()


def causal(length):
    minimum = torch.finfo(torch.float32).min
    return torch.triu(torch.full((1, 1, length, length), minimum), diagonal=1)


def test_exact_mask_coordinate_all_heads_renormalization_and_window_hooks():
    layers = [Layer() for _ in range(5)]
    hidden = torch.zeros(1, 6, 2); base = causal(6); original = base.clone()
    clean = layers[0].self_attn(hidden, attention_mask=base)[1]
    with AttentionBlockContext(layers, layer_indices=(1, 2, 3),
                               edges=AttentionEdges(((5, 2),)), sequence_length=6,
                               row_sum_tolerance=1e-6) as context:
        outputs = [layers[index].self_attn(hidden, attention_mask=base)[1] for index in (1, 2, 3)]
    assert torch.equal(base, original)
    for index, weights in zip((1, 2, 3), outputs):
        changed = layers[index].self_attn.seen_mask != base
        assert changed.nonzero().tolist() == [[0, 0, 5, 2]]
        assert torch.all(weights[0, :, 5, 2] == 0)
        assert torch.allclose(weights[0, :, 5].sum(-1), torch.ones(3))
        assert torch.all(weights[0, :, 5, 1] > clean[0, :, 5, 1])
    diagnostics = context.diagnostics()
    assert diagnostics["layers"] == [1, 2, 3]
    assert all(row["hook_call_count"] == 1 for row in diagnostics["by_layer"].values())
