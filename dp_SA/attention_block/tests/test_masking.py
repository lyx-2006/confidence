from __future__ import annotations

import torch

from dp_SA.attention_block.masking import AttentionBlockContext, AttentionEdges


class ToyAttention(torch.nn.Module):
    def __init__(self, heads: int = 3):
        super().__init__(); self.heads=heads

    def forward(self, hidden_states, attention_mask=None):
        batch, length, _ = hidden_states.shape
        logits=torch.zeros(batch,self.heads,length,length,dtype=hidden_states.dtype)
        logits=logits+attention_mask
        weights=torch.softmax(logits,dim=-1,dtype=torch.float32).to(hidden_states.dtype)
        return hidden_states,weights


class ToyLayer(torch.nn.Module):
    def __init__(self): super().__init__(); self.self_attn=ToyAttention()


def causal(length: int):
    mask=torch.full((1,1,length,length),torch.finfo(torch.float32).min)
    return torch.triu(mask,diagonal=1)


def test_target_to_source_direction_all_heads_and_renormalization():
    layers=[ToyLayer() for _ in range(3)]; hidden=torch.zeros(1,5,2); base=causal(5)
    with AttentionBlockContext(layers,layer_indices=[1],edges=AttentionEdges.from_sets([4],[1,2]),sequence_length=5,row_sum_tolerance=1e-6) as context:
        _out, weights=layers[1].self_attn(hidden,attention_mask=base)
    assert torch.count_nonzero(weights[0,:,4,[1,2]]) == 0
    assert torch.allclose(weights[0,:,4].sum(-1),torch.ones(3))
    assert torch.all(weights[0,:,1,0] > 0)  # source is not itself ablated
    diagnostics=context.diagnostics(); assert diagnostics["by_layer"]["1"]["head_count"]==3


def test_only_requested_layer_and_edges_change():
    layers=[ToyLayer() for _ in range(2)]; hidden=torch.zeros(1,4,2); base=causal(4)
    clean=layers[0].self_attn(hidden,attention_mask=base)[1]
    with AttentionBlockContext(layers,layer_indices=[1],edges=AttentionEdges.from_sets([3],[0]),sequence_length=4,row_sum_tolerance=1e-6):
        untouched=layers[0].self_attn(hidden,attention_mask=base)[1]
        _out, blocked=layers[1].self_attn(hidden,attention_mask=base)
    assert torch.equal(clean,untouched)
    assert torch.all(blocked[0,:,3,0] == 0)
    assert torch.all(blocked[0,:,2,0] > 0)


def test_empty_block_is_exact_clean():
    layers=[ToyLayer()]; hidden=torch.zeros(1,4,2); base=causal(4)
    clean=layers[0].self_attn(hidden,attention_mask=base)[1]
    with AttentionBlockContext(layers,layer_indices=[0],edges=AttentionEdges(()),sequence_length=4) as context:
        empty=layers[0].self_attn(hidden,attention_mask=base)[1]
    assert torch.equal(clean,empty); assert context.diagnostics()["empty"] is True


def test_keep_restores_only_requested_edges():
    edges=AttentionEdges.from_sets([2,3],[0,1]).without([2],[0,1])
    assert edges.pairs == ((3,0),(3,1))
