from __future__ import annotations

import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parents[1]
SHORT_ROOT = HERE.parent
for path in (SHORT_ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from attention_block.preserve_other_weights import (
    PreserveOtherAttentionContext,
    PreservedAttentionEdges,
    zero_edges_without_renormalization,
)
from attention_block.run_sac2panl_preserve import CONDITIONS, METRICS, QUERY, SOURCES, WINDOWS, validate_contract


def test_zero_edge_preserves_every_other_weight_exactly() -> None:
    weights = torch.tensor(
        [[[[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5]]]], dtype=torch.float32
    )
    target = torch.tensor([2])
    source = torch.tensor([0])
    patched, audit = zero_edges_without_renormalization(weights, target, source)
    assert patched[0, 0, 2, 0].item() == 0.0
    mask = torch.ones_like(weights, dtype=torch.bool)
    mask[..., target, source] = False
    assert torch.equal(patched[mask], weights[mask])
    assert patched[0, 0, 2].sum().item() == torch.tensor(0.8).item()
    assert audit["max_other_weight_change"] == 0.0
    assert audit["max_blocked_weight_after"] == 0.0
    assert audit["max_row_accounting_error"] == 0.0


def test_sac2panl_contract_is_exact() -> None:
    assert QUERY == "P1_SAC"
    assert SOURCES == {
        "C1_main_block": "P1_PANL",
        "C2_source_plus_1_control": "P1_PANL_PLUS_1",
    }
    assert CONDITIONS == ("C1_main_block", "C2_source_plus_1_control")
    assert WINDOWS == ((8, 12), (12, 16), (16, 20), (20, 24))
    assert METRICS == ("delta_soft_sa", "logit_change_diff", "token_change_rate")


def test_context_patches_selected_layer_and_restores_function() -> None:
    from transformers.models.qwen3_vl import modeling_qwen3_vl

    class Module:
        layer_idx = 3
        num_key_value_groups = 1
        training = False

    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]])
    key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]])
    value = torch.arange(6, dtype=torch.float32).reshape(1, 1, 3, 2)
    mask = torch.zeros((1, 1, 3, 3), dtype=torch.float32)
    original = modeling_qwen3_vl.eager_attention_forward
    _, baseline = original(Module(), query, key, value, mask, scaling=1.0)
    with PreserveOtherAttentionContext(
        layer_indices=[3], edges=PreservedAttentionEdges(((2, 0),)), sequence_length=3
    ) as context:
        _, blocked = modeling_qwen3_vl.eager_attention_forward(
            Module(), query, key, value, mask, scaling=1.0
        )
    diagnostics = context.diagnostics()
    assert modeling_qwen3_vl.eager_attention_forward is original
    assert blocked[0, 0, 2, 0].item() == 0.0
    keep = torch.ones_like(baseline, dtype=torch.bool)
    keep[..., 2, 0] = False
    assert torch.equal(blocked[keep], baseline[keep])
    assert diagnostics["by_layer"]["3"]["max_other_weight_change"] == 0.0
    assert diagnostics["by_layer"]["3"]["hook_call_count"] == 1


def test_existing_gate_manifest_contract() -> None:
    result = validate_contract()
    assert result["status"] == "validated"
    assert result["case_count"] == 64
    assert result["other_weights_policy"] == "bitwise_unchanged"
