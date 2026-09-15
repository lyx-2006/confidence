"""Post-softmax Qwen3 attention blocking without renormalizing other weights.

This context patches only the eager attention function while active. For selected
text layers it computes the ordinary softmax weights, sets explicit query/key
entries to zero, and performs the value aggregation with those patched weights.
Every non-target entry is therefore bitwise unchanged from the ordinary eager
attention matrix produced in the same forward call.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class PreservedAttentionEdges:
    pairs: tuple[tuple[int, int], ...]


def zero_edges_without_renormalization(
    weights: torch.Tensor,
    local_targets: torch.Tensor,
    sources: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float | bool | int]]:
    """Return weights with target edges zeroed and audit all other entries."""
    if weights.ndim != 4:
        raise ValueError(f"Expected [batch, head, query, key], got {tuple(weights.shape)}")
    if local_targets.numel() != sources.numel():
        raise ValueError("Target/source index count mismatch")
    patched = weights.clone()
    before = weights[..., local_targets, sources]
    patched[..., local_targets, sources] = 0

    # Restore the blocked entries before comparison. Any remaining difference is
    # necessarily an unintended change outside the requested edges.
    comparison = patched.clone()
    comparison[..., local_targets, sources] = weights[..., local_targets, sources]
    max_other_change = float((comparison - weights).abs().max().item())
    blocked_after = patched[..., local_targets, sources]
    targets = torch.unique(local_targets)
    original_rows = weights[..., targets, :].float()
    patched_rows = patched[..., targets, :].float()
    removed_by_row = (original_rows - patched_rows).sum(dim=-1)
    row_accounting_error = float(
        ((patched_rows.sum(dim=-1) + removed_by_row) - original_rows.sum(dim=-1)).abs().max().item()
    )
    return patched, {
        "edge_count": int(local_targets.numel()),
        "max_removed_weight": float(before.abs().max().item()),
        "mean_removed_weight": float(before.float().mean().item()),
        "max_blocked_weight_after": float(blocked_after.abs().max().item()),
        "max_other_weight_change": max_other_change,
        "max_row_accounting_error": row_accounting_error,
        "original_rows_finite": bool(torch.isfinite(original_rows).all()),
        "patched_rows_finite": bool(torch.isfinite(patched_rows).all()),
    }


class PreserveOtherAttentionContext:
    """Temporarily zero selected post-softmax Qwen3 attention edges."""

    def __init__(
        self,
        *,
        layer_indices: Sequence[int],
        edges: PreservedAttentionEdges,
        sequence_length: int,
    ) -> None:
        self.layer_indices = tuple(sorted({int(x) for x in layer_indices}))
        self.edges = edges
        self.sequence_length = int(sequence_length)
        if not self.layer_indices:
            raise ValueError("At least one blocked layer is required")
        for target, source in edges.pairs:
            if not (0 <= source <= target < self.sequence_length):
                raise ValueError(f"Invalid causal attention edge TARGET={target} SOURCE={source}")
        self._calls = {layer: 0 for layer in self.layer_indices}
        self._diagnostics: dict[int, dict[str, Any]] = {}
        self._original = None
        self._module = None

    def __enter__(self) -> "PreserveOtherAttentionContext":
        from transformers.models.qwen3_vl import modeling_qwen3_vl

        if getattr(modeling_qwen3_vl.eager_attention_forward, "_preserve_other_weights_patch", False):
            raise RuntimeError("A preserve-other-weights attention context is already active")
        self._module = modeling_qwen3_vl
        self._original = modeling_qwen3_vl.eager_attention_forward
        original = self._original
        selected_layers = set(self.layer_indices)

        def patched_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            layer = int(getattr(module, "layer_idx", -1))
            if layer not in selected_layers:
                return original(module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs)
            self._calls[layer] += 1
            if int(query.shape[-2]) != self.sequence_length or int(key.shape[-2]) != self.sequence_length:
                raise RuntimeError(
                    "Preserve-other-weights blocking is defined only for the complete prefill forward"
                )
            key_states = modeling_qwen3_vl.repeat_kv(key, module.num_key_value_groups)
            value_states = modeling_qwen3_vl.repeat_kv(value, module.num_key_value_groups)
            weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
            if attention_mask is not None:
                weights = weights + attention_mask[:, :, :, : key_states.shape[-2]]
            weights = torch.nn.functional.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
            weights = torch.nn.functional.dropout(weights, p=dropout, training=module.training)
            targets = torch.tensor([x for x, _ in self.edges.pairs], device=weights.device, dtype=torch.long)
            sources = torch.tensor([x for _, x in self.edges.pairs], device=weights.device, dtype=torch.long)
            weights, audit = zero_edges_without_renormalization(weights, targets, sources)
            previous = self._diagnostics.get(layer)
            if previous is not None:
                raise RuntimeError(f"Selected layer {layer} was patched more than once")
            self._diagnostics[layer] = audit
            output = torch.matmul(weights, value_states).transpose(1, 2).contiguous()
            return output, weights

        patched_forward._preserve_other_weights_patch = True
        modeling_qwen3_vl.eager_attention_forward = patched_forward
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        if self._module is not None and self._original is not None:
            self._module.eager_attention_forward = self._original
        self._module = None
        self._original = None

    def diagnostics(self) -> dict[str, Any]:
        missing = sorted(set(self.layer_indices) - set(self._diagnostics))
        repeated = {layer: count for layer, count in self._calls.items() if count != 1}
        if missing or repeated:
            raise RuntimeError(f"Attention patch audit failed: missing={missing}, calls={repeated}")
        for layer, audit in self._diagnostics.items():
            if audit["max_blocked_weight_after"] != 0.0:
                raise RuntimeError(f"Blocked edge is nonzero at layer {layer}")
            if audit["max_other_weight_change"] != 0.0:
                raise RuntimeError(f"Non-target attention changed at layer {layer}")
            if not audit["original_rows_finite"] or not audit["patched_rows_finite"]:
                raise RuntimeError(f"Non-finite attention at layer {layer}")
            audit["hook_call_count"] = self._calls[layer]
        return {
            "mechanism": "post_softmax_zero_without_renormalization",
            "other_weights_policy": "bitwise_unchanged",
            "layers": list(self.layer_indices),
            "by_layer": {str(k): v for k, v in self._diagnostics.items()},
        }
