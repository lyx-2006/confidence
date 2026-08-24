from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch


@dataclass(frozen=True)
class AttentionEdges:
    """Explicit TARGET -> SOURCE query/key pairs."""

    pairs: tuple[tuple[int, int], ...]

    @classmethod
    def from_sets(cls, targets: Iterable[int], sources: Iterable[int]) -> "AttentionEdges":
        return cls(tuple(sorted({(int(t), int(s)) for t in targets for s in sources})))

    def without(self, targets: Iterable[int], sources: Iterable[int]) -> "AttentionEdges":
        restored = {(int(t), int(s)) for t in targets for s in sources}
        return AttentionEdges(tuple(pair for pair in self.pairs if pair not in restored))


class AttentionBlockContext:
    """Temporarily add exact per-layer attention-mask edges to Qwen eager attention."""

    def __init__(
        self,
        language_layers: Sequence[torch.nn.Module],
        *,
        layer_indices: Sequence[int],
        edges: AttentionEdges,
        sequence_length: int,
        validate_weights: bool = True,
        row_sum_tolerance: float = 0.01,
    ) -> None:
        self.language_layers = list(language_layers)
        self.layer_indices = tuple(sorted({int(x) for x in layer_indices}))
        self.edges = edges
        self.sequence_length = int(sequence_length)
        self.validate_weights = bool(validate_weights)
        self.row_sum_tolerance = float(row_sum_tolerance)
        if not self.layer_indices:
            raise ValueError("At least one blocked layer is required")
        if any(x < 0 or x >= len(self.language_layers) for x in self.layer_indices):
            raise ValueError(f"Blocked layers outside model: {self.layer_indices}")
        for target, source in self.edges.pairs:
            if not (0 <= source <= target < self.sequence_length):
                raise ValueError(f"Invalid causal attention edge TARGET={target} SOURCE={source}")
        self._handles: list[Any] = []
        self._calls = {layer: 0 for layer in self.layer_indices}
        self._applied_counts = {layer: 0 for layer in self.layer_indices}
        self._current_pairs: dict[int, tuple[torch.Tensor, torch.Tensor] | None] = {
            layer: None for layer in self.layer_indices
        }
        self._diagnostics: dict[int, dict[str, Any]] = {}
        self._edge_targets_cpu = torch.tensor(
            [target for target, _source in self.edges.pairs], dtype=torch.long
        )
        self._edge_sources_cpu = torch.tensor(
            [source for _target, source in self.edges.pairs], dtype=torch.long
        )
        self._device_indices: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def _indices_on(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key = str(device)
        if key not in self._device_indices:
            self._device_indices[key] = (
                self._edge_targets_cpu.to(device=device, non_blocking=True),
                self._edge_sources_cpu.to(device=device, non_blocking=True),
            )
        return self._device_indices[key]

    def _pre_hook(self, layer: int, _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
        self._calls[layer] += 1
        mask = kwargs.get("attention_mask")
        if mask is None:
            raise RuntimeError("Qwen eager attention did not receive an additive attention_mask")
        if mask.ndim != 4:
            raise ValueError(f"Unexpected attention mask shape: {tuple(mask.shape)}")
        query_length, key_length = int(mask.shape[-2]), int(mask.shape[-1])
        cache_position = kwargs.get("cache_position")
        if cache_position is not None:
            values = cache_position.detach().cpu().reshape(-1).tolist()
            global_queries = [int(x) for x in values]
        else:
            global_queries = list(range(key_length - query_length, key_length))
        if len(global_queries) != query_length:
            raise ValueError(f"cache_position/query mismatch: {global_queries} vs {query_length}")
        edge_targets, edge_sources = self._indices_on(mask.device)
        if query_length == self.sequence_length and global_queries == list(range(self.sequence_length)):
            local_targets, sources = edge_targets, edge_sources
        else:
            local_by_global = {value: index for index, value in enumerate(global_queries)}
            active = [(local_by_global[target], source) for target, source in self.edges.pairs
                      if target in local_by_global and source < key_length]
            if active:
                local_targets = torch.tensor([pair[0] for pair in active], dtype=torch.long, device=mask.device)
                sources = torch.tensor([pair[1] for pair in active], dtype=torch.long, device=mask.device)
            else:
                local_targets = edge_targets[:0]
                sources = edge_sources[:0]
        self._current_pairs[layer] = (local_targets, sources)
        if local_targets.numel() == 0:
            return args, kwargs
        patched = mask.clone()
        minimum = torch.finfo(patched.dtype).min
        patched[..., local_targets, sources] = minimum
        self._applied_counts[layer] += int(local_targets.numel())
        kwargs["attention_mask"] = patched
        return args, kwargs

    def _post_hook(self, layer: int, _module: Any, _args: Any, output: Any):
        active = self._current_pairs[layer]
        self._current_pairs[layer] = None
        if not self.validate_weights or active is None or active[0].numel() == 0:
            return output
        if not isinstance(output, tuple) or len(output) < 2 or not isinstance(output[1], torch.Tensor):
            raise TypeError("Attention module did not return eager attention weights")
        weights = output[1]
        if weights.ndim != 4:
            raise ValueError(f"Unexpected attention weights shape: {tuple(weights.shape)}")
        local_targets, sources = active
        blocked = weights[0, :, local_targets, sources]
        targets = torch.unique(local_targets)
        rows = weights[0, :, targets, :].float()
        max_blocked = float(blocked.abs().max().item())
        max_row_error = float((rows.sum(dim=-1) - 1.0).abs().max().item())
        finite = bool(torch.isfinite(rows).all())
        if max_blocked != 0.0:
            raise RuntimeError(f"Blocked attention is not exactly zero at layer {layer}: {max_blocked}")
        if max_row_error > self.row_sum_tolerance:
            raise RuntimeError(f"Attention row renormalization failed at layer {layer}: {max_row_error}")
        if not finite:
            raise RuntimeError(f"Non-finite attention at layer {layer}")
        previous = self._diagnostics.get(layer, {})
        self._diagnostics[layer] = {
            "head_count": int(weights.shape[1]),
            "blocked_edge_count": self._applied_counts[layer],
            "max_blocked_weight": max(max_blocked, float(previous.get("max_blocked_weight", 0.0))),
            "max_row_sum_error": max(max_row_error, float(previous.get("max_row_sum_error", 0.0))),
            "finite": finite and bool(previous.get("finite", True)),
        }
        return output

    def __enter__(self) -> "AttentionBlockContext":
        if not self.edges.pairs:
            return self
        for layer in self.layer_indices:
            attention = getattr(self.language_layers[layer], "self_attn", None)
            if attention is None:
                raise AttributeError(f"Decoder layer {layer} has no self_attn")
            self._handles.append(attention.register_forward_pre_hook(
                lambda module, args, kwargs, index=layer: self._pre_hook(index, module, args, kwargs),
                with_kwargs=True,
            ))
            self._handles.append(attention.register_forward_hook(
                lambda module, args, output, index=layer: self._post_hook(index, module, args, output)
            ))
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def diagnostics(self) -> dict[str, Any]:
        if not self.edges.pairs:
            return {"layers": [], "by_layer": {}, "empty": True}
        bad = {layer: {"applied": self._applied_counts[layer], "requested": len(self.edges.pairs)}
               for layer in self.layer_indices if self._applied_counts[layer] != len(self.edges.pairs)}
        if bad:
            raise RuntimeError(f"Not every requested attention edge was applied: {bad}")
        missing = sorted(set(self.layer_indices) - set(self._diagnostics)) if self.validate_weights else []
        if missing:
            raise RuntimeError(f"Missing attention diagnostics for layers: {missing}")
        return {"layers": list(self.layer_indices), "by_layer": {str(k): v for k, v in self._diagnostics.items()}}
