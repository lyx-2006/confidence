from __future__ import annotations

from typing import Any, Mapping

import torch

from layer_metacognition.model_adapter import LanguageModules


def tensor_and_tail(output: Any) -> tuple[torch.Tensor, tuple[Any, ...] | None, type | None]:
    if isinstance(output, torch.Tensor): return output, None, None
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0], tuple(output[1:]), type(output)
    raise TypeError(f"Unsupported decoder output: {type(output)!r}")


class MediationHook:
    """Single-prefill steering, downstream replacement, and exact hidden capture."""

    def __init__(self, modules: LanguageModules, *, prefill_sequence_length: int,
                 upstream_position: int, downstream_position: int,
                 steering_layer: int | None = None, steering_vector: torch.Tensor | None = None,
                 patch_layer: int | None = None, patch_source: torch.Tensor | None = None,
                 capture_targets: Mapping[str, tuple[int, int]] | None = None) -> None:
        self.modules = modules; self.prefill_sequence_length = int(prefill_sequence_length)
        self.upstream_position = int(upstream_position); self.downstream_position = int(downstream_position)
        if not 0 <= self.upstream_position < self.downstream_position < self.prefill_sequence_length:
            raise ValueError("Upstream/downstream positions violate causal order")
        self.steering_layer = steering_layer; self.steering_vector = None if steering_vector is None else steering_vector.detach().reshape(-1)
        self.patch_layer = patch_layer; self.patch_source = None if patch_source is None else patch_source.detach().reshape(-1)
        if (steering_layer is None) != (steering_vector is None): raise ValueError("Steering layer/vector must be paired")
        if (patch_layer is None) != (patch_source is None): raise ValueError("Patch layer/source must be paired")
        if steering_layer is not None and patch_layer is not None and int(patch_layer) <= int(steering_layer):
            raise ValueError("Patch layer must be downstream of steering layer")
        for value in (self.steering_vector, self.patch_source):
            if value is not None and value.numel() != modules.hidden_size: raise ValueError("Hidden-size mismatch")
        self.capture_targets = dict(capture_targets or {})
        layers = {layer for layer, _ in self.capture_targets.values()}
        if steering_layer is not None: layers.add(int(steering_layer))
        if patch_layer is not None: layers.add(int(patch_layer))
        if not layers or any(x < 0 or x >= modules.num_hidden_layers for x in layers): raise ValueError("Invalid hook layers")
        self.layers = tuple(sorted(layers)); self.hook_calls = {x: 0 for x in self.layers}
        self.prefill_hits = {x: 0 for x in self.layers}; self.captured: dict[str, torch.Tensor] = {}
        self.injection_count = 0; self.patch_count = 0; self.non_target_unchanged = True
        self.replacement_equal: bool | None = None; self.activation_dtype: str | None = None; self._handles = []

    @staticmethod
    def _outside_equal(before: torch.Tensor, after: torch.Tensor, positions: set[int]) -> bool:
        keep = [i for i in range(before.shape[1]) if i not in positions]
        return torch.equal(before[:, keep], after[:, keep])

    def _hook(self, layer: int, output: Any) -> Any:
        self.hook_calls[layer] += 1
        tensor, tail, container = tensor_and_tail(output)
        if int(tensor.shape[1]) != self.prefill_sequence_length: return output
        if self.prefill_hits[layer]: raise RuntimeError(f"Repeated full prefill at L{layer}")
        if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[2] != self.modules.hidden_size:
            raise ValueError(f"Bad hidden shape: {tuple(tensor.shape)}")
        self.prefill_hits[layer] += 1; self.activation_dtype = str(tensor.dtype).replace("torch.", "")
        patched = tensor; modified: set[int] = set()
        if layer == self.steering_layer:
            patched = tensor.clone(); patched[0, self.upstream_position] += self.steering_vector.to(tensor.device, tensor.dtype)
            modified.add(self.upstream_position); self.injection_count += 1
        if layer == self.patch_layer:
            if patched is tensor: patched = tensor.clone()
            source = self.patch_source.to(tensor.device, tensor.dtype)
            patched[0, self.downstream_position] = source; modified.add(self.downstream_position); self.patch_count += 1
            self.replacement_equal = torch.equal(patched[0, self.downstream_position].detach().cpu(), source.detach().cpu())
        if modified and not self._outside_equal(tensor, patched, modified):
            self.non_target_unchanged = False; raise RuntimeError("Hook modified non-target positions")
        for name, (capture_layer, position) in self.capture_targets.items():
            if layer == capture_layer: self.captured[name] = patched[0, position].detach().cpu().clone()
        if patched is tensor: return output
        if tail is None: return patched
        values = (patched, *tail)
        return list(values) if container is list else values

    def __enter__(self):
        if self._handles: raise RuntimeError("Hook cannot be entered twice")
        for layer in self.layers:
            self._handles.append(self.modules.language_layers[layer].register_forward_hook(
                lambda _m, _a, output, index=layer: self._hook(index, output)))
        return self

    def __exit__(self, *_args):
        for handle in self._handles: handle.remove()
        self._handles.clear()

    def validate(self) -> None:
        if any(v != 1 for v in self.hook_calls.values()) or any(v != 1 for v in self.prefill_hits.values()):
            raise RuntimeError(f"Hook count gate failed: {self.hook_calls}/{self.prefill_hits}")
        if set(self.captured) != set(self.capture_targets): raise RuntimeError("Hidden capture incomplete")
        if self.injection_count != int(self.steering_vector is not None): raise RuntimeError("Steering count failed")
        if self.patch_count != int(self.patch_source is not None): raise RuntimeError("Patch count failed")
        if self.patch_source is not None and not self.replacement_equal: raise RuntimeError("Replacement equality failed")

    def diagnostics(self) -> dict[str, Any]:
        self.validate()
        return {"layers": list(self.layers), "hook_calls": self.hook_calls, "prefill_hits": self.prefill_hits,
                "activation_dtype": self.activation_dtype, "injection_count": self.injection_count,
                "patch_count": self.patch_count, "replacement_equal": self.replacement_equal,
                "non_target_unchanged": self.non_target_unchanged}
