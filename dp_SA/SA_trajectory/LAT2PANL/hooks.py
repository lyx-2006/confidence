from __future__ import annotations

from typing import Any, Sequence

import torch

from layer_metacognition.model_adapter import LanguageModules


def _tensor_and_trailing(output: Any) -> tuple[torch.Tensor, tuple[Any, ...] | None]:
    if isinstance(output, torch.Tensor): return output, None
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor): return output[0], output[1:]
    raise TypeError(f"Unsupported decoder output: {type(output)!r}")


class LATPANLMediationHook:
    """One-forward hook for exact LAT addition, PANL replacement and trajectory capture.

    L14 steering and L14 PANL replacement are deliberately performed in the same
    callback. This makes the within-layer causal order explicit and testable.
    """

    def __init__(self, modules: LanguageModules, *, prefill_sequence_length: int,
                 lat_position: int, panl_position: int, cle_position: int,
                 steering_vector: torch.Tensor | None = None,
                 patch_layer: int | None = None, patch_source: torch.Tensor | None = None,
                 capture_panl_layers: Sequence[int] = ()) -> None:
        self.modules = modules; self.prefill_sequence_length = int(prefill_sequence_length)
        self.lat_position = int(lat_position); self.panl_position = int(panl_position); self.cle_position = int(cle_position)
        if not (0 <= self.lat_position < self.panl_position < self.cle_position < self.prefill_sequence_length):
            raise ValueError("Invalid LAT/PANL/CLE causal positions")
        self.steering_vector = None if steering_vector is None else steering_vector.detach().reshape(-1)
        if self.steering_vector is not None and self.steering_vector.numel() != modules.hidden_size:
            raise ValueError("Steering vector hidden-size mismatch")
        self.patch_layer = None if patch_layer is None else int(patch_layer)
        self.patch_source = None if patch_source is None else patch_source.detach().reshape(-1)
        if (self.patch_layer is None) != (self.patch_source is None):
            raise ValueError("patch_layer and patch_source must be specified together")
        if self.patch_source is not None and self.patch_source.numel() != modules.hidden_size:
            raise ValueError("Patch source hidden-size mismatch")
        self.capture_panl_layers = tuple(sorted(set(int(value) for value in capture_panl_layers)))
        layers = set(self.capture_panl_layers) | {20}
        if self.steering_vector is not None: layers.add(14)
        if self.patch_layer is not None: layers.add(self.patch_layer)
        if any(value < 0 or value >= modules.num_hidden_layers for value in layers): raise ValueError("Hook layer outside model")
        self.layers = tuple(sorted(layers)); self.hook_calls = {value: 0 for value in self.layers}
        self.prefill_hits = {value: 0 for value in self.layers}; self.captured_panl: dict[int, torch.Tensor] = {}
        self.cle_hidden: torch.Tensor | None = None; self.lat_before: torch.Tensor | None = None; self.lat_after: torch.Tensor | None = None
        self.patch_before: torch.Tensor | None = None; self.patch_after: torch.Tensor | None = None
        self.non_target_unchanged = True; self.replacement_bitwise_equal: bool | None = None; self.activation_dtype: str | None = None
        self._handles: list[Any] = []

    @staticmethod
    def _outside_equal(before: torch.Tensor, after: torch.Tensor, positions: set[int]) -> bool:
        ordered = sorted(positions); cursor = 0
        for position in ordered:
            if not torch.equal(before[:, cursor:position, :], after[:, cursor:position, :]): return False
            cursor = position + 1
        return torch.equal(before[:, cursor:, :], after[:, cursor:, :])

    def _hook(self, layer: int, output: Any) -> Any:
        self.hook_calls[layer] += 1
        tensor, trailing = _tensor_and_trailing(output)
        if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[2] != self.modules.hidden_size:
            raise ValueError(f"Unexpected hidden shape at L{layer}: {tuple(tensor.shape)}")
        if int(tensor.shape[1]) != self.prefill_sequence_length or self.prefill_hits[layer]: return output
        if tensor.dtype != torch.bfloat16: raise TypeError(f"Runtime hidden must be bfloat16, got {tensor.dtype}")
        self.activation_dtype = "bfloat16"; self.prefill_hits[layer] += 1
        modified_positions: set[int] = set(); patched = tensor
        if layer == 14 and self.steering_vector is not None:
            patched = tensor.clone(); modified_positions.add(self.lat_position)
            self.lat_before = tensor[0, self.lat_position].detach().cpu().clone()
            vector = self.steering_vector.to(device=tensor.device, dtype=torch.bfloat16)
            patched[0, self.lat_position] = patched[0, self.lat_position] + vector
            self.lat_after = patched[0, self.lat_position].detach().cpu().clone()
        if layer == self.patch_layer:
            if patched is tensor: patched = tensor.clone()
            modified_positions.add(self.panl_position)
            assert self.patch_source is not None
            source = self.patch_source.to(device=tensor.device)
            if source.dtype != torch.bfloat16: raise TypeError("Patch source must remain bfloat16")
            self.patch_before = tensor[0, self.panl_position].detach().cpu().clone()
            patched[0, self.panl_position] = source
            self.patch_after = patched[0, self.panl_position].detach().cpu().clone()
            self.replacement_bitwise_equal = torch.equal(self.patch_after.view(torch.uint16), self.patch_source.cpu().view(torch.uint16))
            if not self.replacement_bitwise_equal: raise RuntimeError("PANL replacement is not bitwise identical")
        if modified_positions:
            unchanged = self._outside_equal(tensor, patched, modified_positions)
            self.non_target_unchanged = self.non_target_unchanged and unchanged
            if not unchanged: raise RuntimeError("A non-target token changed inside intervention hook")
        visible = patched
        if layer in self.capture_panl_layers:
            self.captured_panl[layer] = visible[0, self.panl_position].detach().cpu().clone()
        if layer == 20: self.cle_hidden = visible[0, self.cle_position].detach().cpu().clone()
        if patched is tensor: return output
        return patched if trailing is None else (patched, *trailing)

    def __enter__(self) -> "LATPANLMediationHook":
        if self._handles: raise RuntimeError("Hook cannot be entered twice")
        for layer in self.layers:
            self._handles.append(self.modules.language_layers[layer].register_forward_hook(
                lambda _module, _args, output, index=layer: self._hook(index, output)))
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        for handle in self._handles: handle.remove()
        self._handles.clear()

    def validate(self) -> None:
        if any(value != 1 for value in self.prefill_hits.values()) or any(value != 1 for value in self.hook_calls.values()):
            raise RuntimeError(f"Hook hit gate failed: calls={self.hook_calls}, prefill={self.prefill_hits}")
        if set(self.captured_panl) != set(self.capture_panl_layers): raise RuntimeError("PANL capture incomplete")
        if self.cle_hidden is None: raise RuntimeError("CLE L20 capture missing")
        if self.steering_vector is not None and (self.lat_before is None or self.lat_after is None): raise RuntimeError("LAT injection missing")
        if self.patch_layer is not None and not self.replacement_bitwise_equal: raise RuntimeError("PANL bitwise replacement gate failed")
        if not self.non_target_unchanged: raise RuntimeError("Non-target token gate failed")

    def diagnostics(self) -> dict[str, Any]:
        self.validate()
        return {"hook_calls": self.hook_calls, "prefill_hits": self.prefill_hits,
                "activation_dtype": self.activation_dtype, "lat_injection_count": int(self.steering_vector is not None),
                "panl_patch_count": int(self.patch_layer is not None), "patch_layer": self.patch_layer,
                "non_target_unchanged": self.non_target_unchanged,
                "replacement_bitwise_equal": self.replacement_bitwise_equal}
