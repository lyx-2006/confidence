from __future__ import annotations

from typing import Any, Sequence

import torch

from layer_metacognition.model_adapter import LanguageModules


def decoder_output_tensor(output: Any) -> torch.Tensor:
    """Normalize Qwen3 Tensor and Qwen2.5 tuple decoder return formats."""

    tensor = output[0] if isinstance(output, (tuple, list)) and output else output
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
        raise TypeError(
            "Decoder block output must be a rank-3 Tensor or a tuple/list whose "
            f"first item is one; got {type(output)!r} shape={getattr(tensor, 'shape', None)}"
        )
    return tensor


class SelectedHiddenCapture:
    """Capture selected token states from selected full-prefill block outputs."""

    def __init__(
        self,
        modules: LanguageModules,
        *,
        positions: dict[str, int],
        layers: Sequence[int],
        prefill_sequence_length: int,
    ) -> None:
        if not positions:
            raise ValueError("At least one position is required")
        self.modules = modules
        self.positions = {str(name): int(index) for name, index in positions.items()}
        self.layers = tuple(int(layer) for layer in layers)
        self.prefill_sequence_length = int(prefill_sequence_length)
        if not self.layers or len(self.layers) != len(set(self.layers)):
            raise ValueError("Capture layers must be non-empty and unique")
        if any(layer < 0 or layer >= modules.num_hidden_layers for layer in self.layers):
            raise ValueError("Capture layer is outside the language model")
        if self.prefill_sequence_length < 1 or any(
            index < 0 or index >= self.prefill_sequence_length
            for index in self.positions.values()
        ):
            raise ValueError("Capture position is outside the full prefill")
        self.values: dict[str, dict[int, torch.Tensor]] = {
            name: {} for name in self.positions
        }
        self.call_counts = {layer: 0 for layer in self.layers}
        self.capture_counts = {layer: 0 for layer in self.layers}
        self._handles: list[Any] = []

    def _hook(self, layer: int, output: Any) -> None:
        self.call_counts[layer] += 1
        tensor = decoder_output_tensor(output)
        if int(tensor.shape[1]) != self.prefill_sequence_length:
            return
        if self.capture_counts[layer] > 0:
            raise RuntimeError(f"Layer {layer} produced the full prefill more than once")
        if int(tensor.shape[0]) < 1 or int(tensor.shape[2]) != self.modules.hidden_size:
            raise ValueError(f"Invalid decoder output at layer {layer}: {tuple(tensor.shape)}")
        for name, position in self.positions.items():
            self.values[name][layer] = tensor[0, position, :].detach().clone()
        self.capture_counts[layer] += 1

    def __enter__(self) -> "SelectedHiddenCapture":
        if self._handles:
            raise RuntimeError("SelectedHiddenCapture cannot be entered twice")
        for layer in self.layers:
            self._handles.append(
                self.modules.language_layers[layer].register_forward_hook(
                    lambda _module, _args, output, index=layer: self._hook(index, output)
                )
            )
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def validate(self) -> dict[str, dict[int, torch.Tensor]]:
        bad = {layer: count for layer, count in self.capture_counts.items() if count != 1}
        if bad:
            raise RuntimeError(f"Hidden capture did not run exactly once per layer: {bad}")
        expected = set(self.layers)
        missing = {
            name: sorted(expected - set(values))
            for name, values in self.values.items()
            if set(values) != expected
        }
        if missing:
            raise RuntimeError(f"Hidden capture is incomplete: {missing}")
        return self.values

    def diagnostics(self) -> dict[str, Any]:
        self.validate()
        return {
            "prefill_sequence_length": self.prefill_sequence_length,
            "layers": list(self.layers),
            "positions": dict(self.positions),
            "hook_call_counts": dict(self.call_counts),
            "capture_counts": dict(self.capture_counts),
        }
