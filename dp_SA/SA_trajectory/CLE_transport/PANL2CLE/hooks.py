from __future__ import annotations

from typing import Any, Mapping

import torch

from layer_metacognition.model_adapter import LanguageModules


def _tensor_output(output: Any) -> tuple[torch.Tensor, tuple[Any, ...] | None]:
    if isinstance(output, torch.Tensor): return output, None
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor): return output[0], output[1:]
    raise TypeError(f"Unsupported decoder output: {type(output)!r}")


class WindowCaptureHook:
    def __init__(self, modules: LanguageModules, *, windows: Mapping[str, list[int]], layers: tuple[int, ...], prefill_length: int):
        self.modules = modules; self.windows = {str(k): list(map(int, v)) for k, v in windows.items()}
        self.layers = tuple(map(int, layers)); self.prefill_length = int(prefill_length)
        self.values: dict[str, torch.Tensor] = {}; self.counts = {layer: 0 for layer in self.layers}; self.handles: list[Any] = []
        if any(len(v) != 8 or v != list(range(v[0], v[0] + 8)) for v in self.windows.values()): raise ValueError("Capture windows must be contiguous length 8")

    def _hook(self, layer: int, output: Any) -> None:
        tensor, _ = _tensor_output(output)
        if self.counts[layer] or int(tensor.shape[1]) != self.prefill_length: return
        if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[2] != self.modules.hidden_size: raise ValueError("Invalid donor block output")
        for name, positions in self.windows.items(): self.values[f"{name}__L{layer}"] = tensor[0, positions, :].detach().cpu().clone()
        self.counts[layer] += 1

    def __enter__(self) -> "WindowCaptureHook":
        for layer in self.layers:
            self.handles.append(self.modules.language_layers[layer].register_forward_hook(lambda _m, _a, out, index=layer: self._hook(index, out)))
        return self

    def __exit__(self, *_args: Any) -> None:
        for handle in self.handles: handle.remove()
        self.handles.clear()

    def validate(self) -> None:
        expected = {f"{name}__L{layer}" for name in self.windows for layer in self.layers}
        if self.counts != {layer: 1 for layer in self.layers} or set(self.values) != expected: raise RuntimeError("Donor capture incomplete")
        if any(value.dtype != torch.bfloat16 or tuple(value.shape) != (8, self.modules.hidden_size) for value in self.values.values()):
            raise RuntimeError("Donor capture dtype/shape mismatch")


class WindowSwapHook:
    def __init__(self, modules: LanguageModules, *, layer: int, positions: list[int], source: torch.Tensor, prefill_length: int):
        self.modules = modules; self.layer = int(layer); self.positions = list(map(int, positions)); self.source = source.detach().cpu()
        self.prefill_length = int(prefill_length); self.handle: Any | None = None; self.hook_count = 0; self.applied_count = 0
        self.target_exact = False; self.outside_exact = False; self.patch_l2: float | None = None
        if self.positions != list(range(self.positions[0], self.positions[0] + 8)): raise ValueError("Swap positions must be contiguous length 8")
        if self.source.dtype != torch.bfloat16 or tuple(self.source.shape) != (8, modules.hidden_size): raise ValueError("Swap source must be [8, hidden] BF16")

    def _hook(self, _module: Any, _args: Any, output: Any) -> Any:
        self.hook_count += 1; tensor, trailing = _tensor_output(output)
        if self.applied_count or int(tensor.shape[1]) != self.prefill_length: return output
        if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[2] != self.modules.hidden_size: raise ValueError("Invalid recipient block output")
        source = self.source.to(tensor.device)
        if source.dtype != tensor.dtype: raise TypeError(f"Lossless swap dtype mismatch: {source.dtype} != {tensor.dtype}")
        before = tensor[0, self.positions, :].detach(); patched = tensor.clone(); patched[0, self.positions, :] = source
        self.target_exact = bool(torch.equal(patched[0, self.positions, :], source))
        mask = torch.ones(tensor.shape[1], dtype=torch.bool, device=tensor.device); mask[self.positions] = False
        self.outside_exact = bool(torch.equal(patched[:, mask, :], tensor[:, mask, :]))
        self.patch_l2 = float(torch.linalg.vector_norm(source.float() - before.float()).item()); self.applied_count += 1
        if not self.target_exact or not self.outside_exact: raise RuntimeError("Window-only replacement invariant failed")
        return patched if trailing is None else (patched, *trailing)

    def __enter__(self) -> "WindowSwapHook":
        self.handle = self.modules.language_layers[self.layer].register_forward_hook(self._hook); return self

    def __exit__(self, *_args: Any) -> None:
        if self.handle is not None: self.handle.remove(); self.handle = None

    def diagnostics(self) -> dict[str, Any]:
        if self.applied_count != 1: raise RuntimeError(f"Swap applied {self.applied_count} times")
        return {"layer": self.layer, "positions": self.positions, "hook_count": self.hook_count, "applied_count": self.applied_count,
                "target_exact": self.target_exact, "outside_exact": self.outside_exact, "patch_l2": self.patch_l2,
                "site": "decoder_block_output_post_mlp_residual"}


class SingleHiddenCapture:
    def __init__(self, modules: LanguageModules, *, layer: int, position: int, prefill_length: int):
        self.module = modules.language_layers[int(layer)]; self.layer = int(layer); self.position = int(position); self.prefill_length = int(prefill_length)
        self.value: torch.Tensor | None = None; self.count = 0; self.handle: Any | None = None
    def _hook(self, _m: Any, _a: Any, output: Any) -> None:
        tensor, _ = _tensor_output(output)
        if self.value is None and int(tensor.shape[1]) == self.prefill_length:
            self.value = tensor[0, self.position].detach().float().cpu(); self.count += 1
    def __enter__(self) -> "SingleHiddenCapture": self.handle = self.module.register_forward_hook(self._hook); return self
    def __exit__(self, *_args: Any) -> None:
        if self.handle is not None: self.handle.remove(); self.handle = None
    def validate(self) -> torch.Tensor:
        if self.count != 1 or self.value is None: raise RuntimeError("Downstream hidden capture failed")
        return self.value


class EmptyHook:
    def __init__(self, modules: LanguageModules, *, layer: int): self.module = modules.language_layers[int(layer)]; self.handle: Any | None = None; self.count = 0
    def _hook(self, _m: Any, _a: Any, output: Any) -> Any: self.count += 1; return output
    def __enter__(self) -> "EmptyHook": self.handle = self.module.register_forward_hook(self._hook); return self
    def __exit__(self, *_args: Any) -> None:
        if self.handle is not None: self.handle.remove(); self.handle = None

