from __future__ import annotations

from typing import Any

import numpy as np
import torch

from layer_metacognition.model_adapter import LanguageModules

from .utils import cosine_distance


class SwapInvariantError(RuntimeError):
    pass


def _tensor_output(output: Any) -> tuple[torch.Tensor, tuple[Any, ...] | None]:
    if isinstance(output, torch.Tensor):
        return output, None
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0], output[1:]
    raise SwapInvariantError("decoder block returned an unsupported output")


class SwapActivationHook:
    """Replace one token at one post-MLP decoder block output."""

    site = "decoder_block_output_post_mlp_residual"

    def __init__(self, modules: LanguageModules, *, layer: int, position: int,
                 source_hidden: torch.Tensor, prefill_sequence_length: int) -> None:
        self.modules = modules
        self.layer = int(layer)
        self.position = int(position)
        self.prefill_sequence_length = int(prefill_sequence_length)
        self.source_hidden = source_hidden.detach().reshape(-1).cpu().float()
        if self.layer < 0 or self.layer >= modules.num_hidden_layers:
            raise SwapInvariantError("swap layer is outside model")
        if self.source_hidden.numel() != modules.hidden_size or not bool(torch.isfinite(self.source_hidden).all()):
            raise SwapInvariantError("source hidden has invalid shape or values")
        self.hook_count = 0
        self.applied_count = 0
        self.before: torch.Tensor | None = None
        self.after: torch.Tensor | None = None
        self.source_on_device: torch.Tensor | None = None
        self._handle: Any | None = None

    def _hook(self, _module: Any, _args: Any, output: Any) -> Any:
        self.hook_count += 1
        tensor, trailing = _tensor_output(output)
        if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[2] != self.modules.hidden_size:
            raise SwapInvariantError(f"invalid decoder output shape {tuple(tensor.shape)}")
        if self.applied_count or tensor.shape[1] != self.prefill_sequence_length:
            return output
        if not 0 <= self.position < tensor.shape[1]:
            raise SwapInvariantError("swap position outside decoder output")
        source = self.source_hidden.to(device=tensor.device, dtype=tensor.dtype)
        before = tensor[0, self.position].detach().clone()
        patched = tensor.clone()
        patched[0, self.position] = source
        after = patched[0, self.position].detach().clone()
        if not torch.equal(after, source):
            raise SwapInvariantError("target activation does not equal donor source after replacement")
        mask = torch.ones(tensor.shape[1], dtype=torch.bool, device=tensor.device)
        mask[self.position] = False
        if not torch.equal(tensor[:, mask, :], patched[:, mask, :]):
            raise SwapInvariantError("swap modified a non-target position")
        self.before = before.float().cpu()
        self.after = after.float().cpu()
        self.source_on_device = source.float().cpu()
        self.applied_count += 1
        return patched if trailing is None else (patched, *trailing)

    def __enter__(self) -> "SwapActivationHook":
        if self._handle is not None:
            raise RuntimeError("swap hook entered twice")
        self._handle = self.modules.language_layers[self.layer].register_forward_hook(self._hook)
        return self

    def __exit__(self, *_args: Any) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def diagnostics(self) -> dict[str, Any]:
        if self.applied_count != 1 or self.before is None or self.after is None or self.source_on_device is None:
            raise SwapInvariantError(f"swap hook applied {self.applied_count} times; expected once")
        before = self.before.numpy()
        source = self.source_on_device.numpy()
        before_norm = float(np.linalg.norm(before))
        source_norm = float(np.linalg.norm(source))
        if before_norm <= 0 or source_norm <= 0 or not np.isfinite(before_norm * source_norm):
            raise SwapInvariantError("activation norm is zero or non-finite")
        return {
            "site": self.site, "layer": self.layer, "position": self.position,
            "hook_count": self.hook_count, "applied_count": self.applied_count,
            "recipient_norm": before_norm, "donor_norm": source_norm,
            "norm_ratio": float(source_norm / before_norm),
            "abs_log_norm_ratio": float(abs(np.log(source_norm / before_norm))),
            "cosine_distance": cosine_distance(before, source),
            "target_exact_after_cast": bool(torch.equal(self.after, self.source_on_device)),
        }


class EmptyHook:
    """Observe a block without changing its output, for bitwise no-op checks."""

    def __init__(self, modules: LanguageModules, *, layer: int, prefill_sequence_length: int) -> None:
        self.module = modules.language_layers[int(layer)]
        self.prefill_sequence_length = int(prefill_sequence_length)
        self.hook_count = 0
        self.prefill_count = 0
        self._handle: Any | None = None

    def _hook(self, _module: Any, _args: Any, output: Any) -> Any:
        self.hook_count += 1
        tensor, _trailing = _tensor_output(output)
        if int(tensor.shape[1]) == self.prefill_sequence_length:
            self.prefill_count += 1
        return output

    def __enter__(self) -> "EmptyHook":
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *_args: Any) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def validate(self) -> None:
        if self.prefill_count != 1:
            raise SwapInvariantError(f"empty hook saw {self.prefill_count} prefills; expected one")


__all__ = ["EmptyHook", "SwapActivationHook", "SwapInvariantError"]
