from __future__ import annotations

from typing import Any

import numpy as np
import torch

from layer_metacognition.model_adapter import LanguageModules


class LATSwapError(RuntimeError):
    pass


def _tensor_output(output: Any) -> tuple[torch.Tensor, tuple[Any, ...] | None]:
    if isinstance(output, torch.Tensor):
        return output, None
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0], output[1:]
    raise LATSwapError("Unsupported decoder block output")


class LATSwapHook:
    site = "decoder_block_output_post_mlp_residual"

    def __init__(self, modules: LanguageModules, *, layer: int, recipient_position: int, donor_hidden: torch.Tensor, prefill_length: int) -> None:
        self.modules = modules; self.layer = int(layer); self.position = int(recipient_position); self.prefill_length = int(prefill_length)
        self.donor = donor_hidden.detach().reshape(-1).cpu().float()
        if self.layer not in range(modules.num_hidden_layers) or self.donor.numel() != modules.hidden_size or not torch.isfinite(self.donor).all():
            raise LATSwapError("Invalid layer or donor hidden")
        self.hook_count = 0; self.applied_count = 0; self.before = None; self.after = None; self.cast_donor = None; self.other_tokens_equal = False; self.dtype = None; self.device = None; self.shape = None; self._handle = None

    def _hook(self, _module: Any, _args: Any, output: Any) -> Any:
        self.hook_count += 1; tensor, trailing = _tensor_output(output)
        if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[2] != self.modules.hidden_size:
            raise LATSwapError(f"Invalid decoder shape: {tuple(tensor.shape)}")
        if self.applied_count or int(tensor.shape[1]) != self.prefill_length:
            return output
        if self.position not in range(int(tensor.shape[1])):
            raise LATSwapError("LAT position outside prefill")
        donor = self.donor.to(device=tensor.device, dtype=tensor.dtype)
        patched = tensor.clone(); before = tensor[0, self.position].detach().clone(); patched[0, self.position] = donor
        mask = torch.ones(tensor.shape[1], dtype=torch.bool, device=tensor.device); mask[self.position] = False
        self.other_tokens_equal = bool(torch.equal(tensor[:, mask, :], patched[:, mask, :]))
        if not self.other_tokens_equal or not torch.equal(patched[0, self.position], donor):
            raise LATSwapError("LAT-only replacement invariant failed")
        self.before = before.float().cpu(); self.after = patched[0, self.position].detach().float().cpu(); self.cast_donor = donor.detach().float().cpu()
        self.dtype = str(tensor.dtype); self.device = str(tensor.device); self.shape = list(tensor.shape); self.applied_count += 1
        return patched if trailing is None else (patched, *trailing)

    def __enter__(self) -> "LATSwapHook":
        self._handle = self.modules.language_layers[self.layer].register_forward_hook(self._hook); return self

    def __exit__(self, *_args: Any) -> None:
        if self._handle is not None: self._handle.remove(); self._handle = None

    def diagnostics(self) -> dict[str, Any]:
        if self.applied_count != 1 or self.before is None or self.after is None or self.cast_donor is None:
            raise LATSwapError(f"LAT hook applied {self.applied_count} times, expected once")
        before, donor = self.before.numpy(), self.cast_donor.numpy(); bn, dn = np.linalg.norm(before), np.linalg.norm(donor)
        if bn <= 0 or dn <= 0 or not np.isfinite([bn, dn]).all():
            raise LATSwapError("Invalid hidden norm")
        cosine = float(np.dot(before, donor) / (bn * dn))
        import hashlib
        digest = lambda tensor: hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
        return {
            "site": self.site, "layer": self.layer, "position": self.position, "hook_count": self.hook_count,
            "applied_count": self.applied_count, "shape": self.shape, "dtype": self.dtype, "device": self.device,
            "recipient_norm": float(bn), "donor_norm": float(dn), "cosine_similarity": cosine,
            "cosine_distance": float(1 - cosine), "l2_distance": float(np.linalg.norm(before - donor)),
            "target_exact_after_cast": bool(torch.equal(self.after, self.cast_donor)), "other_tokens_equal": self.other_tokens_equal,
            "recipient_sha256": digest(self.before), "donor_sha256": digest(self.cast_donor), "after_sha256": digest(self.after),
            "finite": bool(torch.isfinite(self.before).all() and torch.isfinite(self.after).all()),
        }
