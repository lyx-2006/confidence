from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from layer_metacognition.model_adapter import LanguageModules
from layer_metacognition.sa_patching.sa_patching_hook import (
    ActivationReplacementHook as _JointActivationReplacementHook,
    EmbeddingReplacement,
    EmbeddingReplacementHook as _JointEmbeddingReplacementHook,
    PatchingInvariantError,
    ResidualActivationCacheHook,
    resolve_language_model,
)


class EmbeddingReplacementHook(_JointEmbeddingReplacementHook):
    """Joint hook with complete per-span norm/shape diagnostics."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.before_norm: dict[str, float] = {}
        self.source_norm: dict[str, float] = {}

    def _pre_hook(self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
        hidden = kwargs.get("inputs_embeds")
        should_apply = isinstance(hidden, torch.Tensor) and not self.applied_count and int(hidden.shape[1]) == self.prefill_sequence_length
        if should_apply:
            for replacement in self.replacements:
                positions = list(replacement.positions)
                self.before_norm[replacement.name] = float(torch.linalg.vector_norm(hidden[0, positions].float()).item())
                self.source_norm[replacement.name] = float(torch.linalg.vector_norm(replacement.source.float()).item())
        return super()._pre_hook(module, args, kwargs)

    def diagnostics(self) -> dict[str, Any]:
        value = super().diagnostics()
        value["before_norm"] = dict(self.before_norm)
        value["source_norm"] = dict(self.source_norm)
        value["shapes"] = {replacement.name: list(replacement.source.shape) for replacement in self.replacements}
        value["positions"] = {replacement.name: list(replacement.positions) for replacement in self.replacements}
        return value


class ActivationReplacementHook(_JointActivationReplacementHook):
    """Joint single-token patch with an explicit non-target equality check."""

    def _hook(self, module: Any, args: Any, output: Any) -> Any:
        if isinstance(output, torch.Tensor):
            before = output
        elif isinstance(output, tuple) and output:
            before = output[0]
        else:
            before = None
        result = super()._hook(module, args, output)
        if self.applied_count == 1 and isinstance(before, torch.Tensor) and int(before.shape[1]) == self.prefill_sequence_length:
            after = result if isinstance(result, torch.Tensor) else result[0]
            mask = torch.ones(before.shape[1], dtype=torch.bool, device=before.device)
            mask[self.position] = False
            if not torch.equal(before[:, mask, :], after[:, mask, :]):
                raise PatchingInvariantError("Patch modified a non-target token")
        return result


class EmptyActivationHook:
    """Observe a requested decoder layer without changing its output."""

    def __init__(self, modules: LanguageModules, *, layer: int, prefill_sequence_length: int) -> None:
        self.module = modules.language_layers[int(layer)]
        self.prefill_sequence_length = int(prefill_sequence_length)
        self.hook_count = 0
        self.prefill_count = 0
        self._handle: Any | None = None

    def _hook(self, _module: Any, _args: Any, output: Any) -> Any:
        self.hook_count += 1
        tensor = output if isinstance(output, torch.Tensor) else output[0]
        if int(tensor.shape[1]) == self.prefill_sequence_length:
            self.prefill_count += 1
        return output

    def __enter__(self):
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *_args: Any) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def validate(self) -> None:
        if self.prefill_count != 1:
            raise PatchingInvariantError(f"Empty hook observed {self.prefill_count} prefills; expected 1")


__all__ = [
    "ActivationReplacementHook", "EmbeddingReplacement", "EmbeddingReplacementHook",
    "EmptyActivationHook", "PatchingInvariantError", "ResidualActivationCacheHook",
    "resolve_language_model",
]
