"""Strict embedding replacement, residual caching, and activation patch hooks."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from layer_metacognition.model_adapter import LanguageModules


class PatchingInvariantError(RuntimeError):
    """A shape/span/device/hook invariant that must abort the whole run."""


def resolve_language_model(model: torch.nn.Module) -> torch.nn.Module:
    candidates = (
        "model.language_model",
        "language_model",
        "model.model",
        "model",
    )
    for path in candidates:
        current: Any = model
        for component in path.split("."):
            if not hasattr(current, component):
                current = None
                break
            current = getattr(current, component)
        if isinstance(current, torch.nn.Module) and hasattr(current, "layers"):
            return current
    raise PatchingInvariantError("Could not resolve Qwen language model input module")


@dataclass(frozen=True)
class EmbeddingReplacement:
    name: str
    positions: tuple[int, ...]
    source: torch.Tensor


class EmbeddingReplacementHook:
    """Replace selected final multimodal embeddings immediately before decoding."""

    def __init__(
        self,
        language_model: torch.nn.Module,
        *,
        replacements: Sequence[EmbeddingReplacement],
        prefill_sequence_length: int,
        hidden_size: int,
        capture_clean: bool = False,
    ) -> None:
        if prefill_sequence_length < 1 or hidden_size < 1:
            raise ValueError("Invalid prefill length or hidden size")
        self.language_model = language_model
        self.prefill_sequence_length = int(prefill_sequence_length)
        self.hidden_size = int(hidden_size)
        self.capture_clean = bool(capture_clean)
        self.replacements = tuple(replacements)
        occupied: set[int] = set()
        names: set[str] = set()
        for replacement in self.replacements:
            if not replacement.name or replacement.name in names:
                raise ValueError("Embedding replacement names must be non-empty and unique")
            names.add(replacement.name)
            positions = tuple(int(value) for value in replacement.positions)
            if not positions or len(positions) != len(set(positions)):
                raise ValueError(f"{replacement.name}: positions must be non-empty and unique")
            if min(positions) < 0 or max(positions) >= self.prefill_sequence_length:
                raise PatchingInvariantError(
                    f"{replacement.name}: token span is outside prefill length"
                )
            overlap = occupied.intersection(positions)
            if overlap:
                raise PatchingInvariantError(
                    f"Embedding replacement spans overlap at {sorted(overlap)}"
                )
            occupied.update(positions)
            source = replacement.source
            if source.ndim != 2 or tuple(source.shape) != (
                len(positions),
                self.hidden_size,
            ):
                raise PatchingInvariantError(
                    f"{replacement.name}: hidden shape mismatch: "
                    f"source={tuple(source.shape)} expected={(len(positions), self.hidden_size)}"
                )
            if not bool(torch.isfinite(source).all()):
                raise PatchingInvariantError(f"{replacement.name}: non-finite source embedding")
        self.hook_count = 0
        self.applied_count = 0
        self.clean_embeddings: torch.Tensor | None = None
        self.replacement_l2: dict[str, float] = {}
        self._handle: Any | None = None

    def _pre_hook(
        self,
        _module: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        self.hook_count += 1
        hidden = kwargs.get("inputs_embeds")
        if not isinstance(hidden, torch.Tensor):
            raise PatchingInvariantError(
                "Language-model pre-hook did not receive inputs_embeds"
            )
        if hidden.ndim != 3 or int(hidden.shape[0]) != 1:
            raise PatchingInvariantError(
                f"Embedding output must be [1, sequence, hidden], got {tuple(hidden.shape)}"
            )
        if int(hidden.shape[2]) != self.hidden_size:
            raise PatchingInvariantError(
                f"Embedding hidden shape mismatch: {tuple(hidden.shape)}"
            )
        if self.applied_count or int(hidden.shape[1]) != self.prefill_sequence_length:
            return args, kwargs
        if self.capture_clean:
            self.clean_embeddings = hidden[0].detach().float().cpu().clone()
        patched = hidden.clone()
        for replacement in self.replacements:
            positions = list(replacement.positions)
            source = replacement.source
            if source.device.type not in {"cpu", patched.device.type}:
                raise PatchingInvariantError(
                    f"{replacement.name}: device mismatch source={source.device} "
                    f"target={patched.device}"
                )
            source_on_device = source.to(device=patched.device, dtype=patched.dtype)
            before = hidden[0, positions, :]
            difference = torch.linalg.vector_norm(
                source_on_device.float() - before.float()
            ).item()
            if not difference > 0:
                raise PatchingInvariantError(
                    f"{replacement.name}: clean and corrupt embeddings are identical"
                )
            patched[0, positions, :] = source_on_device
            self.replacement_l2[replacement.name] = float(difference)
        updated = dict(kwargs)
        updated["inputs_embeds"] = patched
        self.applied_count += 1
        return args, updated

    def __enter__(self) -> "EmbeddingReplacementHook":
        if self._handle is not None:
            raise RuntimeError("EmbeddingReplacementHook cannot be entered twice")
        self._handle = self.language_model.register_forward_pre_hook(
            self._pre_hook,
            with_kwargs=True,
        )
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def validate_applied_once(self) -> None:
        if self.applied_count != 1:
            raise PatchingInvariantError(
                f"Embedding hook applied {self.applied_count} times; expected 1"
            )
        if self.capture_clean and self.clean_embeddings is None:
            raise PatchingInvariantError("Clean embedding capture is missing")
        if set(self.replacement_l2) != {
            replacement.name for replacement in self.replacements
        }:
            raise PatchingInvariantError("Embedding replacement diagnostics are incomplete")

    def diagnostics(self) -> dict[str, Any]:
        self.validate_applied_once()
        return {
            "hook_count": int(self.hook_count),
            "applied_count": int(self.applied_count),
            "prefill_sequence_length": self.prefill_sequence_length,
            "replacement_l2": dict(self.replacement_l2),
        }


class ResidualActivationCacheHook:
    """Capture requested decoder-block outputs during exactly one prefill."""

    def __init__(
        self,
        modules: LanguageModules,
        *,
        targets: Mapping[int, Mapping[str, int]],
        prefill_sequence_length: int,
    ) -> None:
        self.modules = modules
        self.targets = {
            int(layer): {str(name): int(position) for name, position in values.items()}
            for layer, values in targets.items()
        }
        self.prefill_sequence_length = int(prefill_sequence_length)
        self.cache: dict[int, dict[str, torch.Tensor]] = {
            layer: {} for layer in self.targets
        }
        self.capture_count = {layer: 0 for layer in self.targets}
        self._handles: list[Any] = []
        for layer, values in self.targets.items():
            if layer < 0 or layer >= modules.num_hidden_layers:
                raise PatchingInvariantError(f"Cache layer {layer} is outside the model")
            if any(
                position < 0 or position >= self.prefill_sequence_length
                for position in values.values()
            ):
                raise PatchingInvariantError(f"Cache position outside prefill at layer {layer}")

    def _hook(self, layer: int, output: Any) -> None:
        tensor = output[0] if isinstance(output, tuple) and output else output
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            raise PatchingInvariantError("Decoder block returned an invalid hidden tensor")
        if int(tensor.shape[2]) != self.modules.hidden_size:
            raise PatchingInvariantError("Decoder cache hidden shape mismatch")
        if self.capture_count[layer] or int(tensor.shape[1]) != self.prefill_sequence_length:
            return
        if int(tensor.shape[0]) != 1:
            raise PatchingInvariantError("Residual cache requires batch size 1")
        for name, position in self.targets[layer].items():
            self.cache[layer][name] = tensor[0, position].detach().float().cpu().clone()
        self.capture_count[layer] += 1

    def __enter__(self) -> "ResidualActivationCacheHook":
        for layer in self.targets:
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

    def validate(self) -> None:
        bad = {layer: count for layer, count in self.capture_count.items() if count != 1}
        if bad:
            raise PatchingInvariantError(f"Residual cache hooks did not capture once: {bad}")
        for layer, targets in self.targets.items():
            if set(self.cache[layer]) != set(targets):
                raise PatchingInvariantError(f"Residual cache is incomplete at layer {layer}")


class ActivationReplacementHook:
    """Replace one post-MLP decoder residual token on the initial prefill."""

    def __init__(
        self,
        modules: LanguageModules,
        *,
        layer: int,
        position: int,
        source_hidden: torch.Tensor,
        prefill_sequence_length: int,
    ) -> None:
        self.modules = modules
        self.layer = int(layer)
        self.position = int(position)
        self.prefill_sequence_length = int(prefill_sequence_length)
        self.source_hidden = source_hidden.detach().reshape(-1)
        if self.layer < 0 or self.layer >= modules.num_hidden_layers:
            raise PatchingInvariantError("Patch layer is outside the model")
        if self.position < 0 or self.position >= self.prefill_sequence_length:
            raise PatchingInvariantError("Patch token span mismatch")
        if int(self.source_hidden.numel()) != modules.hidden_size:
            raise PatchingInvariantError("Patch hidden shape mismatch")
        if not bool(torch.isfinite(self.source_hidden).all()):
            raise PatchingInvariantError("Patch source contains non-finite values")
        self.hook_count = 0
        self.applied_count = 0
        self.clean_hidden_norm = float(
            torch.linalg.vector_norm(self.source_hidden.float()).item()
        )
        self.corrupt_hidden_norm: float | None = None
        self.patch_l2: float | None = None
        self._handle: Any | None = None

    def _hook(self, _module: Any, _args: Any, output: Any) -> Any:
        self.hook_count += 1
        if isinstance(output, torch.Tensor):
            tensor = output
            trailing: tuple[Any, ...] | None = None
        elif isinstance(output, tuple) and output:
            tensor = output[0]
            trailing = output[1:]
        else:
            raise PatchingInvariantError("Unsupported decoder block output")
        if tensor.ndim != 3 or int(tensor.shape[2]) != self.modules.hidden_size:
            raise PatchingInvariantError("Patch decoder hidden shape mismatch")
        if self.applied_count or int(tensor.shape[1]) != self.prefill_sequence_length:
            return output
        if int(tensor.shape[0]) != 1:
            raise PatchingInvariantError("Activation patch requires batch size 1")
        source = self.source_hidden.to(device=tensor.device, dtype=tensor.dtype)
        before = tensor[0, self.position].detach()
        patched = tensor.clone()
        patched[0, self.position] = source
        self.corrupt_hidden_norm = float(torch.linalg.vector_norm(before.float()).item())
        self.patch_l2 = float(torch.linalg.vector_norm(source.float() - before.float()).item())
        self.applied_count += 1
        return patched if trailing is None else (patched, *trailing)

    def __enter__(self) -> "ActivationReplacementHook":
        self._handle = self.modules.language_layers[self.layer].register_forward_hook(
            self._hook
        )
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def diagnostics(self) -> dict[str, Any]:
        if self.applied_count != 1 or self.corrupt_hidden_norm is None:
            raise PatchingInvariantError(
                f"Patch hook applied {self.applied_count} times; expected 1"
            )
        return {
            "hook_count": int(self.hook_count),
            "applied_count": int(self.applied_count),
            "layer": self.layer,
            "position": self.position,
            "clean_hidden_norm": self.clean_hidden_norm,
            "corrupt_hidden_norm": self.corrupt_hidden_norm,
            "patch_l2": self.patch_l2,
            "site": "decoder_block_output_post_mlp_residual",
        }


class CombinedHooks:
    """Enter a fixed ordered collection of hook context managers."""

    def __init__(self, *hooks: Any) -> None:
        self.hooks = hooks
        self._stack: ExitStack | None = None

    def __enter__(self) -> "CombinedHooks":
        self._stack = ExitStack()
        for hook in self.hooks:
            self._stack.enter_context(hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        assert self._stack is not None
        self._stack.__exit__(exc_type, exc, traceback)
        self._stack = None
