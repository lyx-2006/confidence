from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch


class ReplacementInvariantError(RuntimeError):
    pass


@dataclass(frozen=True)
class EmbeddingReplacement:
    name: str
    positions: tuple[int, ...]
    source: torch.Tensor


def resolve_language_model(model: torch.nn.Module) -> torch.nn.Module:
    for path in ("model.language_model", "language_model", "model.model", "model"):
        current: Any = model
        for component in path.split("."):
            current = getattr(current, component, None)
            if current is None:
                break
        if isinstance(current, torch.nn.Module) and hasattr(current, "layers"):
            return current
    raise ReplacementInvariantError("Could not resolve language model inputs_embeds module")


class EmbeddingReplacementHook:
    """Replace image/text evidence in one language-model pre-hook."""

    def __init__(self, language_model: torch.nn.Module, *, replacements: Sequence[EmbeddingReplacement],
                 prefill_sequence_length: int, hidden_size: int, interpolation: float = 1.0) -> None:
        if not replacements:
            raise ValueError("At least one replacement is required")
        if interpolation not in (0.0, 1.0):
            raise ValueError("Only smoke lambda=0 and formal lambda=1 are supported")
        self.language_model = language_model
        self.replacements = tuple(replacements)
        self.prefill_sequence_length = int(prefill_sequence_length)
        self.hidden_size = int(hidden_size)
        self.interpolation = float(interpolation)
        occupied: set[int] = set()
        names: set[str] = set()
        for replacement in self.replacements:
            positions = tuple(map(int, replacement.positions))
            if not replacement.name or replacement.name in names or not positions or len(set(positions)) != len(positions):
                raise ValueError("Replacement names and positions must be non-empty and unique")
            if min(positions) < 0 or max(positions) >= self.prefill_sequence_length:
                raise ReplacementInvariantError(f"{replacement.name}: positions outside prefill")
            if occupied & set(positions):
                raise ReplacementInvariantError("Replacement spans overlap")
            if tuple(replacement.source.shape) != (len(positions), self.hidden_size):
                raise ReplacementInvariantError(f"{replacement.name}: source shape mismatch")
            if not bool(torch.isfinite(replacement.source).all()):
                raise ReplacementInvariantError(f"{replacement.name}: non-finite source")
            names.add(replacement.name)
            occupied.update(positions)
        self.hook_count = 0
        self.applied_count = 0
        self.replacement_l2: dict[str, float] = {}
        self.before_norm: dict[str, float] = {}
        self.source_norm: dict[str, float] = {}
        self.non_target_equal: bool | None = None
        self._handle: Any | None = None

    def _pre_hook(self, _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
        self.hook_count += 1
        hidden = kwargs.get("inputs_embeds")
        if not isinstance(hidden, torch.Tensor):
            raise ReplacementInvariantError("Language-model hook did not receive inputs_embeds")
        if hidden.ndim != 3 or int(hidden.shape[0]) != 1 or int(hidden.shape[2]) != self.hidden_size:
            raise ReplacementInvariantError(f"Unexpected inputs_embeds shape: {tuple(hidden.shape)}")
        if self.applied_count or int(hidden.shape[1]) != self.prefill_sequence_length:
            return args, kwargs
        patched = hidden.clone()
        target: set[int] = set()
        for replacement in self.replacements:
            positions = list(replacement.positions)
            target.update(positions)
            source = replacement.source.to(device=hidden.device, dtype=hidden.dtype)
            before = hidden[0, positions]
            value = before + self.interpolation * (source - before)
            patched[0, positions] = value
            self.before_norm[replacement.name] = float(torch.linalg.vector_norm(before.float()).item())
            self.source_norm[replacement.name] = float(torch.linalg.vector_norm(source.float()).item())
            self.replacement_l2[replacement.name] = float(torch.linalg.vector_norm(value.float() - before.float()).item())
        mask = torch.ones(int(hidden.shape[1]), dtype=torch.bool, device=hidden.device)
        mask[list(sorted(target))] = False
        self.non_target_equal = bool(torch.equal(hidden[:, mask], patched[:, mask]))
        if not self.non_target_equal:
            raise ReplacementInvariantError("Replacement modified a non-target embedding")
        updated = dict(kwargs)
        updated["inputs_embeds"] = patched
        self.applied_count += 1
        return args, updated

    def __enter__(self):
        self._handle = self.language_model.register_forward_pre_hook(self._pre_hook, with_kwargs=True)
        return self

    def __exit__(self, *_args: Any) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def diagnostics(self) -> dict[str, Any]:
        if self.applied_count != 1 or self.hook_count < 1 or self.non_target_equal is not True:
            raise ReplacementInvariantError(
                f"Embedding hook invariant failed: hooks={self.hook_count}, applied={self.applied_count}"
            )
        if self.interpolation == 1.0 and any(value <= 0 for value in self.replacement_l2.values()):
            raise ReplacementInvariantError("Formal replacement was identical to clean embedding")
        if self.interpolation == 0.0 and any(value != 0 for value in self.replacement_l2.values()):
            raise ReplacementInvariantError("Lambda=0 replacement was not exactly zero")
        return {
            "hook_count": self.hook_count,
            "applied_count": self.applied_count,
            "interpolation": self.interpolation,
            "replacement_count": len(self.replacements),
            "replacement_l2": dict(self.replacement_l2),
            "before_norm": dict(self.before_norm),
            "source_norm": dict(self.source_norm),
            "positions": {value.name: list(value.positions) for value in self.replacements},
            "shapes": {value.name: list(value.source.shape) for value in self.replacements},
            "non_target_equal": self.non_target_equal,
        }


__all__ = ["EmbeddingReplacement", "EmbeddingReplacementHook", "ReplacementInvariantError", "resolve_language_model"]
