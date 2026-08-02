"""Validated, bounded-cache loading of target-layer AC/PANL vectors."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import HIDDEN_STATE_DEFINITION, POSITION_NAMES


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError(f"Hidden-state shard payload is not an object: {path}")
    return value


class HiddenStateLoader:
    """Read one layer/position vector while caching only a few shards."""

    def __init__(self, experiment_dir: str | Path, cache_size: int = 2):
        if cache_size < 1:
            raise ValueError("cache_size must be positive")
        self.experiment_dir = Path(experiment_dir).resolve()
        self.index_path = self.experiment_dir / "hidden_states" / "index.json"
        if not self.index_path.is_file():
            raise FileNotFoundError(f"Hidden-state index does not exist: {self.index_path}")
        index = json.loads(self.index_path.read_text(encoding="utf-8"))
        cases = index.get("cases")
        if not isinstance(cases, dict):
            raise ValueError(f"Hidden-state index has no cases object: {self.index_path}")
        self.index_cases: dict[str, dict[str, Any]] = cases
        self.cache_size = int(cache_size)
        self._cache: OrderedDict[Path, dict[str, Any]] = OrderedDict()
        self.shard_load_count = 0

    @property
    def cached_shard_count(self) -> int:
        return len(self._cache)

    def _resolve_shard(self, shard_path: str) -> Path:
        raw = Path(shard_path)
        return raw.resolve() if raw.is_absolute() else (self.experiment_dir / raw).resolve()

    def _payload(self, shard_path: str) -> tuple[Path, dict[str, Any]]:
        resolved = self._resolve_shard(shard_path)
        if resolved in self._cache:
            payload = self._cache.pop(resolved)
            self._cache[resolved] = payload
            return resolved, payload
        if not resolved.is_file():
            raise FileNotFoundError(f"Hidden-state shard does not exist: {resolved}")
        payload = _torch_load(resolved)
        self._cache[resolved] = payload
        self.shard_load_count += 1
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return resolved, payload

    @staticmethod
    def _validate_reference_pair(
        case_id: str,
        manifest_reference: dict[str, Any],
        index_reference: dict[str, Any],
    ) -> None:
        for field in (
            "shard_path",
            "offset",
            "hidden_size",
            "hidden_state_definition",
        ):
            if manifest_reference.get(field) != index_reference.get(field):
                raise ValueError(
                    f"Manifest/index mismatch for {case_id}: {field} "
                    f"{manifest_reference.get(field)!r} != {index_reference.get(field)!r}"
                )
        for field in ("layer_indices", "position_names"):
            if list(manifest_reference.get(field) or []) != list(
                index_reference.get(field) or []
            ):
                raise ValueError(f"Manifest/index mismatch for {case_id}: {field}")

    def load_vector(
        self,
        record: dict[str, Any],
        layer: int,
        position_name: str,
    ) -> np.ndarray:
        case_id = str(record.get("case_id"))
        manifest_reference = record.get("hidden_state_reference")
        if not isinstance(manifest_reference, dict):
            raise ValueError(f"Manifest case has no hidden_state_reference: {case_id}")
        index_reference = self.index_cases.get(case_id)
        if not isinstance(index_reference, dict):
            raise KeyError(f"Hidden-state index has no case {case_id!r}")
        self._validate_reference_pair(case_id, manifest_reference, index_reference)
        shard_path, payload = self._payload(str(manifest_reference["shard_path"]))
        if payload.get("hidden_state_definition") != HIDDEN_STATE_DEFINITION:
            raise ValueError(
                f"Unexpected hidden-state definition in {shard_path}: "
                f"{payload.get('hidden_state_definition')!r}"
            )
        layer_indices = [int(value) for value in payload.get("layer_indices", [])]
        position_names = [str(value) for value in payload.get("position_names", [])]
        if not layer_indices:
            legacy_layer = payload.get("layer_index")
            if legacy_layer is not None:
                layer_indices = [int(legacy_layer)]
        if position_name not in POSITION_NAMES:
            raise ValueError(f"Unknown Probe position name: {position_name!r}")
        if position_name not in position_names:
            raise ValueError(
                f"Position {position_name!r} is absent from shard {shard_path}; "
                f"available={position_names}"
            )
        if int(layer) not in layer_indices:
            raise ValueError(
                f"Layer {layer} is absent from shard {shard_path}; available={layer_indices}"
            )
        hidden_states = payload.get("hidden_states")
        case_ids = payload.get("case_ids")
        if not isinstance(hidden_states, torch.Tensor):
            raise ValueError(f"Shard has no hidden_states tensor: {shard_path}")
        if not isinstance(case_ids, list):
            raise ValueError(f"Shard has no case_ids list: {shard_path}")
        offset = int(manifest_reference["offset"])
        if offset < 0 or offset >= int(hidden_states.shape[0]) or offset >= len(case_ids):
            raise IndexError(
                f"Offset {offset} is invalid for shard {shard_path} with "
                f"{hidden_states.shape[0]} tensor rows and {len(case_ids)} case IDs"
            )
        if str(case_ids[offset]) != case_id:
            raise ValueError(
                f"Case ID mismatch at {shard_path} offset {offset}: "
                f"expected {case_id!r}, found {case_ids[offset]!r}"
            )
        position_index = position_names.index(position_name)
        if len(layer_indices) == 1:
            if hidden_states.ndim != 3:
                raise ValueError(
                    f"Single-layer shard must be [case, position, hidden], got "
                    f"{tuple(hidden_states.shape)} in {shard_path}"
                )
            if int(hidden_states.shape[1]) != len(position_names):
                raise ValueError(
                    f"Position count mismatch in {shard_path}: tensor has "
                    f"{hidden_states.shape[1]}, payload names={position_names}"
                )
            vector = hidden_states[offset, position_index, :]
        else:
            if hidden_states.ndim != 4:
                raise ValueError(
                    f"Multi-layer shard must be [case, layer, position, hidden], got "
                    f"{tuple(hidden_states.shape)} in {shard_path}"
                )
            if int(hidden_states.shape[1]) != len(layer_indices):
                raise ValueError(
                    f"Layer count mismatch in {shard_path}: tensor has "
                    f"{hidden_states.shape[1]}, payload layers={layer_indices}"
                )
            if int(hidden_states.shape[2]) != len(position_names):
                raise ValueError(
                    f"Position count mismatch in {shard_path}: tensor has "
                    f"{hidden_states.shape[2]}, payload names={position_names}"
                )
            vector = hidden_states[
                offset,
                layer_indices.index(int(layer)),
                position_index,
                :,
            ]
        expected_hidden = int(manifest_reference["hidden_size"])
        if vector.ndim != 1 or int(vector.shape[0]) != expected_hidden:
            raise ValueError(
                f"Hidden size mismatch for {case_id}: vector={tuple(vector.shape)}, "
                f"reference={expected_hidden}"
            )
        payload_hidden = int(payload.get("hidden_size", vector.shape[0]))
        if payload_hidden != expected_hidden:
            raise ValueError(
                f"Payload hidden size mismatch for {case_id}: "
                f"{payload_hidden} != {expected_hidden}"
            )
        return vector.detach().cpu().float().numpy().astype(np.float32, copy=False)
