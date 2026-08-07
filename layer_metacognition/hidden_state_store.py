"""Atomic, resumable CPU-FP16 hidden-state shards and JSONL helpers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def atomic_write_text(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def append_jsonl(path: str | Path, value: dict[str, Any], fsync: bool = True) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        if fsync:
            os.fsync(handle.fileno())


def load_jsonl(path: str | Path, repair_trailing: bool = False) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    raw_lines = source.read_text(encoding="utf-8").splitlines()
    nonempty_indices = [index for index, line in enumerate(raw_lines) if line.strip()]
    final_nonempty = nonempty_indices[-1] if nonempty_indices else -1
    records: list[dict[str, Any]] = []
    valid_lines: list[str] = []
    repaired = False
    for index, line in enumerate(raw_lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            if repair_trailing and index == final_nonempty:
                repaired = True
                break
            raise ValueError(f"Invalid JSONL at {source}:{index + 1}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record at {source}:{index + 1} is not an object")
        records.append(value)
        valid_lines.append(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    if repaired:
        atomic_write_text(source, "\n".join(valid_lines) + ("\n" if valid_lines else ""))
    return records


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class HiddenStateStore:
    """Buffer completed cases and atomically commit fixed-size PANL shards."""

    def __init__(self, output_dir: str | Path, shard_size: int = 16):
        if shard_size < 1:
            raise ValueError("shard_size must be positive")
        self.output_dir = Path(output_dir)
        self.hidden_dir = self.output_dir / "hidden_states"
        self.hidden_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.hidden_dir / "index.json"
        self.shard_size = shard_size
        self._pending: list[tuple[str, torch.Tensor, list[int], dict[str, Any]]] = []
        indices = [int(path.stem.split("_")[-1]) for path in self.hidden_dir.glob("shard_*.pt")]
        self._next_shard = max(indices, default=-1) + 1

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def add(
        self,
        case_id: str,
        panl_hidden: torch.Tensor,
        layer_indices: list[int],
        result: dict[str, Any],
    ) -> bool:
        tensor = panl_hidden.detach().to(device="cpu", dtype=torch.float16).contiguous()
        if tensor.ndim != 2 or tensor.shape[0] != len(layer_indices):
            raise ValueError(
                f"PANL tensor must have shape [layers, hidden], got {tuple(tensor.shape)} "
                f"for {len(layer_indices)} layers"
            )
        if self._pending and self._pending[0][2] != layer_indices:
            raise ValueError("All cases in a shard must use the same layer selection")
        self._pending.append((case_id, tensor, list(layer_indices), result))
        return len(self._pending) >= self.shard_size

    def flush(self, results_path: str | Path) -> list[str]:
        if not self._pending:
            return []
        shard_name = f"shard_{self._next_shard:05d}.pt"
        shard_path = self.hidden_dir / shard_name
        case_ids = [entry[0] for entry in self._pending]
        layer_indices = self._pending[0][2]
        tensors = torch.stack([entry[1] for entry in self._pending], dim=0)
        panl_positions = [int(entry[3]["positions"]["panl"]["position"]) for entry in self._pending]
        model_num_hidden_layers = int(
            self._pending[0][3]["model_structure"]["num_hidden_layers"]
        )
        payload = {
            "format_version": 1,
            "case_ids": case_ids,
            "layer_indices": layer_indices,
            "hidden_states": tensors,
            "panl_positions": panl_positions,
            "model_num_hidden_layers": model_num_hidden_layers,
            "stored_layer_count": int(tensors.shape[1]),
            "hidden_size": int(tensors.shape[2]),
        }
        _atomic_torch_save(shard_path, payload)
        for offset, (_case_id, _tensor, _layers, result) in enumerate(self._pending):
            result["hidden_state_reference"] = {
                "stage": "stage2",
                "position_name": "panl",
                "panl_shard": str(Path("hidden_states") / shard_name),
                "panl_offset": offset,
                "position": result["positions"]["panl"]["position"],
                "num_layers": result["model_structure"]["num_hidden_layers"],
                "stored_layer_indices": layer_indices,
                "hidden_size": int(tensors.shape[2]),
            }
            append_jsonl(results_path, result, fsync=False)
        with Path(results_path).open("a", encoding="utf-8") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        committed = list(case_ids)
        self._pending.clear()
        self._next_shard += 1
        completed_ids = {record["case_id"] for record in load_jsonl(results_path, repair_trailing=True)}
        self.rebuild_index(completed_ids)
        return committed

    def rebuild_index(self, completed_case_ids: set[str]) -> dict[str, Any]:
        cases: dict[str, Any] = {}
        for shard_path in sorted(self.hidden_dir.glob("shard_*.pt")):
            payload = _torch_load(shard_path)
            case_ids = payload["case_ids"]
            tensor = payload["hidden_states"]
            layer_indices = [int(value) for value in payload["layer_indices"]]
            positions = payload.get("panl_positions", [None] * len(case_ids))
            model_num_layers = int(payload.get("model_num_hidden_layers", len(layer_indices)))
            for offset, case_id in enumerate(case_ids):
                if case_id not in completed_case_ids:
                    continue
                cases[case_id] = {
                    "case_id": case_id,
                    "stage": "stage2",
                    "position_name": "panl",
                    "position": positions[offset],
                    "num_layers": model_num_layers,
                    "stored_layer_count": len(layer_indices),
                    "stored_layer_indices": layer_indices,
                    "hidden_size": int(tensor.shape[2]),
                    "shard_path": str(Path("hidden_states") / shard_path.name),
                    "offset": offset,
                }
        index = {"format_version": 1, "cases": cases}
        atomic_write_json(self.index_path, index)
        return index

    def read_case(self, case_id: str) -> tuple[torch.Tensor, list[int]]:
        if not self.index_path.exists():
            raise KeyError(f"Hidden-state index does not exist: {self.index_path}")
        index = json.loads(self.index_path.read_text(encoding="utf-8"))
        reference = index.get("cases", {}).get(case_id)
        if reference is None:
            raise KeyError(f"No hidden state stored for case {case_id!r}")
        payload = _torch_load(self.output_dir / reference["shard_path"])
        tensor = payload["hidden_states"][int(reference["offset"])].clone()
        return tensor, [int(value) for value in payload["layer_indices"]]


class TargetLayerHiddenStateStore:
    """Atomically store selected decoder layers' named position vectors."""

    SUPPORTED_POSITION_NAMES = ("ac", "panl", "ltt", "ptnl", "sac")
    FORMAT_VERSION = 2
    HIDDEN_STATE_DEFINITION = "decoder_block_output_pre_final_norm"

    def __init__(
        self,
        output_dir: str | Path,
        layer_index: int | list[int] | tuple[int, ...],
        position_names: list[str] | tuple[str, ...] = ("ac", "panl"),
        shard_size: int = 64,
    ):
        raw_layers = (
            [layer_index] if isinstance(layer_index, int) else list(layer_index)
        )
        if not raw_layers or any(value < 0 for value in raw_layers):
            raise ValueError("layer_index must contain non-negative layers")
        if len(set(raw_layers)) != len(raw_layers):
            raise ValueError("layer_index must not contain duplicates")
        if shard_size < 1:
            raise ValueError("shard_size must be positive")
        raw_positions = [str(value) for value in position_names]
        if not raw_positions:
            raise ValueError("position_names must not be empty")
        if len(set(raw_positions)) != len(raw_positions):
            raise ValueError("position_names must not contain duplicates")
        invalid_positions = [
            value
            for value in raw_positions
            if value not in self.SUPPORTED_POSITION_NAMES
        ]
        if invalid_positions:
            raise ValueError(f"Unsupported position_names: {invalid_positions}")
        self.output_dir = Path(output_dir)
        self.layer_indices = tuple(int(value) for value in raw_layers)
        self.position_names = tuple(raw_positions)
        self.layer_index = (
            self.layer_indices[0] if len(self.layer_indices) == 1 else None
        )
        self.hidden_root = self.output_dir / "hidden_states"
        layer_label = "_".join(str(value) for value in self.layer_indices)
        directory_name = (
            f"target_layer_{layer_label}"
            if len(self.layer_indices) == 1
            else f"target_layers_{layer_label}"
        )
        self.hidden_dir = self.hidden_root / directory_name
        self.hidden_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.hidden_root / "index.json"
        self.shard_size = int(shard_size)
        self._pending: list[
            tuple[str, torch.Tensor, dict[str, int], dict[str, str], dict[str, Any]]
        ] = []
        indices = [
            int(path.stem.split("_")[-1])
            for path in self.hidden_dir.glob("shard_*.pt")
        ]
        self._next_shard = max(indices, default=-1) + 1
        self._validate_existing_shards()

    def _validate_existing_shards(self) -> None:
        for shard_path in sorted(self.hidden_dir.glob("shard_*.pt")):
            payload = _torch_load(shard_path)
            payload_layers = tuple(
                int(value)
                for value in payload.get(
                    "layer_indices",
                    [payload.get("layer_index")],
                )
            )
            payload_positions = tuple(
                str(value) for value in payload.get("position_names", [])
            )
            if payload_layers != self.layer_indices:
                raise ValueError(
                    f"Existing hidden-state shard layer schema differs: {shard_path}; "
                    f"existing={list(payload_layers)}, requested={list(self.layer_indices)}"
                )
            if payload_positions != self.position_names:
                raise ValueError(
                    f"Existing hidden-state shard position schema differs: {shard_path}; "
                    f"existing={list(payload_positions)}, requested={list(self.position_names)}"
                )
            tensor = payload.get("hidden_states")
            expected_ndim = 3 if len(self.layer_indices) == 1 else 4
            position_axis = 1 if len(self.layer_indices) == 1 else 2
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.ndim != expected_ndim
                or int(tensor.shape[position_axis]) != len(self.position_names)
            ):
                shape = None if not isinstance(tensor, torch.Tensor) else tuple(tensor.shape)
                raise ValueError(
                    f"Existing hidden-state shard tensor schema differs: {shard_path}; "
                    f"shape={shape}, position_names={list(self.position_names)}"
                )

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def add(
        self,
        case_id: str,
        hidden_states: torch.Tensor,
        positions: dict[str, int],
        stages: dict[str, str],
        result: dict[str, Any],
    ) -> bool:
        tensor = (
            hidden_states.detach()
            .to(device="cpu", dtype=torch.float16)
            .contiguous()
        )
        valid_shape = (
            tensor.ndim == 2
            and len(self.layer_indices) == 1
            and int(tensor.shape[0]) == len(self.position_names)
        ) or (
            tensor.ndim == 3
            and len(self.layer_indices) > 1
            and int(tensor.shape[0]) == len(self.layer_indices)
            and int(tensor.shape[1]) == len(self.position_names)
        )
        if not valid_shape:
            expected = (
                f"[{len(self.position_names)}, hidden]"
                if len(self.layer_indices) == 1
                else (
                    f"[{len(self.layer_indices)}, {len(self.position_names)}, hidden]"
                )
            )
            raise ValueError(
                f"Target-layer hidden state must have shape {expected}, "
                f"got {tuple(tensor.shape)}"
            )
        normalized_positions = {
            name: int(positions[name]) for name in self.position_names
        }
        normalized_stages = {
            name: str(stages[name]) for name in self.position_names
        }
        if self._pending and int(self._pending[0][1].shape[-1]) != int(tensor.shape[-1]):
            raise ValueError("All cases in a shard must use the same hidden size")
        self._pending.append(
            (
                str(case_id),
                tensor,
                normalized_positions,
                normalized_stages,
                result,
            )
        )
        return len(self._pending) >= self.shard_size

    def flush(self, results_path: str | Path) -> list[str]:
        if not self._pending:
            return []
        shard_name = f"shard_{self._next_shard:05d}.pt"
        relative_shard = (
            Path("hidden_states")
            / self.hidden_dir.name
            / shard_name
        )
        shard_path = self.output_dir / relative_shard
        case_ids = [entry[0] for entry in self._pending]
        tensors = torch.stack([entry[1] for entry in self._pending], dim=0)
        positions = [entry[2] for entry in self._pending]
        stages = [entry[3] for entry in self._pending]
        payload = {
            "format_version": self.FORMAT_VERSION,
            "case_ids": case_ids,
            "layer_indices": list(self.layer_indices),
            "position_names": list(self.position_names),
            "hidden_states": tensors,
            "positions": positions,
            "stages": stages,
            "dtype": "float16",
            "hidden_size": int(tensors.shape[-1]),
            "hidden_state_definition": self.HIDDEN_STATE_DEFINITION,
        }
        if self.layer_index is not None:
            payload["layer_index"] = self.layer_index
        _atomic_torch_save(shard_path, payload)
        for offset, entry in enumerate(self._pending):
            _case_id, _tensor, case_positions, case_stages, result = entry
            result["hidden_state_reference"] = {
                "format_version": self.FORMAT_VERSION,
                "layer_indices": list(self.layer_indices),
                "position_names": list(self.position_names),
                "shard_path": str(relative_shard),
                "offset": offset,
                "positions": case_positions,
                "stages": case_stages,
                "dtype": "float16",
                "hidden_size": int(tensors.shape[-1]),
                "hidden_state_definition": self.HIDDEN_STATE_DEFINITION,
            }
            if self.layer_index is not None:
                result["hidden_state_reference"]["layer_index"] = self.layer_index
            append_jsonl(results_path, result, fsync=False)
        with Path(results_path).open("a", encoding="utf-8") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        committed = list(case_ids)
        self._pending.clear()
        self._next_shard += 1
        completed_ids = {
            str(record["case_id"])
            for record in load_jsonl(results_path, repair_trailing=True)
            if record.get("status") == "completed"
        }
        self.rebuild_index(completed_ids)
        return committed

    def rebuild_index(self, completed_case_ids: set[str]) -> dict[str, Any]:
        cases: dict[str, Any] = {}
        for shard_path in sorted(self.hidden_dir.glob("shard_*.pt")):
            payload = _torch_load(shard_path)
            payload_layers = tuple(
                int(value)
                for value in payload.get(
                    "layer_indices",
                    [payload.get("layer_index")],
                )
            )
            if payload_layers != self.layer_indices:
                raise ValueError(f"Unexpected layer indices in {shard_path}")
            payload_positions = tuple(
                str(value) for value in payload.get("position_names", [])
            )
            if payload_positions != self.position_names:
                raise ValueError(f"Unexpected position names in {shard_path}")
            relative_shard = shard_path.relative_to(self.output_dir)
            case_ids = [str(value) for value in payload["case_ids"]]
            positions = payload["positions"]
            stages = payload["stages"]
            tensors = payload["hidden_states"]
            for offset, case_id in enumerate(case_ids):
                if case_id not in completed_case_ids:
                    continue
                cases[case_id] = {
                    "case_id": case_id,
                    "layer_indices": list(self.layer_indices),
                    "position_names": list(self.position_names),
                    "positions": positions[offset],
                    "stages": stages[offset],
                    "dtype": "float16",
                    "hidden_size": int(tensors.shape[-1]),
                    "hidden_state_definition": self.HIDDEN_STATE_DEFINITION,
                    "shard_path": str(relative_shard),
                    "offset": offset,
                }
                if self.layer_index is not None:
                    cases[case_id]["layer_index"] = self.layer_index
        index = {
            "format_version": self.FORMAT_VERSION,
            "layer_indices": list(self.layer_indices),
            "position_names": list(self.position_names),
            "cases": cases,
        }
        if self.layer_index is not None:
            index["layer_index"] = self.layer_index
        atomic_write_json(self.index_path, index)
        return index

    def read_case(self, case_id: str) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self.index_path.exists():
            raise KeyError(f"Hidden-state index does not exist: {self.index_path}")
        index = json.loads(self.index_path.read_text(encoding="utf-8"))
        reference = index.get("cases", {}).get(case_id)
        if reference is None:
            raise KeyError(f"No hidden state stored for case {case_id!r}")
        payload = _torch_load(self.output_dir / reference["shard_path"])
        tensor = payload["hidden_states"][int(reference["offset"])].clone()
        return tensor, dict(reference)
