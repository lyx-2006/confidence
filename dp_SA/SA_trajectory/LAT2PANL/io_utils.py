from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).is_file():
        return []
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_bytes(path: Path, content: bytes) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode())


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_bytes(path, b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode() for row in rows))


def atomic_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames: fieldnames.append(key)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def bf16_to_uint16(tensor: torch.Tensor) -> np.ndarray:
    value = tensor.detach().contiguous().cpu()
    if value.dtype != torch.bfloat16:
        raise TypeError(f"Expected bfloat16 hidden, got {value.dtype}")
    return value.view(torch.uint16).numpy().copy()


def uint16_to_bf16(bits: np.ndarray) -> torch.Tensor:
    array = np.ascontiguousarray(bits)
    if array.dtype != np.uint16:
        raise TypeError(f"Expected uint16 bit array, got {array.dtype}")
    return torch.from_numpy(array.copy()).view(torch.bfloat16)


def bits_hash(bits: np.ndarray) -> str:
    if bits.dtype != np.uint16:
        raise TypeError("bf16 bit hash requires uint16")
    return hashlib.sha256(np.ascontiguousarray(bits).tobytes(order="C")).hexdigest()


def atomic_bf16_npz(path: Path, arrays: dict[str, torch.Tensor]) -> dict[str, Any]:
    payload: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {"format": "raw_bfloat16_bits_v1", "byte_order": "little"}
    for key, tensor in arrays.items():
        bits = bf16_to_uint16(tensor)
        payload[key] = bits
        metadata[key] = {"logical_dtype": "bfloat16", "storage_dtype": "uint16", "shape": list(bits.shape), "bits_sha256": bits_hash(bits)}
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **payload, __metadata__=np.asarray(json.dumps(metadata, sort_keys=True)))
        os.replace(temporary, path)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise
    return metadata


def load_bf16_npz(path: Path, key: str) -> tuple[torch.Tensor, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["__metadata__"].item()))
        bits = np.asarray(archive[key]).copy()
    if metadata.get("format") != "raw_bfloat16_bits_v1" or metadata.get("byte_order") != "little":
        raise ValueError("Unsupported bf16 hidden file metadata")
    if bits_hash(bits) != metadata[key]["bits_sha256"]:
        raise ValueError(f"Corrupt bf16 source: {path}:{key}")
    tensor = uint16_to_bf16(bits)
    if list(tensor.shape) != metadata[key]["shape"]:
        raise ValueError("bf16 hidden shape metadata mismatch")
    return tensor, metadata[key]
