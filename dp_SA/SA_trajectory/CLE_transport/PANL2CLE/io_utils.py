from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def atomic_json(path: str | Path, value: Any) -> None:
    _atomic(Path(path), (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


def atomic_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic(Path(path), b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode() for row in rows))


def atomic_text(path: str | Path, value: str) -> None:
    _atomic(Path(path), value.encode())


def atomic_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    values = list(rows); fields: list[str] = []
    for row in values:
        for key in row:
            if key not in fields: fields.append(key)
    stream = io.StringIO(); writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader(); writer.writerows(values)
    atomic_text(path, stream.getvalue())


def bits_hash(bits: np.ndarray) -> str:
    if bits.dtype != np.uint16: raise TypeError("BF16 storage must be uint16")
    return hashlib.sha256(np.ascontiguousarray(bits).tobytes()).hexdigest()


def bf16_to_uint16(tensor: torch.Tensor) -> np.ndarray:
    value = tensor.detach().contiguous().cpu()
    if value.dtype != torch.bfloat16: raise TypeError(f"Expected bfloat16, got {value.dtype}")
    return value.view(torch.uint16).numpy().copy()


def uint16_to_bf16(bits: np.ndarray) -> torch.Tensor:
    value = np.ascontiguousarray(bits)
    if value.dtype != np.uint16: raise TypeError(f"Expected uint16, got {value.dtype}")
    return torch.from_numpy(value.copy()).view(torch.bfloat16)


def atomic_bf16_npz(path: str | Path, arrays: dict[str, torch.Tensor]) -> dict[str, Any]:
    payload: dict[str, np.ndarray] = {}; metadata: dict[str, Any] = {"format": "raw_bfloat16_bits_v1"}
    for key, tensor in arrays.items():
        bits = bf16_to_uint16(tensor); payload[key] = bits
        metadata[key] = {"shape": list(bits.shape), "logical_dtype": "bfloat16", "bits_sha256": bits_hash(bits)}
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".npz", dir=destination.parent); os.close(descriptor)
    try:
        np.savez_compressed(temporary, **payload, __metadata__=np.asarray(json.dumps(metadata, sort_keys=True)))
        os.replace(temporary, destination)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise
    return metadata


def load_bf16_npz(path: str | Path, key: str) -> tuple[torch.Tensor, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["__metadata__"].item())); bits = np.asarray(archive[key]).copy()
    if metadata.get("format") != "raw_bfloat16_bits_v1" or key not in metadata:
        raise ValueError("Invalid BF16 cache metadata")
    if bits_hash(bits) != metadata[key]["bits_sha256"]: raise ValueError(f"Corrupt BF16 cache: {path}:{key}")
    tensor = uint16_to_bf16(bits)
    if list(tensor.shape) != metadata[key]["shape"]: raise ValueError("BF16 cache shape mismatch")
    return tensor, metadata[key]

