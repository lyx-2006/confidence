from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from .config import CAUSAL_PAIRS, CLE_LAYERS, PANL_LAYERS


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2); handle.write("\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def atomic_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file(): return []
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_bf16_npz(path: str | Path, tensors: dict[str, torch.Tensor]) -> None:
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for key, tensor in tensors.items():
        value = tensor.detach().cpu().reshape(-1)
        if value.dtype != torch.bfloat16:
            raise TypeError(f"{key} must be bfloat16, got {value.dtype}")
        arrays[key] = value.view(torch.uint16).numpy().copy()
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".npz", dir=destination.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, destination)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def load_bf16(path: str | Path, key: str) -> torch.Tensor:
    with np.load(path, allow_pickle=False) as archive:
        array = np.asarray(archive[key], dtype=np.uint16).copy()
    return torch.from_numpy(array).view(torch.bfloat16)


def validate_layer_design(pairs: Sequence[tuple[int, int]] = CAUSAL_PAIRS) -> None:
    expected = {(p, c) for p in PANL_LAYERS for c in CLE_LAYERS if c > p}
    if set(map(tuple, pairs)) != expected or len(pairs) != len(expected) or len(expected) != 9:
        raise ValueError("PANL/CLE layer design must contain exactly the nine causal pairs")


def physical_trial_key(row: dict[str, Any]) -> str:
    return "|".join(map(str, (row["case_id"], row["condition"], row.get("panl_layer"),
                              row.get("cle_layer"), float(row.get("alpha", 0.0)))))


def expected_physical_count(case_count: int) -> int:
    return int(case_count) * (1 + len(PANL_LAYERS) * len(( -5.0, 5.0)) + 2 * len(CAUSAL_PAIRS) * 2)


def expected_logical_count(case_count: int) -> int:
    return int(case_count) * len(CAUSAL_PAIRS) * 2 * 4

