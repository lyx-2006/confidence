from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from dp_SA.unimodal_logit_confidence.io_utils import (
    atomic_csv as _atomic_csv, atomic_json, atomic_jsonl, atomic_text, canonical_hash,
    load_jsonl, sha256_file, stable_shard,
)


LAYOUT = (
    "figures", "tables", "progress", "artifacts", "artifacts/manifests",
    "artifacts/directions", "artifacts/probes", "artifacts/trials",
    "artifacts/diagnostics", "artifacts/hidden",
)


def atomic_csv(path: str | Path, rows: Any) -> None:
    values = list(rows)
    names: list[str] = []
    for row in values:
        for key in row:
            if key not in names: names.append(key)
    _atomic_csv(path, values, fieldnames=names)


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def create_output_root(root: str | Path, *, resume: bool) -> Path:
    destination = Path(root).resolve()
    if destination.exists() and not resume:
        raise FileExistsError(f"Output directory exists; use --resume: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    for relative in LAYOUT:
        (destination / relative).mkdir(parents=True, exist_ok=True)
    return destination


def ensure_layout(root: str | Path) -> Path:
    destination = Path(root).resolve()
    for relative in LAYOUT:
        (destination / relative).mkdir(parents=True, exist_ok=True)
    return destination


def atomic_npz(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent); os.close(fd)
    try:
        with open(temporary, "wb") as handle:
            np.savez(handle, **arrays); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def atomic_joblib(path: str | Path, value: Any) -> None:
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent); os.close(fd)
    try:
        joblib.dump(value, temporary); os.replace(temporary, destination)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def semantic_fingerprint(path: Path, payload: dict[str, Any], *, resume: bool) -> str:
    fingerprint = canonical_hash(payload)
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("fingerprint") != fingerprint:
            raise ValueError(f"Resume fingerprint mismatch: {path}")
        if not resume:
            raise FileExistsError(f"Stage output exists; use --resume: {path}")
    else:
        atomic_json(path, {**payload, "fingerprint": fingerprint})
    return fingerprint


def validate_hashed_file(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"{label} fingerprint mismatch: {path}")
