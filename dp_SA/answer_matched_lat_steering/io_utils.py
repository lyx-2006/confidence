from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _atomic_writer(path: Path, mode: str = "w"):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    return fd, temporary


def atomic_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    fd, temporary = _atomic_writer(destination)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=True)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def atomic_text(path: str | Path, value: str) -> None:
    destination = Path(path)
    fd, temporary = _atomic_writer(destination)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def atomic_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    fd, temporary = _atomic_writer(destination)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=True) + "\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file(): return []
    output = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip(): output.append(json.loads(line))
    return output


def atomic_npz(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    destination = Path(path)
    fd, temporary = _atomic_writer(destination)
    os.close(fd)
    try:
        with open(temporary, "wb") as handle:
            np.savez(handle, **arrays); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def atomic_csv(path: str | Path, rows: Sequence[dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    destination = Path(path)
    if fields is None:
        fields = list(rows[0]) if rows else []
    fd, temporary = _atomic_writer(destination)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise
