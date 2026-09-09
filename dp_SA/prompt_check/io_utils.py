from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return []
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]


def _temporary(destination: Path) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    return tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)


def atomic_json(path: str | Path, value: Any) -> None:
    destination = Path(path); fd, temporary = _temporary(destination)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=True); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def atomic_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path); fd, temporary = _temporary(destination)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows: handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=True) + "\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=True) + "\n"); handle.flush(); os.fsync(handle.fileno())


def atomic_csv(path: str | Path, rows: Sequence[dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    destination = Path(path); fd, temporary = _temporary(destination)
    fields = list(fields or (list(rows[0]) if rows else []))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def atomic_npz(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    destination = Path(path); fd, temporary = _temporary(destination); os.close(fd)
    try:
        with open(temporary, "wb") as handle:
            np.savez(handle, **arrays); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def inventory(paths: Iterable[str | Path]) -> dict[str, str]:
    return {str(Path(path).resolve()): sha256_file(path) for path in sorted(map(Path, paths)) if path.is_file()}


def verify_inventory(expected: dict[str, str]) -> None:
    changed = [path for path, digest in expected.items() if not Path(path).is_file() or sha256_file(path) != digest]
    if changed: raise RuntimeError(f"Frozen historical sources changed: {changed[:5]}")
