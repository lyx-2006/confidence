from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from dp_SA.io_utils import atomic_json, atomic_jsonl, canonical_hash, sha256_file


def load_jsonl_strict(path: str | Path, *, repair_trailing: bool = False) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    lines = source.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if repair_trailing and index == len(lines) - 1:
                atomic_jsonl(source, rows)
                return rows
            raise
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {index + 1} is not an object: {source}")
        rows.append(value)
    return rows


def atomic_append_jsonl(path: str | Path, rows: Sequence[dict[str, Any]], row: dict[str, Any]) -> list[dict[str, Any]]:
    updated = [*rows, row]
    atomic_jsonl(path, updated)
    return updated


def atomic_torch_save(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def directory_code_hash(paths: Iterable[Path]) -> dict[str, str]:
    output: dict[str, str] = {}
    for path in sorted({Path(value).resolve() for value in paths}, key=str):
        if path.is_file():
            output[str(path)] = sha256_file(path)
    return output


def stable_seed(seed: int, *parts: str) -> int:
    raw = ":".join([str(int(seed)), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


__all__ = [
    "atomic_append_jsonl", "atomic_csv", "atomic_json", "atomic_jsonl",
    "atomic_torch_save", "canonical_hash", "directory_code_hash",
    "load_jsonl_strict", "sha256_file", "stable_seed",
]
