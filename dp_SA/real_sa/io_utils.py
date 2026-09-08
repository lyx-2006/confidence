from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from dp_SA.io_utils import canonical_hash, sha256_file


def ensure_layout(root: str | Path) -> Path:
    output = Path(root).resolve()
    for relative in (
        "artifacts/manifests", "artifacts/mean_embeddings", "tables", "figures",
        "progress", "progress/smoke",
    ):
        (output / relative).mkdir(parents=True, exist_ok=True)
    return output


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: str | Path, value: Any) -> None:
    _atomic_bytes(Path(path), json.dumps(value, ensure_ascii=False, indent=2).encode() + b"\n")


def atomic_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    payload = b"".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
        for row in rows
    )
    _atomic_bytes(Path(path), payload)


def atomic_csv(path: str | Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    stream = __import__("io").StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(Path(path), stream.getvalue().encode())


def atomic_torch_save(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
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


def load_jsonl(path: str | Path, *, repair_trailing: bool = False) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    raw = source.read_bytes()
    lines = raw.splitlines(keepends=True)
    rows: list[dict[str, Any]] = []
    valid = 0
    for index, line in enumerate(lines):
        if not line.strip():
            valid += len(line)
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if repair_trailing and index == len(lines) - 1:
                _atomic_bytes(source, raw[:valid])
                break
            raise
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row is not an object: {source}:{index + 1}")
        rows.append(value)
        valid += len(line)
    return rows


def upsert_jsonl(path: str | Path, rows: Sequence[dict[str, Any]], row: dict[str, Any], *, key: str) -> list[dict[str, Any]]:
    value = str(row[key])
    if any(str(old[key]) == value for old in rows):
        raise ValueError(f"Duplicate {key}: {value}")
    updated = [*rows, row]
    atomic_jsonl(path, sorted(updated, key=lambda item: str(item[key])))
    return updated


def validate_fingerprint(path: Path, payload: dict[str, Any]) -> str:
    fingerprint = canonical_hash(payload)
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("fingerprint") != fingerprint:
            raise ValueError(f"Run fingerprint mismatch: {path}")
    else:
        atomic_json(path, {**payload, "fingerprint": fingerprint})
    return fingerprint


__all__ = [
    "atomic_csv", "atomic_json", "atomic_jsonl", "atomic_torch_save", "canonical_hash",
    "ensure_layout", "load_jsonl", "sha256_file", "upsert_jsonl", "validate_fingerprint",
]
