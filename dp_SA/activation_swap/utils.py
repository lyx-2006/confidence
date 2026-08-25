from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from dp_SA.io_utils import canonical_hash, sha256_file


def atomic_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
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


def append_jsonl(path: str | Path, rows: Sequence[dict[str, Any]], row: dict[str, Any]) -> list[dict[str, Any]]:
    updated = [*rows, row]
    atomic_jsonl(path, updated)
    return updated


def stable_seed(seed: int, *parts: Any) -> int:
    raw = ":".join([str(int(seed)), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def stable_key(row: dict[str, Any]) -> tuple[Any, ...]:
    item = str(row.get("item_id", ""))
    item_key: tuple[Any, ...] = (0, int(item)) if item.isdigit() else (1, item)
    return (*item_key, int(row.get("prior_index", 0)), str(row.get("condition", "")),
            str(row.get("version", "v4")), str(row.get("case_id", "")))


def model_fingerprints(model_path: Path) -> dict[str, str]:
    return {name: sha256_file(model_path / name) for name in MODEL_FILES if (model_path / name).is_file()}


MODEL_FILES = (
    "config.json", "generation_config.json", "preprocessor_config.json", "processor_config.json",
    "chat_template.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json",
)


def finite_vector(values: Sequence[float], expected: int = 9) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64)
    if output.shape != (expected,) or not np.isfinite(output).all():
        raise ValueError(f"expected {expected} finite values")
    return output


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 0 or nb <= 0 or not math.isfinite(na * nb):
        raise ValueError("cosine distance requires non-zero finite vectors")
    return float(1.0 - np.dot(a, b) / (na * nb))


def directory_hash(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path.resolve()): sha256_file(path) for path in sorted({Path(p).resolve() for p in paths}) if path.is_file()}


__all__ = ["append_jsonl", "atomic_json", "atomic_jsonl", "canonical_hash", "cosine_distance",
           "directory_hash", "finite_vector", "load_jsonl", "model_fingerprints", "sha256_file",
           "stable_key", "stable_seed"]
