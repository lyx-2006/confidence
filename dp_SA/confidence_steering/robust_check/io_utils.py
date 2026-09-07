from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np

from dp_SA.unimodal_logit_confidence.io_utils import (
    atomic_csv as _atomic_csv,
    atomic_json as _atomic_json,
    atomic_jsonl as _atomic_jsonl,
    atomic_text as _atomic_text,
    canonical_hash,
    load_jsonl,
    sha256_file,
    stable_shard,
)

from .config import OUTPUT_ROOT


def assert_output_path(path: str | Path) -> Path:
    destination = Path(path).resolve()
    root = OUTPUT_ROOT.resolve()
    if destination != root and root not in destination.parents:
        raise ValueError(f"Write outside robust_check/output is forbidden: {destination}")
    return destination


def ensure_layout(root: str | Path) -> Path:
    destination = assert_output_path(root)
    for relative in (
        "artifacts/source_hashes", "artifacts/splits", "artifacts/hidden_reuse",
        "artifacts/probes", "artifacts/directions", "artifacts/trials",
        "artifacts/diagnostics", "tables", "figures", "progress",
    ):
        (destination / relative).mkdir(parents=True, exist_ok=True)
    return destination


def atomic_json(path: str | Path, value: Any) -> None:
    _atomic_json(assert_output_path(path), value)


def atomic_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_jsonl(assert_output_path(path), rows)


def atomic_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    names: list[str] = []
    for row in values:
        for name in row:
            if name not in names:
                names.append(name)
    _atomic_csv(assert_output_path(path), values, fieldnames=names)


def atomic_text(path: str | Path, value: str) -> None:
    _atomic_text(assert_output_path(path), value)


def atomic_joblib(path: str | Path, value: Any) -> None:
    destination = assert_output_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    joblib.dump(value, temporary)
    temporary.replace(destination)


def atomic_npz(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    destination = assert_output_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(destination)


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    destination = assert_output_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        handle.flush()


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def canonical_jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def inventory_hashes(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for path in sorted({p.resolve() for p in paths}, key=str):
        if not path.is_file():
            raise FileNotFoundError(path)
        output[str(path)] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    return output


def verify_inventory(before: dict[str, dict[str, Any]]) -> None:
    current = inventory_hashes(Path(path) for path in before)
    if current != before:
        changed = sorted(path for path in set(before) | set(current) if before.get(path) != current.get(path))
        raise RuntimeError(f"Reused historical files changed: {changed[:10]}")

