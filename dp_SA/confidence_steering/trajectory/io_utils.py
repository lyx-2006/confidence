from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np

from .config import OUTPUT_PARENT


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def stable_shard(value: Any, num_gpus: int) -> int:
    if num_gpus not in (1, 2):
        raise ValueError("num_gpus must be 1 or 2")
    return int(canonical_hash(value), 16) % num_gpus


def require_output_root(path: str | Path) -> Path:
    root = Path(path).resolve()
    try:
        root.relative_to(OUTPUT_PARENT.resolve())
    except ValueError as exc:
        raise ValueError(f"Output root must remain inside {OUTPUT_PARENT.resolve()}: {root}") from exc
    if root == OUTPUT_PARENT.resolve():
        raise ValueError("Output root must be a child of trajectory/output")
    return root


def ensure_layout(path: str | Path, *, resume: bool) -> Path:
    root = require_output_root(path)
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError(f"Output exists; use --resume: {root}")
    for relative in (
        "artifacts/manifests", "artifacts/clean_hidden/by_case", "artifacts/probes",
        "artifacts/trajectory_hidden", "artifacts/trials", "artifacts/diagnostics",
        "tables", "figures", "progress",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    return root


def _atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def atomic_json(path: str | Path, value: Any) -> None:
    _atomic(Path(path), json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode() + b"\n")


def atomic_text(path: str | Path, value: str) -> None:
    _atomic(Path(path), value.encode())


def atomic_bytes(path: str | Path, value: bytes) -> None:
    _atomic(Path(path), value)


def atomic_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    payload = b"".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode() + b"\n" for row in rows)
    _atomic(Path(path), payload)


def atomic_csv(path: str | Path, rows: Sequence[dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    values = list(rows); names = list(fields or [])
    if not names:
        for row in values:
            for key in row:
                if key not in names: names.append(key)
    stream = io.StringIO(); writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
    writer.writeheader(); writer.writerows(values); atomic_text(path, stream.getvalue())


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


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file(): return []
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]


def semantic_lock(path: Path, payload: dict[str, Any], *, resume: bool) -> str:
    fingerprint = canonical_hash(payload)
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("fingerprint") != fingerprint:
            raise ValueError(f"Semantic fingerprint mismatch: {path}")
        if not resume:
            raise FileExistsError(path)
    else:
        atomic_json(path, {**payload, "fingerprint": fingerprint})
    return fingerprint


def inventory(paths: Iterable[str | Path]) -> dict[str, str]:
    output = {}
    for raw in sorted({str(Path(value).resolve()) for value in paths}):
        path = Path(raw)
        if not path.is_file(): raise FileNotFoundError(path)
        output[raw] = sha256_file(path)
    return output


def verify_inventory(expected: dict[str, str]) -> None:
    for raw, digest in expected.items():
        path = Path(raw)
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Protected historical file changed: {path}")


def canonical_forward_key(case_id: str, direction: str | None, alpha: float) -> str:
    return canonical_hash({"case_id": case_id, "direction": direction or "baseline", "alpha": float(alpha)})
