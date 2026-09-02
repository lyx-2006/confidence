from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def stable_shard(value: Any, num_gpus: int) -> int:
    if num_gpus not in (1, 2):
        raise ValueError("--num-gpus must be 1 or 2")
    return int(canonical_hash(value), 16) % num_gpus


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def atomic_json(path: str | Path, value: Any) -> None:
    _atomic(Path(path), json.dumps(value, ensure_ascii=False, indent=2).encode() + b"\n")


def atomic_text(path: str | Path, value: str) -> None:
    _atomic(Path(path), value.encode())


def atomic_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic(Path(path), b"".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode() + b"\n" for row in rows))


def atomic_csv(path: str | Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    rows = list(rows)
    names = list(fieldnames or (list(rows[0]) if rows else []))
    stream = io.StringIO(); writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
    writer.writeheader(); writer.writerows(rows); atomic_text(path, stream.getvalue())


def load_jsonl(path: str | Path, *, repair_trailing: bool = False) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists(): return []
    raw = source.read_bytes(); rows = []; valid = 0
    for index, line in enumerate(raw.splitlines(keepends=True)):
        if not line.strip(): valid += len(line); continue
        try: rows.append(json.loads(line)); valid += len(line)
        except json.JSONDecodeError:
            if not repair_trailing or index != len(raw.splitlines(keepends=True)) - 1: raise
            _atomic(source, raw[:valid]); break
    return rows


def ensure_layout(root: str | Path, *, resume: bool) -> Path:
    root = Path(root).resolve()
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError(f"Output already exists; pass --resume: {root}")
    for name in (
        "shared/manifests", "unimodal_confidence/artifacts/raw_scores", "unimodal_confidence/artifacts/calibrated_scores",
        "unimodal_confidence/artifacts/temperature", "unimodal_confidence/artifacts/predictions", "unimodal_confidence/tables",
        "unimodal_confidence/figures", "unimodal_confidence/progress", "confidence_probe/artifacts/hidden",
        "confidence_probe/artifacts/models", "confidence_probe/artifacts/predictions", "confidence_probe/tables",
        "confidence_probe/figures", "confidence_probe/progress",
    ): (root / name).mkdir(parents=True, exist_ok=True)
    return root


def validate_fingerprint(path: Path, payload: dict[str, Any], *, resume: bool) -> str:
    fingerprint = canonical_hash(payload)
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("fingerprint") != fingerprint: raise ValueError(f"Fingerprint mismatch: {path}")
        if not resume: raise FileExistsError(path)
    else: atomic_json(path, {**payload, "fingerprint": fingerprint})
    return fingerprint

