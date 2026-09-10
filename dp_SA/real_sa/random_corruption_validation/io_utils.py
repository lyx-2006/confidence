from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path: str | Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def atomic_json(path: str | Path, value: Any) -> None:
    atomic_bytes(path, json.dumps(value, ensure_ascii=False, indent=2).encode() + b"\n")


def atomic_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_bytes(path, b"".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode() + b"\n" for row in rows))


def atomic_csv(path: str | Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    if not rows and fieldnames is None:
        raise ValueError("Empty CSV requires explicit fieldnames")
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(fieldnames or rows[0]), extrasaction="ignore")
    writer.writeheader(); writer.writerows(rows)
    atomic_bytes(path, stream.getvalue().encode())


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists(): return []
    rows=[]
    for index, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            value=json.loads(line)
            if not isinstance(value, dict): raise ValueError(f"Non-object JSONL row: {source}:{index}")
            rows.append(value)
    return rows


def load_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def upsert(path: Path, rows: list[dict[str, Any]], row: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value=str(row[key])
    if any(str(existing[key]) == value for existing in rows): raise ValueError(f"Duplicate {key}: {value}")
    result=[*rows,row]
    atomic_jsonl(path, sorted(result,key=lambda item:str(item[key])))
    return result


def ensure_layout(root: Path) -> Path:
    root=root.resolve()
    for relative in ("artifacts/gaussian_images","tables","figures","progress","progress/smoke/artifacts/gaussian_images","progress/smoke/tables","progress/smoke/figures"):
        (root/relative).mkdir(parents=True,exist_ok=True)
    return root


def tree_hashes(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)):sha256_file(path) for path in sorted(root.rglob("*")) if path.is_file()}


__all__=["atomic_bytes","atomic_csv","atomic_json","atomic_jsonl","canonical_hash","canonical_json","ensure_layout","load_csv","load_jsonl","sha256_bytes","sha256_file","tree_hashes","upsert"]
