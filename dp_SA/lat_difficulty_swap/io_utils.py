from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_hash(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _atomic(path: Path, writer: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            writer(handle)
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
    data = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    _atomic(Path(path), lambda handle: handle.write(data))


def atomic_text(path: str | Path, value: str) -> None:
    _atomic(Path(path), lambda handle: handle.write(value.encode("utf-8")))


def atomic_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    def write(handle: Any) -> None:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")
    _atomic(Path(path), write)


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("ab") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl(path: str | Path, *, repair_trailing: bool = False) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    raw = source.read_bytes()
    lines = raw.splitlines(keepends=True)
    rows: list[dict[str, Any]] = []
    valid_end = 0
    for index, line in enumerate(lines):
        if not line.strip():
            valid_end += len(line)
            continue
        try:
            rows.append(json.loads(line))
            valid_end += len(line)
        except json.JSONDecodeError:
            if not repair_trailing or index != len(lines) - 1:
                raise
            _atomic(source, lambda handle: handle.write(raw[:valid_end]))
            break
    return rows


def atomic_csv(path: str | Path, rows: Sequence[dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    destination = Path(path)
    chosen = list(fields or sorted({key for row in rows for key in row}))
    def write(handle: Any) -> None:
        import io
        text = io.TextIOWrapper(handle, encoding="utf-8", newline="", write_through=True)
        writer = csv.DictWriter(text, fieldnames=chosen, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        text.detach()
    _atomic(destination, write)


def ensure_layout(root: str | Path, *, resume: bool, allow_existing_upstream: bool = False) -> Path:
    resolved = Path(root).resolve()
    if resolved.exists() and any(resolved.iterdir()) and not resume and not allow_existing_upstream:
        raise FileExistsError(f"Output exists; pass --resume: {resolved}")
    for name in ("figures", "tables", "progress", "artifacts", "artifacts/hidden", "artifacts/probe_models", "artifacts/probe_predictions"):
        (resolved / name).mkdir(parents=True, exist_ok=True)
    return resolved


def stage_update(root: Path, stage: str, status: str, **extra: Any) -> None:
    payload = {"stage": stage, "status": status, **extra}
    atomic_json(root / "progress" / "stage_status.json", payload)
    atomic_json(root / "progress" / "progress.json", payload)
