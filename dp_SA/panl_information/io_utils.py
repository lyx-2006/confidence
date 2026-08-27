from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .config import RESULTS_ROOT


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
    payload = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    _atomic(Path(path), lambda handle: handle.write(payload))


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


def ensure_output_layout(root: str | Path = RESULTS_ROOT, *, resume: bool, formal: bool = True) -> Path:
    resolved = Path(root).resolve()
    if formal and resolved != RESULTS_ROOT.resolve():
        raise ValueError(f"Formal output root is fixed: {RESULTS_ROOT.resolve()}")
    if resolved.exists() and any(resolved.iterdir()) and not resume:
        raise FileExistsError(f"Output already exists; pass --resume: {resolved}")
    for name in ("progress", "tables", "figures", "artifacts", "artifacts/hidden", "artifacts/probe_models"):
        (resolved / name).mkdir(parents=True, exist_ok=True)
    return resolved


def safe_remove_temp_tree(path: str | Path, allowed: Iterable[str | Path]) -> None:
    import shutil

    target = Path(path).resolve()
    allowed_paths = {Path(value).resolve() for value in allowed}
    if target not in allowed_paths:
        raise ValueError(f"Refusing to delete non-temporary path: {target}")
    if target.is_dir():
        shutil.rmtree(target)


def assert_fingerprint(path: str | Path, payload: dict[str, Any], *, resume: bool) -> str:
    fingerprint = canonical_hash(payload)
    destination = Path(path)
    if destination.exists():
        old = json.loads(destination.read_text(encoding="utf-8"))
        if old.get("fingerprint") != fingerprint:
            raise ValueError("Fingerprint mismatch; refusing resume")
        if not resume:
            raise FileExistsError(f"Run config exists; pass --resume: {destination}")
    else:
        atomic_json(destination, {**payload, "fingerprint": fingerprint})
    return fingerprint


def stage_update(root: Path, stage: str, status: str, **extra: Any) -> None:
    atomic_json(root / "progress" / "stage_status.json", {"stage": stage, "status": status, **extra})
    atomic_json(root / "progress" / "progress.json", {"stage": stage, "status": status, **extra})
