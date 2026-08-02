"""Small JSON/path helpers shared by Probe CLIs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

from layer_metacognition.hidden_state_store import atomic_write_text


def probe_output_dir(experiment_dir: str | Path) -> Path:
    return Path(experiment_dir).resolve() / "probe"


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"JSONL file does not exist: {source}")
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {source}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record at {source}:{line_number} is not an object")
            yield value


def load_optional_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    return list(iter_jsonl(source)) if source.is_file() else []


def atomic_write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    lines = [
        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        for record in records
    ]
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def atomic_write_keyed_jsonl(
    path: str | Path,
    records: dict[tuple[Any, ...], dict[str, Any]],
    *,
    sort_key: Callable[[tuple[Any, ...]], Any] | None = None,
) -> None:
    keys = sorted(records, key=sort_key)
    atomic_write_jsonl(path, (records[key] for key in keys))


def sortable_item_id(value: Any) -> tuple[int, int | str]:
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value
