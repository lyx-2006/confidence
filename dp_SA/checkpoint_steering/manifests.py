from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from dp_SA.selection import select_manifests

from .config import (
    HISTORICAL_CAPTURE,
    HISTORICAL_CONSTRUCTION,
    HISTORICAL_TEST,
    SEED,
)
from .io_utils import atomic_json, atomic_jsonl, canonical_hash, load_jsonl, sha256_file


def manifest_fingerprint(rows: Sequence[dict[str, Any]]) -> str:
    return canonical_hash([str(row["case_id"]) for row in rows])


def _validate(construction: Sequence[dict[str, Any]], test: Sequence[dict[str, Any]]) -> None:
    construction_items = [str(row["item_id"]) for row in construction]
    test_items = [str(row["item_id"]) for row in test]
    if len(construction) != 50 or Counter(row.get("construction_side") for row in construction) != Counter({"high_image": 25, "high_text": 25}):
        raise ValueError("Historical construction manifest is incomplete")
    if len(test) != 100 or Counter(row.get("test_side") for row in test) != Counter({"image_side": 50, "text_side": 50}):
        raise ValueError("Historical test manifest is incomplete")
    if len(set(construction_items)) != 50 or len(set(test_items)) != 100 or set(construction_items) & set(test_items):
        raise ValueError("Historical manifests have duplicate items or construction/test leakage")
    if any(int(row["argmax_hard_class"]) == 4 for row in test):
        raise ValueError("Historical test manifest contains class 4")


def _smoke_subset(construction: Sequence[dict[str, Any]], test: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    small_construction = [
        dict(row)
        for side in ("high_image", "high_text")
        for row in [value for value in construction if value["construction_side"] == side][:2]
    ]
    small_test = [
        dict(row)
        for side in ("image_side", "text_side")
        for row in [value for value in test if value["test_side"] == side][:2]
    ]
    if len(small_construction) != 4 or len(small_test) != 4:
        raise ValueError("Smoke manifests require two records per side")
    return small_construction, small_test


def _load_source() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    historical_construction = load_jsonl(HISTORICAL_CONSTRUCTION)
    historical_test = load_jsonl(HISTORICAL_TEST)
    try:
        _validate(historical_construction, historical_test)
        source = {
            "selection_source": "historical_frozen_steering_manifests",
            "construction_path": str(HISTORICAL_CONSTRUCTION.resolve()),
            "test_path": str(HISTORICAL_TEST.resolve()),
            "construction_sha256": sha256_file(HISTORICAL_CONSTRUCTION),
            "test_sha256": sha256_file(HISTORICAL_TEST),
        }
        return historical_construction, historical_test, source
    except (FileNotFoundError, ValueError):
        completed = [row for row in load_jsonl(HISTORICAL_CAPTURE) if row.get("status") == "completed"]
        construction, test, summary = select_manifests(completed, seed=SEED)
        _validate(construction, test)
        return construction, test, {
            "selection_source": "reselected_from_historical_capture",
            "capture_path": str(HISTORICAL_CAPTURE.resolve()),
            "capture_sha256": sha256_file(HISTORICAL_CAPTURE),
            "selection_summary": summary,
        }


def prepare_manifests(root: Path, *, smoke: bool, resume: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    directory = root / "artifacts" / "manifests"
    construction_path = directory / "construction_manifest.jsonl"
    test_path = directory / "test_manifest.jsonl"
    source_path = directory / "selection_source.json"
    if construction_path.exists() or test_path.exists() or source_path.exists():
        if not resume:
            raise FileExistsError("Checkpoint-steering manifests exist; use --resume")
        construction = load_jsonl(construction_path)
        test = load_jsonl(test_path)
        if not construction or not test or not source_path.is_file():
            raise ValueError("Frozen manifest artifacts are incomplete")
        source = __import__("json").loads(source_path.read_text())
        expected = source.get("frozen_fingerprints", {})
        if expected != {"construction": manifest_fingerprint(construction), "test": manifest_fingerprint(test)}:
            raise ValueError("Frozen manifest fingerprint mismatch")
        return construction, test, source

    construction, test, source = _load_source()
    if smoke:
        construction, test = _smoke_subset(construction, test)
    else:
        _validate(construction, test)
    source = {
        **source,
        "smoke": smoke,
        "seed": SEED,
        "construction_count": len(construction),
        "test_count": len(test),
        "construction_counts": dict(Counter(row["construction_side"] for row in construction)),
        "test_counts": dict(Counter(row["test_side"] for row in test)),
        "frozen_fingerprints": {
            "construction": manifest_fingerprint(construction),
            "test": manifest_fingerprint(test),
        },
    }
    atomic_jsonl(construction_path, construction)
    atomic_jsonl(test_path, test)
    atomic_json(source_path, source)
    return construction, test, source


def load_frozen_manifests(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import json

    directory = root / "artifacts" / "manifests"
    construction = load_jsonl(directory / "construction_manifest.jsonl")
    test = load_jsonl(directory / "test_manifest.jsonl")
    source_path = directory / "selection_source.json"
    if not construction or not test or not source_path.is_file():
        raise ValueError("Frozen checkpoint-steering manifests are missing")
    source = json.loads(source_path.read_text())
    actual = {"construction": manifest_fingerprint(construction), "test": manifest_fingerprint(test)}
    if source.get("frozen_fingerprints") != actual:
        raise ValueError("Frozen checkpoint-steering manifests changed")
    return construction, test, source
