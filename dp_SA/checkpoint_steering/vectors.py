from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import LAYERS, POSITIONS, SMOKE_LAYERS, VECTOR_NORM_FRACTION
from .io_utils import array_hash, atomic_json, atomic_npz, canonical_hash, sha256_file
from .manifests import manifest_fingerprint


def construct_direction(high: np.ndarray, low: np.ndarray, *, fraction: float = VECTOR_NORM_FRACTION) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    high32 = np.asarray(high, dtype=np.float32)
    low32 = np.asarray(low, dtype=np.float32)
    if high32.ndim != 2 or low32.ndim != 2 or high32.shape[1:] != low32.shape[1:] or not len(high32) or not len(low32):
        raise ValueError("Direction inputs must be non-empty [samples, hidden] arrays with matching hidden size")
    if not np.isfinite(high32).all() or not np.isfinite(low32).all():
        raise ValueError("Direction inputs contain non-finite values")
    high_mean = high32.mean(axis=0, dtype=np.float32)
    low_mean = low32.mean(axis=0, dtype=np.float32)
    raw = high_mean - low_mean
    raw_norm = float(np.linalg.norm(raw))
    combined = np.concatenate([high32, low32], axis=0)
    mean_residual_norm = float(np.linalg.norm(combined, axis=1).mean())
    target_norm = float(fraction * mean_residual_norm)
    if not all(math.isfinite(value) and value > 0 for value in (raw_norm, mean_residual_norm, target_norm)):
        raise ValueError("Direction norm is invalid")
    scaled = np.asarray(raw / raw_norm * target_norm, dtype=np.float32)
    arrays = {"raw_vector": raw, "scaled_vector": scaled, "high_mean": high_mean, "low_mean": low_mean}
    metadata = {
        "raw_norm": raw_norm,
        "target_norm": target_norm,
        "scaled_norm": float(np.linalg.norm(scaled)),
        "mean_residual_norm": mean_residual_norm,
        "high_mean_norm": float(np.linalg.norm(high_mean)),
        "low_mean_norm": float(np.linalg.norm(low_mean)),
        "normalization_fraction": float(fraction),
    }
    return arrays, metadata


def _hidden(root: Path, clean_by_case: dict[str, dict[str, Any]], row: dict[str, Any], position: str, layer: int) -> np.ndarray:
    clean = clean_by_case[str(row["case_id"])]
    path = root / clean["hidden_file"]
    with np.load(path) as payload:
        return np.asarray(payload[f"{position}__L{layer}"], dtype=np.float32)


def build_vectors(root: Path, construction: Sequence[dict[str, Any]], clean_rows: Sequence[dict[str, Any]], *, smoke: bool, resume: bool) -> dict[str, Any]:
    directory = root / "artifacts" / "vectors"
    metadata_path = directory / "vector_metadata.json"
    layers = SMOKE_LAYERS if smoke else LAYERS
    clean_by_case = {str(row["case_id"]): row for row in clean_rows if row.get("status") == "completed"}
    expected_cases = {str(row["case_id"]) for row in construction}
    if not expected_cases.issubset(clean_by_case):
        raise ValueError("Construction clean capture is incomplete")
    if metadata_path.exists():
        if not resume:
            raise FileExistsError("Direction vectors exist; use --resume")
        import json
        metadata = json.loads(metadata_path.read_text())
        for cell in metadata["vectors"]:
            path = root / cell["vector_file"]
            if not path.is_file() or sha256_file(path) != cell["file_sha256"]:
                raise ValueError(f"Vector artifact fingerprint mismatch: {path}")
        return {**metadata, "resumed_noop": True}

    high = [row for row in construction if row["construction_side"] == "high_image"]
    low = [row for row in construction if row["construction_side"] == "high_text"]
    cells = []
    for position in POSITIONS:
        for layer in layers:
            h = np.stack([_hidden(root, clean_by_case, row, position, layer) for row in high])
            l = np.stack([_hidden(root, clean_by_case, row, position, layer) for row in low])
            arrays, metrics = construct_direction(h, l)
            relative = Path("artifacts") / "vectors" / f"{position}__L{layer}.npz"
            path = root / relative
            atomic_npz(path, arrays)
            vector_fingerprint = canonical_hash({name: array_hash(value) for name, value in arrays.items()} | metrics)
            cells.append({
                "position": position,
                "layer": int(layer),
                "vector_file": str(relative),
                "file_sha256": sha256_file(path),
                "vector_fingerprint": vector_fingerprint,
                **metrics,
            })
    metadata = {
        "format_version": 1,
        "construction_fingerprint": manifest_fingerprint(construction),
        "vectors": cells,
    }
    metadata["fingerprint"] = canonical_hash(metadata)
    atomic_json(metadata_path, metadata)
    return metadata


def load_scaled_vectors(root: Path, metadata: dict[str, Any]) -> dict[tuple[str, int], np.ndarray]:
    output: dict[tuple[str, int], np.ndarray] = {}
    for row in metadata["vectors"]:
        path = root / row["vector_file"]
        if sha256_file(path) != row["file_sha256"]:
            raise ValueError(f"Vector file changed: {path}")
        with np.load(path) as payload:
            output[(str(row["position"]), int(row["layer"]))] = np.asarray(payload["scaled_vector"], dtype=np.float32)
    return output

