from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

from dp_SA.io_utils import atomic_json, canonical_hash

from .config import CAPTURE_LAYERS, POSITIONS, VARIANTS


def parse_subset(values: Sequence[str], allowed: Sequence[str], label: str) -> tuple[str, ...]:
    parsed = tuple(values)
    if not parsed or len(parsed) != len(set(parsed)) or any(value not in allowed for value in parsed):
        raise ValueError(f"{label} must be a non-empty unique subset of {list(allowed)}")
    return parsed


def parse_positions(values: Sequence[str]) -> tuple[str, ...]:
    return parse_subset(values, POSITIONS, "--positions")


def parse_variants(values: Sequence[str]) -> tuple[str, ...]:
    return parse_subset(values, VARIANTS, "--variants")


def parse_layers(values: Sequence[int]) -> tuple[int, ...]:
    parsed = tuple(int(value) for value in values)
    if not parsed or len(parsed) != len(set(parsed)) or any(value not in CAPTURE_LAYERS for value in parsed):
        raise ValueError(f"--layers must be unique values in {CAPTURE_LAYERS[0]}..{CAPTURE_LAYERS[-1]}")
    return parsed


def parse_alphas(values: Sequence[float]) -> tuple[float, ...]:
    parsed = tuple(float(value) for value in values)
    if not parsed or len(parsed) != len(set(parsed)) or any(not math.isfinite(value) for value in parsed):
        raise ValueError("--alphas must be non-empty, unique, and finite")
    return parsed


def hidden_key(position: str, layer: int) -> str:
    return f"{position}__L{int(layer)}"


def all_hidden_keys() -> tuple[str, ...]:
    return tuple(hidden_key(position, layer) for position in POSITIONS for layer in CAPTURE_LAYERS)


def ensure_fingerprinted_config(path: Path, payload: dict[str, Any], *, resume: bool, label: str) -> dict[str, Any]:
    candidate = dict(payload)
    candidate["fingerprint"] = canonical_hash(candidate)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not resume:
            raise FileExistsError(f"{label} output already exists; use --resume or a new output root: {path}")
        if existing != candidate:
            raise ValueError(f"{label} resume fingerprint mismatch: {path}")
        return existing
    if resume:
        raise FileNotFoundError(f"{label} --resume requested but config does not exist: {path}")
    atomic_json(path, candidate)
    return candidate

