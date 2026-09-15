from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

from dp_SA.io_utils import atomic_json, canonical_hash

from .config import CAPTURE_LAYERS, POSITIONS


def parse_positions(values: Sequence[str]) -> tuple[str, ...]:
    output = tuple(str(value) for value in values)
    if not output or len(output) != len(set(output)):
        raise ValueError("--positions must be non-empty and unique")
    invalid = sorted(set(output) - set(POSITIONS))
    if invalid:
        raise ValueError(f"Unsupported Steering positions: {invalid}; choices={list(POSITIONS)}")
    return output


def parse_layers(values: Sequence[int]) -> tuple[int, ...]:
    output = tuple(int(value) for value in values)
    if not output or len(output) != len(set(output)):
        raise ValueError("--layers must be non-empty and unique")
    invalid = sorted(set(output) - set(CAPTURE_LAYERS))
    if invalid:
        raise ValueError(
            f"Steering layers must be zero-based indices in "
            f"[{CAPTURE_LAYERS[0]}, {CAPTURE_LAYERS[-1]}]: {invalid}"
        )
    return output


def parse_alphas(values: Sequence[float]) -> tuple[float, ...]:
    output = tuple(float(value) for value in values)
    if not output or len(output) != len(set(output)):
        raise ValueError("--alphas must be non-empty and unique")
    if not all(math.isfinite(value) for value in output):
        raise ValueError("--alphas must contain only finite values")
    return output


def hidden_key(position: str, layer: int) -> str:
    parsed_position = parse_positions((position,))[0]
    parsed_layer = parse_layers((layer,))[0]
    return f"{parsed_position}__L{parsed_layer}"


def all_capture_keys() -> tuple[str, ...]:
    return tuple(hidden_key(position, layer) for position in POSITIONS for layer in CAPTURE_LAYERS)


def ensure_fingerprinted_config(
    path: Path,
    payload: dict[str, Any],
    *,
    resume: bool,
    label: str,
) -> dict[str, Any]:
    config = dict(payload)
    config["fingerprint"] = canonical_hash(config)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("fingerprint") != config["fingerprint"]:
            raise ValueError(f"{label} config fingerprint changed; use a fresh output root")
        if not resume:
            raise FileExistsError(f"{label} output exists; use --resume")
    else:
        atomic_json(path, config)
    return config

