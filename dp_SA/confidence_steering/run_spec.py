from __future__ import annotations

import argparse
import math
from typing import Any, Sequence

from .config import ALPHAS, DIRECTIONS, STEERING_LAYERS
from .io_utils import canonical_hash

NATURAL_DIRECTIONS = (
    "confidence_parallel_sa",
    "confidence_perp_sa_natural_scale",
)
ALL_DIRECTIONS = DIRECTIONS[:-1] + NATURAL_DIRECTIONS + (DIRECTIONS[-1],)
SHUFFLE_DIRECTION = "within_answer_shuffled_perp_difficulty_sa"


def normalize_run_spec(
    directions: Sequence[str] | None = None,
    layers: Sequence[int] | None = None,
    alphas: Sequence[float] | None = None,
) -> dict[str, Any]:
    raw_directions = list(DIRECTIONS if directions is None else directions)
    raw_layers = list(STEERING_LAYERS if layers is None else layers)
    raw_alphas = list(ALPHAS if alphas is None else alphas)
    if not raw_directions or not raw_layers or not raw_alphas:
        raise ValueError("directions, layers, and alphas must be non-empty")
    if len(raw_directions) != len(set(raw_directions)):
        raise ValueError("Duplicate direction in run spec")
    unknown = set(raw_directions) - set(ALL_DIRECTIONS)
    if unknown:
        raise ValueError(f"Unknown direction(s): {sorted(unknown)}")
    parsed_layers = [int(value) for value in raw_layers]
    if len(parsed_layers) != len(set(parsed_layers)):
        raise ValueError("Duplicate layer in run spec")
    if not set(parsed_layers) <= set(STEERING_LAYERS):
        raise ValueError(f"Layers must be selected from {STEERING_LAYERS}")
    parsed_alphas = [float(value) for value in raw_alphas]
    if len(parsed_alphas) != len(set(parsed_alphas)):
        raise ValueError("Duplicate alpha in run spec")
    if not all(math.isfinite(value) for value in parsed_alphas):
        raise ValueError("All alphas must be finite")
    if 0.0 not in parsed_alphas:
        raise ValueError("Run spec must include alpha=0 for parity")
    selected = tuple(direction for direction in ALL_DIRECTIONS if direction in raw_directions)
    layer_values = tuple(sorted(parsed_layers))
    alpha_values = tuple(sorted(parsed_alphas))
    alpha_set = set(alpha_values)
    paired = tuple(value for value in alpha_values if value > 0 and -value in alpha_set)
    is_default = selected == tuple(DIRECTIONS) and layer_values == tuple(STEERING_LAYERS) and alpha_values == tuple(ALPHAS)
    payload = {
        "directions": list(selected),
        "layers": list(layer_values),
        "alphas": list(alpha_values),
        "paired_doses": list(paired),
        "shuffle_requested": SHUFFLE_DIRECTION in selected,
        "analysis_kind": "mechanism_diagnostic" if any(d in selected for d in NATURAL_DIRECTIONS) else "legacy_confirmatory" if is_default else "diagnostic",
        "is_default": is_default,
    }
    payload["fingerprint"] = canonical_hash(payload)
    return payload


def add_run_spec_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--directions", nargs="+")
    parser.add_argument("--layers", nargs="+", type=int)
    parser.add_argument("--alphas", nargs="+", type=float)


def run_spec_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return normalize_run_spec(args.directions, args.layers, args.alphas)


def run_spec_cli_args(spec: dict[str, Any]) -> list[str]:
    return [
        "--directions", *map(str, spec["directions"]),
        "--layers", *map(str, spec["layers"]),
        "--alphas", *map(str, spec["alphas"]),
    ]


def expected_runtime_counts(
    spec: dict[str, Any],
    case_count: int,
    *,
    null_repeats: int = 0,
) -> dict[str, int]:
    """Return canonical trial/forward counts for a formal (non-smoke) run."""
    layer_count = len(spec["layers"])
    direction_count = len(spec["directions"])
    nonzero_alpha_count = sum(float(alpha) != 0.0 for alpha in spec["alphas"])
    main_trials = case_count * layer_count * direction_count * len(spec["alphas"])
    main_forwards = case_count * (
        1 + layer_count + layer_count * direction_count * nonzero_alpha_count
    )
    null_layer_count = (1 if spec["is_default"] else layer_count) if spec["shuffle_requested"] else 0
    null_alpha_count = sum(
        float(alpha) != 0.0 and -float(alpha) in spec["alphas"]
        for alpha in spec["alphas"]
    )
    null_trials = case_count * int(null_repeats) * null_layer_count * null_alpha_count
    # Null trials share one clean forward per case with the main run.
    return {
        "main_trials": main_trials,
        "main_forwards": main_forwards,
        "null_trials": null_trials,
        "total_forwards": main_forwards + null_trials,
    }
