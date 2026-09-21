from __future__ import annotations

from pathlib import Path
from typing import Sequence


SECTIONS = ("progress", "figures", "tables")


def ensure_output_layout(root: Path, variants: Sequence[str]) -> None:
    """Create the common experiment and per-variant output directories."""
    for section in SECTIONS:
        (root / section).mkdir(parents=True, exist_ok=True)
    for variant in variants:
        for section in SECTIONS:
            (root / variant / section).mkdir(parents=True, exist_ok=True)


def capture_config_path(root: Path) -> Path:
    return root / "progress" / "config.json"


def capture_phase0_path(root: Path) -> Path:
    return root / "tables" / "phase0_results.jsonl"


def capture_results_path(root: Path, variant: str) -> Path:
    return root / variant / "tables" / "results.jsonl"


def steering_predictions_path(root: Path, variant: str) -> Path:
    return root / variant / "tables" / "predictions.jsonl"
