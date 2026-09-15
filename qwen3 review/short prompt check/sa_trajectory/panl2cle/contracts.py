from __future__ import annotations

from typing import Any

from SA_trajectory.PANL2CLE.contracts import (
    atomic_bf16_npz,
    atomic_json,
    atomic_jsonl,
    canonical_hash,
    load_bf16,
    load_jsonl,
    sha256_file,
)

from .config import ALPHAS, PAIRS, PANL_LAYERS


def physical_trial_key(row: dict[str, Any]) -> str:
    return "|".join(map(str, (
        row["case_id"], row["condition"], row.get("panl_layer"),
        row.get("cle_layer"), float(row.get("alpha", 0.0)),
    )))


def expected_physical_count(case_count: int) -> int:
    # C0 plus C1/C2/C3 for every immediate pair and signed alpha.
    return int(case_count) * (1 + 3 * len(PAIRS) * len(ALPHAS))


def expected_logical_count(case_count: int) -> int:
    return int(case_count) * len(PAIRS) * len(ALPHAS) * 4


def validate_layer_design() -> None:
    expected = tuple(zip(PANL_LAYERS, (15, 17, 19)))
    if PAIRS != expected or any(cle != panl + 1 for panl, cle in PAIRS):
        raise ValueError("Only PANL-to-immediate-next-CLE pairs are allowed")
