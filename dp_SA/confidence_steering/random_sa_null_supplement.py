from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .analyze import _mean, family_draws, symmetric_effect
from .config import BOOTSTRAP_REPEATS, SEED
from .io_utils import atomic_csv, atomic_json, canonical_hash, load_jsonl, sha256_file
from .random_sa_null import (
    FORMAL_REPEATS, GROUPS, NATURAL_FORMAL_ROOT, RANDOM_NULL_DOSE,
    RANDOM_NULL_LAYER, _null_symmetric_effect, protected_hashes,
)

TRUE_DIRECTION = "confidence_perp_sa_natural_scale"
CONFIDENCE_ENDPOINT = "delta_confidence_LAT_immediate"
SA_ENDPOINTS = ("delta_panl_probe_sa", "delta_final_soft_sa")


def _true_effect(main: Sequence[dict[str, Any]], endpoint: str) -> list[dict[str, Any]]:
    selected = [
        row for row in main
        if row["direction"] == TRUE_DIRECTION
        and int(row["layer"]) == RANDOM_NULL_LAYER
        and abs(float(row["alpha"])) == RANDOM_NULL_DOSE
    ]
    return symmetric_effect(selected, endpoint, RANDOM_NULL_DOSE)


def _by_family(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output[str(row["family_id"])].append(row)
    return output


def _draw_sample(index: dict[str, list[dict[str, Any]]], draw: Sequence[str]) -> list[dict[str, Any]]:
    return [row for family in draw for row in index.get(str(family), [])]


def supplemental_rows(
    main: Sequence[dict[str, Any]], null: Sequence[dict[str, Any]],
    draws: Sequence[Sequence[str]], repeats: int = FORMAL_REPEATS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    true = {endpoint: _true_effect(main, endpoint) for endpoint in (CONFIDENCE_ENDPOINT, *SA_ENDPOINTS)}
    null_effects = {
        endpoint: {
            replicate: _null_symmetric_effect(null, endpoint, replicate, RANDOM_NULL_DOSE)
            for replicate in range(1, repeats + 1)
        }
        for endpoint in (CONFIDENCE_ENDPOINT, *SA_ENDPOINTS)
    }
    case_count = len(true[CONFIDENCE_ENDPOINT])
    if case_count == 0 or any(len(rows) != case_count for rows in true.values()):
        raise ValueError("Incomplete true effects for supplemental analysis")
    if any(len(rows) != case_count for endpoint in null_effects.values() for rows in endpoint.values()):
        raise ValueError("Incomplete null effects for supplemental analysis")

    true_indexes = {endpoint: _by_family(rows) for endpoint, rows in true.items()}
    null_indexes = {
        endpoint: {replicate: _by_family(rows) for replicate, rows in per_rep.items()}
        for endpoint, per_rep in null_effects.items()
    }
    normalized: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    for group in GROUPS:
        true_confidence = _mean(true[CONFIDENCE_ENDPOINT], "effect", group)
        null_confidence = [
            _mean(null_effects[CONFIDENCE_ENDPOINT][replicate], "effect", group)
            for replicate in range(1, repeats + 1)
        ]
        if not math.isfinite(true_confidence) or true_confidence == 0 or any(not math.isfinite(value) or value == 0 for value in null_confidence):
            raise ValueError(f"Invalid LAT-confidence denominator for {group}")
        for endpoint in SA_ENDPOINTS:
            true_sa = _mean(true[endpoint], "effect", group)
            null_sa = [
                _mean(null_effects[endpoint][replicate], "effect", group)
                for replicate in range(1, repeats + 1)
            ]
            replicate_ratios = [sa / confidence for sa, confidence in zip(null_sa, null_confidence, strict=True)]
            null_mean_sa = float(np.mean(null_sa)); null_mean_confidence = float(np.mean(null_confidence))
            normalized.append({
                "endpoint": endpoint, "group": group, "dose": RANDOM_NULL_DOSE,
                "true_sa_effect": true_sa, "true_lat_confidence_effect": true_confidence,
                "true_sa_per_lat_confidence": true_sa / true_confidence,
                "null_mean_sa_effect": null_mean_sa,
                "null_mean_lat_confidence_effect": null_mean_confidence,
                "null_mean_sa_per_lat_confidence": null_mean_sa / null_mean_confidence,
                "null_replicate_ratio_mean": float(np.mean(replicate_ratios)),
                "null_replicate_ratio_sd": float(np.std(replicate_ratios, ddof=1)),
                "null_replicate_ratio_min": min(replicate_ratios),
                "null_replicate_ratio_max": max(replicate_ratios),
                "null_repeats": repeats, "case_count": case_count,
            })

            observed = true_sa - null_mean_sa
            boot = []
            for draw in draws:
                true_sample = _draw_sample(true_indexes[endpoint], draw)
                true_value = _mean(true_sample, "effect", group)
                null_values = [
                    _mean(_draw_sample(null_indexes[endpoint][replicate], draw), "effect", group)
                    for replicate in range(1, repeats + 1)
                ]
                value = true_value - float(np.mean(null_values))
                if math.isfinite(value):
                    boot.append(value)
            if len(boot) != len(draws):
                raise ValueError(f"Invalid paired bootstrap draws for {endpoint}/{group}")
            low, high = np.percentile(boot, [2.5, 97.5])
            contrasts.append({
                "endpoint": endpoint, "group": group, "dose": RANDOM_NULL_DOSE,
                "contrast_definition": "S2_true-minus-mean_global_null",
                "true_effect": true_sa, "mean_null_effect": null_mean_sa,
                "paired_contrast": observed,
                "bootstrap_sem": float(np.std(boot, ddof=1)),
                "ci95_low": float(low), "ci95_high": float(high),
                "ci_excludes_zero": bool(high < 0 or low > 0),
                "bootstrap_repeats": len(boot), "null_repeats": repeats,
                "case_count": case_count,
            })
    return normalized, contrasts


def run_supplement(root: Path = NATURAL_FORMAL_ROOT) -> dict[str, Any]:
    root = root.resolve()
    main_path = root / "artifacts/trials/main_trials.jsonl"
    null_path = root / "artifacts/trials/random_sa_subspace_null_trials.jsonl"
    run = json.loads((root / "progress/random_sa_subspace_null_run.json").read_text())
    if run.get("status") != "complete" or int(run.get("null_trial_count", -1)) != 4000:
        raise ValueError("Formal 20-null run is not complete")
    before = protected_hashes(root)
    main = load_jsonl(main_path); null = load_jsonl(null_path)
    true_rows = [
        row for row in main
        if row["direction"] == TRUE_DIRECTION
        and int(row["layer"]) == RANDOM_NULL_LAYER
        and abs(float(row["alpha"])) == RANDOM_NULL_DOSE
    ]
    draws, draw_fingerprint = family_draws(true_rows, BOOTSTRAP_REPEATS, SEED)
    expected_fingerprint = json.loads((root / "progress/random_sa_subspace_null_analyze.json").read_text())["bootstrap_draw_fingerprint"]
    if draw_fingerprint != expected_fingerprint:
        raise ValueError("Supplement does not reproduce the frozen shared family bootstrap draws")
    normalized, contrasts = supplemental_rows(main, null, draws)
    normalized_path = root / "tables/random_sa_subspace_null_unit_confidence.csv"
    contrast_path = root / "tables/random_sa_subspace_null_paired_family_contrast.csv"
    atomic_csv(normalized_path, normalized)
    atomic_csv(contrast_path, contrasts)
    after = protected_hashes(root)
    if after != before:
        raise ValueError("Protected main artifacts changed during supplemental analysis")
    result = {
        "status": "complete", "gpu_forwards": 0,
        "main_trials_sha256": sha256_file(main_path),
        "null_trials_sha256": sha256_file(null_path),
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "bootstrap_draw_fingerprint": draw_fingerprint,
        "normalized_rows": len(normalized), "contrast_rows": len(contrasts),
        "normalized_table": str(normalized_path), "paired_contrast_table": str(contrast_path),
        "protected_artifacts_unchanged": True,
    }
    result["fingerprint"] = canonical_hash(result)
    atomic_json(root / "progress/random_sa_subspace_null_supplement.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=NATURAL_FORMAL_ROOT)
    args = parser.parse_args(argv)
    print(json.dumps(run_supplement(args.output_root), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
