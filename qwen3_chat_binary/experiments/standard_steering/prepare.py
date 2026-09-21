from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from dp_SA.io_utils import atomic_json, atomic_jsonl, canonical_hash, sha256_file

from qwen3_chat_binary.config import (
    CAPTURE_ROOT,
    CONSTRUCTION_PER_SIDE,
    IMAGE_TEST_COUNT,
    STEERING_OUTPUT_ROOT,
    TEXT_TEST_COUNT,
    VARIANTS,
)
from qwen3_chat_binary.contracts import ensure_fingerprinted_config, parse_variants
from qwen3_chat_binary.layout import capture_config_path, capture_results_path, ensure_output_layout
from qwen3_chat_binary.steering import shared_manifests


def prepare_split(
    *, capture_root: Path = CAPTURE_ROOT,
    output_root: Path = STEERING_OUTPUT_ROOT,
    variants: Sequence[str] = VARIANTS,
    resume: bool = False,
) -> dict[str, Any]:
    capture_root, output_root = capture_root.resolve(), output_root.resolve()
    variants = parse_variants(variants)
    construction, test, selection = shared_manifests(
        capture_root, variants=variants, smoke=False
    )
    ensure_output_layout(output_root, variants)
    source_paths = [capture_config_path(capture_root)] + [
        capture_results_path(capture_root, variant) for variant in variants
    ]
    payload = {
        "format_version": 1,
        "experiment": "standard_steering_central_test_split",
        "capture_root": str(capture_root),
        "variants": list(variants),
        "construction_per_side": CONSTRUCTION_PER_SIDE,
        "test_per_side": {"text": TEXT_TEST_COUNT, "image": IMAGE_TEST_COUNT},
        "construction_rule": "clean SA extremes",
        "test_rule": "smallest absolute distance from clean SA 0.5 within each side",
        "source_sha256": {str(path): sha256_file(path) for path in source_paths},
        "construction_fingerprint": canonical_hash([row["case_id"] for row in construction]),
        "test_fingerprint": canonical_hash([row["case_id"] for row in test]),
    }
    config = ensure_fingerprinted_config(
        output_root / "progress" / "split_config.json",
        payload,
        resume=resume,
        label="Standard steering split",
    )
    atomic_jsonl(output_root / "tables" / "construction_manifest.jsonl", construction)
    atomic_jsonl(output_root / "tables" / "test_manifest.jsonl", test)
    text_scores = [float(row["image_attribution_score"]) for row in test if row["test_side"] == "text_side"]
    image_scores = [float(row["image_attribution_score"]) for row in test if row["test_side"] == "image_side"]
    summary = {
        **selection,
        "status": "complete",
        "config_fingerprint": config["fingerprint"],
        "construction_score_ranges": {
            side: [
                min(float(row["image_attribution_score"]) for row in construction if row["construction_side"] == side),
                max(float(row["image_attribution_score"]) for row in construction if row["construction_side"] == side),
            ]
            for side in ("high_text", "high_image")
        },
        "test_score_ranges": {
            "text_side": [min(text_scores), max(text_scores)],
            "image_side": [min(image_scores), max(image_scores)],
        },
        "test_distance_ranges": {
            "text_side": [min(0.5 - value for value in text_scores), max(0.5 - value for value in text_scores)],
            "image_side": [min(value - 0.5 for value in image_scores), max(value - 0.5 for value in image_scores)],
        },
    }
    atomic_json(output_root / "tables" / "selection_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare the standard Steering fixed case split")
    parser.add_argument("--capture-root", type=Path, default=CAPTURE_ROOT)
    parser.add_argument("--output-root", type=Path, default=STEERING_OUTPUT_ROOT)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(prepare_split(**vars(args)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
