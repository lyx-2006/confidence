#!/usr/bin/env python3
"""Build a V2 five-bin (0--100) text-entropy pool.

The legacy confidence PoolBuilder lives in legacy_pool_builder.py and remains
importable here for old experiments, but the CLI uses the shared V2 Qwen batch
runtime and never runs Stage 2 confidence.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import random
import re
import sys
import tempfile
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


ROOT_DIR = Path(__file__).resolve().parents[3]  # data_generation/legacy/generate_color_pool -> repo root
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# The V2 CLI section below imports generation_runtime/generation_v2, which now
# live one level above the legacy folder.
GENERATION_DIR = ROOT_DIR / "data_generation"
if str(GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATION_DIR))


from legacy_pool_builder import PoolBuilder  # noqa: E402

COLOR_SET_A = [
    "red", "orange", "yellow", "green", "blue", "cyan",
    "purple", "pink", "brown", "white", "black", "gray",
]
# Only the first vocabulary is active for colour-pool generation; the legacy
# second vocabulary (COLOR_SET_B) moved to legacy_pool_builder.py.
ALL_COLORS = list(COLOR_SET_A)

# V2 uses explicit 0--100 entropy ranges; the legacy 0--1 confidence ranges
# live in legacy_pool_builder.py as LEGACY_BIN_RANGES.
BIN_RANGES = [(0.0, 20.0), (20.0, 40.0), (40.0, 60.0), (60.0, 80.0), (80.0, 100.0)]


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().strip()
    value = re.sub(r"\s+", " ", value)
    return re.sub(r"[^\w\s]", "", value)


def normalize_answer(value: str) -> str:
    return normalize_text(value).strip()


BIN_SELECTOR_ALIASES = {
    "0": 0, "bin0": 0, "0.0-0.2": 0,
    "1": 1, "bin1": 1, "0.2-0.4": 1,
    "2": 2, "bin2": 2, "0.4-0.6": 2,
    "3": 3, "bin3": 3, "0.6-0.8": 3,
    "4": 4, "bin4": 4, "0.8-1.0": 4,
    "0-20": 0, "20-40": 1, "40-60": 2, "60-80": 3, "80-100": 4,
}


def parse_select_pool(value: str) -> list[int]:
    normalized = value.strip().lower().replace(" ", "")
    if normalized == "all":
        return list(range(5))
    parts = [part for part in normalized.split(",") if part]
    if not parts:
        raise argparse.ArgumentTypeError("--select_pool requires at least one bin")

    selected: set[int] = set()
    boundaries = {0.0, 20.0, 40.0, 60.0, 80.0, 100.0}
    invalid: list[str] = []
    for part in parts:
        if part in BIN_SELECTOR_ALIASES:
            selected.add(BIN_SELECTOR_ALIASES[part])
            continue
        match = re.fullmatch(r"(\d+(?:\.\d+)?)\-(\d+(?:\.\d+)?)", part)
        if not match:
            invalid.append(part)
            continue
        low, high = float(match.group(1)), float(match.group(2))
        if low not in boundaries or high not in boundaries or low >= high:
            invalid.append(part)
            continue
        covered = [
            bin_id for bin_id, (bin_low, bin_high) in enumerate(BIN_RANGES)
            if bin_low >= low and bin_high <= high
        ]
        if not covered:
            invalid.append(part)
            continue
        selected.update(covered)
    if invalid:
        choices = "bin IDs (0-4) or ranges aligned to 0,20,40,60,80,100"
        raise argparse.ArgumentTypeError(f"Unknown bin selector(s): {invalid}; use {choices}")
    return sorted(selected)


def parse_csv_ints(value: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from exc
    if len(values) != 5 or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("Exactly five positive batch sizes are required")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate V2 five-bin text-entropy colour pools")
    parser.add_argument("--find", action="store_true", help="Deprecated legacy flag; V2 always validates before queuing")
    parser.add_argument("--after", choices=ALL_COLORS, help="Start after this colour in the global order")
    parser.add_argument("--round", dest="rounds", type=int, default=5, help="Maximum generator/analyzer rounds per colour")
    parser.add_argument("--input", default="/root/autodl-tmp/datasets/dataset.json")
    parser.add_argument("--output", default=str(ROOT_DIR / "generation_v2_outputs/formal/text/text_entropy_pool.json"))
    parser.add_argument("--target-per-bin", type=int, default=5)
    parser.add_argument(
        "--select_pool", "--select-pool",
        type=parse_select_pool,
        default=list(range(5)),
        help="Entropy bins to generate: e.g. 0,1,4 or ranges such as 40-80",
    )
    parser.add_argument("--bin-batch-sizes", type=parse_csv_ints, default=parse_csv_ints("20,20,20,20,20"))
    parser.add_argument("--deepseek-workers", type=int, default=27)
    parser.add_argument(
        "--color-workers",
        type=int,
        default=6,
        help="Number of colors generated concurrently (1-6); local Qwen evaluation remains serial",
    )
    parser.add_argument("--colors", help="Comma-separated subset of the 24 supported colours")
    parser.add_argument("--resume", action="store_true", help="Explicit alias for the default incremental-create behavior")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.88)
    parser.add_argument("--stability-threshold", type=float, default=0.1)
    parser.add_argument("--model-path", default=str(ROOT_DIR / "qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct"))
    parser.add_argument("--api-config-path", default=str(ROOT_DIR / "api_config.json"))
    parser.add_argument("--qwen-batch-size", type=int, default=4)
    parser.add_argument("--qwen-batch-wait-ms", type=int, default=500)
    parser.add_argument("--qwen-wait-timeout", type=float, default=86400.0)
    args = parser.parse_args(argv)

    if args.rounds <= 0 or args.target_per_bin <= 0 or args.deepseek_workers <= 0 or args.color_workers <= 0:
        parser.error("--round, --target-per-bin, --deepseek-workers, and --color-workers must be positive")
    if args.color_workers > 6:
        parser.error("--color-workers cannot exceed 6")
    if not 0.0 <= args.near_duplicate_threshold <= 1.0:
        parser.error("--near-duplicate-threshold must be between 0 and 1")
    if args.stability_threshold <= 0.0:
        parser.error("--stability-threshold must be positive")
    if not 1 <= args.qwen_batch_size <= 64 or args.qwen_batch_wait_ms < 0:
        parser.error("--qwen-batch-size must be 1-64 and --qwen-batch-wait-ms non-negative")
    if args.qwen_wait_timeout <= 0:
        parser.error("--qwen-wait-timeout must be positive")
    selected = list(ALL_COLORS)
    if args.after:
        selected = selected[ALL_COLORS.index(args.after) + 1:]
    if args.colors:
        requested = [normalize_answer(part) for part in args.colors.split(",") if part.strip()]
        invalid = [color for color in requested if color not in ALL_COLORS]
        if invalid:
            parser.error(f"Unsupported --colors values: {', '.join(invalid)}")
        requested_set = set(requested)
        selected = [color for color in selected if color in requested_set]
    if not selected:
        parser.error("No colours remain after applying --after and --colors")
    # V2 serializes DeepSeek calls through the producer and shares one Qwen
    # scheduler; the legacy worker lower-bound no longer applies.
    args.selected_colors = selected
    args.selected_bins = list(args.select_pool)
    args.batch_sizes = dict(enumerate(args.bin_batch_sizes))
    return args


def _run_v2(args: argparse.Namespace) -> None:
    """Standalone V2 text producer used by this historical CLI entrypoint.

    The legacy ``PoolBuilder`` remains importable for old experiments, but is
    intentionally not called here: V2 has no confidence-analysis stage.
    """
    from generation_runtime import PersistentQwenQueue, QwenBatchScheduler, ensure_isolated_root
    from generation_v2 import TextEntropyProducer

    output_path = Path(args.output).expanduser().resolve()
    run_root = output_path.parent.parent
    ensure_isolated_root(
        run_root,
        (
            ROOT_DIR / "datasets",
            ROOT_DIR / "data_generation" / "legacy" / "generate_color_pool" / "output",
            ROOT_DIR / "data_generation" / "legacy" / "generate_dataset" / "datasets",
        ),
    )
    formal_root = ROOT_DIR / "generation_v2_outputs" / "formal"
    if run_root == formal_root and run_root.exists() and any(run_root.iterdir()) and not args.resume:
        raise ValueError(f"V2 output root already contains files; use --resume: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    # Same transient queue location as the joint pipeline: the run root keeps
    # only final artifacts, and resume of the same run root reuses the queue.
    transient_dir = (
        Path(tempfile.gettempdir())
        / "v2-generation"
        / hashlib.sha256(str(run_root.resolve()).encode("utf-8")).hexdigest()[:12]
    )
    transient_dir.mkdir(parents=True, exist_ok=True)
    queue = PersistentQwenQueue(transient_dir / "qwen_jobs.json")
    from confidence_test.inference_extension import ExtendedQwenVLInference
    inference = ExtendedQwenVLInference(model_path=str(Path(args.model_path).expanduser().resolve()))
    import threading
    stop = threading.Event()
    error: list[BaseException] = []

    def consume() -> None:
        try:
            QwenBatchScheduler(queue, inference, args.qwen_batch_size, args.qwen_batch_wait_ms).run(stop)
        except BaseException as exc:
            queue.fail_unfinished({"type": type(exc).__name__, "message": str(exc)})
            error.append(exc)

    worker = threading.Thread(target=consume, name="qwen-v2-text-scheduler", daemon=True)
    worker.start()
    try:
        producer = TextEntropyProducer(
            input_path=Path(args.input),
            output_path=output_path,
            queue=queue,
            api_config_path=Path(args.api_config_path),
            selected_colors=args.selected_colors,
            selected_bins=args.selected_bins,
            target_per_bin=args.target_per_bin,
            rounds=args.rounds,
            batch_sizes=args.batch_sizes,
            near_duplicate_threshold=args.near_duplicate_threshold,
            qwen_timeout=args.qwen_wait_timeout,
        )
        producer.run()
    finally:
        stop.set()
        worker.join(timeout=120)
        if worker.is_alive():
            queue.fail_unfinished({"type": "SchedulerShutdown", "message": "scheduler did not stop"})
            raise RuntimeError("Qwen scheduler did not stop")
    if error:
        raise RuntimeError(f"Qwen scheduler failed: {error[0]}")


def main() -> int:
    args = parse_args()
    try:
        _run_v2(args)
    except KeyboardInterrupt:
        print("[WARN] Interrupted; accepted priors written before the interruption were preserved.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    print(f"[INFO] Colour prior pool updated at {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
