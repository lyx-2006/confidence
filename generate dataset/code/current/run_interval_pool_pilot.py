#!/usr/bin/env python3
"""Build a three-pair pilot pool for fixed normalized-entropy intervals."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import math
import os
import random
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
ROOT = PROJECT_DIR.parent
DEFAULT_ROOT = PROJECT_DIR / "datasets" / "legacy" / "experiments" / "interval_pool_pilot"
DEFAULT_MODEL = ROOT / "qwen-3-vl" / "model"
PAIRS = (("red", "circle"), ("white", "triangle"), ("black", "star"))
INTERVALS = (
    ("0.0-0.1", 0.0, 0.1),
    ("0.1-0.2", 0.1, 0.2),
    ("0.2-0.3", 0.2, 0.3),
    ("0.3-0.4", 0.3, 0.4),
    ("0.4-0.5", 0.4, 0.5),
    ("0.5-0.6", 0.5, 0.6),
    (">=0.6", 0.6, 1.0000000001),
)
HIGH_INTERVALS = {"0.5-0.6", ">=0.6"}


def _load(name: str, filename: str) -> Any:
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXP = _load("_interval_experiment_support", "explore_image_difficulty.py")
EXTREME = _load("_interval_extreme_support", "run_extreme_difficulty.py")
POOL = EXP.POOL
LEGACY = EXP.LEGACY


@dataclass(frozen=True)
class Construction:
    construction_id: str
    semantic_shape_count: int
    occlusion_target: float
    occluder_count: int
    blur_radius: float


CONSTRUCTIONS: dict[str, tuple[Construction, ...]] = {
    "0.0-0.1": (
        Construction("single_clear_b0", 1, 0.0, 0, 0.0),
        Construction("single_clear_b2", 1, 0.0, 0, 2.0),
        Construction("single_clear_b4", 1, 0.0, 0, 4.0),
        Construction("four_clear_b0", 4, 0.0, 0, 0.0),
        Construction("seven_clear_b2", 7, 0.0, 0, 2.0),
        Construction("ten_clear_b4", 10, 0.0, 0, 4.0),
    ),
    "0.1-0.2": (
        Construction("four_clear_b32", 4, 0.0, 0, 32.0),
        Construction("ten_occ60_o2_b12", 10, 0.60, 2, 12.0),
        Construction("ten_occ20_o1_b16", 10, 0.20, 1, 16.0),
        Construction("ten_occ30_o1_b16", 10, 0.30, 1, 16.0),
    ),
    "0.2-0.3": (
        Construction("four_occ50_o1_b16", 4, 0.50, 1, 16.0),
        Construction("ten_occ60_o2_b8", 10, 0.60, 2, 8.0),
        Construction("twelve_occ80_o4_b8", 12, 0.80, 4, 8.0),
    ),
    "0.3-0.4": (
        Construction("twelve_occ80_o4_b8", 12, 0.80, 4, 8.0),
        Construction("ten_occ70_o3_b12", 10, 0.70, 3, 12.0),
        Construction("four_occ60_o1_b12", 4, 0.60, 1, 12.0),
    ),
    "0.4-0.5": (
        Construction("eleven_occ80_o4_b4", 11, 0.80, 4, 4.0),
        Construction("eleven_occ80_o4_b12", 11, 0.80, 4, 12.0),
        Construction("eleven_occ80_o4_b16", 11, 0.80, 4, 16.0),
        Construction("ten_occ60_o1_b12", 10, 0.60, 1, 12.0),
    ),
    "0.5-0.6": (
        Construction("ten_occ80_o4_b12", 10, 0.80, 4, 12.0),
        Construction("eleven_occ80_o4_b12", 11, 0.80, 4, 12.0),
        Construction("twelve_occ80_o4_b12", 12, 0.80, 4, 12.0),
        Construction("ten_occ30_o1_b32", 10, 0.30, 1, 32.0),
    ),
    ">=0.6": (
        Construction("ten_occ60_o2_b16", 10, 0.60, 2, 16.0),
        Construction("ten_occ70_o3_b16", 10, 0.70, 3, 16.0),
        Construction("ten_occ80_o4_b16", 10, 0.80, 4, 16.0),
        Construction("eleven_occ80_o4_b8", 11, 0.80, 4, 8.0),
        Construction("twelve_occ80_o4_b16", 12, 0.80, 4, 16.0),
    ),
}


def interval_for_entropy(value: float) -> str:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"Invalid normalized entropy: {value}")
    for name, lower, upper in INTERVALS:
        if lower <= value < upper:
            return name
    raise AssertionError(value)


def generation_seed(global_seed: int, color: str, shape: str, target_interval: str, attempt: int) -> int:
    return LEGACY.derive_seed(global_seed, "interval-pool-pilot", color, shape, target_interval, attempt)


def generate_candidate(task: dict[str, Any]) -> dict[str, Any]:
    construction = Construction(**task["construction"])
    seed = int(task["seed"])
    output = Path(task["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if construction.semantic_shape_count in EXP.PROFILE_FOR_COUNT:
        layout = EXP.build_semantic_layout(
            seed, construction.semantic_shape_count, task["shape"], task["color"]
        )
    else:
        layout = EXTREME.build_semantic_layout(
            seed, construction.semantic_shape_count, task["shape"], task["color"]
        )
    if construction.occluder_count == 0:
        final_layout = layout
    elif construction.occluder_count == 1:
        final_layout, _measured, _direction = EXP.add_fixed_direction_occluder(
            layout, construction.occlusion_target, seed
        )
    else:
        final_layout, _measured = EXTREME.add_multiple_occluders(
            layout, construction.occlusion_target, construction.occluder_count, seed
        )
    scene_dir = output / "scene"
    geometry = EXP.render_and_validate(final_layout, scene_dir, construction.occlusion_target)
    image_path = output / "candidate.png"
    POOL.render_blur_variant(scene_dir / "sharp.png", image_path, construction.blur_radius)
    return {
        **task,
        "image_path": str(image_path.resolve()),
        "layout_path": str((scene_dir / "layout.json").resolve()),
        "target_mask_path": str((scene_dir / "target_mask.png").resolve()),
        "occluder_mask_path": str((scene_dir / "occluder_mask.png").resolve()),
        "geometry": geometry,
        "image_sha256": POOL.sha256_file(image_path),
    }


class IntervalPilot:
    def __init__(
        self,
        root: Path,
        model_path: Path,
        seed: int,
        quota: int,
        max_attempts: int,
        workers: int,
        resume: bool,
        runner: Any | None = None,
        pairs: Sequence[tuple[str, str]] = PAIRS,
        active_pair_limit: int | None = None,
        queue_size: int | None = None,
        retain_staging: bool = True,
        migrate_max_attempts_from: int | None = None,
        migrate_quota_from: int | None = None,
    ) -> None:
        self.root = root.resolve(); self.model_path = model_path.resolve(); self.seed = int(seed)
        self.quota = int(quota); self.max_attempts = int(max_attempts); self.workers = int(workers)
        self.pairs = tuple((str(color), str(shape)) for color, shape in pairs)
        self.active_pair_limit = active_pair_limit or len(self.pairs)
        self.queue_size = queue_size or max(
            len(self.pairs) * max(len(values) for values in CONSTRUCTIONS.values()), self.workers
        )
        self.retain_staging = bool(retain_staging)
        if self.quota < 1 or self.max_attempts < 1 or self.workers < 1:
            raise ValueError("quota, max_attempts, and workers must be positive")
        if not self.pairs or len(set(self.pairs)) != len(self.pairs):
            raise ValueError("pairs must be non-empty and unique")
        if self.active_pair_limit < 1 or self.queue_size < 1:
            raise ValueError("active_pair_limit and queue_size must be positive")
        self.root.mkdir(parents=True, exist_ok=True)
        self.results_path = self.root / "candidate_results.jsonl"
        self.rejected_path = self.root / "rejected.jsonl"
        self.manifest_path = self.root / "manifest.json"
        self.config = {
            "schema_version": "interval_pool_pilot.v1", "pairs": [list(pair) for pair in self.pairs],
            "intervals": [list(value) for value in INTERVALS],
            "high_intervals": sorted(HIGH_INTERVALS),
            "quota": self.quota, "max_attempts_per_pair_interval": self.max_attempts,
            "workers": self.workers, "seed": self.seed,
            "constructions": {key: [asdict(value) for value in values] for key, values in CONSTRUCTIONS.items()},
            "model": POOL.model_fingerprint(self.model_path),
            "faithful_runtime_sha256": POOL.sha256_file(POOL.FAITHFUL_DIR / "runtime.py"),
            "faithful_prompts_sha256": POOL.sha256_file(POOL.FAITHFUL_DIR / "prompts.py"),
        }
        if (
            self.pairs != PAIRS or active_pair_limit is not None
            or queue_size is not None or not self.retain_staging
        ):
            self.config["scheduler"] = {
                "active_pair_limit": self.active_pair_limit,
                "queue_size": self.queue_size,
                "task_order": "construction_then_pair",
                "inference_workers": 1,
                "retain_staging": self.retain_staging,
            }
        self.fingerprint = POOL.canonical_hash(self.config)
        if self.manifest_path.is_file():
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if self.manifest.get("config_fingerprint") != self.fingerprint:
                previous_config = dict(self.config)
                migrations = []
                if migrate_max_attempts_from is not None:
                    previous_config["max_attempts_per_pair_interval"] = migrate_max_attempts_from
                    migrations.append({
                        "field": "max_attempts_per_pair_interval",
                        "from": migrate_max_attempts_from,
                        "to": self.max_attempts,
                    })
                if migrate_quota_from is not None:
                    previous_config["quota"] = migrate_quota_from
                    migrations.append({
                        "field": "quota",
                        "from": migrate_quota_from,
                        "to": self.quota,
                    })
                previous_fingerprint = POOL.canonical_hash(previous_config)
                can_migrate = (
                    resume
                    and migrations
                    and (migrate_max_attempts_from is None or self.max_attempts < migrate_max_attempts_from)
                    and (migrate_quota_from is None or self.quota < migrate_quota_from)
                    and self.manifest.get("config_fingerprint") == previous_fingerprint
                )
                if not can_migrate:
                    raise RuntimeError("Pilot configuration changed")
                self.manifest["config_fingerprint"] = self.fingerprint
                self.manifest.setdefault("config_migrations", []).extend(migrations)
                self._persist_manifest()
            if not resume:
                raise RuntimeError("Pilot exists; pass --resume")
        else:
            self.manifest = {
                "schema_version": "interval_pool_pilot.state.v1",
                "config_fingerprint": self.fingerprint, "status": "running",
            }
            self._persist_manifest()
        self._runner_value = runner
        self.records: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.completed: dict[str, dict[str, Any]] = {}
        self.result_rows = EXP.read_jsonl(self.results_path)
        self.accepted_hashes: set[str] = set()
        for color, shape in self.pairs:
            path = self.root / color / f"{shape}.json"
            values = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
            self.records[(color, shape)] = values
            for value in values:
                self.completed[str(value["candidate_id"])] = value
                image_hash = value.get("validation", {}).get("image_sha256")
                if image_hash:
                    self.accepted_hashes.add(str(image_hash))
        for value in self.result_rows:
            self.completed[str(value["candidate_id"])] = value
        if not self.retain_staging:
            self.cleanup_terminal_staging()

    def _persist_manifest(self) -> None:
        POOL.atomic_json(self.manifest_path, self.manifest)

    def _runner(self) -> Any:
        if self._runner_value is None:
            self._runner_value = POOL.load_qwen3_runner(self.model_path)
        return self._runner_value

    def _cleanup_staging_directory(self, directory: Path) -> bool:
        staging_root = (self.root / "_staging").resolve()
        target = directory.resolve()
        if target == staging_root or not target.is_relative_to(staging_root):
            raise ValueError(f"Refusing to clean non-candidate staging path: {target}")
        if not target.exists():
            return False
        try:
            shutil.rmtree(target)
        except OSError as exc:
            EXP.append_jsonl(self.root / "cleanup_errors.jsonl", {
                "path": str(target), "error": f"{type(exc).__name__}: {exc}",
            })
            return False
        return True

    def cleanup_generated(self, generated: dict[str, Any]) -> bool:
        if self.retain_staging:
            return False
        return self._cleanup_staging_directory(Path(generated["output_dir"]))

    def cleanup_terminal_staging(self) -> int:
        """Remove crash-leftover staging only for rows already durably terminal."""
        removed = 0
        for row in self.result_rows:
            image_path = row.get("image_path")
            if image_path:
                candidate_dir = Path(str(image_path)).parent
            elif not row.get("source"):
                interval_dir = str(row["target_interval"]).replace(">=", "gte_")
                construction_id = row["construction"]["construction_id"]
                candidate_dir = (
                    self.root / "_staging" / row["color"] / row["shape"] / interval_dir
                    / f"{int(row['target_attempt_index']):06d}_{construction_id}"
                )
            else:
                continue
            try:
                if candidate_dir.resolve().is_relative_to((self.root / "_staging").resolve()):
                    removed += int(self._cleanup_staging_directory(candidate_dir))
            except (OSError, ValueError):
                continue
        return removed

    def accepted_count(self, pair: tuple[str, str], interval: str) -> int:
        return sum(value.get("actual_interval") == interval for value in self.records[pair])

    def target_rows(self, pair: tuple[str, str], interval: str) -> list[dict[str, Any]]:
        color, shape = pair
        return [
            value for value in self.result_rows
            if value.get("color") == color and value.get("shape") == shape
            and value.get("target_interval") == interval and value.get("status") == "completed"
        ]

    def attempt_limit(self, pair: tuple[str, str], interval: str) -> int:
        return self.max_attempts

    def target_done(self, pair: tuple[str, str], interval: str) -> bool:
        attempts = len(self.target_rows(pair, interval))
        return self.accepted_count(pair, interval) >= self.quota or attempts >= self.attempt_limit(pair, interval)

    def next_attempt(self, pair: tuple[str, str], interval: str) -> int:
        rows = self.target_rows(pair, interval)
        return max((int(value["target_attempt_index"]) for value in rows), default=0) + 1

    def _task(
        self, pair: tuple[str, str], interval: str, attempt: int, construction: Construction
    ) -> dict[str, Any]:
        color, shape = pair
        seed = generation_seed(self.seed, color, shape, interval, attempt)
        candidate_id = POOL.canonical_hash({
            "config": self.fingerprint, "color": color, "shape": shape,
            "target_interval": interval, "attempt": attempt, "construction": construction.construction_id,
        })
        output = self.root / "_staging" / color / shape / interval.replace(">=", "gte_") / f"{attempt:06d}_{construction.construction_id}"
        return {
            "candidate_id": candidate_id, "color": color, "shape": shape,
            "target_interval": interval, "target_attempt_index": attempt,
            "construction": asdict(construction), "seed": seed, "output_dir": str(output),
        }

    def _publish(self, generated: dict[str, Any], validation: dict[str, Any], actual: str) -> dict[str, Any]:
        pair = (generated["color"], generated["shape"])
        records = self.records[pair]
        number = max((int(Path(value["image"]).stem.rsplit("_", 1)[-1]) for value in records), default=0) + 1
        filename = f"{generated['shape']}_{generated['color']}_{number:06d}.png"
        destination = self.root / generated["color"] / filename
        POOL.atomic_copy(Path(generated["image_path"]), destination)
        construction = generated["construction"]
        layout_path = Path(generated["layout_path"])
        if not layout_path.is_file():
            raise RuntimeError(f"Accepted candidate is missing layout: {layout_path}")
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        record = {
            "candidate_id": generated["candidate_id"], "image": filename,
            "Answer": generated["color"], "target_interval": generated["target_interval"],
            "actual_interval": actual, "entropy": validation["normalized_entropy"],
            "construction_id": construction["construction_id"],
            "generation": {
                "seed": generated["seed"], "attempt_index": generated["target_attempt_index"],
                "semantic_shape_count": construction["semantic_shape_count"],
                "occlusion_target": construction["occlusion_target"],
                "occlusion_ratio": generated["geometry"]["occlusion_ratio"],
                "occluder_count": construction["occluder_count"],
                "blur_radius": construction["blur_radius"],
            },
            "validation": {**validation, "image_sha256": generated["image_sha256"]},
            "layout": layout,
            "layout_sha256": POOL.canonical_hash(layout),
        }
        records.append(record)
        POOL.atomic_json(self.root / generated["color"] / f"{generated['shape']}.json", records)
        return record

    def _record_score(self, generated: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
        validation = POOL.validation_from_qwen(raw, generated["color"])
        entropy = validation.get("normalized_entropy")
        actual = interval_for_entropy(float(entropy)) if entropy is not None else "invalid"
        pair = (generated["color"], generated["shape"])
        duplicate_sha = generated["image_sha256"] in self.accepted_hashes
        quota_open = actual != "invalid" and self.accepted_count(pair, actual) < self.quota
        accepted = bool(validation["gate_passed"] and quota_open and not duplicate_sha)
        reasons = list(validation.get("failure_reasons", []))
        if validation["gate_passed"] and not quota_open:
            reasons.append("actual_interval_quota_filled")
        if duplicate_sha:
            reasons.append("duplicate_image_sha256")
        row = {
            "candidate_id": generated["candidate_id"], "status": "completed", "accepted": accepted,
            "color": generated["color"], "shape": generated["shape"],
            "target_interval": generated["target_interval"], "actual_interval": actual,
            "target_attempt_index": generated["target_attempt_index"],
            "construction": generated["construction"], "seed": generated["seed"],
            "geometry": generated["geometry"], "image_path": generated["image_path"],
            "image_sha256": generated["image_sha256"], "validation": validation,
            "rejection_reasons": sorted(set(reasons)),
            "staging_retained": self.retain_staging,
        }
        if accepted:
            published = self._publish(generated, validation, actual)
            self.accepted_hashes.add(generated["image_sha256"])
            row["published_image"] = str((self.root / generated["color"] / published["image"]).resolve())
        else:
            EXP.append_jsonl(self.rejected_path, {
                "candidate_id": row["candidate_id"], "color": row["color"], "shape": row["shape"],
                "target_interval": row["target_interval"], "actual_interval": actual,
                "entropy": entropy, "rejection_reasons": row["rejection_reasons"],
            })
        EXP.append_jsonl(self.results_path, row)
        self.result_rows.append(row)
        self.completed[row["candidate_id"]] = row
        self.cleanup_generated(generated)
        return row

    def score(self, generated: dict[str, Any]) -> dict[str, Any]:
        if generated["candidate_id"] in self.completed:
            return self.completed[generated["candidate_id"]]
        raw = self._runner().image_only(
            LEGACY.QUESTION_TEMPLATE.format(shape=generated["shape"]),
            generated["image_path"], generated["color"],
        )
        return self._record_score(generated, raw)

    def score_batch(self, generated_batch: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        pending = [
            generated for generated in generated_batch
            if generated["candidate_id"] not in self.completed
        ]
        if not pending:
            return []
        runner = self._runner()
        if not hasattr(runner, "image_only_many"):
            return [self.score(generated) for generated in pending]
        requests = [
            (
                LEGACY.QUESTION_TEMPLATE.format(shape=generated["shape"]),
                generated["image_path"], generated["color"],
            )
            for generated in pending
        ]
        raw_results = runner.image_only_many(requests)
        if len(raw_results) != len(pending):
            raise RuntimeError("Qwen worker pool returned the wrong number of results")
        return [
            self._record_score(generated, raw)
            for generated, raw in zip(pending, raw_results, strict=True)
        ]

    def record_generation_failure(self, task: dict[str, Any], exc: Exception) -> None:
        """Consume a failed attempt so a bad construction cannot loop forever."""
        row = {
            "candidate_id": task["candidate_id"], "status": "completed", "accepted": False,
            "color": task["color"], "shape": task["shape"],
            "target_interval": task["target_interval"], "actual_interval": "invalid",
            "target_attempt_index": task["target_attempt_index"],
            "construction": task["construction"], "seed": task["seed"],
            "rejection_reasons": ["generation_failed"],
            "error": f"{type(exc).__name__}: {exc}",
        }
        EXP.append_jsonl(self.root / "generation_errors.jsonl", row)
        EXP.append_jsonl(self.results_path, row)
        self.result_rows.append(row)
        self.completed[row["candidate_id"]] = row
        if not self.retain_staging:
            self._cleanup_staging_directory(Path(task["output_dir"]))

    def run_interval(self, executor: concurrent.futures.ProcessPoolExecutor, interval: str) -> None:
        constructions = CONSTRUCTIONS[interval]
        while True:
            active = [
                pair for pair in self.pairs if not self.target_done(pair, interval)
            ][:self.active_pair_limit]
            if not active:
                return
            tasks_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for pair in active:
                limit = self.attempt_limit(pair, interval)
                remaining = limit - len(self.target_rows(pair, interval))
                start = self.next_attempt(pair, interval)
                pair_tasks = []
                for offset in range(min(len(constructions), remaining)):
                    attempt = start + offset
                    construction = constructions[(attempt - 1) % len(constructions)]
                    pair_tasks.append(self._task(pair, interval, attempt, construction))
                tasks_by_pair[pair] = pair_tasks
            # Construction-major interleaving makes the first worker wave cover
            # every active pair instead of letting the first pair monopolize it.
            tasks = []
            for offset in range(max(map(len, tasks_by_pair.values()), default=0)):
                for pair in active:
                    if offset < len(tasks_by_pair[pair]):
                        tasks.append(tasks_by_pair[pair][offset])
            tasks = tasks[:self.queue_size]
            submitted = [(task, executor.submit(generate_candidate, task)) for task in tasks]
            # Collect in deterministic submission order. With a multi-GPU runner,
            # the whole bounded batch is distributed across Qwen workers and then
            # committed in this same stable order.
            generated_batch = []
            for task, future in submitted:
                try:
                    generated = future.result()
                except Exception as exc:
                    self.record_generation_failure(task, exc)
                    continue
                generated_batch.append(generated)
            self.score_batch(generated_batch)
            for generated in generated_batch:
                pair = (generated["color"], generated["shape"])
                print(
                    f"[{interval}] {pair[0]}/{pair[1]} "
                    f"attempts={len(self.target_rows(pair, interval))}/{self.attempt_limit(pair, interval)} "
                    f"accepted={self.accepted_count(pair, interval)}", flush=True,
                )

    def write_reports(self) -> None:
        rows = self.result_rows
        report = {
            "status": self.manifest.get("status"), "quota": self.quota,
            "max_attempts": self.max_attempts, "pairs": {},
        }
        lines = [
            "# Interval Pool Pilot", "", f"状态：{self.manifest.get('status')}", "",
            "| pair | interval | target attempts | accepted actual | target hit rate | stop reason |",
            "|---|---|---:|---:|---:|---|",
        ]
        for pair in self.pairs:
            pair_key = f"{pair[0]}/{pair[1]}"; report["pairs"][pair_key] = {}
            for interval, _lower, _upper in INTERVALS:
                target = [row for row in rows if row["color"] == pair[0] and row["shape"] == pair[1] and row["target_interval"] == interval]
                hits = [row for row in target if row.get("accepted") and row.get("actual_interval") == interval]
                accepted = self.accepted_count(pair, interval)
                reason = "budget_exhausted" if len(target) >= self.max_attempts else "quota_filled"
                value = {
                    "target_attempts": len(target), "target_hits": len(hits),
                    "accepted_actual_interval": accepted,
                    "target_hit_rate": len(hits) / len(target) if target else 0.0,
                    "stop_reason": reason,
                    "construction_attempts": dict(Counter(row["construction"]["construction_id"] for row in target)),
                    "construction_outcomes": {},
                }
                for construction in CONSTRUCTIONS[interval]:
                    construction_rows = [
                        row for row in target
                        if row["construction"]["construction_id"] == construction.construction_id
                    ]
                    eligible_hits = [
                        row for row in construction_rows
                        if row.get("validation", {}).get("gate_passed")
                        and row.get("actual_interval") == interval
                    ]
                    value["construction_outcomes"][construction.construction_id] = {
                        "attempts": len(construction_rows),
                        "gate_passed": sum(
                            bool(row.get("validation", {}).get("gate_passed"))
                            for row in construction_rows
                        ),
                        "eligible_target_hits": len(eligible_hits),
                        "accepted_target_hits": sum(bool(row.get("accepted")) for row in eligible_hits),
                        "actual_interval_counts": dict(Counter(
                            row.get("actual_interval", "invalid") for row in construction_rows
                        )),
                    }
                report["pairs"][pair_key][interval] = value
                lines.append(
                    f"| {pair_key} | {interval} | {len(target)} | {accepted} | "
                    f"{value['target_hit_rate']:.2%} | {reason} |"
                )
        POOL.atomic_json(self.root / "pilot_report.json", report)
        EXP.atomic_text(self.root / "SUMMARY.md", "\n".join(lines) + "\n")

    def write_round_report(self, round_index: int, interval: str) -> None:
        lines = [
            f"# Round {round_index}: Entropy {interval}", "",
            "## 难度构造", "",
            "本轮把物体数量、目标遮挡率、遮挡物数量和 Gaussian blur 组合起来产生候选；",
            "候选最终归档区间只由 Qwen3 image-only 的 normalized entropy 决定。", "",
            "| construction | objects | occlusion | occluders | blur |", "|---|---:|---:|---:|---:|",
        ]
        for item in CONSTRUCTIONS[interval]:
            lines.append(
                f"| {item.construction_id} | {item.semantic_shape_count} | "
                f"{item.occlusion_target:.0%} | {item.occluder_count} | {item.blur_radius:g} |"
            )
        lines.extend(["", "## 本轮结果", "", "| pair | attempts | hits in target interval | accepted in interval |", "|---|---:|---:|---:|"])
        for pair in self.pairs:
            target = self.target_rows(pair, interval)
            hits = sum(row.get("accepted") and row.get("actual_interval") == interval for row in target)
            lines.append(
                f"| {pair[0]}/{pair[1]} | {len(target)} | {hits} | "
                f"{self.accepted_count(pair, interval)} |"
            )
        lines.extend([
            "", "## 构造命中统计（合并三个组合）", "",
            "| construction | attempts | Qwen gate passed | entropy in target interval | hit rate |",
            "|---|---:|---:|---:|---:|",
        ])
        all_target = [
            row for row in self.result_rows if row.get("target_interval") == interval
        ]
        for construction in CONSTRUCTIONS[interval]:
            selected = [
                row for row in all_target
                if row["construction"]["construction_id"] == construction.construction_id
            ]
            passed = sum(bool(row.get("validation", {}).get("gate_passed")) for row in selected)
            hits = sum(
                bool(row.get("validation", {}).get("gate_passed"))
                and row.get("actual_interval") == interval
                for row in selected
            )
            lines.append(
                f"| {construction.construction_id} | {len(selected)} | {passed} | {hits} | "
                f"{hits / len(selected) if selected else 0.0:.2%} |"
            )
        report_path = self.root / "rounds" / f"round_{round_index:02d}_{interval.replace('>=', 'gte_')}.md"
        EXP.atomic_text(report_path, "\n".join(lines) + "\n")

    def run(self) -> None:
        # Start CPU workers before the CUDA model is loaded.  Each bounded
        # batch contains one task per active pair and construction.
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers) as executor:
            for round_index, (interval, _lower, _upper) in enumerate(INTERVALS, 1):
                self.run_interval(executor, interval)
                self.write_round_report(round_index, interval)
                self.write_reports()
        self.manifest["status"] = "complete"
        self._persist_manifest(); self.write_reports()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a formal interval-pool pilot for three color-shape pairs")
    parser.add_argument("run", nargs="?", default="run")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--quota", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--migrate-max-attempts-from", type=int)
    parser.add_argument("--migrate-quota-from", type=int)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    IntervalPilot(
        args.output_root, args.model_path, args.seed, args.quota,
        args.max_attempts, args.workers, args.resume,
        migrate_max_attempts_from=args.migrate_max_attempts_from,
        migrate_quota_from=args.migrate_quota_from,
    ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
