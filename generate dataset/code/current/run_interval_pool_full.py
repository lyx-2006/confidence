#!/usr/bin/env python3
"""Run the full 12-color x 17-shape entropy-pool experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing
import os
import queue
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
ROOT = PROJECT_DIR.parent
DEFAULT_ROOT = PROJECT_DIR / "datasets" / "current" / "interval_pool_full"
DEFAULT_REUSE_ROOT = PROJECT_DIR / "datasets" / "legacy" / "experiments" / "interval_pool_pilot"


def _load_pilot() -> Any:
    path = SCRIPT_DIR / "run_interval_pool_pilot.py"
    spec = importlib.util.spec_from_file_location("_full_interval_pilot_support", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PILOT = _load_pilot()
POOL = PILOT.POOL
LEGACY = PILOT.LEGACY
ALL_PAIRS = tuple(
    (color, shape) for color in POOL.COLORS for shape in LEGACY.SHAPES
)


def _qwen_gpu_worker(
    device: str,
    model_path: str,
    task_queue: Any,
    result_queue: Any,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        runner = POOL.load_qwen3_runner(Path(model_path))
    except BaseException as exc:
        result_queue.put(("startup_error", device, type(exc).__name__, str(exc)))
        return
    result_queue.put(("ready", device, None, None))
    while True:
        task = task_queue.get()
        if task is None:
            return
        index, question, image_path, expected = task
        try:
            raw = runner.image_only(question, image_path, expected)
        except BaseException as exc:
            result_queue.put(("result_error", index, type(exc).__name__, str(exc)))
        else:
            result_queue.put(("result", index, raw, device))


class MultiGpuQwenRunner:
    """Lazy one-model-per-GPU runner with deterministic ordered results."""

    def __init__(self, model_path: Path, devices: Sequence[str]) -> None:
        normalized = tuple(str(device).strip() for device in devices if str(device).strip())
        if not normalized or len(set(normalized)) != len(normalized):
            raise ValueError("gpu devices must be non-empty and unique")
        self.model_path = model_path.resolve()
        self.devices = normalized
        self.context = multiprocessing.get_context("spawn")
        self.task_queues: list[Any] = []
        self.result_queue: Any | None = None
        self.processes: list[Any] = []

    def _start(self) -> None:
        if self.processes:
            return
        self.result_queue = self.context.Queue()
        for device in self.devices:
            task_queue = self.context.Queue()
            process = self.context.Process(
                target=_qwen_gpu_worker,
                args=(device, str(self.model_path), task_queue, self.result_queue),
                name=f"qwen-gpu-{device}",
            )
            process.start()
            self.task_queues.append(task_queue)
            self.processes.append(process)
        ready: set[str] = set()
        while len(ready) < len(self.devices):
            try:
                kind, key, error_type, message = self.result_queue.get(timeout=600)
            except queue.Empty as exc:
                self.close()
                raise RuntimeError("Timed out loading Qwen GPU workers") from exc
            if kind == "startup_error":
                self.close()
                raise RuntimeError(f"Qwen worker GPU {key} failed: {error_type}: {message}")
            if kind != "ready":
                self.close()
                raise RuntimeError(f"Unexpected Qwen startup message: {kind}")
            ready.add(str(key))

    def image_only_many(
        self, requests: Sequence[tuple[str, str, str]]
    ) -> list[dict[str, Any]]:
        self._start()
        assert self.result_queue is not None
        for index, (question, image_path, expected) in enumerate(requests):
            worker = index % len(self.task_queues)
            self.task_queues[worker].put((index, question, image_path, expected))
        results: list[dict[str, Any] | None] = [None] * len(requests)
        received = 0
        while received < len(requests):
            try:
                kind, key, value, detail = self.result_queue.get(timeout=600)
            except queue.Empty as exc:
                dead = [process.name for process in self.processes if not process.is_alive()]
                raise RuntimeError(f"Timed out waiting for Qwen workers; dead={dead}") from exc
            if kind == "result_error":
                raise RuntimeError(f"Qwen request {key} failed: {value}: {detail}")
            if kind != "result":
                raise RuntimeError(f"Unexpected Qwen result message: {kind}")
            results[int(key)] = value
            received += 1
        if any(value is None for value in results):
            raise RuntimeError("Qwen worker pool returned incomplete results")
        return [value for value in results if value is not None]

    def image_only(self, question: str, image_path: str, expected: str) -> dict[str, Any]:
        return self.image_only_many([(question, image_path, expected)])[0]

    def close(self) -> None:
        for task_queue in self.task_queues:
            try:
                task_queue.put(None)
            except (OSError, ValueError):
                pass
        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        self.task_queues = []
        self.processes = []
        if self.result_queue is not None:
            self.result_queue.close()
            self.result_queue = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FullIntervalPool(PILOT.IntervalPilot):
    def __init__(
        self,
        root: Path,
        model_path: Path,
        seed: int = 20260920,
        quota: int = 3,
        max_attempts: int = 100,
        workers: int = 24,
        active_pairs: int = 12,
        queue_size: int = 48,
        resume: bool = False,
        reuse_roots: Sequence[Path] = (),
        runner: Any | None = None,
        migrate_max_attempts_from: int | None = None,
        migrate_quota_from: int | None = None,
        supplement_missing_intervals: Sequence[str] = (),
        additional_attempts: int = 100,
    ) -> None:
        super().__init__(
            root=root,
            model_path=model_path,
            seed=seed,
            quota=quota,
            max_attempts=max_attempts,
            workers=workers,
            resume=resume,
            runner=runner,
            pairs=ALL_PAIRS,
            active_pair_limit=active_pairs,
            queue_size=queue_size,
            retain_staging=False,
            migrate_max_attempts_from=migrate_max_attempts_from,
            migrate_quota_from=migrate_quota_from,
        )
        for reuse_root in reuse_roots:
            self.import_existing(reuse_root)
        self.backfill_accepted_layouts()
        self._configure_supplement_phase(supplement_missing_intervals, additional_attempts)

    def _configure_supplement_phase(
        self, intervals: Sequence[str], additional_attempts: int
    ) -> None:
        self.supplement_enabled = bool(intervals)
        self.supplement_targets: dict[tuple[str, str, str], int] = {}
        if not intervals:
            return
        selected = tuple(dict.fromkeys(str(value) for value in intervals))
        valid = {name for name, _lower, _upper in PILOT.INTERVALS}
        if any(value not in valid for value in selected):
            raise ValueError(f"Unknown supplement interval: {selected}")
        if additional_attempts < 1:
            raise ValueError("additional_attempts must be positive")
        phase_config = {
            "schema_version": "interval_pool.supplement.v1",
            "intervals": list(selected),
            "quota": self.quota,
            "additional_attempts": int(additional_attempts),
        }
        phase = self.manifest.get("supplement_phase")
        if phase is None:
            baselines = {}
            for color, shape in self.pairs:
                for interval in selected:
                    if self.accepted_count((color, shape), interval) == 0:
                        key = f"{color}/{shape}/{interval}"
                        baselines[key] = len(self.target_rows((color, shape), interval))
            phase = {**phase_config, "status": "running", "baselines": baselines}
            self.manifest["supplement_phase"] = phase
            self._persist_manifest()
        else:
            actual_config = {key: phase.get(key) for key in phase_config}
            if actual_config != phase_config:
                raise RuntimeError("Supplement phase configuration changed")
        for key, baseline in phase.get("baselines", {}).items():
            color, shape, interval = key.split("/", 2)
            self.supplement_targets[(color, shape, interval)] = int(baseline)

    def attempt_limit(self, pair: tuple[str, str], interval: str) -> int:
        if not self.supplement_enabled:
            return super().attempt_limit(pair, interval)
        baseline = self.supplement_targets.get((pair[0], pair[1], interval))
        if baseline is None:
            return len(self.target_rows(pair, interval))
        return baseline + int(self.manifest["supplement_phase"]["additional_attempts"])

    def target_done(self, pair: tuple[str, str], interval: str) -> bool:
        if self.supplement_enabled and (pair[0], pair[1], interval) not in self.supplement_targets:
            return True
        return super().target_done(pair, interval)

    def run(self) -> None:
        try:
            super().run()
        finally:
            close = getattr(self._runner_value, "close", None)
            if callable(close):
                close()

    def _reconstruct_layout(
        self, color: str, shape: str, record: dict[str, Any]
    ) -> dict[str, Any]:
        generation = record["generation"]
        seed = int(generation["seed"])
        semantic_count = int(generation["semantic_shape_count"])
        occlusion_target = float(generation["occlusion_target"])
        occluder_count = int(generation["occluder_count"])
        if semantic_count in PILOT.EXP.PROFILE_FOR_COUNT:
            layout = PILOT.EXP.build_semantic_layout(seed, semantic_count, shape, color)
        else:
            layout = PILOT.EXTREME.build_semantic_layout(seed, semantic_count, shape, color)
        if occluder_count == 0:
            final_layout = layout
        elif occluder_count == 1:
            final_layout, _measured, _direction = PILOT.EXP.add_fixed_direction_occluder(
                layout, occlusion_target, seed
            )
        else:
            final_layout, _measured = PILOT.EXTREME.add_multiple_occluders(
                layout, occlusion_target, occluder_count, seed
            )
        targets = [
            obj for obj in final_layout["objects"] if obj.get("role") == "target"
        ]
        if (
            len(targets) != 1 or targets[0].get("shape") != shape
            or targets[0].get("color") != color
        ):
            raise RuntimeError(f"Reconstructed layout target mismatch: {color}/{shape}")
        return final_layout

    def backfill_accepted_layouts(self) -> dict[str, int]:
        """Atomically add exact deterministic layouts to older accepted records."""
        updated_records = 0
        updated_files = 0
        for color, shape in self.pairs:
            records = self.records[(color, shape)]
            changed = False
            for record in records:
                if isinstance(record.get("layout"), dict):
                    expected_hash = POOL.canonical_hash(record["layout"])
                    if record.get("layout_sha256") != expected_hash:
                        record["layout_sha256"] = expected_hash
                        changed = True
                    continue
                layout = self._reconstruct_layout(color, shape, record)
                record["layout"] = layout
                record["layout_sha256"] = POOL.canonical_hash(layout)
                updated_records += 1
                changed = True
            if changed:
                POOL.atomic_json(self.root / color / f"{shape}.json", records)
                updated_files += 1
        self.manifest["accepted_layouts"] = {
            "status": "complete",
            "record_count": sum(len(values) for values in self.records.values()),
            "updated_on_last_start": updated_records,
        }
        self._persist_manifest()
        return {"records": updated_records, "files": updated_files}

    def _verify_source(self, source: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        manifest_path = source / "manifest.json"
        results_path = source / "candidate_results.jsonl"
        if not manifest_path.is_file() or not results_path.is_file():
            raise FileNotFoundError(f"Reusable pool is incomplete: {source}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise RuntimeError(f"Reusable pool is not complete: {source}")
        if source == DEFAULT_REUSE_ROOT.resolve():
            expected_config = {
                "schema_version": "interval_pool_pilot.v1",
                "pairs": [list(pair) for pair in PILOT.PAIRS],
                "intervals": [list(value) for value in PILOT.INTERVALS],
                "high_intervals": sorted(PILOT.HIGH_INTERVALS),
                "quota": 5,
                "max_attempts_per_pair_interval": 200,
                "workers": 12,
                "seed": 20260920,
                "constructions": {
                    key: [PILOT.asdict(value) for value in values]
                    for key, values in PILOT.CONSTRUCTIONS.items()
                },
                "model": POOL.model_fingerprint(self.model_path),
                "faithful_runtime_sha256": POOL.sha256_file(POOL.FAITHFUL_DIR / "runtime.py"),
                "faithful_prompts_sha256": POOL.sha256_file(POOL.FAITHFUL_DIR / "prompts.py"),
            }
            expected_fingerprint = POOL.canonical_hash(expected_config)
            if manifest.get("config_fingerprint") != expected_fingerprint:
                raise RuntimeError(
                    "Existing pilot was measured with a different model, prompt, or construction config"
                )
        rows = PILOT.EXP.read_jsonl(results_path)
        if not rows:
            raise RuntimeError(f"Reusable pool has no candidate rows: {source}")
        valid_pairs = set(self.pairs)
        valid_intervals = {name for name, _lower, _upper in PILOT.INTERVALS}
        if len({str(row.get("candidate_id")) for row in rows}) != len(rows):
            raise RuntimeError(f"Reusable pool has duplicate candidate IDs: {source}")
        for row in rows:
            if row.get("status") != "completed":
                raise RuntimeError(f"Reusable row is not terminal: {row.get('candidate_id')}")
            pair = (row.get("color"), row.get("shape"))
            if pair not in valid_pairs or row.get("target_interval") not in valid_intervals:
                raise RuntimeError(f"Incompatible reusable row: {row.get('candidate_id')}")
            construction = row.get("construction", {})
            expected = {
                item.construction_id: item
                for item in PILOT.CONSTRUCTIONS[row["target_interval"]]
            }.get(construction.get("construction_id"))
            if expected is None or construction != PILOT.asdict(expected):
                raise RuntimeError(f"Reusable construction changed: {row.get('candidate_id')}")
            if int(row.get("target_attempt_index", 0)) < 1:
                raise RuntimeError(f"Reusable attempt index is invalid: {row.get('candidate_id')}")
        return manifest, rows

    def import_existing(self, reuse_root: Path) -> dict[str, int]:
        """Import measured candidates and accepted PNGs without Qwen inference."""
        source = reuse_root.resolve()
        if source == self.root:
            raise ValueError("reuse root must differ from output root")
        manifest, source_rows = self._verify_source(source)
        source_key = {
            "path": str(source),
            "config_fingerprint": manifest.get("config_fingerprint"),
            "results_sha256": sha256_file(source / "candidate_results.jsonl"),
        }
        reused = self.manifest.setdefault("reuse_sources", [])
        matching = [item for item in reused if item.get("path") == str(source)]
        if matching:
            if any(item != source_key for item in matching):
                raise RuntimeError(f"Reusable source changed after import: {source}")
            return {"candidate_rows": 0, "accepted_images": 0}

        result_ids = {str(row["candidate_id"]) for row in self.result_rows}
        imported_records: dict[str, dict[str, Any]] = {}
        copied = 0
        source_pairs = sorted({(str(row["color"]), str(row["shape"])) for row in source_rows})
        for color, shape in source_pairs:
            source_json = source / color / f"{shape}.json"
            values = json.loads(source_json.read_text(encoding="utf-8")) if source_json.is_file() else []
            destination_records = self.records[(color, shape)]
            known = {str(value["candidate_id"]) for value in destination_records}
            for original in values:
                candidate_id = str(original["candidate_id"])
                if candidate_id in known:
                    imported_records[candidate_id] = next(
                        value for value in destination_records
                        if str(value["candidate_id"]) == candidate_id
                    )
                    continue
                source_image = source / color / str(original["image"])
                expected_hash = str(original["validation"]["image_sha256"])
                if not source_image.is_file() or sha256_file(source_image) != expected_hash:
                    raise RuntimeError(f"Reusable image hash mismatch: {source_image}")
                if expected_hash in self.accepted_hashes:
                    raise RuntimeError(f"Duplicate reusable image SHA256: {expected_hash}")
                number = max(
                    (int(Path(value["image"]).stem.rsplit("_", 1)[-1]) for value in destination_records),
                    default=0,
                ) + 1
                record = deepcopy(original)
                record["image"] = f"{shape}_{color}_{number:06d}.png"
                record["source"] = {
                    "kind": "reused_interval_pool",
                    "root": str(source),
                    "original_image": original["image"],
                }
                POOL.atomic_copy(source_image, self.root / color / record["image"])
                destination_records.append(record)
                known.add(candidate_id)
                imported_records[candidate_id] = record
                self.accepted_hashes.add(expected_hash)
                copied += 1
            POOL.atomic_json(self.root / color / f"{shape}.json", destination_records)

        added_rows = 0
        for original in source_rows:
            candidate_id = str(original["candidate_id"])
            if candidate_id in result_ids:
                continue
            row = deepcopy(original)
            row["source"] = {"kind": "reused_interval_pool", "root": str(source)}
            if row.get("accepted"):
                record = imported_records.get(candidate_id)
                if record is None:
                    raise RuntimeError(f"Accepted reusable row has no JSON record: {candidate_id}")
                row["published_image"] = str(
                    (self.root / row["color"] / record["image"]).resolve()
                )
            else:
                PILOT.EXP.append_jsonl(self.rejected_path, {
                    "candidate_id": candidate_id,
                    "color": row["color"], "shape": row["shape"],
                    "target_interval": row["target_interval"],
                    "actual_interval": row.get("actual_interval", "invalid"),
                    "entropy": row.get("validation", {}).get("normalized_entropy"),
                    "rejection_reasons": row.get("rejection_reasons", []),
                    "source": row["source"],
                })
            PILOT.EXP.append_jsonl(self.results_path, row)
            self.result_rows.append(row)
            self.completed[candidate_id] = row
            result_ids.add(candidate_id)
            added_rows += 1

        reused.append(source_key)
        self._persist_manifest()
        return {"candidate_rows": added_rows, "accepted_images": copied}


def parse_gpu_devices(raw: str) -> tuple[str, ...]:
    devices = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not devices or len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError("gpu devices must be a comma-separated unique list")
    return devices


def plan(devices: Sequence[str] = ("0",)) -> dict[str, Any]:
    return {
        "pair_count": len(ALL_PAIRS),
        "colors": len(POOL.COLORS),
        "shapes": len(LEGACY.SHAPES),
        "active_pairs": 12,
        "generation_workers": 24,
        "bounded_queue": 48,
        "qwen_workers": len(devices),
        "gpu_devices": list(devices),
        "retain_staging": False,
        "permanent_candidate_data": [
            "candidate_results.jsonl", "rejected.jsonl",
            "accepted PNG", "per-shape JSON",
        ],
        "default_reuse_root": str(DEFAULT_REUSE_ROOT.resolve()),
        "output_root": str(DEFAULT_ROOT.resolve()),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "status"), nargs="?", default="plan")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--model-path", type=Path, default=PILOT.DEFAULT_MODEL)
    parser.add_argument("--reuse-root", type=Path, action="append", default=None)
    parser.add_argument("--no-reuse", action="store_true")
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--quota", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--migrate-max-attempts-from", type=int)
    parser.add_argument("--migrate-quota-from", type=int)
    parser.add_argument("--supplement-missing-interval", action="append", default=[])
    parser.add_argument("--additional-attempts", type=int, default=100)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--active-pairs", type=int, default=12)
    parser.add_argument("--queue-size", type=int, default=48)
    parser.add_argument("--gpu-devices", type=parse_gpu_devices, default=("0",))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "plan":
        print(json.dumps(plan(args.gpu_devices), ensure_ascii=False, indent=2))
        return 0
    if args.command == "status":
        report = args.output_root / "pilot_report.json"
        if not report.is_file():
            print(json.dumps({"status": "not_started", "root": str(args.output_root.resolve())}))
        else:
            print(report.read_text(encoding="utf-8"))
        return 0
    reuse_roots: list[Path] = []
    if not args.no_reuse:
        reuse_roots = args.reuse_root or [DEFAULT_REUSE_ROOT]
    runner = MultiGpuQwenRunner(args.model_path, args.gpu_devices)
    experiment = FullIntervalPool(
        root=args.output_root, model_path=args.model_path, seed=args.seed,
        quota=args.quota, max_attempts=args.max_attempts, workers=args.workers,
        active_pairs=args.active_pairs, queue_size=args.queue_size,
        resume=args.resume, reuse_roots=reuse_roots,
        runner=runner, migrate_max_attempts_from=args.migrate_max_attempts_from,
        migrate_quota_from=args.migrate_quota_from,
        supplement_missing_intervals=args.supplement_missing_interval,
        additional_attempts=args.additional_attempts,
    )
    experiment.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
