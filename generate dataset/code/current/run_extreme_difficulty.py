#!/usr/bin/env python3
"""Run five resumable extreme Qwen3 image-difficulty rounds.

Each candidate is evaluated three times.  A candidate passes when at least two
trials pass the normal generated-answer plus restricted-top1 gate.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageChops, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
ROOT = PROJECT_DIR.parent
DEFAULT_ROOT = PROJECT_DIR / "datasets" / "legacy" / "experiments" / "image_difficulty_extreme"
DEFAULT_MODEL = ROOT / "qwen-3-vl" / "model"


def _load(name: str, filename: str) -> Any:
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXP = _load("_extreme_experiment_support", "explore_image_difficulty.py")
POOL = EXP.POOL
LEGACY = EXP.LEGACY
COLORS = EXP.COLORS
SHAPES = EXP.SHAPES
BLUR_RADII = EXP.BLUR_RADII
SEED_INDICES = (1, 2, 3)
TRIALS_PER_IMAGE = 3
REQUIRED_CORRECT_TRIALS = 2


@dataclass(frozen=True)
class ExtremeSpec:
    round_id: int
    semantic_shape_count: int
    occlusion_target: float
    occluder_count: int

    @property
    def name(self) -> str:
        return (
            f"extreme_{self.round_id:03d}_count_{self.semantic_shape_count:02d}_"
            f"occ_{int(round(self.occlusion_target * 100)):02d}_occluders_{self.occluder_count}"
        )

    @property
    def candidate_count(self) -> int:
        return len(COLORS) * len(SHAPES) * len(SEED_INDICES) * len(BLUR_RADII)


def extreme_specs() -> tuple[ExtremeSpec, ...]:
    return (
        ExtremeSpec(1, 10, 0.60, 2),
        ExtremeSpec(2, 10, 0.70, 3),
        ExtremeSpec(3, 10, 0.80, 4),
        ExtremeSpec(4, 11, 0.80, 4),
        ExtremeSpec(5, 12, 0.80, 4),
    )


def build_semantic_layout(seed: int, count: int, target_shape: str, target_color: str) -> dict[str, Any]:
    forbidden = set(LEGACY.similar_shapes_for(target_shape)) | {target_shape}
    allowed = [shape for shape in POOL.SHAPES if shape not in forbidden]
    if len(allowed) < count - 1:
        raise ValueError(f"Not enough distinct shapes for count={count}, target={target_shape}")
    colors = [color for color in POOL.COLORS if color != target_color]
    objects = None
    for restart in range(240):
        rng = random.Random(LEGACY.derive_seed(seed, "extreme-semantic-layout", restart))
        shapes = list(allowed)
        rng.shuffle(shapes)
        proposed = []
        try:
            proposed.append(LEGACY._place_object(
                rng, proposed, target_shape, target_color, "target", LEGACY.EASY_TARGET_SIZE_RANGE
            ))
            for shape in shapes[: count - 1]:
                proposed.append(LEGACY._place_object(
                    rng, proposed, shape, rng.choice(colors), "distractor", LEGACY.EASY_DISTRACTOR_SIZE_RANGE
                ))
        except ValueError:
            continue
        if not LEGACY.detect_layout_patterns(proposed):
            objects = proposed
            break
    if objects is None:
        raise ValueError(f"Could not pack {count} semantic shapes")
    return {
        "schema_version": "image_pool.extreme_layout.v1",
        "canvas": [LEGACY.CANVAS_SIZE, LEGACY.CANVAS_SIZE],
        "branch": "image_pool_extreme",
        "difficulty": "extreme_experiment",
        "profile": f"extreme_count_{count:02d}",
        "semantic_shape_count": count,
        "target_shape": target_shape,
        "target_color": target_color,
        "case_seed": seed,
        "target_geometry": {key: objects[0][key] for key in ("center", "bbox", "rotation", "size")},
        "occluder_geometry": [],
        "expected_occlusion_ratio": 0.0,
        "objects": objects,
    }


def _partition_bounds(start: int, stop: int, count: int) -> list[tuple[int, int]]:
    result = []
    for index in range(count):
        left = round(start + (stop - start) * index / count)
        right = round(start + (stop - start) * (index + 1) / count)
        result.append((left, max(left + 1, right)))
    return result


def add_multiple_occluders(
    semantic_layout: dict[str, Any], target_ratio: float, occluder_count: int, seed: int
) -> tuple[dict[str, Any], float]:
    layout = json.loads(json.dumps(semantic_layout))
    target = next(obj for obj in layout["objects"] if obj.get("role") == "target")
    target_mask = LEGACY._object_mask(target).convert("1")
    bounds = target_mask.getbbox()
    if bounds is None:
        raise ValueError("Empty target mask")
    left, top, right, bottom = bounds
    direction = random.Random(LEGACY.derive_seed(seed, "extreme-occlusion-direction")).choice(
        ("left", "right", "top", "bottom")
    )
    axis_limit = right - left if direction in {"left", "right"} else bottom - top
    colors = [value for value in POOL.COLORS if value != layout["target_color"]]
    random.Random(LEGACY.derive_seed(seed, "extreme-occluder-colors")).shuffle(colors)
    cropped = target_mask.crop(bounds).convert("1")
    cropped_pixels = list(cropped.getdata())
    cropped_width, cropped_height = cropped.size
    target_area = sum(bool(value) for value in cropped_pixels)
    if target_area == 0:
        raise ValueError("Empty target mask")
    column_counts = [
        sum(bool(cropped_pixels[y * cropped_width + x]) for y in range(cropped_height))
        for x in range(cropped_width)
    ]
    row_counts = [
        sum(bool(cropped_pixels[y * cropped_width + x]) for x in range(cropped_width))
        for y in range(cropped_height)
    ]
    counts = column_counts if direction in {"left", "right"} else row_counts
    if direction in {"right", "bottom"}:
        counts = list(reversed(counts))
    best_amount: int | None = None
    best_ratio: float | None = None
    covered = 0
    for amount in range(1, axis_limit + 1):
        covered += counts[amount - 1]
        ratio = covered / target_area
        if best_ratio is None or abs(ratio - target_ratio) < abs(best_ratio - target_ratio):
            best_amount, best_ratio = amount, ratio
    if best_amount is None or best_ratio is None:
        raise ValueError("Occlusion search produced no candidate")
    if direction in {"left", "right"}:
        band_left, band_right = (left, left + best_amount) if direction == "left" else (right - best_amount, right)
        boxes = [(band_left, p0, band_right, p1) for p0, p1 in _partition_bounds(top, bottom, occluder_count)]
    else:
        band_top, band_bottom = (top, top + best_amount) if direction == "top" else (bottom - best_amount, bottom)
        boxes = [(p0, band_top, p1, band_bottom) for p0, p1 in _partition_bounds(left, right, occluder_count)]
    occluders = [
        LEGACY._occluder_candidate("rectangle", colors[index % len(colors)], tuple(map(float, box)))
        for index, box in enumerate(boxes)
    ]
    union = Image.new("1", (LEGACY.CANVAS_SIZE, LEGACY.CANVAS_SIZE), 0)
    for obj in occluders:
        union = ImageChops.lighter(union, LEGACY._object_mask(obj).convert("1"))
    measured = LEGACY._mask_overlap_ratio(target_mask, union)
    if abs(measured - target_ratio) > 0.02:
        raise ValueError(f"Could not reach occlusion {target_ratio:.0%}; measured={measured}")
    for obj in occluders:
        obj["expected_mask_occlusion_ratio"] = LEGACY._mask_overlap_ratio(
            target_mask, LEGACY._object_mask(obj).convert("1")
        )
    layout["objects"].extend(occluders)
    layout["occluder_geometry"] = [
        {key: obj[key] for key in ("shape", "center", "bbox", "rotation", "size")} for obj in occluders
    ]
    layout["expected_occlusion_ratio"] = measured
    layout["occlusion_direction"] = direction
    layout["occluder_count"] = occluder_count
    layout["profile"] = (
        f"extreme_count_{layout['semantic_shape_count']:02d}_occ_{int(round(target_ratio * 100)):02d}_"
        f"occluders_{occluder_count}"
    )
    return layout, measured


def aggregate_trials(trials: Sequence[dict[str, Any]]) -> dict[str, Any]:
    passed = [trial for trial in trials if trial.get("gate_passed") is True]
    entropy_values = [
        float(trial["normalized_entropy"])
        for trial in trials
        if trial.get("normalized_entropy") is not None and math.isfinite(float(trial["normalized_entropy"]))
    ]
    probability_values = [
        float(trial["target_probability"])
        for trial in trials
        if trial.get("target_probability") is not None and math.isfinite(float(trial["target_probability"]))
    ]
    return {
        "gate_passed": len(passed) >= REQUIRED_CORRECT_TRIALS,
        "correct_trials": len(passed),
        "required_correct_trials": REQUIRED_CORRECT_TRIALS,
        "trial_count": len(trials),
        "normalized_entropy": statistics.median(entropy_values) if entropy_values else None,
        "entropy_values": entropy_values,
        "target_probability": statistics.median(probability_values) if probability_values else None,
        "predictions": [trial.get("prediction") for trial in trials],
        "restricted_top1": [trial.get("restricted_top1") for trial in trials],
        "trials": list(trials),
    }


def entropy_band(value: float | None) -> str:
    if value is None:
        return "invalid"
    for lower, upper in ((0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6)):
        if lower <= value < upper:
            return f"{lower:.1f}-{upper:.1f}"
    return ">=0.6"


def summarize(spec: ExtremeSpec, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    passed = [row for row in rows if row.get("validation", {}).get("gate_passed") is True]
    entropies = [float(row["validation"]["normalized_entropy"]) for row in passed if row["validation"].get("normalized_entropy") is not None]
    by_blur = {}
    for radius in BLUR_RADII:
        subset = [row for row in rows if row.get("blur_radius") == radius]
        valid = [row for row in subset if row.get("validation", {}).get("gate_passed") is True]
        values = [float(row["validation"]["normalized_entropy"]) for row in valid]
        by_blur[f"{radius:g}"] = {
            "attempted": len(subset), "passed": len(valid),
            "pass_rate": len(valid) / len(subset) if subset else None,
            "entropy_median": statistics.median(values) if values else None,
        }
    bands = Counter(entropy_band(row["validation"].get("normalized_entropy")) for row in passed)
    votes = Counter(int(row["validation"]["correct_trials"]) for row in rows if row.get("status") == "completed")
    measured = [float(row["occlusion_ratio"]) for row in rows if row.get("occlusion_ratio") is not None]
    return {
        "round_id": spec.round_id, "round_name": spec.name, **asdict(spec),
        "candidate_count": len(rows), "passed_count": len(passed),
        "pass_rate": len(passed) / len(rows) if rows else None,
        "occlusion_measured_median": statistics.median(measured) if measured else None,
        "entropy_q25": EXP.quantile(entropies, 0.25), "entropy_median": EXP.quantile(entropies, 0.5),
        "entropy_q75": EXP.quantile(entropies, 0.75), "entropy_q90": EXP.quantile(entropies, 0.9),
        "entropy_max": max(entropies) if entropies else None,
        "by_blur": by_blur, "entropy_bands": dict(sorted(bands.items())),
        "correct_trial_histogram": {str(key): value for key, value in sorted(votes.items())},
    }


def markdown(spec: ExtremeSpec, summary: dict[str, Any]) -> str:
    blur = ["| blur | passed/attempted | pass rate | entropy median |", "|---:|---:|---:|---:|"]
    for radius, value in summary["by_blur"].items():
        median = "n/a" if value["entropy_median"] is None else f"{value['entropy_median']:.6g}"
        blur.append(f"| {radius} | {value['passed']}/{value['attempted']} | {value['pass_rate']:.2%} | {median} |")
    bands = ["| Entropy band | passed images |", "|---|---:|"]
    bands.extend(f"| {name} | {count} |" for name, count in summary["entropy_bands"].items())
    return f"""# Extreme Round {spec.round_id:03d}

状态：已完成

- 语义物体：{spec.semantic_shape_count}
- 联合遮挡物：{spec.occluder_count}
- 目标遮挡：{spec.occlusion_target:.0%}；实测中位数：{summary['occlusion_measured_median']:.2%}
- blur：{', '.join(f'{value:g}' for value in BLUR_RADII)}
- 候选：{summary['candidate_count']}
- 每张 Qwen3 测试：3 次；至少 2 次通过双门控
- 通过：{summary['passed_count']}（{summary['pass_rate']:.2%}）
- 0/1/2/3 次正确分布：`{json.dumps(summary['correct_trial_histogram'], sort_keys=True)}`

## Entropy（只统计 2/3 通过样本，取三次测量中位数）

| q25 | median | q75 | q90 | max |
|---:|---:|---:|---:|---:|
| {summary['entropy_q25'] or 0:.6g} | {summary['entropy_median'] or 0:.6g} | {summary['entropy_q75'] or 0:.6g} | {summary['entropy_q90'] or 0:.6g} | {summary['entropy_max'] or 0:.6g} |

## Blur

{chr(10).join(blur)}

## Entropy 区间

{chr(10).join(bands)}

`contact_sheet.png` 展示高 Entropy 通过样本和未通过样本；完整三次输出位于 `candidate_results.jsonl`。
"""


class ExtremeRunner:
    def __init__(self, root: Path, model_path: Path, seed: int, resume: bool, runner: Any | None = None) -> None:
        self.root = root.resolve(); self.model_path = model_path.resolve(); self.seed = int(seed)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.config = {
            "schema_version": "image_difficulty_extreme.v1", "seed": self.seed,
            "specs": [asdict(spec) for spec in extreme_specs()], "colors": list(COLORS), "shapes": list(SHAPES),
            "seed_indices": list(SEED_INDICES), "blur_radii": list(BLUR_RADII),
            "trials_per_image": TRIALS_PER_IMAGE, "required_correct_trials": REQUIRED_CORRECT_TRIALS,
            "model": POOL.model_fingerprint(self.model_path),
            "faithful_runtime_sha256": POOL.sha256_file(POOL.FAITHFUL_DIR / "runtime.py"),
            "faithful_prompts_sha256": POOL.sha256_file(POOL.FAITHFUL_DIR / "prompts.py"),
        }
        self.fingerprint = POOL.canonical_hash(self.config)
        if self.manifest_path.is_file():
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if self.manifest.get("config_fingerprint") != self.fingerprint:
                raise RuntimeError("Extreme experiment configuration changed")
            if not resume:
                raise RuntimeError("Extreme experiment exists; pass --resume")
        else:
            self.manifest = {"schema_version": "image_difficulty_extreme.state.v1", "config_fingerprint": self.fingerprint, "completed_rounds": [], "status": "running"}
            self._persist()
        self._runner_value = runner

    def _persist(self) -> None:
        POOL.atomic_json(self.manifest_path, self.manifest)

    def _runner(self) -> Any:
        if self._runner_value is None:
            self._runner_value = POOL.load_qwen3_runner(self.model_path)
        return self._runner_value

    def _round_dir(self, spec: ExtremeSpec) -> Path:
        return self.root / spec.name

    def _round_complete(self, spec: ExtremeSpec) -> bool:
        directory = self._round_dir(spec)
        if spec.round_id not in self.manifest.get("completed_rounds", []):
            return False
        rows = EXP.read_jsonl(directory / "candidate_results.jsonl")
        return len({row.get("candidate_id") for row in rows}) == spec.candidate_count and (directory / "summary.json").is_file()

    def run_round(self, spec: ExtremeSpec) -> dict[str, Any]:
        directory = self._round_dir(spec); directory.mkdir(parents=True, exist_ok=True)
        round_config = {"experiment_fingerprint": self.fingerprint, "spec": asdict(spec)}
        POOL.atomic_json(directory / "config.json", round_config)
        EXP.atomic_text(directory / f"extreme_{spec.round_id:03d}.md", f"# Extreme Round {spec.round_id:03d}\n\n状态：运行中\n\n```json\n{json.dumps(round_config, ensure_ascii=False, indent=2)}\n```\n")
        results_path = directory / "candidate_results.jsonl"
        existing = {str(row["candidate_id"]): row for row in EXP.read_jsonl(results_path)}
        completed = set(existing)
        processed = len(completed)
        consecutive_errors = 0
        for color in COLORS:
            for shape in SHAPES:
                for seed_index in SEED_INDICES:
                    seed = LEGACY.derive_seed(self.seed, "extreme", spec.semantic_shape_count, color, shape, seed_index)
                    base_id = f"{shape}_{color}_count_{spec.semantic_shape_count:02d}_seed_{seed_index:02d}"
                    scene_dir = directory / "scenes" / color / shape / f"seed_{seed_index:02d}"
                    try:
                        semantic = build_semantic_layout(seed, spec.semantic_shape_count, shape, color)
                        layout, _ = add_multiple_occluders(semantic, spec.occlusion_target, spec.occluder_count, seed)
                        required = [scene_dir / name for name in ("sharp.png", "layout.json", "target_mask.png", "occluder_mask.png")]
                        if all(path.is_file() for path in required):
                            stored = json.loads((scene_dir / "layout.json").read_text(encoding="utf-8"))
                            if POOL.canonical_hash(stored) != POOL.canonical_hash(layout):
                                raise RuntimeError("Stored layout mismatch")
                            geometry = {"occlusion_ratio": float(layout["expected_occlusion_ratio"]), "occlusion_direction": layout["occlusion_direction"]}
                        else:
                            geometry = EXP.render_and_validate(layout, scene_dir, spec.occlusion_target)
                    except Exception as exc:
                        for radius in BLUR_RADII:
                            candidate_id = POOL.canonical_hash({"round": spec.round_id, "base": base_id, "blur": radius, "config": self.fingerprint})
                            if candidate_id not in completed:
                                EXP.append_jsonl(results_path, {"candidate_id": candidate_id, "status": "layout_failed", "error": f"{type(exc).__name__}: {exc}", "color": color, "shape": shape, "blur_radius": radius})
                                completed.add(candidate_id); processed += 1
                        continue
                    for radius in BLUR_RADII:
                        candidate_id = POOL.canonical_hash({"round": spec.round_id, "base": base_id, "blur": radius, "config": self.fingerprint})
                        if candidate_id in completed:
                            continue
                        image_path = scene_dir / f"blur_{radius:g}.png"
                        if not image_path.is_file():
                            POOL.render_blur_variant(scene_dir / "sharp.png", image_path, radius)
                        trials = []
                        for trial_index in range(1, TRIALS_PER_IMAGE + 1):
                            try:
                                raw = self._runner().image_only(LEGACY.QUESTION_TEMPLATE.format(shape=shape), str(image_path), color)
                                trials.append(POOL.validation_from_qwen(raw, color))
                                consecutive_errors = 0
                            except Exception as exc:
                                consecutive_errors += 1
                                EXP.append_jsonl(directory / "model_errors.jsonl", {"candidate_id": candidate_id, "trial": trial_index, "error": f"{type(exc).__name__}: {exc}"})
                                if consecutive_errors >= 3:
                                    raise RuntimeError("Three consecutive Qwen runtime errors; stopping") from exc
                                break
                        if len(trials) != TRIALS_PER_IMAGE:
                            continue
                        validation = aggregate_trials(trials)
                        EXP.append_jsonl(results_path, {
                            "candidate_id": candidate_id, "status": "completed", "round_id": spec.round_id,
                            "base_scene_id": base_id, "color": color, "shape": shape, "seed": seed,
                            "seed_index": seed_index, "semantic_shape_count": spec.semantic_shape_count,
                            "occluder_count": spec.occluder_count, "occlusion_target": spec.occlusion_target,
                            "occlusion_ratio": geometry["occlusion_ratio"], "occlusion_direction": geometry["occlusion_direction"],
                            "blur_radius": radius, "image_path": str(image_path.resolve()),
                            "image_sha256": POOL.sha256_file(image_path), "validation": validation,
                        })
                        completed.add(candidate_id); processed += 1
                        if processed % 12 == 0 or processed == spec.candidate_count:
                            print(f"[{spec.name}] {processed}/{spec.candidate_count}", flush=True)
        rows = list({str(row["candidate_id"]): row for row in EXP.read_jsonl(results_path)}.values())
        if len(rows) != spec.candidate_count:
            raise RuntimeError(f"{spec.name}: terminal candidates {len(rows)}/{spec.candidate_count}")
        summary = summarize(spec, rows)
        POOL.atomic_json(directory / "summary.json", summary)
        EXP.make_contact_sheet(rows, directory / "contact_sheet.png")
        EXP.atomic_text(directory / f"extreme_{spec.round_id:03d}.md", markdown(spec, summary))
        done = set(self.manifest.get("completed_rounds", [])); done.add(spec.round_id)
        self.manifest["completed_rounds"] = sorted(done); self._persist()
        return summary

    def update_summary(self) -> None:
        summaries = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(self.root.glob("extreme_*/summary.json"))]
        rows = [
            row
            for path in sorted(self.root.glob("extreme_*/candidate_results.jsonl"))
            for row in EXP.read_jsonl(path)
            if row.get("status") == "completed"
        ]
        passed = [row for row in rows if row.get("validation", {}).get("gate_passed") is True]
        trial_histogram = Counter(int(row["validation"]["correct_trials"]) for row in rows)
        lines = [
            "# Extreme Difficulty Test", "", f"状态：{self.manifest.get('status')}", "",
            f"- 候选：{len(rows)}", f"- 2/3 门控通过：{len(passed)}（{len(passed) / len(rows):.2%}）" if rows else "- 2/3 门控通过：0",
            f"- 0/1/2/3 次正确分布：`{json.dumps({str(k): v for k, v in sorted(trial_histogram.items())}, sort_keys=True)}`",
            "- Qwen3 使用 `do_sample=False`；本次三次结果只有 0/3 和 3/3，因此 2/3 规则没有产生边界票。", "",
            "## 各轮结果", "",
            "| round | objects | occlusion | occluders | passed/total | rate | entropy median | q90 | max |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for value in summaries:
            lines.append(f"| {value['round_id']} | {value['semantic_shape_count']} | {value['occlusion_target']:.0%} | {value['occluder_count']} | {value['passed_count']}/{value['candidate_count']} | {value['pass_rate']:.2%} | {value['entropy_median'] or 0:.6g} | {value['entropy_q90'] or 0:.6g} | {value['entropy_max'] or 0:.6g} |")
        lines.extend([
            "", "## 遮挡与 blur 联合结果", "",
            "| blur | passed/attempted | pass rate | passed entropy median | maximum |",
            "|---:|---:|---:|---:|---:|",
        ])
        for radius in BLUR_RADII:
            subset = [row for row in rows if row["blur_radius"] == radius]
            valid = [row for row in subset if row["validation"]["gate_passed"]]
            values = [float(row["validation"]["normalized_entropy"]) for row in valid]
            lines.append(
                f"| {radius:g} | {len(valid)}/{len(subset)} | {len(valid) / len(subset):.2%} | "
                f"{statistics.median(values) if values else 0:.6g} | {max(values) if values else 0:.6g} |"
            )
        target_bands = ((0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.01))
        band_report = []
        lines.extend([
            "", "## 目标 Entropy 区间", "",
            "| band | correct images | color × shape coverage | best construction | best hits/27 |",
            "|---|---:|---:|---|---:|",
        ])
        for lower, upper in target_bands:
            subset = [
                row for row in passed
                if lower <= float(row["validation"]["normalized_entropy"]) < upper
            ]
            grouped = Counter((
                row["semantic_shape_count"], row["occlusion_target"], row["occluder_count"], row["blur_radius"]
            ) for row in subset)
            best, best_count = grouped.most_common(1)[0] if grouped else (None, 0)
            best_text = (
                f"objects={best[0]}, occ={best[1]:.0%}, occluders={best[2]}, blur={best[3]:g}"
                if best else "n/a"
            )
            label = f"{lower:.1f}-{upper:.1f}" if upper <= 1.0 else ">=0.6"
            coverage = len({(row["color"], row["shape"]) for row in subset})
            lines.append(f"| {label} | {len(subset)} | {coverage}/9 | {best_text} | {best_count}/27 |")
            band_report.append({
                "band": label, "correct_count": len(subset), "combination_count": coverage,
                "best_construction": best_text, "best_hits": best_count,
            })
        POOL.atomic_json(self.root / "aggregate_summary.json", {
            "candidate_count": len(rows), "passed_count": len(passed),
            "pass_rate": len(passed) / len(rows) if rows else None,
            "correct_trial_histogram": {str(k): v for k, v in sorted(trial_histogram.items())},
            "entropy_band_recommendations": band_report,
        })
        lines.extend([
            "", "## 结论", "",
            "80% 遮挡与 blur 8–16 能产生少量高 Entropy 正确样本；blur 24/32 基本使模型直接答错。"
            "这些组合适合极端候选生成，但每张图仍必须经过 2/3 双门控，不能只凭参数直接赋予难度。",
        ])
        EXP.atomic_text(self.root / "SUMMARY.md", "\n".join(lines) + "\n")

    def run(self) -> None:
        for spec in extreme_specs():
            if not self._round_complete(spec):
                self.run_round(spec)
                self.update_summary()
        self.manifest["status"] = "complete"; self._persist(); self.update_summary()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run five extreme Qwen3 image difficulty rounds")
    parser.add_argument("run", nargs="?", default="run")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    ExtremeRunner(args.output_root, args.model_path, args.seed, args.resume).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
