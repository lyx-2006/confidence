#!/usr/bin/env python3
"""Run a resumable Qwen3 image-difficulty experiment matrix.

The experiment varies semantic shape count, measured target occlusion, and
Gaussian blur while keeping the model, prompt, colour order, and semantic
layout deterministic.  Every round owns a durable measurement ledger and a
Markdown report; global CSV/Markdown reports are refreshed after each round.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import os
import random
import statistics
import sys
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageChops, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
ROOT = PROJECT_DIR.parent
DEFAULT_ROOT = PROJECT_DIR / "datasets" / "legacy" / "experiments" / "image_difficulty_experiments"
DEFAULT_MODEL = ROOT / "qwen-3-vl" / "model"


def _load_pool() -> Any:
    path = SCRIPT_DIR / "generate_image_pool.py"
    name = "_qwen3_image_pool_experiment_support"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import image-pool support: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


POOL = _load_pool()
LEGACY = POOL.LEGACY
COLORS = ("red", "white", "black")
SHAPES = ("circle", "triangle", "star")
BLUR_RADII = (0.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0)
SHAPE_COUNTS = (1, 4, 7, 10)
OCCLUSION_LEVELS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60)
BASE_SEED_INDICES = (1, 2, 3)
ENTROPY_CUTOFFS = (0.01, 0.05, 0.10, 0.20, 0.30)
LEVELS = ("very_easy", "easy", "medium", "hard")
PROFILE_FOR_COUNT = {1: "single_clear", 4: "sparse_clear", 7: "medium_clear", 10: "dense_clear"}


@dataclass(frozen=True)
class RoundSpec:
    round_id: int
    shape_count: int
    occlusion_target: float
    seed_indices: tuple[int, ...] = BASE_SEED_INDICES
    phase: str = "main"

    @property
    def name(self) -> str:
        percent = int(round(self.occlusion_target * 100))
        return f"round_{self.round_id:03d}_count_{self.shape_count:02d}_occ_{percent:02d}"

    @property
    def candidate_count(self) -> int:
        return len(COLORS) * len(SHAPES) * len(self.seed_indices) * len(BLUR_RADII)


def main_round_specs() -> list[RoundSpec]:
    specs = [RoundSpec(1, 1, 0.0)]
    specs.extend(RoundSpec(index + 2, count, 0.0) for index, count in enumerate((4, 7, 10)))
    next_id = 5
    for count in (4, 7, 10):
        for occlusion in OCCLUSION_LEVELS:
            specs.append(RoundSpec(next_id, count, occlusion))
            next_id += 1
    assert len(specs) == 22 and sum(spec.candidate_count for spec in specs) == 4752
    return specs


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    POOL.append_jsonl(path, value)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return POOL.load_jsonl(path)


def quantile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None

    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda index: values[index])
        result = [0.0] * len(values)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and values[order[end]] == values[order[start]]:
                end += 1
            rank = (start + end - 1) / 2.0
            for position in range(start, end):
                result[order[position]] = rank
            start = end
        return result

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    numerator = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    dx = sum((x - mx) ** 2 for x in rx)
    dy = sum((y - my) ** 2 for y in ry)
    if dx == 0 or dy == 0:
        return None
    return numerator / math.sqrt(dx * dy)


def base_seed(global_seed: int, count: int, color: str, shape: str, seed_index: int) -> int:
    return LEGACY.derive_seed(global_seed, "difficulty-experiment", count, color, shape, seed_index)


def build_semantic_layout(seed: int, count: int, shape: str, color: str) -> dict[str, Any]:
    layout = POOL.build_pool_layout(seed, PROFILE_FOR_COUNT[count], shape, color)
    layout = json.loads(json.dumps(layout))
    layout["difficulty"] = "entropy_experiment"
    layout["profile"] = f"count_{count:02d}_clear"
    layout["expected_occlusion_ratio"] = 0.0
    return layout


def _direction(seed: int) -> str:
    return random.Random(LEGACY.derive_seed(seed, "fixed-occlusion-direction")).choice(
        ("left", "right", "top", "bottom")
    )


def add_fixed_direction_occluder(
    semantic_layout: dict[str, Any], target_ratio: float, seed: int
) -> tuple[dict[str, Any], float, str]:
    layout = json.loads(json.dumps(semantic_layout))
    if target_ratio == 0.0:
        layout["profile"] = f"count_{layout['semantic_shape_count']:02d}_occ_00"
        return layout, 0.0, "none"
    target = next(obj for obj in layout["objects"] if obj.get("role") == "target")
    target_mask = LEGACY._object_mask(target).convert("1")
    bounds = target_mask.getbbox()
    if bounds is None:
        raise ValueError("Target mask is empty")
    left, top, right, bottom = bounds
    direction = _direction(seed)
    limit = (right - left) if direction in {"left", "right"} else (bottom - top)
    distractor_colors = [value for value in POOL.COLORS if value != layout["target_color"]]
    occluder_color = random.Random(LEGACY.derive_seed(seed, "fixed-occlusion-color")).choice(distractor_colors)
    candidates: list[tuple[float, dict[str, Any]]] = []
    for amount in range(1, limit + 1):
        if direction == "left":
            bbox = (left, top, left + amount, bottom)
        elif direction == "right":
            bbox = (right - amount, top, right, bottom)
        elif direction == "top":
            bbox = (left, top, right, top + amount)
        else:
            bbox = (left, bottom - amount, right, bottom)
        occluder = LEGACY._occluder_candidate("rectangle", occluder_color, tuple(map(float, bbox)))
        ratio = LEGACY._mask_overlap_ratio(target_mask, LEGACY._object_mask(occluder))
        candidates.append((ratio, occluder))
    measured, occluder = min(candidates, key=lambda item: abs(item[0] - target_ratio))
    if abs(measured - target_ratio) > 0.02:
        raise ValueError(f"Occlusion solver missed target {target_ratio:.2f}: measured {measured:.4f}")
    occluder["expected_mask_occlusion_ratio"] = measured
    layout["objects"].append(occluder)
    layout["occluder_geometry"] = [
        {key: occluder[key] for key in ("shape", "center", "bbox", "rotation", "size")}
    ]
    layout["expected_occlusion_ratio"] = measured
    layout["profile"] = (
        f"count_{layout['semantic_shape_count']:02d}_occ_{int(round(target_ratio * 100)):02d}"
    )
    layout["occlusion_direction"] = direction
    return layout, measured, direction


def render_and_validate(layout: dict[str, Any], destination: Path, target_ratio: float) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    style_rng = random.Random(LEGACY.derive_seed(int(layout["case_seed"]), "render-style"))
    style = {
        "background_rgb": list(style_rng.choice(LEGACY.ALLOWED_BACKGROUNDS)),
        "outline_rgb": list(LEGACY.DEFAULT_RENDER_STYLE["outline_rgb"]),
        "outline_width": int(LEGACY.DEFAULT_RENDER_STYLE["outline_width"]),
    }
    with tempfile.TemporaryDirectory(prefix="entropy-round-render-") as temporary:
        rendered = LEGACY.render_scene_locally(layout, Path(temporary), style)
        for source, target in {
            "image.png": "sharp.png",
            "layout.json": "layout.json",
            "target_mask.png": "target_mask.png",
            "occluder_mask.png": "occluder_mask.png",
        }.items():
            POOL.atomic_copy(rendered / source, destination / target)
    with Image.open(destination / "target_mask.png") as target_image, Image.open(
        destination / "occluder_mask.png"
    ) as occluder_image:
        target_mask = target_image.convert("1")
        occluder_mask = occluder_image.convert("1")
        target_area = target_mask.histogram()[255]
        intersection = ImageChops.logical_and(target_mask, occluder_mask).histogram()[255]
    measured = intersection / target_area if target_area else 0.0
    semantic = [obj for obj in layout["objects"] if obj.get("role") in {"target", "distractor"}]
    issues: list[str] = []
    if target_area == 0:
        issues.append("empty_target_mask")
    if abs(measured - target_ratio) > 0.02:
        issues.append("occlusion_tolerance")
    if len(semantic) != int(layout["semantic_shape_count"]):
        issues.append("semantic_shape_count")
    if len([obj for obj in semantic if obj.get("role") == "target"]) != 1:
        issues.append("target_count")
    if len({obj["shape"] for obj in semantic}) != len(semantic):
        issues.append("semantic_shape_uniqueness")
    if issues:
        raise ValueError(f"Rendered scene validation failed: {issues}")
    return {
        "valid": True,
        "semantic_shape_count": len(semantic),
        "rendered_object_count": len(layout["objects"]),
        "occlusion_ratio": measured,
        "occlusion_direction": layout.get("occlusion_direction", "none"),
    }


def round_change_description(spec: RoundSpec) -> str:
    specs = {item.round_id: item for item in main_round_specs()}
    previous = specs.get(spec.round_id - 1)
    if previous is None:
        return "无（基线轮）"
    changes = []
    if previous.shape_count != spec.shape_count:
        changes.append(f"语义图形数 {previous.shape_count} → {spec.shape_count}")
    if previous.occlusion_target != spec.occlusion_target:
        changes.append(f"目标遮挡率 {previous.occlusion_target:.0%} → {spec.occlusion_target:.0%}")
    if previous.seed_indices != spec.seed_indices:
        changes.append(f"seed index {previous.seed_indices} → {spec.seed_indices}")
    return "；".join(changes) if changes else "仅重复测量配置"


def round_design_markdown(spec: RoundSpec, fingerprint: str, global_seed: int) -> str:
    previous = round_change_description(spec)
    return f"""# Round {spec.round_id:03d}: shape count={spec.shape_count}, occlusion={spec.occlusion_target:.0%}

状态：运行中

## 本轮目的

测量 `{spec.shape_count}` 个语义图形、目标实测遮挡率 `{spec.occlusion_target:.0%}` 时，blur 与 Qwen3 normalized Entropy 的联合变化。相对上一轮改变：{previous}。

## 固定配置

- 颜色：{', '.join(COLORS)}
- 图形：{', '.join(SHAPES)}
- 全局 seed：{global_seed}
- 每个 color × shape 的 seed index：{', '.join(map(str, spec.seed_indices))}
- blur radius：{', '.join(f'{value:g}' for value in BLUR_RADII)}
- 计划候选：{spec.candidate_count}
- 遮挡允许误差：±2 个百分点
- 配置指纹：`{fingerprint}`

## 结果

运行完成后自动补充。
"""


def successful_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if row.get("status") == "completed"
        and row.get("validation", {}).get("gate_passed") is True
        and row.get("validation", {}).get("normalized_entropy") is not None
    ]


def summarize_round(spec: RoundSpec, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    correct = successful_rows(rows)
    entropies = [float(row["validation"]["normalized_entropy"]) for row in correct]
    by_blur: dict[str, dict[str, Any]] = {}
    for radius in BLUR_RADII:
        subset = [row for row in rows if row.get("blur_radius") == radius]
        passed = successful_rows(subset)
        values = [float(row["validation"]["normalized_entropy"]) for row in passed]
        by_blur[f"{radius:g}"] = {
            "attempted": len(subset),
            "passed": len(passed),
            "accuracy": len(passed) / len(subset) if subset else None,
            "entropy_median": statistics.median(values) if values else None,
        }
    thresholds: dict[str, dict[str, int]] = {}
    for cutoff in ENTROPY_CUTOFFS:
        subset = [row for row in correct if float(row["validation"]["normalized_entropy"]) >= cutoff]
        thresholds[f"{cutoff:.2f}"] = {
            "count": len(subset),
            "combination_count": len({(row["color"], row["shape"]) for row in subset}),
        }
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("validation", {}).get("normalized_entropy") is not None:
            groups[str(row["base_scene_id"])].append(row)
    correlations: list[float] = []
    for group in groups.values():
        ordered = sorted(group, key=lambda row: float(row["blur_radius"]))
        correlation = spearman(
            [float(row["blur_radius"]) for row in ordered],
            [float(row["validation"]["normalized_entropy"]) for row in ordered],
        )
        if correlation is not None:
            correlations.append(correlation)
    measured_occ = [float(row["occlusion_ratio"]) for row in rows if row.get("occlusion_ratio") is not None]
    return {
        "round_id": spec.round_id,
        "round_name": spec.name,
        "phase": spec.phase,
        "shape_count": spec.shape_count,
        "occlusion_target": spec.occlusion_target,
        "occlusion_measured_median": statistics.median(measured_occ) if measured_occ else None,
        "candidate_count": len(rows),
        "completed_count": sum(row.get("status") == "completed" for row in rows),
        "correct_count": len(correct),
        "accuracy": len(correct) / len(rows) if rows else None,
        "entropy_q25": quantile(entropies, 0.25),
        "entropy_median": quantile(entropies, 0.50),
        "entropy_q75": quantile(entropies, 0.75),
        "entropy_q90": quantile(entropies, 0.90),
        "entropy_max": max(entropies) if entropies else None,
        "blur_entropy_spearman_mean": statistics.mean(correlations) if correlations else None,
        "blur_entropy_spearman_median": statistics.median(correlations) if correlations else None,
        "by_blur": by_blur,
        "entropy_cutoffs": thresholds,
        "failure_count": len(rows) - len(correct),
        "status_counts": dict(sorted(defaultdict(int, {
            status: sum(row.get("status") == status for row in rows)
            for status in {str(row.get("status")) for row in rows}
        }).items())),
    }


def round_result_markdown(
    spec: RoundSpec,
    fingerprint: str,
    summary: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    global_seed: int,
) -> str:
    def number(value: Any, digits: int = 6) -> str:
        return "n/a" if value is None else f"{float(value):.{digits}g}"

    blur_lines = ["| blur | 通过/尝试 | 正确率 | 正确样本 Entropy median |", "|---:|---:|---:|---:|"]
    for radius, value in summary["by_blur"].items():
        blur_lines.append(
            f"| {radius} | {value['passed']}/{value['attempted']} | {value['accuracy']:.2%} | {number(value['entropy_median'])} |"
        )
    cutoff_lines = ["| Entropy 下限 | 正确样本数 | color × shape 覆盖 |", "|---:|---:|---:|"]
    for cutoff, value in summary["entropy_cutoffs"].items():
        cutoff_lines.append(f"| {cutoff} | {value['count']} | {value['combination_count']}/9 |")
    next_plan = "进入下一轮矩阵配置。" if spec.round_id < 22 else "汇总主实验并检查四档阈值；不足时追加最多三轮。"
    correct = successful_rows(rows)
    representatives = sorted(
        correct, key=lambda row: float(row["validation"]["normalized_entropy"]), reverse=True
    )[:3]
    failures = [row for row in rows if row.get("status") != "completed" or row not in correct][:3]
    seed_rows = {}
    for row in rows:
        if row.get("base_scene_id") and row.get("seed") is not None:
            seed_rows[str(row["base_scene_id"])] = int(row["seed"])
    seed_lines = ["| base_scene_id | actual seed |", "|---|---:|"]
    seed_lines.extend(f"| {scene_id} | {seed} |" for scene_id, seed in sorted(seed_rows.items()))
    representative_lines = ["| image | Answer | Entropy | blur |", "|---|---|---:|---:|"]
    representative_lines.extend(
        f"| `{Path(row['image_path']).relative_to(Path(row['image_path']).parents[4])}` | {row['color']} | "
        f"{float(row['validation']['normalized_entropy']):.8g} | {row['blur_radius']:g} |"
        for row in representatives
    )
    if failures:
        failure_lines = ["| image | prediction / restricted | Entropy | reasons |", "|---|---|---:|---|"]
        for row in failures:
            validation = row.get("validation", {})
            image = row.get("image_path", "n/a")
            try:
                image = str(Path(image).relative_to(Path(image).parents[4]))
            except (ValueError, IndexError):
                pass
            failure_lines.append(
                f"| `{image}` | {validation.get('prediction')} / {validation.get('restricted_top1')} | "
                f"{number(validation.get('normalized_entropy'))} | "
                f"{', '.join(validation.get('failure_reasons', [])) or row.get('status')} |"
            )
    else:
        failure_lines = ["本轮没有失败样本。"]
    conclusion = (
        f"本轮双门控正确率为 {summary['accuracy']:.2%}；正确样本 Entropy median="
        f"{number(summary['entropy_median'])}、q90={number(summary['entropy_q90'])}、"
        f"maximum={number(summary['entropy_max'])}。这些数值描述实测范围，不预先指定难度标签。"
    )
    return f"""# Round {spec.round_id:03d}: shape count={spec.shape_count}, occlusion={spec.occlusion_target:.0%}

状态：已完成

## 本轮构造

- 语义图形数：{spec.shape_count}
- 目标遮挡率：目标 {spec.occlusion_target:.0%}，实测中位数 {summary['occlusion_measured_median']:.2%}
- blur radius：{', '.join(f'{value:g}' for value in BLUR_RADII)}
- 全局 seed：{global_seed}
- seed index：{', '.join(map(str, spec.seed_indices))}
- 候选：{summary['candidate_count']}；正确门控：{summary['correct_count']}；正确率：{summary['accuracy']:.2%}
- 配置指纹：`{fingerprint}`
- 相对上一轮：{round_change_description(spec)}

### 实际基础 scene seed

{chr(10).join(seed_lines)}

## Entropy 分布（只统计双门控正确样本）

| q25 | median | q75 | q90 | maximum |
|---:|---:|---:|---:|---:|
| {number(summary['entropy_q25'])} | {number(summary['entropy_median'])} | {number(summary['entropy_q75'])} | {number(summary['entropy_q90'])} | {number(summary['entropy_max'])} |

## Blur 分组

{chr(10).join(blur_lines)}

## 可用高 Entropy 覆盖

{chr(10).join(cutoff_lines)}

## 同源场景关系

blur 与 Entropy 的 Spearman：平均 `{number(summary['blur_entropy_spearman_mean'])}`，中位数 `{number(summary['blur_entropy_spearman_median'])}`。相关性仅记录实测关系，不作为正确性门控。

## 失败与样本

- 未通过或非 completed：{summary['failure_count']}
- 状态计数：`{json.dumps(summary['status_counts'], ensure_ascii=False, sort_keys=True)}`
- 本目录的 `contact_sheet.png` 包含各 blur 代表样本、高 Entropy 正确样本和失败样本。

### 代表样本（正确样本中 Entropy 最高的 3 张）

{chr(10).join(representative_lines)}

### 失败样本（最多 3 张）

{chr(10).join(failure_lines)}

## 本轮结论

{conclusion}

## 下一轮

{next_plan}
"""


def make_contact_sheet(rows: Sequence[dict[str, Any]], output: Path) -> None:
    correct = successful_rows(rows)
    incorrect = [row for row in rows if row.get("status") == "completed" and row not in correct]
    selected: list[dict[str, Any]] = []
    for radius in BLUR_RADII:
        subset = [row for row in correct if row.get("blur_radius") == radius]
        if subset:
            ordered = sorted(subset, key=lambda row: float(row["validation"]["normalized_entropy"]))
            selected.extend([ordered[len(ordered) // 2], ordered[-1]])
    selected.extend(sorted(correct, key=lambda row: float(row["validation"]["normalized_entropy"]), reverse=True)[:12])
    selected.extend(incorrect[:8])
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in selected:
        candidate_id = str(row["candidate_id"])
        if candidate_id not in seen and Path(str(row.get("image_path", ""))).is_file():
            seen.add(candidate_id)
            unique.append(row)
    cell_w, cell_h, columns = 220, 250, 6
    row_count = max(1, math.ceil(min(36, len(unique)) / columns))
    sheet = Image.new("RGB", (columns * cell_w, row_count * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(unique[:36]):
        x, y = (index % columns) * cell_w, (index // columns) * cell_h
        with Image.open(row["image_path"]) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((cell_w - 12, cell_h - 44), Image.Resampling.LANCZOS)
        sheet.paste(thumb, (x + (cell_w - thumb.width) // 2, y + 4))
        entropy = row.get("validation", {}).get("normalized_entropy")
        label = f"{row['shape']}/{row['color']} H={float(entropy):.3g} b={row['blur_radius']:g}" if entropy is not None else f"{row['shape']}/{row['color']} failed"
        draw.text((x + 4, y + cell_h - 34), label, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG")


def make_global_boundary_sheet(
    rows: Sequence[dict[str, Any]], thresholds: Sequence[float], output: Path
) -> None:
    correct = successful_rows(rows)
    selected: list[dict[str, Any]] = []
    remaining = list(correct)
    for threshold in thresholds:
        nearest = sorted(
            remaining,
            key=lambda row: abs(float(row["validation"]["normalized_entropy"]) - threshold),
        )[:12]
        selected.extend(nearest)
        used = {row["candidate_id"] for row in nearest}
        remaining = [row for row in remaining if row["candidate_id"] not in used]
    cell_w, cell_h, columns = 220, 260, 6
    sheet = Image.new("RGB", (columns * cell_w, 6 * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(selected[:36]):
        x, y = (index % columns) * cell_w, (index // columns) * cell_h
        with Image.open(row["image_path"]) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((cell_w - 12, cell_h - 54), Image.Resampling.LANCZOS)
        sheet.paste(thumb, (x + (cell_w - thumb.width) // 2, y + 4))
        entropy = float(row["validation"]["normalized_entropy"])
        draw.text((x + 4, y + cell_h - 46), f"{row['shape']}/{row['color']} H={entropy:.3g}", fill="black")
        draw.text(
            (x + 4, y + cell_h - 30),
            f"n={row['semantic_shape_count']} occ={row['occlusion_target']:.0%} b={row['blur_radius']:g}",
            fill="black",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG")


def threshold_diagnostics(rows: Sequence[dict[str, Any]], thresholds: Sequence[float]) -> dict[str, Any]:
    levels: dict[str, dict[str, Any]] = {}
    medians: list[float | None] = []
    for level in LEVELS:
        subset = [
            row for row in rows
            if POOL.entropy_level(float(row["validation"]["normalized_entropy"]), thresholds) == level
        ]
        radii = [float(row["blur_radius"]) for row in subset]
        median_blur = statistics.median(radii) if radii else None
        medians.append(median_blur)
        levels[level] = {
            "count": len(subset),
            "combination_count": len({(row["color"], row["shape"]) for row in subset}),
            "median_blur_radius": median_blur,
        }
    known = [value for value in medians if value is not None]
    monotonic = len(known) == 4 and all(a <= b for a, b in zip(known, known[1:]))
    eligible = (
        all(levels[level]["count"] >= 20 for level in LEVELS)
        and all(levels[level]["combination_count"] >= 6 for level in LEVELS)
        and monotonic
    )
    return {"thresholds": list(thresholds), "levels": levels, "median_blur_monotonic": monotonic, "eligible": eligible}


def propose_thresholds(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    correct = successful_rows(rows)
    entropies = [float(row["validation"]["normalized_entropy"]) for row in correct]
    candidates = sorted({
        float(value) for fraction in [index / 20 for index in range(1, 20)]
        if (value := quantile(entropies, fraction)) is not None and 0.0 < value < 1.0
    })
    best: tuple[tuple[Any, ...], dict[str, Any]] | None = None
    target = len(correct) / 4 if correct else 0
    for thresholds in itertools.combinations(candidates, 3):
        diagnostics = threshold_diagnostics(correct, thresholds)
        counts = [diagnostics["levels"][level]["count"] for level in LEVELS]
        coverages = [diagnostics["levels"][level]["combination_count"] for level in LEVELS]
        score = (
            int(diagnostics["eligible"]),
            int(diagnostics["median_blur_monotonic"]),
            min(coverages),
            min(counts),
            -sum(abs(count - target) for count in counts),
        )
        if best is None or score > best[0]:
            best = (score, diagnostics)
    if best is None:
        return {"thresholds": None, "levels": {}, "eligible": False, "reason": "fewer_than_three_distinct_entropy_cut_points"}
    return best[1]


def recommended_constructions(
    rows: Sequence[dict[str, Any]], thresholds: Sequence[float], limit: int = 5
) -> dict[str, list[dict[str, Any]]]:
    """Rank measured constructions by usable yield for each entropy level."""
    grouped: dict[tuple[int, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "completed":
            grouped[(
                int(row["semantic_shape_count"]),
                float(row["occlusion_target"]),
                float(row["blur_radius"]),
            )].append(row)
    recommendations: dict[str, list[dict[str, Any]]] = {}
    for level in LEVELS:
        values: list[dict[str, Any]] = []
        for (count, occlusion, blur), group in grouped.items():
            correct = successful_rows(group)
            matching = [
                row for row in correct
                if POOL.entropy_level(float(row["validation"]["normalized_entropy"]), thresholds) == level
            ]
            if not matching:
                continue
            values.append({
                "semantic_shape_count": count,
                "occlusion_target": occlusion,
                "blur_radius": blur,
                "attempted": len(group),
                "gate_correct": len(correct),
                "level_count": len(matching),
                "usable_rate": len(matching) / len(group),
                "level_purity_among_correct": len(matching) / len(correct),
                "entropy_median": statistics.median(
                    float(row["validation"]["normalized_entropy"]) for row in matching
                ),
            })
        values.sort(
            key=lambda value: (
                value["usable_rate"], value["level_count"], value["level_purity_among_correct"]
            ),
            reverse=True,
        )
        recommendations[level] = values[:limit]
    return recommendations


SUMMARY_FIELDS = (
    "round_id", "round_name", "phase", "shape_count", "occlusion_target",
    "occlusion_measured_median", "candidate_count", "completed_count", "correct_count",
    "accuracy", "entropy_q25", "entropy_median", "entropy_q75", "entropy_q90",
    "entropy_max", "blur_entropy_spearman_mean", "blur_entropy_spearman_median",
    "failure_count",
)


class ExperimentRunner:
    def __init__(self, root: Path, model_path: Path, seed: int, resume: bool, runner: Any | None = None) -> None:
        self.root = root.resolve()
        self.model_path = model_path.resolve()
        self.seed = int(seed)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "experiment_manifest.json"
        self.config = {
            "schema_version": "image_difficulty_experiment.v1",
            "seed": self.seed,
            "model": POOL.model_fingerprint(self.model_path),
            "faithful_runtime_sha256": POOL.sha256_file(POOL.FAITHFUL_DIR / "runtime.py"),
            "faithful_prompts_sha256": POOL.sha256_file(POOL.FAITHFUL_DIR / "prompts.py"),
            "colors": list(COLORS), "shapes": list(SHAPES), "blur_radii": list(BLUR_RADII),
            "shape_counts": list(SHAPE_COUNTS), "occlusion_levels": list(OCCLUSION_LEVELS),
            "seed_indices": list(BASE_SEED_INDICES), "entropy_definition": POOL.ENTROPY_DEFINITION,
        }
        self.fingerprint = POOL.canonical_hash(self.config)
        if self.manifest_path.is_file():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if manifest.get("config_fingerprint") != self.fingerprint:
                raise RuntimeError("Experiment configuration differs from existing manifest")
            if not resume:
                raise RuntimeError("Experiment exists; pass --resume")
            self.manifest = manifest
        else:
            self.manifest = {
                "schema_version": "image_difficulty_experiment.state.v1",
                "config_fingerprint": self.fingerprint,
                "completed_rounds": [],
                "main_round_count": 22,
                "main_candidate_count": 4752,
                "status": "running",
            }
            self._persist_manifest()
        self._runner_value = runner

    def _persist_manifest(self) -> None:
        POOL.atomic_json(self.manifest_path, self.manifest)

    def _runner(self) -> Any:
        if self._runner_value is None:
            self._runner_value = POOL.load_qwen3_runner(self.model_path)
        return self._runner_value

    def _round_config(self, spec: RoundSpec) -> dict[str, Any]:
        return {
            "experiment_fingerprint": self.fingerprint,
            "round": asdict(spec),
            "colors": list(COLORS), "shapes": list(SHAPES), "blur_radii": list(BLUR_RADII),
            "occlusion_tolerance": 0.02,
        }

    def _round_dir(self, spec: RoundSpec) -> Path:
        return self.root / spec.name

    def _write_round_start(self, spec: RoundSpec) -> str:
        directory = self._round_dir(spec)
        directory.mkdir(parents=True, exist_ok=True)
        config = self._round_config(spec)
        fingerprint = POOL.canonical_hash(config)
        config_path = directory / "config.json"
        if config_path.is_file():
            existing = json.loads(config_path.read_text(encoding="utf-8"))
            if POOL.canonical_hash(existing) != fingerprint:
                raise RuntimeError(f"Round configuration changed: {spec.name}")
        else:
            POOL.atomic_json(config_path, config)
        report = directory / f"round_{spec.round_id:03d}.md"
        if not report.is_file():
            atomic_text(report, round_design_markdown(spec, fingerprint, self.seed))
        return fingerprint

    def _scene_layout(self, spec: RoundSpec, color: str, shape: str, seed_index: int) -> tuple[dict[str, Any], float, str, int]:
        seed = base_seed(self.seed, spec.shape_count, color, shape, seed_index)
        semantic = build_semantic_layout(seed, spec.shape_count, shape, color)
        layout, measured, direction = add_fixed_direction_occluder(semantic, spec.occlusion_target, seed)
        return layout, measured, direction, seed

    def run_round(self, spec: RoundSpec) -> dict[str, Any]:
        fingerprint = self._write_round_start(spec)
        directory = self._round_dir(spec)
        results_path = directory / "candidate_results.jsonl"
        existing = {str(row.get("candidate_id")): row for row in read_jsonl(results_path) if row.get("candidate_id")}
        completed = {key for key, row in existing.items() if row.get("status") in {"completed", "layout_failed"}}
        consecutive_model_errors = 0
        processed = len(completed)
        for color in COLORS:
            for shape in SHAPES:
                for seed_index in spec.seed_indices:
                    base_id = f"{shape}_{color}_count_{spec.shape_count:02d}_seed_{seed_index:02d}"
                    scene_dir = directory / "scenes" / color / shape / f"seed_{seed_index:02d}"
                    try:
                        layout, _expected, _direction_value, seed = self._scene_layout(spec, color, shape, seed_index)
                        required = [scene_dir / name for name in ("sharp.png", "layout.json", "target_mask.png", "occluder_mask.png")]
                        if all(path.is_file() for path in required):
                            stored = json.loads((scene_dir / "layout.json").read_text(encoding="utf-8"))
                            if POOL.canonical_hash(stored) != POOL.canonical_hash(layout):
                                raise RuntimeError(f"Stored layout mismatch: {scene_dir}")
                            measured = float(layout["expected_occlusion_ratio"])
                            geometry = {
                                "valid": True,
                                "semantic_shape_count": spec.shape_count,
                                "rendered_object_count": len(layout["objects"]),
                                "occlusion_ratio": measured,
                                "occlusion_direction": layout.get("occlusion_direction", "none"),
                            }
                        else:
                            geometry = render_and_validate(layout, scene_dir, spec.occlusion_target)
                    except Exception as exc:
                        for radius in BLUR_RADII:
                            candidate_id = POOL.canonical_hash({"round": fingerprint, "base": base_id, "blur": radius})
                            if candidate_id in completed:
                                continue
                            append_jsonl(results_path, {
                                "candidate_id": candidate_id, "status": "layout_failed", "round_id": spec.round_id,
                                "base_scene_id": base_id, "color": color, "shape": shape, "seed_index": seed_index,
                                "blur_radius": radius, "occlusion_target": spec.occlusion_target,
                                "error": f"{type(exc).__name__}: {exc}",
                            })
                            completed.add(candidate_id)
                            processed += 1
                        continue
                    for radius in BLUR_RADII:
                        candidate_id = POOL.canonical_hash({"round": fingerprint, "base": base_id, "blur": radius})
                        if candidate_id in completed:
                            continue
                        image_path = scene_dir / f"blur_{radius:g}.png"
                        if not image_path.is_file():
                            POOL.render_blur_variant(scene_dir / "sharp.png", image_path, radius)
                        try:
                            raw = self._runner().image_only(LEGACY.QUESTION_TEMPLATE.format(shape=shape), str(image_path), color)
                            validation = POOL.validation_from_qwen(raw, color)
                            consecutive_model_errors = 0
                        except Exception as exc:
                            consecutive_model_errors += 1
                            append_jsonl(directory / "model_errors.jsonl", {
                                "candidate_id": candidate_id, "error": f"{type(exc).__name__}: {exc}",
                                "consecutive_model_errors": consecutive_model_errors,
                            })
                            if consecutive_model_errors >= 3:
                                raise RuntimeError(
                                    f"Three consecutive model failures in {spec.name}; stopped before classifying images"
                                ) from exc
                            continue
                        record = {
                            "candidate_id": candidate_id, "status": "completed", "round_id": spec.round_id,
                            "round_name": spec.name, "phase": spec.phase, "base_scene_id": base_id,
                            "scene_variant_id": f"{base_id}_occ_{int(round(spec.occlusion_target * 100)):02d}",
                            "color": color, "shape": shape, "seed": seed, "seed_index": seed_index,
                            "semantic_shape_count": spec.shape_count,
                            "occlusion_target": spec.occlusion_target,
                            "occlusion_ratio": geometry["occlusion_ratio"],
                            "occlusion_direction": geometry["occlusion_direction"],
                            "blur_radius": radius, "image_path": str(image_path.resolve()),
                            "image_sha256": POOL.sha256_file(image_path), "validation": validation,
                        }
                        append_jsonl(results_path, record)
                        completed.add(candidate_id)
                        processed += 1
                        if processed % 24 == 0 or processed == spec.candidate_count:
                            print(f"[{spec.name}] {processed}/{spec.candidate_count}", flush=True)
        rows_by_id = {str(row["candidate_id"]): row for row in read_jsonl(results_path)}
        rows = list(rows_by_id.values())
        if len(rows) != spec.candidate_count:
            raise RuntimeError(f"{spec.name} has {len(rows)}/{spec.candidate_count} terminal candidates")
        summary = summarize_round(spec, rows)
        POOL.atomic_json(directory / "summary.json", summary)
        make_contact_sheet(rows, directory / "contact_sheet.png")
        atomic_text(
            directory / f"round_{spec.round_id:03d}.md",
            round_result_markdown(spec, fingerprint, summary, rows, self.seed),
        )
        completed_rounds = set(map(int, self.manifest.get("completed_rounds", [])))
        completed_rounds.add(spec.round_id)
        self.manifest["completed_rounds"] = sorted(completed_rounds)
        self._persist_manifest()
        self.update_global_reports()
        return summary

    def all_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("round_*/candidate_results.jsonl")):
            rows.extend(read_jsonl(path))
        return rows

    def round_summaries(self) -> list[dict[str, Any]]:
        summaries = []
        for path in sorted(self.root.glob("round_*/summary.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                summaries.append(value)
        return sorted(summaries, key=lambda item: int(item["round_id"]))

    def update_global_reports(self, final: bool = False) -> dict[str, Any]:
        summaries = self.round_summaries()
        rows = self.all_rows()
        threshold_result = propose_thresholds(rows) if successful_rows(rows) else {"thresholds": None, "eligible": False}
        flat_rows = [{key: summary.get(key) for key in SUMMARY_FIELDS} for summary in summaries]
        atomic_csv(self.root / "experiment_summary.csv", SUMMARY_FIELDS, flat_rows)
        total = len(rows)
        correct = len(successful_rows(rows))
        correct_rows = successful_rows(rows)
        entropies = [float(row["validation"]["normalized_entropy"]) for row in correct_rows]
        construction_advice = (
            recommended_constructions(rows, threshold_result["thresholds"])
            if threshold_result.get("thresholds") else {}
        )
        if threshold_result.get("thresholds"):
            make_global_boundary_sheet(
                rows, threshold_result["thresholds"], self.root / "global_threshold_contact_sheet.png"
            )
        lines = [
            "# Qwen3 Image Difficulty Exploration", "",
            f"状态：{'主实验与补充实验已结束' if final else '运行中'}", "",
            f"- 已完成轮次：{len(summaries)}", f"- 已记录候选：{total}",
            f"- 双门控正确：{correct}（{correct / total:.2%}）" if total else "- 双门控正确：0", "",
            "## 可用 Entropy 范围", "",
            (
                f"正确样本范围 `{min(entropies):.12g}`–`{max(entropies):.12g}`；"
                f"q25=`{quantile(entropies, 0.25):.12g}`，median=`{quantile(entropies, 0.50):.12g}`，"
                f"q75=`{quantile(entropies, 0.75):.12g}`，q90=`{quantile(entropies, 0.90):.12g}`。"
                if entropies else "尚无正确样本。"
            ), "",
            "## 各轮可用难度", "",
            "| round | shapes | occlusion | correct/total | entropy median | q90 | max |", "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for summary in summaries:
            lines.append(
                f"| {summary['round_id']:03d} | {summary['shape_count']} | {summary['occlusion_target']:.0%} | "
                f"{summary['correct_count']}/{summary['candidate_count']} | {summary['entropy_median'] or 0:.6g} | "
                f"{summary['entropy_q90'] or 0:.6g} | {summary['entropy_max'] or 0:.6g} |"
            )
        lines.extend(["", "## 当前全局阈值候选", "", "```json", json.dumps(threshold_result, ensure_ascii=False, indent=2), "```", ""])
        if construction_advice:
            lines.extend([
                "## 推荐构造参数", "",
                "推荐按该配置在对应档位的可用产出率排序。每种配置在主实验中包含 27 张候选；正式生成仍需逐张通过双门控。", "",
                "| level | shapes | occlusion | blur | usable/attempted | usable rate | level purity | entropy median |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for level in LEVELS:
                for value in construction_advice[level]:
                    lines.append(
                        f"| {level} | {value['semantic_shape_count']} | {value['occlusion_target']:.0%} | "
                        f"{value['blur_radius']:g} | {value['level_count']}/{value['attempted']} | "
                        f"{value['usable_rate']:.2%} | {value['level_purity_among_correct']:.2%} | "
                        f"{value['entropy_median']:.6g} |"
                    )
            lines.append("")
        lines.extend([
            "## 人工边界检查", "",
            "请检查 [global_threshold_contact_sheet.png](global_threshold_contact_sheet.png) 以及各轮 `contact_sheet.png`。"
            "阈值只统计生成答案与 restricted top-1 同时正确的样本；视觉可辨认性需单独确认。",
        ])
        atomic_text(self.root / "SUMMARY.md", "\n".join(lines) + "\n")
        return threshold_result

    def choose_supplement(self, round_id: int, supplement_index: int) -> RoundSpec:
        summaries = [summary for summary in self.round_summaries() if summary.get("phase") == "main"]
        viable = [summary for summary in summaries if int(summary.get("correct_count", 0)) >= 20]
        source = max(
            viable or summaries,
            key=lambda summary: (float(summary.get("entropy_q75") or 0), float(summary.get("accuracy") or 0)),
        )
        start = 4 + supplement_index * 3
        return RoundSpec(
            round_id, int(source["shape_count"]), float(source["occlusion_target"]),
            tuple(range(start, start + 3)), phase="supplement",
        )

    def round_is_complete(self, spec: RoundSpec) -> bool:
        if spec.round_id not in set(map(int, self.manifest.get("completed_rounds", []))):
            return False
        directory = self._round_dir(spec)
        required = (
            directory / "config.json",
            directory / f"round_{spec.round_id:03d}.md",
            directory / "summary.json",
            directory / "contact_sheet.png",
            directory / "candidate_results.jsonl",
        )
        if not all(path.is_file() for path in required):
            return False
        rows = read_jsonl(directory / "candidate_results.jsonl")
        return (
            len({row.get("candidate_id") for row in rows}) == spec.candidate_count
            and all(row.get("status") in {"completed", "layout_failed"} for row in rows)
        )

    def run(self, max_supplements: int = 3) -> None:
        specs = main_round_specs()
        for spec in specs:
            if not self.round_is_complete(spec):
                self.run_round(spec)
        if self.manifest.get("status") == "complete" and all(self.round_is_complete(spec) for spec in specs):
            self.update_global_reports(final=True)
            return
        thresholds = self.update_global_reports()
        supplement_index = 0
        while not thresholds.get("eligible") and supplement_index < max_supplements:
            spec = self.choose_supplement(23 + supplement_index, supplement_index)
            self.run_round(spec)
            supplement_index += 1
            thresholds = self.update_global_reports()
        self.manifest["status"] = "complete"
        self.manifest["threshold_candidate"] = thresholds
        self.manifest["manual_contact_sheet_review_required"] = True
        self._persist_manifest()
        self.update_global_reports(final=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore Qwen3 image difficulty across 22 resumable rounds")
    parser.add_argument("run", nargs="?", default="run")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-supplements", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run != "run":
        raise ValueError("Only the 'run' command is supported")
    if not 0 <= args.max_supplements <= 3:
        raise ValueError("--max-supplements must be between 0 and 3")
    ExperimentRunner(args.output_root, args.model_path, args.seed, args.resume).run(args.max_supplements)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
