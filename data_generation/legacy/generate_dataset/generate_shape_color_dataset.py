#!/usr/bin/env python3
"""Generate a resumable, image-only shape/colour dataset staging file.

The input dataset is used only to discover existing shape/colour pairs and to
reuse its irrelevant/null images.  Newly generated items are written to a
separate dataset and intentionally omit ``selected_text_priors``.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import fcntl
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
import random
import re
import resource
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageChops, ImageDraw, ImageFilter


ROOT_DIR = Path(__file__).resolve().parents[3]  # data_generation/legacy/generate_dataset -> repo root
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


COLORS = [
    "red", "orange", "yellow", "green", "blue", "cyan",
    "purple", "pink", "brown", "white", "black", "gray",
]

SHAPES = [
    "rectangle", "square", "parallelogram", "trapezoid", "diamond",
    "circle", "oval", "semicircle", "crescent",
    "triangle", "pentagon", "hexagon", "octagon",
    "star", "heart", "arrow", "cross"
]

SIMILAR_SHAPE_GROUPS = [
    # 第一组：四边形家族
    {"rectangle", "square", "parallelogram", "trapezoid", "diamond"},
    
    # 第二组：圆形与弧形家族
    {"circle", "oval", "semicircle", "crescent"},
    
    # 第三组：常见多边形（精简掉了七、九、十边形）
    {"triangle", "pentagon", "hexagon", "octagon"},
    
    # 第四组：符号与特殊形状（新增组）
    {"star", "heart", "arrow", "cross"},
]

QUESTION_TEMPLATE = (
    "What is the color of the {shape}? Choose from: " + ", ".join(COLORS) + "."
)

IMAGE_TEST_PROMPT = """Question:
{question}



Answer the question using only the question and image clue above. 

Answer as concisely as possible.
Do not provide reasoning, explanation, confidence, hedging, or any additional text.

Output exactly:

**Answer**: <your answer>

Do not include any additional text."""

QUESTION_PATTERN = re.compile(
    r"^What is the color of the (.+?)\?\s*Choose from:", re.IGNORECASE
)
CANVAS_SIZE = 1024
MAX_OBJECT_BBOX_SIDE = 250.0
EASY_TARGET_SIZE_RANGE = (180.0, 250.0)
EASY_DISTRACTOR_SIZE_RANGE = (180.0, 250.0)
HARD_EXTRA_DISTRACTOR_SIZE_RANGE = (180.0, 250.0)
EASY_MAX_ATTEMPTS = 5
HARD_MAX_ATTEMPTS = 10
RECREATE_HARD_MAX_ATTEMPTS = 20
ENTROPY_GAP_THRESHOLD = 0.25
OCCLUSION_RANGE = (0.70, 0.80)
DEEPSEEK_MODEL = "deepseek-v4-flash-aistar"
LAYOUT_SCALE_VERSION = 4
LOCAL_RENDERER_VERSION = 1
COLOR_RGB = {
    "red": (220, 45, 45), "orange": (242, 132, 30), "yellow": (240, 205, 35),
    "green": (45, 160, 75), "blue": (45, 95, 215), "cyan": (35, 185, 200),
    "purple": (130, 70, 185), "pink": (225, 100, 160), "brown": (135, 82, 45),
    "white": (250, 250, 250), "black": (25, 25, 25), "gray": (125, 125, 125),
}
ALLOWED_BACKGROUNDS = ((235, 238, 242), (242, 239, 232), (232, 240, 237))
DEFAULT_RENDER_STYLE = {
    "background_rgb": list(ALLOWED_BACKGROUNDS[0]),
    "outline_rgb": [35, 35, 35],
    "outline_width": 2,
}
DEEPSEEK_TEMPERATURES = {
    "planner": 0.2,
    "planner_test_feedback": 0.2,
    "code_generator": 0.1,
    "code_generator_syntax_repair": 0.1,
    "code_validator": 0.1,
    "recreate_failure_analyst": 0.1,
    "recreate_generator": 0.1,
}
DEEPSEEK_MAX_TOKENS = {
    "planner": 1024,
    "planner_test_feedback": 1024,
    "recreate_failure_analyst": 1536,
    "recreate_generator": 1024,
}

def _log(level: str, message: str, *, flush: bool = True) -> None:
    """Timestamped log line to stderr so stdout stays clean for dry-run JSON."""
    timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    dest = sys.stderr if level in ("WARN", "ERROR", "STOPPED") else sys.stdout
    print(f"[{timestamp}] [{level}] {message}", file=dest, flush=flush)


class GenerationStopped(RuntimeError):
    """A recoverable item failure stopped the run before a partial item write."""


class WorkerShutdown(BaseException):
    """Internal signal used to unwind a worker and clean its sandbox children."""


def _raise_worker_shutdown(signum: int, _frame: Any) -> None:
    raise WorkerShutdown(f"worker received signal {signum}")


@dataclass
class ModelRun:
    top1_answer: str | None
    ground_truth_answer: str
    answer_prob: float | None
    answer_class_probabilities: dict[str, float]
    entropy: float | None
    normalized_entropy: float | None
    parse_success: bool
    error: dict[str, str] | None
    elapsed_seconds: float


@dataclass
class CandidateResult:
    attempt: int
    seed: int
    difficulty: str = ""
    geometry_valid: bool = False
    geometry: dict[str, Any] = field(default_factory=dict)
    agent_results: dict[str, Any] = field(default_factory=dict)
    code_sha256: str | None = None
    runs: list[dict[str, Any]] = field(default_factory=list)
    calibration: dict[str, Any] | None = None
    correct_count: int = 0
    all_correct: bool = False
    entropy_gap: float | None = None
    failure_reason: str | None = None
    artifact_dir: str | None = None
    planner_retry_feedback: dict[str, Any] | None = None

    def state_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("artifact_dir", None)
        return value

    def public_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "case_seed": self.seed,
            "geometry_valid": self.geometry_valid,
            "geometry": self.geometry,
            "code_sha256": self.code_sha256,
            "runs": self.runs,
            "calibration": self.calibration,
            "correct_count": self.correct_count,
            "all_correct": self.all_correct,
            "entropy_gap": self.entropy_gap,
            "failure_reason": self.failure_reason,
            "planner_retry_feedback": self.planner_retry_feedback,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


def question_text(item: dict[str, Any]) -> str:
    question = item.get("question")
    if isinstance(question, dict):
        question = question.get("text")
    return str(question or "").strip()


def iter_dataset_items(payload: Any) -> Iterable[dict[str, Any]]:
    if not isinstance(payload, list):
        raise ValueError("Dataset root must be an array")
    for group in payload:
        if not isinstance(group, dict):
            continue
        items = group.get("items")
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    yield item


def parse_shape(question: str) -> str | None:
    match = QUESTION_PATTERN.match(question.strip())
    return normalize(match.group(1)) if match else None


def extract_existing_combinations(payload: Any) -> set[tuple[str, str]]:
    existing: set[tuple[str, str]] = set()
    for item in iter_dataset_items(payload):
        shape = parse_shape(question_text(item))
        answer = normalize(item.get("answer"))
        text_answer = normalize(item.get("text_ans"))
        if answer != text_answer:
            continue
        if shape in SHAPES and text_answer in COLORS:
            existing.add((str(shape), text_answer))
    return existing


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def derive_seed(seed: int, *parts: Any) -> int:
    material = "::".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def random_shape_order(seed: int) -> list[str]:
    rng = random.Random(seed)
    remaining = list(SHAPES)
    ordered: list[str] = []
    while remaining:
        ordered.append(remaining.pop(rng.randrange(len(remaining))))
    return ordered


def _conflict_counts(
    offsets: dict[str, int], missing_by_shape: dict[str, list[str]]
) -> list[int]:
    counts = [0] * len(COLORS)
    for shape, text_colors in missing_by_shape.items():
        offset = offsets[shape]
        for color in text_colors:
            counts[(COLORS.index(color) + offset) % len(COLORS)] += 1
    return counts


def build_conflict_maps(
    seed: int,
    missing_by_shape: dict[str, list[str]],
    max_iterations: int = 200_000,
) -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    """Build seeded cyclic derangements balanced over combinations being generated."""
    rng = random.Random(derive_seed(seed, "conflict-maps"))
    offsets = {shape: rng.randrange(1, len(COLORS)) for shape in SHAPES}

    def score(values: list[int]) -> tuple[int, float]:
        if not values:
            return (0, 0.0)
        mean = sum(values) / len(values)
        return max(values) - min(values), sum((value - mean) ** 2 for value in values)

    current_counts = _conflict_counts(offsets, missing_by_shape)
    current_score = score(current_counts)
    if sum(current_counts) >= len(COLORS):
        for _iteration in range(max_iterations):
            if current_score[0] <= 1:
                break
            shape = rng.choice(SHAPES)
            previous = offsets[shape]
            proposed = rng.randrange(1, len(COLORS))
            offsets[shape] = proposed
            proposed_counts = _conflict_counts(offsets, missing_by_shape)
            proposed_score = score(proposed_counts)
            if proposed_score < current_score or rng.random() < 0.0005:
                current_counts = proposed_counts
                current_score = proposed_score
            else:
                offsets[shape] = previous
        if current_score[0] > 1:
            raise ValueError(
                "Could not construct globally balanced conflict colors; "
                f"counts={current_counts}"
            )
    mappings = {
        shape: {
            color: COLORS[(index + offsets[shape]) % len(COLORS)]
            for index, color in enumerate(COLORS)
        }
        for shape in SHAPES
    }
    for mapping in mappings.values():
        if set(mapping.values()) != set(COLORS) or any(key == value for key, value in mapping.items()):
            raise AssertionError("Conflict map is not a derangement")
    return mappings, dict(zip(COLORS, current_counts, strict=True))


def build_manifest(payload: Any, seed: int) -> dict[str, Any]:
    existing = extract_existing_combinations(payload)
    shape_order = random_shape_order(seed)
    missing_by_shape = {
        shape: [color for color in COLORS if (shape, color) not in existing]
        for shape in SHAPES
    }
    conflict_maps, conflict_counts = build_conflict_maps(seed, missing_by_shape)
    max_id = max(
        (int(str(item.get("id"))) for item in iter_dataset_items(payload) if str(item.get("id", "")).isdigit()),
        default=0,
    )
    combinations: list[dict[str, Any]] = []
    next_id = max_id + 1
    for shape in shape_order:
        for color in COLORS:
            if (shape, color) in existing:
                continue
            entry = {
                "id": str(next_id),
                "shape": shape,
                "text_color": color,
                "conflict_color": conflict_maps[shape][color],
                "case_seeds": {
                    condition: derive_seed(seed, next_id, condition)
                    for condition in (
                        "consistent_easy", "consistent_hard",
                        "conflict_easy", "conflict_hard",
                    )
                },
            }
            combinations.append(entry)
            next_id += 1
    return {
        "seed": seed,
        "shape_order": shape_order,
        "conflict_maps": conflict_maps,
        "conflict_counts": conflict_counts,
        "existing_count": len(existing),
        "missing_count": len(SHAPES) * len(COLORS) - len(existing),
        "planned_count": len(combinations),
        "combinations": combinations,
    }


def validate_prior_pool(payload: Any) -> dict[str, int]:
    if not isinstance(payload, list):
        raise ValueError("Prior pool root must be an array")
    by_color = {entry.get("color"): entry for entry in payload if isinstance(entry, dict)}
    counts: dict[str, int] = {}
    for color in COLORS:
        entry = by_color.get(color)
        if not isinstance(entry, dict):
            raise ValueError(f"Prior pool is missing color {color!r}")
        usable = []
        for level in entry.get("prior_levels", []):
            if not isinstance(level, dict):
                continue
            for prior in level.get("priors", []):
                if (
                    isinstance(prior, dict)
                    and prior.get("accepted") is True
                    and isinstance(prior.get("text_clue"), str)
                    and prior["text_clue"].strip()
                ):
                    usable.append(prior)
        if not usable:
            raise ValueError(f"Prior pool has no accepted non-empty prior for {color!r}")
        counts[color] = len(usable)
    return counts


def find_shared_clue_paths(input_path: Path, payload: Any, output_path: Path) -> tuple[str, str]:
    irrelevant: Path | None = None
    null_image: Path | None = None
    for item in iter_dataset_items(payload):
        image_clue = item.get("image_clue")
        if not isinstance(image_clue, dict):
            continue
        raw_irr = image_clue.get("irr", image_clue.get("irrelevant"))
        if isinstance(raw_irr, dict):
            raw_irr = raw_irr.get("image")
        raw_null = image_clue.get("null")
        for raw, destination in ((raw_irr, "irr"), (raw_null, "null")):
            if not isinstance(raw, str) or not raw.strip():
                continue
            candidate = Path(raw)
            resolved = candidate.resolve() if candidate.is_absolute() else (input_path.parent / candidate).resolve()
            if not resolved.is_file():
                continue
            if destination == "irr" and irrelevant is None:
                irrelevant = resolved
            if destination == "null" and null_image is None:
                null_image = resolved
        if irrelevant is not None and null_image is not None:
            break
    if irrelevant is None or null_image is None:
        raise ValueError("Input dataset must provide at least one readable irr and null image")
    return (
        Path(os.path.relpath(irrelevant, output_path.parent.resolve())).as_posix(),
        Path(os.path.relpath(null_image, output_path.parent.resolve())).as_posix(),
    )


def similar_shapes_for(target: str) -> set[str]:
    for group in SIMILAR_SHAPE_GROUPS:
        if target in group:
            return set(group) - {target}
    return set()


def _bbox_intersection(first: Sequence[float], second: Sequence[float]) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    return max(0.0, right - left) * max(0.0, bottom - top)


def _bbox_area(bbox: Sequence[float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def _near(value: float, other: float, tolerance: float) -> bool:
    return abs(value - other) <= tolerance


def detect_layout_patterns(objects: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    centers = [tuple(map(float, obj["center"])) for obj in objects]
    count = len(centers)
    if count < 5:
        return issues
    alignment_limit = max(3, math.ceil(count * 0.30))
    for axis, name in ((0, "vertical"), (1, "horizontal")):
        for center in centers:
            aligned = sum(_near(point[axis], center[axis], 8.0) for point in centers)
            if aligned >= alignment_limit:
                issues.append(f"obvious_{name}_alignment")
                break

    sizes = [float(obj["size"]) for obj in objects]
    rotations = [float(obj["rotation"]) % 360.0 for obj in objects]
    repeat_limit = max(4, math.ceil(count * 0.50))
    if any(sum(abs(other - value) <= max(2.0, abs(value) * 0.02) for other in sizes) >= repeat_limit for value in sizes):
        issues.append("repeated_sizes")
    if any(sum(abs(other - value) <= 3.0 for other in rotations) >= repeat_limit for value in rotations):
        issues.append("repeated_rotations")

    def symmetry_ratio(transform: Any) -> float:
        matched = 0
        for point in centers:
            transformed = transform(point)
            if any(math.dist(transformed, other) <= 12.0 for other in centers):
                matched += 1
        return matched / count

    if symmetry_ratio(lambda p: (CANVAS_SIZE - p[0], p[1])) >= 0.70:
        issues.append("vertical_symmetry")
    if symmetry_ratio(lambda p: (p[0], CANVAS_SIZE - p[1])) >= 0.70:
        issues.append("horizontal_symmetry")
    if symmetry_ratio(lambda p: (CANVAS_SIZE - p[0], CANVAS_SIZE - p[1])) >= 0.70:
        issues.append("central_symmetry")

    radii = [math.dist(point, (CANVAS_SIZE / 2, CANVAS_SIZE / 2)) for point in centers]
    radius_mean = sum(radii) / len(radii)
    if radius_mean and math.sqrt(sum((r - radius_mean) ** 2 for r in radii) / len(radii)) / radius_mean <= 0.05:
        issues.append("ring_layout")

    for axis, name in ((0, "x"), (1, "y")):
        ordered = sorted(point[axis] for point in centers)
        gaps = [b - a for a, b in zip(ordered, ordered[1:]) if b - a > 8.0]
        if len(gaps) >= 4:
            mean_gap = sum(gaps) / len(gaps)
            near_equal = sum(abs(gap - mean_gap) <= max(5.0, mean_gap * 0.05) for gap in gaps)
            if near_equal >= max(4, math.ceil(len(gaps) * 0.70)):
                issues.append(f"equal_{name}_spacing")
    return sorted(set(issues))


def _object_signature(obj: dict[str, Any]) -> tuple[Any, ...]:
    return (
        obj.get("role"), obj.get("shape"), obj.get("color"),
        tuple(round(float(value), 3) for value in obj.get("center", [])),
        tuple(round(float(value), 3) for value in obj.get("bbox", [])),
        round(float(obj.get("rotation", 0)), 3), round(float(obj.get("size", 0)), 3),
    )


def validate_layout(
    layout: Any,
    target_shape: str,
    target_color: str,
    difficulty: str,
    target_mask: Image.Image | None = None,
    occluder_mask: Image.Image | None = None,
    expected_layout: dict[str, Any] | None = None,
) -> dict[str, Any]:
    issues: list[str] = []
    if not isinstance(layout, dict):
        raise ValueError("layout.json must contain an object")
    if layout.get("canvas") != [CANVAS_SIZE, CANVAS_SIZE]:
        issues.append("canvas_must_be_1024x1024")
    objects = layout.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("layout.objects must be a non-empty array")
    required = {"shape", "color", "center", "bbox", "rotation", "size", "role"}
    for index, obj in enumerate(objects):
        if not isinstance(obj, dict) or not required.issubset(obj):
            issues.append(f"object_{index}_schema")
            continue
        if obj["shape"] not in SHAPES:
            issues.append(f"object_{index}_unknown_shape")
        if obj["color"] not in COLORS:
            issues.append(f"object_{index}_unknown_color")
        if not isinstance(obj["center"], list) or len(obj["center"]) != 2:
            issues.append(f"object_{index}_center")
        if not isinstance(obj["bbox"], list) or len(obj["bbox"]) != 4:
            issues.append(f"object_{index}_bbox")
            continue
        bbox = [float(value) for value in obj["bbox"]]
        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > CANVAS_SIZE or bbox[3] > CANVAS_SIZE or _bbox_area(bbox) <= 0:
            issues.append(f"object_{index}_bbox_bounds")
        if (
            bbox[2] - bbox[0] > MAX_OBJECT_BBOX_SIDE + 1e-6
            or bbox[3] - bbox[1] > MAX_OBJECT_BBOX_SIDE + 1e-6
        ):
            issues.append(f"object_{index}_bbox_too_large")
    valid_objects = [obj for obj in objects if isinstance(obj, dict) and required.issubset(obj)]
    targets = [obj for obj in valid_objects if obj["role"] == "target"]
    if len(targets) != 1:
        issues.append("target_must_appear_once")
    elif targets[0]["shape"] != target_shape or targets[0]["color"] != target_color:
        issues.append("target_shape_or_color_mismatch")
    if sum(obj["shape"] == target_shape for obj in valid_objects) != 1:
        issues.append("target_shape_must_be_unique")
    if any(obj["color"] == target_color and obj.get("role") != "target" for obj in valid_objects):
        issues.append("target_color_used_by_non_target")
    forbidden = similar_shapes_for(target_shape)
    if any(obj["shape"] in forbidden and obj.get("role") != "occluder" for obj in valid_objects):
        issues.append("similar_shape_present")
    distractors = [obj for obj in valid_objects if obj.get("role") != "target"]
    distinct_distractors = len({obj["shape"] for obj in distractors})
    minimum = 7 if difficulty == "easy" else 10
    if distinct_distractors < minimum:
        issues.append(f"insufficient_distinct_distractors:{distinct_distractors}<{minimum}")

    for first_index, first in enumerate(valid_objects):
        for second in valid_objects[first_index + 1:]:
            overlap = _bbox_intersection(first["bbox"], second["bbox"])
            if overlap <= 0:
                continue
            roles = {first.get("role"), second.get("role")}
            if difficulty == "easy":
                issues.append("easy_bbox_overlap")
            elif roles == {"target", "occluder"}:
                continue
            else:
                fraction = overlap / max(1.0, min(_bbox_area(first["bbox"]), _bbox_area(second["bbox"])))
                if fraction > 0.15:
                    issues.append("excessive_non_target_overlap")
    if targets:
        target_bbox = [float(value) for value in targets[0]["bbox"]]
        if min(target_bbox[0], target_bbox[1], CANVAS_SIZE - target_bbox[2], CANVAS_SIZE - target_bbox[3]) < 32:
            issues.append("target_safety_margin")
    issues.extend(detect_layout_patterns(valid_objects))

    occlusion_ratio: float | None = None
    if target_mask is not None and occluder_mask is not None:
        if target_mask.size != (CANVAS_SIZE, CANVAS_SIZE) or occluder_mask.size != target_mask.size:
            issues.append("mask_size")
        else:
            for mask_name, mask in (("target", target_mask), ("occluder", occluder_mask)):
                mask_bbox = mask.getbbox()
                if mask_bbox is not None and (
                    mask_bbox[2] - mask_bbox[0] > MAX_OBJECT_BBOX_SIDE
                    or mask_bbox[3] - mask_bbox[1] > MAX_OBJECT_BBOX_SIDE
                ):
                    issues.append(f"{mask_name}_mask_too_large")
            target_binary = target_mask.convert("1")
            occluder_binary = occluder_mask.convert("1")
            target_area = target_binary.histogram()[255]
            intersection = ImageChops.logical_and(target_binary, occluder_binary).histogram()[255]
            occlusion_ratio = 0.0 if target_area == 0 else intersection / target_area
            if target_area == 0:
                issues.append("empty_target_mask")
            elif difficulty == "easy" and occlusion_ratio != 0.0:
                issues.append("easy_target_occluded")
            elif difficulty == "hard" and not (OCCLUSION_RANGE[0] <= occlusion_ratio <= OCCLUSION_RANGE[1]):
                issues.append("hard_occlusion_out_of_range")

    if expected_layout is not None:
        expected = expected_layout.get("objects", [])
        if [_object_signature(obj) for obj in valid_objects] != [_object_signature(obj) for obj in expected]:
            issues.append("layout_does_not_match_local_rng_plan")
        for key in ("case_seed", "branch", "difficulty", "target_shape", "target_color"):
            if layout.get(key) != expected_layout.get(key):
                issues.append(f"layout_{key}_mismatch")
    result = {
        "valid": not issues,
        "issues": sorted(set(issues)),
        "distinct_distractor_shapes": distinct_distractors,
        "occlusion_ratio": occlusion_ratio,
        "object_count": len(valid_objects),
    }
    return result


def _sample_bbox(rng: random.Random, size: float) -> tuple[list[float], list[float]]:
    aspect = rng.uniform(0.72, 1.32)
    # ``size`` is the actual maximum bbox side, so neither dimension can
    # silently exceed the configured 150 px limit through the aspect ratio.
    if aspect >= 1.0:
        width, height = size, size / aspect
    else:
        width, height = size * aspect, size
    margin = 45.0
    center = [rng.uniform(margin + width / 2, CANVAS_SIZE - margin - width / 2),
              rng.uniform(margin + height / 2, CANVAS_SIZE - margin - height / 2)]
    bbox = [center[0] - width / 2, center[1] - height / 2,
            center[0] + width / 2, center[1] + height / 2]
    return center, bbox


def _place_object(
    rng: random.Random,
    objects: list[dict[str, Any]],
    shape: str,
    color: str,
    role: str,
    size_range: tuple[float, float],
) -> dict[str, Any]:
    for _ in range(1000):
        size = rng.uniform(*size_range)
        center, bbox = _sample_bbox(rng, size)
        if all(_bbox_intersection(bbox, other["bbox"]) == 0 for other in objects):
            return {
                "shape": shape, "color": color, "center": [round(v, 3) for v in center],
                "bbox": [round(v, 3) for v in bbox], "rotation": round(rng.uniform(3, 357), 3),
                "size": round(size, 3), "role": role,
            }
    raise ValueError("Local RNG could not create a non-overlapping random layout")


def _preferred_occluder_shape(target_shape: str) -> str:
    # Occluders are not distractors. Use a trusted bar shape that differs from
    # the target itself so exact mask coverage can be solved deterministically.
    return "square" if target_shape == "rectangle" else "rectangle"


def build_easy_layout(
    seed: int, branch: str, target_shape: str, target_color: str
) -> dict[str, Any]:
    rng = random.Random(seed)
    allowed = [shape for shape in SHAPES if shape != target_shape and shape not in similar_shapes_for(target_shape)]
    rng.shuffle(allowed)
    # Reserve one predictable, dissimilar shape for the hard occluder so the
    # trusted renderer can solve its mask coverage deterministically.
    reserved_occluder = _preferred_occluder_shape(target_shape)
    distractor_shapes = [shape for shape in allowed if shape != reserved_occluder][:10]
    available_colors = [color for color in COLORS if color != target_color]
    objects: list[dict[str, Any]] | None = None
    for restart in range(50):
        layout_rng = random.Random(derive_seed(seed, "joint-easy-layout", restart))
        proposed: list[dict[str, Any]] = []
        try:
            target = _place_object(
                layout_rng, proposed, target_shape, target_color, "target", EASY_TARGET_SIZE_RANGE
            )
            proposed.append(target)
            for shape in distractor_shapes:
                proposed.append(_place_object(
                    layout_rng,
                    proposed,
                    shape,
                    layout_rng.choice(available_colors),
                    "distractor",
                    EASY_DISTRACTOR_SIZE_RANGE,
                ))
        except ValueError:
            continue
        if not detect_layout_patterns(proposed):
            objects = proposed
            break
    if objects is None:
        raise ValueError("Local RNG could not jointly pack a non-grid easy layout")
    target = objects[0]
    layout = {
        "canvas": [CANVAS_SIZE, CANVAS_SIZE], "branch": branch, "difficulty": "easy",
        "target_shape": target_shape, "target_color": target_color, "case_seed": seed,
        "target_geometry": {
            key: target[key] for key in ("center", "bbox", "rotation", "size")
        },
        "occluder_geometry": [],
        "mask_method": "binary target silhouette and binary occluder union",
        "expected_occlusion_ratio": 0.0, "objects": objects,
    }
    return layout


def build_hard_layout(seed: int, easy_layout: dict[str, Any]) -> dict[str, Any]:
    rng = random.Random(seed)
    objects = [dict(obj) for obj in easy_layout["objects"]]
    target = next(obj for obj in objects if obj["role"] == "target")
    target_shape = str(easy_layout["target_shape"])
    target_color = str(easy_layout["target_color"])
    existing_shapes = {obj["shape"] for obj in objects}
    occluder_shape = _preferred_occluder_shape(target_shape)
    if occluder_shape in existing_shapes or occluder_shape == target_shape:
        raise ValueError("Reserved hard occluder is not available")
    allowed = [
        shape for shape in SHAPES
        if shape != target_shape and shape not in similar_shapes_for(target_shape)
        and shape not in existing_shapes and shape != occluder_shape
    ]
    rng.shuffle(allowed)
    colors = [color for color in COLORS if color != target_color]
    while len({obj["shape"] for obj in objects if obj["role"] != "target"}) < 10:
        if not allowed:
            raise ValueError("Not enough dissimilar distractor shapes for hard layout")
        shape = allowed.pop()
        objects.append(_place_object(
            rng, objects, shape, rng.choice(colors), "distractor", HARD_EXTRA_DISTRACTOR_SIZE_RANGE
        ))
    occluder = _solve_occluder_object(
        target,
        occluder_shape,
        rng.choice(colors),
        seed,
    )
    objects.append(occluder)
    return {
        "canvas": [CANVAS_SIZE, CANVAS_SIZE], "branch": easy_layout["branch"],
        "difficulty": "hard", "target_shape": target_shape, "target_color": target_color,
        "case_seed": seed, "base_easy_seed": easy_layout["case_seed"],
        "target_geometry": {
            key: target[key] for key in ("center", "bbox", "rotation", "size")
        },
        "occluder_geometry": [
            {key: occluder[key] for key in ("shape", "center", "bbox", "rotation", "size")}
        ],
        "mask_method": "binary target silhouette intersected with binary occluder union",
        "expected_occlusion_ratio": 0.75, "objects": objects,
    }


def build_recreated_hard_layout(seed: int, easy_layout: dict[str, Any]) -> dict[str, Any]:
    """Create a materially new hard arrangement while preserving the easy target."""
    target = dict(next(obj for obj in easy_layout["objects"] if obj["role"] == "target"))
    source_distractors = [
        dict(obj) for obj in easy_layout["objects"] if obj.get("role") == "distractor"
    ]
    target_shape = str(easy_layout["target_shape"])
    target_color = str(easy_layout["target_color"])
    occluder_shape = _preferred_occluder_shape(target_shape)
    source_shapes = {obj["shape"] for obj in source_distractors} | {target_shape}
    if occluder_shape in source_shapes or occluder_shape == target_shape:
        raise ValueError("Reserved recreate hard occluder is not available")
    colors = [color for color in COLORS if color != target_color]
    objects: list[dict[str, Any]] | None = None
    occluder: dict[str, Any] | None = None
    for restart in range(50):
        restart_seed = derive_seed(seed, "recreate-hard-layout", restart)
        rng = random.Random(restart_seed)
        proposed = [dict(target)]
        shuffled = list(source_distractors)
        rng.shuffle(shuffled)
        try:
            for source in shuffled:
                proposed.append(_place_object(
                    rng,
                    proposed,
                    str(source["shape"]),
                    str(source["color"]),
                    "distractor",
                    HARD_EXTRA_DISTRACTOR_SIZE_RANGE,
                ))
        except ValueError:
            continue
        current_occluder = _solve_occluder_object(
            target,
            occluder_shape,
            rng.choice(colors),
            restart_seed,
            seeded_variant=True,
        )
        proposed.append(current_occluder)
        if not detect_layout_patterns(proposed):
            objects = proposed
            occluder = current_occluder
            break
    if objects is None or occluder is None:
        raise ValueError("Local RNG could not repack a valid recreated hard layout")
    return {
        "canvas": [CANVAS_SIZE, CANVAS_SIZE],
        "branch": "conflict",
        "difficulty": "hard",
        "target_shape": target_shape,
        "target_color": target_color,
        "case_seed": seed,
        "base_easy_seed": easy_layout.get("case_seed"),
        "target_geometry": {
            key: target[key] for key in ("center", "bbox", "rotation", "size")
        },
        "occluder_geometry": [
            {key: occluder[key] for key in ("shape", "center", "bbox", "rotation", "size")}
        ],
        "mask_method": "binary target silhouette intersected with binary occluder union",
        "expected_occlusion_ratio": occluder["expected_mask_occlusion_ratio"],
        "objects": objects,
    }


def sanitize_render_style(value: Any) -> dict[str, Any]:
    """Reduce DeepSeek style JSON to a tiny, safe, deterministic schema."""
    raw = value if isinstance(value, dict) else {}
    background = raw.get("background_rgb")
    background_tuple = tuple(background) if isinstance(background, list) and len(background) == 3 else None
    if background_tuple not in ALLOWED_BACKGROUNDS:
        background_tuple = ALLOWED_BACKGROUNDS[0]
    outline = raw.get("outline_rgb")
    if not (
        isinstance(outline, list) and len(outline) == 3
        and all(isinstance(channel, int) and 0 <= channel <= 80 for channel in outline)
    ):
        outline = list(DEFAULT_RENDER_STYLE["outline_rgb"])
    width = raw.get("outline_width")
    if not isinstance(width, int) or not 1 <= width <= 3:
        width = int(DEFAULT_RENDER_STYLE["outline_width"])
    return {
        "background_rgb": list(background_tuple),
        "outline_rgb": list(outline),
        "outline_width": width,
    }


def _regular_polygon(count: int, radius: float = 0.92, offset: float = -math.pi / 2) -> list[tuple[float, float]]:
    return [
        (math.cos(offset + 2 * math.pi * index / count) * radius,
         math.sin(offset + 2 * math.pi * index / count) * radius)
        for index in range(count)
    ]


def _normalized_shape_mask(shape: str, side: int = 256) -> Image.Image:
    """Draw one canonical shape in a local square without external code."""
    mask = Image.new("L", (side, side), 0)
    draw = ImageDraw.Draw(mask)
    center = side / 2
    radius = side * 0.44

    def points(values: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
        return [(center + x * radius, center + y * radius) for x, y in values]

    bounds = (center - radius, center - radius, center + radius, center + radius)
    if shape in {"rectangle", "square"}:
        draw.rectangle(bounds, fill=255)
    elif shape == "parallelogram":
        draw.polygon(points([(-0.70, -0.88), (1.0, -0.88), (0.70, 0.88), (-1.0, 0.88)]), fill=255)
    elif shape == "trapezoid":
        draw.polygon(points([(-0.65, -0.88), (0.65, -0.88), (1.0, 0.88), (-1.0, 0.88)]), fill=255)
    elif shape == "diamond":
        draw.polygon(points([(0, -1), (1, 0), (0, 1), (-1, 0)]), fill=255)
    elif shape in {"circle", "oval"}:
        if shape == "circle":
            draw.ellipse(bounds, fill=255)
        else:
            draw.ellipse((center - radius, center - radius * 0.68, center + radius, center + radius * 0.68), fill=255)
    elif shape == "semicircle":
        draw.pieslice(bounds, start=0, end=180, fill=255)
    elif shape == "crescent":
        draw.ellipse(bounds, fill=255)
        draw.ellipse(
            (center - radius * 0.35, center - radius * 0.92,
             center + radius * 0.95, center + radius * 0.92),
            fill=0,
        )
    elif shape in {"triangle", "pentagon", "hexagon", "octagon"}:
        sides = {"triangle": 3, "pentagon": 5, "hexagon": 6, "octagon": 8}[shape]
        draw.polygon(points(_regular_polygon(sides)), fill=255)
    elif shape == "star":
        star: list[tuple[float, float]] = []
        for index in range(10):
            current_radius = 1.0 if index % 2 == 0 else 0.42
            angle = -math.pi / 2 + index * math.pi / 5
            star.append((math.cos(angle) * current_radius, math.sin(angle) * current_radius))
        draw.polygon(points(star), fill=255)
    elif shape == "heart":
        heart: list[tuple[float, float]] = []
        for index in range(101):
            angle = 2 * math.pi * index / 100
            x = 16 * math.sin(angle) ** 3 / 17
            y = -(13 * math.cos(angle) - 5 * math.cos(2 * angle)
                  - 2 * math.cos(3 * angle) - math.cos(4 * angle)) / 17
            heart.append((x, y))
        draw.polygon(points(heart), fill=255)
    elif shape == "arrow":
        draw.polygon(points([
            (-1.0, -0.28), (0.25, -0.28), (0.25, -0.62),
            (1.0, 0), (0.25, 0.62), (0.25, 0.28), (-1.0, 0.28),
        ]), fill=255)
    elif shape == "cross":
        draw.polygon(points([
            (-0.30, -1.0), (0.30, -1.0), (0.30, -0.30),
            (1.0, -0.30), (1.0, 0.30), (0.30, 0.30),
            (0.30, 1.0), (-0.30, 1.0), (-0.30, 0.30),
            (-1.0, 0.30), (-1.0, -0.30), (-0.30, -0.30),
        ]), fill=255)
    else:
        raise ValueError(f"No trusted renderer for shape {shape!r}")
    return mask


def _object_mask(obj: dict[str, Any]) -> Image.Image:
    left, top, right, bottom = map(float, obj["bbox"])
    width = max(1, min(round(right - left), int(MAX_OBJECT_BBOX_SIDE)))
    height = max(1, min(round(bottom - top), int(MAX_OBJECT_BBOX_SIDE)))
    role = str(obj.get("role"))
    shape = str(obj["shape"])
    if role == "occluder" and shape in {"rectangle", "square"}:
        local = Image.new("L", (width, height), 0)
        local_draw = ImageDraw.Draw(local)
        local_draw.rectangle((0, 0, width - 1, height - 1), fill=255)
    else:
        base = _normalized_shape_mask(shape)
        rotated = base.rotate(float(obj.get("rotation", 0.0)), resample=Image.Resampling.BICUBIC, expand=True)
        content = rotated.getbbox()
        if content is None:
            raise ValueError(f"Trusted renderer produced an empty {shape} mask")
        local = rotated.crop(content)
        local.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
    x = round((left + right - local.width) / 2)
    y = round((top + bottom - local.height) / 2)
    canvas.paste(local, (x, y))
    return canvas


def _mask_overlap_ratio(target_mask: Image.Image, occluder_mask: Image.Image) -> float:
    bounds = target_mask.getbbox()
    if bounds is None:
        return 0.0
    target_binary = target_mask.crop(bounds).convert("1")
    occluder_binary = occluder_mask.crop(bounds).convert("1")
    target_area = target_binary.histogram()[255]
    if target_area == 0:
        return 0.0
    intersection = ImageChops.logical_and(target_binary, occluder_binary).histogram()[255]
    return intersection / target_area


def _occluder_candidate(
    shape: str,
    color: str,
    bbox: Sequence[float],
    rotation: float = 0.0,
) -> dict[str, Any]:
    left, top, right, bottom = map(float, bbox)
    return {
        "shape": shape,
        "color": color,
        "center": [round((left + right) / 2, 3), round((top + bottom) / 2, 3)],
        "bbox": [round(left, 3), round(top, 3), round(right, 3), round(bottom, 3)],
        "rotation": round(rotation, 3),
        "size": round(max(right - left, bottom - top), 3),
        "role": "occluder",
    }


def _solve_occluder_object(
    target: dict[str, Any],
    shape: str,
    color: str,
    seed: int,
    seeded_variant: bool = False,
) -> dict[str, Any]:
    """Find trusted geometry whose rendered mask covers 70%-80% of the target."""
    target_mask = _object_mask(target)
    bounds = target_mask.getbbox()
    if bounds is None:
        raise ValueError("Cannot solve occlusion for an empty target mask")
    left, top, right, bottom = bounds
    binary = target_mask.crop(bounds).convert("1")
    width, height = binary.size
    pixels = list(binary.getdata())
    target_area = sum(1 for value in pixels if value)
    candidates: list[tuple[float, tuple[int, int, int, int], str]] = []

    column_counts = [sum(1 for y in range(height) if pixels[y * width + x]) for x in range(width)]
    row_counts = [sum(1 for x in range(width) if pixels[y * width + x]) for y in range(height)]
    for counts, orientation in ((column_counts, "vertical"), (row_counts, "horizontal")):
        cumulative = 0
        for index, count in enumerate(counts, start=1):
            cumulative += count
            ratio = cumulative / target_area
            if orientation == "vertical":
                boxes = ((left, top, left + index, bottom), (right - index, top, right, bottom))
            else:
                boxes = ((left, top, right, top + index), (left, bottom - index, right, bottom))
            for bbox in boxes:
                candidates.append((ratio, bbox, orientation))

    legal = [value for value in candidates if OCCLUSION_RANGE[0] <= value[0] <= OCCLUSION_RANGE[1]]
    if not legal:
        closest = min(candidates, key=lambda value: abs(value[0] - 0.75), default=None)
        raise ValueError(f"Could not solve legal occluder geometry; closest={closest[0] if closest else None}")
    def materialize(value: tuple[float, tuple[int, int, int, int], str]) -> dict[str, Any] | None:
        _estimated, raw_bbox, orientation = value
        bbox = list(map(float, raw_bbox))
        if orientation == "vertical" and bbox[3] - bbox[1] < EASY_TARGET_SIZE_RANGE[0]:
            midpoint = (bbox[1] + bbox[3]) / 2
            bbox[1] = max(0.0, midpoint - EASY_TARGET_SIZE_RANGE[0] / 2)
            bbox[3] = bbox[1] + EASY_TARGET_SIZE_RANGE[0]
            if bbox[3] > CANVAS_SIZE:
                bbox[3] = float(CANVAS_SIZE)
                bbox[1] = bbox[3] - EASY_TARGET_SIZE_RANGE[0]
        elif orientation == "horizontal" and bbox[2] - bbox[0] < EASY_TARGET_SIZE_RANGE[0]:
            midpoint = (bbox[0] + bbox[2]) / 2
            bbox[0] = max(0.0, midpoint - EASY_TARGET_SIZE_RANGE[0] / 2)
            bbox[2] = bbox[0] + EASY_TARGET_SIZE_RANGE[0]
            if bbox[2] > CANVAS_SIZE:
                bbox[2] = float(CANVAS_SIZE)
                bbox[0] = bbox[2] - EASY_TARGET_SIZE_RANGE[0]
        selected = _occluder_candidate(shape, color, bbox)
        measured = _mask_overlap_ratio(target_mask, _object_mask(selected))
        if not (OCCLUSION_RANGE[0] <= measured <= OCCLUSION_RANGE[1]):
            return None
        selected["expected_mask_occlusion_ratio"] = measured
        return selected

    if seeded_variant:
        ordered = list(legal)
        random.Random(seed).shuffle(ordered)
        ordered.sort(key=lambda value: abs(value[0] - 0.75))
        verified = []
        for value in ordered:
            selected = materialize(value)
            if selected is not None:
                verified.append(selected)
                if len(verified) >= 24:
                    break
        if not verified:
            raise ValueError("Could not verify any seeded legal occluder geometry")
        near_target = sorted(
            verified,
            key=lambda value: abs(float(value["expected_mask_occlusion_ratio"]) - 0.75),
        )[:min(24, len(verified))]
        return random.Random(derive_seed(seed, "verified-occluder")).choice(near_target)

    selected = materialize(min(legal, key=lambda value: abs(value[0] - 0.75)))
    if selected is None:
        raise ValueError("Solved occluder failed verification")
    return selected


def render_scene_locally(
    layout: dict[str, Any],
    work_dir: Path,
    render_style: dict[str, Any] | None = None,
) -> Path:
    """Render a validated layout using only trusted repository code."""
    style = sanitize_render_style(render_style)
    output = work_dir / "rendered"
    output.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), tuple(style["background_rgb"]))
    target_mask = Image.new("L", image.size, 0)
    occluder_mask = Image.new("L", image.size, 0)
    outline_width = int(style["outline_width"])

    ordered = [obj for obj in layout["objects"] if obj.get("role") != "occluder"] + [
        obj for obj in layout["objects"] if obj.get("role") == "occluder"
    ]
    for obj in ordered:
        mask = _object_mask(obj)
        role = str(obj.get("role"))
        if role == "target":
            target_mask = ImageChops.lighter(target_mask, mask)
        elif role == "occluder":
            occluder_mask = ImageChops.lighter(occluder_mask, mask)

        mask_bounds = mask.getbbox()
        if mask_bounds is None:
            raise ValueError(f"Trusted renderer produced empty object mask: {obj['shape']}")
        mask_crop = mask.crop(mask_bounds)
        padded = Image.new(
            "L",
            (mask_crop.width + outline_width * 2, mask_crop.height + outline_width * 2),
            0,
        )
        padded.paste(mask_crop, (outline_width, outline_width))
        expanded_crop = padded.filter(ImageFilter.MaxFilter(outline_width * 2 + 1))
        expanded = Image.new("L", image.size, 0)
        expanded.paste(expanded_crop, (mask_bounds[0] - outline_width, mask_bounds[1] - outline_width))
        left, top, right, bottom = [round(float(value)) for value in obj["bbox"]]
        bbox_clip = Image.new("L", image.size, 0)
        ImageDraw.Draw(bbox_clip).rectangle((left, top, right - 1, bottom - 1), fill=255)
        expanded = ImageChops.multiply(expanded, bbox_clip)
        image.paste(tuple(style["outline_rgb"]), mask=expanded)
        image.paste(COLOR_RGB[str(obj["color"])], mask=mask)

    image.save(output / "image.png", format="PNG")
    target_mask.save(output / "target_mask.png", format="PNG")
    occluder_mask.save(output / "occluder_mask.png", format="PNG")
    atomic_write_json(output / "layout.json", layout)
    atomic_write_json(output / "agent_layout.json", layout)
    return output


ALLOWED_IMPORTS = {
    "math": None,
    "random": None,
    "PIL": {"Image", "ImageDraw", "ImageChops"},
}
FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
    "globals", "locals", "vars", "getattr", "setattr", "delattr", "help",
    "os", "sys", "subprocess", "socket", "pathlib", "shutil", "requests",
}
FORBIDDEN_ATTRIBUTES = {
    "text", "multiline_text", "textbbox", "textlength", "truetype", "load_default",
    "open", "save", "write", "write_text", "write_bytes", "read_text", "read_bytes",
    "system", "popen", "spawn", "fork", "connect", "request", "urlopen",
}


def syntax_diagnostics(code: str) -> dict[str, Any]:
    """Return stable local parse/compile diagnostics without changing code."""
    try:
        tree = ast.parse(code, filename="generated_scene.py", mode="exec")
        compile(tree, "generated_scene.py", "exec")
        return {"valid": True, "error": None}
    except (SyntaxError, ValueError, TypeError) as exc:
        return {
            "valid": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "line": getattr(exc, "lineno", None),
                "offset": getattr(exc, "offset", None),
                "end_line": getattr(exc, "end_lineno", None),
                "end_offset": getattr(exc, "end_offset", None),
                "source_line": (getattr(exc, "text", None) or "").rstrip("\n"),
            },
        }


def minimal_patch_summary(before: str, after: str) -> dict[str, Any]:
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    changed = sum(
        max(old_end - old_start, new_end - new_start)
        for tag, old_start, old_end, new_start, new_end
        in SequenceMatcher(a=before_lines, b=after_lines, autojunk=False).get_opcodes()
        if tag != "equal"
    )
    allowed = max(12, math.ceil(max(1, len(before_lines)) * 0.15))
    return {
        "before_lines": len(before_lines),
        "after_lines": len(after_lines),
        "changed_line_positions": changed,
        "allowed_changed_line_positions": allowed,
        "minimal": changed <= allowed,
    }


def validate_generated_code(code: str) -> ast.AST:
    if not isinstance(code, str) or not code.strip():
        raise ValueError("Generated code is empty")
    tree = ast.parse(code, filename="generated_scene.py", mode="exec")
    function_names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    if "render_scene" not in function_names:
        raise ValueError("Generated code must define render_scene()")
    for top_level in tree.body:
        if not isinstance(top_level, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.Assign, ast.AnnAssign)):
            raise ValueError(f"Unsafe top-level statement: {type(top_level).__name__}")
        if isinstance(top_level, ast.FunctionDef):
            if top_level.decorator_list or top_level.args.defaults or top_level.args.kw_defaults:
                raise ValueError("Generated functions may not use decorators or default expressions")
        if isinstance(top_level, (ast.Assign, ast.AnnAssign)):
            value = top_level.value
            try:
                ast.literal_eval(value)
            except Exception as exc:
                raise ValueError("Top-level assignments must contain literal constants only") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in {"math", "random"}:
                    raise ValueError(f"Unsafe import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            allowed = ALLOWED_IMPORTS.get(node.module or "")
            if allowed is None or any(alias.name not in allowed for alias in node.names):
                raise ValueError(f"Unsafe import from {node.module}")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ValueError(f"Unsafe name: {node.id}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in FORBIDDEN_ATTRIBUTES:
                raise ValueError(f"Unsafe attribute: {node.attr}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_NAMES:
                raise ValueError(f"Unsafe call: {node.func.id}")
        elif isinstance(node, (ast.Global, ast.Nonlocal, ast.ClassDef, ast.Lambda, ast.AsyncFunctionDef, ast.Await)):
            raise ValueError(f"Unsafe syntax: {type(node).__name__}")
    compile(tree, "generated_scene.py", "exec")
    return tree


def ensure_syntax_with_minimal_repairs(
    agents: Any,
    plan: dict[str, Any],
    code: str,
    max_repairs: int = 3,
) -> tuple[str, list[dict[str, Any]]]:
    """Ask the Code Generator to patch only syntax errors, never rewrite locally."""
    history: list[dict[str, Any]] = []
    current = code.strip()
    if current.startswith("```"):
        current = re.sub(r"^```(?:python)?\s*", "", current, flags=re.IGNORECASE)
        current = re.sub(r"\s*```$", "", current)
    for repair_index in range(max_repairs + 1):
        diagnostics = syntax_diagnostics(current)
        history.append({"repair_index": repair_index, "diagnostics": diagnostics})
        if diagnostics["valid"]:
            return current, history
        if repair_index == max_repairs:
            break
        response = agents.repair_syntax(plan, current, diagnostics["error"])
        patched = response.get("patched_code")
        if not isinstance(patched, str) or not patched.strip():
            raise ValueError("Syntax repair returned no patched_code")
        history[-1]["repair_response"] = response
        summary = minimal_patch_summary(current, patched.strip())
        history[-1]["minimal_patch"] = summary
        if not summary["minimal"]:
            history[-1]["patch_rejected"] = "syntax repair changed too many line positions"
            continue
        current = patched.strip()
    raise SyntaxError(f"Generated code still has syntax errors after {max_repairs} minimal repairs: {history[-1]['diagnostics']['error']}")


RUNNER_SOURCE = r'''import importlib.util
import json
import sys
from pathlib import Path
from PIL import Image

source = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
expected_layout_path = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else None
spec = importlib.util.spec_from_file_location("generated_scene", source)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load generated module")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
result = module.render_scene()
if not isinstance(result, dict):
    raise TypeError("render_scene must return a dict")
image = result.get("image")
target = result.get("target_mask")
occluder = result.get("occluder_mask")
layout = result.get("layout")
if not isinstance(image, Image.Image) or not isinstance(target, Image.Image) or not isinstance(occluder, Image.Image):
    raise TypeError("render_scene returned invalid Pillow objects")
if not isinstance(layout, dict):
    raise TypeError("render_scene returned invalid layout")
output.mkdir(parents=True, exist_ok=True)
image.save(output / "image.png", format="PNG")
target.save(output / "target_mask.png", format="PNG")
occluder.save(output / "occluder_mask.png", format="PNG")
(output / "layout.json").write_text(json.dumps(layout, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
if expected_layout_path is not None:
    (output / "agent_layout.json").write_text(json.dumps(layout, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    expected_layout = json.loads(expected_layout_path.read_text(encoding="utf-8"))
    (output / "layout.json").write_text(json.dumps(expected_layout, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
'''


def _child_limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (128 * 1024 * 1024, 128 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _terminate_subprocess_group(process: subprocess.Popen[Any], grace_seconds: float = 2.0) -> None:
    """Stop a sandbox process and every descendant in its dedicated session."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass


def execute_generated_code(
    code: str,
    work_dir: Path,
    timeout: float = 30.0,
    expected_layout: dict[str, Any] | None = None,
) -> Path:
    validate_generated_code(code)
    source = work_dir / "generated_scene.py"
    runner = work_dir / "trusted_runner.py"
    output = work_dir / "rendered"
    source.write_text(code, encoding="utf-8")
    runner.write_text(RUNNER_SOURCE, encoding="utf-8")
    command = [sys.executable, "-I", str(runner), str(source), str(output)]
    if expected_layout is not None:
        expected_path = work_dir / "expected_layout.json"
        expected_path.write_text(json.dumps(expected_layout, ensure_ascii=False), encoding="utf-8")
        command.append(str(expected_path))
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": "",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    process = subprocess.Popen(
        command,
        cwd=work_dir,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=_child_limits,
        start_new_session=True,
    )
    try:
        _stdout, stderr = process.communicate(timeout=timeout)
    except BaseException:
        _terminate_subprocess_group(process)
        raise
    if process.returncode != 0:
        cleaned_stderr = stderr[-2000:].replace("api_key", "[redacted]")
        raise RuntimeError(f"Generated scene subprocess failed ({process.returncode}): {cleaned_stderr}")
    required = ["image.png", "target_mask.png", "occluder_mask.png", "layout.json"]
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"Generated scene omitted files: {missing}")
    return output


def validate_rendered_scene(
    output: Path,
    expected_layout: dict[str, Any],
) -> dict[str, Any]:
    image_path = output / "image.png"
    target_path = output / "target_mask.png"
    occluder_path = output / "occluder_mask.png"
    layout = load_json(output / "layout.json")
    agent_layout_path = output / "agent_layout.json"
    agent_layout = load_json(agent_layout_path) if agent_layout_path.is_file() else layout
    with Image.open(image_path) as image:
        image.load()
        if image.size != (CANVAS_SIZE, CANVAS_SIZE) or image.mode != "RGB" or image.format != "PNG":
            raise ValueError(f"Invalid image properties: size={image.size}, mode={image.mode}, format={image.format}")
    with Image.open(target_path) as target_image, Image.open(occluder_path) as occluder_image:
        target_image.load()
        occluder_image.load()
        if target_image.mode != "L" or occluder_image.mode != "L":
            raise ValueError("Masks must use mode L")
        result = validate_layout(
            layout,
            str(expected_layout["target_shape"]),
            str(expected_layout["target_color"]),
            str(expected_layout["difficulty"]),
            target_image,
            occluder_image,
            expected_layout=expected_layout,
        )
    if not result["valid"]:
        raise ValueError("Local geometry validation failed: " + ", ".join(result["issues"]))
    semantic_issue_prefixes = (
        "object_", "target_", "similar_shape", "insufficient_distinct", "canvas_",
    )
    agent_check = validate_layout(
        agent_layout,
        str(expected_layout["target_shape"]),
        str(expected_layout["target_color"]),
        str(expected_layout["difficulty"]),
    )
    semantic_issues = [
        issue for issue in agent_check["issues"]
        if issue.startswith(semantic_issue_prefixes)
    ]
    if semantic_issues:
        raise ValueError("Agent layout semantic validation failed: " + ", ".join(semantic_issues))
    result["agent_layout_matches_local_plan"] = (
        [_object_signature(obj) for obj in agent_layout.get("objects", [])]
        == [_object_signature(obj) for obj in expected_layout.get("objects", [])]
    )
    result["agent_layout_nonsemantic_issues"] = [
        issue for issue in agent_check["issues"] if issue not in semantic_issues
    ]
    return result


def parse_json_response(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("DeepSeek response JSON must be an object")
    return parsed


def _redact(message: str, secret: str = "") -> str:
    cleaned = str(message)
    if secret:
        cleaned = cleaned.replace(secret, "[redacted]")
    cleaned = re.sub(r"(?i)(authorization|api[_-]?key)\s*[:=]\s*\S+", r"\1=[redacted]", cleaned)
    return cleaned


class DeepSeekAgents:
    def __init__(self, config_path: Path):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("DeepSeek generation requires the openai package") from exc
        config = load_json(config_path)
        self.api_key = str(config.get("api_key", ""))
        base_url = str(config.get("base_url", ""))
        if not self.api_key or not base_url:
            raise RuntimeError("api_config.json must contain api_key and base_url")
        self.client = OpenAI(api_key=self.api_key, base_url=base_url, timeout=150.0, max_retries=0)

    def call(
        self,
        prompt: str,
        role: str,
        retries: int = 3,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        request_temperature = (
            DEEPSEEK_TEMPERATURES.get(role, 0.2)
            if temperature is None else float(temperature)
        )
        for attempt in range(1, retries + 1):
            started = time.perf_counter()
            try:
                _log("INFO", f"  DeepSeek {role} attempt {attempt}/{retries}")
                response = self.client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=request_temperature,
                    max_tokens=DEEPSEEK_MAX_TOKENS.get(role, 8192),
                )
                content = response.choices[0].message.content or ""
                parsed = parse_json_response(content)
                _log("INFO", f"  DeepSeek {role} attempt {attempt}/{retries} ok ({time.perf_counter() - started:.1f}s)")
                return parsed
            except Exception as exc:
                last_error = exc
                _log("WARN", f"  DeepSeek {role} attempt {attempt}/{retries} failed: {_redact(str(exc), self.api_key)}")
                if attempt < retries:
                    time.sleep(min(2 ** (attempt - 1), 4))
        raise RuntimeError(f"DeepSeek {role} failed after {retries} attempts: {_redact(str(last_error), self.api_key)}")

    def plan(
        self,
        local_layout: dict[str, Any],
        retry_feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        feedback_text = (
            "\nThe previous candidate was tested by the local model. Use this feedback to avoid the same failure "
            "without changing the immutable local geometry:\n"
            + json.dumps(retry_feedback, ensure_ascii=False)
            if retry_feedback else ""
        )
        prompt = f"""You are a visual style Planner. Return strict JSON only; never write Python or Pillow code.
Trusted local code owns all geometry, objects, colors, masks, drawing, and validation. You may select only a
small render style for the immutable scene below.
Return exactly:
{{"render_style": {{"background_rgb": [r,g,b], "outline_rgb": [r,g,b], "outline_width": 1}},
  "scene_notes": ["short visual observations"]}}.
Rules:
- background_rgb must be exactly one of {json.dumps([list(value) for value in ALLOWED_BACKGROUNDS])}.
- outline_rgb has three integer channels in 0..80; outline_width is an integer from 1 to 3.
- Do not repeat or modify the layout. Do not add coordinates, objects, text, labels, code, or drawing commands.
- Prefer strong separation for white/black shapes and a visually clear target.
LOCAL_LAYOUT:
{json.dumps(local_layout, ensure_ascii=False)}
{feedback_text}"""
        result = self.call(prompt, "planner")
        notes = result.get("scene_notes")
        return {
            "plan": local_layout,
            "render_style": sanitize_render_style(result.get("render_style")),
            "scene_notes": notes if isinstance(notes, list) else [],
            "planner_json": result,
            "planner_issues": [],
        }

    def plan_recreated_hard(
        self,
        local_layout: dict[str, Any],
        retry_feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Let the generation agent style a freshly regenerated trusted layout."""
        feedback_text = (
            "\nFAILURE_ANALYST_FEEDBACK_FROM_PREVIOUS_ATTEMPT:\n"
            + json.dumps(retry_feedback, ensure_ascii=False)
            if retry_feedback else ""
        )
        prompt = f"""You are the conflict-hard Generation Agent. Return strict JSON only; never write code.
Trusted local code has regenerated the complete layout below and will render and validate it. Select a small
render style that follows the Failure Analyst's advice where possible without altering any layout value.
Return exactly:
{{"render_style": {{"background_rgb": [r,g,b], "outline_rgb": [r,g,b], "outline_width": 1}},
  "applied_advice": ["short notes"]}}.
Rules:
- background_rgb must be exactly one of {json.dumps([list(value) for value in ALLOWED_BACKGROUNDS])}.
- outline_rgb has integer channels in 0..80; outline_width is an integer from 1 to 3.
- Preserve the target shape/color and every object, bbox, role, coordinate, mask and occlusion constraint.
- Do not output objects, coordinates, drawing commands, Python, Markdown, or additional keys.
FRESH_TRUSTED_LAYOUT:
{json.dumps(local_layout, ensure_ascii=False)}
{feedback_text}"""
        result = self.call(prompt, "recreate_generator")
        notes = result.get("applied_advice")
        return {
            "plan": local_layout,
            "render_style": sanitize_render_style(result.get("render_style")),
            "scene_notes": notes if isinstance(notes, list) else [],
            "planner_json": result,
            "planner_issues": [],
        }

    def review_recreate_failure(
        self,
        layout: dict[str, Any],
        failure_reason: str,
        calibration: dict[str, Any] | None,
        easy_entropy: float,
        target: str,
    ) -> dict[str, Any]:
        """Analyze one failed recreate result before the next generation attempt."""
        compact_result = {
            "failure_reason": failure_reason,
            "required_ground_truth_answer": target,
            "top1_answer": calibration.get("top1_answer") if calibration else None,
            "top1_answers": calibration.get("top1_answers") if calibration else None,
            "correct_count": calibration.get("correct_count") if calibration else 0,
            "hard_normalized_entropy": calibration.get("normalized_entropy") if calibration else None,
            "easy_normalized_entropy": easy_entropy,
            "entropy_gap": (
                float(calibration["normalized_entropy"]) - easy_entropy
                if calibration and calibration.get("normalized_entropy") is not None else None
            ),
            "required_entropy_gap_strictly_greater_than": ENTROPY_GAP_THRESHOLD,
        }
        prompt = f"""You are the conflict-hard Failure Analyst. Return strict JSON only:
{{"analysis": ["specific causes"], "retry_instructions": ["minimal actionable suggestions"]}}.
Analyze why the local model did not answer the required conflict ground truth in all three runs and/or why
hard normalized entropy minus easy normalized entropy was not strictly greater than 0.25. The complete failed
layout JSON and measured result are provided below. Give concise advice to the next Generation Agent. Trusted
local code will create a fresh layout and enforces: one target, target color only on target, dissimilar shapes,
180-250px bboxes, non-grid layout, and 70%-80% exact target occlusion. Do not output code or a replacement layout.
FAILED_LAYOUT_JSON:
{json.dumps(layout, ensure_ascii=False)}
FAILED_VALIDATION_RESULT:
{json.dumps(compact_result, ensure_ascii=False)}"""
        result = self.call(prompt, "recreate_failure_analyst")
        if not isinstance(result.get("analysis"), list) or not isinstance(result.get("retry_instructions"), list):
            raise ValueError("Recreate failure analyst schema is invalid")
        result["test_result"] = compact_result
        return result

    def repair_syntax(
        self,
        plan: dict[str, Any],
        code: str,
        diagnostics: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = f"""You are the Code Generator performing a minimal syntax-only repair.
Return strict JSON only:
{{"patched_code": "complete patched Python code", "changes": ["short descriptions"]}}.
The trusted local parser reported the exact diagnostic below. Change only the smallest token range needed
to make the code parse and compile. Do not redesign shapes, layout, masks, colors, functions, imports,
or any already-valid line. Preserve the complete program and render_scene interface.
DIAGNOSTIC:
{json.dumps(diagnostics, ensure_ascii=False)}
IMMUTABLE PLAN:
{json.dumps(plan, ensure_ascii=False)}
CODE:
{code}"""
        result = self.call(prompt, "code_generator_syntax_repair")
        if not isinstance(result.get("patched_code"), str):
            raise ValueError("Syntax repair response has no patched_code")
        if not isinstance(result.get("changes", []), list):
            raise ValueError("Syntax repair changes must be an array")
        return result

    def review_test_result(
        self,
        plan: dict[str, Any],
        calibration: dict[str, Any],
        easy_entropy: float | None,
    ) -> dict[str, Any]:
        compact = {
            "top1_answers": calibration.get("top1_answers"),
            "correct_count": calibration.get("correct_count"),
            "all_correct": calibration.get("all_correct"),
            "normalized_entropy": calibration.get("normalized_entropy"),
            "easy_entropy": easy_entropy,
            "entropy_gap": (
                float(calibration["normalized_entropy"]) - easy_entropy
                if easy_entropy is not None and calibration.get("normalized_entropy") is not None
                else None
            ),
        }
        prompt = f"""You are the Planner reviewing an actual three-run local Qwen image test.
Return strict JSON only:
{{"analysis": ["concise findings"], "retry_instructions": ["minimal actionable changes for the next candidate"]}}.
Do not change the target, branch, colors, candidate set, safety rules, or local RNG authority.
Suggest only changes to the next allowed render_style JSON (neutral background, dark outline, outline width 1-3).
Do not provide Python, Pillow code, geometry, coordinates, colors for objects, or new objects. For hard images,
improve visual uncertainty while retaining the correct answer.
PLAN:
{json.dumps(plan, ensure_ascii=False)}
TEST_RESULT:
{json.dumps(compact, ensure_ascii=False)}"""
        result = self.call(prompt, "planner_test_feedback")
        if not isinstance(result.get("analysis"), list) or not isinstance(result.get("retry_instructions"), list):
            raise ValueError("Planner test feedback schema is invalid")
        result["test_result"] = compact
        return result

    def generate_code(
        self,
        plan: dict[str, Any],
        guidance: list[str] | None = None,
    ) -> dict[str, Any]:
        guidance_text = (
            "\nPlanner guidance based on the previous local-model test:\n"
            + json.dumps(guidance, ensure_ascii=False)
            if guidance else ""
        )
        prompt = f"""You are the Code Generator agent. Return one valid JSON object only:
{{"code": "complete executable Python source"}}.

Generate conservative, deterministic Pillow code. Before responding, mentally parse it and trace one full
render_scene() call. The code must satisfy every item in this checklist:
1. Define all helpers and render_scene(); no placeholders, undefined names, lambda expressions, ellipses,
   leading-zero numbers, prose, Markdown fences, or incomplete branches.
2. Create exactly one 1024x1024 RGB image and two 1024x1024 L masks. Every RGB tuple contains exactly
   three integer values in 0..255. Pillow coordinates may be floats, but color channel values may not.
   Keep each shape entirely inside its listed bbox; no shape may exceed 250 pixels in width or height.
3. Return exactly {{"image": image, "target_mask": target_mask,
   "occluder_mask": occluder_mask, "layout": PLAN_DICT}} from render_scene().
4. Copy PLAN_DICT literally and completely from PLAN. Do not recalculate, round, resize, reorder, correct,
   randomize, or otherwise alter any canvas, seed, object, center, bbox, rotation, size, color, or role value.
5. Draw every PLAN object once using its listed geometry. The target shape/color appears once. Draw the target
   silhouette identically into the RGB image and target_mask before occluders. Draw only role=occluder objects
   into occluder_mask; distractors must never enter occluder_mask.
6. For easy, masks must have zero target/occluder intersection. For hard, the occluder must cover 70%-80% of
   target-mask pixels. Use direct ImageDraw operations on L masks; do not use ImageChops.logical_or on L images.
7. Use only `from PIL import Image, ImageDraw, ImageChops`, math, and random. No files, text/fonts, network,
   shell, subprocess, environment access, dynamic imports, introspection, or side effects outside render_scene().
8. Keep the implementation simple: one generic polygon/ellipse drawing path where practical. Do not invent
   decorative elements, extra objects, shadows, labels, borders, or alternative canvas dimensions.

Return only after checking that the source compiles and that every referenced helper and variable is defined.
PLAN:
{json.dumps(plan, ensure_ascii=False)}
{guidance_text}"""
        result = self.call(prompt, "code_generator")
        if not isinstance(result.get("code"), str) or not result["code"].strip():
            raise ValueError("Code Generator response has no code")
        return result

    def validate_code(
        self,
        plan: dict[str, Any],
        code: str,
        local_failure: str | None = None,
    ) -> dict[str, Any]:
        failure_context = (
            "\nA trusted local AST/subprocess/geometry check reported this failure. You must correct it:\n"
            + local_failure
            if local_failure else ""
        )
        prompt = f"""You are the Code Validator agent. Inspect the Pillow code against the immutable plan and safety contract.
Return strict JSON only: {{"approved": true_or_false, "issues": [strings], "corrected_code": "code or empty"}}.
Mentally parse and execute render_scene() before deciding. Check undefined names, numeric literals, RGB integer
channels, helper signatures, Pillow image/mask modes and dimensions, and the exact four-key return value.
Compare every PLAN value to the returned layout literally: never change, round, reorder, resize, or recompute
canvas, case_seed, objects, centers, bboxes, rotations, sizes, colors, or roles. Verify target uniqueness,
disallowed similar shapes, absence of text, mask correctness, difficulty constraints, and absence of
file/network/shell/subprocess/dynamic-import access. For easy require zero target occlusion; for hard require
70%-80% measured target-mask coverage. Reject any shape or bbox wider or taller than 250 pixels. Do not use
ImageChops.logical_or directly on L masks.
If the code is correct, approve it and return an empty corrected_code. If anything is wrong, make only the
smallest necessary correction, then return one complete executable corrected_code after rechecking this list.
PLAN:
{json.dumps(plan, ensure_ascii=False)}
CODE:
{code}
{failure_context}"""
        result = self.call(prompt, "code_validator")
        if not isinstance(result.get("approved"), bool) or not isinstance(result.get("issues"), list):
            raise ValueError("Code Validator response schema is invalid")
        corrected = result.get("corrected_code", "")
        if corrected is not None and not isinstance(corrected, str):
            raise ValueError("Code Validator corrected_code must be a string")
        return result


def aggregate_model_runs(runs: list[dict[str, Any]], target: str) -> dict[str, Any]:
    top_answers = [normalize(run.get("top1_answer")) or None for run in runs]
    correct_count = sum(
        bool(run.get("parse_success")) and normalize(run.get("top1_answer")) == target
        for run in runs
    )

    def mean_field(name: str) -> float | None:
        values = [float(run[name]) for run in runs if run.get(name) is not None]
        return sum(values) / len(values) if len(values) == len(runs) and values else None

    class_names = sorted({name for run in runs for name in run.get("answer_class_probabilities", {})})
    class_probabilities = {
        name: sum(float(run.get("answer_class_probabilities", {}).get(name, 0.0)) for run in runs) / len(runs)
        for name in class_names
    } if runs else {}
    unanimous = top_answers[0] if top_answers and len(set(top_answers)) == 1 else None
    return {
        "top1_answer": unanimous,
        "top1_answers": top_answers,
        "ground_truth_answer": target,
        "answer_prob": mean_field("answer_prob"),
        "answer_class_probabilities": class_probabilities,
        "entropy": mean_field("entropy"),
        "normalized_entropy": mean_field("normalized_entropy"),
        "parse_success": len(runs) == 3 and all(bool(run.get("parse_success")) for run in runs),
        "correct_count": correct_count,
        "all_correct": len(runs) == 3 and correct_count == 3,
        "elapsed_seconds": sum(float(run.get("elapsed_seconds", 0.0)) for run in runs),
        "runs": runs,
    }


def test_image_three_times(
    inference: Any,
    image_path: Path,
    question: str,
    target: str,
) -> dict[str, Any]:
    prompt = IMAGE_TEST_PROMPT.format(question=question)
    runs: list[dict[str, Any]] = []
    for _ in range(3):
        started = time.perf_counter()
        try:
            result = inference.generate_answer_with_metrics(
                prompt=prompt,
                answer_classes=list(COLORS),
                image_path=str(image_path),
            )
            run = ModelRun(
                top1_answer=getattr(result, "normalized_answer", None),
                ground_truth_answer=target,
                answer_prob=getattr(result, "answer_prob", None),
                answer_class_probabilities=dict(getattr(result, "answer_class_probabilities", {}) or {}),
                entropy=getattr(result, "raw_answer_entropy", None),
                normalized_entropy=getattr(result, "answer_entropy", None),
                parse_success=bool(getattr(result, "parse_success", False)),
                error=getattr(result, "error", None),
                elapsed_seconds=float(getattr(result, "elapsed_seconds", time.perf_counter() - started)),
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            run = ModelRun(
                top1_answer=None, ground_truth_answer=target, answer_prob=None,
                answer_class_probabilities={}, entropy=None, normalized_entropy=None,
                parse_success=False,
                error={"type": type(exc).__name__, "message": str(exc)},
                elapsed_seconds=time.perf_counter() - started,
            )
        runs.append(asdict(run))
    return aggregate_model_runs(runs, target)


class PersistentGPUQueue:
    """Cross-process FIFO JSON queue with a single GPU consumer."""

    def __init__(self, path: str | Path, poll_seconds: float = 0.2):
        self.path = Path(path).resolve()
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.poll_seconds = poll_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            atomic_write_json(self.path, {"version": 1, "next_sequence": 1, "jobs": []})

    def _mutate(self, callback: Any) -> Any:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                payload = load_json(self.path) if self.path.exists() else {
                    "version": 1, "next_sequence": 1, "jobs": [],
                }
                result, changed = callback(payload)
                if changed:
                    payload["updated_at"] = utc_now()
                    atomic_write_json(self.path, payload)
                return result
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def reset_inflight(self) -> None:
        def reset(payload: dict[str, Any]) -> tuple[None, bool]:
            changed = False
            for job in payload.get("jobs", []):
                if job.get("status") == "running":
                    job["status"] = "queued"
                    job["recovered_at"] = utc_now()
                    changed = True
            return None, changed
        self._mutate(reset)

    def enqueue(
        self,
        job_id: str,
        image_path: Path,
        question: str,
        target: str,
        metadata: dict[str, Any],
    ) -> None:
        def add(payload: dict[str, Any]) -> tuple[None, bool]:
            jobs = payload.setdefault("jobs", [])
            existing = next((job for job in jobs if job.get("job_id") == job_id), None)
            if existing is not None:
                error_type = (existing.get("error") or {}).get("type")
                if existing.get("status") == "failed" and error_type == "KeyboardInterrupt":
                    existing.update({
                        "status": "queued",
                        "requeued_at": utc_now(),
                        "image_path": str(image_path.resolve()),
                        "question": question,
                        "target": target,
                        "metadata": metadata,
                        "result": None,
                        "error": None,
                    })
                    return None, True
                return None, False
            sequence = int(payload.get("next_sequence", 1))
            payload["next_sequence"] = sequence + 1
            jobs.append({
                "job_id": job_id,
                "sequence": sequence,
                "status": "queued",
                "created_at": utc_now(),
                "image_path": str(image_path.resolve()),
                "question": question,
                "target": target,
                "metadata": metadata,
                "result": None,
                "error": None,
            })
            return None, True
        self._mutate(add)

    def wait(self, job_id: str, timeout: float) -> dict[str, Any]:
        started = time.monotonic()
        while True:
            def read(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
                job = next((value for value in payload.get("jobs", []) if value.get("job_id") == job_id), None)
                return job, False
            job = self._mutate(read)
            if job is None:
                raise RuntimeError(f"GPU queue lost job {job_id}")
            if job.get("status") == "completed":
                result = job.get("result")
                if not isinstance(result, dict):
                    raise RuntimeError(f"GPU queue job {job_id} has no result")
                return result
            if job.get("status") == "failed":
                error = job.get("error") or {}
                raise RuntimeError(f"GPU queue job {job_id} failed: {error}")
            if time.monotonic() - started > timeout:
                raise TimeoutError(f"Timed out waiting for GPU queue job {job_id}")
            time.sleep(self.poll_seconds)

    def submit_and_wait(
        self,
        job_id: str,
        image_path: Path,
        question: str,
        target: str,
        metadata: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        self.enqueue(job_id, image_path, question, target, metadata)
        return self.wait(job_id, timeout)

    def claim_next(self) -> dict[str, Any] | None:
        def claim(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
            queued = sorted(
                (job for job in payload.get("jobs", []) if job.get("status") == "queued"),
                key=lambda job: int(job.get("sequence", 0)),
            )
            if not queued:
                return None, False
            job = queued[0]
            job["status"] = "running"
            job["started_at"] = utc_now()
            return dict(job), True
        return self._mutate(claim)

    def complete(self, job_id: str, result: dict[str, Any] | None, error: dict[str, str] | None) -> None:
        def finish(payload: dict[str, Any]) -> tuple[None, bool]:
            job = next((value for value in payload.get("jobs", []) if value.get("job_id") == job_id), None)
            if job is None:
                raise RuntimeError(f"GPU queue lost claimed job {job_id}")
            job["status"] = "failed" if error else "completed"
            job["completed_at"] = utc_now()
            job["result"] = result
            job["error"] = error
            return None, True
        self._mutate(finish)

    def has_unfinished(self) -> bool:
        def check(payload: dict[str, Any]) -> tuple[bool, bool]:
            return any(job.get("status") in {"queued", "running"} for job in payload.get("jobs", [])), False
        return bool(self._mutate(check))

    def fail_unfinished(self, error: dict[str, str]) -> None:
        def fail(payload: dict[str, Any]) -> tuple[None, bool]:
            changed = False
            for job in payload.get("jobs", []):
                if job.get("status") in {"queued", "running"}:
                    job["status"] = "failed"
                    job["completed_at"] = utc_now()
                    job["error"] = error
                    changed = True
            return None, changed
        self._mutate(fail)


def consume_gpu_queue(
    queue: PersistentGPUQueue,
    inference: Any,
    producers_done: threading.Event,
    fatal_errors: list[BaseException],
) -> None:
    """Own the model and process queued images strictly by sequence."""
    try:
        queue.reset_inflight()
        while not producers_done.is_set() or queue.has_unfinished():
            job = queue.claim_next()
            if job is None:
                time.sleep(queue.poll_seconds)
                continue
            try:
                result = test_image_three_times(
                    inference,
                    Path(str(job["image_path"])),
                    str(job["question"]),
                    str(job["target"]),
                )
                queue.complete(str(job["job_id"]), result, None)
            except Exception as exc:
                queue.complete(
                    str(job["job_id"]), None,
                    {"type": type(exc).__name__, "message": str(exc)},
                )
    except BaseException as exc:
        fatal_errors.append(exc)
        try:
            queue.fail_unfinished({"type": type(exc).__name__, "message": str(exc)})
        except Exception:
            pass


def select_best_hard(candidates: list[CandidateResult]) -> tuple[CandidateResult | None, str]:
    eligible = [candidate for candidate in candidates if candidate.geometry_valid and candidate.calibration is not None]
    if not eligible:
        return None, "no_geometry_valid_model_tested_candidate"

    def key(candidate: CandidateResult) -> tuple[int, float, float, int]:
        hard_entropy = candidate.calibration.get("normalized_entropy") if candidate.calibration else None
        return (
            candidate.correct_count,
            float(candidate.entropy_gap) if candidate.entropy_gap is not None else -math.inf,
            float(hard_entropy) if hard_entropy is not None else -math.inf,
            -candidate.attempt,
        )

    best = max(eligible, key=key)
    reason = (
        f"selected attempt {best.attempt} by correct_count={best.correct_count}, "
        f"entropy_gap={best.entropy_gap}, hard_entropy="
        f"{best.calibration.get('normalized_entropy') if best.calibration else None}"
    )
    return best, reason


def hard_candidate_passes(candidate: CandidateResult) -> bool:
    return bool(
        candidate.geometry_valid
        and candidate.all_correct
        and candidate.entropy_gap is not None
        and candidate.entropy_gap > ENTROPY_GAP_THRESHOLD
    )


def _load_extended_inference(model_path: Path) -> Any:
    from confidence_test.inference_extension import ExtendedQwenVLInference
    return ExtendedQwenVLInference(model_path=str(model_path))


def _publish_artifacts(source: Path, image_dir: Path, stem: str) -> dict[str, str]:
    image_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "image.png": f"{stem}.png",
        "layout.json": f"{stem}.layout.json",
        "target_mask.png": f"{stem}.target_mask.png",
        "occluder_mask.png": f"{stem}.occluder_mask.png",
    }
    published: dict[str, str] = {}
    for source_name, target_name in mapping.items():
        target = image_dir / target_name
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=image_dir)
        os.close(fd)
        try:
            shutil.copyfile(source / source_name, temporary)
            with open(temporary, "rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        published[source_name] = str(target.resolve())
    return published


def _branch_checkpoint_path(checkpoint_dir: Path, item_id: str, branch: str) -> Path:
    return checkpoint_dir / f"{item_id}_{branch}.json"


def _upgrade_incomplete_branch_layout_scale(state: dict[str, Any]) -> bool:
    """Stamp the current profile without discarding any existing candidate work."""
    if state.get("status") == "completed" or state.get("layout_scale_version") == LAYOUT_SCALE_VERSION:
        return False
    state["layout_scale_migration"] = {
        "from": state.get("layout_scale_version", 1),
        "to": LAYOUT_SCALE_VERSION,
        "at": utc_now(),
        "preserved_easy_attempts": len(state.get("easy_attempts", [])),
        "preserved_hard_attempts": len(state.get("hard_attempts", [])),
        "preserved_accepted_easy": isinstance(state.get("easy"), dict),
    }
    state["layout_scale_version"] = LAYOUT_SCALE_VERSION
    return True


def _stage_candidate_for_resume(
    candidate: CandidateResult,
    checkpoint_dir: Path,
    item_id: str,
    branch: str,
) -> dict[str, Any]:
    value = candidate.state_dict()
    if candidate.geometry_valid and candidate.artifact_dir:
        stage = checkpoint_dir / "artifacts" / f"{item_id}_{branch}_{candidate.difficulty}_{candidate.attempt}"
        stage.mkdir(parents=True, exist_ok=True)
        for name in ("image.png", "layout.json", "agent_layout.json", "target_mask.png", "occluder_mask.png"):
            source = Path(candidate.artifact_dir) / name
            if source.is_file():
                shutil.copyfile(source, stage / name)
        value["artifact_dir"] = str(stage.resolve())
    return value


def _candidate_from_checkpoint(value: dict[str, Any]) -> CandidateResult:
    candidate = CandidateResult(
        attempt=int(value["attempt"]),
        seed=int(value["seed"]),
        difficulty=str(value.get("difficulty", "hard")),
    )
    for key in (
        "geometry_valid", "geometry", "agent_results", "code_sha256", "runs",
        "calibration", "correct_count", "all_correct", "entropy_gap",
        "failure_reason", "artifact_dir", "planner_retry_feedback",
    ):
        if key in value:
            setattr(candidate, key, value[key])
    return candidate


def _terminate_process_pool(
    executor: concurrent.futures.ProcessPoolExecutor,
    grace_seconds: float = 5.0,
) -> None:
    """Cancel queued work, then terminate and reap every live pool worker."""
    processes = list((getattr(executor, "_processes", None) or {}).values())
    executor.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic() + grace_seconds
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        process.join(timeout=remaining)
    for process in processes:
        if process.is_alive():
            process.kill()
    for process in processes:
        process.join(timeout=1.0)


def _run_branch_process(spec: dict[str, Any]) -> dict[str, Any]:
    """Process one complete branch (easy then hard) without loading the GPU model."""
    signal.signal(signal.SIGTERM, _raise_worker_shutdown)
    combo = dict(spec["combo"])
    item_id = str(combo["id"])
    branch = str(spec["branch"])
    target = str(spec["target"])
    output_path = Path(spec["output_path"]).resolve()
    image_dir = Path(spec["image_dir"]).resolve()
    checkpoint_dir = Path(spec["checkpoint_dir"]).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = _branch_checkpoint_path(checkpoint_dir, item_id, branch)
    state = load_json(checkpoint_path) if checkpoint_path.is_file() else {
        "version": 1,
        "layout_scale_version": LAYOUT_SCALE_VERSION,
        "item_id": item_id,
        "branch": branch,
        "target": target,
        "status": "in_progress",
        "easy_attempts": [],
        "hard_attempts": [],
        "created_at": utc_now(),
    }
    if state.get("status") == "completed":
        return {"status": "completed", "item_id": item_id, "branch": branch, "result": state["result"], "state": state}

    def persist() -> None:
        state["updated_at"] = utc_now()
        atomic_write_json(checkpoint_path, state)

    if _upgrade_incomplete_branch_layout_scale(state):
        persist()

    agents = DeepSeekAgents(Path(spec["api_config_path"]))
    queue = PersistentGPUQueue(spec["gpu_queue_path"])
    harness = DatasetGenerator.__new__(DatasetGenerator)
    harness.agents = agents
    harness.inference = None
    harness.output_path = output_path
    harness.image_dir = image_dir

    def publish(candidate: CandidateResult, stem: str) -> dict[str, str]:
        if candidate.artifact_dir is None:
            raise ValueError("Cannot publish candidate without artifacts")
        published = _publish_artifacts(Path(candidate.artifact_dir), image_dir, stem)
        return {
            key: Path(os.path.relpath(Path(value), output_path.parent)).as_posix()
            for key, value in published.items()
        }

    question = QUESTION_TEMPLATE.format(shape=combo["shape"])

    def queued_test(
        difficulty: str,
        attempt: int,
        seed: int,
    ) -> Any:
        def submit(image_path: Path, queued_question: str, queued_target: str) -> dict[str, Any]:
            image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
            identity = f"{item_id}:{branch}:{difficulty}:{attempt}:{seed}:{image_sha256}"
            job_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            return queue.submit_and_wait(
                job_id,
                image_path,
                queued_question,
                queued_target,
                {
                    "item_id": item_id,
                    "branch": branch,
                    "difficulty": difficulty,
                    "attempt": attempt,
                    "case_seed": seed,
                    "image_sha256": image_sha256,
                    "worker_pid": os.getpid(),
                },
                timeout=float(spec["gpu_wait_timeout"]),
            )
        return submit

    try:
        with tempfile.TemporaryDirectory(prefix=f"shape-color-{item_id}-{branch}-") as temporary:
            root = Path(temporary)
            easy_state = state.get("easy")
            if isinstance(easy_state, dict) and easy_state.get("status") == "accepted":
                easy_result = easy_state["result"]
                easy_layout = load_json((output_path.parent / easy_result["layout_path"]).resolve())
                easy_entropy = easy_result["calibration"].get("normalized_entropy")
            else:
                easy_candidate: CandidateResult | None = None
                retry_feedback = (
                    state["easy_attempts"][-1].get("planner_retry_feedback")
                    if state.get("easy_attempts") else None
                )
                start_attempt = len(state.get("easy_attempts", [])) + 1
                for attempt in range(start_attempt, EASY_MAX_ATTEMPTS + 1):
                    attempt_seed = derive_seed(int(combo["case_seeds"][f"{branch}_easy"]), attempt)
                    layout = build_easy_layout(attempt_seed, branch, combo["shape"], target)
                    work = root / f"easy-{attempt}"
                    work.mkdir()
                    candidate = harness._candidate_pipeline(
                        layout, work, question, target, attempt, None,
                        retry_feedback=retry_feedback,
                        model_test=queued_test("easy", attempt, attempt_seed),
                    )
                    state.setdefault("easy_attempts", []).append(
                        _stage_candidate_for_resume(candidate, checkpoint_dir, item_id, branch)
                    )
                    persist()
                    if candidate.geometry_valid and candidate.all_correct:
                        easy_candidate = candidate
                        break
                    retry_feedback = candidate.planner_retry_feedback
                if easy_candidate is None:
                    state.update({
                        "status": "failed",
                        "failure_reason": "easy_attempt_limit_exhausted",
                        "attempt_count": len(state.get("easy_attempts", [])),
                    })
                    persist()
                    return {"status": "failed", "item_id": item_id, "branch": branch, "state": state}
                easy_paths = publish(
                    easy_candidate,
                    f"{item_id}_{'consist' if branch == 'consistent' else 'conflict'}_easy",
                )
                easy_result = {
                    "image_path": easy_paths["image.png"],
                    "layout_path": easy_paths["layout.json"],
                    "target_mask_path": easy_paths["target_mask.png"],
                    "occluder_mask_path": easy_paths["occluder_mask.png"],
                    "case_seed": easy_candidate.seed,
                    "code_sha256": easy_candidate.code_sha256,
                    "geometry": easy_candidate.geometry,
                    "calibration": easy_candidate.calibration,
                }
                state["easy"] = {"status": "accepted", "result": easy_result}
                persist()
                easy_layout = load_json((output_path.parent / easy_result["layout_path"]).resolve())
                easy_entropy = easy_result["calibration"].get("normalized_entropy")

            prior_hard = list(state.get("hard_attempts", []))
            hard_candidates = [_candidate_from_checkpoint(value) for value in prior_hard]
            retry_feedback = prior_hard[-1].get("planner_retry_feedback") if prior_hard else None
            start_attempt = len(prior_hard) + 1
            if not any(hard_candidate_passes(candidate) for candidate in hard_candidates):
                for attempt in range(start_attempt, HARD_MAX_ATTEMPTS + 1):
                    attempt_seed = derive_seed(int(combo["case_seeds"][f"{branch}_hard"]), attempt)
                    layout = build_hard_layout(attempt_seed, easy_layout)
                    work = root / f"hard-{attempt}"
                    work.mkdir()
                    candidate = harness._candidate_pipeline(
                        layout, work, question, target, attempt, easy_entropy,
                        retry_feedback=retry_feedback,
                        model_test=queued_test("hard", attempt, attempt_seed),
                    )
                    hard_candidates.append(candidate)
                    state.setdefault("hard_attempts", []).append(
                        _stage_candidate_for_resume(candidate, checkpoint_dir, item_id, branch)
                    )
                    persist()
                    if hard_candidate_passes(candidate):
                        break
                    retry_feedback = candidate.planner_retry_feedback

            # Resume can retain model results but temporary images from prior failed
            # hard candidates are gone, so only candidates produced by this worker
            # invocation are eligible for final publishing.
            successful = next((candidate for candidate in hard_candidates if hard_candidate_passes(candidate)), None)
            if successful is not None:
                selected = successful
                hard_success = True
                failure_reason = "pass"
                selection_reason = f"attempt {selected.attempt} passed all hard acceptance conditions"
            else:
                selected, selection_reason = select_best_hard(hard_candidates)
                hard_success = False
                failure_reason = "no_hard_candidate_met_accuracy_and_entropy_gap"
                if selected is None:
                    state.update({
                        "status": "failed",
                        "failure_reason": "no_geometry_valid_model_tested_candidate",
                        "attempt_count": len(state.get("hard_attempts", [])),
                    })
                    persist()
                    return {"status": "failed", "item_id": item_id, "branch": branch, "state": state}
            hard_paths = publish(
                selected,
                f"{item_id}_{'consist' if branch == 'consistent' else 'conflict'}_hard",
            )
            hard_result = {
                "image_path": hard_paths["image.png"],
                "layout_path": hard_paths["layout.json"],
                "target_mask_path": hard_paths["target_mask.png"],
                "occluder_mask_path": hard_paths["occluder_mask.png"],
                "case_seed": selected.seed,
                "code_sha256": selected.code_sha256,
                "geometry": selected.geometry,
                "calibration": selected.calibration,
            }
            result = {
                "easy": easy_result,
                "hard": hard_result,
                "entropy_check": {
                    "easy_entropy": easy_entropy,
                    "hard_entropy": selected.calibration.get("normalized_entropy") if selected.calibration else None,
                    "entropy_gap": selected.entropy_gap,
                    "entropy_gap_threshold": ENTROPY_GAP_THRESHOLD,
                    "hard_success": hard_success,
                    "failure_reason": failure_reason,
                    "attempt_count": len(state.get("hard_attempts", [])),
                    "attempts": state.get("hard_attempts", []),
                    "best_selection_reason": selection_reason,
                    "require_hard_top1_correct": True,
                },
            }
            state.update({"status": "completed", "completed_at": utc_now(), "result": result})
            persist()
            return {"status": "completed", "item_id": item_id, "branch": branch, "result": result, "state": state}
    except Exception as exc:
        state.update({
            "status": "failed",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "failed_at": utc_now(),
        })
        persist()
        return {"status": "failed", "item_id": item_id, "branch": branch, "state": state}


def _compact_summary_calibration(calibration: dict[str, Any]) -> dict[str, Any]:
    """Convert full calibration data to the compact valid/invalid dataset schema."""
    runs = []
    for run in calibration.get("runs", []):
        runs.append({
            "answer": run.get("top1_answer"),
            "ground_truth_answer": run.get("ground_truth_answer"),
            "entropy": run.get("entropy"),
            "normalized_entropy": run.get("normalized_entropy"),
            "parse_success": run.get("parse_success"),
            "error": run.get("error"),
        })
    return {
        "answer": calibration.get("top1_answer"),
        "ground_truth_answer": calibration.get("ground_truth_answer"),
        "entropy": calibration.get("entropy"),
        "normalized_entropy": calibration.get("normalized_entropy"),
        "correct_count": calibration.get("correct_count"),
        "all_correct": calibration.get("all_correct"),
        "parse_success": calibration.get("parse_success"),
        "runs": runs,
    }


def _run_recreate_process(spec: dict[str, Any]) -> dict[str, Any]:
    """Regenerate one invalid conflict-hard image for at most twenty attempts."""
    signal.signal(signal.SIGTERM, _raise_worker_shutdown)
    source_id = str(spec["source_id"])
    checkpoint_dir = Path(spec["checkpoint_dir"]).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{source_id}_conflict_hard.json"
    state = load_json(checkpoint_path) if checkpoint_path.is_file() else {
        "version": 1,
        "source_id": source_id,
        "status": "in_progress",
        "target": str(spec["target"]),
        "easy_entropy": float(spec["easy_entropy"]),
        "attempts": [],
        "created_at": utc_now(),
    }
    if state.get("status") == "completed":
        return {"status": "completed", "source_id": source_id, "result": state["result"]}

    def persist() -> None:
        state["updated_at"] = utc_now()
        atomic_write_json(checkpoint_path, state)

    agents = DeepSeekAgents(Path(spec["api_config_path"]))
    queue = PersistentGPUQueue(spec["gpu_queue_path"])
    harness = DatasetGenerator.__new__(DatasetGenerator)
    harness.agents = agents
    harness.inference = None
    easy_layout = load_json(Path(spec["easy_layout_path"]))
    question = str(spec["question"])
    target = str(spec["target"])
    easy_entropy = float(spec["easy_entropy"])
    retry_feedback = (
        state["attempts"][-1].get("planner_retry_feedback")
        if state.get("attempts") else None
    )

    def queued_test(attempt: int, seed: int) -> Any:
        def submit(image_path: Path, queued_question: str, queued_target: str) -> dict[str, Any]:
            image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
            identity = f"recreate:{source_id}:{attempt}:{seed}:{image_sha256}"
            job_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            return queue.submit_and_wait(
                job_id,
                image_path,
                queued_question,
                queued_target,
                {
                    "mode": "recreate",
                    "source_id": source_id,
                    "branch": "conflict",
                    "difficulty": "hard",
                    "attempt": attempt,
                    "case_seed": seed,
                    "image_sha256": image_sha256,
                    "worker_pid": os.getpid(),
                },
                timeout=float(spec["gpu_wait_timeout"]),
            )
        return submit

    try:
        with tempfile.TemporaryDirectory(prefix=f"shape-color-recreate-{source_id}-") as temporary:
            root = Path(temporary)
            start_attempt = len(state.get("attempts", [])) + 1
            for attempt in range(start_attempt, RECREATE_HARD_MAX_ATTEMPTS + 1):
                attempt_seed = derive_seed(int(spec["case_seed"]), "recreate", source_id, attempt)
                layout = build_recreated_hard_layout(attempt_seed, easy_layout)
                work = root / f"hard-{attempt}"
                work.mkdir()
                candidate = harness._candidate_pipeline(
                    layout,
                    work,
                    question,
                    target,
                    attempt,
                    easy_entropy,
                    retry_feedback=retry_feedback,
                    model_test=queued_test(attempt, attempt_seed),
                    recreate_hard=True,
                )
                if candidate.failure_reason != "pass" and candidate.planner_retry_feedback is None:
                    try:
                        candidate.planner_retry_feedback = agents.review_recreate_failure(
                            layout,
                            str(candidate.failure_reason),
                            candidate.calibration,
                            easy_entropy,
                            target,
                        )
                        candidate.agent_results["recreate_failure_feedback"] = candidate.planner_retry_feedback
                    except Exception as feedback_exc:
                        candidate.agent_results["recreate_failure_feedback_error"] = {
                            "type": type(feedback_exc).__name__, "message": str(feedback_exc),
                        }
                staged = _stage_candidate_for_resume(
                    candidate, checkpoint_dir, source_id, "recreate_conflict"
                )
                state.setdefault("attempts", []).append(staged)
                persist()
                if hard_candidate_passes(candidate):
                    result = {
                        "candidate": staged,
                        "easy_entropy": easy_entropy,
                        "hard_entropy": candidate.calibration.get("normalized_entropy"),
                        "entropy_gap": candidate.entropy_gap,
                        "attempt_count": attempt,
                    }
                    state.update({
                        "status": "completed", "completed_at": utc_now(), "result": result,
                    })
                    persist()
                    return {"status": "completed", "source_id": source_id, "result": result}
                retry_feedback = candidate.planner_retry_feedback
            state.update({
                "status": "failed",
                "failure_reason": "recreate_hard_attempt_limit_exhausted",
                "attempt_count": len(state.get("attempts", [])),
                "failed_at": utc_now(),
            })
            persist()
            return {"status": "failed", "source_id": source_id, "state": state}
    except Exception as exc:
        state.update({
            "status": "failed",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "failed_at": utc_now(),
        })
        persist()
        return {"status": "failed", "source_id": source_id, "state": state}


class DatasetGenerator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.input_path = Path(args.input_dataset).resolve()
        self.prior_path = Path(args.prior_pool).resolve()
        self.output_path = Path(args.output_dataset).resolve()
        self.image_dir = Path(args.image_dir).resolve()
        self.model_path = Path(args.model_path).resolve()
        self.state_path = self.output_path.with_suffix(".state.json")
        self.gpu_queue_path = Path(
            getattr(args, "gpu_queue", None) or self.output_path.with_suffix(".gpu_queue.json")
        ).resolve()
        self.branch_checkpoint_dir = self.output_path.with_suffix(".branches")
        self.worker_count = int(getattr(args, "workers", 1))
        self.gpu_wait_timeout = float(getattr(args, "gpu_wait_timeout", 86400.0))
        self.input_payload = load_json(self.input_path)
        self.prior_counts = validate_prior_pool(load_json(self.prior_path))
        self.irr_path, self.null_path = find_shared_clue_paths(
            self.input_path, self.input_payload, self.output_path
        )
        self.state: dict[str, Any]
        self.output: list[dict[str, Any]]
        self.agents: DeepSeekAgents | None = None
        self.inference: Any = None
        self._initialize()

    def _configuration(self) -> dict[str, str]:
        return {
            "input_dataset": str(self.input_path),
            "prior_pool": str(self.prior_path),
            "output_dataset": str(self.output_path),
            "image_dir": str(self.image_dir),
            "model_path": str(self.model_path),
            "gpu_queue": str(self.gpu_queue_path),
            "branch_checkpoint_dir": str(self.branch_checkpoint_dir),
        }

    def _initialize(self) -> None:
        if not self.input_path.is_file() or not self.prior_path.is_file():
            raise FileNotFoundError("Input dataset and prior pool must exist")
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")
        if self.args.dry_run:
            seed = self.args.seed if self.args.seed is not None else secrets.randbits(64)
            manifest = build_manifest(self.input_payload, seed)
            self.state = {"seed": seed, "manifest": manifest, "config": self._configuration()}
            self.output = [{"category": "colour", "items": []}]
            _log("INFO", f"Dry-run initialized: seed={seed} planned={manifest['planned_count']}")
            return
        if self.args.resume:
            if not self.state_path.is_file():
                raise ValueError(f"--resume requires state file: {self.state_path}")
            self.state = load_json(self.state_path)
            current_config = self._configuration()
            saved_config = self.state.get("config")
            if not isinstance(saved_config, dict) or any(
                saved_config.get(key) != value
                for key, value in current_config.items()
                if key not in {"gpu_queue", "branch_checkpoint_dir"}
            ):
                raise ValueError("Resume configuration does not match the saved state")
            saved_config.update({
                "gpu_queue": current_config["gpu_queue"],
                "branch_checkpoint_dir": current_config["branch_checkpoint_dir"],
            })
            saved_seed = self.state.get("seed")
            if self.args.seed is not None and self.args.seed != saved_seed:
                raise ValueError(f"Resume seed mismatch: requested {self.args.seed}, saved {saved_seed}")
            if self.output_path.is_file():
                self.output = load_json(self.output_path)
            else:
                self.output = [{"category": "colour", "items": []}]
            self._validate_output()
            self._reconcile_output()
            self._migrate_legacy_branch_state()
            completed = len(self.output[0]["items"])
            planned = self.state["manifest"]["planned_count"]
            _log("INFO", f"Resumed: seed={saved_seed} completed={completed}/{planned} remaining={planned - completed}")
            self._persist_state()
            return
        if self.output_path.exists() or self.state_path.exists():
            raise ValueError("Output or state already exists; choose new paths or use --resume")
        seed = self.args.seed if self.args.seed is not None else secrets.randbits(64)
        manifest = build_manifest(self.input_payload, seed)
        self.state = {
            "version": 1,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "seed": seed,
            "config": self._configuration(),
            "manifest": manifest,
            "prior_pool_preflight": self.prior_counts,
            "combinations": {},
            "failures": [],
        }
        self.output = [{"category": "colour", "items": []}]
        self._persist_state()
        atomic_write_json(self.output_path, self.output)
        _log("INFO", f"Initialized: seed={seed} existing={manifest['existing_count']} planned={manifest['planned_count']}")

    def _migrate_legacy_branch_state(self) -> None:
        """Move pre-concurrency branch checkpoints into independent worker files."""
        for item_id, item_state in self.state.get("combinations", {}).items():
            if not isinstance(item_state, dict):
                continue
            for branch, legacy in item_state.get("branches", {}).items():
                if not isinstance(legacy, dict):
                    continue
                path = _branch_checkpoint_path(self.branch_checkpoint_dir, str(item_id), str(branch))
                if path.exists():
                    continue
                target = None
                for combo in self.state.get("manifest", {}).get("combinations", []):
                    if str(combo.get("id")) == str(item_id):
                        target = combo.get("text_color") if branch == "consistent" else combo.get("conflict_color")
                        break
                migrated = {
                    "version": 1,
                    "item_id": str(item_id),
                    "branch": branch,
                    "target": target,
                    "status": legacy.get("status", "in_progress"),
                    "easy_attempts": list(legacy.get("easy_attempts", [])),
                    "hard_attempts": list(legacy.get("hard_attempts", [])),
                    "migrated_at": utc_now(),
                }
                if isinstance(legacy.get("easy"), dict):
                    migrated["easy"] = legacy["easy"]
                if legacy.get("status") == "completed" and isinstance(legacy.get("result"), dict):
                    migrated["result"] = legacy["result"]
                atomic_write_json(path, migrated)

    def _validate_output(self) -> None:
        if not isinstance(self.output, list) or len(self.output) != 1:
            raise ValueError("Output dataset must contain exactly one group")
        group = self.output[0]
        if not isinstance(group, dict) or group.get("category") != "colour" or not isinstance(group.get("items"), list):
            raise ValueError("Output dataset group schema is invalid")

    def _reconcile_output(self) -> None:
        complete = self.state.setdefault("combinations", {})
        for item in self.output[0]["items"]:
            item_id = str(item.get("id", ""))
            if item_id:
                complete.setdefault(item_id, {})["status"] = "completed"

    def _persist_state(self) -> None:
        self.state["updated_at"] = utc_now()
        atomic_write_json(self.state_path, self.state)

    def _record_candidate(self, item_id: str, branch_key: str, candidate: CandidateResult) -> None:
        item_state = self.state.setdefault("combinations", {}).setdefault(item_id, {"status": "in_progress", "branches": {}})
        branch = item_state.setdefault("branches", {}).setdefault(branch_key, {"status": "in_progress", "attempts": []})
        attempts = branch.setdefault(f"{candidate.difficulty}_attempts", [])
        attempts[:] = [value for value in attempts if value.get("attempt") != candidate.attempt]
        attempts.append(candidate.state_dict())
        self._persist_state()

    def _candidate_pipeline(
        self,
        expected_layout: dict[str, Any],
        work_dir: Path,
        question: str,
        target: str,
        attempt: int,
        easy_entropy: float | None,
        retry_feedback: dict[str, Any] | None = None,
        model_test: Any | None = None,
        recreate_hard: bool = False,
    ) -> CandidateResult:
        assert self.agents is not None
        if model_test is None:
            assert self.inference is not None
        t_start = time.perf_counter()
        candidate = CandidateResult(
            attempt=attempt,
            seed=int(expected_layout["case_seed"]),
            difficulty=str(expected_layout["difficulty"]),
        )
        difficulty = expected_layout["difficulty"]
        try:
            t0 = time.perf_counter()
            planner = (
                self.agents.plan_recreated_hard(expected_layout, retry_feedback=retry_feedback)
                if recreate_hard else
                self.agents.plan(expected_layout, retry_feedback=retry_feedback)
            )
            _log("INFO", f"    [{difficulty}] planner JSON done in {time.perf_counter() - t0:.1f}s")
            candidate.agent_results = {"planner": planner, "renderer": {
                "type": "trusted_local_renderer",
                "version": LOCAL_RENDERER_VERSION,
                "render_style": planner["render_style"],
            }}
            renderer_identity = json.dumps(
                {
                    "version": LOCAL_RENDERER_VERSION,
                    "layout": expected_layout,
                    "style": planner["render_style"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            candidate.code_sha256 = hashlib.sha256(renderer_identity.encode("utf-8")).hexdigest()
            t_render = time.perf_counter()
            rendered = render_scene_locally(expected_layout, work_dir, planner["render_style"])
            geometry = validate_rendered_scene(rendered, expected_layout)
            _log("INFO", f"    [{difficulty}] trusted render+validate done in {time.perf_counter() - t_render:.2f}s")
            candidate.geometry_valid = True
            candidate.geometry = geometry
            candidate.artifact_dir = str(rendered)
            t_vlm = time.perf_counter()
            calibration = (
                model_test(rendered / "image.png", question, target)
                if model_test is not None
                else test_image_three_times(self.inference, rendered / "image.png", question, target)
            )
            _log("INFO", f"    [{difficulty}] VLM test done in {time.perf_counter() - t_vlm:.1f}s, correct={calibration['correct_count']}/3")
            candidate.runs = calibration["runs"]
            candidate.calibration = calibration
            candidate.correct_count = int(calibration["correct_count"])
            candidate.all_correct = bool(calibration["all_correct"])
            if easy_entropy is not None and calibration.get("normalized_entropy") is not None:
                candidate.entropy_gap = float(calibration["normalized_entropy"]) - easy_entropy
            if not candidate.all_correct:
                candidate.failure_reason = "image_not_correct_in_all_three_runs"
            elif expected_layout["difficulty"] == "hard" and not (
                candidate.entropy_gap is not None and candidate.entropy_gap > ENTROPY_GAP_THRESHOLD
            ):
                candidate.failure_reason = "entropy_gap_not_strictly_greater_than_0.25"
            else:
                candidate.failure_reason = "pass"
            if candidate.failure_reason != "pass":
                try:
                    if recreate_hard:
                        if easy_entropy is None:
                            raise ValueError("Recreate feedback requires easy entropy")
                        candidate.planner_retry_feedback = self.agents.review_recreate_failure(
                            planner["plan"], candidate.failure_reason, calibration,
                            float(easy_entropy), target,
                        )
                    else:
                        candidate.planner_retry_feedback = self.agents.review_test_result(
                            planner["plan"], calibration, easy_entropy
                        )
                    candidate.agent_results["planner_test_feedback"] = candidate.planner_retry_feedback
                except Exception as feedback_exc:
                    candidate.agent_results["planner_test_feedback_error"] = {
                        "type": type(feedback_exc).__name__, "message": str(feedback_exc),
                    }
            _log("INFO", f"    [{difficulty}] pipeline done in {time.perf_counter() - t_start:.1f}s, result={candidate.failure_reason}")
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            candidate.failure_reason = f"{type(exc).__name__}: {exc}"
            _log("ERROR", f"    [{difficulty}] pipeline exception after {time.perf_counter() - t_start:.1f}s: {candidate.failure_reason}")
        return candidate

    def _published_relative(self, absolute: str) -> str:
        return Path(os.path.relpath(Path(absolute), self.output_path.parent)).as_posix()

    def _publish_candidate(self, candidate: CandidateResult, stem: str) -> dict[str, str]:
        if candidate.artifact_dir is None:
            raise ValueError("Cannot publish candidate without artifacts")
        published = _publish_artifacts(Path(candidate.artifact_dir), self.image_dir, stem)
        return {key: self._published_relative(value) for key, value in published.items()}

    def _run_branch(
        self,
        combo: dict[str, Any],
        branch: str,
        target: str,
    ) -> dict[str, Any]:
        item_id = combo["id"]
        branch_started = time.perf_counter()
        _log("INFO", f"[{item_id}] {branch} branch start, target={target}")
        item_state = self.state.setdefault("combinations", {}).setdefault(item_id, {"status": "in_progress", "branches": {}})
        branches = item_state.setdefault("branches", {})
        if branches.get(branch, {}).get("status") == "completed":
            _log("INFO", f"[{item_id}] {branch} branch already completed, skipped")
            return branches[branch]["result"]
        question = QUESTION_TEMPLATE.format(shape=combo["shape"])
        with tempfile.TemporaryDirectory(prefix=f"shape-color-{item_id}-{branch}-") as temporary:
            root = Path(temporary)
            easy_state = branches.get(branch, {}).get("easy")
            if isinstance(easy_state, dict) and easy_state.get("status") == "accepted":
                easy_result = easy_state["result"]
                easy_layout = load_json((self.output_path.parent / easy_result["layout_path"]).resolve())
                easy_entropy = easy_result["calibration"].get("normalized_entropy")
                _log("INFO", f"[{item_id}] {branch}: easy locked (resume), entropy={easy_entropy}")
            else:
                easy_candidate: CandidateResult | None = None
                retry_feedback: dict[str, Any] | None = None
                for attempt in range(1, EASY_MAX_ATTEMPTS + 1):
                    _log("INFO", f"[{item_id}] {branch} easy attempt {attempt}/{EASY_MAX_ATTEMPTS}")
                    attempt_seed = derive_seed(int(combo["case_seeds"][f"{branch}_easy"]), attempt)
                    layout = build_easy_layout(attempt_seed, branch, combo["shape"], target)
                    work = root / f"easy-{attempt}"
                    work.mkdir()
                    candidate = self._candidate_pipeline(
                        layout, work, question, target, attempt, None,
                        retry_feedback=retry_feedback,
                    )
                    self._record_candidate(item_id, branch, candidate)
                    if candidate.geometry_valid and candidate.all_correct:
                        easy_candidate = candidate
                        _log("INFO", f"[{item_id}] {branch} easy accepted on attempt {attempt}")
                        break
                    else:
                        _log("WARN", f"[{item_id}] {branch} easy attempt {attempt} failed: {candidate.failure_reason}")
                        retry_feedback = candidate.planner_retry_feedback
                if easy_candidate is None:
                    _log("STOPPED", f"[{item_id}] {branch} easy failed after {EASY_MAX_ATTEMPTS} attempts")
                    branches.setdefault(branch, {})["easy"] = {
                        "status": "failed", "failure_reason": "easy_attempt_limit_exhausted",
                        "attempt_count": EASY_MAX_ATTEMPTS,
                    }
                    self.state["failures"].append({
                        "item_id": item_id, "branch": branch,
                        "reason": "easy_attempt_limit_exhausted", "timestamp": utc_now(),
                    })
                    self._persist_state()
                    raise GenerationStopped(f"Item {item_id} {branch} easy failed after {EASY_MAX_ATTEMPTS} attempts")
                easy_paths = self._publish_candidate(easy_candidate, f"{item_id}_{'consist' if branch == 'consistent' else 'conflict'}_easy")
                _log("INFO", f"[{item_id}] {branch} easy published, entropy={easy_candidate.calibration.get('normalized_entropy') if easy_candidate.calibration else None}")
                easy_result = {
                    "image_path": easy_paths["image.png"],
                    "layout_path": easy_paths["layout.json"],
                    "target_mask_path": easy_paths["target_mask.png"],
                    "occluder_mask_path": easy_paths["occluder_mask.png"],
                    "case_seed": easy_candidate.seed,
                    "code_sha256": easy_candidate.code_sha256,
                    "geometry": easy_candidate.geometry,
                    "calibration": easy_candidate.calibration,
                }
                branch_state = branches.setdefault(branch, {"status": "in_progress", "attempts": []})
                branch_state["easy"] = {"status": "accepted", "result": easy_result}
                self._persist_state()
                easy_layout = load_json((self.output_path.parent / easy_result["layout_path"]).resolve())
                easy_entropy = easy_result["calibration"].get("normalized_entropy")

            hard_candidates: list[CandidateResult] = []
            retry_feedback = None
            for attempt in range(1, HARD_MAX_ATTEMPTS + 1):
                _log("INFO", f"[{item_id}] {branch} hard attempt {attempt}/{HARD_MAX_ATTEMPTS}")
                attempt_seed = derive_seed(int(combo["case_seeds"][f"{branch}_hard"]), attempt)
                layout = build_hard_layout(attempt_seed, easy_layout)
                work = root / f"hard-{attempt}"
                work.mkdir()
                candidate = self._candidate_pipeline(
                    layout, work, question, target, attempt, easy_entropy,
                    retry_feedback=retry_feedback,
                )
                hard_candidates.append(candidate)
                self._record_candidate(item_id, branch, candidate)
                if hard_candidate_passes(candidate):
                    _log("INFO", f"[{item_id}] {branch} hard accepted on attempt {attempt} (gap={candidate.entropy_gap})")
                    break
                else:
                    _log("WARN", f"[{item_id}] {branch} hard attempt {attempt} failed: {candidate.failure_reason} gap={candidate.entropy_gap}")
                    retry_feedback = candidate.planner_retry_feedback
            successful = next((candidate for candidate in hard_candidates if hard_candidate_passes(candidate)), None)
            if successful is not None:
                selected = successful
                selection_reason = f"attempt {selected.attempt} passed all hard acceptance conditions"
                hard_success = True
                failure_reason = "pass"
            else:
                selected, selection_reason = select_best_hard(hard_candidates)
                hard_success = False
                failure_reason = "no_hard_candidate_met_accuracy_and_entropy_gap"
                if selected is not None:
                    _log("WARN", f"[{item_id}] {branch} hard: no candidate passed, selected best-effort {selection_reason}")
                if selected is None:
                    _log("STOPPED", f"[{item_id}] {branch} hard: no geometry-valid candidate")
                    branch_state = branches.setdefault(branch, {"status": "in_progress"})
                    branch_state["hard"] = {
                        "status": "failed", "hard_success": False,
                        "failure_reason": "no_geometry_valid_model_tested_candidate",
                        "attempt_count": len(hard_candidates),
                    }
                    self._persist_state()
                    raise GenerationStopped(f"Item {item_id} {branch} has no valid hard candidate")
            hard_paths = self._publish_candidate(selected, f"{item_id}_{'consist' if branch == 'consistent' else 'conflict'}_hard")
            hard_result = {
                "image_path": hard_paths["image.png"],
                "layout_path": hard_paths["layout.json"],
                "target_mask_path": hard_paths["target_mask.png"],
                "occluder_mask_path": hard_paths["occluder_mask.png"],
                "case_seed": selected.seed,
                "code_sha256": selected.code_sha256,
                "geometry": selected.geometry,
                "calibration": selected.calibration,
            }
            result = {
                "easy": easy_result,
                "hard": hard_result,
                "entropy_check": {
                    "easy_entropy": easy_entropy,
                    "hard_entropy": selected.calibration.get("normalized_entropy") if selected.calibration else None,
                    "entropy_gap": selected.entropy_gap,
                    "entropy_gap_threshold": ENTROPY_GAP_THRESHOLD,
                    "hard_success": hard_success,
                    "failure_reason": failure_reason,
                    "attempt_count": len(hard_candidates),
                    "attempts": [candidate.public_dict() for candidate in hard_candidates],
                    "best_selection_reason": selection_reason,
                    "require_hard_top1_correct": True,
                },
            }
            branch_state = branches.setdefault(branch, {})
            branch_state.update({"status": "completed", "result": result})
            self._persist_state()
            _log("INFO", f"[{item_id}] {branch} branch done in {time.perf_counter() - branch_started:.1f}s (easy_ok=True, hard_success={hard_success})")
            return result

    def _build_item(self, combo: dict[str, Any], consistent: dict[str, Any], conflict: dict[str, Any]) -> dict[str, Any]:
        def branch_payload(value: dict[str, Any]) -> dict[str, Any]:
            return {
                "easy": value["easy"]["image_path"],
                "easy_calibration": value["easy"]["calibration"],
                "hard": value["hard"]["image_path"],
                "hard_calibration": value["hard"]["calibration"],
                "entropy_check": value["entropy_check"],
            }
        return {
            "id": combo["id"],
            "order": "text_image",
            "question": {"text": QUESTION_TEMPLATE.format(shape=combo["shape"])},
            "answer": combo["text_color"],
            "text_ans": combo["text_color"],
            "candidate_colors": list(COLORS),
            "conflict_ans": combo["conflict_color"],
            "image_clue": {
                "consistent": branch_payload(consistent),
                "conflict": branch_payload(conflict),
                "irr": self.irr_path,
                "null": self.null_path,
            },
        }

    def _append_item(self, item: dict[str, Any]) -> None:
        existing_ids = {str(value.get("id")) for value in self.output[0]["items"]}
        if item["id"] not in existing_ids:
            self.output[0]["items"].append(item)
            atomic_write_json(self.output_path, self.output)
        state = self.state["combinations"].setdefault(item["id"], {})
        state["status"] = "completed"
        state["completed_at"] = utc_now()
        self._persist_state()

    def _merge_branch_result(self, payload: dict[str, Any]) -> None:
        item_id = str(payload["item_id"])
        branch = str(payload["branch"])
        item_state = self.state.setdefault("combinations", {}).setdefault(
            item_id, {"status": "in_progress", "branches": {}}
        )
        branch_state: dict[str, Any] = {
            "status": payload["status"],
            "checkpoint": str(_branch_checkpoint_path(self.branch_checkpoint_dir, item_id, branch)),
        }
        if payload.get("status") == "completed":
            branch_state["result"] = payload["result"]
        else:
            branch_state["failure_reason"] = payload.get("state", {}).get("failure_reason")
            failure = {
                "item_id": item_id,
                "branch": branch,
                "reason": branch_state["failure_reason"],
                "timestamp": utc_now(),
            }
            failures = self.state.setdefault("failures", [])
            if not any(
                value.get("item_id") == item_id
                and value.get("branch") == branch
                and value.get("reason") == branch_state["failure_reason"]
                for value in failures
            ):
                failures.append(failure)
        item_state.setdefault("branches", {})[branch] = branch_state
        self._rebuild_completed_output()
        self._persist_state()

    def _rebuild_completed_output(self) -> None:
        items: list[dict[str, Any]] = []
        existing_items = {
            str(item.get("id")): item
            for item in self.output[0].get("items", [])
            if isinstance(item, dict) and str(item.get("id", ""))
        }
        combinations = self.state.get("combinations", {})
        for combo in self.state["manifest"]["combinations"]:
            if str(combo["id"]) in existing_items:
                items.append(existing_items[str(combo["id"])])
                continue
            item_state = combinations.get(str(combo["id"]), {})
            branches = item_state.get("branches", {}) if isinstance(item_state, dict) else {}
            consistent = branches.get("consistent", {})
            conflict = branches.get("conflict", {})
            if consistent.get("status") == "completed" and conflict.get("status") == "completed":
                items.append(self._build_item(combo, consistent["result"], conflict["result"]))
                item_state["status"] = "completed"
                item_state.setdefault("completed_at", utc_now())
        self.output[0]["items"] = items
        atomic_write_json(self.output_path, self.output)

    def _run_concurrent(self) -> None:
        output_ids = {str(item.get("id")) for item in self.output[0]["items"]}
        specs: list[dict[str, Any]] = []
        for combo in self.state["manifest"]["combinations"]:
            if str(combo["id"]) in output_ids:
                continue
            for branch, target in (
                ("consistent", combo["text_color"]),
                ("conflict", combo["conflict_color"]),
            ):
                specs.append({
                    "combo": combo,
                    "branch": branch,
                    "target": target,
                    "output_path": str(self.output_path),
                    "image_dir": str(self.image_dir),
                    "checkpoint_dir": str(self.branch_checkpoint_dir),
                    "gpu_queue_path": str(self.gpu_queue_path),
                    "gpu_wait_timeout": self.gpu_wait_timeout,
                    "api_config_path": str(ROOT_DIR / "api_config.json"),
                })
        if not specs:
            _log("INFO", "No incomplete branches remain")
            return

        queue = PersistentGPUQueue(self.gpu_queue_path)
        queue.reset_inflight()
        self.inference = _load_extended_inference(self.model_path)
        producers_done = threading.Event()
        fatal_errors: list[BaseException] = []
        consumer = threading.Thread(
            target=consume_gpu_queue,
            args=(queue, self.inference, producers_done, fatal_errors),
            name="gpu-fifo-consumer",
            daemon=True,
        )
        consumer.start()
        failures: list[dict[str, Any]] = []
        max_workers = min(self.worker_count, 64, len(specs))
        _log("INFO", f"Starting {len(specs)} branch tasks with {max_workers} worker processes; GPU queue={self.gpu_queue_path}")
        context = multiprocessing.get_context("spawn")
        executor: concurrent.futures.ProcessPoolExecutor | None = None
        futures: list[concurrent.futures.Future[dict[str, Any]]] = []
        interrupted = False
        try:
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=context,
            )
            futures = [executor.submit(_run_branch_process, spec) for spec in specs]
            for future in concurrent.futures.as_completed(futures):
                payload = future.result()
                self._merge_branch_result(payload)
                if payload.get("status") != "completed":
                    failures.append(payload)
                _log(
                    "INFO" if payload.get("status") == "completed" else "ERROR",
                    f"[{payload['item_id']}] {payload['branch']} worker {payload['status']}",
                )
            executor.shutdown(wait=True, cancel_futures=True)
            executor = None
        except BaseException as exc:
            interrupted = isinstance(exc, KeyboardInterrupt)
            for future in futures:
                future.cancel()
            try:
                queue.fail_unfinished({
                    "type": "KeyboardInterrupt" if interrupted else type(exc).__name__,
                    "message": "generation interrupted; queued GPU work cancelled" if interrupted else str(exc),
                })
            except Exception as queue_exc:
                _log("WARN", f"Could not mark GPU queue jobs cancelled: {queue_exc}")
            if executor is not None:
                _log("WARN", "Stopping all branch worker processes")
                _terminate_process_pool(executor)
                executor = None
            raise
        finally:
            producers_done.set()
            consumer.join(timeout=10.0 if interrupted else 60.0)
        if consumer.is_alive():
            raise RuntimeError("GPU queue consumer did not stop")
        if fatal_errors:
            raise RuntimeError(f"GPU queue consumer failed: {fatal_errors[0]}")
        if failures:
            raise GenerationStopped(f"{len(failures)} branch worker(s) failed; resume checkpoints were preserved")

    def run(self) -> None:
        manifest = self.state["manifest"]
        total_planned = manifest["planned_count"]
        already_done = len(self.output[0]["items"])
        _log("INFO", f"seed={self.state['seed']} existing={manifest['existing_count']} planned={total_planned} already_completed={already_done}")
        if self.args.dry_run:
            self.run_dry_run()
            return
        t_run = time.perf_counter()
        self._run_concurrent()
        total_elapsed = time.perf_counter() - t_run
        final_count = len(self.output[0]["items"])
        failures = self.state.get("failures", [])
        _log("INFO", f"=== DONE: {final_count}/{total_planned} items, {len(failures)} failures, total time {total_elapsed:.0f}s ({total_elapsed/60:.1f}min) ===")

    def run_dry_run(self) -> None:
        manifest = self.state["manifest"]
        if not manifest["combinations"]:
            raise ValueError("Dry-run requires at least one missing combination")
        combo = manifest["combinations"][0]
        self.agents = DeepSeekAgents(ROOT_DIR / "api_config.json")
        self.inference = _load_extended_inference(self.model_path)
        with tempfile.TemporaryDirectory(prefix="shape-color-real-dry-run-") as temporary:
            root = Path(temporary)
            failures: list[str] = []
            candidate: CandidateResult | None = None
            retry_feedback: dict[str, Any] | None = None
            for attempt in range(1, EASY_MAX_ATTEMPTS + 1):
                seed = derive_seed(int(combo["case_seeds"]["consistent_easy"]), attempt)
                layout = build_easy_layout(seed, "consistent", combo["shape"], combo["text_color"])
                work = root / f"easy-{attempt}"
                work.mkdir()
                current = self._candidate_pipeline(
                    layout, work, QUESTION_TEMPLATE.format(shape=combo["shape"]),
                    combo["text_color"], attempt, None,
                    retry_feedback=retry_feedback,
                )
                if current.geometry_valid and current.calibration is not None:
                    candidate = current
                    break
                failures.append(str(current.failure_reason))
                retry_feedback = current.planner_retry_feedback
            if candidate is None:
                raise RuntimeError(
                    f"Dry-run pipeline failed after {EASY_MAX_ATTEMPTS} candidates: {failures}"
                )
            print(json.dumps({
                "dry_run": True,
                "persisted": False,
                "candidate_attempt": candidate.attempt,
                "prior_candidate_failures": failures,
                "item": {key: combo[key] for key in ("id", "shape", "text_color", "conflict_color")},
                "geometry": candidate.geometry,
                "model_all_correct": candidate.all_correct,
                "model_runs": candidate.runs,
            }, ensure_ascii=False, indent=2))


class RecreateDatasetGenerator:
    """Recreate invalid conflict-hard images and append only passing items."""

    GROUP_STEMS = {
        "consistent_easy": "consist_easy",
        "consistent_hard": "consist_hard",
        "conflict_easy": "conflict_easy",
        "conflict_hard": "conflict_hard",
    }

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.datasets_root = Path(__file__).resolve().parent / "datasets"
        self.invalid_root = self.datasets_root / "invalid_datasets"
        self.valid_root = self.datasets_root / "valid_datasets"
        self.invalid_path = self.invalid_root / "generated_shape_color_dataset.json"
        self.output_path = self.valid_root / "generated_shape_color_dataset.json"
        self.image_dir = self.valid_root / "images"
        self.source_summary_path = self.datasets_root / "generated_shape_color_dataset.summary.json"
        self.state_path = self.valid_root / "generated_shape_color_dataset.recreate.state.json"
        self.checkpoint_dir = self.valid_root / "generated_shape_color_dataset.recreate.branches"
        self.gpu_queue_path = Path(
            args.gpu_queue or self.valid_root / "generated_shape_color_dataset.recreate.gpu_queue.json"
        ).resolve()
        self.model_path = Path(args.model_path).resolve()
        self.gpu_wait_timeout = float(args.gpu_wait_timeout)
        if not self.invalid_path.is_file() or not self.output_path.is_file():
            raise FileNotFoundError(
                "--recreate requires invalid_datasets and valid_datasets generated_shape_color_dataset.json"
            )
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")
        self.invalid = load_json(self.invalid_path)
        self.output = load_json(self.output_path)
        self._validate_summary_dataset(self.invalid, "invalid")
        self._validate_summary_dataset(self.output, "valid")
        self.records, recovered = self._resolve_source_records()
        self.state: dict[str, Any]
        self._initialize_state(recovered)

    @staticmethod
    def _validate_summary_dataset(payload: Any, name: str) -> None:
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise ValueError(f"{name} recreate dataset must be an object with an items array")
        if str(payload.get("schema_version", "")).startswith("shape_color_dataset."):
            raise ValueError("Legacy split/refine/recreate tools cannot process shape_color_dataset.v2")
        ids = [str(item.get("id")) for item in payload["items"] if isinstance(item, dict)]
        if len(ids) != len(payload["items"]) or len(ids) != len(set(ids)):
            raise ValueError(f"{name} recreate dataset item IDs must be present and unique")

    @staticmethod
    def _group_artifacts(root: Path, item: dict[str, Any], group_name: str) -> dict[str, Path]:
        groups = item.get("groups")
        group = groups.get(group_name) if isinstance(groups, dict) else None
        image_value = group.get("image") if isinstance(group, dict) else None
        if not isinstance(image_value, str):
            raise ValueError(f"item {item.get('id')} has no {group_name} image")
        image = (root / image_value).resolve()
        return {
            "image.png": image,
            "layout.json": image.with_suffix(".layout.json"),
            "target_mask.png": image.with_suffix(".target_mask.png"),
            "occluder_mask.png": image.with_suffix(".occluder_mask.png"),
        }

    def _item_assets_match_labels(self, root: Path, item: dict[str, Any]) -> bool:
        shape = parse_shape(str(item.get("question", "")))
        if shape is None:
            return False
        for group_name in self.GROUP_STEMS:
            expected_color = item.get("answer") if group_name.startswith("consistent") else item.get("conflict_answer")
            try:
                artifacts = self._group_artifacts(root, item, group_name)
                if not all(path.is_file() for path in artifacts.values()):
                    return False
                layout = load_json(artifacts["layout.json"])
            except Exception:
                return False
            if layout.get("target_shape") != shape or layout.get("target_color") != expected_color:
                return False
        return True

    def _resolve_source_records(self) -> tuple[list[dict[str, Any]], int]:
        original_items: list[dict[str, Any]] = []
        if self.source_summary_path.is_file():
            source_summary = load_json(self.source_summary_path)
            if isinstance(source_summary, dict) and isinstance(source_summary.get("items"), list):
                original_items = [item for item in source_summary["items"] if isinstance(item, dict)]
        original_index: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for item in original_items:
            key = (str(item.get("question")), str(item.get("answer")), str(item.get("conflict_answer")))
            original_index.setdefault(key, []).append(item)

        records: list[dict[str, Any]] = []
        recovered = 0
        for invalid_item in self.invalid["items"]:
            source_item = invalid_item
            source_root = self.invalid_root
            source_kind = "invalid"
            if not self._item_assets_match_labels(source_root, source_item):
                key = (
                    str(invalid_item.get("question")),
                    str(invalid_item.get("answer")),
                    str(invalid_item.get("conflict_answer")),
                )
                matches = original_index.get(key, [])
                if len(matches) != 1 or not self._item_assets_match_labels(self.datasets_root, matches[0]):
                    raise ValueError(
                        f"invalid item {invalid_item.get('id')} has mismatched assets and no unique recoverable source"
                    )
                source_item = matches[0]
                source_root = self.datasets_root
                source_kind = "original_fallback"
                recovered += 1

            easy_group = source_item["groups"]["conflict_easy"]
            easy_entropy = easy_group.get("normalized_entropy")
            if not isinstance(easy_entropy, (int, float)):
                raise ValueError(f"invalid item {invalid_item.get('id')} has no numeric conflict-easy entropy")
            easy_layout_path = self._group_artifacts(
                source_root, source_item, "conflict_easy"
            )["layout.json"]
            easy_layout = load_json(easy_layout_path)
            target = str(invalid_item.get("conflict_answer"))
            shape = parse_shape(str(invalid_item.get("question", "")))
            if shape is None or easy_layout.get("target_shape") != shape or easy_layout.get("target_color") != target:
                raise ValueError(f"invalid item {invalid_item.get('id')} conflict-easy baseline is inconsistent")
            records.append({
                "source_id": str(invalid_item["id"]),
                "invalid_item": invalid_item,
                "source_item": source_item,
                "source_root": source_root,
                "source_kind": source_kind,
                "easy_layout_path": easy_layout_path,
                "easy_entropy": float(easy_entropy),
                "target": target,
                "question": str(invalid_item["question"]),
            })
        return records, recovered

    def _configuration(self) -> dict[str, Any]:
        return {
            "mode": "recreate",
            "invalid_dataset": str(self.invalid_path.resolve()),
            "valid_dataset": str(self.output_path.resolve()),
            "valid_image_dir": str(self.image_dir.resolve()),
            "model_path": str(self.model_path),
            "gpu_queue": str(self.gpu_queue_path),
            "max_attempts": RECREATE_HARD_MAX_ATTEMPTS,
            "invalid_sha256": hashlib.sha256(self.invalid_path.read_bytes()).hexdigest(),
        }

    def _initialize_state(self, recovered: int) -> None:
        current = self._configuration()
        if self.args.resume:
            if not self.state_path.is_file():
                raise ValueError(f"--recreate --resume requires state file: {self.state_path}")
            self.state = load_json(self.state_path)
            saved = self.state.get("config")
            if not isinstance(saved, dict) or any(saved.get(key) != value for key, value in current.items()):
                raise ValueError("Recreate resume configuration does not match the saved state")
            if self.args.seed is not None and self.args.seed != self.state.get("seed"):
                raise ValueError(
                    f"Recreate resume seed mismatch: requested {self.args.seed}, saved {self.state.get('seed')}"
                )
        else:
            if self.state_path.exists():
                raise ValueError("Recreate state already exists; use --recreate --resume")
            seed = self.args.seed if self.args.seed is not None else secrets.randbits(64)
            self.state = {
                "version": 1,
                "mode": "recreate",
                "cycle": 1,
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "seed": seed,
                "config": current,
                "input_count": len(self.records),
                "recovered_source_count": recovered,
                "items": {},
                "failures": [],
            }
            self._persist_state()

    def _persist_state(self) -> None:
        self.state["updated_at"] = utc_now()
        atomic_write_json(self.state_path, self.state)

    def _next_output_id(self) -> str:
        existing = [int(str(item["id"])) for item in self.output["items"] if str(item.get("id", "")).isdigit()]
        assigned = [
            int(str(value["assigned_id"]))
            for value in self.state.get("items", {}).values()
            if isinstance(value, dict) and str(value.get("assigned_id", "")).isdigit()
        ]
        return f"{max(existing + assigned, default=0) + 1:03d}"

    @staticmethod
    def _atomic_copy(source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        os.close(fd)
        try:
            shutil.copyfile(source, temporary)
            with open(temporary, "rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    def _write_invalid_dataset(self) -> None:
        self.invalid["item_count"] = len(self.invalid["items"])
        self.invalid["group_count"] = len(self.invalid["items"]) * 4
        self.invalid["generated_at_utc"] = utc_now()
        atomic_write_json(self.invalid_path, self.invalid)

    def _verify_valid_publication(self, source_id: str) -> str:
        item_state = self.state.get("items", {}).get(source_id, {})
        assigned_id = str(item_state.get("assigned_id", ""))
        valid_item = next(
            (item for item in self.output["items"] if str(item.get("id")) == assigned_id),
            None,
        )
        valid_files = list(self.image_dir.glob(f"{assigned_id}_*")) if assigned_id else []
        if valid_item is None or len(valid_files) != 16:
            raise ValueError(
                f"Cannot remove invalid {source_id}: valid publication {assigned_id!r} is incomplete"
            )
        return assigned_id

    def _remove_invalid_source(self, source_id: str) -> int:
        """Remove one source only after its complete valid copy has been verified."""
        source_id = str(source_id)
        matching = [item for item in self.invalid["items"] if str(item.get("id")) == source_id]
        if not matching:
            return 0
        self._verify_valid_publication(source_id)
        source_files = sorted(self.invalid_root.joinpath("images").glob(f"{source_id}_*"))
        if len(source_files) != 16:
            raise ValueError(
                f"Cannot remove invalid {source_id}: expected 16 source artifacts, found {len(source_files)}"
            )
        self.invalid["items"] = [
            item for item in self.invalid["items"] if str(item.get("id")) != source_id
        ]
        self._write_invalid_dataset()
        for path in source_files:
            path.unlink()
        return len(source_files)

    def _cleanup_published_residuals(self) -> None:
        remaining_ids = {str(item.get("id")) for item in self.invalid["items"]}
        cleaned = False
        for source_id, item_state in self.state.get("items", {}).items():
            if not isinstance(item_state, dict) or item_state.get("status") != "published":
                continue
            if str(source_id) in remaining_ids:
                removed = self._remove_invalid_source(str(source_id))
                item_state["invalid_removed_at"] = utc_now()
                item_state["invalid_removed_file_count"] = removed
                cleaned = True
        if cleaned:
            self._persist_state()

    def _renumber_invalid_dataset(self) -> dict[str, str]:
        """Compact all remaining invalid IDs and artifact names after a full worker cycle."""
        items = list(self.invalid["items"])
        mapping = {
            str(item["id"]): f"{index:03d}"
            for index, item in enumerate(items, start=1)
        }
        if all(old == new for old, new in mapping.items()):
            self._write_invalid_dataset()
            return mapping

        image_dir = self.invalid_root / "images"
        staging = Path(tempfile.mkdtemp(prefix=".invalid-renumber-", dir=self.invalid_root))
        try:
            expected_count = 0
            for old_id, new_id in mapping.items():
                source_files = sorted(image_dir.glob(f"{old_id}_*"))
                if len(source_files) != 16:
                    raise ValueError(
                        f"Cannot renumber invalid {old_id}: expected 16 artifacts, found {len(source_files)}"
                    )
                for source in source_files:
                    suffix = source.name[len(old_id):]
                    shutil.copyfile(source, staging / f"{new_id}{suffix}")
                    expected_count += 1
            if len(list(staging.iterdir())) != expected_count:
                raise RuntimeError("Invalid renumber staging file count mismatch")

            for item in items:
                old_id = str(item["id"])
                new_id = mapping[old_id]
                item["id"] = new_id
                for group in item.get("groups", {}).values():
                    if not isinstance(group, dict) or not isinstance(group.get("image"), str):
                        continue
                    old_name = Path(group["image"]).name
                    if not old_name.startswith(old_id + "_"):
                        raise ValueError(f"Invalid image path does not match item {old_id}: {old_name}")
                    group["image"] = f"images/{new_id}{old_name[len(old_id):]}"

            for old_id in mapping:
                for source in image_dir.glob(f"{old_id}_*"):
                    source.unlink()
            for staged in sorted(staging.iterdir()):
                os.replace(staged, image_dir / staged.name)
            self.invalid["items"] = items
            self._write_invalid_dataset()
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return mapping

    def _finalize_recreate_cycle(self, failures: list[dict[str, Any]]) -> None:
        cycle = int(self.state.get("cycle", 1))
        mapping = self._renumber_invalid_dataset()
        statuses = Counter(
            value.get("status", "unknown")
            for value in self.state.get("items", {}).values()
            if isinstance(value, dict)
        )
        self.state.setdefault("cycles", []).append({
            "cycle": cycle,
            "completed_at": utc_now(),
            "published_count": int(statuses.get("published", 0)),
            "failed_count": len(failures),
            "remaining_invalid_count": len(self.invalid["items"]),
            "renumber_mapping": mapping,
        })
        self.state["cycle"] = cycle + 1
        self.state["items"] = {}
        self.state["failures"] = []
        self.state["input_count"] = len(self.invalid["items"])
        self.state["config"] = self._configuration()
        self._persist_state()

    def _publish_success(self, record: dict[str, Any], payload: dict[str, Any]) -> None:
        source_id = record["source_id"]
        item_state = self.state.setdefault("items", {}).setdefault(source_id, {})
        final_id = str(item_state.get("assigned_id") or self._next_output_id())
        item_state.update({"status": "publishing", "assigned_id": final_id})
        self._persist_state()

        candidate = payload["result"]["candidate"]
        calibration = candidate.get("calibration")
        artifact_dir = Path(str(candidate.get("artifact_dir", "")))
        if not isinstance(calibration, dict) or not artifact_dir.is_dir():
            raise ValueError(f"recreate result {source_id} has no durable calibrated artifacts")
        easy_entropy = float(record["easy_entropy"])
        hard_entropy = calibration.get("normalized_entropy")
        gap = float(hard_entropy) - easy_entropy if hard_entropy is not None else None
        if not (
            calibration.get("all_correct") is True
            and normalize(calibration.get("top1_answer")) == normalize(record["target"])
            and gap is not None and gap > ENTROPY_GAP_THRESHOLD
        ):
            raise ValueError(f"recreate result {source_id} failed final answer/entropy validation")
        generated_layout = load_json(artifact_dir / "layout.json")
        validate_rendered_scene(artifact_dir, generated_layout)

        destination_paths: dict[str, dict[str, Path]] = {}
        for group_name, stem_suffix in self.GROUP_STEMS.items():
            stem = f"{final_id}_{stem_suffix}"
            destination_paths[group_name] = {
                "image.png": self.image_dir / f"{stem}.png",
                "layout.json": self.image_dir / f"{stem}.layout.json",
                "target_mask.png": self.image_dir / f"{stem}.target_mask.png",
                "occluder_mask.png": self.image_dir / f"{stem}.occluder_mask.png",
            }
            sources = (
                {name: artifact_dir / name for name in destination_paths[group_name]}
                if group_name == "conflict_hard" else
                self._group_artifacts(record["source_root"], record["source_item"], group_name)
            )
            if not all(path.is_file() for path in sources.values()):
                raise FileNotFoundError(f"source artifacts missing for {source_id}/{group_name}")
            for name, destination in destination_paths[group_name].items():
                self._atomic_copy(sources[name], destination)

        groups: dict[str, Any] = {}
        for group_name, stem_suffix in self.GROUP_STEMS.items():
            if group_name == "conflict_hard":
                value = _compact_summary_calibration(calibration)
            else:
                value = dict(record["source_item"]["groups"][group_name])
            value["image"] = f"images/{final_id}_{stem_suffix}.png"
            groups[group_name] = value
        new_item = {
            "id": final_id,
            "question": record["invalid_item"]["question"],
            "answer": record["invalid_item"]["answer"],
            "conflict_answer": record["target"],
            "groups": groups,
        }
        existing = next((item for item in self.output["items"] if str(item.get("id")) == final_id), None)
        if existing is None:
            self.output["items"].append(new_item)
        elif existing != new_item:
            raise ValueError(f"valid output ID {final_id} already contains different data")
        self.output["item_count"] = len(self.output["items"])
        self.output["group_count"] = len(self.output["items"]) * 4
        self.output["generated_at_utc"] = utc_now()
        atomic_write_json(self.output_path, self.output)
        item_state.update({
            "status": "published",
            "published_at": utc_now(),
            "source_kind": record["source_kind"],
            "attempt_count": payload["result"]["attempt_count"],
            "easy_entropy": easy_entropy,
            "hard_entropy": hard_entropy,
            "entropy_gap": gap,
        })
        self._persist_state()
        removed = self._remove_invalid_source(source_id)
        item_state.update({
            "invalid_removed_at": utc_now(),
            "invalid_removed_file_count": removed,
        })
        self._persist_state()
        _log("INFO", f"[recreate {source_id}->{final_id}] published, entropy_gap={gap}")

    def _spec(self, record: dict[str, Any]) -> dict[str, Any]:
        cycle = int(self.state.get("cycle", 1))
        return {
            "source_id": record["source_id"],
            "question": record["question"],
            "target": record["target"],
            "easy_entropy": record["easy_entropy"],
            "easy_layout_path": str(record["easy_layout_path"]),
            "case_seed": derive_seed(int(self.state["seed"]), "recreate-item", record["source_id"]),
            "checkpoint_dir": str(self.checkpoint_dir / f"cycle_{cycle:03d}"),
            "gpu_queue_path": str(self.gpu_queue_path),
            "gpu_wait_timeout": self.gpu_wait_timeout,
            "api_config_path": str(ROOT_DIR / "api_config.json"),
        }

    def run(self) -> None:
        self._cleanup_published_residuals()
        pending = [
            record for record in self.records
            if self.state.get("items", {}).get(record["source_id"], {}).get("status") != "published"
        ]
        if not pending:
            if self.state.get("items"):
                self._finalize_recreate_cycle([])
            _log("INFO", "No invalid conflict-hard images remain to recreate")
            return
        queue = PersistentGPUQueue(self.gpu_queue_path)
        queue.reset_inflight()
        inference = _load_extended_inference(self.model_path)
        producers_done = threading.Event()
        fatal_errors: list[BaseException] = []
        consumer = threading.Thread(
            target=consume_gpu_queue,
            args=(queue, inference, producers_done, fatal_errors),
            name="recreate-gpu-fifo-consumer",
            daemon=True,
        )
        consumer.start()
        executor: concurrent.futures.ProcessPoolExecutor | None = None
        futures: list[concurrent.futures.Future[dict[str, Any]]] = []
        failures: list[dict[str, Any]] = []
        interrupted = False
        try:
            worker_count = len(pending)
            _log(
                "INFO",
                f"Recreating {len(pending)} conflict-hard images with one worker per image; "
                f"max_attempts={RECREATE_HARD_MAX_ATTEMPTS}",
            )
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=multiprocessing.get_context("spawn"),
            )
            futures = [executor.submit(_run_recreate_process, self._spec(record)) for record in pending]
            # Read results in input order so successful items are appended and numbered deterministically.
            for record, future in zip(pending, futures):
                payload = future.result()
                if payload.get("status") == "completed":
                    self._publish_success(record, payload)
                else:
                    failure_reason = payload.get("state", {}).get("failure_reason")
                    self.state.setdefault("items", {}).setdefault(record["source_id"], {}).update({
                        "status": "failed", "failure_reason": failure_reason, "failed_at": utc_now(),
                    })
                    self.state.setdefault("failures", []).append({
                        "source_id": record["source_id"], "reason": failure_reason, "timestamp": utc_now(),
                    })
                    self._persist_state()
                    failures.append(payload)
                    _log("ERROR", f"[recreate {record['source_id']}] failed: {failure_reason}")
            executor.shutdown(wait=True, cancel_futures=True)
            executor = None
            self._finalize_recreate_cycle(failures)
        except BaseException as exc:
            interrupted = isinstance(exc, KeyboardInterrupt)
            for future in futures:
                future.cancel()
            try:
                queue.fail_unfinished({
                    "type": "KeyboardInterrupt" if interrupted else type(exc).__name__,
                    "message": "recreate interrupted; queued GPU work cancelled" if interrupted else str(exc),
                })
            except Exception as queue_exc:
                _log("WARN", f"Could not cancel recreate GPU jobs: {queue_exc}")
            if executor is not None:
                _log("WARN", "Stopping all recreate worker processes")
                _terminate_process_pool(executor)
                executor = None
            raise
        finally:
            producers_done.set()
            consumer.join(timeout=10.0 if interrupted else 60.0)
        if consumer.is_alive():
            raise RuntimeError("Recreate GPU queue consumer did not stop")
        if fatal_errors:
            raise RuntimeError(f"Recreate GPU queue consumer failed: {fatal_errors[0]}")
        if failures:
            raise GenerationStopped(
                f"{len(failures)} recreate image(s) exhausted {RECREATE_HARD_MAX_ATTEMPTS} attempts"
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a resumable standalone shape × text-colour image dataset"
    )
    parser.add_argument("--input-dataset", default=str(ROOT_DIR / "datasets/dataset_test.json"))
    parser.add_argument("--prior-pool", default=str(ROOT_DIR / "datasets/color_prior_pool.json"))
    parser.add_argument("--output-dataset", default=str(ROOT_DIR / "generation_v2_outputs/formal/image/shape_color_dataset.json"))
    parser.add_argument("--image-dir", default=str(ROOT_DIR / "generation_v2_outputs/formal/image/images"))
    parser.add_argument(
        "--model-path",
        default=str(ROOT_DIR / "qwen-2.5-vl/models/Qwen2.5-VL-7B-Instruct"),
    )
    parser.add_argument("--api-config-path", default=str(ROOT_DIR / "api_config.json"))
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--workers", "--processes",
        type=int,
        default=16,
        help="Concurrent branch worker processes (1-64); each worker runs easy then hard",
    )
    parser.add_argument(
        "--gpu-queue",
        help="Persistent FIFO JSON queue path (default: output dataset with .gpu_queue.json suffix)",
    )
    parser.add_argument(
        "--gpu-wait-timeout",
        type=float,
        default=86400.0,
        help="Maximum seconds a branch worker waits for its queued GPU test",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help=(
            "Recreate only invalid_datasets conflict-hard images (20 attempts, one worker per image) "
            "and append passing items to valid_datasets"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run one real temporary DeepSeek + Qwen smoke case without persisting outputs",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use the legacy confidence/prior-pool generator instead of the V2 image producer",
    )
    parser.add_argument("--branches", default="conflict", help="V2 branches: conflict or conflict,consistent")
    parser.add_argument("--images-per-difficulty", type=int, default=1)
    parser.add_argument("--conflict-easy-count", type=int)
    parser.add_argument("--conflict-hard-count", type=int)
    parser.add_argument("--target-rotation-mode", choices=("safe", "range", "none"), default="safe")
    parser.add_argument("--target-rotation-min", type=float, default=-30.0)
    parser.add_argument("--target-rotation-max", type=float, default=30.0)
    parser.add_argument("--distractor-rotation-min", type=float, default=0.0)
    parser.add_argument("--distractor-rotation-max", type=float, default=360.0)
    parser.add_argument("--similarity-model-path")
    parser.add_argument("--download-similarity-model", action="store_true")
    parser.add_argument("--old-image-dataset", default=str(ROOT_DIR / "generate dataset/datasets/generated_shape_color_dataset.json"))
    parser.add_argument("--old-image-root", default=str(ROOT_DIR / "generate dataset/datasets/generated_shape_color_images"))
    parser.add_argument("--qwen-batch-size", type=int, default=8)
    parser.add_argument("--qwen-batch-wait-ms", type=int, default=500)
    parser.add_argument("--qwen-wait-timeout", type=float, default=86400.0)
    args = parser.parse_args(argv)
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be non-negative")
    if not 1 <= args.workers <= 64:
        parser.error("--workers must be between 1 and 64")
    if args.gpu_wait_timeout <= 0:
        parser.error("--gpu-wait-timeout must be positive")
    if args.resume and args.dry_run:
        parser.error("--resume and --dry-run cannot be combined")
    if args.recreate and args.dry_run:
        parser.error("--recreate and --dry-run cannot be combined")
    branches = [part.strip().casefold() for part in str(args.branches).split(",") if part.strip()]
    if branches not in (["conflict"], ["conflict", "consistent"]):
        parser.error("--branches must be conflict or conflict,consistent")
    args.branches = branches
    if not 1 <= args.images_per_difficulty <= 32:
        parser.error("--images-per-difficulty must be between 1 and 32")
    for name in ("conflict_easy_count", "conflict_hard_count"):
        value = getattr(args, name)
        if value is not None and not 1 <= value <= 32:
            parser.error(f"--{name.replace('_', '-')} must be between 1 and 32")
    easy = args.conflict_easy_count or args.images_per_difficulty
    hard = args.conflict_hard_count or args.images_per_difficulty
    if hard > easy:
        parser.error("conflict hard count cannot exceed conflict easy count")
    if args.target_rotation_min > args.target_rotation_max:
        parser.error("target rotation min cannot exceed max")
    if args.distractor_rotation_min > args.distractor_rotation_max:
        parser.error("distractor rotation min cannot exceed max")
    if not 1 <= args.qwen_batch_size <= 64 or args.qwen_batch_wait_ms < 0:
        parser.error("invalid Qwen batch size/wait")
    if args.qwen_wait_timeout <= 0:
        parser.error("--qwen-wait-timeout must be positive")
    return args


def _resolve_similarity_model(args: argparse.Namespace, run_root: Path) -> Path | None:
    """Resolve the DINOv2 model path (inlined from the removed joint pipeline)."""
    skip_validation = bool(getattr(args, "skip_similarity_validation", False))
    selected_modes = sum(
        bool(value)
        for value in (
            args.similarity_model_path,
            args.download_similarity_model,
            skip_validation,
        )
    )
    if selected_modes > 1:
        raise ValueError(
            "--similarity-model-path, --download-similarity-model, and "
            "--skip-similarity-validation are mutually exclusive"
        )
    if skip_validation:
        return None
    if args.download_similarity_model:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("--download-similarity-model requires huggingface_hub") from exc
        destination = ROOT_DIR / "generation_v2_outputs" / "models" / "facebook-dinov2-base"
        destination.mkdir(parents=True, exist_ok=True)
        try:
            revision = snapshot_download("facebook/dinov2-base", local_dir=str(destination), local_dir_use_symlinks=False)
        except TypeError:
            revision = snapshot_download("facebook/dinov2-base", local_dir=str(destination))
        if revision:
            (destination / "v2_revision.txt").write_text(str(revision), encoding="utf-8")
        return destination.resolve()
    if args.similarity_model_path is None:
        raise ValueError(
            "Provide --similarity-model-path, explicitly enable --download-similarity-model, "
            "or explicitly enable --skip-similarity-validation"
        )
    path = args.similarity_model_path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"DINOv2 model path does not exist: {path}")
    return path


def _run_v2_cli(args: argparse.Namespace) -> None:
    """Run the isolated V2 image producer while keeping legacy classes intact."""
    from generation_runtime import PersistentQwenQueue, QwenBatchScheduler, ensure_isolated_root
    from generation_v2 import ImageDatasetV2Producer

    output_path = Path(args.output_dataset).expanduser().resolve()
    image_dir = Path(args.image_dir).expanduser().resolve()
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
    if run_root == formal_root and run_root.exists() and any(run_root.iterdir()) and not args.resume and not args.dry_run:
        raise ValueError(f"V2 output root already contains files; use --resume: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    if args.download_similarity_model:
        model_path = _resolve_similarity_model(
            argparse.Namespace(similarity_model_path=None, download_similarity_model=True), run_root
        )
    elif args.similarity_model_path:
        model_path = Path(args.similarity_model_path).expanduser().resolve()
    else:
        raise ValueError("Provide --similarity-model-path or explicitly enable --download-similarity-model")
    runtime_dir = run_root / "runtime"
    queue = PersistentQwenQueue(runtime_dir / "qwen_jobs.json")
    from confidence_test.inference_extension import ExtendedQwenVLInference
    inference = ExtendedQwenVLInference(model_path=str(Path(args.model_path).expanduser().resolve()))
    stop = threading.Event()
    scheduler_errors: list[BaseException] = []

    def consume() -> None:
        try:
            QwenBatchScheduler(queue, inference, args.qwen_batch_size, args.qwen_batch_wait_ms).run(stop)
        except BaseException as exc:
            queue.fail_unfinished({"type": type(exc).__name__, "message": str(exc)})
            scheduler_errors.append(exc)

    scheduler = threading.Thread(target=consume, name="qwen-v2-image-scheduler", daemon=True)
    scheduler.start()
    try:
        producer = ImageDatasetV2Producer(
            input_path=Path(args.input_dataset), output_path=output_path, image_dir=image_dir,
            state_path=output_path.with_suffix(".state.json"), queue=queue,
            api_config_path=Path(args.api_config_path), old_image_dataset=Path(args.old_image_dataset),
            old_image_root=Path(args.old_image_root), similarity_model_path=model_path,
            similarity_cache=runtime_dir / "dinov2_embeddings.pt", branches=args.branches,
            seed=args.seed if args.seed is not None else 42,
            images_per_difficulty=args.images_per_difficulty,
            conflict_easy_count=args.conflict_easy_count, conflict_hard_count=args.conflict_hard_count,
            target_rotation_mode=args.target_rotation_mode, target_rotation_min=args.target_rotation_min,
            target_rotation_max=args.target_rotation_max,
            distractor_rotation_min=args.distractor_rotation_min,
            distractor_rotation_max=args.distractor_rotation_max, qwen_timeout=args.qwen_wait_timeout,
        )
        producer.run()
    finally:
        stop.set()
        scheduler.join(timeout=120)
        if scheduler.is_alive():
            queue.fail_unfinished({"type": "SchedulerShutdown", "message": "scheduler did not stop"})
            raise RuntimeError("Qwen scheduler did not stop")
    if scheduler_errors:
        raise RuntimeError(f"Qwen scheduler failed: {scheduler_errors[0]}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.recreate or args.legacy or args.dry_run:
            generator = RecreateDatasetGenerator(args) if args.recreate else DatasetGenerator(args)
            generator.run()
        else:
            _run_v2_cli(args)
    except KeyboardInterrupt:
        message = (
            "[WARN] Recreate interrupted; completed attempts and published items are recoverable with "
            "--recreate --resume."
            if args.recreate else
            "[WARN] Interrupted; completed items and locked easy images are recoverable with --resume."
        )
        print(message, file=sys.stderr)
        return 130
    except GenerationStopped as exc:
        print(f"[STOPPED] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"[ERROR] {_redact(str(exc))}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
