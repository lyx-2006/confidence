#!/usr/bin/env python3
"""Build an entropy-calibrated, image-only shape/colour pool with Qwen3.

This entry point deliberately stays separate from both the legacy paired
text/image dataset pipeline and Generation V2.  Layouts and rasterisation are
reused from ``generate_shape_color_dataset.py``; admission uses the exact
Qwen3 image-only gate used by the faithful CMA experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageChops, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "qwen-3-vl" / "model"
DEFAULT_PILOT_ROOT = SCRIPT_DIR / "datasets" / "image_pool_pilot"
DEFAULT_POOL_ROOT = SCRIPT_DIR / "datasets" / "image_pool"
FAITHFUL_DIR = ROOT / "qwen3 review" / "short prompt check" / "faithful check"

LEVELS = ("very_easy", "easy", "medium", "hard")
PILOT_COLORS = ("red", "white", "black")
PILOT_SHAPES = ("circle", "triangle", "star")
BLUR_RADII = (0.0, 2.0, 4.0, 8.0, 12.0, 16.0)
ENTROPY_DEFINITION = "H_norm=-sum(p_i*ln(p_i))/ln(12), p over restricted color classes"


@dataclass(frozen=True)
class SceneProfile:
    name: str
    semantic_shape_count: int
    occlusion_range: tuple[float, float] | None


PROFILES = (
    SceneProfile("single_clear", 1, None),
    SceneProfile("sparse_clear", 4, None),
    SceneProfile("medium_clear", 7, None),
    SceneProfile("medium_occluded", 7, (0.20, 0.30)),
    SceneProfile("dense_clear", 10, None),
    SceneProfile("dense_occluded", 10, (0.35, 0.45)),
)
PROFILE_BY_NAME = {profile.name: profile for profile in PROFILES}


def _load_legacy() -> Any:
    path = SCRIPT_DIR / "generate_shape_color_dataset.py"
    name = "_image_pool_legacy_renderer"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import legacy renderer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LEGACY = _load_legacy()
COLORS = tuple(LEGACY.COLORS)
SHAPES = tuple(LEGACY.SHAPES)


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    LEGACY.atomic_write_json(path, value)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number} is not a JSON object")
            result.append(value)
    return result


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    try:
        shutil.copyfile(source, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_thresholds(raw: str | Sequence[float] | None) -> tuple[float, float, float] | None:
    if raw is None:
        return None
    values = [float(value) for value in raw.split(",")] if isinstance(raw, str) else [float(value) for value in raw]
    if len(values) != 3 or not (0.0 < values[0] < values[1] < values[2] < 1.0):
        raise ValueError("Thresholds must be three strictly increasing values inside (0, 1)")
    return values[0], values[1], values[2]


def entropy_level(entropy: float, thresholds: Sequence[float]) -> str:
    value = float(entropy)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"Invalid normalized entropy: {entropy!r}")
    first, second, third = parse_thresholds(tuple(thresholds)) or ()
    if value < first:
        return "very_easy"
    if value < second:
        return "easy"
    if value < third:
        return "medium"
    return "hard"


def _solve_pool_occluder(
    target: dict[str, Any], color: str, seed: int, ratio_range: tuple[float, float]
) -> dict[str, Any]:
    """Create a deterministic rectangular strip with measured target coverage."""
    target_mask = LEGACY._object_mask(target)
    bounds = target_mask.getbbox()
    if bounds is None:
        raise ValueError("Cannot occlude an empty target mask")
    left, top, right, bottom = bounds
    binary = target_mask.crop(bounds).convert("1")
    width, height = binary.size
    pixels = list(binary.getdata())
    target_area = sum(bool(value) for value in pixels)
    if target_area == 0:
        raise ValueError("Cannot occlude a zero-area target")
    column_counts = [sum(bool(pixels[y * width + x]) for y in range(height)) for x in range(width)]
    row_counts = [sum(bool(pixels[y * width + x]) for x in range(width)) for y in range(height)]
    candidates: list[tuple[float, tuple[float, float, float, float]]] = []
    for counts, orientation in ((column_counts, "vertical"), (row_counts, "horizontal")):
        cumulative = 0
        for index, count in enumerate(counts, start=1):
            cumulative += count
            ratio = cumulative / target_area
            if not ratio_range[0] <= ratio <= ratio_range[1]:
                continue
            if orientation == "vertical":
                boxes = ((left, top, left + index, bottom), (right - index, top, right, bottom))
            else:
                boxes = ((left, top, right, top + index), (left, bottom - index, right, bottom))
            candidates.extend((ratio, tuple(map(float, box))) for box in boxes)
    if not candidates:
        raise ValueError(f"Could not solve occlusion in range {ratio_range}")
    midpoint = sum(ratio_range) / 2.0
    shape = LEGACY._preferred_occluder_shape(str(target["shape"]))
    # Raster bbox rounding and mask placement can move the measured ratio away
    # from the analytic strip estimate, especially for triangles and stars.
    # Verify every promising candidate against the exact trusted mask.
    candidates.sort(key=lambda item: abs(item[0] - midpoint))
    ordered = candidates[:]
    random.Random(seed).shuffle(ordered)
    ordered.sort(key=lambda item: abs(item[0] - midpoint))
    verified: list[dict[str, Any]] = []
    for _estimated, bbox in ordered:
        result = LEGACY._occluder_candidate(shape, color, bbox)
        measured = LEGACY._mask_overlap_ratio(target_mask, LEGACY._object_mask(result))
        if ratio_range[0] <= measured <= ratio_range[1]:
            result["expected_mask_occlusion_ratio"] = measured
            verified.append(result)
            if len(verified) >= 24:
                break
    if not verified:
        raise ValueError(f"No raster-verified occluder in range {ratio_range}")
    verified.sort(key=lambda item: abs(float(item["expected_mask_occlusion_ratio"]) - midpoint))
    near = verified[: min(12, len(verified))]
    return random.Random(LEGACY.derive_seed(seed, "verified-pool-occluder")).choice(near)


def build_pool_layout(
    seed: int, profile_name: str, target_shape: str, target_color: str
) -> dict[str, Any]:
    if profile_name not in PROFILE_BY_NAME:
        raise ValueError(f"Unknown scene profile: {profile_name}")
    if target_shape not in SHAPES or target_color not in COLORS:
        raise ValueError(f"Unsupported target: {target_shape}/{target_color}")
    profile = PROFILE_BY_NAME[profile_name]
    forbidden = set(LEGACY.similar_shapes_for(target_shape)) | {target_shape}
    allowed = [shape for shape in SHAPES if shape not in forbidden]
    if len(allowed) < profile.semantic_shape_count - 1:
        raise ValueError(f"Not enough distinct distractors for {target_shape}/{profile_name}")
    available_colors = [color for color in COLORS if color != target_color]
    objects: list[dict[str, Any]] | None = None
    for restart in range(100):
        rng = random.Random(LEGACY.derive_seed(seed, "image-pool-layout", restart))
        shapes = list(allowed)
        rng.shuffle(shapes)
        proposed: list[dict[str, Any]] = []
        try:
            target = LEGACY._place_object(
                rng, proposed, target_shape, target_color, "target", LEGACY.EASY_TARGET_SIZE_RANGE
            )
            proposed.append(target)
            for shape in shapes[: profile.semantic_shape_count - 1]:
                proposed.append(LEGACY._place_object(
                    rng,
                    proposed,
                    shape,
                    rng.choice(available_colors),
                    "distractor",
                    LEGACY.EASY_DISTRACTOR_SIZE_RANGE,
                ))
        except ValueError:
            continue
        if not LEGACY.detect_layout_patterns(proposed):
            objects = proposed
            break
    if objects is None:
        raise ValueError(f"Could not pack image-pool layout for {profile_name}")
    target = objects[0]
    occluder_geometry: list[dict[str, Any]] = []
    expected_occlusion = 0.0
    if profile.occlusion_range is not None:
        occluder = _solve_pool_occluder(
            target,
            random.Random(LEGACY.derive_seed(seed, "occluder-color")).choice(available_colors),
            LEGACY.derive_seed(seed, "occluder-geometry"),
            profile.occlusion_range,
        )
        objects.append(occluder)
        expected_occlusion = float(occluder["expected_mask_occlusion_ratio"])
        occluder_geometry = [
            {key: occluder[key] for key in ("shape", "center", "bbox", "rotation", "size")}
        ]
    return {
        "schema_version": "image_pool.layout.v1",
        "canvas": [LEGACY.CANVAS_SIZE, LEGACY.CANVAS_SIZE],
        "branch": "image_pool",
        "difficulty": profile_name,
        "profile": profile_name,
        "semantic_shape_count": profile.semantic_shape_count,
        "target_shape": target_shape,
        "target_color": target_color,
        "case_seed": seed,
        "target_geometry": {key: target[key] for key in ("center", "bbox", "rotation", "size")},
        "occluder_geometry": occluder_geometry,
        "expected_occlusion_ratio": expected_occlusion,
        "objects": objects,
    }


def validate_pool_rendered(rendered: Path, expected: dict[str, Any]) -> dict[str, Any]:
    required = ("image.png", "layout.json", "target_mask.png", "occluder_mask.png")
    missing = [name for name in required if not (rendered / name).is_file()]
    if missing:
        return {"valid": False, "issues": [f"missing:{name}" for name in missing]}
    layout = json.loads((rendered / "layout.json").read_text(encoding="utf-8"))
    issues: list[str] = []
    if layout != expected:
        issues.append("layout_mismatch")
    objects = layout.get("objects", [])
    target = [obj for obj in objects if obj.get("role") == "target"]
    semantic = [obj for obj in objects if obj.get("role") in {"target", "distractor"}]
    distractors = [obj for obj in objects if obj.get("role") == "distractor"]
    if len(target) != 1:
        issues.append("target_count")
    if len(semantic) != int(expected["semantic_shape_count"]):
        issues.append("semantic_shape_count")
    if len({obj.get("shape") for obj in semantic}) != len(semantic):
        issues.append("semantic_shapes_not_distinct")
    if any(obj.get("color") == expected["target_color"] for obj in distractors):
        issues.append("target_color_reused")
    if any(obj.get("shape") in LEGACY.similar_shapes_for(expected["target_shape"]) for obj in distractors):
        issues.append("similar_distractor")
    for index, first in enumerate(semantic):
        bbox = list(map(float, first.get("bbox", [])))
        if len(bbox) != 4 or bbox[0] < 0 or bbox[1] < 0 or bbox[2] > LEGACY.CANVAS_SIZE or bbox[3] > LEGACY.CANVAS_SIZE:
            issues.append("bbox_bounds")
            continue
        for second in semantic[index + 1 :]:
            if LEGACY._bbox_intersection(first["bbox"], second["bbox"]) > 0:
                issues.append("semantic_bbox_overlap")
    with Image.open(rendered / "image.png") as image:
        if image.mode != "RGB" or image.size != (LEGACY.CANVAS_SIZE, LEGACY.CANVAS_SIZE):
            issues.append("image_mode_or_size")
    with Image.open(rendered / "target_mask.png") as target_mask, Image.open(
        rendered / "occluder_mask.png"
    ) as occluder_mask:
        target_binary = target_mask.convert("1")
        occluder_binary = occluder_mask.convert("1")
        target_area = target_binary.histogram()[255]
        intersection = ImageChops.logical_and(target_binary, occluder_binary).histogram()[255]
        ratio = 0.0 if target_area == 0 else intersection / target_area
    profile = PROFILE_BY_NAME[str(expected["profile"])]
    if target_area == 0:
        issues.append("empty_target_mask")
    elif profile.occlusion_range is None and ratio != 0.0:
        issues.append("unexpected_occlusion")
    elif profile.occlusion_range is not None and not (
        profile.occlusion_range[0] <= ratio <= profile.occlusion_range[1]
    ):
        issues.append("occlusion_out_of_range")
    return {
        "valid": not issues,
        "issues": sorted(set(issues)),
        "semantic_shape_count": len(semantic),
        "rendered_object_count": len(objects),
        "occlusion_ratio": ratio,
    }


def render_base_scene(layout: dict[str, Any], destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    style_rng = random.Random(LEGACY.derive_seed(int(layout["case_seed"]), "render-style"))
    style = {
        "background_rgb": list(style_rng.choice(LEGACY.ALLOWED_BACKGROUNDS)),
        "outline_rgb": list(LEGACY.DEFAULT_RENDER_STYLE["outline_rgb"]),
        "outline_width": int(LEGACY.DEFAULT_RENDER_STYLE["outline_width"]),
    }
    with tempfile.TemporaryDirectory(prefix="image-pool-render-") as temporary:
        rendered = LEGACY.render_scene_locally(layout, Path(temporary), style)
        geometry = validate_pool_rendered(rendered, layout)
        if not geometry["valid"]:
            raise ValueError(f"Rendered scene failed validation: {geometry['issues']}")
        mapping = {
            "image.png": "sharp.png",
            "layout.json": "layout.json",
            "target_mask.png": "target_mask.png",
            "occluder_mask.png": "occluder_mask.png",
        }
        for source, target in mapping.items():
            atomic_copy(rendered / source, destination / target)
    return geometry


def render_blur_variant(sharp: Path, destination: Path, radius: float) -> None:
    if radius < 0:
        raise ValueError("Blur radius must be non-negative")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if radius == 0:
        atomic_copy(sharp, destination)
        return
    with Image.open(sharp) as source:
        image = source.convert("RGB").filter(ImageFilter.GaussianBlur(radius=float(radius)))
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    try:
        image.save(temporary, format="PNG")
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def model_fingerprint(model_path: Path) -> dict[str, Any]:
    names = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.json",
        "preprocessor_config.json",
        "model.safetensors.index.json",
    )
    missing = [name for name in names if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Qwen3 model directory is incomplete: {missing}")
    values = {name: sha256_file(model_path / name) for name in names}
    return {"path": str(model_path.resolve()), "files": values, "fingerprint": canonical_hash(values)}


def load_qwen3_runner(model_path: Path) -> Any:
    for path in (ROOT, FAITHFUL_DIR.parent.parent, FAITHFUL_DIR.parent, FAITHFUL_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    existing = sys.modules.get("runtime")
    if existing is not None and Path(getattr(existing, "__file__", "")).resolve() != (FAITHFUL_DIR / "runtime.py").resolve():
        raise RuntimeError(f"Top-level runtime module collision: {getattr(existing, '__file__', None)}")
    from runtime import FaithfulQwenRunner

    return FaithfulQwenRunner(model_path)


def validation_from_qwen(result: dict[str, Any], expected: str) -> dict[str, Any]:
    logits = result.get("answer_class_logits")
    probabilities = result.get("answer_class_probabilities")
    entropy = result.get("normalized_entropy")
    target_probability = result.get("target_probability")
    finite_maps = all(
        isinstance(mapping, dict)
        and set(mapping) == set(COLORS)
        and all(math.isfinite(float(value)) for value in mapping.values())
        for mapping in (logits, probabilities)
    )
    metrics_valid = (
        finite_maps
        and entropy is not None
        and target_probability is not None
        and math.isfinite(float(entropy))
        and math.isfinite(float(target_probability))
        and 0.0 <= float(entropy) <= 1.0
    )
    generated_correct = bool(result.get("parse_success")) and normalize(result.get("normalized_answer")) == expected
    restricted_correct = normalize(result.get("restricted_top1")) == expected
    passed = bool(
        result.get("gate_passed") is True
        and generated_correct
        and restricted_correct
        and result.get("answer_metric_status") == "completed"
        and metrics_valid
    )
    reasons: list[str] = []
    if not result.get("parse_success"):
        reasons.append("answer_parse_failed")
    if not generated_correct:
        reasons.append("generated_answer_incorrect")
    if not restricted_correct:
        reasons.append("restricted_top1_incorrect")
    if result.get("answer_metric_status") != "completed" or not metrics_valid:
        reasons.append("invalid_answer_metrics")
    return {
        "prediction": normalize(result.get("normalized_answer")),
        "restricted_top1": normalize(result.get("restricted_top1")),
        "correct": passed,
        "gate_passed": passed,
        "failure_reasons": sorted(set(reasons)),
        "target_probability": float(target_probability) if target_probability is not None else None,
        "target_margin": result.get("target_margin"),
        "normalized_entropy": float(entropy) if entropy is not None else None,
        "answer_class_logits": logits if isinstance(logits, dict) else {},
        "answer_class_probabilities": probabilities if isinstance(probabilities, dict) else {},
        "actual_output": result.get("actual_output"),
        "prompt_hash": result.get("prompt_hash"),
        "rendered_hash": result.get("rendered_hash"),
        "model_fingerprint": result.get("model_fingerprint"),
    }


class ImagePoolPipeline:
    def __init__(
        self,
        *,
        mode: str,
        output_root: Path,
        model_path: Path,
        seed: int,
        thresholds: tuple[float, float, float] | None,
        quota_per_level: int,
        resume: bool,
        runner: Any | None = None,
    ) -> None:
        self.command = mode
        # ``build`` and ``import-legacy`` are two producers for one formal
        # pool and therefore must share a resume/config fingerprint.
        self.mode = "pilot" if mode == "pilot" else "formal"
        self.root = output_root.resolve()
        self.model_path = model_path.resolve()
        self.seed = int(seed)
        self.thresholds = thresholds
        self.quota = int(quota_per_level)
        if self.quota < 1:
            raise ValueError("quota_per_level must be positive")
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.results_path = self.root / "candidate_results.jsonl"
        self.rejected_path = self.root / "rejected.jsonl"
        self._runner_value = runner
        self._records_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.config = self._config()
        self.fingerprint = canonical_hash(self.config)
        if self.state_path.is_file():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("config_fingerprint") != self.fingerprint:
                raise RuntimeError("Existing image-pool state has a different configuration")
            if not resume:
                raise RuntimeError("Output already has state; pass --resume or choose another output root")
            self.state = state
        else:
            self.state = {
                "schema_version": "image_pool.state.v1",
                "config_fingerprint": self.fingerprint,
                "completed_count": 0,
                "accepted_sha256": [],
                "accepted_counts": {},
            }
            self._persist_state()
        published_hashes: set[str] = set()
        recovered_completed: dict[str, dict[str, Any]] = dict(self.state.pop("completed", {}))
        for color in COLORS:
            for shape in SHAPES:
                for record in self._shape_records(color, shape):
                    value = record.get("validation", {}).get("image_sha256")
                    if isinstance(value, str) and value:
                        published_hashes.add(value)
                    candidate_id = record.get("candidate_id")
                    if isinstance(candidate_id, str) and candidate_id:
                        recovered_completed[candidate_id] = {
                            "accepted": True,
                            "difficulty_level": record.get("difficulty", {}).get("level"),
                            "image_sha256": value,
                            "rejection_reasons": [],
                        }
        for record in load_jsonl(self.results_path):
            candidate_id = record.get("candidate_id")
            if isinstance(candidate_id, str) and candidate_id:
                recovered_completed[candidate_id] = {
                    "accepted": bool(record.get("accepted")),
                    "difficulty_level": record.get("difficulty_level"),
                    "image_sha256": record.get("image_sha256"),
                    "rejection_reasons": list(record.get("rejection_reasons", [])),
                }
        self.completed = recovered_completed
        self.state["completed_count"] = len(self.completed)
        self.accepted_hashes = set(map(str, self.state.get("accepted_sha256", []))) | published_hashes
        self.state["accepted_sha256"] = sorted(self.accepted_hashes)
        self._persist_state()
        self._write_meta()

    def _config(self) -> dict[str, Any]:
        prompt_path = FAITHFUL_DIR / "prompts.py"
        runtime_path = FAITHFUL_DIR / "runtime.py"
        return {
            "schema_version": "image_pool.config.v1",
            "mode": self.mode,
            "output_root": str(self.root),
            "seed": self.seed,
            "model": model_fingerprint(self.model_path),
            "faithful_runtime_sha256": sha256_file(runtime_path),
            "faithful_prompts_sha256": sha256_file(prompt_path),
            "colors": list(COLORS),
            "shapes": list(SHAPES),
            "profiles": [asdict(profile) for profile in PROFILES],
            "blur_radii": list(BLUR_RADII),
            "entropy_definition": ENTROPY_DEFINITION,
            "thresholds": list(self.thresholds) if self.thresholds else None,
            "quota_per_level": self.quota,
        }

    def _write_meta(self) -> None:
        atomic_json(
            self.root / "dataset_meta.json",
            {
                "schema_version": "image_pool.v1",
                "config_fingerprint": self.fingerprint,
                "mode": self.mode,
                "model": self.config["model"],
                "colors": list(COLORS),
                "shapes": list(SHAPES),
                "difficulty_levels": list(LEVELS),
                "difficulty_thresholds": list(self.thresholds) if self.thresholds else None,
                "entropy_definition": ENTROPY_DEFINITION,
                "blur_radii": list(BLUR_RADII),
                "profiles": [asdict(profile) for profile in PROFILES],
                "quota_per_level": self.quota,
                "split_rule": "Keep every base_scene_id in exactly one dataset split.",
            },
        )

    def _persist_state(self) -> None:
        self.state["accepted_sha256"] = sorted(set(map(str, self.state.get("accepted_sha256", []))))
        atomic_json(self.state_path, self.state)

    def _runner(self) -> Any:
        if self._runner_value is None:
            self._runner_value = load_qwen3_runner(self.model_path)
        return self._runner_value

    def _question(self, shape: str) -> str:
        return LEGACY.QUESTION_TEMPLATE.format(shape=shape)

    def _shape_json_path(self, color: str, shape: str) -> Path:
        return self.root / color / f"{shape}.json"

    def _shape_records(self, color: str, shape: str) -> list[dict[str, Any]]:
        key = (color, shape)
        if key in self._records_cache:
            return self._records_cache[key]
        path = self._shape_json_path(color, shape)
        if not path.is_file():
            self._records_cache[key] = []
            return self._records_cache[key]
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError(f"Shape JSON root must be an array: {path}")
        self._records_cache[key] = [record for record in value if isinstance(record, dict)]
        return self._records_cache[key]

    def _next_filename(self, color: str, shape: str) -> str:
        pattern = re.compile(rf"^{re.escape(shape)}_{re.escape(color)}_(\d{{6}})\.png$")
        numbers = [
            int(match.group(1))
            for record in self._shape_records(color, shape)
            if (match := pattern.match(str(record.get("image", ""))))
        ]
        return f"{shape}_{color}_{max(numbers, default=0) + 1:06d}.png"

    def _accepted_count(self, color: str, shape: str, level: str) -> int:
        return sum(
            record.get("difficulty", {}).get("level") == level
            for record in self._shape_records(color, shape)
        )

    def _scene_has_level(self, color: str, shape: str, scene_id: str, level: str) -> bool:
        return any(
            record.get("generation", {}).get("base_scene_id") == scene_id
            and record.get("difficulty", {}).get("level") == level
            for record in self._shape_records(color, shape)
        )

    def _publish(
        self,
        *,
        candidate_id: str,
        candidate_image: Path,
        color: str,
        shape: str,
        scene_id: str,
        seed: int,
        profile: SceneProfile,
        radius: float | None,
        geometry: dict[str, Any],
        validation: dict[str, Any],
        level: str,
        source: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        image_hash = sha256_file(candidate_image)
        filename = self._next_filename(color, shape)
        destination = self.root / color / filename
        atomic_copy(candidate_image, destination)
        record: dict[str, Any] = {
            "candidate_id": candidate_id,
            "image": filename,
            "Answer": color,
            "difficulty": {"level": level, "entropy": validation["normalized_entropy"]},
            "generation": {
                "base_scene_id": scene_id,
                "seed": seed,
                "profile": profile.name,
                "blur_radius": radius,
                "semantic_shape_count": geometry.get("semantic_shape_count"),
                "occlusion_ratio": geometry.get("occlusion_ratio"),
            },
            "validation": {
                **validation,
                "image_sha256": image_hash,
            },
        }
        if source is not None:
            record["source"] = source
        path = self._shape_json_path(color, shape)
        records = self._shape_records(color, shape)
        records.append(record)
        atomic_json(path, records)
        self.accepted_hashes.add(image_hash)
        self.state["accepted_sha256"] = sorted(self.accepted_hashes)
        key = f"{color}/{shape}/{level}"
        self.state.setdefault("accepted_counts", {})[key] = self._accepted_count(color, shape, level)
        return record

    def _candidate_paths(self, color: str, shape: str, scene_id: str, radius: float) -> tuple[Path, Path]:
        scene_dir = self.root / "_scenes" / color / shape / scene_id
        radius_name = str(radius).replace(".", "p")
        candidate = self.root / "_candidates" / color / shape / f"{scene_id}__blur_{radius_name}.png"
        return scene_dir, candidate

    def _score_candidate(
        self,
        *,
        candidate_id: str,
        image: Path,
        color: str,
        shape: str,
        scene_id: str,
        seed: int,
        profile: SceneProfile,
        radius: float | None,
        geometry: dict[str, Any],
        source: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if candidate_id in self.completed:
            return dict(self.completed[candidate_id])
        raw = self._runner().image_only(self._question(shape), str(image), color)
        validation = validation_from_qwen(raw, color)
        image_hash = sha256_file(image)
        level = "pending" if self.thresholds is None else entropy_level(
            float(validation["normalized_entropy"]), self.thresholds
        ) if validation.get("normalized_entropy") is not None else "invalid"
        rejection: list[str] = list(validation["failure_reasons"])
        if image_hash in self.accepted_hashes:
            rejection.append("duplicate_image_sha256")
        if self.thresholds is not None and level in LEVELS:
            if self._accepted_count(color, shape, level) >= self.quota:
                rejection.append("quota_filled")
            if self._scene_has_level(color, shape, scene_id, level):
                rejection.append("scene_level_duplicate")
        accepted = validation["gate_passed"] and not rejection
        record = {
            "candidate_id": candidate_id,
            "status": "completed",
            "accepted": accepted,
            "rejection_reasons": sorted(set(rejection)),
            "image_path": str(image.resolve()),
            "image_sha256": image_hash,
            "color": color,
            "shape": shape,
            "difficulty_level": level,
            "generation": {
                "base_scene_id": scene_id,
                "seed": seed,
                "profile": profile.name,
                "blur_radius": radius,
                "semantic_shape_count": geometry.get("semantic_shape_count"),
                "occlusion_ratio": geometry.get("occlusion_ratio"),
            },
            "geometry": geometry,
            "validation": validation,
            "source": source,
        }
        if accepted:
            published = self._publish(
                candidate_id=candidate_id,
                candidate_image=image,
                color=color,
                shape=shape,
                scene_id=scene_id,
                seed=seed,
                profile=profile,
                radius=radius,
                geometry=geometry,
                validation=validation,
                level=level,
                source=source,
            )
            record["published_image"] = str((self.root / color / published["image"]).resolve())
        else:
            append_jsonl(self.rejected_path, {
                "candidate_id": candidate_id,
                "color": color,
                "shape": shape,
                "generation": record["generation"],
                "prediction": validation.get("prediction"),
                "restricted_top1": validation.get("restricted_top1"),
                "normalized_entropy": validation.get("normalized_entropy"),
                "rejection_reasons": record["rejection_reasons"],
                "source": source,
            })
        append_jsonl(self.results_path, record)
        self.completed[candidate_id] = {
            "accepted": accepted,
            "difficulty_level": level,
            "image_sha256": image_hash,
            "rejection_reasons": record["rejection_reasons"],
        }
        self.state["completed_count"] = len(self.completed)
        # candidate_results.jsonl is fsync'd and is the durable completion
        # ledger.  Rewrite state immediately only when publication changed;
        # rejected candidates are recovered from the ledger on resume.
        if accepted:
            self._persist_state()
        return record

    def _run_scene(self, color: str, shape: str, profile: SceneProfile, scene_index: int) -> None:
        seed = LEGACY.derive_seed(self.seed, self.mode, color, shape, profile.name, scene_index)
        scene_id = f"{shape}_{color}_scene_{scene_index:06d}_{profile.name}"
        scene_dir, _unused = self._candidate_paths(color, shape, scene_id, 0.0)
        layout_path = scene_dir / "layout.json"
        stored_artifacts = (
            layout_path,
            scene_dir / "sharp.png",
            scene_dir / "image.png",
            scene_dir / "target_mask.png",
            scene_dir / "occluder_mask.png",
        )
        if all(path.is_file() for path in stored_artifacts):
            layout = json.loads(layout_path.read_text(encoding="utf-8"))
            geometry = validate_pool_rendered(scene_dir, layout)
        else:
            layout = build_pool_layout(seed, profile.name, shape, color)
            geometry = render_base_scene(layout, scene_dir)
            # ``validate_pool_rendered`` expects image.png; public scene storage
            # calls the sharp artifact sharp.png.
            if not (scene_dir / "image.png").exists():
                atomic_copy(scene_dir / "sharp.png", scene_dir / "image.png")
        if not geometry.get("valid", True):
            raise RuntimeError(f"Stored scene is invalid: {scene_id}: {geometry.get('issues')}")
        sharp = scene_dir / "sharp.png"
        try:
            for radius in BLUR_RADII:
                if self.thresholds is not None and all(
                    self._accepted_count(color, shape, level) >= self.quota for level in LEVELS
                ):
                    break
                _scene_dir, candidate = self._candidate_paths(color, shape, scene_id, radius)
                if not candidate.is_file():
                    render_blur_variant(sharp, candidate, radius)
                candidate_id = canonical_hash(
                    {"kind": "generated", "scene": scene_id, "radius": radius, "config": self.fingerprint}
                )
                self._score_candidate(
                    candidate_id=candidate_id,
                    image=candidate,
                    color=color,
                    shape=shape,
                    scene_id=scene_id,
                    seed=seed,
                    profile=profile,
                    radius=radius,
                    geometry=geometry,
                )
        finally:
            if self.mode == "formal":
                for radius in BLUR_RADII:
                    _scene_dir, candidate = self._candidate_paths(color, shape, scene_id, radius)
                    candidate.unlink(missing_ok=True)
                shutil.rmtree(scene_dir, ignore_errors=True)

    def run_pilot(self) -> None:
        for color in PILOT_COLORS:
            for shape in PILOT_SHAPES:
                for index, profile in enumerate(PROFILES, start=1):
                    self._run_scene(color, shape, profile, index)
        self._persist_state()

    def run_build(self, max_scenes_per_pair: int) -> None:
        if self.thresholds is None:
            raise ValueError("Formal build requires --thresholds")
        for color in COLORS:
            for shape in SHAPES:
                for scene_index in range(1, max_scenes_per_pair + 1):
                    if all(self._accepted_count(color, shape, level) >= self.quota for level in LEVELS):
                        break
                    profile = PROFILES[(scene_index - 1) % len(PROFILES)]
                    self._run_scene(color, shape, profile, scene_index)
        missing = {
            f"{color}/{shape}/{level}": self.quota - self._accepted_count(color, shape, level)
            for color in COLORS
            for shape in SHAPES
            for level in LEVELS
            if self._accepted_count(color, shape, level) < self.quota
        }
        atomic_json(self.root / "build_report.json", {"complete": not missing, "missing": missing})
        self._persist_state()
        if missing:
            raise RuntimeError(f"Formal pool did not fill {len(missing)} quota cells; see build_report.json")

    def import_legacy(self, sources: Sequence[Path]) -> None:
        if self.thresholds is None:
            raise ValueError("Legacy import requires --thresholds")
        candidates = collect_legacy_candidates(sources)
        for candidate in candidates:
            color = candidate["color"]
            shape = candidate["shape"]
            source_path = Path(candidate["image_path"])
            source_hash = sha256_file(source_path)
            scene_id = f"legacy_{shape}_{color}_{source_hash[:16]}"
            candidate_id = canonical_hash(
                {"kind": "legacy", "sha256": source_hash, "color": color, "shape": shape, "config": self.fingerprint}
            )
            layout_path = source_path.with_suffix(".layout.json")
            semantic_count = None
            occlusion = None
            if layout_path.is_file():
                layout = json.loads(layout_path.read_text(encoding="utf-8"))
                semantic_count = sum(obj.get("role") in {"target", "distractor"} for obj in layout.get("objects", []))
                occlusion = layout.get("expected_occlusion_ratio")
            profile = SceneProfile("legacy", int(semantic_count or 0), None)
            geometry = {
                "valid": True,
                "issues": [],
                "semantic_shape_count": semantic_count,
                "rendered_object_count": None,
                "occlusion_ratio": occlusion,
            }
            self._score_candidate(
                candidate_id=candidate_id,
                image=source_path,
                color=color,
                shape=shape,
                scene_id=scene_id,
                seed=0,
                profile=profile,
                radius=None,
                geometry=geometry,
                source=candidate["source"],
            )
        self._persist_state()


def _question_shape(question: Any) -> str | None:
    if isinstance(question, dict):
        question = question.get("text")
    match = re.search(r"color\s+of\s+(?:the\s+)?(.+?)\?", str(question or ""), re.IGNORECASE)
    return normalize(match.group(1)) if match else None


def _resolve_reference(dataset_path: Path, raw: str) -> Path:
    path = Path(raw)
    candidates = [
        path if path.is_absolute() else dataset_path.parent / path,
        dataset_path.parent / "generated_shape_color_images" / path.name,
        dataset_path.parent / "images" / path.name,
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(f"Cannot resolve image reference {raw!r} from {dataset_path}")


def collect_legacy_candidates(sources: Sequence[Path]) -> list[dict[str, Any]]:
    by_hash: dict[str, dict[str, Any]] = {}
    semantic_by_hash: dict[str, tuple[str, str]] = {}
    for source_path in sources:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            items = [item for group in payload if isinstance(group, dict) for item in group.get("items", []) if isinstance(item, dict)]
        elif isinstance(payload, dict):
            items = [item for item in payload.get("items", []) if isinstance(item, dict)]
        else:
            raise ValueError(f"Unsupported legacy dataset root: {source_path}")
        for item in items:
            shape = _question_shape(item.get("question"))
            if shape not in SHAPES:
                continue
            entries: list[tuple[str, str, str]] = []
            clue = item.get("image_clue")
            if isinstance(clue, dict):
                for branch in ("consistent", "conflict"):
                    branch_value = clue.get(branch)
                    answer = normalize(item.get("answer")) if branch == "consistent" else normalize(
                        item.get("conflict_ans", item.get("conflict_answer"))
                    )
                    if answer not in COLORS or not isinstance(branch_value, dict):
                        continue
                    for old_difficulty in ("easy", "hard"):
                        raw = branch_value.get(old_difficulty)
                        if isinstance(raw, str):
                            entries.append((answer, f"{branch}_{old_difficulty}", raw))
            groups = item.get("groups")
            if isinstance(groups, dict):
                for group_name, group in groups.items():
                    if not isinstance(group, dict) or not isinstance(group.get("image"), str):
                        continue
                    answer = normalize(group.get("ground_truth_answer", group.get("answer")))
                    if answer in COLORS and (str(group_name).startswith("consistent_") or str(group_name).startswith("conflict_")):
                        entries.append((answer, str(group_name), str(group["image"])))
            for answer, source_group, raw in entries:
                image_path = _resolve_reference(source_path, raw)
                image_hash = sha256_file(image_path)
                semantic = (shape, answer)
                previous = semantic_by_hash.get(image_hash)
                if previous is not None and previous != semantic:
                    raise ValueError(f"Legacy SHA label conflict {image_hash}: {previous} vs {semantic}")
                semantic_by_hash[image_hash] = semantic
                by_hash.setdefault(
                    image_hash,
                    {
                        "shape": shape,
                        "color": answer,
                        "image_path": str(image_path),
                        "source": {
                            "dataset": str(source_path.resolve()),
                            "item_id": str(item.get("id", "")),
                            "group": source_group,
                            "original_image": raw,
                            "original_sha256": image_hash,
                        },
                    },
                )
    return [by_hash[key] for key in sorted(by_hash)]


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(map(float, values))
    if not ordered:
        raise ValueError("Cannot calculate quantile of an empty sequence")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _rank(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank = (index + end - 1) / 2.0
        for original, _value in indexed[index:end]:
            ranks[original] = rank
        index = end
    return ranks


def _correlation(first: Sequence[float], second: Sequence[float]) -> float | None:
    if len(first) != len(second) or len(first) < 2:
        return None
    left, right = _rank(first), _rank(second)
    left_mean, right_mean = statistics.mean(left), statistics.mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left) * sum((b - right_mean) ** 2 for b in right)
    )
    return numerator / denominator if denominator else None


def _rounded_thresholds(values: Sequence[float]) -> tuple[float, float, float] | None:
    raw = [_quantile(values, fraction) for fraction in (0.25, 0.50, 0.75)]
    rounded = [min(0.95, max(0.05, round(value / 0.05) * 0.05)) for value in raw]
    for index in range(1, 3):
        if rounded[index] <= rounded[index - 1]:
            rounded[index] = min(0.95, rounded[index - 1] + 0.05)
    if not rounded[0] < rounded[1] < rounded[2]:
        return (raw[0], raw[1], raw[2]) if raw[0] < raw[1] < raw[2] else None
    return rounded[0], rounded[1], rounded[2]


def threshold_diagnostics(rows: Sequence[dict[str, Any]], thresholds: Sequence[float]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {level: [] for level in LEVELS}
    for row in rows:
        entropy = row.get("validation", {}).get("normalized_entropy")
        if entropy is None:
            continue
        groups[entropy_level(float(entropy), thresholds)].append(row)
    values: dict[str, Any] = {}
    medians: list[float | None] = []
    for level in LEVELS:
        subset = groups[level]
        radii = [float(row["generation"]["blur_radius"]) for row in subset if row["generation"].get("blur_radius") is not None]
        median = statistics.median(radii) if radii else None
        medians.append(median)
        values[level] = {
            "count": len(subset),
            "combination_count": len({(row["color"], row["shape"]) for row in subset}),
            "median_blur_radius": median,
        }
    known_medians = [value for value in medians if value is not None]
    monotonic = all(left <= right for left, right in zip(known_medians, known_medians[1:]))
    return {
        "thresholds": list(thresholds),
        "levels": values,
        "minimum_count_passed": all(values[level]["count"] >= 20 for level in LEVELS),
        "minimum_combination_coverage_passed": all(values[level]["combination_count"] >= 6 for level in LEVELS),
        "median_blur_monotonic_passed": monotonic,
    }


def create_contact_sheet(rows: Sequence[dict[str, Any]], thresholds: Sequence[float], output: Path) -> None:
    boundaries = list(thresholds)
    selected: list[dict[str, Any]] = []
    remaining = list(rows)
    for boundary in boundaries:
        nearest = sorted(
            remaining,
            key=lambda row: abs(float(row["validation"]["normalized_entropy"]) - boundary),
        )[:12]
        selected.extend(nearest)
        used = {row["candidate_id"] for row in nearest}
        remaining = [row for row in remaining if row["candidate_id"] not in used]
    cell_w, cell_h = 220, 250
    sheet = Image.new("RGB", (6 * cell_w, 6 * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(selected[:36]):
        x, y = (index % 6) * cell_w, (index // 6) * cell_h
        path = Path(row["image_path"])
        with Image.open(path) as raw:
            thumb = raw.convert("RGB")
            thumb.thumbnail((cell_w - 12, cell_h - 42), Image.Resampling.LANCZOS)
        sheet.paste(thumb, (x + (cell_w - thumb.width) // 2, y + 4))
        text = (
            f"{row['shape']}/{row['color']} "
            f"H={float(row['validation']['normalized_entropy']):.3f} "
            f"b={row['generation'].get('blur_radius')}"
        )
        draw.text((x + 4, y + cell_h - 32), text, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG")


def analyze_pilot(output_root: Path) -> dict[str, Any]:
    results_path = output_root / "candidate_results.jsonl"
    indexed = {
        str(row["candidate_id"]): row
        for row in load_jsonl(results_path)
        if isinstance(row.get("candidate_id"), str)
    }
    rows = list(indexed.values())
    if len(rows) != 324:
        raise ValueError(f"Pilot requires exactly 324 completed candidates, found {len(rows)}")
    passing = [
        row for row in rows
        if row.get("accepted") is True
        and row.get("validation", {}).get("gate_passed") is True
        and row.get("validation", {}).get("normalized_entropy") is not None
    ]
    if len(passing) < 4:
        raise ValueError("Pilot has too few passing candidates to propose four entropy levels")
    entropies = [float(row["validation"]["normalized_entropy"]) for row in passing]
    quantiles = {f"q{int(fraction * 100)}": _quantile(entropies, fraction) for fraction in (0.25, 0.50, 0.75)}
    proposed = _rounded_thresholds(entropies)
    by_profile: dict[str, dict[str, Any]] = {}
    for profile in PROFILE_BY_NAME:
        subset = [row for row in rows if row["generation"]["profile"] == profile]
        valid = [row for row in subset if row.get("accepted") is True]
        by_profile[profile] = {
            "attempted": len(subset),
            "passed": len(valid),
            "accuracy": len(valid) / len(subset) if subset else None,
            "entropy_mean": statistics.mean(
                float(row["validation"]["normalized_entropy"])
                for row in subset
                if row["validation"].get("normalized_entropy") is not None
            ) if any(row["validation"].get("normalized_entropy") is not None for row in subset) else None,
        }
    by_blur: dict[str, dict[str, Any]] = {}
    for radius in BLUR_RADII:
        subset = [row for row in rows if row["generation"].get("blur_radius") == radius]
        valid = [row for row in subset if row.get("accepted") is True]
        by_blur[str(radius)] = {
            "attempted": len(subset),
            "passed": len(valid),
            "accuracy": len(valid) / len(subset) if subset else None,
            "entropy_mean": statistics.mean(
                float(row["validation"]["normalized_entropy"])
                for row in subset
                if row["validation"].get("normalized_entropy") is not None
            ) if any(row["validation"].get("normalized_entropy") is not None for row in subset) else None,
        }
    scene_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["validation"].get("normalized_entropy") is not None:
            scene_groups[row["generation"]["base_scene_id"]].append(row)
    correlations = []
    for group in scene_groups.values():
        ordered = sorted(group, key=lambda row: float(row["generation"]["blur_radius"]))
        correlation = _correlation(
            [float(row["generation"]["blur_radius"]) for row in ordered],
            [float(row["validation"]["normalized_entropy"]) for row in ordered],
        )
        if correlation is not None:
            correlations.append(correlation)
    diagnostics = threshold_diagnostics(passing, proposed) if proposed is not None else {
        "thresholds": None,
        "levels": {},
        "minimum_count_passed": False,
        "minimum_combination_coverage_passed": False,
        "median_blur_monotonic_passed": False,
        "failure_reason": "passing_entropy_distribution_has_fewer_than_four_separable_regions",
    }
    report = {
        "schema_version": "image_pool.pilot_report.v1",
        "candidate_count": len(rows),
        "passing_count": len(passing),
        "accuracy": len(passing) / len(rows),
        "entropy": {
            "minimum": min(entropies),
            "maximum": max(entropies),
            "mean": statistics.mean(entropies),
            "quantiles": quantiles,
        },
        "by_profile": by_profile,
        "by_blur": by_blur,
        "blur_entropy_spearman": {
            "scene_count": len(correlations),
            "mean": statistics.mean(correlations) if correlations else None,
            "median": statistics.median(correlations) if correlations else None,
        },
        "high_entropy_correct_count": sum(value >= quantiles["q75"] for value in entropies),
        "proposed_global_thresholds": list(proposed) if proposed is not None else None,
        "threshold_diagnostics": diagnostics,
        "automatic_freeze_eligible": bool(
            diagnostics["minimum_count_passed"]
            and diagnostics["minimum_combination_coverage_passed"]
            and diagnostics["median_blur_monotonic_passed"]
        ),
        "manual_contact_sheet_review_required": True,
        "thresholds_applied": False,
    }
    atomic_json(output_root / "pilot_report.json", report)
    contact_boundaries = proposed or (quantiles["q25"], quantiles["q50"], quantiles["q75"])
    create_contact_sheet(passing, contact_boundaries, output_root / "pilot_contact_sheet.png")
    return report


def _common_parser(parser: argparse.ArgumentParser, default_root: Path) -> None:
    parser.add_argument("--output-root", type=Path, default=default_root)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Qwen3 entropy-calibrated image pool")
    subparsers = parser.add_subparsers(dest="command", required=True)
    pilot = subparsers.add_parser("pilot", help="Generate and score the fixed 324-image pilot")
    _common_parser(pilot, DEFAULT_PILOT_ROOT)
    analyze = subparsers.add_parser("analyze", help="Analyze a completed pilot without applying thresholds")
    analyze.add_argument("--output-root", type=Path, default=DEFAULT_PILOT_ROOT)
    build = subparsers.add_parser("build", help="Build the complete balanced image pool")
    _common_parser(build, DEFAULT_POOL_ROOT)
    build.add_argument("--thresholds", required=True, help="Three comma-separated normalized entropy cut points")
    build.add_argument("--quota-per-level", type=int, default=10)
    build.add_argument("--max-scenes-per-pair", type=int, default=200)
    legacy = subparsers.add_parser("import-legacy", help="Recalibrate and import legacy images")
    _common_parser(legacy, DEFAULT_POOL_ROOT)
    legacy.add_argument("--thresholds", required=True, help="Three comma-separated normalized entropy cut points")
    legacy.add_argument("--quota-per-level", type=int, default=10)
    legacy.add_argument(
        "--source",
        type=Path,
        action="append",
        default=None,
        help="Legacy dataset JSON; may be repeated",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "analyze":
        report = analyze_pilot(args.output_root.resolve())
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    thresholds = parse_thresholds(getattr(args, "thresholds", None))
    pipeline = ImagePoolPipeline(
        mode=args.command,
        output_root=args.output_root,
        model_path=args.model_path,
        seed=args.seed,
        thresholds=thresholds,
        quota_per_level=getattr(args, "quota_per_level", 10),
        resume=args.resume,
    )
    if args.command == "pilot":
        pipeline.run_pilot()
    elif args.command == "build":
        if args.max_scenes_per_pair < 1:
            raise ValueError("--max-scenes-per-pair must be positive")
        pipeline.run_build(args.max_scenes_per_pair)
    elif args.command == "import-legacy":
        sources = args.source or [
            ROOT / "datasets" / "datasets.json",
            SCRIPT_DIR / "datasets" / "generated_shape_color_dataset.summary.json",
        ]
        pipeline.import_legacy([path.resolve() for path in sources])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
