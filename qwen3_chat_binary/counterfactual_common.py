from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFilter

from confidence_test.answer_metrics import compute_answer_metrics, normalize_answer, parse_answer_output
from dp_SA.io_utils import canonical_hash, sha256_file
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.model_adapter import model_input_device

from .config import DATASET_PATH, MODEL_PATH, REPOSITORY_ROOT
from .dataset import COLORS, ConflictCase, load_conflict_cases
from .prompts import ANSWER_PREFILL, phase0_prompt
from .runtime import load_qwen3_inference


CURRENT_DATA_ROOT = DATASET_PATH.parent
TEXT_POOL_PATH = CURRENT_DATA_ROOT / "text_entropy_calibration.json"
IMAGE_POOL_ROOT = CURRENT_DATA_ROOT / "interval_pool_full"
CAPTURE_PHASE0_PATH = REPOSITORY_ROOT / "qwen3_chat_binary" / "output" / "Capture" / "tables" / "phase0_results.jsonl"
TEXT_ENTROPY_TOLERANCE = 0.05
TEXT_PROBABILITY_TOLERANCE = 0.10
EPSILON = 1e-8

TEXT_ONLY_TEMPLATE = """Question:
{question}

Text clue:
{text_clue}

Answer the question using the text clue.

Answer as concisely as possible.
Do not provide reasoning, explanation, confidence, source attribution, or any additional text.

Output exactly:

**Answer**: <your answer>"""

IMAGE_ONLY_TEMPLATE = """Question:
{question}

Answer the question using the image.

Answer as concisely as possible.
Do not provide reasoning, explanation, confidence, source attribution, or any additional text.

Output exactly:

**Answer**: <your answer>"""


def text_only_prompt(question: str, clue: str) -> str:
    return TEXT_ONLY_TEMPLATE.format(question=question, text_clue=clue)


def image_only_prompt(question: str) -> str:
    return IMAGE_ONLY_TEMPLATE.format(question=question)


def cma_scores(p00: float, p10: float, p01: float, p11: float) -> dict[str, Any]:
    values = tuple(map(float, (p00, p10, p01, p11)))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("CMA inputs must be finite")
    phi_image = 0.5 * ((values[0] - values[1]) + (values[2] - values[3]))
    phi_text = 0.5 * ((values[0] - values[2]) + (values[1] - values[3]))
    denominator = abs(phi_image) + abs(phi_text)
    identifiable = denominator >= EPSILON
    share = abs(phi_image) / denominator if identifiable else None
    return {
        "phi_image": phi_image,
        "phi_text": phi_text,
        "attribution_denominator": denominator,
        "identifiable": identifiable,
        "image_share": share,
        "text_share": 1.0 - share if share is not None else None,
        "cma_signed": 2.0 * share - 1.0 if share is not None else None,
        "interaction": values[0] - values[1] - values[2] + values[3],
    }


def stable_key(*values: Any) -> str:
    return canonical_hash(list(values))


def load_text_pool() -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(TEXT_POOL_PATH.read_text(encoding="utf-8"))
    result = {str(entry["color"]): list(entry["clues"]) for entry in payload}
    if set(result) != set(COLORS):
        raise ValueError("Text pool colors do not match configured colors")
    return result


def load_image_record(case: ConflictCase) -> dict[str, Any]:
    path = IMAGE_POOL_ROOT / case.image_answer / f"{case.shape}.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    matches = [row for row in rows if row.get("image") == case.image_path.name]
    if len(matches) != 1 or not isinstance(matches[0].get("layout"), dict):
        raise ValueError(f"Could not resolve one image-pool record for {case.case_id}")
    return matches[0]


def _renderer_module() -> Any:
    path = REPOSITORY_ROOT / "generate dataset" / "code" / "current" / "generate_image_pool.py"
    name = "qwen3_chat_binary_counterfactual_renderer"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import renderer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _render_layout(layout: dict[str, Any], directory: Path) -> None:
    pool = _renderer_module()
    directory.mkdir(parents=True, exist_ok=True)
    style_rng = random.Random(pool.LEGACY.derive_seed(int(layout["case_seed"]), "render-style"))
    style = {
        "background_rgb": list(style_rng.choice(pool.LEGACY.ALLOWED_BACKGROUNDS)),
        "outline_rgb": list(pool.LEGACY.DEFAULT_RENDER_STYLE["outline_rgb"]),
        "outline_width": int(pool.LEGACY.DEFAULT_RENDER_STYLE["outline_width"]),
    }
    with tempfile.TemporaryDirectory(prefix="cf-render-") as temporary:
        rendered = pool.LEGACY.render_scene_locally(layout, Path(temporary), style)
        for source, target in (
            ("image.png", "sharp.png"), ("layout.json", "layout.json"),
            ("target_mask.png", "target_mask.png"), ("occluder_mask.png", "occluder_mask.png"),
        ):
            shutil.copy2(rendered / source, directory / target)


def _blur(source: Path, destination: Path, radius: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        value = image.convert("RGB")
        if radius > 0:
            value = value.filter(ImageFilter.GaussianBlur(radius=radius))
        value.save(destination, format="PNG")


def render_counterfactual(case: ConflictCase, third_color: str, output: Path) -> dict[str, Any]:
    record = load_image_record(case)
    layout = copy.deepcopy(record["layout"])
    original_dir, counterfactual_dir = output / "original", output / "counterfactual"
    _render_layout(layout, original_dir)
    blur_radius = float(record["generation"]["blur_radius"])
    _blur(original_dir / "sharp.png", original_dir / "image.png", blur_radius)
    with Image.open(case.image_path) as expected, Image.open(original_dir / "image.png") as reproduced:
        original_equal = np.array_equal(np.asarray(expected.convert("RGB")), np.asarray(reproduced.convert("RGB")))
    if not original_equal:
        raise RuntimeError("Original image could not be reproduced pixel-exactly")

    changed_layout = copy.deepcopy(layout)
    changed_layout["target_color"] = third_color
    targets = [obj for obj in changed_layout["objects"] if obj.get("role") == "target"]
    if len(targets) != 1:
        raise ValueError("Layout does not have exactly one target")
    targets[0]["color"] = third_color
    distractor_collision = any(
        obj.get("role") != "target" and obj.get("color") == third_color
        for obj in changed_layout["objects"]
    )
    _render_layout(changed_layout, counterfactual_dir)
    _blur(counterfactual_dir / "sharp.png", counterfactual_dir / "image.png", blur_radius)
    with Image.open(original_dir / "sharp.png") as first, Image.open(counterfactual_dir / "sharp.png") as second:
        difference = np.any(np.asarray(first.convert("RGB")) != np.asarray(second.convert("RGB")), axis=-1)
    with Image.open(original_dir / "target_mask.png") as handle:
        target = np.asarray(handle.convert("L")) > 0
    with Image.open(original_dir / "occluder_mask.png") as handle:
        occluder = np.asarray(handle.convert("L")) > 0
    with Image.open(counterfactual_dir / "target_mask.png") as handle:
        target_cf = np.asarray(handle.convert("L")) > 0
    with Image.open(counterfactual_dir / "occluder_mask.png") as handle:
        occluder_cf = np.asarray(handle.convert("L")) > 0
    if not np.array_equal(target, target_cf) or not np.array_equal(occluder, occluder_cf):
        raise RuntimeError("Counterfactual masks changed")
    visible_target = target & ~occluder
    if not difference.any() or bool((difference & ~visible_target).any()):
        raise RuntimeError("Sharp counterfactual changed pixels outside the visible target")
    return {
        "source_record": str(IMAGE_POOL_ROOT / case.image_answer / f"{case.shape}.json"),
        "source_candidate_id": record["candidate_id"],
        "original_image": str(case.image_path),
        "counterfactual_image": str((counterfactual_dir / "image.png").resolve()),
        "counterfactual_layout": str((counterfactual_dir / "layout.json").resolve()),
        "blur_radius": blur_radius,
        "original_reproduced_pixel_exact": True,
        "sharp_changed_pixel_count": int(difference.sum()),
        "sharp_changes_within_visible_target": True,
        "masks_equal": True,
        "distractor_color_collision": distractor_collision,
        "original_image_sha256": sha256_file(case.image_path),
        "counterfactual_image_sha256": sha256_file(counterfactual_dir / "image.png"),
    }


class BehaviorRunner:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path.resolve()
        self.inference = load_qwen3_inference(self.model_path)
        self.processor = self.inference.processor
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.device = model_input_device(self.inference)
        self.color_ids = tuple(self._one_token(color) for color in COLORS)

    def _one_token(self, value: str) -> int:
        ids = self.tokenizer.encode(value, add_special_tokens=False)
        if len(ids) != 1 or self.tokenizer.decode(ids, skip_special_tokens=False) != value:
            raise ValueError(f"Color is not one token: {value!r} -> {ids}")
        return int(ids[0])

    def _inputs(self, prompt: str, image_path: str | None):
        content: list[dict[str, str]] = []
        if image_path:
            content.append({"type": "image", "image": str(Path(image_path).resolve())})
        content.append({"type": "text", "text": prompt})
        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": ANSWER_PREFILL}]},
        ]
        rendered = render_continued_assistant(self.processor, messages, ANSWER_PREFILL)
        prefix = self.tokenizer.encode(rendered, add_special_tokens=False)
        for color, token_id in zip(COLORS, self.color_ids):
            combined = self.tokenizer.encode(rendered + color, add_special_tokens=False)
            if combined[:-1] != prefix or combined[-1] != token_id:
                raise ValueError(f"Color token boundary failed for {color}")
        inputs = prepare_multimodal_inputs(self.processor, messages, rendered, device=self.device)
        return rendered, inputs

    def score(self, prompt: str, image_path: str | None, expected: str, *, generate: bool) -> dict[str, Any]:
        rendered, inputs = self._inputs(prompt, image_path)
        generated_ids: list[int] = []
        raw = normalized = None
        parsed = False
        if generate:
            with torch.inference_mode():
                output = self.inference.model.generate(
                    **inputs, max_new_tokens=24, do_sample=False, use_cache=True
                )
            generated_ids = [int(x) for x in output[0, inputs.input_ids.shape[1]:].tolist()]
            continuation = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            raw = ANSWER_PREFILL + continuation
            _answer, normalized, parsed = parse_answer_output(raw)
        with torch.inference_mode():
            logits = self.inference.model(**inputs, use_cache=False).logits[0, -1].detach().float().cpu()
        metrics = asdict(compute_answer_metrics(logits, COLORS, expected, self.tokenizer))
        selected = logits[list(self.color_ids)]
        mass = float(torch.exp(torch.logsumexp(selected, 0) - torch.logsumexp(logits, 0)))
        target_logit = float(metrics["answer_class_logits"][expected])
        others = [float(v) for k, v in metrics["answer_class_logits"].items() if k != expected]
        return {
            "prompt_hash": canonical_hash(prompt), "rendered_hash": canonical_hash(rendered),
            "image_path": str(Path(image_path).resolve()) if image_path else None,
            "image_sha256": sha256_file(image_path) if image_path else None,
            "actual_output": raw, "normalized_answer": normalized, "parse_success": bool(parsed),
            "generated_token_ids": generated_ids, "expected_color": expected,
            "target_logit": target_logit,
            "target_probability": float(metrics["answer_class_probabilities"][expected]),
            "target_margin": target_logit - max(others),
            "normalized_entropy": float(metrics["answer_entropy"]),
            "label_probability_mass": mass,
            **metrics,
        }

    def text_only(self, case: ConflictCase, clue: str, expected: str) -> dict[str, Any]:
        return self.score(text_only_prompt(case.question, clue), None, expected, generate=True)

    def image_only(self, case: ConflictCase, image: str, expected: str) -> dict[str, Any]:
        return self.score(image_only_prompt(case.question), image, expected, generate=True)

    def multimodal(self, case: ConflictCase, clue: str, image: str, fixed: str) -> dict[str, Any]:
        return self.score(phase0_prompt(case.question, clue), image, fixed, generate=True)
