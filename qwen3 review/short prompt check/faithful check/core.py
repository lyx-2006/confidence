from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from config import COLORS, SEED


COLOR_RGB = {
    "red": (220, 45, 45), "orange": (242, 132, 30), "yellow": (240, 205, 35),
    "green": (45, 160, 75), "blue": (45, 95, 215), "cyan": (35, 185, 200),
    "purple": (130, 70, 185), "pink": (225, 100, 160), "brown": (135, 82, 45),
    "white": (250, 250, 250), "black": (25, 25, 25), "gray": (125, 125, 125),
}


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


def stable_key(*values: Any, seed: int = SEED) -> str:
    raw = json.dumps([seed, *values], ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _question_text(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("text")
    text = str(value or "").strip()
    if not text:
        raise ValueError("Dataset item has no question")
    return text


def _shape_from_question(question: str) -> str:
    match = re.search(r"color of the (.+?)\?", question, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse shape from question: {question!r}")
    return normalize(match.group(1))


def _sidecars(image: Path) -> dict[str, Path]:
    stem = image.name[:-4] if image.name.endswith(".png") else image.stem
    return {
        "image": image,
        "layout": image.with_name(f"{stem}.layout.json"),
        "target_mask": image.with_name(f"{stem}.target_mask.png"),
        "occluder_mask": image.with_name(f"{stem}.occluder_mask.png"),
    }


def load_image_candidates(dataset_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError("Faithful-check source dataset must be an object with items")
    rows: list[dict[str, Any]] = []
    for item in payload["items"]:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id", "")).strip()
        question = _question_text(item.get("question"))
        text_color = normalize(item.get("answer"))
        image_color = normalize(item.get("conflict_answer", item.get("conflict_ans")))
        if text_color not in COLORS or image_color not in COLORS or text_color == image_color:
            continue
        groups = item.get("groups")
        if not isinstance(groups, dict):
            raise ValueError(f"Item {item_id} has no groups object")
        for difficulty in ("easy", "hard"):
            group = groups.get(f"conflict_{difficulty}")
            raw = group.get("image") if isinstance(group, dict) else None
            if not isinstance(raw, str) or not raw.strip():
                continue
            image = (dataset_path.parent / raw).resolve()
            paths = _sidecars(image)
            missing = [str(path) for path in paths.values() if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Item {item_id}/{difficulty} missing artifacts: {missing}")
            layout = json.loads(paths["layout"].read_text(encoding="utf-8"))
            if normalize(layout.get("target_color")) != image_color:
                raise ValueError(f"Item {item_id}/{difficulty} layout target color mismatch")
            targets = [obj for obj in layout.get("objects", []) if obj.get("role") == "target"]
            if len(targets) != 1 or normalize(targets[0].get("color")) != image_color:
                raise ValueError(f"Item {item_id}/{difficulty} has invalid target object")
            rows.append({
                "case_id": f"{item_id}__conflict_{difficulty}",
                "item_id": item_id,
                "difficulty": difficulty,
                "question": question,
                "shape": _shape_from_question(question),
                "text_color": text_color,
                "image_color": image_color,
                "source_group": f"conflict_{difficulty}",
                "source_image_calibration": group,
                **{f"source_{name}": str(path) for name, path in paths.items()},
            })
    return rows


def order_image_candidates(
    rows: Iterable[dict[str, Any]],
    difficulty: str,
    used_items: set[str] | None = None,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    remaining = [dict(row) for row in rows if row["difficulty"] == difficulty]
    used_items = set(used_items or ())
    counts: dict[str, Counter[str]] = {
        "text": Counter(), "image": Counter(), "shape": Counter(),
    }
    ordered: list[dict[str, Any]] = []
    while remaining:
        chosen = min(
            remaining,
            key=lambda row: (
                int(str(row["item_id"]) in used_items),
                counts["text"][row["text_color"]],
                counts["image"][row["image_color"]],
                counts["shape"][row["shape"]],
                stable_key(row["case_id"], seed=seed),
            ),
        )
        remaining.remove(chosen)
        ordered.append(chosen)
        used_items.add(str(chosen["item_id"]))
        counts["text"][chosen["text_color"]] += 1
        counts["image"][chosen["image_color"]] += 1
        counts["shape"][chosen["shape"]] += 1
    return ordered


def load_text_pool(pool_path: Path) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(pool_path.read_text(encoding="utf-8"))
    entries = payload.get("colors", []) if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError("Text pool must be a legacy array or text-pool object")
    by_color: dict[str, list[dict[str, Any]]] = {color: [] for color in COLORS}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        color = normalize(entry.get("color"))
        if color not in by_color:
            continue
        levels = entry.get("entropy_bins", entry.get("prior_levels", []))
        seen: set[str] = set()
        bins: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for level in levels if isinstance(levels, list) else []:
            if not isinstance(level, dict):
                continue
            raw_bin = level.get("entropy_bin_id", level.get("bin_id", -1))
            try:
                bin_id = int(raw_bin)
            except (TypeError, ValueError):
                bin_id = -1
            for prior in level.get("priors", []):
                if not isinstance(prior, dict) or prior.get("accepted") is not True:
                    continue
                clue = str(prior.get("text_clue", prior.get("clue", ""))).strip()
                key = normalize(clue)
                if not clue or key in seen:
                    continue
                seen.add(key)
                bins[bin_id].append({
                    "color": color,
                    "text_clue": clue,
                    "source_bin_id": bin_id,
                    "source_candidate_id": prior.get("candidate_id"),
                    "source_strategy_family": prior.get("strategy_family"),
                    "source_difficulty_type": prior.get("difficulty_type"),
                })
        for values in bins.values():
            values.sort(key=lambda row: stable_key(color, row["text_clue"]))
        # Round-robin bins so the first ten do not collapse to one source bin.
        active = sorted(bins)
        while active:
            next_active = []
            for bin_id in active:
                if bins[bin_id]:
                    by_color[color].append(bins[bin_id].pop(0))
                if bins[bin_id]:
                    next_active.append(bin_id)
            active = next_active
    missing = [color for color, values in by_color.items() if not values]
    if missing:
        raise ValueError(f"Text pool has no accepted clues for colors: {missing}")
    return by_color


def eligible_third_colors(layout: dict[str, Any], text_color: str, image_color: str) -> list[str]:
    used = {
        normalize(obj.get("color"))
        for obj in layout.get("objects", [])
        if isinstance(obj, dict) and obj.get("role") != "target"
    }
    return [
        color for color in COLORS
        if color not in used and color not in {normalize(text_color), normalize(image_color)}
    ]


def difficulty_tuple(row: dict[str, Any]) -> tuple[float, float, float, int]:
    return (
        float(row["normalized_entropy"]),
        float(row["target_probability"]),
        float(row["target_margin"]),
        len(str(row["text_clue"])),
    )


def text_match_deltas(first: dict[str, Any], second: dict[str, Any]) -> dict[str, float]:
    a, b = difficulty_tuple(first), difficulty_tuple(second)
    return {
        "entropy_delta": abs(a[0] - b[0]),
        "target_probability_delta": abs(a[1] - b[1]),
        "target_margin_delta": abs(a[2] - b[2]),
        "length_delta": float(abs(a[3] - b[3])),
    }


def matched_text_pairs(
    originals: Iterable[dict[str, Any]],
    counterfactuals: Iterable[dict[str, Any]],
    entropy_tolerance: float,
    probability_tolerance: float,
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, float]]]:
    matches = []
    for original in originals:
        for counterfactual in counterfactuals:
            delta = text_match_deltas(original, counterfactual)
            if (
                delta["entropy_delta"] <= entropy_tolerance
                and delta["target_probability_delta"] <= probability_tolerance
            ):
                matches.append((original, counterfactual, delta))
    matches.sort(key=lambda value: (
        value[2]["entropy_delta"], value[2]["target_probability_delta"],
        value[2]["target_margin_delta"], value[2]["length_delta"],
        stable_key(value[0]["text_clue"], value[1]["text_clue"]),
    ))
    return matches


def _atomic_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".png", dir=path.parent)
    os.close(fd)
    try:
        image.save(temporary, format="PNG")
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def recolor_target(
    *, source_image: Path, source_layout: Path, target_mask: Path,
    occluder_mask: Path, destination_image: Path, destination_layout: Path,
    destination_target_mask: Path, destination_occluder_mask: Path,
    new_color: str,
) -> dict[str, Any]:
    if new_color not in COLOR_RGB:
        raise ValueError(f"Unknown replacement color: {new_color}")
    layout = json.loads(source_layout.read_text(encoding="utf-8"))
    old_color = normalize(layout.get("target_color"))
    if old_color not in COLOR_RGB or old_color == new_color:
        raise ValueError(f"Invalid target recolor {old_color!r} -> {new_color!r}")
    targets = [obj for obj in layout.get("objects", []) if obj.get("role") == "target"]
    if len(targets) != 1 or normalize(targets[0].get("color")) != old_color:
        raise ValueError("Layout must contain one matching target object")
    if new_color not in eligible_third_colors(layout, "__none__", old_color):
        raise ValueError("Replacement color is already used by a non-target object")

    with Image.open(source_image) as handle:
        source = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
    with Image.open(target_mask) as handle:
        target = np.asarray(handle.convert("L"), dtype=np.uint8) > 0
    with Image.open(occluder_mask) as handle:
        occluder = np.asarray(handle.convert("L"), dtype=np.uint8) > 0
    if source.shape[:2] != target.shape or target.shape != occluder.shape:
        raise ValueError("Image and mask dimensions differ")
    visible = target & ~occluder
    old_rgb = np.asarray(COLOR_RGB[old_color], dtype=np.uint8)
    new_rgb = np.asarray(COLOR_RGB[new_color], dtype=np.uint8)
    change = visible & np.all(source == old_rgb, axis=-1)
    if not bool(change.any()):
        raise ValueError("No visible target fill pixels matched the trusted RGB map")
    result = source.copy()
    result[change] = new_rgb
    actual_diff = np.any(result != source, axis=-1)
    if not np.array_equal(actual_diff, change) or bool((actual_diff & ~visible).any()):
        raise RuntimeError("Counterfactual recolor modified non-target pixels")

    updated = copy.deepcopy(layout)
    updated["target_color"] = new_color
    updated_targets = [obj for obj in updated["objects"] if obj.get("role") == "target"]
    updated_targets[0]["color"] = new_color
    _atomic_image(destination_image, Image.fromarray(result, mode="RGB"))
    _atomic_json(destination_layout, updated)
    destination_target_mask.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target_mask, destination_target_mask)
    shutil.copy2(occluder_mask, destination_occluder_mask)
    written_layout = json.loads(destination_layout.read_text(encoding="utf-8"))
    expected_layout = copy.deepcopy(layout)
    expected_layout["target_color"] = new_color
    [obj for obj in expected_layout["objects"] if obj.get("role") == "target"][0]["color"] = new_color
    if written_layout != expected_layout:
        raise RuntimeError("Counterfactual layout changed outside the two target color fields")
    target_hash = sha256_path(target_mask)
    occluder_hash = sha256_path(occluder_mask)
    if target_hash != sha256_path(destination_target_mask):
        raise RuntimeError("Target mask changed while creating counterfactual")
    if occluder_hash != sha256_path(destination_occluder_mask):
        raise RuntimeError("Occluder mask changed while creating counterfactual")
    return {
        "old_color": old_color,
        "new_color": new_color,
        "changed_pixel_count": int(change.sum()),
        "visible_target_pixel_count": int(visible.sum()),
        "all_changed_pixels_within_visible_target": True,
        "non_target_pixels_bitwise_equal": True,
        "layout_only_target_color_changed": True,
        "target_mask_sha256": target_hash,
        "counterfactual_target_mask_sha256": sha256_path(destination_target_mask),
        "occluder_mask_sha256": occluder_hash,
        "counterfactual_occluder_mask_sha256": sha256_path(destination_occluder_mask),
    }


def cma_scores(
    original: float,
    image_counterfactual: float,
    text_counterfactual: float,
    joint_counterfactual: float,
    epsilon: float = 1e-8,
) -> dict[str, Any]:
    phi_image = 0.5 * (
        (float(original) - float(image_counterfactual))
        + (float(text_counterfactual) - float(joint_counterfactual))
    )
    phi_text = 0.5 * (
        (float(original) - float(text_counterfactual))
        + (float(image_counterfactual) - float(joint_counterfactual))
    )
    denominator = abs(phi_image) + abs(phi_text)
    identifiable = bool(math.isfinite(denominator) and denominator >= epsilon)
    signed = (abs(phi_image) - abs(phi_text)) / denominator if identifiable else None
    return {
        "phi_image": phi_image,
        "phi_text": phi_text,
        "attribution_denominator": denominator,
        "identifiable": identifiable,
        "cma_signed": signed,
        "image_share": (abs(phi_image) / denominator) if identifiable else None,
        "text_share": (abs(phi_text) / denominator) if identifiable else None,
        "interaction": (
            float(original) - float(image_counterfactual)
            - float(text_counterfactual) + float(joint_counterfactual)
        ),
    }


def signed_soft_sa(soft_sa: float) -> float:
    value = (float(soft_sa) - 0.5) / 0.45
    if value >= 1.0 - 1e-12:
        return 1.0
    if value <= -1.0 + 1e-12:
        return -1.0
    return float(value)
