from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parents[2]
for candidate in (REPOSITORY_ROOT, ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from core import COLOR_RGB, cma_scores, matched_text_pairs, recolor_target, signed_soft_sa
from prompts import (
    ANSWER_PREFILL, PHASE0_IMAGE_ONLY_TEMPLATE, PHASE0_TEXT_ONLY_TEMPLATE,
    image_only_prompt, text_only_prompt,
)


def test_exact_single_modal_templates_and_prefill() -> None:
    question = "What color?"
    clue = "A red clue."
    assert ANSWER_PREFILL == "**Answer**:"
    assert text_only_prompt(question, clue) == PHASE0_TEXT_ONLY_TEMPLATE.format(
        question=question, text_clue=clue,
    )
    assert "using the text clue and the image" not in text_only_prompt(question, clue)
    assert image_only_prompt(question) == PHASE0_IMAGE_ONLY_TEMPLATE.format(question=question)
    assert "Text clue:" not in image_only_prompt(question)


def test_cma_direction_interaction_and_unidentifiable() -> None:
    image_dominant = cma_scores(4, 1, 3, 0)
    assert image_dominant["phi_image"] == 3
    assert image_dominant["phi_text"] == 1
    assert image_dominant["cma_signed"] == 0.5
    assert image_dominant["interaction"] == 0
    text_dominant = cma_scores(4, 3, 1, 0)
    assert text_dominant["cma_signed"] == -0.5
    undefined = cma_scores(1, 1, 1, 1)
    assert undefined["identifiable"] is False
    assert undefined["cma_signed"] is None


def test_signed_soft_sa_mapping() -> None:
    assert signed_soft_sa(0.05) == -1
    assert signed_soft_sa(0.5) == 0
    assert signed_soft_sa(0.95) == 1
    assert signed_soft_sa(-10) == -1
    assert signed_soft_sa(10) == 1


def test_text_matching_enforces_tolerances_and_order() -> None:
    original = {
        "text_clue": "original", "normalized_entropy": .20,
        "target_probability": .70, "target_margin": 2.0,
    }
    near = {
        "text_clue": "near", "normalized_entropy": .22,
        "target_probability": .75, "target_margin": 2.2,
    }
    far = {
        "text_clue": "far", "normalized_entropy": .30,
        "target_probability": .70, "target_margin": 2.0,
    }
    matches = matched_text_pairs([original], [far, near], .05, .10)
    assert len(matches) == 1
    assert matches[0][1]["text_clue"] == "near"


def test_recolor_changes_only_visible_target_and_two_layout_fields(tmp_path: Path) -> None:
    source = np.full((8, 8, 3), (245, 245, 245), dtype=np.uint8)
    source[1:7, 1:7] = COLOR_RGB["red"]
    source[0, 0] = COLOR_RGB["green"]
    target = np.zeros((8, 8), dtype=np.uint8); target[1:7, 1:7] = 255
    occluder = np.zeros((8, 8), dtype=np.uint8); occluder[3:5, 3:5] = 255
    source[3:5, 3:5] = COLOR_RGB["black"]
    layout = {
        "target_color": "red",
        "objects": [
            {"role": "target", "color": "red", "shape": "circle", "center": [4, 4]},
            {"role": "distractor", "color": "green", "shape": "square"},
        ],
    }
    source_path = tmp_path / "source.png"
    target_path = tmp_path / "target.png"
    occluder_path = tmp_path / "occluder.png"
    layout_path = tmp_path / "source.layout.json"
    Image.fromarray(source).save(source_path)
    Image.fromarray(target).save(target_path)
    Image.fromarray(occluder).save(occluder_path)
    layout_path.write_text(json.dumps(layout), encoding="utf-8")
    destinations = {
        "image": tmp_path / "cf.png", "layout": tmp_path / "cf.layout.json",
        "target": tmp_path / "cf.target.png", "occluder": tmp_path / "cf.occluder.png",
    }
    audit = recolor_target(
        source_image=source_path, source_layout=layout_path,
        target_mask=target_path, occluder_mask=occluder_path,
        destination_image=destinations["image"], destination_layout=destinations["layout"],
        destination_target_mask=destinations["target"],
        destination_occluder_mask=destinations["occluder"], new_color="blue",
    )
    result = np.asarray(Image.open(destinations["image"]).convert("RGB"))
    changed = np.any(result != source, axis=-1)
    expected = (target > 0) & ~(occluder > 0)
    assert np.array_equal(changed, expected)
    assert audit["changed_pixel_count"] == int(expected.sum())
    assert result[0, 0].tolist() == list(COLOR_RGB["green"])
    assert result[3, 3].tolist() == list(COLOR_RGB["black"])
    updated = json.loads(destinations["layout"].read_text(encoding="utf-8"))
    assert updated["target_color"] == "blue"
    assert updated["objects"][0]["color"] == "blue"
    assert updated["objects"][1] == layout["objects"][1]
    assert audit["target_mask_sha256"] == audit["counterfactual_target_mask_sha256"]
    assert audit["occluder_mask_sha256"] == audit["counterfactual_occluder_mask_sha256"]

