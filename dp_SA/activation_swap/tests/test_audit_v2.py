from __future__ import annotations

from pathlib import Path

import pytest

from dp_SA.activation_swap.audit_v2 import (
    INPUT_FILES,
    _position_effect_rows,
    _prepare_output,
    _source_hashes,
    classify_ood_status,
    margin_regression_example,
    matched_drift_pairs,
    oriented_effect,
    original_pipeline_fields,
    panl_drift_contrast,
)
from dp_SA.activation_swap.metrics import bootstrap_values
from dp_SA.activation_swap.utils import canonical_hash, sha256_file


@pytest.mark.parametrize(
    ("side", "metric", "same", "cross", "expected"),
    [
        ("image_side", "soft_sa", .8, .3, .5),
        ("text_side", "soft_sa", .3, .8, .5),
        ("image_side", "hard_midpoint", .75, .25, .5),
        ("text_side", "hard_midpoint", .25, .75, .5),
        ("image_side", "fixed_clean_class_margin", .2, -.4, .6),
        ("text_side", "fixed_clean_class_margin", .2, -.4, .6),
        ("image_side", "first_token_change", 0, 1, 1),
        ("text_side", "first_token_change", 0, 1, 1),
    ],
)
def test_audit_metric_orientation(side, metric, same, cross, expected):
    assert oriented_effect(same, cross, side, metric) == pytest.approx(expected)


def test_margin_regression_values_do_not_flip_text_side():
    result = margin_regression_example()
    assert result["image_margin_effect"] == pytest.approx(.0690)
    assert result["text_margin_effect"] == pytest.approx(.0184)
    assert result["combined_margin_effect"] == pytest.approx(.0437)


def _diagnostic_rows():
    rows = []
    for position, offset in (("P1_PANL", .04), ("P1_PANL_PLUS_1", .02)):
        for index, side in enumerate(("image_side", "image_side", "text_side", "text_side")):
            for arm, value in (("same", .01 + index * .001), ("cross", .01 + index * .001 + offset)):
                rows.append({"position": position, "layer": 14, "recipient_case_id": f"r{index}",
                             "recipient_side": side, "swap_kind": arm, "cosine_distance": value,
                             "abs_log_norm_ratio": value / 2})
    return rows


def test_matched_cosine_and_panl_position_drift_contrast():
    rows = _diagnostic_rows()
    paired = matched_drift_pairs(rows, "P1_PANL", 14, "cosine_distance")
    assert [row["effect"] for row in paired] == pytest.approx([.04] * 4)
    interaction = panl_drift_contrast(rows, 14, "cosine_distance")
    assert [row["effect"] for row in interaction] == pytest.approx([.02] * 4)
    stats = bootstrap_values([row["effect"] for row in interaction], repeats=100, seed=42)
    assert stats["ci_low"] == pytest.approx(.02)


def test_heuristic_warning_is_not_automatic_scientific_failure():
    summary = [{"analysis": "condition", "cosine_gt_0_1_count": 5}]
    paired = [
        {"analysis": "cross_minus_same", "position": "P1_PANL", "layer": layer,
         "group": "all", "diagnostic_metric": "cosine_distance", "ci_low": -.01, "ci_high": .02}
        for layer in (14, 16)
    ] + [
        {"analysis": "PANL_minus_PANL_PLUS_1_cross_specific_drift", "position": "", "layer": layer,
         "group": "all", "diagnostic_metric": "cosine_distance", "ci_low": -.02, "ci_high": .01}
        for layer in (14, 16)
    ]
    status, details = classify_ood_status(summary, paired, natural_status="available")
    assert status == "caveat"
    assert details["heuristic_cosine_gt_0_1_count"] == 5


def test_original_pipeline_status_is_preserved_separately():
    original = {"panl_transfer_supported": False, "interpretation": "historical gate failed"}
    assert original_pipeline_fields(original) == {
        "original_pipeline_success": False,
        "original_pipeline_interpretation": "historical gate failed",
    }


def test_audit_resume_rejects_changed_inputs_and_never_overwrites_source(tmp_path: Path):
    source = tmp_path / "formal"
    source.mkdir()
    for name in INPUT_FILES:
        (source / name).write_text(f"original:{name}\n", encoding="utf-8")
    original_summary_hash = sha256_file(source / "summary.json")
    output = source / "analysis_audit_v2"
    config = {"format_version": 2, "source_root": str(source), "output_root": str(output)}
    config["fingerprint"] = canonical_hash(config)
    hashes = _source_hashes(source)
    _prepare_output(source, output, config, hashes, resume=False)
    assert sha256_file(source / "summary.json") == original_summary_hash
    with pytest.raises(FileExistsError):
        _prepare_output(source, output, config, hashes, resume=False)
    (source / "summary.md").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="input fingerprint changed"):
        _prepare_output(source, output, config, _source_hashes(source), resume=True)


def test_position_contrast_uses_metric_specific_orientations():
    grouped = {}
    for position, effect in (("P1_PANL", .4), ("P1_PANL_PLUS_1", .1)):
        grouped[(position, 14)] = [
            {"recipient_case_id": "i", "recipient_side": "image_side", "swap_kind": "same",
             "swap_soft_sa": effect},
            {"recipient_case_id": "i", "recipient_side": "image_side", "swap_kind": "cross",
             "swap_soft_sa": 0},
            {"recipient_case_id": "t", "recipient_side": "text_side", "swap_kind": "same",
             "swap_soft_sa": 0},
            {"recipient_case_id": "t", "recipient_side": "text_side", "swap_kind": "cross",
             "swap_soft_sa": effect},
        ]
    contrast = _position_effect_rows(grouped, "P1_PANL", "P1_PANL_PLUS_1", 14, "soft_sa")
    assert [row["effect"] for row in contrast] == pytest.approx([.3, .3])
