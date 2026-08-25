from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from layer_metacognition.model_adapter import LanguageModules

from dp_SA.activation_swap.analyze import _activation_comparisons, _contrast_rows, _lodo_effect
from dp_SA.activation_swap.hooks import EmptyHook, SwapActivationHook
from dp_SA.activation_swap.matching import add_length_bins, build_swap_pairs, quantile_edges
from dp_SA.activation_swap.metrics import (
    bh_fdr, paired_rows, score_logits, sign_flip_p, stratified_effect_summary,
)
from dp_SA.activation_swap.run import _check_or_write_config, _load_cache, _write_or_compare_jsonl
from dp_SA.activation_swap.utils import atomic_jsonl, load_jsonl
from dp_SA.activation_swap.utils import sha256_file


def _row(case_id: str, side: str, length: int, *, construction: bool = False) -> dict:
    return {
        "case_id": case_id, "item_id": case_id, "test_side": None if construction else side,
        "construction_side": side if construction else None, "question_token_length": length,
        "answer_token_length": 1, "phase0_normalized_answer": "answer",
    }


def test_quantile_bins_collapse_duplicate_boundaries_and_match_is_deterministic():
    assert quantile_edges([1, 1, 1, 2], 10) == [1.0, 2.0]
    recipients = [_row(f"r{i}", "image_side" if i < 2 else "text_side", 35) for i in range(4)]
    donors = [_row(f"d{i}", "high_image" if i < 2 else "high_text", 35, construction=True) for i in range(4)]
    first, diagnostics = build_swap_pairs(recipients, donors, seed=42)
    second, diagnostics2 = build_swap_pairs(recipients, donors, seed=42)
    assert first == second and diagnostics == diagnostics2
    assert len(first) == 8
    assert {row["condition"] for row in first} == {"I_from_I", "I_from_T", "T_from_T", "T_from_I"}
    assert max(diagnostics["donor_reuse_counts"].values()) - min(diagnostics["donor_reuse_counts"].values()) <= 1


def test_matching_rejects_donor_recipient_item_overlap():
    recipient = [_row("same", "image_side", 35)]
    donor = [_row("same", "high_image", 35, construction=True)]
    with pytest.raises(ValueError, match="overlap"):
        build_swap_pairs(recipient, donor, seed=1)


def test_k8_metrics_and_paired_orientation():
    score = score_logits([0, 0, 0, 0, 0, 0, 0, 0, 4])
    assert score["hard_class"] == 8 and score["hard_midpoint"] == 1.0
    assert score["soft_sa"] > 0.5
    fixed = score_logits([0, 2, 0, 0, 0, 0, 0, 0, 1], clean_class=8)
    assert fixed["fixed_clean_class_margin"] == pytest.approx(0.75)
    rows = []
    for side in ("image_side", "text_side"):
        rows.extend([
            {"recipient_case_id": side, "recipient_side": side, "swap_kind": "same", "swap_soft_sa": .7},
            {"recipient_case_id": side, "recipient_side": side, "swap_kind": "cross", "swap_soft_sa": .3 if side == "image_side" else .9},
        ])
    pairs = paired_rows(rows, "swap_soft_sa")
    assert pairs[0]["effect"] == pytest.approx(.4)
    assert pairs[1]["effect"] == pytest.approx(.2)
    summary = stratified_effect_summary(pairs, repeats=100, seed=42)
    assert summary["mean"] == pytest.approx(.3)
    assert paired_rows([
        {"recipient_case_id": "change", "recipient_side": "image_side", "swap_kind": "same", "first_token_changed": False},
        {"recipient_case_id": "change", "recipient_side": "image_side", "swap_kind": "cross", "first_token_changed": True},
    ], "first_token_changed")[0]["effect"] == 1.0


class _Layer(torch.nn.Module):
    def forward(self, hidden):
        return hidden


def _modules():
    return LanguageModules([_Layer(), _Layer()], torch.nn.Identity(), torch.nn.Linear(3, 9, bias=False), 3, 2)


def test_full_hidden_swap_only_target_and_noop_parity():
    modules = _modules()
    hidden = torch.arange(24, dtype=torch.float32).reshape(1, 8, 3)
    source = torch.tensor([-1.0, -2.0, -3.0])
    hook = SwapActivationHook(modules, layer=1, position=3, source_hidden=source, prefill_sequence_length=8)
    with hook:
        swapped = modules.language_layers[1](hidden)
    assert torch.equal(swapped[0, 3], source)
    assert torch.equal(swapped[0, [0, 1, 2, 4, 5, 6, 7]], hidden[0, [0, 1, 2, 4, 5, 6, 7]])
    assert hook.diagnostics()["hook_count"] == 1
    empty = EmptyHook(modules, layer=0, prefill_sequence_length=8)
    with empty:
        output = modules.language_layers[0](hidden)
    empty.validate()
    assert output is hidden


def test_bootstrap_signflip_fdr_and_jsonl_tail_repair(tmp_path: Path):
    assert sign_flip_p([1, 1, 1, 1], repeats=100, seed=42) < .2
    q = bh_fdr([.01, .04, .2])
    assert q[0] <= q[1] <= q[2]
    path = tmp_path / "rows.jsonl"
    atomic_jsonl(path, [{"a": 1}])
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{bad")
    assert load_jsonl(path, repair_trailing=True) == [{"a": 1}]


def test_resume_rejects_fingerprint_and_mapping_changes(tmp_path: Path):
    config_path = tmp_path / "run_config.json"
    config = {"fingerprint": "run-a", "layers": [14]}
    _check_or_write_config(config_path, config, resume=False)
    with pytest.raises(FileExistsError):
        _check_or_write_config(config_path, config, resume=False)
    with pytest.raises(ValueError, match="fingerprint"):
        _check_or_write_config(config_path, {"fingerprint": "run-b", "layers": [14]}, resume=True)

    mapping_path = tmp_path / "swap_pair_manifest.jsonl"
    _write_or_compare_jsonl(mapping_path, [{"recipient": "r", "donor": "d1"}], resume=False)
    with pytest.raises(ValueError, match="manifest changed"):
        _write_or_compare_jsonl(mapping_path, [{"recipient": "r", "donor": "d2"}], resume=True)


def test_clean_cache_bytes_are_immutable_on_resume(tmp_path: Path):
    path = tmp_path / "cache.pt"
    torch.save({"run_fingerprint": "fp", "case_id": "case", "hidden": {"x": torch.ones(2)}}, path)
    digest = sha256_file(path)
    assert _load_cache(path, "fp", "case", digest)["case_id"] == "case"
    with path.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        _load_cache(path, "fp", "case", digest)


def test_activation_matched_comparison_flags_cross_only_panl_drift():
    grouped = {}
    for position in ("P1_PANL", "P1_PANL_PLUS_1"):
        rows = []
        for index in range(4):
            side = "image_side" if index < 2 else "text_side"
            for kind in ("same", "cross"):
                drift = 0.5 if position == "P1_PANL" and kind == "cross" else 0.1
                rows.append({"recipient_case_id": f"r{index}", "recipient_side": side,
                             "swap_kind": kind, "activation_diagnostics": {
                                 "norm_ratio": 1.0 + drift, "abs_log_norm_ratio": drift,
                                 "cosine_distance": drift,
                             }})
        grouped[(position, 14)] = rows
    comparisons, warnings = _activation_comparisons(grouped, repeats=100, seed=42)
    assert comparisons
    assert any(row["cross_only_panl_drift_warning"] for row in warnings
               if row["diagnostic_metric"] == "cosine_distance")


def test_matched_position_contrast_and_lodo_side_effects():
    grouped = {}
    for position, delta in (("P1_PANL", 0.4), ("P1_PANL_PLUS_1", 0.1)):
        grouped[(position, 14)] = [
            {"recipient_case_id": "image", "recipient_side": "image_side", "swap_kind": "same", "swap_soft_sa": delta, "first_token_changed": 0.0},
            {"recipient_case_id": "image", "recipient_side": "image_side", "swap_kind": "cross", "swap_soft_sa": 0.0, "first_token_changed": delta},
            {"recipient_case_id": "text", "recipient_side": "text_side", "swap_kind": "same", "swap_soft_sa": 0.0, "first_token_changed": 0.0},
            {"recipient_case_id": "text", "recipient_side": "text_side", "swap_kind": "cross", "swap_soft_sa": delta, "first_token_changed": delta},
        ]
    contrast = _contrast_rows(grouped, position="P1_PANL", control="P1_PANL_PLUS_1",
                              layer=14, metric="swap_soft_sa")
    assert [row["effect"] for row in contrast] == pytest.approx([0.3, 0.3])
    assert _lodo_effect(contrast, "all") == pytest.approx(0.3)
    assert _lodo_effect(contrast, "image_side") == pytest.approx(0.3)
    assert _lodo_effect([contrast[0]], "all") is None
    change_contrast = _contrast_rows(grouped, position="P1_PANL", control="P1_PANL_PLUS_1",
                                     layer=14, metric="first_token_changed")
    assert [row["effect"] for row in change_contrast] == pytest.approx([0.3, 0.3])
