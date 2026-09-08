from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from dp_SA.io_utils import atomic_json, atomic_jsonl
from dp_SA.SA_trajectory.CLE_transport.analyze import FamilyBootstrap, analyze, bh_fdr
from dp_SA.SA_trajectory.CLE_transport.config import CONDITIONS, WINDOWS, WINDOW_NAMES
from dp_SA.SA_trajectory.CLE_transport.run import class_margin


def _diagnostics(start, end):
    return {"layers": list(range(start, end + 1)), "by_layer": {
        str(layer): {"head_count": 28, "blocked_edge_count": 1, "max_blocked_weight": 0.0,
                     "max_row_sum_error": 0.0, "finite": True, "hook_call_count": 1}
        for layer in range(start, end + 1)}}


def _base(case, family, answer, side):
    logits = list(map(float, range(9)))
    return {"status": "completed", "experiment": "PANL2CLE", "case_id": case,
            "family_id": family, "item_id": case, "answer": answer, "test_side": side,
            "condition": "C0_clean", "window_name": None, "window_start": None, "window_end": None,
            "query_name": None, "source_name": None, "query_index": None, "source_index": None,
            "clean_class_logits": logits, "blocked_class_logits": logits,
            "clean_soft_sa": .5, "blocked_soft_sa": .5,
            "clean_hard_sa_class": 8, "blocked_hard_sa_class": 8,
            "clean_margin": class_margin(logits, 8), "blocked_margin": class_margin(logits, 8),
            "delta_soft_sa": 0., "abs_delta_soft_sa": 0., "token_changed": False,
            "token_change_rate": 0., "logit_change_diff": 0., "positions": {},
            "attention_diagnostics": None, "parity": {"passed": True},
            "elapsed_seconds": .1, "fingerprint": "test"}


def test_margin_uses_mean_of_other_classes():
    assert class_margin(range(9), 8) == pytest.approx(4.5)


def test_answer_equal_macro_and_bh():
    manifest = [
        {"family_id": "a", "test_answer": "brown", "test_side": "high_image"},
        {"family_id": "b", "test_answer": "brown", "test_side": "high_text"},
        {"family_id": "c", "test_answer": "purple", "test_side": "high_text"},
    ]
    result = FamilyBootstrap(manifest, repeats=50).aggregate({"a": 0., "b": 2., "c": 4.}, "answer_equal_macro")
    assert result["mean"] == pytest.approx(2.5)
    assert bh_fdr([.01, .04, .03, .002]) == pytest.approx([.02, .04, .04, .008])


def test_analysis_outputs_complete_schema_and_figures(tmp_path):
    manifest = [
        {"case_id": "a", "family_id": "fa", "item_id": "a", "test_answer": "brown", "test_side": "high_image"},
        {"case_id": "b", "family_id": "fb", "item_id": "b", "test_answer": "purple", "test_side": "high_text"},
    ]
    atomic_jsonl(tmp_path / "artifacts/manifests/test_manifest.jsonl", manifest)
    rows = []
    for row in manifest:
        clean = _base(row["case_id"], row["family_id"], row["test_answer"], row["test_side"])
        rows.append(clean); atomic_json(tmp_path / "artifacts/trials" / f"{row['case_id']}__clean.json", clean)
        for wi, (start, end) in enumerate(WINDOWS):
            for ci, condition in enumerate(CONDITIONS):
                delta = .01 * (wi + 1) * (1 if ci == 0 else .5)
                trial = {**clean, "condition": condition, "window_name": WINDOW_NAMES[start, end],
                         "window_start": start, "window_end": end, "query_name": "P1_CLASS_LIST_END",
                         "source_name": "P1_PANL", "query_index": 9, "source_index": 2,
                         "blocked_soft_sa": .5 + delta, "delta_soft_sa": delta,
                         "abs_delta_soft_sa": abs(delta), "token_changed": ci == 0,
                         "token_change_rate": float(ci == 0), "logit_change_diff": delta,
                         "attention_diagnostics": _diagnostics(start, end), "parity": None}
                rows.append(trial)
                atomic_json(tmp_path / "artifacts/trials" / f"{row['case_id']}__{condition}__{start}.json", trial)
    result = analyze(experiment="PANL2CLE", output_root=tmp_path, smoke=True, repeats=50)
    assert result["status"] == "complete" and result["trial_count"] == 18
    required = ("case_level_trials.csv", "window_effects.csv", "paired_main_vs_control.csv",
                "attention_audit.csv", "README_zh.md")
    assert all((tmp_path / "tables" / name).stat().st_size for name in required)
    assert all((tmp_path / "figures" / name).stat().st_size for name in (
        "fig1_delta_sa_by_window.png", "fig2_token_change_rate_by_window.png",
        "fig3_logit_change_diff_by_window.png"))
    with (tmp_path / "tables/paired_main_vs_control.csv").open(newline="") as handle:
        paired = list(csv.DictReader(handle))
    primary = [row for row in paired if row["group"] == "answer_equal_macro"]
    assert len(primary) == 12 and all(row["q_bh"] for row in primary)
