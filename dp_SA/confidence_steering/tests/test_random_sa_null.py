from __future__ import annotations

import json

import numpy as np
import pytest

from dp_SA.confidence_steering.io_utils import atomic_jsonl, canonical_hash
from dp_SA.confidence_steering.random_sa_null import (
    CANDIDATE_POOL_SIZE, ENDPOINTS, GROUPS, NATURAL_RUN_SPEC,
    _frozen_position_object, _historical_clean_baselines, candidate_metrics,
    random_null_comparison_rows, select_global_candidates,
)
from dp_SA.confidence_steering.random_sa_null_supplement import supplemental_rows
from dp_SA.confidence_steering.config import CANONICAL_COLORS


def _metric(recipient_index: int, candidate_id: int):
    size = 64
    raw = np.linspace(1.0, 2.0, size)
    true = raw * 0.4
    basis = np.eye(size, 2)
    hidden = np.random.default_rng(7).normal(size=(30, size))
    return candidate_metrics(
        raw, true, basis, hidden, raw,
        recipient_index=recipient_index, candidate_id=candidate_id,
    )


def test_global_candidates_are_complete_deterministic_and_globally_ranked() -> None:
    per_candidate = {}
    hashes = set()
    for candidate_id in range(1, CANDIDATE_POOL_SIZE + 1):
        rows = []
        for recipient_index, answer in enumerate(CANONICAL_COLORS):
            metric, basis, vector = _metric(recipient_index, candidate_id)
            assert metric["valid"] and basis.shape == (64, 2)
            assert np.max(np.abs(basis.T @ (vector / np.linalg.norm(vector)))) <= 1e-5
            rows.append({"recipient_answer": answer, **metric})
            hashes.add(metric["vector_sha256"])
        per_candidate[candidate_id] = rows
    selected, summaries = select_global_candidates(per_candidate, 20)
    selected_again, _ = select_global_candidates(per_candidate, 20)
    assert selected == selected_again and len(selected) == 20
    expected = sorted(
        range(1, CANDIDATE_POOL_SIZE + 1),
        key=lambda candidate_id: (
            np.mean([row["matching_distance"] for row in per_candidate[candidate_id]]),
            candidate_id,
        ),
    )[:20]
    assert selected == expected
    assert all(len(per_candidate[candidate_id]) == len(CANONICAL_COLORS) for candidate_id in selected)
    assert len(hashes) == CANDIDATE_POOL_SIZE * len(CANONICAL_COLORS)
    assert sum(row["selected_rank"] is not None for row in summaries) == 20


def test_global_candidate_rejects_one_bad_recipient() -> None:
    rows = [{"recipient_answer": answer, "valid": True, "matching_distance": 1.0}
            for answer in CANONICAL_COLORS]
    rows[-1]["valid"] = False
    with pytest.raises(ValueError, match="globally valid"):
        select_global_candidates({1: rows}, 1)


def _main_row(case_id: str, direction: str, layer: int, alpha: float) -> dict:
    row = {
        "case_id": case_id, "direction": direction, "layer": layer, "alpha": alpha,
        "clean_sa_logits": [1.0, 2.0, 3.0], "clean_sa_probabilities": [.1, .2, .7],
        "clean_soft_sa": .8, "clean_hard_sa": 2, "clean_class_margin": .5,
        "clean_panl_confidence_probe": .2, "clean_panl_sa_probe": .3,
        "panl_clean_hidden_hash": "panl", "config_fingerprint": "cfg",
        "hidden_definition": "hidden", "clean_lat_confidence_probe": .4,
    }
    return row


def test_historical_clean_requires_all_30_rows_and_unique_fields(tmp_path) -> None:
    trials = [
        _main_row("case", direction, layer, alpha)
        for direction in NATURAL_RUN_SPEC["directions"]
        for layer in NATURAL_RUN_SPEC["layers"]
        for alpha in NATURAL_RUN_SPEC["alphas"]
    ]
    manifest = [{
        "case_id": "case", "phase1_prompt_hash": "prompt",
        "phase0_answer_fingerprint": "answer", "positions": {"x": 1},
    }]
    atomic_jsonl(tmp_path / "artifacts/trials/main_trials.jsonl", trials)
    atomic_jsonl(tmp_path / "artifacts/manifests/runtime_manifest.jsonl", manifest)
    baselines, audit = _historical_clean_baselines(tmp_path, 1)
    assert len(baselines) == len(audit) == 1
    trials[1]["clean_soft_sa"] = .81
    atomic_jsonl(tmp_path / "artifacts/trials/main_trials.jsonl", trials)
    with pytest.raises(ValueError, match="Historical clean baseline mismatch"):
        _historical_clean_baselines(tmp_path, 1)


def test_negative_tail_p_and_all_three_aggregations() -> None:
    cases = (("a", "f1", "red"), ("b", "f2", "blue"))
    main = []
    null = []
    for case_id, family_id, answer in cases:
        base = {"case_id": case_id, "item_id": case_id, "family_id": family_id,
                "condition": "conflict_easy", "answer_origin": "follow_text",
                "fixed_answer": answer, "layer": 14}
        for alpha in (-2.0, 2.0):
            main.append({**base, "direction": "confidence_perp_sa_natural_scale", "alpha": alpha,
                         **{endpoint: -alpha for endpoint in ENDPOINTS}})
            for replicate, slope in ((1, -.5), (2, 0.0), (3, .5)):
                null.append({**base, "direction": "random_sa_subspace_null", "alpha": alpha,
                             "null_replicate": replicate,
                             **{endpoint: slope * alpha for endpoint in ENDPOINTS}})
    draws = [["f1", "f2"] for _ in range(10)]
    rows = random_null_comparison_rows(main, null, 3, draws)
    comparisons = [row for row in rows if row["row_type"] == "comparison"]
    assert {row["group"] for row in comparisons} == set(GROUPS)
    assert len(comparisons) == len(ENDPOINTS) * len(GROUPS)
    assert all(row["empirical_p_negative_one_sided"] == .25 for row in comparisons)
    assert all(row["minimum_attainable_p"] == .25 for row in comparisons)


def test_context_fingerprint_changes_with_any_context_component() -> None:
    context = {"prompt": "a", "position": "b", "model": "c", "processor": "d"}
    initial = canonical_hash(context)
    assert canonical_hash({**context, "position": "changed"}) != initial


def test_frozen_position_object_is_exact_and_only_allows_new_auxiliary_key() -> None:
    manifest = {"P1_LAT": {"processed_index": 10}, "P1_PANL": {"processed_index": 11}}
    located = {**manifest, "P1_CLASS_LIST_END": {"processed_index": 20}}
    assert _frozen_position_object(located, manifest) == manifest
    with pytest.raises(ValueError, match="values differ"):
        _frozen_position_object({**located, "P1_LAT": {"processed_index": 9}}, manifest)
    with pytest.raises(ValueError, match="schema mismatch"):
        _frozen_position_object({**located, "unexpected": {}}, manifest)


def test_unit_confidence_and_paired_family_contrast() -> None:
    main = []
    null = []
    for case_id, family_id, answer in (("a", "f1", "red"), ("b", "f2", "blue")):
        base = {"case_id": case_id, "item_id": case_id, "family_id": family_id,
                "condition": "conflict_easy", "answer_origin": "follow_text",
                "fixed_answer": answer, "layer": 14}
        for alpha in (-2.0, 2.0):
            main.append({**base, "direction": "confidence_perp_sa_natural_scale", "alpha": alpha,
                         "delta_confidence_LAT_immediate": 5 * alpha,
                         "delta_panl_probe_sa": -2 * alpha,
                         "delta_final_soft_sa": -alpha})
            for replicate in (1, 2):
                null.append({**base, "direction": "random_sa_subspace_null", "alpha": alpha,
                             "null_replicate": replicate,
                             "delta_confidence_LAT_immediate": 4 * alpha,
                             "delta_panl_probe_sa": alpha,
                             "delta_final_soft_sa": .5 * alpha})
    normalized, contrasts = supplemental_rows(main, null, [["f1", "f2"]] * 20, repeats=2)
    panl = next(row for row in normalized if row["endpoint"] == "delta_panl_probe_sa" and row["group"] == "answer_equal_macro")
    assert panl["true_sa_per_lat_confidence"] == pytest.approx(-.4)
    assert panl["null_mean_sa_per_lat_confidence"] == pytest.approx(.25)
    paired = next(row for row in contrasts if row["endpoint"] == "delta_panl_probe_sa" and row["group"] == "answer_equal_macro")
    assert paired["paired_contrast"] == pytest.approx(-6.0)
    assert paired["ci95_low"] == pytest.approx(-6.0)
    assert paired["ci95_high"] == pytest.approx(-6.0)
    assert paired["ci_excludes_zero"] is True
