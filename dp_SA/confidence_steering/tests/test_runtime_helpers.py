from __future__ import annotations

import numpy as np

from dp_SA.confidence_steering.analyze import empirical_null, symmetric_effect
from dp_SA.confidence_steering.io_utils import stable_shard
from dp_SA.confidence_steering.run import alpha_zero_parity, trial_key
from dp_SA.confidence_steering.prepare import prepare_gate_status
from dp_SA.confidence_steering.run_spec import expected_runtime_counts, normalize_run_spec


def test_two_gpu_sharding_is_stable_complete_and_disjoint() -> None:
    cases = [f"case-{i}" for i in range(100)]
    shards = [{c for c in cases if stable_shard(c, 2) == worker} for worker in (0, 1)]
    assert not shards[0] & shards[1] and shards[0] | shards[1] == set(cases)
    rows = [{"case_id": case, "direction": "confidence_raw", "layer": 14, "alpha": 2.0} for case in cases]
    assert sorted(trial_key(row) for row in rows) == sorted(trial_key(row) for shard in shards for row in rows if row["case_id"] in shard)


def test_symmetric_effect_and_empirical_null_extension() -> None:
    main = []; null = []
    for case in ("a", "b"):
        base = {"case_id": case, "item_id": case, "family_id": case, "condition": "conflict_easy", "answer_origin": "follow_text", "fixed_answer": "red", "layer": 14}
        for alpha in (-10.0, -2.0, 2.0, 10.0):
            main.append({**base, "direction": "confidence_perp_sa", "alpha": alpha, "delta_final_soft_sa": alpha})
            for rep in range(1, 21): null.append({**base, "direction": "rebuilt_shuffle_null", "alpha": alpha, "delta_final_soft_sa": alpha * .01, "null_replicate": rep})
    assert symmetric_effect(main, "delta_final_soft_sa", 2)[0]["effect"] == 2
    rows, expand = empirical_null(main, null, "delta_final_soft_sa")
    assert expand and all(r["empirical_p_one_sided"] == 1 / 21 for r in rows)


def test_alpha_zero_parity_covers_lat_panl_probe_and_sac() -> None:
    score = {"class_logits": list(range(9)), "class_probabilities": [1 / 9] * 9, "soft_sa_image_score": .5, "argmax_hard_class": 4}
    hidden = np.asarray([1.0, 2.0], dtype=np.float32)
    result = alpha_zero_parity(score, dict(score), hidden, hidden.copy(), hidden, hidden.copy(), 0.0)
    assert result["passed"]


def test_run_spec_is_dynamic_and_requires_explicit_shuffle() -> None:
    natural = normalize_run_spec(["confidence_raw", "confidence_parallel_sa", "confidence_perp_sa_natural_scale"], [16, 14], [2, -2, 1, -1, 0])
    assert natural["layers"] == [14, 16]
    assert natural["alphas"] == [-2.0, -1.0, 0.0, 1.0, 2.0]
    assert natural["paired_doses"] == [1.0, 2.0]
    assert natural["analysis_kind"] == "mechanism_diagnostic" and not natural["shuffle_requested"]
    shuffled = normalize_run_spec(["confidence_raw", "within_answer_shuffled_perp_difficulty_sa"], [14], [-1, 0, 1])
    assert shuffled["shuffle_requested"] and shuffled["paired_doses"] == [1.0]
    counts = expected_runtime_counts(natural, 100)
    assert counts == {"main_trials": 3000, "main_forwards": 2700, "null_trials": 0, "total_forwards": 2700}


def test_run_spec_rejects_invalid_and_fingerprints_semantics() -> None:
    import pytest

    with pytest.raises(ValueError, match="Duplicate direction"):
        normalize_run_spec(["confidence_raw", "confidence_raw"], [14], [-1, 0, 1])
    with pytest.raises(ValueError, match="Unknown direction"):
        normalize_run_spec(["unknown"], [14], [-1, 0, 1])
    with pytest.raises(ValueError, match="alpha=0"):
        normalize_run_spec(["confidence_raw"], [14], [-1, 1])
    left = normalize_run_spec(["confidence_raw"], [14], [-1, 0, 1])
    right = normalize_run_spec(["confidence_raw"], [14], [-2, 0, 2])
    assert left["fingerprint"] != right["fingerprint"]


def test_non_l14_probe_failure_is_reported_but_does_not_block_l14() -> None:
    gates = prepare_gate_status(
        {14: True, 16: False}, {14: True, 16: False},
        numerical_gate=True, panl_sa_gate=True,
    )
    assert gates["formal_eligible"]
    assert not gates["all_selected_confidence_probes_reliable"]
    assert gates["l14_confidence_probe_gate"] and gates["direction_sensitivity_gate"]


def test_panl_final_sa_and_l14_are_hard_gates() -> None:
    assert not prepare_gate_status({14: False}, {14: True}, numerical_gate=True, panl_sa_gate=True)["formal_eligible"]
    assert not prepare_gate_status({14: True}, {14: True}, numerical_gate=True, panl_sa_gate=False)["formal_eligible"]
