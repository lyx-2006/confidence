from __future__ import annotations

import numpy as np

from dp_SA.confidence_steering.analyze import empirical_null, symmetric_effect
from dp_SA.confidence_steering.io_utils import stable_shard
from dp_SA.confidence_steering.run import alpha_zero_parity, trial_key


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
