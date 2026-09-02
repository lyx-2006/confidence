from __future__ import annotations

import numpy as np
import torch

from dp_SA.checkpoint_steering.run import validate_alpha_zero
from dp_SA.confidence_steering.io_utils import stable_shard
from dp_SA.soft_score import soft_sa_from_logits


def test_case_sharding_is_stable_and_complete() -> None:
    cases=[f"case-{i}" for i in range(50)]; shards=[{c for c in cases if stable_shard(c,2)==worker} for worker in (0,1)]
    assert not shards[0]&shards[1] and shards[0]|shards[1]==set(cases)


def test_alpha_zero_parity_gate() -> None:
    logits=np.arange(9,dtype=float); scored=soft_sa_from_logits(logits,list(range(9)))
    hidden=np.asarray([1.,2.],dtype=np.float32)
    result=validate_alpha_zero(clean_logits=np.asarray(scored["class_logits"]),clean_probabilities=np.asarray(scored["class_probabilities"]),clean_soft_sa=scored["soft_sa_image_score"],clean_hard_class=scored["argmax_hard_class"],scored=scored,before=hidden,after=hidden.copy(),diagnostics={"hook_call_count":1,"steering_applied_count":1})
    assert result["passed"]
