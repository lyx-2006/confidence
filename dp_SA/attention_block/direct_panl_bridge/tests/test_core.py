from __future__ import annotations

import json

import pytest

from dp_SA.attention_block.direct_panl_bridge.core import (
    CONDITIONS, base_pairs, edges_for_condition, effects, one_sided_sign_flip,
    restored_pairs, validate_symmetry,
)
from dp_SA.attention_block.direct_panl_bridge.run import ensure_config


def spans():
    return {"ANSWER":[2,4], "PANL":7, "PANL_PLUS_1":8, "SAC":10, "sequence_length":11}


def test_conditions_are_exact_base_minus_requested_restorations():
    value=spans(); base=base_pairs(value)
    for condition in CONDITIONS:
        assert set(edges_for_condition(value,condition).pairs) == base-restored_pairs(value,condition)
    assert restored_pairs(value,"C11") == {(7,2),(7,3),(10,7)}
    assert restored_pairs(value,"CTRL") == {(8,2),(8,3),(10,8)}
    validate_symmetry(value)


def test_c11_and_control_keep_sac_to_answer_blocked_and_only_restore_two_groups():
    value=spans()
    for condition in ("C11","CTRL"):
        blocked=set(edges_for_condition(value,condition).pairs)
        assert {(10,2),(10,3)} <= blocked
        assert base_pairs(value)-blocked == restored_pairs(value,condition)


def test_interaction_bridge_and_matched_formulas():
    assert effects(1,2,3,8,4) == (4,7,4)


def test_one_sided_sign_flip_is_deterministic_and_directional():
    positive=one_sided_sign_flip([1.0]*20,seed=7,repeats=2000)
    assert positive < .01
    assert positive == one_sided_sign_flip([1.0]*20,seed=7,repeats=2000)
    assert one_sided_sign_flip([-1.0]*20,seed=7,repeats=2000) > .99


def test_resume_accepts_exact_fingerprint_and_rejects_change(tmp_path):
    path=tmp_path/"run_config.json"; config={"fingerprint":"abc","value":1}
    ensure_config(path,config,resume=False)
    ensure_config(path,config,resume=True)
    with pytest.raises(FileExistsError): ensure_config(path,config,resume=False)
    with pytest.raises(ValueError,match="Fingerprint changed"):
        ensure_config(path,{"fingerprint":"changed"},resume=True)
    assert json.loads(path.read_text()) == config
