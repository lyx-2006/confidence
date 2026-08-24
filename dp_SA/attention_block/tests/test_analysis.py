from __future__ import annotations

import pytest

from dp_SA.attention_block.analyze import _mean_ci, _paired_test, _selection, bh_fdr
from dp_SA.attention_block.run import _margin
from dp_SA.io_utils import load_jsonl


def test_bootstrap_and_bh():
    stats=_mean_ci([1,2,3],repeats=2000,seed=42); assert stats["mean"]==2
    paired=_paired_test([1]*100,repeats=2000,seed=42); assert paired["ci_low"]>0 and paired["p_raw"]<.01
    q=bh_fdr([.01,.04,.03]); assert q[0]<=q[1] and all(0<=x<=1 for x in q)


def test_refine_gate_requires_ci_and_q():
    tests=[]
    for arm in ("joint","delayed"):
        for name in ("panl_cache","panl_gather","jit_all_content"):
            for start in (0,4,8,12,16):
                tests.append({"arm":arm,"phase":"coarse","comparison":name,"ci_low":.1 if arm=="joint" and name=="panl_cache" and start==8 else -.1,
                              "q_bh":.01,"window_start":start})
    selected=_selection(tests,.05)
    assert selected["selected_pairs"]=={"joint":["panl_cache"],"delayed":[]}


def test_fixed_clean_class_margin():
    assert _margin([1, 9, 3, 4, 5, 6, 7, 8, 2], 1) == 4.5


def test_jsonl_repairs_only_trailing_partial_record(tmp_path):
    path=tmp_path/"rows.jsonl"; path.write_text('{"x":1}\n{"x":',encoding="utf-8")
    assert load_jsonl(path)==[{"x":1}]
    assert path.read_text(encoding="utf-8")=="{\"x\":1}\n"
