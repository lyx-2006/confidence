from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dp_SA.confidence_steering.trajectory.analyze import component_additivity, persistent_onset
from dp_SA.confidence_steering.trajectory.capture import hidden_key, position_payload
from dp_SA.confidence_steering.trajectory.config import DIRECTIONS, EXPECTED_HASHES, LAYERS, OUTPUT_PARENT, PARENT_FAST_CLEAN, PARENT_FAST_CONFIG, POSITIONS, RAW_EXPRESSION_ATOL, RAW_EXPRESSION_STRICT_ATOL, TABLE_SCHEMAS
from dp_SA.confidence_steering.trajectory.io_utils import array_hash, canonical_forward_key, canonical_hash, load_jsonl, require_output_root, sha256_file, stable_shard
from dp_SA.confidence_steering.trajectory.processor import PROCESSOR_MODE
from dp_SA.confidence_steering.trajectory.run import canonical_merge_equivalence, symmetric_derivative
from dp_SA.confidence_steering.trajectory.train_probes import cluster_bootstrap_draws, choose_alpha, predict_raw, raw_parameters, regression_metrics


def test_four_position_thirteen_layer_schema():
    assert len(POSITIONS)==4 and LAYERS==tuple(range(14,27))
    assert len({hidden_key(p,l) for p in POSITIONS for l in LAYERS})==52


def test_position_payload_and_strict_order():
    located={p:{"processed_index":i,"rendered_index":i+10,"token_id":i+20,"token_text":p} for i,p in enumerate(POSITIONS)}
    assert position_payload(located,"prompt")["causal_order_valid"]
    located["P1_PANL"]["processed_index"]=0
    with pytest.raises(ValueError):position_payload(located,"prompt")


def test_symmetric_hidden_and_readout_derivative():
    assert np.array_equal(symmetric_derivative(np.array([2.,4.]),np.array([0.,2.])),np.array([2.,2.]))


def test_raw_probe_including_intercept():
    rng=np.random.default_rng(4);x=rng.normal(size=(40,7));y=x@np.arange(7)+3
    model=Pipeline([("scale",StandardScaler()),("ridge",Ridge(alpha=.1,solver="lsqr"))]).fit(x,y)
    w,b=raw_parameters(model)
    assert np.max(np.abs(model.predict(x)-predict_raw(w,b,x)))<1e-10
    assert b!=0
    legacy=Pipeline([("scaler",StandardScaler()),("ridge",Ridge(alpha=.1,solver="lsqr"))]).fit(x,y)
    lw,lb=raw_parameters(legacy)
    assert np.max(np.abs(legacy.predict(x)-predict_raw(lw,lb,x)))<1e-10


def test_family_fold_isolation_and_alpha_selection():
    rng=np.random.default_rng(1);x=rng.normal(size=(16,4));y=rng.normal(size=16);folds=np.repeat([1,2,3,4],4)
    alpha,trace=choose_alpha(x,y,folds)
    assert alpha in {r["alpha"] for r in trace} and len(trace)>1
    with pytest.raises(ValueError):choose_alpha(x,y,np.repeat([1,2],8))


def test_unreliable_rule():
    m=regression_metrics(np.arange(6.),np.arange(6.)[::-1])
    assert not (m["r2"]>0 and m["pearson"]>0)


def test_persistent_onset_requires_two_reliable_layers():
    rows=[{"layer":14,"mean":1.,"ci_low":.1,"ci_high":2.,"readout_reliable":True},{"layer":15,"mean":2.,"ci_low":.2,"ci_high":3.,"readout_reliable":True}]
    assert persistent_onset(rows)==14
    rows[1]["ci_low"]=-1
    assert persistent_onset(rows) is None


def test_component_additivity(): assert component_additivity(1.0,.4,.5)==pytest.approx(.1)


def test_shared_bootstrap_draws():
    a=cluster_bootstrap_draws(["a","b","c"],20,42);b=cluster_bootstrap_draws(["c","b","a"],20,42)
    assert a==b and all(len(x)==3 for x in a)


def test_fingerprint_alpha0_dedup_resume_key():
    assert canonical_forward_key("c",None,0)==canonical_forward_key("c","baseline",0)
    assert canonical_forward_key("c",DIRECTIONS[0],.5)!=canonical_forward_key("c",DIRECTIONS[1],.5)


def test_hash_protection_primitive():
    a=np.arange(9,dtype=np.float32);before=array_hash(a);a[0]=7
    assert array_hash(a)!=before and canonical_hash({"x":1})==canonical_hash({"x":1})


def test_output_boundary():
    assert require_output_root(OUTPUT_PARENT/"unit").name=="unit"
    with pytest.raises(ValueError):require_output_root(OUTPUT_PARENT.parent/"forbidden")


def test_single_double_worker_canonical_merge():
    rows=[]
    for i in range(8):rows.append({"case_id":f"c{i}","canonical_key":canonical_forward_key(f"c{i}",None,0),"x":i})
    result=canonical_merge_equivalence(rows)
    assert result["canonical_equal"] and not result["real_two_gpu_inference"]


def test_raw_direction_is_formal_and_smoke_direction():
    assert DIRECTIONS[0]=="confidence_raw" and len(DIRECTIONS)==3


def test_parent_formal_reference_is_explicit_fast():
    assert PROCESSOR_MODE=="explicit_fast"
    assert sha256_file(PARENT_FAST_CLEAN)==EXPECTED_HASHES[PARENT_FAST_CLEAN]
    assert sha256_file(PARENT_FAST_CONFIG)==EXPECTED_HASHES[PARENT_FAST_CONFIG]
    rows=load_jsonl(PARENT_FAST_CLEAN)
    assert len(rows)==100 and len({r["family_id"] for r in rows})==50
    assert all(r["processor_identity"]["is_fast"] for r in rows)


def test_table_and_heatmap_reliability_schema():
    assert "target" in TABLE_SCHEMAS["hidden_transport.csv"]
    assert "readout_reliable" in TABLE_SCHEMAS["trajectory_readouts.csv"]
    assert (len(POSITIONS)*len(DIRECTIONS),len(LAYERS))==(12,13)


def test_raw_error_dual_threshold_policy():
    assert RAW_EXPRESSION_STRICT_ATOL == 1e-5
    assert RAW_EXPRESSION_ATOL == 1e-4
    error=1.0568529949850358e-5
    assert error>RAW_EXPRESSION_STRICT_ATOL and error<RAW_EXPRESSION_ATOL
