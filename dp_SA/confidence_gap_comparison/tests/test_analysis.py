from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from dp_SA.confidence_gap_comparison.analysis import (
    clipped_logit, family_bootstrap_delta, fit_model, predict, prepare_rows,
    reparameterize_mit, run_analysis,
)


def _fixtures() -> tuple[list[dict], list[dict], list[dict]]:
    confidence=[]; train=[]; test=[]
    for split, base, families in (("train", 0, 4), ("test", 100, 3)):
        for family_index in range(families):
            item=str(base+family_index); family=f"{split}-f{family_index}"
            for hard in (0, 1):
                case=f"{item}-{hard}"; condition="conflict_hard" if hard else "conflict_easy"
                c_i=.2+.1*family_index+.02*hard; c_t=.7-.08*family_index
                manifest={"case_id":case,"family_id":family,"item_id":item,"condition":condition,"split":split,
                          "phase0_normalized_answer":"red","soft_sa_image_score":.3+.05*family_index+.01*hard}
                confidence.append({"case_id":case,"family_id":family,"item_id":item,"condition":condition,
                                   "split":split,"fixed_answer":"red","image_fixed_answer_confidence":c_i,
                                   "text_fixed_answer_confidence":c_t})
                (train if split=="train" else test).append(manifest)
    return confidence,train,test


def test_logit_clips_and_is_symmetric() -> None:
    assert np.isfinite(clipped_logit(0.0)) and np.isfinite(clipped_logit(1.0))
    assert clipped_logit(0.0) == pytest.approx(-clipped_logit(1.0))


def test_join_transform_and_no_forbidden_filter() -> None:
    rows,audit=prepare_rows(*_fixtures(),require_frozen_counts=False)
    assert len(rows)==14 and audit["family_overlap_count"]==audit["item_overlap_count"]==0
    assert rows[0]["V_SA"]==pytest.approx(2*rows[0]["soft_SA"]-1)
    assert rows[0]["G_L"]==pytest.approx(rows[0]["L_i"]-rows[0]["L_t"])


def test_scalers_and_models_use_declared_features() -> None:
    rows,_=prepare_rows(*_fixtures(),require_frozen_counts=False); train=[r for r in rows if r["split"]=="train"]
    gap,dual=fit_model("M_G",train),fit_model("M_IT",train)
    assert gap.features==("G_L",) and dual.features==("L_i","L_t")
    assert np.allclose(gap.scaler.mean_,[np.mean([r["G_L"] for r in train])])
    assert predict(gap,train).shape==(len(train),)


def test_gap_mean_is_exact_dual_reparameterization() -> None:
    rows,_=prepare_rows(*_fixtures(),require_frozen_counts=False)
    train=[r for r in rows if r["split"]=="train"]; test=[r for r in rows if r["split"]=="test"]
    dual=fit_model("M_IT",train)
    semantic_prediction,coefficients=reparameterize_mit(dual,train,test)
    assert np.allclose(semantic_prediction,predict(dual,test),atol=1e-14)
    assert set(("beta_G","beta_M","G_L_mean","M_L_mean")) <= set(coefficients)


def test_paired_family_bootstrap_is_deterministic() -> None:
    rows,_=prepare_rows(*_fixtures(),require_frozen_counts=False); test=[r for r in rows if r["split"]=="test"]
    y=np.asarray([r["V_SA"] for r in test]); p1=y+.01; p2=y+.03
    first=family_bootstrap_delta(test,y,p1,p2,repeats=50,seed=42)
    second=family_bootstrap_delta(test,y,p1,p2,repeats=50,seed=42)
    assert first==second and first[2]==50


def test_output_schema(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import dp_SA.confidence_gap_comparison.analysis as module
    confidence,train,test=_fixtures()
    paths=[]
    for name,rows in (("confidence.jsonl",confidence),("train.jsonl",train),("test.jsonl",test)):
        path=tmp_path/name; path.write_text("".join(__import__("json").dumps(row)+"\n" for row in rows)); paths.append(path)
    monkeypatch.setattr(module,"CONFIDENCE_PATH",paths[0]); monkeypatch.setattr(module,"TRAIN_MANIFEST_PATH",paths[1]); monkeypatch.setattr(module,"TEST_MANIFEST_PATH",paths[2])
    original=module.prepare_rows
    monkeypatch.setattr(module,"prepare_rows",lambda a,b,c: original(a,b,c,require_frozen_counts=False))
    result=run_analysis(tmp_path/"output",bootstrap_repeats=20,seed=42)
    assert result["status"]=="complete" and result["valid_bootstrap_repeats"]==20
    for relative in ("tables/model_performance.csv","tables/standardized_coefficients.csv",
                     "tables/paired_delta_r2.csv","artifacts/test_predictions.csv",
                     "figures/fig1_test_predictions.png","figures/fig2_confidence_plane.png"):
        assert (tmp_path/"output"/relative).stat().st_size>0
    with (tmp_path/"output/tables/paired_delta_r2.csv").open(newline="") as handle:
        contrasts=list(csv.DictReader(handle))
    assert contrasts[0]["contrast"]=="M_GM_minus_M_G"
    assert contrasts[1]["contrast"]=="M_GM_minus_M_IT"
