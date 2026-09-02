from __future__ import annotations

import numpy as np
import pytest

from dp_SA.unimodal_logit_confidence.explanatory_comparison import (
    clipped_logit,
    cluster_bootstrap_indices,
    design_matrices,
    prepare_analysis_rows,
)


def test_clipped_logit_is_finite_and_symmetric() -> None:
    assert np.isfinite(clipped_logit(0.0))
    assert np.isfinite(clipped_logit(1.0))
    assert clipped_logit(0.0) == pytest.approx(-clipped_logit(1.0))


def _fixture_rows() -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    confidence=[]; scores=[]; sa=[]; train_manifest=[]; test_manifest=[]
    for split, item, family in (("train", "1", "f1"), ("test", "2", "f2")):
        for condition in ("conflict_easy", "conflict_hard"):
            case=f"{item}-{condition}"; image_hash=f"h-{item}-{condition}"
            confidence.append({"case_id":case,"item_id":item,"family_id":family,"split":split,"condition":condition,"fixed_answer":"red","text_score_unique_key":[item,0],"image_score_unique_key":[item,condition,image_hash],"text_fixed_answer_confidence":.2,"image_fixed_answer_confidence":.8})
            sa.append({"case_id":case,"phase0_normalized_answer":"red","Hard":int(condition=="conflict_hard"),"final_sa":.4,"panl_l14_oof_sa_prediction":.5})
            (train_manifest if split=="train" else test_manifest).append({"case_id":case})
            scores.append({"stable_key":["image",item,condition,image_hash],"entropy_difficulty":20.0})
        scores.append({"stable_key":["text",item,0],"entropy_difficulty":10.0})
    return confidence,scores,sa,train_manifest,test_manifest


def test_join_uses_unique_difficulty_and_record_confidence() -> None:
    args=_fixture_rows()
    # Production requires 100/50; extend the fixture to that exact locked shape.
    confidence,scores,sa,train_manifest,test_manifest=args
    base_test=confidence[-2:]; base_sa=sa[-2:]; confidence=confidence[:-2];sa=sa[:-2];test_manifest=[]
    scores=[row for row in scores if not (len(row["stable_key"])>1 and row["stable_key"][1]=="2")]
    for n in range(50):
        item=str(100+n); family=f"tf{n}"
        for source,condition in zip(base_test,("conflict_easy","conflict_hard"),strict=True):
            case=f"{item}-{condition}"; image_hash=f"h-{item}-{condition}"
            confidence.append({**source,"case_id":case,"item_id":item,"family_id":family,"condition":condition,"text_score_unique_key":[item,0],"image_score_unique_key":[item,condition,image_hash]})
            sa.append({**base_sa[0],"case_id":case,"Hard":int(condition=="conflict_hard")});test_manifest.append({"case_id":case})
            scores.append({"stable_key":["image",item,condition,image_hash],"entropy_difficulty":20.0})
        scores.append({"stable_key":["text",item,0],"entropy_difficulty":10.0})
    rows,audit=prepare_analysis_rows(confidence,scores,sa,train_manifest,test_manifest)
    assert audit["item_overlap_count"]==audit["family_overlap_count"]==0
    assert len([row for row in rows if row["split"]=="test"])==100
    assert rows[0]["D_t"]==10.0 and rows[0]["L_i"]==pytest.approx(clipped_logit(.8))


def test_item_or_family_leakage_is_rejected() -> None:
    confidence,scores,sa,train_manifest,test_manifest=_fixture_rows()
    # Bypass the locked-size check only after forcing a cross-split item collision.
    confidence[2]["item_id"]="1"
    with pytest.raises(ValueError,match="item leakage"):
        prepare_analysis_rows(confidence,scores,sa,train_manifest,test_manifest)


def test_scaler_is_fit_on_train_and_hard_is_not_scaled() -> None:
    train=[{"D_t":0.0,"Hard":0},{"D_t":2.0,"Hard":1}]
    test=[{"D_t":101.0,"Hard":1}]
    X_train,X_test,params=design_matrices(train,test,("D_t","Hard"))
    assert params["D_t"]["train_mean"]==1.0
    assert X_train[:,0].tolist()==[-1.0,1.0]
    assert X_test[0,0]==100.0 and X_test[0,1]==1.0
    assert params["Hard"]["standardized"] is False


def test_cluster_bootstrap_keeps_family_records_together_and_is_deterministic() -> None:
    rows=[{"family_id":"a"},{"family_id":"a"},{"family_id":"b"},{"family_id":"b"}]
    first,fp1=cluster_bootstrap_indices(rows,10,42);second,fp2=cluster_bootstrap_indices(rows,10,42)
    assert fp1==fp2
    assert all(len(draw)==4 for draw in first)
    assert all(np.array_equal(a,b) for a,b in zip(first,second,strict=True))
