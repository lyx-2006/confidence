from __future__ import annotations

import math

import numpy as np
import pytest

from dp_SA.confidence_steering.core import (
    answer_origin, build_vectors, nuisance_matrix, nuisance_pipeline, oof_residualize,
    tail_assignments, validate_frozen_design,
)


def test_answer_origin_keeps_neither_as_explicit_category() -> None:
    assert answer_origin({"answer_matches_text":True,"answer_matches_image":False})=="follow_text"
    assert answer_origin({"answer_matches_text":False,"answer_matches_image":True})=="follow_image"
    assert answer_origin({"answer_matches_text":False,"answer_matches_image":False})=="neither_match"


def _residual_rows() -> list[dict]:
    rows=[]
    for family in range(10):
        for hard in (0,1):
            rows.append({"case_id":f"c{family}-{hard}","family_id":f"f{family}","outer_fold":family%5,"D_t":float(family),"D_i":float(10-family),
                         "Hard":hard,"prior_bin":str(family%3),"answer_origin":("follow_text","follow_image","neither_match")[family%3],
                         "fixed_answer_color":("red","blue")[family%2],"G_L":float(family-hard)})
    return rows


def test_family_oof_residual_is_complete_and_not_in_sample() -> None:
    rows,models=oof_residualize(_residual_rows())
    assert len(rows)==20 and len(models)==5 and len({r["case_id"] for r in rows})==20
    assert all(r["R_C"]==pytest.approx(r["G_L"]-r["predicted_G_L_oof"]) for r in rows)
    assert nuisance_pipeline().named_steps["ridge"].alpha==1.0


def test_sa_and_intervention_fields_cannot_enter_nuisance_matrix() -> None:
    rows=_residual_rows(); first=nuisance_matrix(rows)
    for row in rows:
        row.update({"soft_sa_image_score":999,"argmax_hard_class":8,"panl_probe_sa":-999,"delta_soft_sa":123})
    assert np.array_equal(first,nuisance_matrix(rows))


def test_leakage_gate_checks_all_six_identifiers() -> None:
    def row(split,item,family,condition,image):
        return {"case_id":f"{item}-{condition}","item_id":item,"family_id":family,"condition":condition,"image_sha256":image,"prior_index":0,"outer_fold":0,"split":split}
    train=[row("train","a","fa","conflict_easy","ia")]
    test=[row("test",str(i),f"f{i}",c,f"i{i}-{c}") for i in range(50) for c in ("conflict_easy","conflict_hard")]
    # This fixture only reaches the overlap gate after satisfying formal cardinality.
    train=train*1112
    with pytest.raises(ValueError): validate_frozen_design(train,test)


def _cells() -> tuple[list[dict],dict[int,dict[str,np.ndarray]]]:
    cells=[]; arrays={8:{},14:{}}
    for ai,answer in enumerate(("black","brown","cyan","gray","green","orange","pink","purple","white")):
        for fi in range(8):
            key=f"{answer}_{fi}"; cells.append({"array_key":key,"family_id":f"f{ai}_{fi}","fixed_answer_color":answer,"mean_residual":float(fi),"case_ids":[key],"record_count":1})
            arrays[8][key]=np.asarray([1.0+ai,fi+1.0],dtype=np.float32); arrays[14][key]=arrays[8][key]*2
    return cells,arrays


def test_tail_shuffle_loao_and_shared_norm() -> None:
    cells,arrays=_cells(); assignments,audit=tail_assignments(cells)
    assert all(r["tail_count"]==2 for r in audit if r["eligible"])
    for answer,mapping in assignments.items():
        assert list(mapping.values()).count("high")==4  # true + shuffled
        assert list(mapping.values()).count("low")==4
    vectors,meta=build_vectors(cells,arrays,assignments,audit,["black","blue"],layers=(8,14))
    assert len(meta)==8
    for layer in (8,14):
        for recipient in ("black","blue"):
            rows=[r for r in meta if r["layer"]==layer and r["recipient_answer"]==recipient]
            assert len(rows)==2 and rows[0]["target_norm"]==pytest.approx(rows[1]["target_norm"])
            assert all(recipient not in r["included_answers"] and len(r["included_answers"])>=8 for r in rows)
            assert all(np.linalg.norm(vectors[layer][r["scaled_key"]])==pytest.approx(r["target_norm"],rel=2e-6) for r in rows)


def test_loao_rejects_fewer_than_eight_other_answers() -> None:
    cells,arrays=_cells(); cells=[c for c in cells if c["fixed_answer_color"]!="white"]
    arrays={l:{k:v for k,v in payload.items() if not k.startswith("white_")} for l,payload in arrays.items()}
    assignments,audit=tail_assignments(cells)
    with pytest.raises(ValueError,match="LOAO gate"): build_vectors(cells,arrays,assignments,audit,["black"],layers=(8,))
