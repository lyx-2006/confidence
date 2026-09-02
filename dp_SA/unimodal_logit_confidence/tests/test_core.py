from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dp_SA.unimodal_logit_confidence.analyze import plot_probe_metrics
from dp_SA.unimodal_logit_confidence.build_split import build_split
from dp_SA.unimodal_logit_confidence.fit_temperature import join_phase1, search_temperature, temperature_grid
from dp_SA.unimodal_logit_confidence.io_utils import atomic_jsonl, ensure_layout, stable_shard, validate_fingerprint
from dp_SA.unimodal_logit_confidence.metrics import family_bootstrap, regression_values, restricted_probabilities, score_metrics
from dp_SA.unimodal_logit_confidence.score_unimodal import candidate_suffix_ids, unique_specs
from dp_SA.unimodal_logit_confidence.train_probe import HiddenResolver, fit_probe


class Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: return [ord(value) for value in text]


def test_single_and_multi_token_extension() -> None:
    assert candidate_suffix_ids(Tokenizer(),"base:","r")==[ord("r")]
    assert candidate_suffix_ids(Tokenizer(),"base:","red")==list(map(ord,"red"))
    scores=np.arange(12,dtype=float); p=restricted_probabilities(scores); assert p.sum()==pytest.approx(1); assert int(np.argmax(p))==11


def test_chosen_and_fixed_confidence_are_distinct() -> None:
    names=[f"c{i}" for i in range(12)]; result=score_metrics(names,list(range(12)),"c0")
    assert result["chosen_answer"]=="c11" and result["chosen_confidence"] != result["probabilities"]["c0"]


def _score(modality: str, key: list, probabilities: dict[str,float], chosen: str) -> dict:
    return {"modality":modality,"unique_key":key,"answer_classes":list(probabilities),"calibrated_probabilities":probabilities,"chosen_answer":chosen,"chosen_confidence":probabilities[chosen],"temperature":2.0,"temperature_fingerprint":modality+"fp"}


def test_record_level_join_preserves_one_to_many_fixed_answers() -> None:
    colors=["red","blue",*[f"c{i}" for i in range(10)]]; pt={c:.01 for c in colors}; pi={c:.01 for c in colors}; pt.update(red=.7,blue=.2); pi.update(red=.1,blue=.8)
    scores=[_score("text",["1",0],pt,"red"),_score("image",["1","conflict_easy","h1"],pi,"blue"),_score("image",["1","conflict_hard","h2"],pi,"blue")]
    records=[]
    for condition,image_hash,answer in (("conflict_easy","h1","red"),("conflict_hard","h2","blue")):
        records.append({"case_id":condition,"item_id":"1","family_id":"f1","prior_index":0,"condition":condition,"image_sha256":image_hash,"split":"test","phase0_normalized_answer":answer})
    joined=join_phase1(records,scores,"scorefp")
    assert joined[0]["text_chosen_confidence"]==joined[1]["text_chosen_confidence"]
    assert joined[0]["text_fixed_answer_confidence"] != joined[1]["text_fixed_answer_confidence"]
    assert "fixed_answer" not in scores[0] and joined[0]["G_C"]==pytest.approx(.1-.7)


def test_temperature_grid_search_and_tie_break() -> None:
    assert temperature_grid()[0]==pytest.approx(.05) and temperature_grid()[-1]==pytest.approx(100) and 1.0 in temperature_grid()
    names=[f"c{i}" for i in range(12)]; rows=[]
    for index in range(8): rows.append({"answer_classes":names,"raw_candidate_scores":{name:(2 if name==("c0" if index<4 else "c1") else 0) for name in names},"target_answer":"c0" if index<4 else "c2"})
    best,nll,trace=search_temperature(rows); assert .05<=best["temperature"]<=100 and .05<=nll["temperature"]<=100 and len(trace)>=4096


def test_actual_strict_split_and_calibration(tmp_path: Path) -> None:
    root=ensure_layout(tmp_path/"run",resume=False); audit=build_split(root)
    assert audit["test"]["selected_record_count"]==100 and audit["test"]["selected_family_count"]==50
    assert set(audit["test"]["cell_counts"].values())=={25}
    assert audit["probe_train"]=={"record_count":1112,"family_count":128,"item_count":128,"image_hash_count":256}
    assert audit["test"]["excluded_variant_count"]==444 and not any(audit["overlaps"].values())
    assert audit["calibration"]["text_count"]==audit["calibration"]["image_count"]==200
    assert audit["calibration"]["image_conditions"]=={"conflict_easy":100,"conflict_hard":100}


def test_stable_sharding_and_fingerprint(tmp_path: Path) -> None:
    keys=[("text",str(i),i%3) for i in range(50)]; assert [stable_shard(k,2) for k in keys]==[stable_shard(k,2) for k in keys]
    assert all(stable_shard(k,1)==0 for k in keys)
    path=tmp_path/"config.json"; fp=validate_fingerprint(path,{"x":1},resume=False); assert validate_fingerprint(path,{"x":1},resume=True)==fp
    with pytest.raises(ValueError): validate_fingerprint(path,{"x":2},resume=True)


def test_hidden_resolver_reuse_and_delta(tmp_path: Path) -> None:
    root=tmp_path; (root/"source").mkdir(); (root/"delta").mkdir(); np.savez(root/"source/a.npz",P1_AC__L6=np.ones(3,dtype=np.float16)); np.savez(root/"delta/a.npz",P1_AC__L8=np.full(3,2,dtype=np.float16))
    atomic_jsonl(root/"confidence_probe/artifacts/hidden/reuse_manifest.jsonl",[{"case_id":"a","cell_sources":{"P1_AC__L6":{"path":str(root/"source/a.npz")}}}]); atomic_jsonl(root/"confidence_probe/artifacts/hidden/capture_results.jsonl",[{"case_id":"a","delta_file":"delta/a.npz","delta_keys":["P1_AC__L8"]}])
    resolver=HiddenResolver(root); assert resolver.load("a","P1_AC__L6").tolist()==[1,1,1]; assert resolver.load("a","P1_AC__L8").tolist()==[2,2,2]


def test_scaler_is_train_only_and_metrics_bootstrap() -> None:
    X=np.asarray([[0.],[1.],[2.]]); y=np.asarray([0.,1.,2.]); model=fit_probe(X,y); assert model.named_steps["scaler"].mean_[0]==pytest.approx(1)
    assert model.predict([[100.]])[0]>2
    rows=[{"family_id":f"f{i}","true":float(i),"predicted":float(i)+.1} for i in range(10) for _ in range(2)]; result=family_bootstrap(rows,repeats=50,seed=42); assert result["r2"]["valid"]==50
    assert regression_values([0,1,2],[0,1,2])["r2"]==pytest.approx(1)


def test_three_figure_schema(tmp_path: Path) -> None:
    (tmp_path/"confidence_probe/figures").mkdir(parents=True); rows=[]
    targets=("text_chosen_confidence","image_chosen_confidence","text_fixed_answer_confidence","image_fixed_answer_confidence")
    positions=("P1_AC","P1_LAT","P1_PANL","P1_PANL_PLUS_1"); layers=(6,8,10,12,14,16,18,22,24,26)
    for target in targets:
        for position in positions:
            for layer in layers:
                row={"target":target,"position":position,"layer":layer}
                for metric in ("r2","spearman","pearson"): row[metric]=.1; row[f"{metric}_ci_low"]=0.; row[f"{metric}_ci_high"]=.2
                rows.append(row)
    files=plot_probe_metrics(tmp_path,rows); assert sorted(Path(f).name for f in files)==["probe_pearson.png","probe_r2.png","probe_spearman.png"]
