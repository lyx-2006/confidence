from __future__ import annotations

from pathlib import Path

from PIL import Image

from dp_SA.confidence_steering.analyze import analyze, build_delta_table, make_wide
from dp_SA.confidence_steering.io_utils import atomic_jsonl, ensure_layout


def _data():
    test=[];trials=[]
    for family,origin,answer in (("f1","follow_text","red"),("f2","follow_image","blue")):
        for hard,condition in ((0,"conflict_easy"),(1,"conflict_hard")):
            case=f"{family}-{hard}"; test.append({"case_id":case,"family_id":family})
            for direction,sign in (("residual_confidence_loao",1),("within_answer_shuffled",0.2)):
                for layer in (8,14):
                    for alpha in (-2.0,0.0,2.0):
                        trials.append({"status":"completed","case_id":case,"item_id":family,"family_id":family,"condition":condition,"answer_origin":origin,"fixed_answer":answer,
                                       "direction":direction,"layer":layer,"alpha":alpha,"delta_soft_sa":sign*alpha*.01,"hard_class_changed":False,"margin_change":sign*alpha*.02,
                                       "alpha_zero_parity":{"passed":True} if alpha==0 else None})
    return test,trials


def test_paired_bootstrap_wide_and_metrics() -> None:
    test,trials=_data(); rows,draws=build_delta_table(trials,test,repeats=20,seed=42)
    assert len(rows)==2*6*2*3 and len(draws)==20
    true=next(r for r in rows if r["direction"]=="residual_confidence_loao" and r["group"]=="all" and r["layer"]==8 and r["alpha"]==2)
    assert true["mean_delta_sa"]==.02 and true["valid_bootstrap_repeats"]==20
    wide=make_wide(rows); assert "L8_a+2" in wide[0] and len(wide)==2*6*4


def test_analyze_writes_three_tables_two_300dpi_figures(tmp_path: Path) -> None:
    root=ensure_layout(tmp_path); test,trials=_data(); atomic_jsonl(root/"artifacts/audits/test_manifest.jsonl",test); atomic_jsonl(root/"artifacts/trials/trials.jsonl",trials)
    result=analyze(output_root=root,smoke=True,repeats=20)
    assert result["tables"]==3 and result["figures"]==2
    for name in ("delta_sa_by_layer.png","symmetric_effect_s10.png"):
        path=root/"figures"/name
        with Image.open(path) as image: assert image.info.get("dpi",(0,))[0]>=299
    assert "LAT→PANL→SA" in (root/"summary.md").read_text()

