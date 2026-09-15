from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent;SHORT_ROOT=HERE.parent;REVIEW_ROOT=SHORT_ROOT.parent;REPO=REVIEW_ROOT.parent
for p in (REPO,REVIEW_ROOT,SHORT_ROOT):
    if str(p) not in sys.path:sys.path.insert(0,str(p))

import numpy as np

from AttentionBlock.run import class_margin
from cle_swap.config import LAYERS, LOGIT_PARITY_ATOL, MODEL_PATH, SHORT_CAPTURE
from cle_swap.run import _clean, _patched
from dp_SA.io_utils import atomic_json, atomic_jsonl, canonical_hash, load_jsonl, sha256_file
from dp_SA.soft_score import class_token_ids
from layer_metacognition.model_adapter import resolve_language_modules
from sa_trajectory.panl2cle.contracts import atomic_bf16_npz, load_bf16
from Steering.runtime import load_qwen3_inference

OUTPUT_ROOT=SHORT_ROOT/"output"/"cle_swap"/"within_short_answer_matched_smoke"
SMOKE_PAIR_COUNT=5
FORMAL_PAIR_COUNT=11
REPEATS=2000
SEED=42

def select_pairs(rows, pair_count):
    groups=defaultdict(list)
    for r in rows:
        if r.get("status")=="completed":groups[r["phase0_normalized_answer"]].append(r)
    candidates=[]
    for answer,cell in groups.items():
        lows=sorted((r for r in cell if float(r["soft_sa_image_score"])<.5),key=lambda r:(float(r["soft_sa_image_score"]),r["case_id"]))
        highs=sorted((r for r in cell if float(r["soft_sa_image_score"])>.5),key=lambda r:(-float(r["soft_sa_image_score"]),r["case_id"]))
        pair=next(((lo,hi) for lo in lows for hi in highs if str(lo["item_id"])!=str(hi["item_id"])),None)
        if pair:
            lo,hi=pair;distance=min(.5-float(lo["soft_sa_image_score"]),float(hi["soft_sa_image_score"])-.5)
            candidates.append((distance,answer,lo,hi))
    candidates.sort(key=lambda x:(-x[0],x[1]))
    if pair_count == len(candidates):
        # Solve the small bipartite item-disjoint matching exactly.  Each
        # variable is one answer-matched low/high pair; maximize distance from
        # the midpoint subject to one pair per answer and one use per item.
        from scipy.optimize import LinearConstraint, Bounds, milp
        options=[]; answer_names=[]
        for answer,cell in groups.items():
            lows=sorted((r for r in cell if float(r["soft_sa_image_score"])<.5),key=lambda r:(float(r["soft_sa_image_score"]),r["case_id"]))
            highs=sorted((r for r in cell if float(r["soft_sa_image_score"])>.5),key=lambda r:(-float(r["soft_sa_image_score"]),r["case_id"]))
            if not lows or not highs: continue
            answer_names.append(answer)
            for lo in lows:
                for hi in highs:
                    if str(lo["item_id"]) != str(hi["item_id"]):
                        options.append((answer,lo,hi,min(.5-float(lo["soft_sa_image_score"]),float(hi["soft_sa_image_score"])-.5)))
        items=sorted({str(r["item_id"]) for _,lo,hi,_ in options for r in (lo,hi)})
        ai={a:i for i,a in enumerate(answer_names)};ii={x:i+len(answer_names) for i,x in enumerate(items)}
        A=np.zeros((len(answer_names)+len(items),len(options)),dtype=float)
        for j,(a,lo,hi,_) in enumerate(options):
            A[ai[a],j]=1;A[ii[str(lo["item_id"])],j]=1;A[ii[str(hi["item_id"])],j]=1
        result=milp(c=np.asarray([-o[3]+j*1e-10 for j,o in enumerate(options)]),integrality=np.ones(len(options)),bounds=Bounds(0,1),constraints=LinearConstraint(A,lb=np.full(A.shape[0],-np.inf),ub=np.r_[np.ones(len(answer_names)),np.ones(len(items))]),options={"time_limit":30})
        if not result.success: raise RuntimeError(f"Answer-matched item matching failed: {result.message}")
        candidates=[(d,a,lo,hi) for value,(a,lo,hi,d) in zip(result.x,options) if value>.5]
        candidates.sort(key=lambda x:(-x[0],x[1]))
    selected=[];used=set()
    for distance,answer,lo,hi in candidates:
        items={str(lo["item_id"]),str(hi["item_id"])}
        if items&used:continue
        selected.append({"pair_id":len(selected)+1,"answer":answer,"distance_floor":distance,
                         "text_case_id":lo["case_id"],"text_item_id":str(lo["item_id"]),"text_clean_sa":float(lo["soft_sa_image_score"]),
                         "image_case_id":hi["case_id"],"image_item_id":str(hi["item_id"]),"image_clean_sa":float(hi["soft_sa_image_score"])})
        used|=items
        if len(selected)==pair_count:break
    if len(selected)!=pair_count or len(used)!=2*pair_count:raise RuntimeError(f"Could not select {pair_count} item-disjoint answer-matched pairs")
    return selected

def _csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True);fields=sorted({k for r in rows for k in r});fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
    try:
        with os.fdopen(fd,"w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    except Exception:
        try:os.unlink(tmp)
        except FileNotFoundError:pass
        raise

def run(output_root=OUTPUT_ROOT, pair_count=SMOKE_PAIR_COUNT):
    root=Path(output_root);all_rows={r["case_id"]:r for r in load_jsonl(SHORT_CAPTURE/"results.jsonl") if r.get("status")=="completed"}
    pairs=select_pairs(list(all_rows.values()),pair_count);atomic_jsonl(root/"manifest.jsonl",pairs)
    config={"experiment":"within_short_answer_matched_cle_swap_formal" if pair_count==FORMAL_PAIR_COUNT else "within_short_answer_matched_cle_swap_smoke","pair_count":pair_count,"layers":list(LAYERS),"source_capture_sha256":sha256_file(SHORT_CAPTURE/"results.jsonl")}
    config["fingerprint"]=canonical_hash(config);atomic_json(root/"config.json",config)
    runtime=load_qwen3_inference(MODEL_PATH);modules=resolve_language_modules(runtime.model);ids=class_token_ids(runtime.processor.tokenizer)
    trials=[];self_audits=[];physical=0
    for pair in pairs:
        clean={};contexts={};hidden={}
        for side,key in (("text","text_case_id"),("image","image_case_id")):
            case=pair[key];inputs,located,pos,logits,score,hiddens,diag=_clean(runtime,modules,"short",all_rows[case],ids);physical+=1
            err=max(abs(float(a)-float(b)) for a,b in zip(score["class_logits"],all_rows[case]["class_logits"]))
            if err>LOGIT_PARITY_ATOL or abs(float(score["soft_sa_image_score"])-float(all_rows[case]["soft_sa_image_score"]))>LOGIT_PARITY_ATOL:raise RuntimeError(f"Clean parity failed: {case}")
            path=root/"hidden"/f"{case}__CLE_bf16.npz";atomic_bf16_npz(path,{f"CLE_L{x}":hiddens[x] for x in LAYERS})
            clean[side]=(logits,score);contexts[side]=(inputs,located,pos);hidden[side]=path
        # Different cases can have different sequence lengths and therefore
        # different processed CLE indices.  Each source is captured at its own
        # semantic CLE and injected at the recipient's independently aligned CLE.
        if any(contexts[side][2]["CLE"] >= contexts[side][2]["SAC"] for side in ("text", "image")):
            raise RuntimeError(f"Within-short CLE/SAC causal order failed: pair {pair['pair_id']}")
        for layer in LAYERS:
            if pair["pair_id"]==1:
                for side in ("text","image"):
                    source=load_bf16(hidden[side],f"CLE_L{layer}");logits,score,hook=_patched(runtime,modules,"short",contexts[side][0],contexts[side][2],ids,layer,source);physical+=1
                    base_logits,base_score=clean[side];errors={"logit":float(np.max(np.abs(logits.numpy()-base_logits.numpy()))),"probability":float(np.max(np.abs(np.asarray(score["class_probabilities"])-np.asarray(base_score["class_probabilities"])))),"sa":abs(float(score["soft_sa_image_score"])-float(base_score["soft_sa_image_score"]))}
                    audit={"pair_id":pair["pair_id"],"side":side,"layer":layer,**errors,"hard_equal":score["argmax_hard_class"]==base_score["argmax_hard_class"],"hook":hook.diagnostics()};audit["passed"]=max(errors.values())<=LOGIT_PARITY_ATOL and audit["hard_equal"]
                    if not audit["passed"]:raise RuntimeError(f"Self swap failed: {audit}")
                    self_audits.append(audit)
            for direction,target,donor,expected_sign in (("image_to_text","text","image",1),("text_to_image","image","text",-1)):
                source=load_bf16(hidden[donor],f"CLE_L{layer}");logits,score,hook=_patched(runtime,modules,"short",contexts[target][0],contexts[target][2],ids,layer,source);physical+=1
                base_logits,base_score=clean[target];delta=float(score["soft_sa_image_score"])-float(base_score["soft_sa_image_score"]);raw=int(base_score["argmax_hard_class"])
                trials.append({"status":"completed","pair_id":pair["pair_id"],"answer":pair["answer"],"layer":layer,"direction":direction,
                               "target_case_id":pair[f"{target}_case_id"],"donor_case_id":pair[f"{donor}_case_id"],"target_clean_sa":float(base_score["soft_sa_image_score"]),"donor_clean_sa":float(clean[donor][1]["soft_sa_image_score"]),"swapped_sa":float(score["soft_sa_image_score"]),"delta_sa":delta,"semantic_movement":delta*expected_sign,"direction_success":int(delta*expected_sign>0),
                               "logit_change_diff":class_margin(base_logits.tolist(),raw)-class_margin(logits.tolist(),raw),"token_changed":int(raw!=int(score["argmax_hard_class"])),"clean_class":raw,"swapped_class":int(score["argmax_hard_class"]),"clean_logits":base_logits.tolist(),"swapped_logits":logits.tolist(),"hook":hook.diagnostics()})
    atomic_jsonl(root/"trials.jsonl",trials);atomic_json(root/"self_swap_gate.json",{"status":"passed","checks":len(self_audits),"audits":self_audits})
    summary=[];rng=np.random.default_rng(SEED)
    for direction in ("image_to_text","text_to_image"):
        for layer in LAYERS:
            cell=[r for r in trials if r["direction"]==direction and r["layer"]==layer]
            for metric in ("delta_sa","semantic_movement","direction_success","logit_change_diff","token_changed"):
                vals=np.asarray([r[metric] for r in cell],float);boot=vals[rng.integers(0,len(vals),size=(REPEATS,len(vals)))].mean(axis=1);lo,hi=np.quantile(boot,[.025,.975])
                summary.append({"direction":direction,"layer":layer,"metric":metric,"mean":float(vals.mean()),"ci95_low":float(lo),"ci95_high":float(hi),"pair_count":len(vals),"bootstrap_repeats":REPEATS})
    _csv(root/"summary.csv",summary);_plot(root,summary)
    result={"status":"complete","pairs":pair_count,"clean_forwards":2*pair_count,"cross_swap_trials":pair_count*len(LAYERS)*2,"self_swap_checks":18,"physical_forwards":physical,"layers":list(LAYERS)};atomic_json(root/"completion.json",result);return result

def _plot(root,rows):
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,4.5));labels={"image_to_text":"Image-side CLE → text-side case","text_to_image":"Text-side CLE → image-side case"}
    for ax,metric,title in zip(axes,("delta_sa","semantic_movement","token_changed"),("Canonical ΔSA","Movement toward donor semantics","Token change rate")):
        for d in labels:
            p=sorted([r for r in rows if r["direction"]==d and r["metric"]==metric],key=lambda r:r["layer"]);x=[r["layer"] for r in p];y=np.asarray([r["mean"] for r in p]);err=np.asarray([[r["mean"]-r["ci95_low"] for r in p],[r["ci95_high"]-r["mean"] for r in p]])
            ax.errorbar(x,y,yerr=err,marker="o",capsize=3,label=labels[d])
        ax.axhline(0,color="black",lw=.8);ax.set_xticks(LAYERS);ax.set_xlabel("CLE swap layer");ax.set_title(title);ax.grid(axis="y",alpha=.2)
    axes[-1].legend(frameon=False,fontsize=8);fig.tight_layout();fig.savefig(root/"within_short_swap.png",dpi=220);plt.close(fig)

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--output-root",type=Path,default=None);p.add_argument("--formal",action="store_true");a=p.parse_args(argv)
    root=a.output_root or (SHORT_ROOT/"output"/"cle_swap"/("within_short_answer_matched_formal" if a.formal else "within_short_answer_matched_smoke"))
    print(json.dumps(run(root,pair_count=FORMAL_PAIR_COUNT if a.formal else SMOKE_PAIR_COUNT),ensure_ascii=False));return 0
if __name__=="__main__":raise SystemExit(main())
