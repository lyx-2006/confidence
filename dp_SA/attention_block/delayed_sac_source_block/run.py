from __future__ import annotations

import argparse,json,time
from pathlib import Path
from typing import Any,Sequence

import torch

from confidence_test.runtime_imports import load_runtime
from dp_SA.attention_block.config import INFERENCE_PATH,LOGIT_PARITY_TOLERANCE,MODEL_PATH,ROW_SUM_TOLERANCE,SOFT_PARITY_TOLERANCE
from dp_SA.attention_block.masking import AttentionBlockContext,AttentionEdges
from dp_SA.attention_block.run import _forward,_margin
from dp_SA.attention_block.sources import prepare_case
from dp_SA.attention_block.spans import locate_spans
from dp_SA.io_utils import append_jsonl,atomic_json,atomic_jsonl,canonical_hash,load_jsonl,sha256_file
from dp_SA.soft_score import class_token_ids
from layer_metacognition.model_adapter import resolve_language_modules

ROOT=Path(__file__).resolve().parents[3]
DEFAULT_BASE=ROOT/"dp_SA/attention_block/outputs/formal_both_seed42_w12_20260823T093446Z"
DEFAULT_OUTPUT_PARENT=Path(__file__).resolve().parent/"outputs"
WINDOWS=((0,11),(4,15),(8,19),(12,23),(16,27))
CONDITIONS=("sac_to_evidence","sac_to_answer","sac_to_evidence_answer","sac_to_panl","sac_to_panl_plus_1")
REUSED_CONDITIONS=tuple(c for c in CONDITIONS if c!="sac_to_evidence_answer")
NEW_CONDITION="sac_to_evidence_answer"


def evidence_answer_edges(spans:dict[str,Any])->AttentionEdges:
    sources=sorted(set(spans["EVIDENCE"])|set(range(int(spans["ANSWER"][0]),int(spans["ANSWER"][1]))))
    return AttentionEdges.from_sets([int(spans["SAC"])],sources)


def _selection(base:Path,smoke:bool)->list[dict[str,Any]]:
    rows=json.loads((base/"delayed_case_manifest.json").read_text())
    counts={side:sum(r["test_side"]==side for r in rows) for side in ("image_side","text_side")}
    if len(rows)!=100 or counts!={"image_side":50,"text_side":50}:raise ValueError(f"Frozen delayed manifest invalid: {counts}")
    return [r for side in counts for r in [x for x in rows if x["test_side"]==side][:2]] if smoke else rows


def _config(base:Path,selection:list[dict[str,Any]],smoke:bool)->dict[str,Any]:
    package=Path(__file__).resolve().parent;files=[package/x for x in ("run.py","analyze.py","run_pipeline.py")]+[package.parent/"masking.py",package.parent/"spans.py"]
    value={"format_version":1,"experiment":"delayed_sac_source_block","base_output":str(base.resolve()),
           "base_run_config_sha256":sha256_file(base/"run_config.json"),"base_manifest_sha256":sha256_file(base/"delayed_case_manifest.json"),
           "base_clean_sha256":sha256_file(base/"clean_baselines.jsonl"),"base_spans_sha256":sha256_file(base/"delayed_token_spans.jsonl"),
           "base_blocked_sha256":sha256_file(base/"blocked_results.jsonl"),"selection_hash":canonical_hash(selection),
           "prompt_hash":canonical_hash([r["phase1_prompt_hash"] for r in selection]),"model_path":str(MODEL_PATH.resolve()),
           "model_config_sha256":sha256_file(MODEL_PATH/"config.json"),"processor_config_sha256":sha256_file(MODEL_PATH/"preprocessor_config.json"),
           "windows":WINDOWS,"conditions":CONDITIONS,"reused_conditions":REUSED_CONDITIONS,"new_condition":NEW_CONDITION,
           "seed":42,"bootstrap_repeats":2000,"smoke":smoke,"implementation_sha256":{str(p):sha256_file(p) for p in files}}
    value["fingerprint"]=canonical_hash(value);return value


def ensure_config(path:Path,config:dict[str,Any],*,resume:bool)->None:
    if path.exists():
        if json.loads(path.read_text()).get("fingerprint")!=config["fingerprint"]:raise ValueError("Fingerprint changed; refusing resume")
        if not resume:raise FileExistsError(f"{path.parent} exists; use --resume")
    else:atomic_json(path,config)


def _seed_reused(output:Path,base:Path,selection:list[dict[str,Any]])->int:
    selected={r["case_id"] for r in selection};source=[]
    for row in load_jsonl(base/"blocked_results.jsonl"):
        if (row.get("arm")=="delayed" and row.get("phase")=="coarse" and row.get("case_id") in selected and
                row.get("condition") in REUSED_CONDITIONS and (int(row["window_start"]),int(row["window_end"])) in WINDOWS):
            source.append({**row,"provenance":"reused_frozen_formal","source_output":str(base.resolve())})
    expected=len(selection)*len(REUSED_CONDITIONS)*len(WINDOWS)
    if len(source)!=expected or len({(r["case_id"],r["condition"],r["window_start"]) for r in source})!=expected:raise ValueError(f"Reusable cells incomplete: {len(source)}/{expected}")
    atomic_jsonl(output/"blocked_results.jsonl",source);return len(source)


def run(output:Path,base:Path=DEFAULT_BASE,*,smoke:bool=False,resume:bool=False)->dict[str,Any]:
    output=output.resolve();output.mkdir(parents=True,exist_ok=True);selection=_selection(base,smoke);config=_config(base,selection,smoke);cp=output/"run_config.json";existed=cp.exists();ensure_config(cp,config,resume=resume)
    if not existed:
        atomic_jsonl(output/"selection_manifest.jsonl",selection);reused=_seed_reused(output,base,selection)
    else:reused=sum(1 for r in load_jsonl(output/"blocked_results.jsonl") if r.get("provenance")=="reused_frozen_formal")
    clean={r["case_id"]:r for r in load_jsonl(base/"clean_baselines.jsonl") if r["arm"]=="delayed"};frozen={r["case_id"]:r for r in load_jsonl(base/"delayed_token_spans.jsonl")}
    blocked_path=output/"blocked_results.jsonl";parity_path=output/"clean_parity.jsonl";spans_path=output/"token_spans.jsonl";failure_path=output/"failures.jsonl"
    for path in (parity_path,spans_path,failure_path):path.touch(exist_ok=True)
    all_rows=load_jsonl(blocked_path);completed={(r["case_id"],r["condition"],int(r["window_start"])) for r in all_rows};parity={r["case_id"] for r in load_jsonl(parity_path)};saved={r["case_id"] for r in load_jsonl(spans_path)}
    runtime=load_runtime(INFERENCE_PATH);inference=runtime.QwenVLInference(str(MODEL_PATH));modules=resolve_language_modules(inference.model);tokenizer=getattr(inference.processor,"tokenizer",inference.processor);ids=class_token_ids(tokenizer)
    if getattr(inference.model.config,"_attn_implementation",None)!="eager" or modules.num_hidden_layers!=28:raise RuntimeError("Expected eager 28-layer model")
    started=time.time();expected=len(selection)*(1+len(WINDOWS))
    try:
        for row in selection:
            try:
                rendered,inputs=prepare_case(inference,row);spans=locate_spans(tokenizer,rendered,inputs,row);f={k:v for k,v in frozen[row["case_id"]].items() if k not in {"arm","case_id"}}
                if canonical_hash(spans)!=canonical_hash(f):raise RuntimeError(f"Frozen spans changed: {row['case_id']}")
                if row["case_id"] not in saved:append_jsonl(spans_path,{"case_id":row["case_id"],"item_id":row["item_id"],"test_side":row["test_side"],**spans});saved.add(row["case_id"])
                baseline=clean[row["case_id"]];target=int(baseline["clean_class"])
                if row["case_id"] not in parity:
                    logits,score=_forward(inference.model,inputs,spans["SAC"],ids);ld=max(abs(float(a)-float(b)) for a,b in zip(logits,baseline["class_logits"]));sd=abs(float(score["soft_sa_image_score"])-float(baseline["soft_sa_image_score"]))
                    if ld>LOGIT_PARITY_TOLERANCE or sd>SOFT_PARITY_TOLERANCE or int(score["argmax_hard_class"])!=target:raise RuntimeError(f"Clean parity failed: {row['case_id']}")
                    append_jsonl(parity_path,{"case_id":row["case_id"],"max_abs_logit_difference":ld,"abs_soft_sa_difference":sd,"hard_equal":True});parity.add(row["case_id"])
                edges=evidence_answer_edges(spans)
                expected_edges={(spans["SAC"],source) for source in sorted(set(spans["EVIDENCE"])|set(range(spans["ANSWER"][0],spans["ANSWER"][1])))}
                if set(edges.pairs)!=expected_edges:raise RuntimeError("E+A edge set mismatch")
                for start,end in WINDOWS:
                    key=(row["case_id"],NEW_CONDITION,start)
                    if key in completed:continue
                    before=time.perf_counter()
                    with AttentionBlockContext(modules.language_layers,layer_indices=range(start,end+1),edges=edges,sequence_length=spans["sequence_length"],row_sum_tolerance=ROW_SUM_TOLERANCE) as context:logits,score=_forward(inference.model,inputs,spans["SAC"],ids)
                    margin=_margin(logits,target);diag=context.diagnostics()
                    append_jsonl(blocked_path,{"arm":"delayed","case_id":row["case_id"],"item_id":str(row["item_id"]),"test_side":row["test_side"],"phase":"coarse","condition":NEW_CONDITION,"refine_pair":None,
                        "window_start":start,"window_end":end,"window_center":(start+end)/2,"blocked_layer_count":12,"class_logits":logits,"blocked_class":int(score["argmax_hard_class"]),
                        "blocked_soft_sa":float(score["soft_sa_image_score"]),"clean_class":target,"clean_margin":float(baseline["clean_margin"]),"blocked_margin":margin,
                        "logit_margin_disruption":float(baseline["clean_margin"])-margin,"first_token_changed":int(score["argmax_hard_class"])!=target,
                        "delta_soft_sa":float(score["soft_sa_image_score"])-float(baseline["soft_sa_image_score"]),"abs_delta_soft_sa":abs(float(score["soft_sa_image_score"])-float(baseline["soft_sa_image_score"])),
                        "elapsed_seconds":time.perf_counter()-before,"attention_diagnostics":diag,"provenance":"new_forward"});completed.add(key)
                    elapsed=time.time()-started;done=len(parity)+sum(1 for c in completed if c[1]==NEW_CONDITION);atomic_json(output/"progress.json",{"status":"running","completed":done,"expected":expected,"new_blocked":done-len(parity),"reused_blocked":reused,"failed":len(load_jsonl(failure_path)),"elapsed_seconds":elapsed,"estimated_remaining_seconds":elapsed/max(done,1)*(expected-done)})
                del inputs
            except Exception as exc:append_jsonl(failure_path,{"case_id":row.get("case_id"),"error_type":type(exc).__name__,"error":str(exc)});raise
        new_count=sum(1 for r in load_jsonl(blocked_path) if r.get("provenance")=="new_forward");result={"status":"complete","clean_parity":len(parity),"new_blocked":new_count,"reused_blocked":reused,"total_blocked":new_count+reused,"failures":len(load_jsonl(failure_path)),"elapsed_seconds":time.time()-started,"estimated_remaining_seconds":0.0};atomic_json(output/"completion.json",result);atomic_json(output/"progress.json",result);return result
    finally:
        del inference
        if torch.cuda.is_available():torch.cuda.empty_cache()


def main(argv:Sequence[str]|None=None)->int:
    p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path);p.add_argument("--base-output",type=Path,default=DEFAULT_BASE);p.add_argument("--smoke",action="store_true");p.add_argument("--resume",action="store_true");a=p.parse_args(argv);output=a.output_dir or DEFAULT_OUTPUT_PARENT/time.strftime(("smoke" if a.smoke else "formal")+"_seed42_%Y%m%dT%H%M%SZ",time.gmtime());run(output,a.base_output.resolve(),smoke=a.smoke,resume=a.resume);return 0


if __name__=="__main__":raise SystemExit(main())
