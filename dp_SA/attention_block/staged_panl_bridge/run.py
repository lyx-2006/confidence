from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import time
from pathlib import Path
from typing import Any,Sequence

import torch

from confidence_test.runtime_imports import load_runtime
from dp_SA.attention_block.config import INFERENCE_PATH,LOGIT_PARITY_TOLERANCE,MODEL_PATH,ROW_SUM_TOLERANCE,SOFT_PARITY_TOLERANCE
from dp_SA.attention_block.masking import AttentionBlockContext
from dp_SA.attention_block.run import _forward,_margin
from dp_SA.attention_block.sources import prepare_case
from dp_SA.attention_block.spans import locate_spans
from dp_SA.io_utils import append_jsonl,atomic_json,atomic_jsonl,canonical_hash,load_jsonl,sha256_file
from dp_SA.soft_score import class_token_ids
from layer_metacognition.model_adapter import resolve_language_modules

from .core import G_WINDOWS,R_WINDOWS,WINDOW_PAIRS,Cell,cells,layer_edges

ROOT=Path(__file__).resolve().parents[3]
DEFAULT_BASE=ROOT/"dp_SA/attention_block/outputs/formal_both_seed42_w12_20260823T093446Z"
DEFAULT_OUTPUT_PARENT=Path(__file__).resolve().parent/"outputs"
C10_PARITY_TOLERANCE=1e-6


def _selection(base:Path,smoke:bool)->list[dict[str,Any]]:
    rows=json.loads((base/"delayed_case_manifest.json").read_text())
    counts={side:sum(r["test_side"]==side for r in rows) for side in ("image_side","text_side")}
    if len(rows)!=100 or counts!={"image_side":50,"text_side":50} or len({r["item_id"] for r in rows})!=100:
        raise ValueError(f"Frozen delayed manifest invalid: n={len(rows)}, sides={counts}")
    if smoke:return [r for side in ("image_side","text_side") for r in [x for x in rows if x["test_side"]==side][:2]]
    return rows


def _config(base:Path,selection:list[dict[str,Any]],smoke:bool)->dict[str,Any]:
    package=Path(__file__).resolve().parent
    implementation=[package/name for name in ("run.py","core.py","analyze.py","run_pipeline.py")]
    implementation += [package.parent/"masking.py",package.parent/"spans.py"]
    value={"format_version":1,"experiment":"staged_panl_bridge","arm":"delayed","base_output":str(base.resolve()),
           "base_config_sha256":sha256_file(base/"run_config.json"),"manifest_sha256":sha256_file(base/"delayed_case_manifest.json"),
           "clean_sha256":sha256_file(base/"clean_baselines.jsonl"),"spans_sha256":sha256_file(base/"delayed_token_spans.jsonl"),
           "selection_hash":canonical_hash(selection),"prompt_hash":canonical_hash([r["phase1_prompt_hash"] for r in selection]),
           "model_path":str(MODEL_PATH.resolve()),"model_config_sha256":sha256_file(MODEL_PATH/"config.json"),
           "processor_config_sha256":sha256_file(MODEL_PATH/"preprocessor_config.json"),"inference_path":str(INFERENCE_PATH.resolve()),
           "attention_backend":"eager","num_layers":28,"gathering_windows":G_WINDOWS,"readout_windows":R_WINDOWS,
           "window_pairs":WINDOW_PAIRS,"cells":[c.__dict__ for c in cells()],"seed":42,"bootstrap_repeats":2000,
           "sign_flip_repeats":20000,"c10_parity_tolerance":C10_PARITY_TOLERANCE,"smoke":smoke,
           "implementation_sha256":{str(p):sha256_file(p) for p in implementation}}
    value["fingerprint"]=canonical_hash(value);return value


def ensure_config(path:Path,config:dict[str,Any],*,resume:bool)->None:
    if path.exists():
        if json.loads(path.read_text()).get("fingerprint")!=config["fingerprint"]:raise ValueError("Fingerprint changed; refusing resume")
        if not resume:raise FileExistsError(f"{path.parent} exists; use --resume")
    else:atomic_json(path,config)


def pending_cells(case_id:str,completed:set[tuple[str,str]])->tuple[Cell,...]:
    return tuple(cell for cell in cells() if (case_id,cell.key) not in completed)


def c10_parity(c00:dict[str,Any],c10:dict[str,Any],*,tolerance:float=C10_PARITY_TOLERANCE)->dict[str,Any]:
    max_logits=max(abs(float(a)-float(b)) for a,b in zip(c00["class_logits"],c10["class_logits"]))
    margin=abs(float(c00["margin"])-float(c10["margin"]));soft=abs(float(c00["soft_sa"])-float(c10["soft_sa"]));hard_equal=int(c00["blocked_class"])==int(c10["blocked_class"])
    result={"max_abs_logit_difference":max_logits,"abs_margin_difference":margin,"abs_soft_sa_difference":soft,"hard_equal":hard_equal,
            "tolerance":tolerance,"passed":bool(max_logits<=tolerance and margin<=tolerance and soft<=tolerance and hard_equal)}
    if not result["passed"]:raise RuntimeError(f"C10 isolation parity failed: {result}")
    return result


def _blocked_forward(model:Any,modules:Any,inputs:Any,spans:dict[str,Any],class_ids:list[int],cell:Cell):
    per_layer=layer_edges(spans,cell);contexts=[]
    with ExitStack() as stack:
        for layer in range(28):
            context=AttentionBlockContext(modules.language_layers,layer_indices=[layer],edges=per_layer[layer],
                                          sequence_length=spans["sequence_length"],row_sum_tolerance=ROW_SUM_TOLERANCE)
            contexts.append(stack.enter_context(context))
        logits,score=_forward(model,inputs,spans["SAC"],class_ids)
        by_layer={}
        for layer,context in enumerate(contexts):
            diagnostic=context.diagnostics();by_layer[str(layer)]=diagnostic["by_layer"][str(layer)]
    edge_spec={str(layer):{"count":len(per_layer[layer].pairs),"sha256":canonical_hash(per_layer[layer].pairs)} for layer in range(28)}
    return logits,score,{"layers":list(range(28)),"by_layer":by_layer},edge_spec


def run(output:Path,base:Path=DEFAULT_BASE,*,smoke:bool=False,resume:bool=False)->dict[str,Any]:
    output=output.resolve();output.mkdir(parents=True,exist_ok=True);selection=_selection(base,smoke);config=_config(base,selection,smoke)
    config_path=output/"run_config.json";existed=config_path.exists();ensure_config(config_path,config,resume=resume)
    if not existed:atomic_jsonl(output/"selection_manifest.jsonl",selection)
    clean={r["case_id"]:r for r in load_jsonl(base/"clean_baselines.jsonl") if r["arm"]=="delayed"}
    frozen={r["case_id"]:r for r in load_jsonl(base/"delayed_token_spans.jsonl")}
    blocked_path=output/"blocked_results.jsonl";parity_path=output/"clean_parity.jsonl";spans_path=output/"token_spans.jsonl";failures_path=output/"failures.jsonl"
    for path in (blocked_path,parity_path,spans_path,failures_path):path.touch(exist_ok=True)
    existing_blocked=load_jsonl(blocked_path);completed={(r["case_id"],r["cell_key"]) for r in existing_blocked};result_by_key={(r["case_id"],r["cell_key"]):r for r in existing_blocked}
    parity={r["case_id"] for r in load_jsonl(parity_path)};saved_spans={r["case_id"] for r in load_jsonl(spans_path)}
    runtime=load_runtime(INFERENCE_PATH);inference=runtime.QwenVLInference(str(MODEL_PATH));modules=resolve_language_modules(inference.model)
    tokenizer=getattr(inference.processor,"tokenizer",inference.processor);class_ids=class_token_ids(tokenizer)
    if getattr(inference.model.config,"_attn_implementation",None)!="eager" or modules.num_hidden_layers!=28:raise RuntimeError("Expected eager 28-layer model")
    started=time.time();expected=len(selection)*(1+len(cells()))
    try:
        for row in selection:
            try:
                rendered,inputs=prepare_case(inference,row);spans=locate_spans(tokenizer,rendered,inputs,row)
                frozen_span={k:v for k,v in frozen[row["case_id"]].items() if k not in {"arm","case_id"}}
                if canonical_hash(spans)!=canonical_hash(frozen_span):raise RuntimeError(f"Frozen spans changed: {row['case_id']}")
                if row["case_id"] not in saved_spans:
                    append_jsonl(spans_path,{"case_id":row["case_id"],"item_id":row["item_id"],"test_side":row["test_side"],**spans});saved_spans.add(row["case_id"])
                baseline=clean[row["case_id"]];target=int(baseline["clean_class"])
                if row["case_id"] not in parity:
                    logits,score=_forward(inference.model,inputs,spans["SAC"],class_ids)
                    max_diff=max(abs(float(a)-float(b)) for a,b in zip(logits,baseline["class_logits"]));soft_diff=abs(float(score["soft_sa_image_score"])-float(baseline["soft_sa_image_score"]))
                    if max_diff>LOGIT_PARITY_TOLERANCE or soft_diff>SOFT_PARITY_TOLERANCE or int(score["argmax_hard_class"])!=target:
                        raise RuntimeError(f"Clean parity failed {row['case_id']}: logits={max_diff}, soft={soft_diff}")
                    append_jsonl(parity_path,{"case_id":row["case_id"],"max_abs_logit_difference":max_diff,"abs_soft_sa_difference":soft_diff,"hard_equal":True});parity.add(row["case_id"])
                for cell in pending_cells(row["case_id"],completed):
                    key=(row["case_id"],cell.key)
                    if key in completed:continue
                    before=time.perf_counter();logits,score,diagnostics,edge_spec=_blocked_forward(inference.model,modules,inputs,spans,class_ids,cell)
                    margin=_margin(logits,target)
                    record={"case_id":row["case_id"],"item_id":row["item_id"],"test_side":row["test_side"],
                        "cell_key":cell.key,"condition":cell.family,"gathering_window":cell.gathering,"readout_window":cell.readout,
                        "class_logits":logits,"clean_class":target,"blocked_class":int(score["argmax_hard_class"]),"margin":margin,
                        "clean_margin":float(baseline["clean_margin"]),"clean_margin_disruption":float(baseline["clean_margin"])-margin,
                        "soft_sa":float(score["soft_sa_image_score"]),"soft_sa_delta":float(score["soft_sa_image_score"])-float(baseline["soft_sa_image_score"]),
                        "token_changed":int(score["argmax_hard_class"])!=target,"edge_specification":edge_spec,
                        "attention_diagnostics":diagnostics,"elapsed_seconds":time.perf_counter()-before}
                    if cell.family=="C10":record["c10_parity_to_c00"]=c10_parity(result_by_key[(row["case_id"],"C00")],record)
                    append_jsonl(blocked_path,record);completed.add(key);result_by_key[key]=record
                    elapsed=time.time()-started;done=len(parity)+len(completed)
                    atomic_json(output/"progress.json",{"status":"running","completed":done,"expected":expected,"blocked_completed":len(completed),
                                "clean_completed":len(parity),"failed":len(load_jsonl(failures_path)),"elapsed_seconds":elapsed,
                                "estimated_remaining_seconds":elapsed/max(1,done)*(expected-done),"current_case_id":row["case_id"],"current_cell":cell.key})
                del inputs
            except Exception as exc:
                append_jsonl(failures_path,{"case_id":row.get("case_id"),"error_type":type(exc).__name__,"error":str(exc)});raise
        result={"status":"complete","clean_parity":len(parity),"blocked":len(completed),"failures":len(load_jsonl(failures_path)),
                "elapsed_seconds":time.time()-started,"estimated_remaining_seconds":0.0}
        atomic_json(output/"completion.json",result);atomic_json(output/"progress.json",result);return result
    finally:
        del inference
        if torch.cuda.is_available():torch.cuda.empty_cache()


def main(argv:Sequence[str]|None=None)->int:
    p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path);p.add_argument("--base-output",type=Path,default=DEFAULT_BASE);p.add_argument("--smoke",action="store_true");p.add_argument("--resume",action="store_true");a=p.parse_args(argv)
    output=a.output_dir or DEFAULT_OUTPUT_PARENT/time.strftime(("smoke" if a.smoke else "formal")+"_seed42_%Y%m%dT%H%M%SZ",time.gmtime())
    run(output,a.base_output.resolve(),smoke=a.smoke,resume=a.resume);return 0


if __name__=="__main__":raise SystemExit(main())
