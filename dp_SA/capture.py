from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from confidence_test.answer_metrics import normalize_answer, parse_answer_output
from confidence_test.dataset_utils import load_evaluation_cases
from confidence_test.runtime_imports import load_runtime
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.model_adapter import model_input_device, resolve_language_modules, run_hooked_forward

from .config import (
    CONDITIONS, DATASET_PATH, ERROR_RATE_LIMIT, HIDDEN_DEFINITION, INFERENCE_PATH,
    LAYERS, MODEL_PATH, OUTPUT_ROOT, POSITIONS, SUPPORTED_CAPTURE_POSITIONS,
)
from .io_utils import append_jsonl, atomic_json, canonical_hash, load_jsonl, sha256_file
from .positions import locate_phase1_positions
from .prompts import ANSWER_PREFILL, SA_PREFILL, phase0_prompt, phase1_prompt
from .soft_score import class_token_ids, soft_sa_from_logits

def _content(prompt: str, image_path: str) -> list[dict[str,str]]:
    return [{"type":"image","image":str(Path(image_path).resolve())},{"type":"text","text":prompt}]

def _messages(prompt: str, image_path: str, prefill: str) -> list[dict[str,Any]]:
    return [{"role":"user","content":_content(prompt,image_path)},
            {"role":"assistant","content":[{"type":"text","text":prefill}]}]

def _generate(inference: Any, inputs: Any, max_new_tokens: int, allowed_first_tokens: Sequence[int]|None=None) -> tuple[list[int],str,bool]:
    generation_kwargs: dict[str, Any] = {"max_new_tokens":max_new_tokens,"do_sample":False,"use_cache":True}
    if allowed_first_tokens is not None:
        allowed=tuple(int(value) for value in allowed_first_tokens)
        generation_kwargs["prefix_allowed_tokens_fn"] = lambda _batch_id, _input_ids: list(allowed)
    with torch.inference_mode():
        generated=inference.model.generate(**inputs,**generation_kwargs)
    input_length=int(inputs.input_ids.shape[1]); tokens=[int(x) for x in generated[0,input_length:].tolist()]
    tokenizer=getattr(inference.processor,"tokenizer",inference.processor)
    text=tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    eos_ids=getattr(inference.model.generation_config,"eos_token_id",[])
    if isinstance(eos_ids,int): eos_ids=[eos_ids]
    return tokens,text,bool(tokens and tokens[-1] in set(eos_ids or []))

def _atomic_npz(path: Path, arrays: dict[str,np.ndarray]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent); os.close(fd)
    try:
        with open(tmp,"wb") as handle: np.savez(handle,**arrays); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp,path)
    except Exception:
        try: os.unlink(tmp)
        except FileNotFoundError: pass
        raise

def _case_rows(max_items: int|None, max_samples: int|None) -> list[dict[str,Any]]:
    cases,_=load_evaluation_cases(DATASET_PATH,item_limit=max_items)
    rows=[]
    for case in cases:
        for condition in CONDITIONS:
            image=case.conditions[condition]
            if image.error: continue
            rows.append({"case":case,"condition":condition,"image_path":image.resolved_image_path,
                         "case_id":f"{case.item_id}__prior_{case.prior_index}__{condition}__v4__delayed_sa"})
    if not max_samples:
        return rows
    # Smoke coverage should span items instead of taking many priors from the
    # first few items.  Select one stable record per item first.
    selected=[]; used=set()
    for row in rows:
        item=str(row["case"].item_id)
        if item in used: continue
        selected.append(row); used.add(item)
        if len(selected)==max_samples: return selected
    return selected

def _parse_positions(values: Sequence[str]) -> tuple[str, ...]:
    output=tuple(str(value) for value in values)
    if not output or len(set(output)) != len(output):
        raise ValueError("--positions must be non-empty and unique")
    invalid=sorted(set(output)-set(SUPPORTED_CAPTURE_POSITIONS))
    if invalid:
        raise ValueError(f"Unsupported capture positions: {invalid}")
    return output

def _parse_layers(values: Sequence[int]) -> tuple[int, ...]:
    output=tuple(int(value) for value in values)
    if not output or len(set(output)) != len(output):
        raise ValueError("--layers must be non-empty and unique")
    if min(output) < 0:
        raise ValueError("--layers must contain zero-based non-negative indices")
    return output

def run_capture(*, output_root: Path=OUTPUT_ROOT, max_items: int|None=None, max_samples: int|None=None,
                resume: bool=False, positions: Sequence[str]=POSITIONS, layers: Sequence[int]=LAYERS) -> dict[str,Any]:
    positions=_parse_positions(positions); layers=_parse_layers(layers)
    capture_dir=output_root/"capture"; capture_dir.mkdir(parents=True,exist_ok=True)
    pid_path=capture_dir/"active.pid"
    if pid_path.exists():
        try:
            pid=int(pid_path.read_text()); os.kill(pid,0); raise RuntimeError(f"Capture already active: PID {pid}")
        except ProcessLookupError: pid_path.unlink()
    pid_path.write_text(str(os.getpid()))
    try:
        model_files=("config.json","tokenizer.json","tokenizer_config.json","preprocessor_config.json","model.safetensors.index.json")
        model_fingerprint={name:sha256_file(MODEL_PATH/name) for name in model_files}
        config={"format_version":2,"dataset":str(DATASET_PATH.resolve()),"dataset_sha256":sha256_file(DATASET_PATH),"model":str(MODEL_PATH.resolve()),"model_fingerprint":model_fingerprint,
                "conditions":list(CONDITIONS),"positions":list(positions),"layers":list(layers),"max_items":max_items,"max_samples":max_samples,
                "hidden_definition":HIDDEN_DEFINITION,"phase0_generation":{"max_new_tokens":24,"do_sample":False,"use_cache":True},
                "phase1_generation":{"max_new_tokens":1,"do_sample":False,"use_cache":True,"constraint":"validated_class_token_ids"},"phase0_template_hash":canonical_hash(phase0_prompt("{question}","{text_clue}")),
                "phase1_template_hash":canonical_hash(phase1_prompt("{question}","{text_clue}","{answer}"))}
        config["fingerprint"]=canonical_hash(config)
        config_path=capture_dir/"config.json"
        if config_path.exists():
            old=json.loads(config_path.read_text())
            if old.get("fingerprint")!=config["fingerprint"]:
                raise ValueError("Capture config fingerprint changed; use a fresh --output-root")
            if not resume: raise FileExistsError("Capture output exists; use --resume")
        else: atomic_json(config_path,config)
        phase0_path=capture_dir/"phase0_results.jsonl"; results_path=capture_dir/"results.jsonl"
        phase0={r["case_id"]:r for r in load_jsonl(phase0_path)}; completed={r["case_id"] for r in load_jsonl(results_path) if r.get("status")=="completed"}
        runtime=load_runtime(INFERENCE_PATH); inference=runtime.QwenVLInference(str(MODEL_PATH)); modules=resolve_language_modules(inference.model)
        if any(layer >= modules.num_hidden_layers for layer in layers):
            raise ValueError(f"Requested layer outside model with {modules.num_hidden_layers} layers")
        tokenizer=getattr(inference.processor,"tokenizer",inference.processor); class_ids=class_token_ids(tokenizer); device=model_input_device(inference)
        rows=_case_rows(max_items,max_samples); failures=0; started=time.time(); image_hashes: dict[str,str]={}
        for ordinal,spec in enumerate(rows,1):
            case=spec["case"]; case_id=spec["case_id"]
            if case_id in completed: continue
            try:
                p0=phase0.get(case_id)
                if p0 is None:
                    prompt0=phase0_prompt(case.question,case.text_clue); messages0=_messages(prompt0,spec["image_path"],ANSWER_PREFILL)
                    rendered0=render_continued_assistant(inference.processor,messages0,ANSWER_PREFILL)
                    inputs0=prepare_multimodal_inputs(inference.processor,messages0,rendered0,device=device)
                    token_ids,continuation,eos=_generate(inference,inputs0,24); raw=ANSWER_PREFILL+continuation
                    answer,normalized,ok=parse_answer_output(raw)
                    answer_ids=tokenizer.encode(str(answer),add_special_tokens=False) if answer else []
                    if str(spec["image_path"]) not in image_hashes:
                        image_hashes[str(spec["image_path"])]=sha256_file(spec["image_path"])
                    p0={"case_id":case_id,"item_id":case.item_id,"prior_index":case.prior_index,"condition":spec["condition"],"version":"v4",
                        "question":case.question,"text_clue":case.text_clue,"image_path":spec["image_path"],"phase0_raw_output":raw,"phase0_raw_answer":answer,
                        "image_sha256":image_hashes[str(spec["image_path"])],"phase0_normalized_answer":normalized,"phase0_answer_token_ids":answer_ids,
                        "phase0_generated_token_ids":token_ids,"phase0_eos_generated":eos,"phase0_generation_config":config["phase0_generation"],"phase0_prompt":prompt0,
                        "phase0_prompt_hash":canonical_hash(prompt0),"phase0_answer_fingerprint":canonical_hash(answer),"parse_success":ok,"status":"completed" if ok else "failed"}
                    append_jsonl(phase0_path,p0); phase0[case_id]=p0
                if p0.get("status")!="completed": raise ValueError("Phase 0 answer parser failed")
                answer=str(p0["phase0_raw_answer"]); prompt1=phase1_prompt(case.question,case.text_clue,answer); messages1=_messages(prompt1,spec["image_path"],SA_PREFILL)
                rendered1=render_continued_assistant(inference.processor,messages1,SA_PREFILL)
                inputs1=prepare_multimodal_inputs(inference.processor,messages1,rendered1,device=device)
                located=locate_phase1_positions(tokenizer,rendered1,inputs1,answer); pos={name:int(located[name]["processed_index"]) for name in positions}
                sac=int(located["P1_SAC"]["processed_index"])
                forward=run_hooked_forward(inference.model,inputs1,modules,pos,logits_positions=[sac])
                score=soft_sa_from_logits(forward.logits_by_position[sac],class_ids)
                generated_ids,generated_text,_=_generate(inference,inputs1,1,class_ids); valid=generated_text in set(map(str,range(9)))
                if not valid: raise ValueError(f"Constrained Phase 1 generation was invalid: {generated_text!r} {generated_ids}")
                if int(generated_text)!=score["argmax_hard_class"]: raise ValueError("Forward argmax differs from greedy generation")
                arrays={f"{position}__L{layer}":forward.hidden_by_name[position][layer].detach().float().cpu().numpy().astype(np.float16) for position in positions for layer in layers}
                hidden_rel=Path("capture")/"hidden"/f"{case_id}.npz"; _atomic_npz(output_root/hidden_rel,arrays)
                normalized=normalize_answer(answer); image_answer=case.conflict_answer
                result={"status":"completed","case_id":case_id,"item_id":case.item_id,"prior_index":case.prior_index,"condition":spec["condition"],"version":"v4",
                        "question":case.question,"text_clue":case.text_clue,"image_path":spec["image_path"],"phase0_raw_answer":answer,"phase0_normalized_answer":normalized,
                        "phase1_inserted_raw_answer":answer,"phase1_inserted_normalized_answer":normalize_answer(answer),"phase1_prompt":prompt1,"phase1_prompt_hash":canonical_hash(prompt1),
                        "phase1_answer_span":located["phase1_answer_span"],"phase1_answer_token_ids":located["phase1_answer_token_ids"],"positions":located,
                        **score,"raw_generated_class":generated_text,"valid_class":valid,"generated_token_ids":generated_ids,"phase1_generation_config":config["phase1_generation"],
                        "phase0_answer_fingerprint":p0["phase0_answer_fingerprint"],"image_sha256":p0["image_sha256"],"hidden_file":str(hidden_rel),
                        "phase0_correct":normalized==case.ground_truth_answer,"answer_matches_text":normalized==case.text_answer,"answer_matches_image":normalized==image_answer,
                        "answer_length":len(answer),"elapsed_ordinal":ordinal}
                if result["phase1_inserted_raw_answer"]!=p0["phase0_raw_answer"]: raise AssertionError("Raw answer changed")
                append_jsonl(results_path,result)
            except Exception as exc:
                failures+=1; append_jsonl(results_path,{"status":"failed","case_id":case_id,"item_id":case.item_id,"prior_index":case.prior_index,"condition":spec["condition"],"error":{"type":type(exc).__name__,"message":str(exc)}})
                if failures/max(1,ordinal)>ERROR_RATE_LIMIT: raise RuntimeError(f"Capture failure rate exceeded limit at {ordinal}: {failures}") from exc
            if ordinal%10==0: atomic_json(capture_dir/"progress.json",{"completed":ordinal-failures,"failed":failures,"total":len(rows),"elapsed_seconds":time.time()-started})
        summary={"status":"complete","total":len(rows),"completed":len([r for r in load_jsonl(results_path) if r.get("status")=="completed"]),"failed":failures}
        atomic_json(capture_dir/"progress.json",{"completed":summary["completed"],"failed":summary["failed"],"total":summary["total"],"elapsed_seconds":time.time()-started})
        atomic_json(capture_dir/"summary.json",summary); return summary
    finally:
        if pid_path.exists() and pid_path.read_text().strip()==str(os.getpid()): pid_path.unlink()

def main(argv: Sequence[str]|None=None)->int:
    p=argparse.ArgumentParser(); p.add_argument("--output-root",default=str(OUTPUT_ROOT)); p.add_argument("--max-items",type=int); p.add_argument("--max-samples",type=int); p.add_argument("--resume",action="store_true")
    p.add_argument("--positions",nargs="+",default=list(POSITIONS)); p.add_argument("--layers",nargs="+",type=int,default=list(LAYERS))
    a=p.parse_args(argv); run_capture(output_root=Path(a.output_root),max_items=a.max_items,max_samples=a.max_samples,resume=a.resume,positions=a.positions,layers=a.layers); return 0
if __name__=="__main__": raise SystemExit(main())
