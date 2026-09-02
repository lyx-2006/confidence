from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

from confidence_test.inference_extension import _user_content
from confidence_test.runtime_imports import load_runtime
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.model_adapter import model_input_device, resolve_language_modules, run_logits_forward

from .build_split import build_split
from .config import ANSWER_PREFILL, IMAGE_PHASE0_TEMPLATE, INFERENCE_PATH, MODEL_PATH, RESULTS_ROOT, TEXT_PHASE0_TEMPLATE
from .io_utils import atomic_json, atomic_jsonl, canonical_hash, ensure_layout, load_jsonl, sha256_file, stable_shard, validate_fingerprint
from .metrics import entropy_difficulty, score_metrics


def spec_key(spec: dict[str, Any]) -> tuple[Any, ...]:
    return ("text",str(spec["item_id"]),int(spec["prior_index"])) if spec["modality"]=="text" else ("image",str(spec["item_id"]),str(spec["condition"]),str(spec["image_hash"]))


def unique_specs(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    text={}; image={}
    for row in rows:
        tk=(str(row["item_id"]),int(row["prior_index"])); candidate={"modality":"text","item_id":tk[0],"prior_index":tk[1],"unique_key":list(tk),"question":row["question"],"text_clue":row["text_clue"],"image_path":None,"image_hash":None,"condition":None,"target_answer":row["text_answer"],"answer_classes":list(row["answer_classes"])}
        old=text.setdefault(tk,candidate)
        if old != candidate: raise ValueError(f"Inconsistent text key: {tk}")
        ik=(str(row["item_id"]),str(row["condition"]),str(row["image_sha256"])); candidate={"modality":"image","item_id":ik[0],"prior_index":None,"unique_key":list(ik),"question":row["question"],"text_clue":None,"image_path":row["image_path"],"image_hash":ik[2],"condition":ik[1],"target_answer":row["image_answer"],"answer_classes":list(row["answer_classes"])}
        old=image.setdefault(ik,candidate)
        if old != candidate: raise ValueError(f"Inconsistent image key: {ik}")
    return sorted([*text.values(),*image.values()],key=spec_key)


def wire(spec: dict[str, Any]) -> tuple[str,list[dict[str,Any]]]:
    if spec["modality"] == "text":
        prompt=TEXT_PHASE0_TEMPLATE.format(question=spec["question"],text_clue=spec["text_clue"]); content=_user_content(prompt,None)
        if [part.get("type") for part in content] != ["text"]: raise AssertionError("Text input contains non-text modality")
    else:
        prompt=IMAGE_PHASE0_TEMPLATE.format(question=spec["question"]); content=_user_content(prompt,spec["image_path"])
        if [part.get("type") for part in content] != ["image","text"] or "Text clue:" in prompt: raise AssertionError("Image input is not isolated")
    return prompt,[{"role":"user","content":content},{"role":"assistant","content":[{"type":"text","text":ANSWER_PREFILL}]}]


def candidate_suffix_ids(tokenizer: Any, rendered: str, candidate: str) -> list[int]:
    base=list(map(int,tokenizer.encode(rendered,add_special_tokens=False))); full=list(map(int,tokenizer.encode(rendered+candidate,add_special_tokens=False)))
    if full[:len(base)] != base or len(full)<=len(base): raise ValueError(f"Candidate does not append cleanly: {candidate}")
    return full[len(base):]


def tokenizer_preflight(processor: Any, specs: Sequence[dict[str,Any]]) -> tuple[str,dict[tuple[Any,...],dict[str,Any]]]:
    tokenizer=getattr(processor,"tokenizer",processor); audits={}; any_multi=False
    for spec in specs:
        _prompt,messages=wire(spec); rendered=render_continued_assistant(processor,messages,ANSWER_PREFILL)
        ids={name:candidate_suffix_ids(tokenizer,rendered,name) for name in spec["answer_classes"]}
        if len(ids)!=12 or len({tuple(value) for value in ids.values()})!=12: raise ValueError("Candidate tokenization is not twelve distinct sequences")
        any_multi |= any(len(value)!=1 for value in ids.values()); audits[spec_key(spec)]={"rendered":rendered,"candidate_token_ids":ids}
    return ("sequence_score_temperature_extension" if any_multi else "single_token_next_token_logits"),audits


def score_single(inference: Any, modules: Any, spec: dict[str,Any], rendered: str, messages: list[dict[str,Any]], ids: dict[str,list[int]]) -> list[float]:
    inputs=prepare_multimodal_inputs(inference.processor,messages,rendered,device=model_input_device(inference)); position=int(inputs.input_ids.shape[1])-1
    logits=run_logits_forward(inference.model,inputs,[position],modules)[position]
    return [float(logits[ids[name][0]]) for name in spec["answer_classes"]]


def score_sequences(inference: Any, modules: Any, spec: dict[str,Any], rendered: str, messages: list[dict[str,Any]], ids: dict[str,list[int]]) -> list[float]:
    base=prepare_multimodal_inputs(inference.processor,messages,rendered,device=model_input_device(inference)); base_length=int(base.input_ids.shape[1]); output=[]
    for name in spec["answer_classes"]:
        full=prepare_multimodal_inputs(inference.processor,messages,rendered+name,device=model_input_device(inference)); suffix=[int(v) for v in full.input_ids[0,base_length:].tolist()]
        if suffix != ids[name]: raise ValueError(f"Processed candidate suffix mismatch: {name}")
        positions=list(range(base_length-1,int(full.input_ids.shape[1])-1)); logits=run_logits_forward(inference.model,full,positions,modules); total=0.0
        for offset,token in enumerate(suffix): total += float(torch.log_softmax(logits[positions[offset]].double(),dim=-1)[token])
        output.append(total)
    return output


def _worker(root: Path, worker_id: int, num_gpus: int, *, resume: bool) -> dict[str,Any]:
    all_specs=unique_specs(load_jsonl(root/"shared/manifests/probe_manifest.jsonl")); specs=[spec for spec in all_specs if stable_shard(spec_key(spec),num_gpus)==worker_id]
    output=root/f"unimodal_confidence/artifacts/raw_scores/shard_{worker_id}.jsonl"; existing=load_jsonl(output,repair_trailing=resume); completed={tuple(row["stable_key"]) for row in existing}
    if len(completed)==len(specs) and completed=={spec_key(s) for s in specs}: return {"worker":worker_id,"count":len(specs),"resumed_noop":True}
    runtime=load_runtime(INFERENCE_PATH); inference=runtime.QwenVLInference(str(MODEL_PATH)); modules=resolve_language_modules(inference.model); policy,audits=tokenizer_preflight(inference.processor,all_specs); rows=list(existing)
    for spec in specs:
        key=spec_key(spec)
        if key in completed: continue
        prompt,messages=wire(spec); audit=audits[key]; rendered=audit["rendered"]; ids=audit["candidate_token_ids"]
        scores=score_single(inference,modules,spec,rendered,messages,ids) if policy=="single_token_next_token_logits" else score_sequences(inference,modules,spec,rendered,messages,ids)
        metrics=score_metrics(spec["answer_classes"],scores,str(spec["target_answer"])); input_payload={"prompt":prompt,"rendered_prompt":rendered,"modalities":[part["type"] for part in messages[0]["content"]],"candidate_token_ids":ids,"image_hash":spec.get("image_hash")}
        rows.append({"modality":spec["modality"],"unique_key":spec["unique_key"],"stable_key":list(key),"item_id":spec["item_id"],"prior_index":spec.get("prior_index"),"condition":spec.get("condition"),"image_hash":spec.get("image_hash"),"question":spec["question"],"text_clue":spec.get("text_clue"),"image_path":spec.get("image_path"),"target_answer":spec["target_answer"],"answer_classes":spec["answer_classes"],"prompt":prompt,"rendered_prompt":rendered,"input_modalities":input_payload["modalities"],"candidate_token_ids":ids,"input_hash":canonical_hash(input_payload),"tokenization_policy":policy,"raw_candidate_scores":{name:float(scores[i]) for i,name in enumerate(spec["answer_classes"])},"uncalibrated_probabilities":metrics["probabilities"],"probability_sum":metrics["probability_sum"],"chosen_answer":metrics["chosen_answer"],"uncalibrated_chosen_confidence":metrics["chosen_confidence"],"correct":metrics["correct"],"uncalibrated_nll":metrics["nll"],"uncalibrated_brier":metrics["brier"],"entropy_difficulty":entropy_difficulty(list(metrics["probabilities"].values()))})
        atomic_jsonl(output,sorted(rows,key=lambda r:tuple(r["stable_key"])))
    return {"worker":worker_id,"count":len(specs),"policy":policy}


def score_unimodal(root: Path, *, num_gpus: int, resume: bool) -> dict[str,Any]:
    if num_gpus not in (1,2): raise ValueError("--num-gpus must be 1 or 2")
    if not (root/"shared/manifests/probe_manifest.jsonl").is_file(): build_split(root)
    validate_fingerprint(root/"unimodal_confidence/progress/score_config.json",{"manifest_sha256":sha256_file(root/"shared/manifests/probe_manifest.jsonl"),"model_config_sha256":sha256_file(MODEL_PATH/"config.json"),"text_template":canonical_hash(TEXT_PHASE0_TEMPLATE),"image_template":canonical_hash(IMAGE_PHASE0_TEMPLATE),"shard_policy":"sha256(canonical_json(stable_key)) % num_gpus","num_gpus":num_gpus},resume=resume)
    if not torch.cuda.is_available() or torch.cuda.device_count()<num_gpus: raise RuntimeError(f"Requested {num_gpus} GPUs, visible={torch.cuda.device_count()}")
    commands=[]
    for worker in range(num_gpus):
        env=dict(os.environ); env["CUDA_VISIBLE_DEVICES"]=str(worker)
        command=[sys.executable,"-m","dp_SA.unimodal_logit_confidence.score_unimodal","--output-root",str(root),"--num-gpus",str(num_gpus),"--worker-id",str(worker)]
        if resume: command.append("--resume")
        commands.append(subprocess.Popen(command,cwd=Path(__file__).resolve().parents[2],env=env))
    codes=[process.wait() for process in commands]
    if any(codes): raise RuntimeError(f"Scoring worker failure: {codes}")
    specs=unique_specs(load_jsonl(root/"shared/manifests/probe_manifest.jsonl")); rows=[row for worker in range(num_gpus) for row in load_jsonl(root/f"unimodal_confidence/artifacts/raw_scores/shard_{worker}.jsonl")]
    keys=[tuple(row["stable_key"]) for row in rows]; expected={spec_key(s) for s in specs}
    if len(keys)!=len(set(keys)) or set(keys)!=expected: raise ValueError("Shard merge has duplicates or omissions")
    policies={row["tokenization_policy"] for row in rows}
    if len(policies)!=1: raise ValueError("Workers used different tokenization policies")
    rows.sort(key=lambda r:tuple(r["stable_key"])); atomic_jsonl(root/"unimodal_confidence/artifacts/raw_scores/unimodal_scores.jsonl",rows)
    summary={"status":"complete","unique_key_count":len(rows),"text_count":sum(r["modality"]=="text" for r in rows),"image_count":sum(r["modality"]=="image" for r in rows),"tokenization_policy":next(iter(policies)),"num_gpus":num_gpus}
    atomic_json(root/"unimodal_confidence/progress/score_unimodal.json",summary); return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-root",default=str(RESULTS_ROOT)); parser.add_argument("--num-gpus",type=int,choices=(1,2),default=1); parser.add_argument("--worker-id",type=int); parser.add_argument("--resume",action="store_true")
    args=parser.parse_args(argv); root=ensure_layout(args.output_root,resume=True)
    result=_worker(root,args.worker_id,args.num_gpus,resume=args.resume) if args.worker_id is not None else score_unimodal(root,num_gpus=args.num_gpus,resume=args.resume)
    print(json.dumps(result,ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
