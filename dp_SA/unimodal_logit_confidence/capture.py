from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from confidence_test.runtime_imports import load_runtime
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.model_adapter import model_input_device, resolve_language_modules, run_hooked_forward
from dp_SA.positions import locate_phase1_positions
from dp_SA.prompts import SA_PREFILL, phase1_prompt

from .config import (DATASET_PATH, HIDDEN_DEFINITION, INFERENCE_PATH, MODEL_PATH, PROBE_LAYERS, PROBE_POSITIONS, RESULTS_ROOT,
                     SOURCE_CAPTURE, SOURCE_CAPTURE_AUDIT, SOURCE_CONFIG, SOURCE_HIDDEN_ROOT, SUPPLEMENT_CAPTURE, SUPPLEMENT_HIDDEN_ROOT)
from .io_utils import atomic_json, atomic_jsonl, ensure_layout, load_jsonl, sha256_file, stable_shard, validate_fingerprint


def hidden_key(position: str, layer: int) -> str: return f"{position}__L{layer}"


def tensor_sha256(value: np.ndarray) -> str: return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _atomic_npz(path: Path, arrays: dict[str,np.ndarray]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); fd,temp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent); os.close(fd)
    try:
        with open(temp,"wb") as handle: np.savez(handle,**arrays); handle.flush(); os.fsync(handle.fileno())
        os.replace(temp,path)
    except Exception:
        try: os.unlink(temp)
        except FileNotFoundError: pass
        raise


def _messages(prompt: str, image_path: str) -> list[dict[str,Any]]:
    path=str(Path(image_path).resolve())
    return [{"role":"user","content":[{"type":"image","image":path},{"type":"text","text":prompt}]},{"role":"assistant","content":[{"type":"text","text":SA_PREFILL}]}]


def _source_index(rows: Sequence[dict[str,Any]], source_root: Path, source_name: str) -> dict[str,dict[str,Any]]:
    output={}
    for row in rows:
        if row.get("status") != "completed": continue
        path=source_root/str(row["hidden_file"])
        if not path.is_file(): continue
        output[str(row["case_id"])]={"row":row,"path":path,"source":source_name}
    return output


def build_reuse_manifest(root: Path, records: Sequence[dict[str,Any]]) -> tuple[list[dict[str,Any]],dict[str,Any]]:
    source_config=json.loads(SOURCE_CONFIG.read_text()); audit=json.loads(SOURCE_CAPTURE_AUDIT.read_text())
    if source_config.get("hidden_definition") != HIDDEN_DEFINITION or audit.get("status") != "passed": raise ValueError("Historical capture is incompatible")
    if Path(source_config.get("model","")).resolve() != MODEL_PATH.resolve() or Path(source_config.get("dataset","")).resolve() != DATASET_PATH.resolve(): raise ValueError("Historical model or dataset is incompatible")
    primary=_source_index(load_jsonl(SOURCE_CAPTURE),SOURCE_HIDDEN_ROOT,"panl_information")
    supplement=_source_index(load_jsonl(SUPPLEMENT_CAPTURE),SUPPLEMENT_HIDDEN_ROOT,"checkpoint_steering")
    requested=[hidden_key(p,l) for p in PROBE_POSITIONS for l in PROBE_LAYERS]; manifests=[]; reused=0; missing=0; supplementary=0; file_hash_cache={}
    for record in records:
        case=str(record["case_id"]); sources={}; candidates=[primary.get(case),supplement.get(case)]
        for candidate in candidates:
            if candidate is None: continue
            row=candidate["row"]
            if row.get("phase1_prompt_hash") and row["phase1_prompt_hash"] != record["phase1_prompt_hash"]: continue
            for position in PROBE_POSITIONS:
                if position in row.get("positions",{}) and row["positions"][position]["token_id"] != record["positions"][position]["token_id"]: raise ValueError(f"Position token mismatch: {case} {position}")
            with np.load(candidate["path"]) as payload:
                for key in requested:
                    if key in sources or key not in payload: continue
                    vector=np.asarray(payload[key])
                    if vector.dtype != np.float16 or vector.ndim != 1 or not np.isfinite(vector).all(): raise ValueError(f"Invalid reusable tensor: {case} {key}")
                    path_text=str(candidate["path"].resolve())
                    if path_text not in file_hash_cache: file_hash_cache[path_text]=sha256_file(candidate["path"])
                    sources[key]={"source":candidate["source"],"path":path_text,"file_sha256":file_hash_cache[path_text],"tensor_sha256":tensor_sha256(vector)}
                    reused += 1; supplementary += int(candidate["source"]=="checkpoint_steering")
        absent=sorted(set(requested)-set(sources)); missing += len(absent)
        manifests.append({"case_id":case,"item_id":record["item_id"],"family_id":record["family_id"],"split":record["split"],"cell_sources":sources,"missing_keys":absent,"hidden_definition":HIDDEN_DEFINITION})
    summary={"record_count":len(records),"required_cell_count":len(records)*len(requested),"reused_cell_count":reused,"supplementary_reused_cell_count":supplementary,"missing_cell_count":missing,"records_requiring_forward":sum(bool(r["missing_keys"]) for r in manifests)}
    atomic_jsonl(root/"confidence_probe/artifacts/hidden/reuse_manifest.jsonl",manifests); atomic_json(root/"confidence_probe/artifacts/hidden/reuse_audit.json",summary)
    return manifests,summary


def _position_parity(record: dict[str,Any], located: dict[str,Any]) -> None:
    indices=[]
    for position in (*PROBE_POSITIONS,"P1_SAC"):
        old=record["positions"][position]; new=located[position]
        for field in ("processed_index","token_id","token_text"):
            if old[field] != new[field]: raise ValueError(f"Position parity failed: {record['case_id']} {position} {field}")
        indices.append(int(new["processed_index"]))
    if not all(a<b for a,b in zip(indices,indices[1:])): raise ValueError(f"Causal order failed: {record['case_id']}")
    if located["P1_LAT"]["processed_index"] != located["phase1_answer_span"][1]-1 or "\n" not in located["P1_PANL"]["token_text"]: raise ValueError(f"LAT/PANL definition failed: {record['case_id']}")


def _worker(root: Path, worker_id: int, num_gpus: int, *, resume: bool) -> dict[str,Any]:
    records={str(r["case_id"]):r for r in load_jsonl(root/"shared/manifests/probe_manifest.jsonl")}; reuse=[r for r in load_jsonl(root/"confidence_probe/artifacts/hidden/reuse_manifest.jsonl") if stable_shard(("hidden",r["case_id"]),num_gpus)==worker_id and r["missing_keys"]]
    result_path=root/f"confidence_probe/artifacts/hidden/shard_{worker_id}.jsonl"; existing=load_jsonl(result_path,repair_trailing=resume); completed={r["case_id"] for r in existing}; output=list(existing)
    if completed=={r["case_id"] for r in reuse}: return {"worker":worker_id,"count":len(reuse),"resumed_noop":True}
    runtime=load_runtime(INFERENCE_PATH); inference=runtime.QwenVLInference(str(MODEL_PATH)); modules=resolve_language_modules(inference.model); tokenizer=getattr(inference.processor,"tokenizer",inference.processor)
    for item in reuse:
        case=item["case_id"]
        if case in completed: continue
        record=records[case]; answer=str(record["phase0_raw_answer"]); prompt=phase1_prompt(record["question"],record["text_clue"],answer)
        if prompt != record["phase1_prompt"]: raise ValueError(f"Frozen Phase1 prompt mismatch: {case}")
        messages=_messages(prompt,record["image_path"]); rendered=render_continued_assistant(inference.processor,messages,SA_PREFILL); inputs=prepare_multimodal_inputs(inference.processor,messages,rendered,device=model_input_device(inference)); located=locate_phase1_positions(tokenizer,rendered,inputs,answer); _position_parity(record,located)
        positions={name:int(located[name]["processed_index"]) for name in PROBE_POSITIONS}; forward=run_hooked_forward(inference.model,inputs,modules,positions)
        captured={hidden_key(position,layer):forward.hidden_by_name[position][layer].detach().float().cpu().numpy().astype(np.float16) for position in PROBE_POSITIONS for layer in PROBE_LAYERS}
        # Every historical overlap is a parity oracle; save only genuinely absent cells.
        by_path={}
        for key,source in item["cell_sources"].items(): by_path.setdefault(source["path"],[]).append(key)
        for source_path,keys in by_path.items():
            with np.load(source_path) as payload:
                for key in keys:
                    if not np.array_equal(np.asarray(payload[key]),captured[key]): raise ValueError(f"Reusable hidden parity failed: {case} {key}")
        arrays={key:captured[key] for key in item["missing_keys"]}; relative=Path("confidence_probe/artifacts/hidden")/f"shard_{worker_id}"/f"{case}.npz"; _atomic_npz(root/relative,arrays)
        row={"status":"completed","case_id":case,"worker_id":worker_id,"delta_file":str(relative),"delta_keys":sorted(arrays),"delta_file_sha256":sha256_file(root/relative),"positions":{name:located[name] for name in PROBE_POSITIONS},"hidden_definition":HIDDEN_DEFINITION}; output.append(row); completed.add(case); atomic_jsonl(result_path,sorted(output,key=lambda r:r["case_id"]))
    return {"worker":worker_id,"count":len(reuse)}


def capture(root: Path, *, num_gpus: int, resume: bool) -> dict[str,Any]:
    if num_gpus not in (1,2): raise ValueError("--num-gpus must be 1 or 2")
    if not torch.cuda.is_available() or torch.cuda.device_count()<num_gpus: raise RuntimeError(f"Requested {num_gpus} GPUs, visible={torch.cuda.device_count()}")
    records=load_jsonl(root/"shared/manifests/probe_manifest.jsonl"); reuse_path=root/"confidence_probe/artifacts/hidden/reuse_manifest.jsonl"
    if not reuse_path.is_file(): _manifest,audit=build_reuse_manifest(root,records)
    else: audit=json.loads((root/"confidence_probe/artifacts/hidden/reuse_audit.json").read_text())
    validate_fingerprint(root/"confidence_probe/progress/capture_config.json",{"probe_manifest_sha256":sha256_file(root/"shared/manifests/probe_manifest.jsonl"),"reuse_manifest_sha256":sha256_file(reuse_path),"model_config_sha256":sha256_file(MODEL_PATH/"config.json"),"positions":list(PROBE_POSITIONS),"layers":list(PROBE_LAYERS),"hidden_definition":HIDDEN_DEFINITION,"shard_policy":"sha256(canonical_json(['hidden',case_id])) % num_gpus","num_gpus":num_gpus},resume=resume)
    processes=[]
    for worker in range(num_gpus):
        env=dict(os.environ); env["CUDA_VISIBLE_DEVICES"]=str(worker); command=[sys.executable,"-m","dp_SA.unimodal_logit_confidence.capture","--output-root",str(root),"--num-gpus",str(num_gpus),"--worker-id",str(worker)]
        if resume: command.append("--resume")
        processes.append(subprocess.Popen(command,cwd=Path(__file__).resolve().parents[2],env=env))
    codes=[p.wait() for p in processes]
    if any(codes): raise RuntimeError(f"Capture worker failure: {codes}")
    rows=[r for worker in range(num_gpus) for r in load_jsonl(root/f"confidence_probe/artifacts/hidden/shard_{worker}.jsonl")]; requested={r["case_id"] for r in load_jsonl(reuse_path) if r["missing_keys"]}
    if len(rows)!=len({r["case_id"] for r in rows}) or {r["case_id"] for r in rows}!=requested: raise ValueError("Capture shards have duplicates or omissions")
    atomic_jsonl(root/"confidence_probe/artifacts/hidden/capture_results.jsonl",sorted(rows,key=lambda r:r["case_id"])); summary={"status":"complete","num_gpus":num_gpus,**audit}; atomic_json(root/"confidence_probe/progress/capture.json",summary); return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-root",default=str(RESULTS_ROOT)); parser.add_argument("--num-gpus",type=int,choices=(1,2),default=1); parser.add_argument("--worker-id",type=int); parser.add_argument("--resume",action="store_true"); parser.add_argument("--audit-only",action="store_true")
    args=parser.parse_args(argv); root=ensure_layout(args.output_root,resume=True)
    if args.audit_only: result=build_reuse_manifest(root,load_jsonl(root/"shared/manifests/probe_manifest.jsonl"))[1]
    else: result=_worker(root,args.worker_id,args.num_gpus,resume=args.resume) if args.worker_id is not None else capture(root,num_gpus=args.num_gpus,resume=args.resume)
    print(json.dumps(result,ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
