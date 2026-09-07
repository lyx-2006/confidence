from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from confidence_test.runtime_imports import load_runtime
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.model_adapter import model_input_device, resolve_language_modules, run_hooked_forward
from dp_SA.positions import locate_phase1_positions
from dp_SA.prompts import SA_PREFILL
from .processor import enforce_parent_fast_image_processor

from .config import (
    BASE_CAPTURE_CONFIG, BASE_CAPTURE_ROOT, BASE_CAPTURE_ROWS, CALIBRATED_SCORES,
    CHECKPOINT_ROOT, CLASS_CAPTURE_CONFIG, CLASS_CAPTURE_ROOT, CLASS_CAPTURE_ROWS,
    CONFIDENCE_CAPTURE, CONFIDENCE_REUSE, EXPECTED_HASHES, FORMAL_ONLY_SOURCES, HIDDEN_DEFINITION,
    HIDDEN_SIZE, IMAGE_TAU, IMAGE_TEMPERATURE, INFERENCE_PATH, JOINED_CONFIDENCE,
    LAYERS, MODEL_PATH, PARENT_FAST_CLEAN, PARENT_FAST_CONFIG, PARENT_PROCESSOR_FILE, POSITION_DEFINITIONS, POSITIONS, RIDGE_ALPHAS, SEED,
    SPLIT_AUDIT, TARGETS, TEXT_TAU, TEXT_TEMPERATURE, TRAIN_MANIFEST,
)
from .io_utils import (
    array_hash, atomic_json, atomic_jsonl, atomic_npz, canonical_hash, inventory,
    load_jsonl, semantic_lock, sha256_file, stable_shard, verify_inventory,
    require_output_root,
)


def hidden_key(position: str, layer: int) -> str:
    return f"{position}__L{int(layer)}"


def messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": [
            {"type": "image", "image": str(Path(row["image_path"]).resolve())},
            {"type": "text", "text": str(row["phase1_prompt"])},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": SA_PREFILL}]},
    ]


def position_definition_hashes() -> dict[str, str]:
    locator = Path(__file__).resolve().parents[2] / "positions.py"
    source = sha256_file(locator)
    return {name: canonical_hash({"position": name, "definition": definition, "locator_sha256": source}) for name, definition in POSITION_DEFINITIONS.items()}


def position_payload(located: dict[str, Any], rendered: str) -> dict[str, Any]:
    hashes = position_definition_hashes(); output = {}
    for name in POSITIONS:
        row = located[name]
        output[name] = {
            "processed_index": int(row["processed_index"]),
            "rendered_index": int(row["rendered_index"]),
            "token_id": int(row["token_id"]),
            "decoded_token": str(row["token_text"]),
            "position_definition_sha256": hashes[name],
        }
    indices = [output[name]["processed_index"] for name in POSITIONS]
    if not all(left < right for left, right in zip(indices, indices[1:])):
        raise ValueError(f"LAT<PANL<CLASS_LIST_END<SAC failed: {indices}")
    return {"positions": output, "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(), "causal_order_valid": True}


def _validate_frozen_sources(*, include_formal: bool = False) -> dict[str, str]:
    selected = {path: digest for path, digest in EXPECTED_HASHES.items() if include_formal or path not in FORMAL_ONLY_SOURCES}
    for path, digest in selected.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Frozen source mismatch: {path}")
    text = json.loads(TEXT_TEMPERATURE.read_text()); image = json.loads(IMAGE_TEMPERATURE.read_text())
    if not math.isclose(float(text["ece_optimal"]["temperature"]), TEXT_TAU, abs_tol=1e-15) or not math.isclose(float(image["ece_optimal"]["temperature"]), IMAGE_TAU, abs_tol=1e-15):
        raise ValueError("Frozen temperature changed")
    return {str(path.resolve()): digest for path, digest in selected.items()}


def prepare_training_records(root: Path, *, resume: bool) -> dict[str, Any]:
    root=require_output_root(root)
    frozen = _validate_frozen_sources(include_formal=False)
    manifests = load_jsonl(TRAIN_MANIFEST)
    if len(manifests) != 1112 or len({str(r["family_id"]) for r in manifests}) != 128:
        raise ValueError("Frozen train cardinality changed")
    joined = {str(r["case_id"]): r for r in load_jsonl(JOINED_CONFIDENCE)}
    prepared = []
    for row in manifests:
        case = str(row["case_id"]); confidence = joined[case]
        if str(confidence["image_hash"]) != str(row["image_sha256"]):
            raise ValueError(f"Image identity mismatch: {case}")
        ci = float(confidence["image_fixed_answer_confidence"]); ct = float(confidence["text_fixed_answer_confidence"])
        gl = float(confidence["image_fixed_answer_log_odds"]) - float(confidence["text_fixed_answer_log_odds"])
        if not math.isclose(gl, float(confidence["G_L"]), rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"G_L mismatch: {case}")
        if not math.isclose(float(confidence["text_temperature"]), TEXT_TAU, abs_tol=1e-15) or not math.isclose(float(confidence["image_temperature"]), IMAGE_TAU, abs_tol=1e-15):
            raise ValueError(f"Temperature mismatch: {case}")
        prepared.append({**row, "C_i": ci, "C_t": ct, "G_L": gl, "final_soft_sa": float(row["soft_sa_image_score"])})
    construction = [r for r in prepared if int(r["outer_fold"]) != 0]
    audit = [r for r in prepared if int(r["outer_fold"]) == 0]
    if (len(construction), len({r["family_id"] for r in construction}), len(audit), len({r["family_id"] for r in audit})) != (882, 103, 230, 25):
        raise ValueError("Frozen construction/audit split changed")
    split_rows = {"construction": construction, "audit": audit}
    overlaps = {}
    for field in ("case_id", "family_id", "item_id", "image_sha256"):
        overlaps[field] = len({str(r[field]) for r in construction} & {str(r[field]) for r in audit})
    if any(overlaps.values()): raise ValueError(f"Construction/audit leakage: {overlaps}")
    atomic_jsonl(root / "artifacts/manifests/construction_manifest.jsonl", construction)
    atomic_jsonl(root / "artifacts/manifests/audit_manifest.jsonl", audit)
    atomic_json(root / "artifacts/manifests/split_audit.json", {
        "status": "passed", "construction_cases": 882, "construction_families": 103,
        "audit_cases": 230, "audit_families": 25, "overlaps": overlaps,
        "formal_test_opened": False, "upstream_split_audit_sha256": frozen[str(SPLIT_AUDIT.resolve())],
    })
    package=Path(__file__).resolve().parent
    code_inventory=inventory(sorted(p for p in package.glob("*.py")))
    model_inventory=inventory([*(MODEL_PATH/name for name in ("config.json","tokenizer.json","tokenizer_config.json","preprocessor_config.json","model.safetensors.index.json")),PARENT_PROCESSOR_FILE])
    config = {
        "format_version": 1, "experiment": "lat_panl_sac_layerwise_trajectory", "seed": SEED,
        "positions": list(POSITIONS), "layers": list(LAYERS), "targets": list(TARGETS),
        "ridge_alphas": list(RIDGE_ALPHAS), "hidden_definition": HIDDEN_DEFINITION,
        "directions": ["confidence_raw", "confidence_parallel_sa", "confidence_perp_sa_natural_scale"],
        "perpendicular_term": "SA-subspace-orthogonal confidence-related component",
        "injection": {"position": "P1_LAT", "layer": 14, "epsilon": 0.5, "site": "block_output"},
        "temperatures": {"text": TEXT_TAU, "image": IMAGE_TAU},
        "formal_reference": {"processor_mode":"explicit_fast","clean_manifest":str(PARENT_FAST_CLEAN.resolve()),"clean_manifest_sha256":EXPECTED_HASHES[PARENT_FAST_CLEAN],"parent_config_sha256":EXPECTED_HASHES[PARENT_FAST_CONFIG]},
        "implementation_sha256": code_inventory, "model_processor_identity": model_inventory,
        "position_definition_hashes": position_definition_hashes(), "frozen_sources": frozen,
    }
    fingerprint = semantic_lock(root / "artifacts/config_and_fingerprint.json", config, resume=resume)
    return {"construction": construction, "audit": audit, "fingerprint": fingerprint, "frozen_sources": frozen}


def _completed_by_case(path: Path) -> dict[str, dict[str, Any]]:
    output = {}
    for row in load_jsonl(path):
        if row.get("status") == "completed": output[str(row["case_id"])] = row
    return output


def build_reuse_manifest(root: Path, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    root=require_output_root(root)
    requested = {hidden_key(position, layer) for position in POSITIONS for layer in LAYERS}
    confidence_reuse = {str(r["case_id"]): r for r in load_jsonl(CONFIDENCE_REUSE)}
    confidence_capture = {str(r["case_id"]): r for r in load_jsonl(CONFIDENCE_CAPTURE)}
    base = _completed_by_case(BASE_CAPTURE_ROWS); class_rows = _completed_by_case(CLASS_CAPTURE_ROWS)
    checkpoint_rows = {}
    checkpoint_identity = {}
    for name in ("construction_manifest.jsonl", "test_manifest.jsonl"):
        for row in load_jsonl(CHECKPOINT_ROOT / "artifacts/manifests" / name): checkpoint_identity[str(row["case_id"])] = row
    checkpoint_files = (
        CHECKPOINT_ROOT / "artifacts/diagnostics/clean_capture.jsonl",
        CHECKPOINT_ROOT / "artifacts/diagnostics/clean_capture_layer_12_16_22.jsonl",
    )
    for source in checkpoint_files:
        for row in load_jsonl(source):
            if row.get("status") == "completed": checkpoint_rows.setdefault(str(row["case_id"]), []).append(row)
    protected = {BASE_CAPTURE_CONFIG, BASE_CAPTURE_ROWS, CLASS_CAPTURE_CONFIG, CLASS_CAPTURE_ROWS, CONFIDENCE_REUSE, CONFIDENCE_CAPTURE, *checkpoint_files}
    manifests=[]; source_counter=Counter()
    for record in records:
        case=str(record["case_id"]); sources={}
        cr=confidence_reuse[case]; cc=confidence_capture.get(case)
        for key in sorted(requested):
            info=cr.get("cell_sources",{}).get(key)
            if info:
                path=Path(info["path"]); sources[key]={"source":"confidence_hidden","path":str(path.resolve()),"file_sha256":info["file_sha256"],"tensor_sha256":info["tensor_sha256"]}; protected.add(path)
            elif cc and key in cc.get("delta_keys",[]):
                path=SOURCE_ROOT / cc["delta_file"]
                with np.load(path) as payload: tensor=np.asarray(payload[key])
                sources[key]={"source":"confidence_hidden_delta","path":str(path.resolve()),"file_sha256":cc["delta_file_sha256"],"tensor_sha256":array_hash(tensor)}; protected.add(path)
        for label, row, source_root in (("base_capture",base.get(case),BASE_CAPTURE_ROOT),("class_list_capture",class_rows.get(case),CLASS_CAPTURE_ROOT)):
            if not row or row.get("phase1_prompt_hash") != record["phase1_prompt_hash"] or row.get("image_sha256") != record["image_sha256"]: continue
            path=source_root / row["hidden_file"]
            with np.load(path) as payload:
                for key in sorted(requested-set(sources)):
                    position=key.split("__",1)[0]
                    if key in payload.files and position in row.get("positions",{}):
                        tensor=np.asarray(payload[key]); sources[key]={"source":label,"path":str(path.resolve()),"file_sha256":sha256_file(path),"tensor_sha256":array_hash(tensor)}
            if any(v["path"]==str(path.resolve()) for v in sources.values()): protected.add(path)
        identity=checkpoint_identity.get(case)
        if identity and identity.get("phase1_prompt_hash")==record["phase1_prompt_hash"] and identity.get("image_sha256")==record["image_sha256"]:
            for row in checkpoint_rows.get(case,[]):
                path=CHECKPOINT_ROOT / row["hidden_file"]
                with np.load(path) as payload:
                    for key in sorted(requested-set(sources)):
                        position=key.split("__",1)[0]
                        if key in payload.files and position in row.get("positions",{}):
                            tensor=np.asarray(payload[key]); sources[key]={"source":"checkpoint_capture","path":str(path.resolve()),"file_sha256":row["hidden_sha256"],"tensor_sha256":array_hash(tensor)}
                if any(v["path"]==str(path.resolve()) for v in sources.values()): protected.add(path)
        for info in sources.values(): source_counter[info["source"]]+=1
        manifests.append({"case_id":case,"item_id":str(record["item_id"]),"family_id":str(record["family_id"]),"split":"audit" if int(record["outer_fold"])==0 else "construction","cell_sources":sources,"missing_keys":sorted(requested-set(sources)),"hidden_definition":HIDDEN_DEFINITION})
    summary={"record_count":len(records),"required_cell_count":len(records)*len(requested),"candidate_reuse_cell_count":sum(len(r["cell_sources"]) for r in manifests),"missing_cell_count":sum(len(r["missing_keys"]) for r in manifests),"records_requiring_forward":sum(bool(r["missing_keys"]) for r in manifests),"source_counts":dict(source_counter)}
    if len(records)==1112 and (summary["candidate_reuse_cell_count"],summary["missing_cell_count"],summary["records_requiring_forward"]) != (25891,31933,1112):
        raise ValueError(f"Historical hidden inventory changed: {summary}")
    atomic_jsonl(root/"artifacts/clean_hidden/reuse_manifest.jsonl",manifests)
    before=inventory(protected); atomic_json(root/"artifacts/diagnostics/source_hashes_before.json",before)
    atomic_json(root/"artifacts/diagnostics/hidden_reuse_audit.json",summary)
    return {**summary,"protected":before}


def _capture_worker(root: Path, worker: int, num_gpus: int, fingerprint: str) -> dict[str, Any]:
    records={str(r["case_id"]):r for split in ("construction","audit") for r in load_jsonl(root/f"artifacts/manifests/{split}_manifest.jsonl")}
    reuse=[r for r in load_jsonl(root/"artifacts/clean_hidden/reuse_manifest.jsonl") if stable_shard(r["case_id"],num_gpus)==worker]
    completed={str(r["case_id"]):r for r in load_jsonl(root/"artifacts/clean_hidden/capture_manifest.jsonl")}
    pending=[r for r in reuse if r["missing_keys"] and r["case_id"] not in completed]
    if not pending: return {"worker":worker,"new_gpu_forwards":0,"resumed_noop":True}
    runtime=load_runtime(INFERENCE_PATH); inference=runtime.QwenVLInference(str(MODEL_PATH)); enforce_parent_fast_image_processor(inference.processor)
    modules=resolve_language_modules(inference.model); tokenizer=getattr(inference.processor,"tokenizer",inference.processor); device=model_input_device(inference)
    worker_path=root/f"artifacts/clean_hidden/capture_manifest.worker_{worker}.jsonl"
    new=load_jsonl(worker_path); completed.update({str(r["case_id"]):r for r in new}); pending=[r for r in reuse if r["missing_keys"] and r["case_id"] not in completed]; forwards=0
    for item in pending:
        record=records[item["case_id"]]; wire=messages(record); rendered=render_continued_assistant(inference.processor,wire,SA_PREFILL)
        inputs=prepare_multimodal_inputs(inference.processor,wire,rendered,device=device); located=locate_phase1_positions(tokenizer,rendered,inputs,str(record["phase0_raw_answer"])); positions={name:int(located[name]["processed_index"]) for name in POSITIONS}
        forward=run_hooked_forward(inference.model,inputs,modules,positions,logits_positions=[positions["P1_SAC"]]); forwards+=1
        captured={hidden_key(position,layer):forward.hidden_by_name[position][layer].detach().float().cpu().numpy().astype(np.float16) for position in POSITIONS for layer in LAYERS}
        by_path={}
        for key,source in item["cell_sources"].items(): by_path.setdefault(source["path"],[]).append((key,source))
        incompatible=[]
        for path,pairs in by_path.items():
            if sha256_file(path)!=pairs[0][1]["file_sha256"]: raise ValueError(f"Historical file changed: {path}")
            with np.load(path) as payload:
                for key,source in pairs:
                    old=np.asarray(payload[key])
                    if old.dtype!=np.float16 or old.shape!=(HIDDEN_SIZE,) or array_hash(old)!=source["tensor_sha256"]: raise ValueError(f"Reusable hidden integrity failed: {item['case_id']} {key}")
                    if not np.array_equal(old,captured[key]): incompatible.append(key)
        arrays={key:captured[key] for key in sorted(set(item["missing_keys"])|set(incompatible))}; relative=Path("artifacts/clean_hidden/by_case")/f"{item['case_id']}.npz"; atomic_npz(root/relative,arrays)
        new.append({"status":"completed","case_id":item["case_id"],"worker":worker,"delta_file":str(relative),"delta_file_sha256":sha256_file(root/relative),"delta_keys":sorted(arrays),"candidate_compatible_keys":sorted(set(item["cell_sources"])-set(incompatible)),"candidate_incompatible_keys":sorted(incompatible),**position_payload(located,rendered),"hidden_definition":HIDDEN_DEFINITION,"config_fingerprint":fingerprint})
        atomic_jsonl(worker_path,new)
    return {"worker":worker,"new_gpu_forwards":forwards,"resumed_noop":forwards==0}


def capture_clean(root: Path, *, num_gpus: int, resume: bool, worker: int | None = None) -> dict[str, Any]:
    root=require_output_root(root)
    config=json.loads((root/"artifacts/config_and_fingerprint.json").read_text()); fingerprint=config["fingerprint"]
    if worker is not None: return _capture_worker(root,worker,num_gpus,fingerprint)
    if num_gpus not in (1,2) or not torch.cuda.is_available() or torch.cuda.device_count()<num_gpus: raise RuntimeError(f"Requested {num_gpus} GPU(s), visible={torch.cuda.device_count()}")
    processes=[]; started=time.time()
    for index in range(num_gpus):
        env=dict(os.environ);env["CUDA_VISIBLE_DEVICES"]=str(index)
        processes.append(subprocess.Popen([sys.executable,"-m","dp_SA.confidence_steering.trajectory.capture","--output-root",str(root),"--worker",str(index),"--num-gpus",str(num_gpus)],cwd=Path(__file__).resolve().parents[3],env=env))
    codes=[p.wait() for p in processes]
    if any(codes): raise RuntimeError(f"Clean capture worker failure: {codes}")
    rows=[]; reports=[]
    for index in range(num_gpus):
        rows+=load_jsonl(root/f"artifacts/clean_hidden/capture_manifest.worker_{index}.jsonl"); reports.append(json.loads((root/f"progress/capture_worker_{index}.json").read_text()))
    expected={r["case_id"] for r in load_jsonl(root/"artifacts/clean_hidden/reuse_manifest.jsonl") if r["missing_keys"]}
    if {r["case_id"] for r in rows}!=expected or len(rows)!=len(expected): raise ValueError("Clean capture merge incomplete")
    atomic_jsonl(root/"artifacts/clean_hidden/capture_manifest.jsonl",sorted(rows,key=lambda r:r["case_id"]))
    reuse_rows=load_jsonl(root/"artifacts/clean_hidden/reuse_manifest.jsonl"); captured_by={r["case_id"]:r for r in rows}
    for item in reuse_rows:
        bad=set(captured_by[item["case_id"]].get("candidate_incompatible_keys",[]))
        for key in bad:item["cell_sources"].pop(key,None)
        item["missing_keys"]=sorted(set(item["missing_keys"])|bad);item["actual_reuse_keys"]=sorted(item["cell_sources"]);item["compatibility_validated_by_float16_bitwise_parity"]=True
    atomic_jsonl(root/"artifacts/clean_hidden/reuse_manifest.jsonl",reuse_rows)
    actual=sum(len(r["cell_sources"]) for r in reuse_rows);atomic_json(root/"artifacts/diagnostics/hidden_reuse_validated.json",{"actual_reuse_cell_count":actual,"incompatible_candidate_cell_count":25891-actual,"new_saved_cell_count":57824-actual})
    result={"status":"complete","new_gpu_forwards":sum(r["new_gpu_forwards"] for r in reports),"resumed_noop":all(r["resumed_noop"] for r in reports),"elapsed_seconds":time.time()-started}
    atomic_json(root/"progress/capture.json",result);return result


class HiddenStore:
    def __init__(self, root: Path):
        self.root=root;self.reuse={r["case_id"]:r for r in load_jsonl(root/"artifacts/clean_hidden/reuse_manifest.jsonl")};self.capture={r["case_id"]:r for r in load_jsonl(root/"artifacts/clean_hidden/capture_manifest.jsonl")};self.cache={}
    def load(self,case:str,key:str)->np.ndarray:
        item=self.reuse[case]; source=item["cell_sources"].get(key)
        if source:path=Path(source["path"]);expected=source["tensor_sha256"]
        else:
            row=self.capture[case];path=self.root/row["delta_file"];expected=None
        p=str(path.resolve())
        if p not in self.cache:
            with np.load(path) as z:self.cache[p]={name:np.asarray(z[name]) for name in z.files}
        value=self.cache[p][key]
        if value.dtype!=np.float16 or value.shape!=(HIDDEN_SIZE,) or (expected and array_hash(value)!=expected):raise ValueError(f"Invalid hidden {case} {key}")
        return value.astype(np.float32)


def main(argv: Sequence[str] | None=None)->int:
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--output-root",required=True);parser.add_argument("--worker",type=int,required=True);parser.add_argument("--num-gpus",type=int,choices=(1,2),required=True)
    args=parser.parse_args(argv);root=Path(args.output_root);result=capture_clean(root,num_gpus=args.num_gpus,resume=True,worker=args.worker);atomic_json(root/f"progress/capture_worker_{args.worker}.json",result);print(json.dumps(result));return 0


if __name__=="__main__":raise SystemExit(main())
