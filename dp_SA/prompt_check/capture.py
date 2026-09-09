from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import AUDIT_MANIFEST, CANDIDATE_MANIFEST, CELL_MANIFEST, CONSTRUCTION_DISTRIBUTION, GEOMETRY_LAYERS, POSITIONS, PROBE_LAYERS, TEMPLATE_NAMES, TEST_MANIFEST
from .io_utils import append_jsonl, array_hash, atomic_json, atomic_jsonl, atomic_npz, canonical_hash, load_jsonl, sha256_file
from .runtime import capture_and_score, greedy_t3_audit, load_inference, prepare_case
from .templates import TEMPLATES
from .sampling import validation_audit, validation_geometry_cells, validation_test


def stable_shard(value: str, count: int) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:16], 16) % count


def smoke_geometry_cells() -> list[dict[str,Any]]:
    distribution=load_jsonl(CONSTRUCTION_DISTRIBUTION);eligible=sorted(r["answer"] for r in distribution if int(r["fold"])==0 and r["eligible_for_direction"] )[:4]
    selected=[]
    for answer in eligible:
        for side in ("high_text","high_image"):
            matches=[r for r in load_jsonl(CELL_MANIFEST) if int(r["fold"])==0 and r["answer"]==answer and r["sa_side"]==side]
            selected.extend(matches[:2])
    if len(eligible)<4 or any(not any(r["answer"]==a and r["sa_side"]==s for r in selected) for a in eligible for s in ("high_text","high_image")):raise ValueError("Cannot form frozen smoke geometry cells")
    return selected


def select_smoke_audit(rows:Sequence[dict[str,Any]],count:int=4)->list[dict[str,Any]]:
    selected=[];seen=set()
    for row in rows:
        if str(row["family_id"]) in seen:continue
        selected.append(row);seen.add(str(row["family_id"]))
        if len(selected)==count:return selected
    raise ValueError(f"Audit contains fewer than {count} distinct smoke families")


def build_capture_manifest(root: Path, *, include_audit: bool, include_geometry: bool, smoke: bool, validation_cases: int | None = None) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    test_rows=validation_test(validation_cases) if validation_cases else (load_jsonl(TEST_MANIFEST)[:4] if smoke else load_jsonl(TEST_MANIFEST))
    steering_ids={str(row["case_id"]) for row in test_rows}
    if include_audit:
        audit = validation_audit(validation_cases) if validation_cases else load_jsonl(AUDIT_MANIFEST)
        if smoke: audit = select_smoke_audit(audit,4)
        for row in audit:
            records[str(row["case_id"])] = {**row, "is_audit": True, "is_candidate": False}
    if include_geometry:
        candidates = load_jsonl(CANDIDATE_MANIFEST)
        if validation_cases:
            case_ids={str(case) for cell in validation_geometry_cells(validation_cases) for case in cell["case_ids"]};case_ids.update(steering_ids)
            candidates=[row for row in candidates if str(row["case_id"]) in case_ids]
        elif smoke:
            case_ids={str(case) for cell in smoke_geometry_cells() for case in cell["case_ids"]}
            case_ids.update(str(r["case_id"]) for r in load_jsonl(TEST_MANIFEST)[:4])
            candidates = [row for row in candidates if str(row["case_id"]) in case_ids]
        for row in candidates:
            case = str(row["case_id"]); previous = records.get(case, {})
            records[case] = {**row, "is_audit": bool(previous.get("is_audit")), "is_candidate": True}
    output = []
    for case in sorted(records):
        row = records[case]; requested: dict[str, set[int]] = {}
        if row["is_audit"]:
            for position in POSITIONS: requested.setdefault(position, set()).update(PROBE_LAYERS)
        if row["is_candidate"]: requested.setdefault("P1_LAT", set()).update(GEOMETRY_LAYERS)
        output.append({**row, "score_required":bool(row["is_audit"] or case in steering_ids), "requested_hidden": {key: sorted(value) for key, value in requested.items()}})
    path = root / "artifacts/manifests/capture_manifest.jsonl"
    if path.is_file():
        combined={str(row["case_id"]):row for row in load_jsonl(path)}
        for row in output:
            case=str(row["case_id"]);old=combined.get(case,{})
            requested={key:set(value) for key,value in old.get("requested_hidden",{}).items()}
            for key,value in row["requested_hidden"].items():requested.setdefault(key,set()).update(value)
            combined[case]={**old,**row,"is_audit":bool(old.get("is_audit") or row.get("is_audit")),"is_candidate":bool(old.get("is_candidate") or row.get("is_candidate")),"requested_hidden":{key:sorted(value) for key,value in requested.items()}}
        output=[combined[key] for key in sorted(combined)];atomic_jsonl(path,output)
    else: atomic_jsonl(path, output)
    return output


def _latest_rows(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for row in load_jsonl(path):
        if row.get("status") == "completed": result[str(row["case_id"])] = row
    return result


def capture_worker(root: Path, *, worker: int, num_gpus: int, templates: Sequence[str], resume: bool, smoke: bool = False) -> dict[str, Any]:
    records = load_jsonl(root / "artifacts/manifests/capture_manifest.jsonl")
    assigned = [row for row in records if stable_shard(str(row["case_id"]), num_gpus) == worker]
    output_path = root / f"artifacts/clean/capture.worker_{worker}.jsonl"; completed = _latest_rows(output_path)
    expected = {(template, str(row["case_id"])) for template in templates for row in assigned}
    requested_by_case={str(row["case_id"]):{f"{position}__L{layer}" for position,layers in row["requested_hidden"].items() for layer in layers} for row in assigned}
    def row_complete(row:dict[str,Any])->bool:
        case=str(row["case_id"]);needs_greedy=smoke and str(row.get("template"))=="T3" and bool(next((r.get("is_audit") for r in assigned if str(r["case_id"])==case),False))
        return row.get("status")=="completed" and requested_by_case.get(case,set()).issubset(set(row.get("hidden_keys",[]))) and (not needs_greedy or row.get("greedy_parse_status")!="not_audited")
    complete_keys = {(str(row["template"]), str(row["case_id"])) for row in load_jsonl(output_path) if row_complete(row)}
    if expected.issubset(complete_keys): return {"worker": worker, "status": "complete", "new_gpu_forwards": 0, "resumed_noop": True}
    inference, modules, tokenizer, device, processor = load_inference(); forwards = 0; started = time.time()
    for template in templates:
        spec = TEMPLATES[template]
        template_done = {(str(row["template"]), str(row["case_id"])) for row in load_jsonl(output_path) if row_complete(row)}
        for record in assigned:
            key = (template, str(record["case_id"]))
            if key in template_done:
                if not resume: raise FileExistsError(f"Capture exists; use --resume: {key}")
                continue
            inputs, rendered, located = prepare_case(inference.processor, tokenizer, device, record, spec)
            hidden, score = capture_and_score(inference.model, modules, tokenizer, inputs, located, spec, record["requested_hidden"],score_required=bool(record.get("score_required",True)))
            forwards += 6 if spec.kind == "labels" and record.get("score_required",True) else 1
            greedy={"greedy_parse_status":"not_audited"}
            if smoke and template=="T3" and record.get("is_audit"):
                greedy=greedy_t3_audit(inference.model,tokenizer,inputs);forwards += 1
            relative = Path("artifacts/hidden") / template / f"{record['case_id']}.npz"; destination = root / relative
            atomic_npz(destination, hidden)
            result = {
                "status": "completed", "template": template, "case_id": record["case_id"], "family_id": record["family_id"], "item_id": str(record["item_id"]),
                "condition": record["condition"], "answer": record.get("phase0_raw_answer"), "is_audit": record["is_audit"], "is_candidate": record["is_candidate"],
                "answer_matches_text": record.get("answer_matches_text"), "answer_matches_image": record.get("answer_matches_image"),
                "positions": located, "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(), "hidden_file": str(relative), "hidden_sha256": sha256_file(destination),
                "hidden_keys": sorted(hidden), "hidden_tensor_sha256": {name: array_hash(value) for name, value in hidden.items()},
                "processor": processor, **greedy, **score,
            }
            append_jsonl(output_path, result); template_done.add(key)
            if forwards % 10 == 0: atomic_json(root / f"progress/capture_worker_{worker}.json", {"status": "running", "worker": worker, "new_gpu_forwards": forwards, "last": key, "elapsed_seconds": time.time()-started})
    summary = {"worker": worker, "status": "complete", "new_gpu_forwards": forwards, "resumed_noop": forwards == 0, "elapsed_seconds": time.time()-started}
    atomic_json(root / f"progress/capture_worker_{worker}.json", summary); return summary


def merge_capture(root: Path, *, num_gpus: int, templates: Sequence[str]) -> dict[str, Any]:
    records = load_jsonl(root / "artifacts/manifests/capture_manifest.jsonl")
    latest={}
    for worker in range(num_gpus):
        for row in load_jsonl(root / f"artifacts/clean/capture.worker_{worker}.jsonl"):
            if row.get("status") == "completed" and row.get("template") in templates:latest[row["template"],str(row["case_id"])]=row
    rows=list(latest.values())
    keys = [(row["template"], row["case_id"]) for row in rows]
    expected = len(records) * len(templates)
    if len(keys) != len(set(keys)) or len(keys) != expected: raise ValueError(f"Capture merge incomplete/duplicate: rows={len(keys)} expected={expected}")
    rows.sort(key=lambda row: (row["template"], row["case_id"])); atomic_jsonl(root / "artifacts/clean/capture.jsonl", rows)
    result = {"status": "complete", "case_template_count": len(rows), "unique_case_count": len(records), "templates": list(templates), "num_workers": num_gpus}
    atomic_json(root / "progress/capture.json", result); return result


def run_capture(root: Path, *, num_gpus:int, templates:Sequence[str], resume:bool, smoke:bool, validation_cases:int|None=None, include_audit:bool=True, include_geometry:bool=True)->dict[str,Any]:
    root.mkdir(parents=True, exist_ok=True); build_capture_manifest(root, include_audit=include_audit, include_geometry=include_geometry, smoke=smoke,validation_cases=validation_cases)
    if num_gpus == 1:
        workers = [capture_worker(root, worker=0, num_gpus=1, templates=templates, resume=resume, smoke=smoke)]
    else:
        processes = []
        for worker in range(num_gpus):
            environment = dict(os.environ); environment["CUDA_VISIBLE_DEVICES"] = str(worker)
            command = [sys.executable, "-m", "dp_SA.prompt_check.capture", "--worker", str(worker), "--num-gpus", str(num_gpus), "--output-root", str(root), "--templates", *templates]
            if resume: command.append("--resume")
            if smoke: command.append("--smoke")
            processes.append(subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2], env=environment))
        codes = [process.wait() for process in processes]
        if any(codes): raise RuntimeError(f"Capture workers failed: {codes}")
        workers = [json.loads((root / f"progress/capture_worker_{worker}.json").read_text()) for worker in range(num_gpus)]
    return {**merge_capture(root, num_gpus=num_gpus, templates=templates), "workers": workers}


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--worker",type=int,required=True); parser.add_argument("--num-gpus",type=int,choices=(1,2),required=True); parser.add_argument("--output-root",required=True); parser.add_argument("--templates",nargs="+",choices=TEMPLATE_NAMES,required=True); parser.add_argument("--resume",action="store_true");parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args(argv); result=capture_worker(Path(args.output_root),worker=args.worker,num_gpus=args.num_gpus,templates=args.templates,resume=args.resume,smoke=args.smoke); print(json.dumps(result,ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
