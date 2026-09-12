"""Independent L12--L17 steering extension for the image-polarity sentence end."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from dp_SA.SA_probe.config import TRAIN_MANIFEST
from dp_SA.SA_probe.runtime import load_inference, messages
from dp_SA.SA_probe.positions import locate_probe_positions
from dp_SA.prompts import SA_PREFILL
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.model_adapter import run_hooked_forward

from . import run_panl2cle_position as base
from .io_utils import array_hash, atomic_json, atomic_jsonl, atomic_npz, load_jsonl, sha256_file
from .manifests import prepare_manifests

POSITION = "P1_IMAGE_POLARITY_SENTENCE_END"
LAYERS = (12, 13, 14, 15, 16, 17)
ROOT = base.OUTPUT_ROOT / "P1_IMAGE_POLARITY_SENTENCE_END_L12_17"


def _configure() -> None:
    base.POSITIONS = (POSITION,)
    base.LAYERS = LAYERS
    base.OUTPUT_ROOT = ROOT


def _capture(root: Path, construction: Sequence[dict[str, Any]], config: dict[str, Any], *, resume: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = root / "artifacts/diagnostics/clean_capture.jsonl"
    existing = {str(r["case_id"]): r for r in load_jsonl(path) if r.get("status") == "completed"}
    expected = {str(r["case_id"]) for r in construction}
    if set(existing) == expected:
        out = {"status":"complete","case_count":len(existing),"new_gpu_forwards":0,"resumed_noop":True}
        atomic_json(root/"progress/capture.json",out); return list(existing.values()), out
    parent = ROOT.parent
    parent_clean = {str(r["case_id"]): r for r in load_jsonl(parent/"artifacts/diagnostics/clean_capture.jsonl") if r.get("status")=="completed"}
    source_capture = {str(r["case_id"]):r for r in load_jsonl(base.SOURCE_CAPTURE) if r.get("status")=="completed"}
    source_manifest = {str(r["case_id"]):r for r in load_jsonl(TRAIN_MANIFEST) if r.get("status")=="completed"}
    inference = modules = tokenizer = device = processor = None
    forwards = 0; started = time.time()
    try:
        inference, modules, tokenizer, device, processor = load_inference()
        ids = base.class_token_ids(tokenizer)
        for target in construction:
            case = str(target["case_id"])
            if case in existing: continue
            clean = parent_clean[case]
            source = source_capture.get(case)
            arrays: dict[str,np.ndarray] = {}
            if source is not None and case in source_manifest:
                base._validate_probe_identity(target, source_manifest[case], source)
                with np.load(base.SOURCE_CAPTURE_ROOT/source["hidden_file"]) as z:
                    for layer in (12,14,16):
                        arrays[f"{POSITION}__L{layer}"] = np.asarray(z[f"{POSITION}__L{layer}"], dtype=np.float16)
            wire = messages(target)
            rendered = render_continued_assistant(inference.processor, wire, SA_PREFILL)
            inputs = prepare_multimodal_inputs(inference.processor, wire, rendered, device=device)
            located = locate_probe_positions(tokenizer, rendered, inputs, str(target["phase0_raw_answer"]))
            if hashlib.sha256(rendered.encode()).hexdigest() != clean["rendered_prompt_sha256"]:
                raise ValueError(f"Rendered prompt changed: {case}")
            sac = int(located["P1_SAC"]["processed_index"])
            forward = run_hooked_forward(inference.model, inputs, modules, {POSITION:int(located[POSITION]["processed_index"])}, logits_positions=[sac])
            forwards += 1
            score = base.soft_sa_from_logits(forward.logits_by_position[sac], ids)
            base._validate_score(score, clean, case)
            for layer in (13,15,17):
                arrays[f"{POSITION}__L{layer}"] = forward.hidden_by_name[POSITION][layer].detach().float().cpu().numpy().astype(np.float16)
            # Missing SA_probe overlap keys (if any) are filled by the fresh forward.
            for layer in (12,14,16):
                arrays.setdefault(f"{POSITION}__L{layer}", forward.hidden_by_name[POSITION][layer].detach().float().cpu().numpy().astype(np.float16))
            rel=Path("artifacts/hidden")/f"{case}.npz"; atomic_npz(root/rel,arrays)
            row={**clean,"capture_source":"SA_probe_12_14_16_plus_fast_odd_layers","hidden_file":str(rel),"hidden_sha256":sha256_file(root/rel),"hidden_keys":sorted(arrays),"hidden_tensor_sha256":{k:array_hash(v) for k,v in arrays.items()},"config_fingerprint":config["fingerprint"],"processor":processor,"positions":{POSITION:located[POSITION],"P1_SAC":located["P1_SAC"]},"capture_forward":True}
            existing[case]=row; atomic_jsonl(path,sorted(existing.values(),key=lambda x:str(x["case_id"])))
    finally:
        if inference is not None: base._release(inference)
    if set(existing)!=expected: raise RuntimeError(f"Capture incomplete: {len(existing)}/{len(expected)}")
    result={"status":"complete","case_count":len(existing),"new_gpu_forwards":forwards,"resumed_noop":forwards==0,"elapsed_seconds":time.time()-started}
    atomic_json(root/"progress/capture.json",result); return list(existing.values()),result


def run_once(root: Path, *, smoke: bool, resume: bool) -> dict[str,Any]:
    _configure(); root.mkdir(parents=True,exist_ok=True)
    construction,test,source=prepare_manifests(root,smoke=smoke,resume=resume)
    config=base._config(root,construction,test,smoke=smoke); base._check_config(root,config,resume=resume)
    # The parent experiment supplies the already validated clean baselines for all 100 test cases.
    clean,capture=_capture(root,construction,config,resume=resume)
    parent_clean=ROOT.parent/"artifacts/diagnostics/clean_capture.jsonl"
    clean_all=load_jsonl(parent_clean)
    vectors,meta=base.build_vectors(root,construction,clean,config,resume=resume)
    steering=base.steer(root,test,clean_all,vectors,config)
    analysis=base.analyze(output_root=root,smoke=smoke,resume=resume,repeats=base.SMOKE_BOOTSTRAP_REPEATS if smoke else base.BOOTSTRAP_REPEATS,positions=(POSITION,),alphas=base.SMOKE_ALPHAS if smoke else base.ALPHAS,generic_summary=True)
    result={"status":"complete","smoke":smoke,"capture":capture,"steering":steering,"analysis":analysis,"vector_fingerprint":meta["fingerprint"],"selection_source":source}
    atomic_json(root/"progress/completion.json",result); return result


def main(argv: Sequence[str]|None=None)->int:
    parser=argparse.ArgumentParser(description="PANL2CLE image-polarity sentence end steering at L12-L17")
    parser.add_argument("--smoke",action="store_true"); parser.add_argument("--resume",action="store_true")
    args=parser.parse_args(argv)
    if args.smoke and args.resume: parser.error("--smoke and --resume are mutually exclusive")
    if args.smoke:
        root=ROOT/"smoke_tmp"/"round_1"
        result=run_once(root,smoke=True,resume=(root/"progress/config.json").exists())
        resumed=run_once(root,smoke=True,resume=True)
        report={"status":"passed","capture_forwards":result["capture"]["new_gpu_forwards"],"steering_forwards":result["steering"]["new_gpu_forwards"],"resume_noop":resumed["steering"]["new_gpu_forwards"]==0}
        atomic_json(root/"progress/smoke_report.json",report)
    else:
        result=run_once(ROOT,smoke=False,resume=args.resume)
    print(json.dumps(result if not args.smoke else report,ensure_ascii=False)); return 0


if __name__=="__main__": raise SystemExit(main())
