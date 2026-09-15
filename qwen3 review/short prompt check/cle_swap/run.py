from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
SHORT_ROOT = HERE.parent
REVIEW_ROOT = SHORT_ROOT.parent
REPOSITORY_ROOT = REVIEW_ROOT.parent
for p in (REPOSITORY_ROOT, REVIEW_ROOT, SHORT_ROOT):
    if str(p) not in sys.path: sys.path.insert(0, str(p))

import numpy as np
import torch

from AttentionBlock.run import class_margin
from SA_trajectory.PANL2CLE.hooks import PANLCLEMediationHook
from Steering.hooks import SelectedHiddenCapture
from Steering.runtime import load_qwen3_inference
from capture.run import _context as short_context
from capture_reverse.run import _context as reverse_context
from capture_reverse.scoring import REVERSE_MIDPOINTS, reverse_soft_sa_from_logits
from dp_SA.config import MIDPOINTS
from dp_SA.io_utils import atomic_json, atomic_jsonl, load_jsonl, sha256_file
from dp_SA.soft_score import class_token_ids, soft_sa_from_logits
from layer_metacognition.model_adapter import model_input_device, resolve_language_modules, run_logits_forward
from sa_trajectory.panl2cle.contracts import atomic_bf16_npz, load_bf16

from .config import (DIRECTIONS, EXPECTED_CASES, HIDDEN_SIZE, LAYERS, LOGIT_PARITY_ATOL,
                     MODEL_PATH, NUM_LAYERS, OUTPUT_ROOT, REVERSE_CAPTURE, SHORT_CAPTURE)


def _rows(path: Path):
    return {r["case_id"]: r for r in load_jsonl(path) if r.get("status") == "completed"}


def _score(kind: str, logits: torch.Tensor, ids: Sequence[int]):
    return soft_sa_from_logits(logits, ids) if kind == "short" else reverse_soft_sa_from_logits(logits, ids)


def _raw_hard(kind: str, score: dict):
    return int(score["argmax_hard_class"] if kind == "short" else score["raw_argmax_class"])


def _context(runtime: Any, kind: str, row: dict):
    fn = short_context if kind == "short" else reverse_context
    _prompt, _rendered, _messages, inputs, located = fn(runtime.processor, row, model_input_device(runtime))
    positions = {"CLE": int(located["P1_CLASS_LIST_END"]["processed_index"]),
                 "SAC": int(located["P1_SAC"]["processed_index"])}
    return inputs, located, positions


def _clean(runtime, modules, kind, row, ids):
    inputs, located, positions = _context(runtime, kind, row)
    capture = SelectedHiddenCapture(modules, positions={"CLE": positions["CLE"]}, layers=LAYERS,
                                    prefill_sequence_length=int(inputs.input_ids.shape[1]))
    with capture:
        vocab_logits = run_logits_forward(runtime.model, inputs, [positions["SAC"]], modules)[positions["SAC"]]
    hidden = capture.validate()["CLE"]
    score = _score(kind, vocab_logits, ids)
    class_logits = torch.tensor(score["class_logits"], dtype=torch.float64)
    return inputs, located, positions, class_logits, score, hidden, capture.diagnostics()


def _patched(runtime, modules, kind, inputs, positions, ids, layer, source):
    hook = PANLCLEMediationHook(
        modules, prefill_sequence_length=int(inputs.input_ids.shape[1]),
        panl_position=0, cle_position=positions["CLE"], patch_layer=layer,
        patch_source=source, capture_cle_layers=(layer,),
    )
    with hook:
        vocab_logits = run_logits_forward(runtime.model, inputs, [positions["SAC"]], modules)[positions["SAC"]]
    score = _score(kind, vocab_logits, ids)
    return torch.tensor(score["class_logits"], dtype=torch.float64), score, hook


def _parity(kind, logits, score, stored):
    err = max(abs(float(a)-float(b)) for a,b in zip(logits.tolist(), stored["class_logits"]))
    soft = abs(float(score["soft_sa_image_score"])-float(stored["soft_sa_image_score"]))
    raw = _raw_hard(kind, score)
    stored_raw = int(stored["argmax_hard_class"] if kind == "short" else stored["raw_argmax_class"])
    return {"logit_max_abs_error": err, "soft_sa_abs_error": soft, "raw_hard_equal": raw == stored_raw,
            "passed": err <= LOGIT_PARITY_ATOL and soft <= LOGIT_PARITY_ATOL and raw == stored_raw}


def _trial(case, side, direction, layer, target_kind, source_kind, target_score, source_score,
           swap_score, target_logits, swap_logits, hook, fingerprint):
    target_sa=float(target_score["soft_sa_image_score"]); source_sa=float(source_score["soft_sa_image_score"])
    swapped_sa=float(swap_score["soft_sa_image_score"]); delta=swapped_sa-target_sa
    target_raw=_raw_hard(target_kind,target_score); swap_raw=_raw_hard(target_kind,swap_score)
    target_midpoints=np.asarray(MIDPOINTS if target_kind=="short" else REVERSE_MIDPOINTS)
    source_probs=np.asarray(source_score["class_probabilities"],dtype=float)
    raw_expected=float(source_probs @ target_midpoints)
    semantic_gap=source_sa-target_sa; raw_gap=raw_expected-target_sa
    semantic_distance_change=abs(swapped_sa-source_sa)-abs(target_sa-source_sa)
    raw_distance_change=abs(swapped_sa-raw_expected)-abs(target_sa-raw_expected)
    clean_margin=class_margin(target_logits.tolist(),target_raw)
    swap_margin=class_margin(swap_logits.tolist(),target_raw)
    return {
        "status":"completed","case_id":case,"test_side":side,"direction":direction,
        "layer":layer,"target_prompt":target_kind,"source_prompt":source_kind,
        "target_clean_sa":target_sa,"source_clean_sa":source_sa,"swapped_sa":swapped_sa,
        "delta_sa":delta,"abs_delta_sa":abs(delta),
        "target_clean_logits":target_logits.tolist(),"swapped_logits":swap_logits.tolist(),
        "target_clean_probabilities":target_score["class_probabilities"],
        "swapped_probabilities":swap_score["class_probabilities"],
        "target_clean_raw_class":target_raw,"swapped_raw_class":swap_raw,
        "target_clean_canonical_class":int(target_score["argmax_hard_class"]),
        "swapped_canonical_class":int(swap_score["argmax_hard_class"]),
        "token_changed":int(target_raw!=swap_raw),"hard_class_changed":int(int(target_score["argmax_hard_class"])!=int(swap_score["argmax_hard_class"])),
        "clean_margin":clean_margin,"swapped_margin":swap_margin,"logit_change_diff":clean_margin-swap_margin,
        "logit_linf_change":float(np.max(np.abs(np.asarray(swap_logits.tolist())-np.asarray(target_logits.tolist())))),
        "semantic_gap":semantic_gap,"semantic_movement":delta*np.sign(semantic_gap),
        "semantic_distance_change":semantic_distance_change,
        "moved_toward_source":int(semantic_distance_change<0),
        "source_raw_expected_in_target_scale":raw_expected,"raw_label_gap":raw_gap,
        "raw_label_movement":delta*np.sign(raw_gap),"raw_label_distance_change":raw_distance_change,
        "semantic_preference_contrast":raw_distance_change-semantic_distance_change,
        "hook":hook.diagnostics(),"fingerprint":fingerprint,
    }


def run(output_root: Path=OUTPUT_ROOT, *, resume=False, smoke=False):
    root=Path(output_root).resolve(); manifest=load_jsonl(root/"artifacts/manifests/test_manifest.jsonl")
    if len(manifest)!=EXPECTED_CASES: raise ValueError("Prepare the 50-case manifest first")
    if smoke:
        manifest=[next(r for r in manifest if r["test_side"]==s) for s in ("text_side","image_side")]
    short=_rows(SHORT_CAPTURE/"results.jsonl"); reverse=_rows(REVERSE_CAPTURE/"results.jsonl")
    fingerprint=json.loads((root/"fingerprint.json").read_text())["fingerprint"]
    runtime=load_qwen3_inference(MODEL_PATH); modules=resolve_language_modules(runtime.model)
    if (modules.num_hidden_layers,modules.hidden_size)!=(NUM_LAYERS,HIDDEN_SIZE): raise RuntimeError("Unexpected model architecture")
    ids=class_token_ids(runtime.processor.tokenizer); new=0; self_audits=[]; started=time.time()
    trial_dir=root/"artifacts/trials"; hidden_dir=root/"artifacts/hidden"; trial_dir.mkdir(parents=True,exist_ok=True)
    for entry in manifest:
        case=entry["case_id"]; clean={}; contexts={}; hidden={}
        expected_case_trials = [trial_dir/f"{case}__{direction}__L{layer}.json" for layer in LAYERS for direction in DIRECTIONS]
        if resume and not smoke and all(path.exists() for path in expected_case_trials):
            continue
        for kind,stored in (("short",short[case]),("reverse",reverse[case])):
            inputs,located,pos,logits,score,hiddens,diag=_clean(runtime,modules,kind,stored,ids)
            parity=_parity(kind,logits,score,stored)
            if not parity["passed"]: raise RuntimeError(f"Clean parity failed {case}/{kind}: {parity}")
            path=hidden_dir/f"{case}__{kind}__clean_cle_bf16.npz"
            atomic_bf16_npz(path,{f"CLE_L{x}":hiddens[x].cpu() for x in LAYERS})
            clean[kind]=(logits,score); contexts[kind]=(inputs,located,pos); hidden[kind]=path
            atomic_json(trial_dir/f"{case}__{kind}__clean.json",{
                "status":"completed","condition":"clean","case_id":case,"test_side":entry["test_side"],
                "prompt":kind,"class_logits":logits.tolist(),"class_probabilities":score["class_probabilities"],
                "soft_sa_image_score":float(score["soft_sa_image_score"]),"raw_class":_raw_hard(kind,score),
                "canonical_class":int(score["argmax_hard_class"]),"positions":located,"capture":diag,
                "hidden_file":str(path.relative_to(root)),"hidden_sha256":sha256_file(path),"parity":parity,"fingerprint":fingerprint})
            new+=1
        if contexts["short"][2] != contexts["reverse"][2]: raise RuntimeError(f"Cross-prompt position mismatch: {case}")
        for layer in LAYERS:
            if smoke:
                for kind in ("short","reverse"):
                    source=load_bf16(hidden[kind],f"CLE_L{layer}")
                    logits,score,hook=_patched(runtime,modules,kind,contexts[kind][0],contexts[kind][2],ids,layer,source)
                    base_logits,base_score=clean[kind]
                    err=max(abs(float(a)-float(b)) for a,b in zip(logits.tolist(),base_logits.tolist()))
                    prob=max(abs(float(a)-float(b)) for a,b in zip(score["class_probabilities"],base_score["class_probabilities"]))
                    sa=abs(float(score["soft_sa_image_score"])-float(base_score["soft_sa_image_score"]))
                    audit={"case_id":case,"prompt":kind,"layer":layer,"logit_max_abs_error":err,
                           "probability_max_abs_error":prob,"soft_sa_abs_error":sa,
                           "hard_equal":_raw_hard(kind,score)==_raw_hard(kind,base_score),"hook":hook.diagnostics()}
                    audit["passed"]=max(err,prob,sa)<=LOGIT_PARITY_ATOL and audit["hard_equal"]
                    if not audit["passed"]: raise RuntimeError(f"Self swap gate failed: {audit}")
                    self_audits.append(audit); new+=1
            for direction,target,source_kind in (("reverse_to_short","short","reverse"),("short_to_reverse","reverse","short")):
                dest=trial_dir/f"{case}__{direction}__L{layer}.json"
                if resume and dest.exists(): continue
                source=load_bf16(hidden[source_kind],f"CLE_L{layer}")
                logits,score,hook=_patched(runtime,modules,target,contexts[target][0],contexts[target][2],ids,layer,source)
                row=_trial(case,entry["test_side"],direction,layer,target,source_kind,
                           clean[target][1],clean[source_kind][1],score,clean[target][0],logits,hook,fingerprint)
                atomic_json(dest,row); new+=1
        atomic_json(root/"progress"/("smoke_run.json" if smoke else "run.json"),{
            "status":"running","cases_completed":manifest.index(entry)+1,"case_target":len(manifest),
            "new_forwards":new,"elapsed_seconds":time.time()-started})
    if smoke:
        atomic_json(root/"progress/self_swap_gate.json",{"status":"passed","checks":len(self_audits),"audits":self_audits})
    swaps=[json.loads(p.read_text()) for p in trial_dir.glob("*__L*.json") if json.loads(p.read_text()).get("fingerprint")==fingerprint]
    if not smoke and len(swaps)!=EXPECTED_CASES*len(LAYERS)*len(DIRECTIONS):
        raise RuntimeError(f"Formal swap grid incomplete: {len(swaps)}")
    atomic_jsonl(root/"artifacts/trials.jsonl",sorted(swaps,key=lambda r:(r["case_id"],r["direction"],r["layer"])))
    result={"status":"complete","smoke":smoke,"case_count":len(manifest),"swap_trials":len(swaps),"new_forwards":new}
    atomic_json(root/"progress"/("smoke_run.json" if smoke else "run.json"),result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--output-root",type=Path,default=OUTPUT_ROOT);p.add_argument("--resume",action="store_true");p.add_argument("--smoke",action="store_true")
    a=p.parse_args(argv);print(json.dumps(run(a.output_root,resume=a.resume,smoke=a.smoke),ensure_ascii=False));return 0
if __name__=="__main__": raise SystemExit(main())
