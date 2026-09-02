from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import ECE_BINS, FIXED_EPSILON, RESULTS_ROOT, TEMPERATURE_GRID_SIZE, TEMPERATURE_MAX, TEMPERATURE_MIN
from .io_utils import atomic_csv, atomic_json, atomic_jsonl, canonical_hash, ensure_layout, load_jsonl
from .metrics import calibration_summary, ece, entropy_difficulty, score_metrics


def temperature_grid() -> np.ndarray:
    return np.unique(np.r_[np.geomspace(TEMPERATURE_MIN,TEMPERATURE_MAX,TEMPERATURE_GRID_SIZE),1.0])


def evaluate_temperature(rows: Sequence[dict[str,Any]], temperature: float) -> dict[str,Any]:
    converted=[]
    for row in rows:
        scores=[float(row["raw_candidate_scores"][name]) for name in row["answer_classes"]]
        metrics=score_metrics(row["answer_classes"],scores,row["target_answer"],temperature)
        converted.append({"correct":metrics["correct"],"chosen_confidence":metrics["chosen_confidence"],"nll":metrics["nll"],"brier":metrics["brier"]})
    return calibration_summary(converted)


def search_temperature(rows: Sequence[dict[str,Any]]) -> tuple[dict[str,Any],dict[str,Any],list[dict[str,Any]]]:
    trace=[]
    for temperature in temperature_grid():
        metrics=evaluate_temperature(rows,float(temperature)); trace.append({"temperature":float(temperature),**metrics})
    ece_best=min(trace,key=lambda row:(row["ece"],row["nll"],abs(math.log(row["temperature"])),row["temperature"]))
    nll_best=min(trace,key=lambda row:(row["nll"],row["ece"],abs(math.log(row["temperature"])),row["temperature"]))
    baseline=next(row for row in trace if row["temperature"]==1.0)
    if ece_best["ece"] > baseline["ece"]: raise AssertionError("ECE-optimal temperature is worse than tau=1")
    return ece_best,nll_best,trace


def _key(row: dict[str,Any]) -> tuple[Any,...]: return tuple(row["unique_key"])


def join_phase1(records: Sequence[dict[str,Any]], scores: Sequence[dict[str,Any]], score_fingerprint: str) -> list[dict[str,Any]]:
    text={_key(row):row for row in scores if row["modality"]=="text"}; image={_key(row):row for row in scores if row["modality"]=="image"}; output=[]
    if len(text)+len(image)!=len(scores): raise ValueError("Duplicate score unique key")
    for record in records:
        tk=(str(record["item_id"]),int(record["prior_index"])); ik=(str(record["item_id"]),str(record["condition"]),str(record["image_sha256"]))
        if tk not in text or ik not in image: raise KeyError(f"Missing score join key: {record['case_id']}")
        t,i=text[tk],image[ik]; answer=str(record["phase0_normalized_answer"])
        if answer not in t["answer_classes"] or answer not in i["answer_classes"]: raise ValueError(f"Fixed answer outside canonical candidates: {record['case_id']}")
        tc=float(t["calibrated_probabilities"][answer]); ic=float(i["calibrated_probabilities"][answer])
        tl=math.log((tc+FIXED_EPSILON)/(1-tc+FIXED_EPSILON)); il=math.log((ic+FIXED_EPSILON)/(1-ic+FIXED_EPSILON))
        payload={"case_id":record["case_id"],"item_id":record["item_id"],"family_id":record["family_id"],"prior_index":record["prior_index"],"condition":record["condition"],"image_hash":record["image_sha256"],"split":record["split"],"fixed_answer":answer,"text_score_unique_key":list(tk),"image_score_unique_key":list(ik),"text_chosen_answer":t["chosen_answer"],"image_chosen_answer":i["chosen_answer"],"text_chosen_confidence":float(t["chosen_confidence"]),"image_chosen_confidence":float(i["chosen_confidence"]),"text_fixed_answer_confidence":tc,"image_fixed_answer_confidence":ic,"text_fixed_answer_log_odds":tl,"image_fixed_answer_log_odds":il,"G_C":ic-tc,"G_L":il-tl,"text_temperature":t["temperature"],"image_temperature":i["temperature"],"text_temperature_fingerprint":t["temperature_fingerprint"],"image_temperature_fingerprint":i["temperature_fingerprint"],"score_fingerprint":score_fingerprint}
        payload["join_fingerprint"]=canonical_hash(payload); output.append(payload)
    return output


def fit_temperature(root: Path) -> dict[str,Any]:
    raw=load_jsonl(root/"unimodal_confidence/artifacts/raw_scores/unimodal_scores.jsonl")
    if not raw: raise FileNotFoundError("Run score_unimodal first")
    manifests={"text":load_jsonl(root/"shared/manifests/text_calibration_manifest.jsonl"),"image":load_jsonl(root/"shared/manifests/image_calibration_manifest.jsonl")}; by_modality={m:{_key(row):row for row in raw if row["modality"]==m} for m in ("text","image")}
    selected={m:[by_modality[m][_key(row)] for row in manifests[m]] for m in ("text","image")}
    best={}; sensitivity={}; trace=[]
    for modality in ("text","image"):
        best[modality],sensitivity[modality],values=search_temperature(selected[modality])
        trace.extend({"modality":modality,"objective":"grid","grid_index":index,**row} for index,row in enumerate(values))
        fingerprint=canonical_hash({"modality":modality,"objective":"ECE","selection":best[modality],"calibration_keys":[row["unique_key"] for row in manifests[modality]]})
        atomic_json(root/f"unimodal_confidence/artifacts/temperature/{modality}_temperature.json",{"modality":modality,"ece_optimal":best[modality],"nll_optimal":sensitivity[modality],"fingerprint":fingerprint})
    calibrated=[]
    temp_fp={m:json.loads((root/f"unimodal_confidence/artifacts/temperature/{m}_temperature.json").read_text())["fingerprint"] for m in ("text","image")}
    for row in raw:
        temperature=float(best[row["modality"]]["temperature"]); values=[row["raw_candidate_scores"][name] for name in row["answer_classes"]]; metrics=score_metrics(row["answer_classes"],values,row["target_answer"],temperature)
        calibrated.append({**row,"calibrated_probabilities":metrics["probabilities"],"calibrated_probability_sum":metrics["probability_sum"],"chosen_answer":metrics["chosen_answer"],"chosen_confidence":metrics["chosen_confidence"],"correct":metrics["correct"],"nll":metrics["nll"],"brier":metrics["brier"],"temperature":temperature,"temperature_fingerprint":temp_fp[row["modality"]]})
    forbidden={"fixed_answer","fixed_answer_confidence","fixed_answer_log_odds","G_C","G_L"}
    if any(forbidden & set(row) for row in calibrated): raise AssertionError("Unique score contains record-level fixed-answer fields")
    calibrated.sort(key=lambda row:tuple(row["stable_key"])); score_fp=canonical_hash([{k:row[k] for k in ("stable_key","raw_candidate_scores","calibrated_probabilities","temperature_fingerprint")} for row in calibrated])
    atomic_jsonl(root/"unimodal_confidence/artifacts/calibrated_scores/unimodal_scores.jsonl",calibrated)
    records=load_jsonl(root/"shared/manifests/probe_manifest.jsonl"); joined=join_phase1(records,calibrated,score_fp); atomic_jsonl(root/"unimodal_confidence/artifacts/predictions/phase1_confidence_joined.jsonl",joined)
    atomic_csv(root/"unimodal_confidence/artifacts/temperature/search_trace.csv",trace)
    table=[]
    test_records=[r for r in records if r["split"]=="test"]
    for modality in ("text","image"):
        baseline=evaluate_temperature(selected[modality],1.0); calibrated_cal=evaluate_temperature(selected[modality],best[modality]["temperature"])
        test_score_map=by_modality[modality]; test_score_rows=[]
        for record in test_records:
            key=(str(record["item_id"]),int(record["prior_index"])) if modality=="text" else (str(record["item_id"]),str(record["condition"]),str(record["image_sha256"]))
            test_score_rows.append(test_score_map[key])
        test_uncal=evaluate_temperature(test_score_rows,1.0); test_cal=evaluate_temperature(test_score_rows,best[modality]["temperature"])
        table.append({"modality":modality,"temperature_objective":"ECE","temperature":best[modality]["temperature"],"calibration_count":len(selected[modality]),"test_count":len(test_records),"test_unique_key_count":len({_key(r) for r in test_score_rows}),"uncalibrated_ece":baseline["ece"],"calibrated_ece":calibrated_cal["ece"],"uncalibrated_nll":baseline["nll"],"calibrated_nll":calibrated_cal["nll"],"uncalibrated_brier":baseline["brier"],"calibrated_brier":calibrated_cal["brier"],"accuracy":calibrated_cal["accuracy"],"auroc":calibrated_cal["auroc"],"test_uncalibrated_ece":test_uncal["ece"],"test_calibrated_ece":test_cal["ece"]})
    atomic_csv(root/"unimodal_confidence/tables/temperature_calibration.csv",table)
    summary={"status":"complete","score_count":len(calibrated),"joined_count":len(joined),"score_fingerprint":score_fp,"temperatures":{m:best[m]["temperature"] for m in best}}
    atomic_json(root/"unimodal_confidence/progress/fit_temperature.json",summary); return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-root",default=str(RESULTS_ROOT)); parser.add_argument("--resume",action="store_true")
    args=parser.parse_args(argv); root=ensure_layout(args.output_root,resume=True); print(json.dumps(fit_temperature(root),ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
