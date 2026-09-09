from __future__ import annotations

import itertools
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import AUDIT_MANIFEST, BOOTSTRAPS, MIDPOINTS_9, SEED, SMOKE_BOOTSTRAPS, VALIDATION_BOOTSTRAPS
from .io_utils import atomic_csv, atomic_json, load_jsonl
from .stats import add_ci, behavior_metrics, bootstrap_rows, hard_agreement, shared_family_draws
from .capture import select_smoke_audit
from .sampling import validation_audit


def _capture_rows(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["template"]), str(row["case_id"])): row for row in load_jsonl(root / "artifacts/clean/capture.jsonl") if row.get("status") == "completed" and row.get("is_audit")}


def _t0_row(row: dict[str, Any]) -> dict[str, Any]:
    hard = int(row["argmax_hard_class"]); soft = float(row["soft_sa_image_score"])
    return {"template": "T0", "case_id": row["case_id"], "family_id": row["family_id"], "item_id": str(row["item_id"]), "condition": row["condition"], "answer": row["phase0_raw_answer"], "answer_matches_text": row.get("answer_matches_text"), "answer_matches_image": row.get("answer_matches_image"), "canonical_soft_sa": soft, "signed_sa": 2*soft-1, "canonical_hard_class": hard, "canonical_hard_label": None, "canonical_hard_group": (0 if hard <= 1 else 1 if hard <= 3 else 2 if hard == 4 else 3 if hard <= 6 else 4), "canonical_hard_score": MIDPOINTS_9[hard], "raw_hard_class": hard, "scoring_status": "historical_valid" if row.get("valid_class") else "historical_invalid", "greedy_parse_status": "historical_constrained", "length_normalized_soft_sa": None, "length_normalized_hard_label": None, "length_definition_agrees": None}


def _tx_row(source: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    hard = source.get("canonical_hard_class")
    return {"template": source["template"], "case_id": source["case_id"], "family_id": source["family_id"], "item_id": source["item_id"], "condition": source["condition"], "answer": source["answer"], "answer_matches_text": source.get("answer_matches_text"), "answer_matches_image": source.get("answer_matches_image"), "canonical_soft_sa": source["canonical_soft_sa"], "signed_sa": source["signed_sa"], "canonical_hard_class": hard, "canonical_hard_label": source.get("canonical_hard_label"), "canonical_hard_group": source.get("canonical_hard_group"), "canonical_hard_score": source.get("canonical_hard_score", MIDPOINTS_9[int(hard)] if hard is not None else None), "raw_hard_class": source.get("raw_hard_class"), "scoring_status": source["scoring_status"], "greedy_parse_status": source.get("greedy_parse_status", "not_audited"), "length_normalized_soft_sa": source.get("length_normalized_soft_sa"), "length_normalized_hard_label": source.get("length_normalized_hard_label"), "length_definition_agrees": source.get("length_definition_agrees"), "t3_candidates": json.dumps(source.get("t3_candidates"), separators=(",", ":")) if source.get("t3_candidates") else None}


def analyze_behavior(root:Path,*,templates:Sequence[str],smoke:bool,validation_cases:int|None=None)->dict[str,Any]:
    audit = validation_audit(validation_cases) if validation_cases else load_jsonl(AUDIT_MANIFEST)
    if smoke: audit = select_smoke_audit(audit,4)
    captured = _capture_rows(root); long_rows = []
    for meta in audit:
        long_rows.append(_t0_row(meta))
        for template in templates:
            key=(template,str(meta["case_id"]))
            if key not in captured: raise ValueError(f"Missing audit capture: {key}")
            long_rows.append(_tx_row(captured[key], meta))
    atomic_csv(root / "artifacts/behavior_case_level.csv", long_rows)
    by_template = {name: {str(r["case_id"]): r for r in long_rows if r["template"] == name} for name in ["T0", *templates]}
    families = [str(row["family_id"]) for row in audit]; repeats = SMOKE_BOOTSTRAPS if smoke else (VALIDATION_BOOTSTRAPS if validation_cases else BOOTSTRAPS)
    ordered, draws = shared_family_draws(families, repeats, SEED + 100)
    pairs = list(itertools.combinations(["T0", *templates], 2)); metrics_rows = []
    for left, right in pairs:
        case_ids=[str(row["case_id"]) for row in audit]; x=np.asarray([by_template[left][case]["canonical_soft_sa"] for case in case_ids]); y=np.asarray([by_template[right][case]["canonical_soft_sa"] for case in case_ids])
        observed=behavior_metrics(x,y); boots,valid=bootstrap_rows(lambda idx: behavior_metrics(x[idx],y[idx]),families,ordered,draws)
        row={"left_template":left,"right_template":right,"comparison_scope":"confirmatory" if left=="T0" else "supplementary",**add_ci(observed,boots,list(observed)),"case_count":len(case_ids),"family_count":len(ordered),"valid_bootstrap_repeats":valid}
        if left=="T0" and right in {"T1","T2"}:
            hx=[int(by_template[left][c]["canonical_hard_class"]) for c in case_ids];hy=[int(by_template[right][c]["canonical_hard_class"]) for c in case_ids];hard=hard_agreement(hx,hy,within_one=True)
            hard_boots,n=bootstrap_rows(lambda idx: hard_agreement(np.asarray(hx)[idx],np.asarray(hy)[idx],within_one=True),families,ordered,draws);row.update(add_ci(hard,hard_boots,list(hard)));row["hard_valid_bootstrap_repeats"]=n
        elif left=="T0" and right=="T3":
            hx=[int(by_template[left][c]["canonical_hard_group"]) for c in case_ids];hy=[int(by_template[right][c]["canonical_hard_group"]) for c in case_ids];hard=hard_agreement(hx,hy,within_one=False)
            hard_boots,n=bootstrap_rows(lambda idx: hard_agreement(np.asarray(hx)[idx],np.asarray(hy)[idx],within_one=False),families,ordered,draws);row.update(add_ci(hard,hard_boots,list(hard)));row["hard_valid_bootstrap_repeats"]=n
        metrics_rows.append(row)
    atomic_csv(root / "tables/behavior_pairwise_correlations.csv", metrics_rows)
    distributions=[]
    for name, rows in by_template.items():
        values=np.asarray([r["canonical_soft_sa"] for r in rows.values()],float); labels=Counter(str(r.get("canonical_hard_label") if name=="T3" else r.get("canonical_hard_class")) for r in rows.values())
        length_agreement=(float(np.mean([bool(r["length_definition_agrees"]) for r in rows.values()])) if name=="T3" else math.nan)
        distributions.append({"template":name,"case_count":len(values),"mean":float(values.mean()),"std":float(values.std(ddof=1)),"q05":float(np.quantile(values,.05)),"median":float(np.median(values)),"q95":float(np.quantile(values,.95)),"image_direction_fraction":float(np.mean(values>.5)),"text_direction_fraction":float(np.mean(values<.5)),"hard_distribution":json.dumps(labels,sort_keys=True),"length_definition_agreement_rate":length_agreement})
    atomic_csv(root / "tables/behavior_template_distributions.csv", distributions)
    atomic_json(root / "artifacts/diagnostics/behavior_bootstrap_draws.json", {"seed":SEED+100,"repeats":repeats,"families":ordered,"draws":draws.tolist()})
    return {"status":"complete","case_rows":len(long_rows),"pair_count":len(metrics_rows),"bootstrap_repeats":repeats}
