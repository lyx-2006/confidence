from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

from .config import BOOTSTRAPS, SMOKE_BOOTSTRAPS, VALIDATION_BOOTSTRAPS
from .io_utils import atomic_json, load_jsonl


SCHEMAS = {
    "tables/behavior_pairwise_correlations.csv": {"left_template", "right_template", "comparison_scope", "pearson", "spearman", "mae", "mean_signed_difference", "slope", "intercept", "direction_agreement"},
    "tables/t0_probe_transfer_metrics.csv": {"template", "position", "layer", "target_reference", "r2", "pearson", "spearman", "mae"},
    "tables/vector_geometry.csv": {"row_type", "template", "fold", "layer", "raw_cosine", "absolute_cosine", "cosine_ci_low", "cosine_ci_high"},
    "tables/steering_transfer_effects.csv": {"template", "layer", "alpha", "aggregation", "mean_delta_sa", "hard_label_change_rate"},
    "tables/steering_transfer_vs_t0.csv": {"template", "layer", "aggregation", "s2_tx", "s2_t0", "absolute_contrast", "retention_ratio", "retention_eligible", "t0_denominator_same_sign_fraction"},
}


def _csv_rows(path:Path, required:set[str])->list[dict[str,str]]:
    if not path.is_file() or path.stat().st_size==0:raise ValueError(f"Required output missing/empty: {path}")
    with path.open(newline="",encoding="utf-8") as handle:
        reader=csv.DictReader(handle);fields=set(reader.fieldnames or []);rows=list(reader)
    if not required.issubset(fields):raise ValueError(f"Schema mismatch {path}: missing {sorted(required-fields)}")
    if not rows:raise ValueError(f"Required table has no rows: {path}")
    return rows


def validate_outputs(root:Path,*,templates:Sequence[str],layers:Sequence[int],smoke:bool,stages:set[str],validation_cases:int|None=None)->dict[str,Any]:
    checked=[]
    relevant={path:schema for path,schema in SCHEMAS.items() if ("behavior" in path and "behavior" in stages) or ("probe" in path and "probe" in stages) or ("geometry" in path and "geometry" in stages) or ("steering" in path and "steering" in stages)}
    for relative,schema in relevant.items():_csv_rows(root/relative,schema);checked.append(relative)
    capture=load_jsonl(root/"artifacts/clean/capture.jsonl")
    if "behavior" in stages or "probe" in stages:
        expected=validation_cases or (4 if smoke else 230)
        for template in templates:
            count=sum(r.get("template")==template and r.get("is_audit") for r in capture)
            if count<expected:raise ValueError(f"Audit capture incomplete: {template} {count}/{expected}")
    if "geometry" in stages:
        expected=None if (smoke or validation_cases) else 1625
        for template in templates:
            count=len({str(r["case_id"]) for r in capture if r.get("template")==template and r.get("is_candidate")})
            if expected is not None and count!=expected:raise ValueError(f"Geometry capture incomplete: {template} {count}/{expected}")
        design=__import__("json").loads((root/"artifacts/diagnostics/geometry_bootstrap_draws.json").read_text());wanted=SMOKE_BOOTSTRAPS if smoke else (VALIDATION_BOOTSTRAPS if validation_cases else BOOTSTRAPS)
        if len(design["multiplicity_draws"])!=wanted:raise ValueError("Geometry bootstrap repeat count changed")
    if "steering" in stages:
        trials=load_jsonl(root/"artifacts/steering_trials.jsonl");expected=(validation_cases or (4 if smoke else 174))*len(templates)*len(layers)*3
        keys={(r["template"],str(r["case_id"]),int(r["layer"]),float(r["alpha"])) for r in trials}
        if len(trials)!=expected or len(keys)!=expected:raise ValueError(f"Steering trial completeness failed: {len(trials)}/{expected}")
    figures=list((root/"figures").glob("*.png"))
    if stages.intersection({"behavior","probe","geometry","steering"}) and (not figures or any(p.stat().st_size==0 for p in figures)):raise ValueError("Figures missing/empty")
    result={"status":"passed","checked_tables":checked,"figure_count":len(figures),"smoke_only":smoke,"validation_cases":validation_cases}
    atomic_json(root/"artifacts/diagnostics/output_validation.json",result);return result
