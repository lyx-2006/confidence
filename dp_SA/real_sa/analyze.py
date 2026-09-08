from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .config import BOOTSTRAP_REPEATS, CONDITIONS, SEED, SENSITIVITY_THRESHOLDS
from .data import FrozenCohort
from .io_utils import atomic_csv, atomic_json, load_jsonl
from .metrics import case_metrics, family_cluster_bootstrap


PER_CASE_FIELDS = (
    "case_id", "family_id", "item_id", "condition", "fixed_answer", "verbal_sa",
    "v11", "v10", "v01", "v00", "D_I", "D_T", "phi_I", "phi_T", "J",
    "phi_sum", "efficiency_error", "sign_type", "total_effect", "R_I",
    "R_I_eligible", "G_R", "answer_side",
)
SUMMARY_FIELDS = (
    "group_type", "group_value", "metric", "threshold", "case_count", "family_count",
    "valid_case_count", "estimate", "ci_low", "ci_high", "bootstrap_valid_repeats",
)


def combine_cases(score_rows: Sequence[dict[str, Any]], cohort: FrozenCohort) -> list[dict[str, Any]]:
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for row in score_rows:
        conditions = by_case.setdefault(str(row["case_id"]), {})
        condition = str(row["corruption_condition"])
        if condition in conditions:
            raise ValueError(f"Duplicate condition score: {row['case_id']} {condition}")
        conditions[condition] = row
    source = {str(row["case_id"]): row for row in cohort.tests}
    if set(by_case) != set(source):
        raise ValueError("Formal scores do not cover the frozen test cohort")
    output: list[dict[str, Any]] = []
    for case_id in sorted(source):
        conditions = by_case[case_id]
        if set(conditions) != set(CONDITIONS):
            raise ValueError(f"Case does not have all four conditions: {case_id}")
        v11 = float(conditions["clean"]["fixed_answer_probability"])
        v10 = float(conditions["10_text_corrupt"]["fixed_answer_probability"])
        v01 = float(conditions["01_image_corrupt"]["fixed_answer_probability"])
        v00 = float(conditions["00_both_corrupt"]["fixed_answer_probability"])
        row = source[case_id]
        output.append({
            "case_id": case_id, "family_id": row["family_id"], "item_id": str(row["item_id"]),
            "condition": row["condition"], "fixed_answer": row["phase0_normalized_answer"],
            "verbal_sa": float(row["soft_sa_image_score"]), "v11": v11, "v10": v10,
            "v01": v01, "v00": v00, **case_metrics(v11, v10, v01, v00),
            "answer_side": row["answer_side"],
        })
    return output


def _mean(field: str, *, eligible_only: bool = False) -> Callable[[Sequence[dict[str, Any]]], float | None]:
    def calculate(rows: Sequence[dict[str, Any]]) -> float | None:
        values = [float(row[field]) for row in rows
                  if row.get(field) is not None and (not eligible_only or bool(row["R_I_eligible"]))]
        return None if not values else float(np.mean(values))
    return calculate


def _summary_cell(rows: Sequence[dict[str, Any]], group_type: str, group_value: str,
                  metric: str, value: Callable[[Sequence[dict[str, Any]]], float | None],
                  *, threshold: float | None = None) -> dict[str, Any]:
    estimate = value(rows)
    interval = family_cluster_bootstrap(rows, value, repeats=BOOTSTRAP_REPEATS, seed=SEED)
    if metric == "R_I":
        valid = sum(row.get("R_I") is not None for row in rows)
    elif metric == "R_I_sensitivity" and threshold is not None:
        valid = sum(row["phi_I"] >= 0 and row["phi_T"] >= 0 and row["total_effect"] >= threshold for row in rows)
    else:
        valid = len(rows)
    return {
        "group_type": group_type, "group_value": group_value, "metric": metric,
        "threshold": "" if threshold is None else threshold, "case_count": len(rows),
        "family_count": len({row["family_id"] for row in rows}), "valid_case_count": valid,
        "estimate": estimate, "ci_low": interval["low"], "ci_high": interval["high"],
        "bootstrap_valid_repeats": interval["valid"],
    }


def build_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[tuple[str, str, list[dict[str, Any]]]] = [("all", "all", list(rows))]
    for kind, field, values in (
        ("difficulty", "condition", ("conflict_easy", "conflict_hard")),
        ("answer_side", "answer_side", ("follow_image", "follow_text")),
        ("fixed_answer", "fixed_answer", tuple(sorted({row["fixed_answer"] for row in rows}))),
        ("sign_type", "sign_type", tuple(sorted({row["sign_type"] for row in rows}))),
    ):
        groups.extend((kind, value, [row for row in rows if row[field] == value]) for value in values)
    output: list[dict[str, Any]] = []
    for group_type, group_value, members in groups:
        for field in ("v11", "v10", "v01", "v00", "D_I", "D_T", "phi_I", "phi_T", "J", "total_effect", "G_R"):
            output.append(_summary_cell(members, group_type, group_value, field, _mean(field)))
        output.append(_summary_cell(members, group_type, group_value, "R_I", _mean("R_I", eligible_only=True)))
        for threshold in SENSITIVITY_THRESHOLDS:
            def eligible(sample: Sequence[dict[str, Any]], t: float = threshold) -> float | None:
                return None if not sample else float(np.mean([
                    row["phi_I"] >= 0 and row["phi_T"] >= 0 and row["total_effect"] >= t for row in sample
                ]))
            def ratio(sample: Sequence[dict[str, Any]], t: float = threshold) -> float | None:
                values = [row["phi_I"] / row["total_effect"] for row in sample
                          if row["phi_I"] >= 0 and row["phi_T"] >= 0 and row["total_effect"] >= t]
                return None if not values else float(np.mean(values))
            output.append(_summary_cell(members, group_type, group_value, "R_I_eligible_rate", eligible, threshold=threshold))
            output.append(_summary_cell(members, group_type, group_value, "R_I_sensitivity", ratio, threshold=threshold))
    return output


def sign_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    total = len(rows)
    for name in ("both_support", "image_support_text_suppress", "text_support_image_suppress",
                 "both_suppress", "weak_total_effect"):
        members = [row for row in rows if row["sign_type"] == name]
        def share(sample: Sequence[dict[str, Any]], target: str = name) -> float | None:
            return None if not sample else float(np.mean([row["sign_type"] == target for row in sample]))
        interval = family_cluster_bootstrap(rows, share, repeats=BOOTSTRAP_REPEATS, seed=SEED)
        output.append({
            "sign_type": name, "count": len(members), "proportion": len(members) / total,
            "proportion_ci_low": interval["low"], "proportion_ci_high": interval["high"],
            "family_count": len({row["family_id"] for row in members}),
            "mean_phi_I": None if not members else float(np.mean([row["phi_I"] for row in members])),
            "mean_phi_T": None if not members else float(np.mean([row["phi_T"] for row in members])),
            "mean_total_effect": None if not members else float(np.mean([row["total_effect"] for row in members])),
            "mean_G_R": None if not members else float(np.mean([row["G_R"] for row in members])),
            "R_I_eligible_count": sum(bool(row["R_I_eligible"]) for row in members),
            "mean_R_I_eligible": None if not any(row["R_I_eligible"] for row in members) else float(np.mean([
                row["R_I"] for row in members if row["R_I_eligible"]
            ])),
        })
    return output


def corruption_diagnostics(scores: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    clean = {str(row["case_id"]): row for row in scores if row["corruption_condition"] == "clean"}
    output: list[dict[str, Any]] = []
    for dataset_group in ("all", "conflict_easy", "conflict_hard"):
        for condition in CONDITIONS:
            members = [row for row in scores if row["corruption_condition"] == condition and
                       (dataset_group == "all" or row["dataset_condition"] == dataset_group)]
            image_norms = [row["replacement_diagnostics"].get("replacement_l2", {}).get("image") for row in members]
            text_norms = [row["replacement_diagnostics"].get("replacement_l2", {}).get("text") for row in members]
            image_norms = [float(value) for value in image_norms if value is not None]
            text_norms = [float(value) for value in text_norms if value is not None]
            supports = [float(row["fixed_answer_probability"]) for row in members]
            deltas = [float(row["fixed_answer_probability"]) - float(clean[str(row["case_id"])]["fixed_answer_probability"])
                      for row in members]
            changes = [row["condition_argmax_answer"] != clean[str(row["case_id"])]["condition_argmax_answer"] for row in members]
            output.append({
                "dataset_group": dataset_group, "corruption_condition": condition, "case_count": len(members),
                "mean_fixed_answer_probability": None if not supports else float(np.mean(supports)),
                "mean_delta_from_clean": None if not deltas else float(np.mean(deltas)),
                "argmax_change_rate": None if not changes else float(np.mean(changes)),
                "max_probability_sum_error": None if not members else max(
                    abs(float(row["probability_sum"]) - 1.0) for row in members
                ),
                "image_replacement_norm_mean": None if not image_norms else float(np.mean(image_norms)),
                "image_replacement_norm_min": None if not image_norms else min(image_norms),
                "image_replacement_norm_max": None if not image_norms else max(image_norms),
                "text_replacement_norm_mean": None if not text_norms else float(np.mean(text_norms)),
                "text_replacement_norm_min": None if not text_norms else min(text_norms),
                "text_replacement_norm_max": None if not text_norms else max(text_norms),
            })
    return output


def analyze(root: Path, cohort: FrozenCohort) -> dict[str, Any]:
    scores = load_jsonl(root / "artifacts" / "condition_scores.jsonl")
    rows = combine_cases(scores, cohort)
    tables = root / "tables"
    atomic_csv(tables / "real_sa_per_case.csv", rows, PER_CASE_FIELDS)
    summary = build_summary(rows)
    atomic_csv(tables / "real_sa_summary.csv", summary, SUMMARY_FIELDS)
    signs = sign_summary(rows)
    atomic_csv(tables / "sign_type_summary.csv", signs, tuple(signs[0]))
    diagnostics = corruption_diagnostics(scores)
    atomic_csv(tables / "corruption_diagnostics.csv", diagnostics, tuple(diagnostics[0]))
    result = {
        "status": "complete", "case_count": len(rows), "condition_score_count": len(scores),
        "sign_counts": dict(Counter(row["sign_type"] for row in rows)),
        "bootstrap_repeats": BOOTSTRAP_REPEATS, "seed": SEED,
    }
    atomic_json(root / "progress" / "analysis.json", result)
    return result


__all__ = ["analyze", "build_summary", "combine_cases", "corruption_diagnostics", "sign_summary"]
