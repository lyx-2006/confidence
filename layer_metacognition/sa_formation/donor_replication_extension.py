"""Prospective donor-replication extension for Actual Source Reliance.

This module does not reopen the failed method-v2 confirmatory gate.  It adds
two previously unmeasured, deterministic replacement donors to the same fixed
answer endpoint and asks two narrower questions:

1. does the original two-donor mean replicate in a fresh two-donor mean; and
2. does deletion agree with that fresh replacement mean on the raw scale?

The extension is intentionally answer-only, performs four logits-only forwards
per item, and never captures hidden states.  A passing result is evidence for
intervention replication conditional on the existing items, not a new-item
confirmation of a choice-independent latent reliance construct.
"""

from __future__ import annotations

import hashlib
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from confidence_test.dataset_utils import EvaluationCase
from layer_metacognition.hidden_state_store import (
    append_jsonl,
    atomic_write_json,
    atomic_write_text,
    load_jsonl,
)

from .core import (
    SAFormationArtifacts,
    canonical_message_hash,
    item_cluster_bootstrap,
    stable_hash,
    write_jsonl_atomic,
)
from .reliance_measurement import (
    BRIDGE_DIR,
    MEASUREMENT_METHOD_VERSION,
    RELIANCE_DIR,
    _numeric_item_key,
    _pre_answer_condition,
    _prior_bin,
    build_answer_only_messages,
    canonical_answer_token_ids,
    contains_verbal_sa_request,
    load_source_rows,
    nuisance_vector,
    plan_split,
    select_two_donors,
)
from .runtime import Stage3Runtime


EXTENSION_DIR = "02_donor_replication_extension"
EXTENSION_METHOD_VERSION = 3
SEED = 42
NEW_DONOR_INDICES = (3, 4)
NEW_CONDITIONS = (
    "replace_text_d3",
    "replace_image_d3",
    "replace_text_d4",
    "replace_image_d4",
)


def method_v2_root(experiment_dir: str | Path) -> Path:
    return Path(experiment_dir) / BRIDGE_DIR / RELIANCE_DIR


def extension_root(experiment_dir: str | Path) -> Path:
    return Path(experiment_dir) / BRIDGE_DIR / EXTENSION_DIR


def _latest_rows(path: str | Path) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path, repair_trailing=True):
        key = str(row.get("intervention_key", row.get("case_id", "")))
        if key:
            latest[key] = row
    return list(latest.values())


def load_method_v2_rows(
    experiment_dir: str | Path,
    split: str,
) -> list[dict[str, Any]]:
    if split not in {"development", "confirmatory"}:
        raise ValueError(f"Unknown split: {split}")
    root = method_v2_root(experiment_dir)
    path = root / f"{split}_analysis.jsonl"
    rows = [
        row
        for row in load_jsonl(path)
        if row.get("status") == "completed"
        and int(row.get("measurement_method_version", -1)) == MEASUREMENT_METHOD_VERSION
    ]
    expected_minimum = 90 if split == "development" else 70
    if len(rows) < expected_minimum:
        raise ValueError(
            f"Method-v2 {split} has only {len(rows)} completed rows; expected at least "
            f"{expected_minimum}"
        )
    case_ids = [str(row["case_id"]) for row in rows]
    item_ids = [str(row["item_id"]) for row in rows]
    if len(set(case_ids)) != len(rows) or len(set(item_ids)) != len(rows):
        raise ValueError("Method-v2 extension input must contain unique cases and items")
    required = {
        "answer_star",
        "selection_rendered_hash",
        "behavior_delete_imageward",
        "behavior_replace_imageward_d1",
        "behavior_replace_imageward_d2",
        "donor1_case_id",
        "donor1_item_id",
        "donor2_case_id",
        "donor2_item_id",
    }
    for row in rows:
        missing = required.difference(row)
        if missing:
            raise ValueError(f"Method-v2 row {row['case_id']} omits {sorted(missing)}")
    return rows


def _donor_rank(
    target: dict[str, Any],
    row: dict[str, Any],
    case_by_key: dict[tuple[str, int], EvaluationCase],
) -> tuple[Any, ...]:
    target_case = case_by_key[(str(target["item_id"]), int(target["prior_index"]))]
    donor_case = case_by_key[(str(row["item_id"]), int(row["prior_index"]))]
    return (
        int(_prior_bin(donor_case) != _prior_bin(target_case)),
        int(str(row["condition"]) != str(target["condition"])),
        abs(len(donor_case.text_clue) - len(target_case.text_clue)),
        _numeric_item_key(row["item_id"]),
        int(row["prior_index"]),
        str(row["case_id"]),
    )


def select_extension_donors(
    target: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    case_by_key: dict[tuple[str, int], EvaluationCase],
    method_v2_row: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select the third/fourth ranked distinct donor items under the v2 rule."""

    original = select_two_donors(target, candidates, case_by_key)
    expected = [
        (str(method_v2_row["donor1_case_id"]), str(method_v2_row["donor1_item_id"])),
        (str(method_v2_row["donor2_case_id"]), str(method_v2_row["donor2_item_id"])),
    ]
    observed = [(str(row["case_id"]), str(row["item_id"])) for row in original]
    if observed != expected:
        raise ValueError(
            f"Method-v2 donor selection drift for {target['case_id']}: "
            f"expected={expected}, observed={observed}"
        )

    used_items = {
        str(target["item_id"]),
        str(method_v2_row["donor1_item_id"]),
        str(method_v2_row["donor2_item_id"]),
    }
    eligible = [
        row
        for row in candidates
        if str(row["item_id"]) not in used_items
        and int(row["fold"]) == int(target["fold"])
        and str(row["difficulty"]) == str(target["difficulty"])
        and int(row["final_image"]) == int(target["final_image"])
    ]
    chosen: list[dict[str, Any]] = []
    for row in sorted(eligible, key=lambda value: _donor_rank(target, value, case_by_key)):
        item = str(row["item_id"])
        if item in used_items:
            continue
        chosen.append(row)
        used_items.add(item)
        if len(chosen) == 2:
            break
    if len(chosen) != 2:
        raise ValueError(f"Fewer than four total distinct donors for {target['case_id']}")
    return chosen[0], chosen[1]


def extension_condition_sources(
    target_case: EvaluationCase,
    target_row: dict[str, Any],
    donor_rows: tuple[dict[str, Any], dict[str, Any]],
    case_by_key: dict[tuple[str, int], EvaluationCase],
) -> dict[str, dict[str, Any]]:
    target_image = str(
        target_case.conditions[str(target_row["condition"])].resolved_image_path
    )
    result: dict[str, dict[str, Any]] = {}
    for donor_index, donor_row in zip(NEW_DONOR_INDICES, donor_rows, strict=True):
        donor_case = case_by_key[
            (str(donor_row["item_id"]), int(donor_row["prior_index"]))
        ]
        donor_image = str(
            donor_case.conditions[str(donor_row["condition"])].resolved_image_path
        )
        result[f"replace_text_d{donor_index}"] = {
            "text_clue": donor_case.text_clue,
            "image_path": target_image,
            "text_source_item": str(donor_row["item_id"]),
            "image_source_item": str(target_row["item_id"]),
        }
        result[f"replace_image_d{donor_index}"] = {
            "text_clue": target_case.text_clue,
            "image_path": donor_image,
            "text_source_item": str(target_row["item_id"]),
            "image_source_item": str(donor_row["item_id"]),
        }
    if tuple(result) != NEW_CONDITIONS:
        raise RuntimeError(f"Extension condition order drifted: {tuple(result)}")
    for donor_index in NEW_DONOR_INDICES:
        text = result[f"replace_text_d{donor_index}"]
        image = result[f"replace_image_d{donor_index}"]
        if text["text_source_item"] != image["image_source_item"]:
            raise RuntimeError(f"Donor {donor_index} is asymmetric")
        if text["image_source_item"] != image["text_source_item"]:
            raise RuntimeError(f"Target context for donor {donor_index} is asymmetric")
    return result


def _method_v2_full_messages_hash(
    case: EvaluationCase,
    target_row: dict[str, Any],
) -> str:
    image_path = str(
        case.conditions[str(target_row["condition"])].resolved_image_path
    )
    messages = build_answer_only_messages(
        case,
        text_clue=case.text_clue,
        image_path=image_path,
    )
    return canonical_message_hash(messages)


def measure_extension_case(
    runtime: Stage3Runtime,
    target_row: dict[str, Any],
    method_v2_row: dict[str, Any],
    donor_rows: tuple[dict[str, Any], dict[str, Any]],
    case_by_key: dict[tuple[str, int], EvaluationCase],
) -> dict[str, Any]:
    case = case_by_key[
        (str(target_row["item_id"]), int(target_row["prior_index"]))
    ]
    if str(target_row["case_id"]) != str(method_v2_row["case_id"]):
        raise ValueError("Method-v2 row does not match extension target")
    full_messages_hash = _method_v2_full_messages_hash(case, target_row)
    stored_messages_hash = str(method_v2_row["selection"]["messages_hash"])
    if full_messages_hash != stored_messages_hash:
        raise ValueError(
            f"Answer-only Full message hash drift for {target_row['case_id']}"
        )
    answer = str(method_v2_row["answer_star"])
    token_ids = canonical_answer_token_ids(runtime.generator.tokenizer, case.answer_classes)
    if answer not in token_ids:
        raise ValueError(f"Fixed answer {answer!r} is not in the canonical answer vocabulary")
    sources = extension_condition_sources(case, target_row, donor_rows, case_by_key)
    measurements: dict[str, dict[str, Any]] = {}
    for condition in NEW_CONDITIONS:
        measured, hidden = _pre_answer_condition(
            runtime,
            case,
            sources[condition],
            answer,
            token_ids,
            capture_hidden=False,
        )
        if hidden is not None:
            raise RuntimeError("Donor extension unexpectedly captured hidden state")
        measurements[condition] = measured
    effects: dict[str, float] = {}
    for donor_index in NEW_DONOR_INDICES:
        text_probability = float(
            measurements[f"replace_text_d{donor_index}"][
                "answer_class_probabilities"
            ][answer]
        )
        image_probability = float(
            measurements[f"replace_image_d{donor_index}"][
                "answer_class_probabilities"
            ][answer]
        )
        if (
            not math.isfinite(text_probability)
            or not math.isfinite(image_probability)
            or text_probability <= 0.0
            or image_probability <= 0.0
        ):
            raise ValueError(
                f"Non-positive fixed-answer probability for donor {donor_index}"
            )
        effects[f"behavior_replace_imageward_d{donor_index}"] = (
            math.log(text_probability) - math.log(image_probability)
        )
    mean = statistics.fmean(effects.values())
    donor_metadata: dict[str, Any] = {}
    support_biases: dict[int, int] = {}
    for donor_index, donor_row in zip(NEW_DONOR_INDICES, donor_rows, strict=True):
        text_answer = str(donor_row["text_answer"])
        image_answer = str(donor_row["image_answer"])
        text_match = text_answer == answer
        image_match = image_answer == answer
        support_bias = int(text_match) - int(image_match)
        support_biases[donor_index] = support_bias
        donor_metadata.update(
            {
                f"donor{donor_index}_text_answer": text_answer,
                f"donor{donor_index}_image_answer": image_answer,
                f"donor{donor_index}_text_matches_answer_star": text_match,
                f"donor{donor_index}_image_matches_answer_star": image_match,
                f"donor{donor_index}_answer_support_bias": support_bias,
                f"donor{donor_index}_compatibility_class": (
                    "text"
                    if text_match and not image_match
                    else "image"
                    if image_match and not text_match
                    else "both"
                    if text_match and image_match
                    else "neither"
                ),
            }
        )
    return {
        "status": "completed",
        "answer_star": answer,
        "answer_star_reused": True,
        "method_v2_selection_rendered_hash": method_v2_row[
            "selection_rendered_hash"
        ],
        "method_v2_full_messages_hash": stored_messages_hash,
        "reconstructed_full_messages_hash": full_messages_hash,
        "full_messages_hash_equal": full_messages_hash == stored_messages_hash,
        "selection_reused_without_forward": True,
        "measurements": measurements,
        "condition_sources": {
            condition: {
                key: value
                for key, value in source.items()
                if key != "text_clue"
            }
            for condition, source in sources.items()
        },
        "verbal_sa_leakage": any(
            bool(measurement["verbal_sa_leakage"])
            for measurement in measurements.values()
        ),
        "hidden_captured": False,
        **donor_metadata,
        **effects,
        "behavior_replace_imageward_d34_mean": mean,
        "behavior_replace_imageward_d3_minus_d4": (
            effects["behavior_replace_imageward_d3"]
            - effects["behavior_replace_imageward_d4"]
        ),
        "donor_match_asymmetry_d3_minus_d4": (
            support_biases[3] - support_biases[4]
        ),
        "replacement_d34_disagreement": 0.5
        * (
            effects["behavior_replace_imageward_d3"]
            - effects["behavior_replace_imageward_d4"]
        ),
    }


def _safe_record(
    base: dict[str, Any], operation: Callable[[], dict[str, Any]]
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        return {**base, **operation(), "elapsed_seconds": time.perf_counter() - started}
    except Exception as exc:
        return {
            **base,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": time.perf_counter() - started,
        }


def build_extension_plan(
    artifacts: SAFormationArtifacts,
    split: str,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, tuple[dict[str, Any], dict[str, Any]]],
    dict[tuple[str, int], EvaluationCase],
]:
    cohort, original_donors, case_by_key = plan_split(artifacts, split)
    method_v2_rows = load_method_v2_rows(artifacts.experiment_dir, split)
    method_v2_by_case = {str(row["case_id"]): row for row in method_v2_rows}
    completed_cohort = [
        row for row in cohort if str(row["case_id"]) in method_v2_by_case
    ]
    if len(completed_cohort) != len(method_v2_rows):
        raise ValueError("Method-v2 completed rows do not match the authoritative cohort")
    source_rows = load_source_rows(artifacts.experiment_dir)
    extension_donors: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for row in completed_cohort:
        case_id = str(row["case_id"])
        method_row = method_v2_by_case[case_id]
        expected = original_donors[case_id]
        if [str(value["case_id"]) for value in expected] != [
            str(method_row["donor1_case_id"]),
            str(method_row["donor2_case_id"]),
        ]:
            raise ValueError(f"Original donor manifest drift for {case_id}")
        extension_donors[case_id] = select_extension_donors(
            row, source_rows, case_by_key, method_row
        )
    return completed_cohort, method_v2_by_case, extension_donors, case_by_key


def build_extension_manifest(
    split: str,
    cohort: Sequence[dict[str, Any]],
    method_v2_by_case: dict[str, dict[str, Any]],
    extension_donors: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    *,
    method_v2_sha256: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for target in cohort:
        case_id = str(target["case_id"])
        old = method_v2_by_case[case_id]
        d3, d4 = extension_donors[case_id]
        donor_items = [
            str(old["donor1_item_id"]),
            str(old["donor2_item_id"]),
            str(d3["item_id"]),
            str(d4["item_id"]),
        ]
        if len(set(donor_items)) != 4 or str(target["item_id"]) in donor_items:
            raise RuntimeError(f"Donor items are not target-distinct for {case_id}")
        rows.append(
            {
                "case_id": case_id,
                "item_id": target["item_id"],
                "prior_index": int(target["prior_index"]),
                "condition": target["condition"],
                "difficulty": target["difficulty"],
                "fold": int(target["fold"]),
                "final_image": int(target["final_image"]),
                "answer_star": old["answer_star"],
                "method_v2_selection_rendered_hash": old[
                    "selection_rendered_hash"
                ],
                "method_v2_full_messages_hash": old["selection"][
                    "messages_hash"
                ],
                "donor1_case_id": old["donor1_case_id"],
                "donor1_item_id": old["donor1_item_id"],
                "donor2_case_id": old["donor2_case_id"],
                "donor2_item_id": old["donor2_item_id"],
                "donor3_case_id": d3["case_id"],
                "donor3_item_id": d3["item_id"],
                "donor3_text_answer": d3["text_answer"],
                "donor3_image_answer": d3["image_answer"],
                "donor3_text_matches_answer_star": str(d3["text_answer"])
                == str(old["answer_star"]),
                "donor3_image_matches_answer_star": str(d3["image_answer"])
                == str(old["answer_star"]),
                "donor4_case_id": d4["case_id"],
                "donor4_item_id": d4["item_id"],
                "donor4_text_answer": d4["text_answer"],
                "donor4_image_answer": d4["image_answer"],
                "donor4_text_matches_answer_star": str(d4["text_answer"])
                == str(old["answer_star"]),
                "donor4_image_matches_answer_star": str(d4["image_answer"])
                == str(old["answer_star"]),
            }
        )
    payload = {
        "format_version": 1,
        "extension_method_version": EXTENSION_METHOD_VERSION,
        "split": split,
        "n": len(rows),
        "conditions": list(NEW_CONDITIONS),
        "forward_count": 4 * len(rows),
        "hidden_capture": False,
        "answer_star": "exact reuse from completed method-v2 Full answer-only run",
        "donor_selection": (
            "third/fourth distinct item under the frozen method-v2 lexicographic "
            "rule; same fold/difficulty/final-side; d1-d4 and target distinct"
        ),
        "claim_scope": (
            "prospective intervention replication on existing items; not a "
            "new-item confirmation and not a reversal of the method-v2 gate"
        ),
        "method_v2_analysis_sha256": method_v2_sha256,
        "rows": rows,
    }
    payload["manifest_fingerprint"] = stable_hash(payload)
    return payload


def _association(
    rows: Sequence[dict[str, Any]], left: str, right: str
) -> dict[str, Any]:
    valid = [
        row
        for row in rows
        if row.get(left) is not None
        and row.get(right) is not None
        and np.isfinite(float(row[left]))
        and np.isfinite(float(row[right]))
    ]
    if len(valid) < 3:
        return {
            "n": len(valid),
            "pearson": None,
            "spearman": None,
            "spearman_item_bootstrap": None,
        }
    x = np.asarray([float(row[left]) for row in valid], dtype=np.float64)
    y = np.asarray([float(row[right]) for row in valid], dtype=np.float64)
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return {
            "n": len(valid),
            "pearson": None,
            "spearman": None,
            "spearman_item_bootstrap": None,
        }
    return {
        "n": len(valid),
        "unique_items": len({str(row["item_id"]) for row in valid}),
        "pearson": float(pearsonr(x, y).statistic),
        "spearman": float(spearmanr(x, y).statistic),
        "spearman_item_bootstrap": item_cluster_bootstrap(
            valid,
            lambda sample: spearmanr(
                [float(row[left]) for row in sample],
                [float(row[right]) for row in sample],
            ).statistic,
        ),
    }


def _donor_reuse_audit(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def audit(indices: Sequence[int]) -> dict[str, Any]:
        counts = Counter(
            str(row[f"donor{index}_item_id"])
            for row in rows
            for index in indices
        )
        pairs = Counter(
            tuple(sorted(str(row[f"donor{index}_item_id"]) for index in indices))
            for row in rows
        )
        values = list(counts.values())
        return {
            "donor_indices": list(indices),
            "assignments": int(sum(values)),
            "unique_donors": len(counts),
            "reused_donors": sum(value > 1 for value in values),
            "maximum_reuse": max(values, default=0),
            "mean_reuse": float(statistics.fmean(values)) if values else None,
            "unique_donor_sets": len(pairs),
            "repeated_donor_sets": sum(value > 1 for value in pairs.values()),
            "counts": dict(sorted(counts.items(), key=lambda value: (-value[1], value[0]))),
        }

    target_items = {str(row["item_id"]) for row in rows}
    all_donors = {
        str(row[f"donor{index}_item_id"])
        for row in rows
        for index in (1, 2, 3, 4)
    }
    return {
        "note": (
            "Donors are reused across targets. Target-item bootstrap is primary; "
            "multi-membership donor clustering and leave-one-donor-out are sensitivity analyses."
        ),
        "all_four": audit((1, 2, 3, 4)),
        "fresh_pair": audit((3, 4)),
        "target_items_also_used_as_donors_n": len(target_items.intersection(all_donors)),
    }


def _donor_cluster_bootstrap(
    rows: Sequence[dict[str, Any]],
    left: str,
    right: str,
    donor_indices: Sequence[int],
    *,
    iterations: int = 1000,
    seed: int = SEED,
) -> dict[str, Any]:
    """Multi-membership cluster bootstrap over reused donor items.

    Each target belongs to every donor cluster that contributes to the relevant
    statistic.  Resampling donor IDs therefore duplicates all target rows using
    each sampled donor.  Because every target has the same number of relevant
    donors, the unresampled expansion has exactly the ordinary target-level
    Spearman estimate.
    """

    donors = sorted(
        {
            str(row[f"donor{index}_item_id"])
            for row in rows
            for index in donor_indices
        }
    )
    by_donor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for donor in {
            str(row[f"donor{index}_item_id"]) for index in donor_indices
        }:
            by_donor[donor].append(row)
    observed = float(
        spearmanr(
            [float(row[left]) for row in rows],
            [float(row[right]) for row in rows],
        ).statistic
    )
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(iterations):
        chosen = rng.choice(donors, size=len(donors), replace=True)
        sample = [row for donor in chosen for row in by_donor[str(donor)]]
        value = float(
            spearmanr(
                [float(row[left]) for row in sample],
                [float(row[right]) for row in sample],
            ).statistic
        )
        if np.isfinite(value):
            samples.append(value)
    return {
        "method": "multi-membership donor-cluster bootstrap",
        "donor_indices": list(donor_indices),
        "unique_donors": len(donors),
        "estimate": observed,
        "ci95": (
            [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]
            if samples
            else [None, None]
        ),
        "iterations": iterations,
        "valid": len(samples),
        "gate_bearing": False,
    }


def _leave_one_donor_out(
    rows: Sequence[dict[str, Any]],
    left: str,
    right: str,
    donor_indices: Sequence[int],
) -> dict[str, Any]:
    donors = sorted(
        {
            str(row[f"donor{index}_item_id"])
            for row in rows
            for index in donor_indices
        }
    )
    estimates: list[dict[str, Any]] = []
    for donor in donors:
        selected = [
            row
            for row in rows
            if donor
            not in {
                str(row[f"donor{index}_item_id"])
                for index in donor_indices
            }
        ]
        value = (
            float(
                spearmanr(
                    [float(row[left]) for row in selected],
                    [float(row[right]) for row in selected],
                ).statistic
            )
            if len(selected) >= 3
            else None
        )
        estimates.append({"donor_item_id": donor, "n": len(selected), "spearman": value})
    valid = [
        float(entry["spearman"])
        for entry in estimates
        if entry["spearman"] is not None and np.isfinite(float(entry["spearman"]))
    ]
    return {
        "method": "leave every target using one donor item out",
        "donor_indices": list(donor_indices),
        "gate_bearing": False,
        "minimum_spearman": min(valid) if valid else None,
        "median_spearman": float(np.median(valid)) if valid else None,
        "maximum_spearman": max(valid) if valid else None,
        "negative_estimate_n": sum(value < 0.0 for value in valid),
        "estimates": estimates,
    }


def _within_group_centered_association(
    rows: Sequence[dict[str, Any]],
    left: str,
    right: str,
    group: str,
) -> dict[str, Any]:
    means: dict[str, dict[str, float]] = {}
    for value in sorted({str(row[group]) for row in rows}):
        selected = [row for row in rows if str(row[group]) == value]
        means[value] = {
            left: float(statistics.fmean(float(row[left]) for row in selected)),
            right: float(statistics.fmean(float(row[right]) for row in selected)),
        }
    centered = [
        {
            **row,
            "_centered_left": float(row[left]) - means[str(row[group])][left],
            "_centered_right": float(row[right]) - means[str(row[group])][right],
        }
        for row in rows
    ]
    return {
        "group": group,
        "group_means": means,
        "association": _association(centered, "_centered_left", "_centered_right"),
        "gate_bearing": False,
    }


def _sign_agreement(rows: Sequence[dict[str, Any]], left: str, right: str) -> float:
    pairs = [
        (float(row[left]), float(row[right]))
        for row in rows
        if float(row[left]) != 0.0 and float(row[right]) != 0.0
    ]
    if not pairs:
        return float("nan")
    return float(statistics.fmean((left > 0.0) == (right > 0.0) for left, right in pairs))


def _cronbach_two_indicator(pearson: float | None) -> float | None:
    if pearson is None or pearson <= -1.0:
        return None
    return float(2.0 * pearson / (1.0 + pearson))


def _icc_consistency(values: np.ndarray) -> float | None:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] < 2:
        return None
    n, k = matrix.shape
    subject_means = matrix.mean(axis=1)
    grand = float(matrix.mean())
    ms_between = float(k * np.sum((subject_means - grand) ** 2) / (n - 1))
    residual = matrix - subject_means[:, None] - matrix.mean(axis=0)[None, :] + grand
    ms_error = float(np.sum(residual**2) / ((n - 1) * (k - 1)))
    denominator = ms_between + (k - 1) * ms_error
    return None if denominator <= 0.0 else float((ms_between - ms_error) / denominator)


def apply_legacy_graded_diagnostic(
    row: dict[str, Any], calibration: dict[str, Any]
) -> dict[str, float]:
    fold = str(int(row["fold"]))
    specification = calibration["nuisance"]
    vector = nuisance_vector(row, specification)
    parameters = calibration["folds"][fold]["methods"]["replacement"]
    beta = np.asarray(parameters["nuisance_beta"], dtype=np.float64)
    residual = float(row["behavior_replace_imageward_d34_mean"] - vector @ beta)
    z = (residual - float(parameters["graded_mean"])) / float(
        parameters["graded_sd"]
    )
    return {
        "legacy_graded_residual_replace_d34": residual,
        "legacy_graded_z_replace_d34": float(z),
    }


def fit_full_margin_protocol_repair(
    development_rows: Sequence[dict[str, Any]],
    legacy_calibration: dict[str, Any],
) -> dict[str, Any]:
    """Fit the omitted linear Full-margin nuisance term on development only.

    This repairs a protocol/implementation discrepancy for sensitivity
    analysis.  It is deliberately non-gate-bearing because the omission was
    discovered after the method-v2 confirmatory result was observed.
    """

    specification = legacy_calibration["nuisance"]
    design = np.stack(
        [
            np.concatenate(
                [
                    nuisance_vector(row, specification),
                    np.asarray([float(row["full_margin"])], dtype=np.float64),
                ]
            )
            for row in development_rows
        ]
    )
    folds = np.asarray([int(row["fold"]) for row in development_rows], dtype=np.int64)
    repair: dict[str, Any] = {
        "format_version": 1,
        "definition": (
            "post-confirmatory protocol-repair sensitivity: original nuisance "
            "vector plus a linear Full answer margin"
        ),
        "gate_bearing": False,
        "feature_names": [*specification["feature_names"], "full_margin"],
        "legacy_nuisance": specification,
        "folds": {},
    }
    columns = {
        "deletion": "behavior_delete_imageward",
        "replacement": "behavior_replace_imageward",
    }
    for fold in sorted(set(folds.tolist())):
        train = folds != fold
        entry: dict[str, Any] = {"train_n": int(train.sum()), "methods": {}}
        for method, column in columns.items():
            outcome = np.asarray(
                [float(row[column]) for row in development_rows], dtype=np.float64
            )
            beta = np.linalg.lstsq(design[train], outcome[train], rcond=None)[0]
            residual = outcome[train] - design[train] @ beta
            sd = float(np.std(residual, ddof=1))
            if sd <= 0.0:
                raise RuntimeError(f"Degenerate protocol-repair scale: {method}, fold={fold}")
            entry["methods"][method] = {
                "beta": beta.tolist(),
                "residual_mean": float(np.mean(residual)),
                "residual_sd": sd,
            }
        repair["folds"][str(fold)] = entry
    repair["calibration_fingerprint"] = stable_hash(repair)
    return repair


def apply_full_margin_protocol_repair(
    row: dict[str, Any], calibration: dict[str, Any]
) -> dict[str, Any]:
    unsigned = dict(calibration)
    fingerprint = str(unsigned.pop("calibration_fingerprint"))
    if stable_hash(unsigned) != fingerprint:
        raise ValueError("Full-margin protocol-repair calibration fingerprint mismatch")
    specification = calibration["legacy_nuisance"]
    vector = np.concatenate(
        [
            nuisance_vector(row, specification),
            np.asarray([float(row["full_margin"])], dtype=np.float64),
        ]
    )
    fold = str(int(row["fold"]))
    output: dict[str, Any] = {
        "margin_repair_calibration_fingerprint": fingerprint,
    }
    for method, value in {
        "deletion": float(row["behavior_delete_imageward"]),
        "replacement": float(row["behavior_replace_imageward_d34_mean"]),
    }.items():
        parameters = calibration["folds"][fold]["methods"][method]
        residual = value - float(vector @ np.asarray(parameters["beta"], dtype=np.float64))
        z = (residual - float(parameters["residual_mean"])) / float(
            parameters["residual_sd"]
        )
        output[f"margin_repair_residual_{method}"] = residual
        output[f"margin_repair_z_{method}"] = float(z)
    return output


def merge_extension_rows(
    extension_rows: Sequence[dict[str, Any]],
    method_v2_by_case: dict[str, dict[str, Any]],
    calibration: dict[str, Any],
    protocol_repair_calibration: dict[str, Any],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for fresh in extension_rows:
        case_id = str(fresh["case_id"])
        old = method_v2_by_case[case_id]
        row = {
            **fresh,
            "behavior_delete_imageward": float(old["behavior_delete_imageward"]),
            "behavior_replace_imageward_d1": float(
                old["behavior_replace_imageward_d1"]
            ),
            "behavior_replace_imageward_d2": float(
                old["behavior_replace_imageward_d2"]
            ),
            "behavior_replace_imageward_d12_mean": float(
                old["behavior_replace_imageward"]
            ),
            "graded_residual_delete": float(old["graded_residual_delete"]),
            "full_margin": float(old["full_margin"]),
            "answer_star_side": old["answer_star_side"],
            "prior_strength": float(old["prior_strength"]),
            "difficulty": old["difficulty"],
            "fold": int(old["fold"]),
        }
        row.update(apply_legacy_graded_diagnostic(row, calibration))
        row.update(apply_full_margin_protocol_repair(row, protocol_repair_calibration))
        row["behavior_replace_imageward_d1234_mean"] = statistics.fmean(
            [
                row["behavior_replace_imageward_d1"],
                row["behavior_replace_imageward_d2"],
                row["behavior_replace_imageward_d3"],
                row["behavior_replace_imageward_d4"],
            ]
        )
        merged.append(row)
    return merged


def summarize_extension(
    rows: Sequence[dict[str, Any]],
    *,
    split: str,
    expected_n: int,
    failed: int,
) -> dict[str, Any]:
    minimum_n = 90 if split == "development" else 70
    leakage_n = sum(bool(row.get("verbal_sa_leakage")) for row in rows)
    answer_reuse_failure_n = sum(not bool(row.get("answer_star_reused")) for row in rows)
    hash_failure_n = sum(not bool(row.get("full_messages_hash_equal")) for row in rows)
    selection_forward_failure_n = sum(
        not bool(row.get("selection_reused_without_forward")) for row in rows
    )
    hidden_capture_n = sum(bool(row.get("hidden_captured")) for row in rows)
    donor_distinct_failure_n = 0
    symmetry_failure_n = 0
    for row in rows:
        items = [
            str(row[f"donor{index}_item_id"])
            for index in (1, 2, 3, 4)
        ]
        donor_distinct_failure_n += int(
            len(set(items)) != 4 or str(row["item_id"]) in items
        )
        for index in NEW_DONOR_INDICES:
            sources = row["condition_sources"]
            symmetry_failure_n += int(
                sources[f"replace_text_d{index}"]["text_source_item"]
                != sources[f"replace_image_d{index}"]["image_source_item"]
                or sources[f"replace_text_d{index}"]["image_source_item"]
                != sources[f"replace_image_d{index}"]["text_source_item"]
            )
    technical_gate = bool(
        len(rows) >= minimum_n
        and len(rows) == expected_n
        and failed == 0
        and leakage_n == 0
        and answer_reuse_failure_n == 0
        and hash_failure_n == 0
        and selection_forward_failure_n == 0
        and hidden_capture_n == 0
        and donor_distinct_failure_n == 0
        and symmetry_failure_n == 0
    )

    split_half = _association(
        rows,
        "behavior_replace_imageward_d12_mean",
        "behavior_replace_imageward_d34_mean",
    )
    split_half_sign = _sign_agreement(
        rows,
        "behavior_replace_imageward_d12_mean",
        "behavior_replace_imageward_d34_mean",
    )
    split_half_icc = _icc_consistency(
        np.asarray(
            [
                [
                    float(row["behavior_replace_imageward_d12_mean"]),
                    float(row["behavior_replace_imageward_d34_mean"]),
                ]
                for row in rows
            ],
            dtype=np.float64,
        )
    )
    split_half_ci = split_half.get("spearman_item_bootstrap") or {
        "ci95": [None, None]
    }
    split_half_gate = bool(
        split_half_ci["ci95"][0] is not None
        and split_half_ci["ci95"][0] > 0.0
        and split_half_sign >= 0.70
        and split_half_icc is not None
        and split_half_icc >= 0.60
    )

    raw = _association(
        rows, "behavior_delete_imageward", "behavior_replace_imageward_d34_mean"
    )
    raw_sign = _sign_agreement(
        rows, "behavior_delete_imageward", "behavior_replace_imageward_d34_mean"
    )
    raw_alpha = _cronbach_two_indicator(raw["pearson"])
    raw_ci = raw.get("spearman_item_bootstrap") or {"ci95": [None, None]}
    fold_metrics: list[dict[str, Any]] = []
    for fold in sorted({int(row["fold"]) for row in rows}):
        selected = [row for row in rows if int(row["fold"]) == fold]
        association = _association(
            selected,
            "behavior_delete_imageward",
            "behavior_replace_imageward_d34_mean",
        )
        fold_metrics.append(
            {
                "fold": fold,
                "n": len(selected),
                "raw_spearman": association["spearman"],
            }
        )
    positive_raw_folds = sum(
        metric["raw_spearman"] is not None and metric["raw_spearman"] > 0.0
        for metric in fold_metrics
    )
    raw_gate = bool(
        raw_ci["ci95"][0] is not None
        and raw_ci["ci95"][0] > 0.0
        and raw_sign >= 0.70
        and raw_alpha is not None
        and raw_alpha >= 0.60
        and positive_raw_folds >= 4
    )

    legacy_graded = _association(
        rows,
        "graded_residual_delete",
        "legacy_graded_residual_replace_d34",
    )
    legacy_graded_sign = _sign_agreement(
        rows,
        "graded_residual_delete",
        "legacy_graded_residual_replace_d34",
    )
    margin_repair = _association(
        rows,
        "margin_repair_residual_deletion",
        "margin_repair_residual_replacement",
    )
    margin_repair_sign = _sign_agreement(
        rows,
        "margin_repair_residual_deletion",
        "margin_repair_residual_replacement",
    )
    donor_reuse = _donor_reuse_audit(rows)
    dependence_sensitivity = {
        "note": (
            "Target-item cluster bootstrap is primary. Donor-cluster and "
            "leave-one-donor-out results diagnose dependence from donor reuse "
            "and do not alter the predeclared gates."
        ),
        "m12_vs_m34": {
            "target_item_bootstrap": split_half["spearman_item_bootstrap"],
            "donor_cluster_bootstrap": _donor_cluster_bootstrap(
                rows,
                "behavior_replace_imageward_d12_mean",
                "behavior_replace_imageward_d34_mean",
                (1, 2, 3, 4),
            ),
            "leave_one_donor_out": _leave_one_donor_out(
                rows,
                "behavior_replace_imageward_d12_mean",
                "behavior_replace_imageward_d34_mean",
                (1, 2, 3, 4),
            ),
        },
        "deletion_vs_m34": {
            "target_item_bootstrap": raw["spearman_item_bootstrap"],
            "donor_cluster_bootstrap": _donor_cluster_bootstrap(
                rows,
                "behavior_delete_imageward",
                "behavior_replace_imageward_d34_mean",
                (3, 4),
                seed=SEED + 1,
            ),
            "leave_one_donor_out": _leave_one_donor_out(
                rows,
                "behavior_delete_imageward",
                "behavior_replace_imageward_d34_mean",
                (3, 4),
            ),
        },
    }
    within_answer_side = {
        "note": "diagnostic only; primary gates use the raw, choice-coupled estimands",
        "m12_vs_m34": _within_group_centered_association(
            rows,
            "behavior_replace_imageward_d12_mean",
            "behavior_replace_imageward_d34_mean",
            "answer_star_side",
        ),
        "deletion_vs_m34": _within_group_centered_association(
            rows,
            "behavior_delete_imageward",
            "behavior_replace_imageward_d34_mean",
            "answer_star_side",
        ),
    }
    compatibility = {
        "note": (
            "diagnostic of donor-content mismatch: support bias is +1 when "
            "donor text matches A*, -1 when donor image matches A*, and 0 otherwise"
        ),
        "asymmetry_vs_r3_minus_r4": _association(
            rows,
            "donor_match_asymmetry_d3_minus_d4",
            "behavior_replace_imageward_d3_minus_d4",
        ),
    }
    margins = np.asarray([float(row["full_margin"]) for row in rows])
    donor_half_disagreement = np.asarray(
        [
            abs(
                float(row["behavior_replace_imageward_d12_mean"])
                - float(row["behavior_replace_imageward_d34_mean"])
            )
            for row in rows
        ]
    )
    method_disagreement = np.asarray(
        [
            abs(
                float(row["behavior_delete_imageward"])
                - float(row["behavior_replace_imageward_d34_mean"])
            )
            for row in rows
        ]
    )
    margin_diagnostics = {
        "note": "post-confirmatory sensitivity only; no margin-based exclusion changes a gate",
        "quantiles": np.quantile(margins, [0.0, 0.25, 0.5, 0.75, 1.0]).tolist(),
        "margin_vs_abs_donor_half_disagreement_spearman": float(
            spearmanr(margins, donor_half_disagreement).statistic
        ),
        "margin_vs_abs_raw_method_disagreement_spearman": float(
            spearmanr(margins, method_disagreement).statistic
        ),
        "thresholds": [],
    }
    for threshold in (0.5, 1.0, 2.0):
        selected = [row for row in rows if float(row["full_margin"]) >= threshold]
        association = _association(
            selected,
            "behavior_delete_imageward",
            "behavior_replace_imageward_d34_mean",
        )
        margin_diagnostics["thresholds"].append(
            {
                "minimum_margin": threshold,
                "n": len(selected),
                "raw_spearman": association["spearman"],
            }
        )

    overall = bool(technical_gate and split_half_gate and raw_gate)
    return {
        "title": f"Prospective donor replication extension — {split}",
        "status": "completed",
        "split": split,
        "planned": expected_n,
        "completed": len(rows),
        "failed": failed,
        "unique_items": len({str(row["item_id"]) for row in rows}),
        "technical": {
            "verbal_sa_leakage_n": leakage_n,
            "answer_star_reuse_failure_n": answer_reuse_failure_n,
            "full_messages_hash_failure_n": hash_failure_n,
            "selection_forward_failure_n": selection_forward_failure_n,
            "hidden_capture_n": hidden_capture_n,
            "donor_distinct_failure_n": donor_distinct_failure_n,
            "symmetry_failure_n": symmetry_failure_n,
            "gate_passed": technical_gate,
        },
        "donor_split_half": {
            "m12_vs_m34": split_half,
            "sign_agreement": split_half_sign,
            "icc_consistency": split_half_icc,
            "gate_passed": split_half_gate,
        },
        "raw_cross_method_replication": {
            "deletion_vs_fresh_m34": raw,
            "sign_agreement": raw_sign,
            "cronbach_alpha_two_indicator": raw_alpha,
            "positive_fold_count": positive_raw_folds,
            "fold_metrics": fold_metrics,
            "gate_passed": raw_gate,
        },
        "legacy_graded_secondary": {
            "gate_bearing": False,
            "note": (
                "secondary diagnostic under the original method-v2 nuisance rule; "
                "it cannot reverse the failed method-v2 confirmatory gate"
            ),
            "deletion_vs_fresh_m34": legacy_graded,
            "sign_agreement": legacy_graded_sign,
            "cronbach_alpha_two_indicator": _cronbach_two_indicator(
                legacy_graded["pearson"]
            ),
        },
        "full_margin_protocol_repair_sensitivity": {
            "gate_bearing": False,
            "note": (
                "The written method-v2 protocol specified Full margin but the "
                "implemented frozen nuisance vector omitted it. This development-fit "
                "linear-margin repair is post-confirmatory and cannot overwrite the "
                "original gate."
            ),
            "deletion_vs_fresh_m34": margin_repair,
            "sign_agreement": margin_repair_sign,
            "cronbach_alpha_two_indicator": _cronbach_two_indicator(
                margin_repair["pearson"]
            ),
        },
        "margin_sensitivity": margin_diagnostics,
        "donor_reuse_audit": donor_reuse,
        "dependence_sensitivity": dependence_sensitivity,
        "within_answer_side_centered": within_answer_side,
        "donor_match_asymmetry": compatibility,
        "extension_gate_passed": overall,
        "gate_rule": {
            "technical": (
                f"n={expected_n} and n>={minimum_n}; zero failures/leakage/hash "
                "drift/selection reruns/hidden capture/donor asymmetry"
            ),
            "donor_split_half": (
                "M12-vs-M34 bootstrap Spearman lower>0, sign>=.70, ICC>=.60"
            ),
            "raw_cross_method": (
                "D-vs-M34 bootstrap Spearman lower>0, sign>=.70, alpha>=.60, "
                ">=4/5 positive folds"
            ),
        },
        "claim": (
            "raw choice-coupled reliance prospectively replicated across fresh donor interventions on the existing items"
            if overall
            else "fresh donor intervention replication failed; retain deletion and replacement as separate estimands"
        ),
        "claim_limit": (
            "This post-confirmatory extension neither reverses the method-v2 gate nor "
            "establishes choice-independent or new-item generalization."
        ),
    }


def write_summary_markdown(directory: Path, summary: dict[str, Any]) -> None:
    split_half = summary["donor_split_half"]
    raw = summary["raw_cross_method_replication"]
    graded = summary["legacy_graded_secondary"]
    lines = [
        f"# {summary['title']}",
        "",
        f"- Evaluable: {summary['completed']}/{summary['planned']}; failures={summary['failed']}.",
        f"- Technical gate: {summary['technical']['gate_passed']}.",
        f"- M12↔M34 Spearman: {split_half['m12_vs_m34']['spearman']}; sign={split_half['sign_agreement']}; ICC={split_half['icc_consistency']}; gate={split_half['gate_passed']}.",
        f"- Deletion↔fresh M34 Spearman: {raw['deletion_vs_fresh_m34']['spearman']}; sign={raw['sign_agreement']}; alpha={raw['cronbach_alpha_two_indicator']}; gate={raw['gate_passed']}.",
        f"- Legacy graded diagnostic Spearman: {graded['deletion_vs_fresh_m34']['spearman']} (not gate-bearing).",
        f"- Overall extension gate: {summary['extension_gate_passed']}.",
        "",
        summary["claim"],
        "",
        summary["claim_limit"],
        "",
    ]
    atomic_write_text(directory / f"{summary['split']}_summary.md", "\n".join(lines))


def _load_method_v2_calibration(experiment_dir: str | Path) -> dict[str, Any]:
    path = method_v2_root(experiment_dir) / "frozen_measurement_rule.json"
    payload = __import__("json").loads(path.read_text(encoding="utf-8"))
    fingerprint = str(payload.get("rule_fingerprint", ""))
    unsigned = dict(payload)
    unsigned.pop("rule_fingerprint", None)
    if stable_hash(unsigned) != fingerprint:
        raise ValueError("Method-v2 frozen rule fingerprint mismatch")
    return payload["calibration"]


def run_extension_panel(
    runtime: Stage3Runtime,
    artifacts: SAFormationArtifacts,
    output: Path,
    *,
    split: str,
    deadline: Callable[[], None],
) -> dict[str, Any]:
    cohort, method_v2_by_case, donors, case_by_key = build_extension_plan(
        artifacts, split
    )
    method_v2_path = method_v2_root(artifacts.experiment_dir) / f"{split}_analysis.jsonl"
    method_v2_sha256 = hashlib.sha256(method_v2_path.read_bytes()).hexdigest()
    manifest = build_extension_manifest(
        split,
        cohort,
        method_v2_by_case,
        donors,
        method_v2_sha256=method_v2_sha256,
    )
    manifest_path = output / f"{split}_cohort_manifest.json"
    if manifest_path.is_file():
        existing = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("manifest_fingerprint") != manifest["manifest_fingerprint"]:
            raise ValueError(f"{split} extension manifest fingerprint changed")
    else:
        atomic_write_json(manifest_path, manifest)

    result_path = output / f"{split}_results.jsonl"
    terminal = {
        str(row["intervention_key"])
        for row in _latest_rows(result_path)
        if int(row.get("extension_method_version", -1)) == EXTENSION_METHOD_VERSION
        and row.get("status") == "completed"
    }
    for target in cohort:
        deadline()
        case_id = str(target["case_id"])
        key = f"donor_replication|{split}|{case_id}"
        if key in terminal:
            continue
        old = method_v2_by_case[case_id]
        d3, d4 = donors[case_id]
        base = {
            "intervention_key": key,
            "experiment": "prospective_donor_replication_extension",
            "extension_method_version": EXTENSION_METHOD_VERSION,
            "split": split,
            "case_id": case_id,
            "item_id": target["item_id"],
            "prior_index": int(target["prior_index"]),
            "condition": target["condition"],
            "difficulty": target["difficulty"],
            "fold": int(target["fold"]),
            "text_answer": target["text_answer"],
            "image_answer": target["image_answer"],
            "prior_strength": float(old["prior_strength"]),
            "answer_star": old["answer_star"],
            "donor1_case_id": old["donor1_case_id"],
            "donor1_item_id": old["donor1_item_id"],
            "donor2_case_id": old["donor2_case_id"],
            "donor2_item_id": old["donor2_item_id"],
            "donor3_case_id": d3["case_id"],
            "donor3_item_id": d3["item_id"],
            "donor3_text_answer": d3["text_answer"],
            "donor3_image_answer": d3["image_answer"],
            "donor4_case_id": d4["case_id"],
            "donor4_item_id": d4["item_id"],
            "donor4_text_answer": d4["text_answer"],
            "donor4_image_answer": d4["image_answer"],
            "method_v2_analysis_sha256": method_v2_sha256,
            "method_v2_row_fingerprint": stable_hash(old),
            "manifest_fingerprint": manifest["manifest_fingerprint"],
        }
        append_jsonl(
            result_path,
            _safe_record(
                base,
                lambda target=target, old=old, donor_pair=(d3, d4): measure_extension_case(
                    runtime, target, old, donor_pair, case_by_key
                ),
            ),
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    latest = [
        row
        for row in _latest_rows(result_path)
        if int(row.get("extension_method_version", -1)) == EXTENSION_METHOD_VERSION
    ]
    completed = [row for row in latest if row.get("status") == "completed"]
    failed_rows = [row for row in latest if row.get("status") == "failed"]
    if len(latest) != len(cohort):
        raise RuntimeError(
            f"{split} extension reached {len(latest)}/{len(cohort)} terminal rows"
        )
    if failed_rows:
        raise RuntimeError(
            f"{split} extension has {len(failed_rows)} retryable failures; resume first"
        )
    calibration = _load_method_v2_calibration(artifacts.experiment_dir)
    method_v2_development = load_method_v2_rows(
        artifacts.experiment_dir, "development"
    )
    protocol_repair = fit_full_margin_protocol_repair(
        method_v2_development, calibration
    )
    repair_path = output / "full_margin_protocol_repair.json"
    if repair_path.is_file():
        existing_repair = __import__("json").loads(
            repair_path.read_text(encoding="utf-8")
        )
        if existing_repair.get("calibration_fingerprint") != protocol_repair.get(
            "calibration_fingerprint"
        ):
            raise ValueError("Full-margin protocol-repair calibration drifted")
    else:
        atomic_write_json(repair_path, protocol_repair)
    analysis = merge_extension_rows(
        completed,
        method_v2_by_case,
        calibration,
        protocol_repair,
    )
    write_jsonl_atomic(output / f"{split}_analysis.jsonl", analysis)
    summary = summarize_extension(
        analysis,
        split=split,
        expected_n=len(cohort),
        failed=len(failed_rows),
    )
    atomic_write_json(output / f"{split}_summary.json", summary)
    write_summary_markdown(output, summary)
    if split == "confirmatory":
        frozen = {
            "format_version": 1,
            "extension_method_version": EXTENSION_METHOD_VERSION,
            "confirmatory_manifest_fingerprint": manifest["manifest_fingerprint"],
            "method_v2_analysis_sha256": method_v2_sha256,
            "gate_formula": summary["gate_rule"],
            "confirmatory_extension_gate_passed": summary[
                "extension_gate_passed"
            ],
            "development_allowed": summary["extension_gate_passed"],
            "claim_scope": summary["claim_limit"],
        }
        frozen["rule_fingerprint"] = stable_hash(frozen)
        atomic_write_json(output / "frozen_extension_rule.json", frozen)
    aggregate: dict[str, Any] = {}
    for name in ("confirmatory", "development"):
        path = output / f"{name}_summary.json"
        if path.is_file():
            aggregate[name] = __import__("json").loads(path.read_text(encoding="utf-8"))
    atomic_write_json(output / "summary.json", aggregate)
    return summary


def verify_development_allowed(output: str | Path) -> dict[str, Any]:
    path = Path(output) / "frozen_extension_rule.json"
    if not path.is_file():
        raise ValueError("Development extension requires completed confirmatory donor replication")
    frozen = __import__("json").loads(path.read_text(encoding="utf-8"))
    fingerprint = str(frozen.get("rule_fingerprint", ""))
    unsigned = dict(frozen)
    unsigned.pop("rule_fingerprint", None)
    if stable_hash(unsigned) != fingerprint:
        raise ValueError("Frozen extension rule fingerprint mismatch")
    if not frozen.get("development_allowed"):
        raise ValueError("Confirmatory donor-replication gate failed; development is prohibited")
    return frozen
