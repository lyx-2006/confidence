"""Read-only post-hoc sensitivities for the completed Stage-09 History panel.

This module contains no model code and never participates in the frozen
qualification gate.  It consumes the completed report-formation branch grid,
reconstructs the same twelve factorial contrasts used by the formal analysis,
and asks whether prompt length, the frozen endpoint side, or replayed-answer
identity changes their interpretation.

The inferential unit is always the recipient item.  History cells belonging to
one recipient are averaged or differenced before bootstrapping; the 320 History
cells are never treated as 320 independent observations.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .prospective_history_response_stats import (
    BRANCHES,
    build_factorial_rows,
    association_summary,
    paired_effect_summary,
)


BOOTSTRAP_ITERATIONS = 1000
BASE_SEED = 42
OUTCOMES = ("A_prediction", "V")
EFFECTS = ("modality", "replay", "interaction", "history_vs_none")
FAMILIES = ("relevant", "irrelevant", "bundle_difference")
ENDPOINT_SIDES = ("image", "text")


def contrast_specs() -> tuple[dict[str, str], ...]:
    """Return the frozen twelve post-hoc contrast identifiers."""

    output: list[dict[str, str]] = []
    for family in FAMILIES:
        prefix = "did" if family == "bundle_difference" else family
        for effect in EFFECTS:
            output.append(
                {
                    "id": f"{family}.{effect}",
                    "family": family,
                    "effect": effect,
                    "key_prefix": f"{prefix}_{effect}",
                }
            )
    return tuple(output)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _descriptive(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return {
            "n": 0,
            "mean": None,
            "sd": None,
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "max": None,
            "unique_n": 0,
        }
    if not np.isfinite(array).all():
        raise ValueError("descriptive statistics received a non-finite value")
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if len(array) > 1 else None,
        "min": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
        "unique_n": int(len(np.unique(array))),
    }


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if _finite(row.get(key))]
    return float(np.mean(values)) if values else math.nan


def _annotate_paired_summary(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    *,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    value = paired_effect_summary(rows, key, iterations=iterations, seed=seed)
    value["bootstrap"]["seed"] = int(seed)
    return value


def _annotate_association(
    rows: Sequence[Mapping[str, Any]],
    left: str,
    right: str,
    *,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    value = association_summary(
        rows,
        left,
        right,
        iterations=iterations,
        seed=seed,
    )
    value["spearman_bootstrap"]["seed"] = int(seed)
    return value


def within_fold_group_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    statistic: Callable[[Sequence[Mapping[str, Any]]], float],
    *,
    group_key: str,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BASE_SEED,
) -> dict[str, Any]:
    """Resample recipient items within immutable fold and baseline group.

    For endpoint-side differences, holding the fold-by-side cell counts fixed
    avoids a bootstrap replicate changing the observed 29/11 composition.  A
    recipient item must occur exactly once in ``rows``.
    """

    values = [dict(row) for row in rows]
    item_ids = [str(row.get("item_id", "")) for row in values]
    if any(not item_id for item_id in item_ids) or len(set(item_ids)) != len(values):
        raise ValueError("bootstrap requires exactly one row per recipient item")
    if not values:
        return {
            "estimate": None,
            "ci95": [None, None],
            "iterations": int(iterations),
            "valid": 0,
            "seed": int(seed),
            "resampling": f"recipient-item-within-fixed-fold-and-{group_key}",
        }
    strata: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in values:
        group = str(row.get(group_key, ""))
        if not group:
            raise ValueError(f"bootstrap row lacks {group_key}")
        strata[(int(row["fold"]), group)].append(row)
    estimate = float(statistic(values))
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(int(iterations)):
        sample: list[dict[str, Any]] = []
        for stratum, group in sorted(strata.items()):
            indices = rng.integers(0, len(group), size=len(group))
            sample.extend(
                {
                    **group[int(index)],
                    "item_id": f"{stratum[0]}:{stratum[1]}:{slot}",
                }
                for slot, index in enumerate(indices)
            )
        try:
            result = float(statistic(sample))
        except (ValueError, ZeroDivisionError, FloatingPointError):
            continue
        if math.isfinite(result):
            samples.append(result)
    return {
        "estimate": estimate if math.isfinite(estimate) else None,
        "ci95": (
            [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]
            if samples
            else [None, None]
        ),
        "iterations": int(iterations),
        "valid": len(samples),
        "seed": int(seed),
        "resampling": f"recipient-item-within-fixed-fold-and-{group_key}",
    }


def validate_and_normalize_rows(
    records: Sequence[Mapping[str, Any]],
    endpoint_rows: Sequence[Mapping[str, Any]],
    *,
    require_frozen_shape: bool = True,
) -> list[dict[str, Any]]:
    """Validate the report grid against the immutable endpoint manifest."""

    endpoints: dict[str, dict[str, Any]] = {}
    for raw in endpoint_rows:
        row = dict(raw)
        case_id = str(row.get("case_id", ""))
        if not case_id or case_id in endpoints:
            raise ValueError("endpoint manifest has a missing or duplicate case_id")
        endpoints[case_id] = row
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in records:
        row = dict(raw)
        if row.get("status") != "completed":
            raise ValueError("post-hoc input contains a non-completed branch")
        case_id = str(row.get("case_id", ""))
        branch = str(row.get("branch", ""))
        if case_id not in endpoints:
            raise ValueError(f"report row is absent from endpoint manifest: {case_id}")
        if branch not in BRANCHES:
            raise ValueError(f"unknown report branch: {branch}")
        key = (case_id, branch)
        if key in seen:
            raise ValueError(f"duplicate report branch: {case_id}/{branch}")
        seen.add(key)
        endpoint = endpoints[case_id]
        for field in ("item_id", "fold", "answer_star", "answer_star_side"):
            if str(row.get(field)) != str(endpoint.get(field)):
                raise ValueError(f"endpoint/report mismatch for {case_id}: {field}")
        side = str(endpoint.get("answer_star_side", "")).lower()
        if side not in ENDPOINT_SIDES:
            raise ValueError(f"unsupported endpoint side for {case_id}: {side}")
        common9 = row.get("joint_common9")
        if not isinstance(common9, Mapping) or not _finite(common9.get("input_token_count")):
            raise ValueError(f"missing joint input token count: {case_id}/{branch}")
        if not _finite(row.get("A_prediction")) or not _finite(row.get("V")):
            raise ValueError(f"missing A/V outcome: {case_id}/{branch}")
        branch_factors = row.get("branch_factors")
        if not isinstance(branch_factors, Mapping):
            raise ValueError(f"missing branch factors: {case_id}/{branch}")
        normalized = dict(row)
        normalized["input_token_count"] = int(common9["input_token_count"])
        normalized["endpoint_side"] = side
        output.append(normalized)

    by_case: dict[str, set[str]] = defaultdict(set)
    for row in output:
        by_case[str(row["case_id"])].add(str(row["branch"]))
    incomplete = [case_id for case_id, branches in by_case.items() if branches != set(BRANCHES)]
    if incomplete:
        raise ValueError(f"incomplete report branch grids: {len(incomplete)}")
    if set(by_case) != set(endpoints):
        raise ValueError("endpoint and report case sets differ")

    if require_frozen_shape:
        if len(endpoints) != 40 or len(output) != 360:
            raise ValueError("frozen Stage-09 report shape must be 40 items x 9 branches")
        item_ids = {str(row["item_id"]) for row in endpoints.values()}
        if len(item_ids) != 40:
            raise ValueError("frozen endpoint must contain 40 unique items")
        fold_counts = Counter(int(row["fold"]) for row in endpoints.values())
        if fold_counts != Counter({fold: 8 for fold in range(5)}):
            raise ValueError(f"frozen endpoint fold counts changed: {dict(fold_counts)}")
        side_counts = Counter(
            str(row["answer_star_side"]).lower() for row in endpoints.values()
        )
        if side_counts != Counter({"image": 29, "text": 11}):
            raise ValueError(f"frozen endpoint side counts changed: {dict(side_counts)}")
    return sorted(output, key=lambda row: (str(row["case_id"]), str(row["branch"])))


def build_posthoc_factorial_rows(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build item-level A, V, and token-count contrasts."""

    rows = build_factorial_rows(
        records,
        outcomes=("A_prediction", "V", "input_token_count"),
    )
    endpoint_by_case = {
        str(row["case_id"]): str(row["endpoint_side"])
        for row in records
    }
    for row in rows:
        row["endpoint_side"] = endpoint_by_case[str(row["case_id"])]
    return rows


def branch_token_count_statistics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Describe total common-9 input length and paired History increments."""

    no_history = {
        str(row["case_id"]): int(row["input_token_count"])
        for row in records
        if row["branch"] == "no_history"
    }
    output: dict[str, Any] = {}
    for branch in BRANCHES:
        rows = [row for row in records if row["branch"] == branch]
        total = [int(row["input_token_count"]) for row in rows]
        increment = [
            int(row["input_token_count"]) - no_history[str(row["case_id"])]
            for row in rows
        ]
        output[branch] = {
            "total_input_token_count": _descriptive(total),
            "paired_increment_vs_no_history": _descriptive(increment),
        }
    return output


def token_contrast_alignment(
    factorial_rows: Sequence[Mapping[str, Any]],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    """Correlate each item-level token contrast with its A and V contrast."""

    output: dict[str, Any] = {}
    for index, spec in enumerate(contrast_specs()):
        prefix = spec["key_prefix"]
        token_key = f"{prefix}_input_token_count"
        seed = BASE_SEED + 1000 + 10 * index
        output[spec["id"]] = {
            "family": spec["family"],
            "effect": spec["effect"],
            "token_contrast": _annotate_paired_summary(
                factorial_rows,
                token_key,
                seed=seed,
                iterations=iterations,
            ),
            "token_vs_A_prediction": _annotate_association(
                factorial_rows,
                token_key,
                f"{prefix}_A_prediction",
                seed=seed + 1,
                iterations=iterations,
            ),
            "token_vs_V": _annotate_association(
                factorial_rows,
                token_key,
                f"{prefix}_V",
                seed=seed + 2,
                iterations=iterations,
            ),
        }
    return output


def endpoint_side_stratified_effects(
    factorial_rows: Sequence[Mapping[str, Any]],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    """Summarize all twelve contrasts separately by the frozen endpoint side."""

    output: dict[str, Any] = {}
    for index, spec in enumerate(contrast_specs()):
        prefix = spec["key_prefix"]
        entry: dict[str, Any] = {
            "family": spec["family"],
            "effect": spec["effect"],
            "strata": {},
        }
        for side_index, side in enumerate(ENDPOINT_SIDES):
            rows = [row for row in factorial_rows if row["endpoint_side"] == side]
            base_seed = BASE_SEED + 2000 + 20 * index + 5 * side_index
            entry["strata"][side] = {
                "n": len(rows),
                "fold_counts": dict(sorted(Counter(int(row["fold"]) for row in rows).items())),
                "outcomes": {
                    outcome: _annotate_paired_summary(
                        rows,
                        f"{prefix}_{outcome}",
                        seed=base_seed + outcome_index,
                        iterations=iterations,
                    )
                    for outcome_index, outcome in enumerate(OUTCOMES)
                },
            }
        output[spec["id"]] = entry
    return output


def endpoint_side_differences(
    factorial_rows: Sequence[Mapping[str, Any]],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    """Estimate endpoint-image minus endpoint-text effect heterogeneity."""

    output: dict[str, Any] = {}
    for index, spec in enumerate(contrast_specs()):
        prefix = spec["key_prefix"]
        entry: dict[str, Any] = {
            "family": spec["family"],
            "effect": spec["effect"],
            "endpoint_image_n": sum(
                row["endpoint_side"] == "image" for row in factorial_rows
            ),
            "endpoint_text_n": sum(
                row["endpoint_side"] == "text" for row in factorial_rows
            ),
            "outcomes": {},
        }
        for outcome_index, outcome in enumerate(OUTCOMES):
            key = f"{prefix}_{outcome}"

            def statistic(sample: Sequence[Mapping[str, Any]], value_key: str = key) -> float:
                image_mean = _mean(
                    [row for row in sample if row["endpoint_side"] == "image"],
                    value_key,
                )
                text_mean = _mean(
                    [row for row in sample if row["endpoint_side"] == "text"],
                    value_key,
                )
                return image_mean - text_mean

            seed = BASE_SEED + 3000 + 10 * index + outcome_index
            result = within_fold_group_bootstrap(
                factorial_rows,
                statistic,
                group_key="endpoint_side",
                iterations=iterations,
                seed=seed,
            )
            result["endpoint_image_mean"] = _mean(
                [row for row in factorial_rows if row["endpoint_side"] == "image"],
                key,
            )
            result["endpoint_text_mean"] = _mean(
                [row for row in factorial_rows if row["endpoint_side"] == "text"],
                key,
            )
            entry["outcomes"][outcome] = result
        output[spec["id"]] = entry
    return output


def av_change_alignment(
    factorial_rows: Sequence[Mapping[str, Any]],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    """Measure item-level alignment between A and V changes for 12 contrasts."""

    output: dict[str, Any] = {}
    for index, spec in enumerate(contrast_specs()):
        prefix = spec["key_prefix"]
        seed = BASE_SEED + 4000 + index
        output[spec["id"]] = {
            "family": spec["family"],
            "effect": spec["effect"],
            "A_vs_V": _annotate_association(
                factorial_rows,
                f"{prefix}_A_prediction",
                f"{prefix}_V",
                seed=seed,
                iterations=iterations,
            ),
        }
    return output


def _normalized_answer(value: Any) -> str:
    return " ".join(str(value).strip().casefold().split())


def build_answer_match_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    relevance: str | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Create one matched-minus-mismatched row per eligible recipient item.

    Match is deliberately recomputed as ``replayed_answer == answer_star``.
    It does *not* use ``answer_identity_matches_target``, whose frozen meaning
    is equality to the target's same-side answer rather than equality to A*.
    """

    if relevance not in {None, "relevant", "irrelevant"}:
        raise ValueError(f"unknown relevance scope: {relevance}")
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    no_history: dict[str, dict[str, Any]] = {}
    for row in records:
        case_id = str(row["case_id"])
        if row["branch"] == "no_history":
            no_history[case_id] = dict(row)
            continue
        if relevance is not None and not str(row["branch"]).startswith(f"{relevance}_"):
            continue
        branch_factors = row["branch_factors"]
        replayed = _normalized_answer(branch_factors.get("replayed_answer"))
        answer_star = _normalized_answer(row["answer_star"])
        enriched = dict(row)
        enriched["replayed_answer_matches_A_star"] = replayed == answer_star
        by_case[case_id].append(enriched)

    output: list[dict[str, Any]] = []
    matched_cell_n = mismatch_cell_n = 0
    paired_matched_cell_n = paired_mismatch_cell_n = 0
    items_with_match = items_with_mismatch = 0
    for case_id, cells in sorted(by_case.items()):
        matched = [row for row in cells if row["replayed_answer_matches_A_star"]]
        mismatched = [row for row in cells if not row["replayed_answer_matches_A_star"]]
        matched_cell_n += len(matched)
        mismatch_cell_n += len(mismatched)
        items_with_match += bool(matched)
        items_with_mismatch += bool(mismatched)
        if not matched or not mismatched:
            continue
        paired_matched_cell_n += len(matched)
        paired_mismatch_cell_n += len(mismatched)
        base = no_history[case_id]
        item: dict[str, Any] = {
            "case_id": case_id,
            "item_id": str(base["item_id"]),
            "fold": int(base["fold"]),
            "endpoint_side": str(base["endpoint_side"]),
            "matched_cell_n": len(matched),
            "mismatched_cell_n": len(mismatched),
        }
        for outcome in OUTCOMES:
            matched_mean = _mean(matched, outcome)
            mismatch_mean = _mean(mismatched, outcome)
            no_history_value = float(base[outcome])
            item[f"matched_mean_{outcome}"] = matched_mean
            item[f"mismatched_mean_{outcome}"] = mismatch_mean
            item[f"matched_minus_mismatched_{outcome}"] = matched_mean - mismatch_mean
            item[f"matched_minus_no_history_{outcome}"] = matched_mean - no_history_value
            item[f"mismatched_minus_no_history_{outcome}"] = mismatch_mean - no_history_value
        output.append(item)
    counts = {
        "history_cell_n": matched_cell_n + mismatch_cell_n,
        "matched_history_cell_n": matched_cell_n,
        "mismatched_history_cell_n": mismatch_cell_n,
        "recipient_item_n": len(by_case),
        "items_with_at_least_one_match_n": items_with_match,
        "items_with_at_least_one_mismatch_n": items_with_mismatch,
        "items_without_any_match_n": len(by_case) - items_with_match,
        "paired_eligible_item_n": len(output),
        "paired_subset_matched_cell_n": paired_matched_cell_n,
        "paired_subset_mismatched_cell_n": paired_mismatch_cell_n,
        "paired_coverage_fraction": (
            float(len(output) / len(by_case)) if by_case else None
        ),
    }
    return output, counts


def answer_match_reinterpretation(
    records: Sequence[Mapping[str, Any]],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    """Reorient replay cells by whether their answer equals frozen A*."""

    output: dict[str, Any] = {}
    # Deliberately do not pool Relevant and Irrelevant cells into an
    # ``all_history`` contrast.  In recipients without an Irrelevant A* match,
    # pooling would make the matched mean Relevant-only while the mismatched
    # mean mixed Relevant and Irrelevant: a composition-changing estimand.
    for scope_index, (scope, relevance) in enumerate(
        (("relevant_history", "relevant"), ("irrelevant_history", "irrelevant"))
    ):
        rows, counts = build_answer_match_rows(records, relevance=relevance)
        entry: dict[str, Any] = {
            **counts,
            "outcomes": {},
            "inferential_unit": "recipient_item",
            "interpretation": (
                "primary complete-pair reorientation"
                if scope == "relevant_history"
                else "coverage-limited sensitivity among recipients with an Irrelevant replay matching A*"
            ),
        }
        for outcome_index, outcome in enumerate(OUTCOMES):
            base_seed = BASE_SEED + 5000 + 20 * scope_index + 5 * outcome_index
            entry["outcomes"][outcome] = {
                contrast: _annotate_paired_summary(
                    rows,
                    f"{contrast}_{outcome}",
                    seed=base_seed + contrast_index,
                    iterations=iterations,
                )
                for contrast_index, contrast in enumerate(
                    (
                        "matched_minus_mismatched",
                        "matched_minus_no_history",
                        "mismatched_minus_no_history",
                    )
                )
            }
        output[scope] = entry
    output["definition"] = {
        "match": "normalized replayed_answer equals frozen phase-0 answer_star (A*)",
        "not_used": "branch_factors.answer_identity_matches_target",
        "why_not_used": (
            "That field means equality to the target's same-side answer and is not "
            "the requested equality to A*."
        ),
        "cell_independence_assumed": False,
        "all_history_pooled_contrast_reported": False,
        "all_history_omission_reason": (
            "Pooling would be composition-changing: for recipients without an "
            "Irrelevant A* match, matched cells are Relevant-only whereas mismatched "
            "cells contain both Relevant and Irrelevant History."
        ),
    }
    return output


def run_posthoc_analysis(
    records: Sequence[Mapping[str, Any]],
    endpoint_rows: Sequence[Mapping[str, Any]],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    require_frozen_shape: bool = True,
) -> dict[str, Any]:
    """Run all pure Stage-09 post-hoc calculations."""

    normalized = validate_and_normalize_rows(
        records,
        endpoint_rows,
        require_frozen_shape=require_frozen_shape,
    )
    factorial = build_posthoc_factorial_rows(normalized)
    expected_n = 40 if require_frozen_shape else len(endpoint_rows)
    if len(factorial) != expected_n:
        raise ValueError(
            f"expected {expected_n} complete item-level factorial rows, got {len(factorial)}"
        )
    endpoint_counts = Counter(row["endpoint_side"] for row in factorial)
    exact_n = sum(bool(row["history_ordered_pair_exact"]) for row in factorial)
    return {
        "analysis_counts": {
            "report_branch_n": len(normalized),
            "recipient_item_n": len(factorial),
            "endpoint_side_counts": dict(sorted(endpoint_counts.items())),
            "exact_ordered_pair_item_n": exact_n,
            "fallback_item_n": len(factorial) - exact_n,
            "contrast_n": len(contrast_specs()),
        },
        "contrast_definitions": {
            "positive_modality": "Image-history minus Text-history, averaged over replay side",
            "positive_replay": "AI replay minus AT replay, averaged over History modality",
            "interaction": "image_AI - image_AT - text_AI + text_AT",
            "history_vs_none": "mean of the four History cells minus no-History",
            "bundle_difference": "Relevant-History bundle contrast minus Irrelevant-History bundle contrast",
            "endpoint_side_difference": "frozen endpoint-image stratum minus frozen endpoint-text stratum",
        },
        "branch_token_count_statistics": branch_token_count_statistics(normalized),
        "token_count_contrast_alignment": token_contrast_alignment(
            factorial, iterations=iterations
        ),
        "endpoint_side_stratified_contrasts": endpoint_side_stratified_effects(
            factorial, iterations=iterations
        ),
        "endpoint_image_minus_text_differences": endpoint_side_differences(
            factorial, iterations=iterations
        ),
        "A_V_change_alignment": av_change_alignment(
            factorial, iterations=iterations
        ),
        "answer_match_reinterpretation": answer_match_reinterpretation(
            normalized, iterations=iterations
        ),
    }
