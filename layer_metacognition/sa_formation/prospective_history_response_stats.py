"""Statistics for the preregistered Stage-09 History response pilot.

The module is deliberately independent of model code.  Its input is one flat
record per item and branch; this keeps the estimands testable with synthetic
data and makes partial/resumed GPU output easy to audit before analysis.

Branch names are frozen as ``no_history`` and
``{relevant|irrelevant}_{text|image}_{at|ai}``.  Positive modality effects are
Image-history minus Text-history.  Positive replay effects are AI minus AT.
The relevance difference-in-differences is Relevant minus Irrelevant.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BOOTSTRAP_ITERATIONS = 1000
SEED = 42
HISTORY_BRANCHES = tuple(
    f"{relevance}_{modality}_{replay}"
    for relevance in ("relevant", "irrelevant")
    for modality in ("text", "image")
    for replay in ("at", "ai")
)
BRANCHES = ("no_history", *HISTORY_BRANCHES)

PRIMARY_OUTCOMES = (
    "B_D",
    "B_M56",
    "U_prediction",
    "A_prediction",
    "V",
)
SECONDARY_OUTCOMES = (
    "M5",
    "M6",
    "U_coordinate",
    "U_L18_prediction",
    "U_L18_coordinate",
    "A_coordinate",
    "full_logp",
    "full_margin",
    "full_entropy",
    "hard_answer_image",
    "hard_answer_other",
)

QUALIFICATION_BEHAVIOR_NUMERIC_FIELDS = (
    "B_D",
    "B_M56",
    "M5",
    "M6",
    "B_target_shared",
    "U_prediction",
    "U_nuisance_prediction",
)
QUALIFICATION_REPORT_NUMERIC_FIELDS = (
    "A_prediction",
    "V",
)
QUALIFICATION_NUMERIC_FIELDS = (
    *QUALIFICATION_BEHAVIOR_NUMERIC_FIELDS,
    *QUALIFICATION_REPORT_NUMERIC_FIELDS,
)
QUALIFICATION_COMMON_TRUE_FIELDS = (
    "answer_only_no_sa",
    "causal_prefix_equal",
)
QUALIFICATION_BEHAVIOR_TRUE_FIELDS = ("answer_hook_exactly_once",)
QUALIFICATION_REPORT_TRUE_FIELDS = ("joint_hook_exactly_once",)
QUALIFICATION_COMMON_FALSE_FIELDS = ("steering_applied",)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _finite_rows(
    rows: Sequence[Mapping[str, Any]], *keys: str
) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if all(_finite(row.get(key)) for key in keys)]


def _safe_spearman(rows: Sequence[Mapping[str, Any]], left: str, right: str) -> float:
    values = _finite_rows(rows, left, right)
    if len(values) < 3:
        return math.nan
    x = np.asarray([float(row[left]) for row in values], dtype=np.float64)
    y = np.asarray([float(row[right]) for row in values], dtype=np.float64)
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return math.nan
    return float(spearmanr(x, y).statistic)


def fixed_fold_item_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    statistic: Callable[[Sequence[Mapping[str, Any]]], float],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = SEED,
) -> dict[str, Any]:
    """Resample recipient items inside each immutable fold.

    Exactly one analysis row per recipient item is required.  Donor reuse is
    reported elsewhere and must not be hidden by pretending donor assignments
    are independent observations.
    """

    values = [dict(row) for row in rows]
    if len({str(row.get("item_id")) for row in values}) != len(values):
        raise ValueError("fixed-fold bootstrap requires one row per recipient item")
    if not values:
        return {
            "estimate": None,
            "ci95": [None, None],
            "iterations": int(iterations),
            "valid": 0,
            "resampling": "recipient-item-within-fixed-fold",
        }
    by_fold: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in values:
        by_fold[int(row["fold"])].append(row)
    try:
        estimate = float(statistic(values))
    except (ValueError, ZeroDivisionError, FloatingPointError):
        estimate = math.nan
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(int(iterations)):
        sample: list[dict[str, Any]] = []
        for fold in sorted(by_fold):
            group = by_fold[fold]
            indices = rng.integers(0, len(group), size=len(group))
            # Give duplicate resamples unique bootstrap item ids so nested
            # statistics can still enforce the one-row-per-unit invariant.
            sample.extend(
                {**group[int(index)], "item_id": f"{fold}:{slot}"}
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
        "resampling": "recipient-item-within-fixed-fold",
    }


def paired_effect_summary(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = SEED,
) -> dict[str, Any]:
    values = _finite_rows(rows, key)
    array = np.asarray([float(row[key]) for row in values], dtype=np.float64)
    bootstrap = fixed_fold_item_bootstrap(
        values,
        lambda sample: float(np.mean([float(row[key]) for row in sample])),
        iterations=iterations,
        seed=seed,
    )
    nonzero = array[array != 0.0]
    return {
        "n": len(values),
        "unique_items": len({str(row["item_id"]) for row in values}),
        "mean": float(array.mean()) if len(array) else None,
        "sd": float(array.std(ddof=1)) if len(array) > 1 else None,
        "ci95": bootstrap["ci95"],
        "positive_direction_rate": (
            float(np.mean(nonzero > 0)) if len(nonzero) else None
        ),
        "bootstrap": bootstrap,
    }


def association_summary(
    rows: Sequence[Mapping[str, Any]],
    left: str,
    right: str,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = SEED,
) -> dict[str, Any]:
    values = _finite_rows(rows, left, right)
    x = np.asarray([float(row[left]) for row in values], dtype=np.float64)
    y = np.asarray([float(row[right]) for row in values], dtype=np.float64)
    pearson = None
    spearman = None
    if len(values) >= 3 and np.std(x) > 1e-12 and np.std(y) > 1e-12:
        pearson = float(pearsonr(x, y).statistic)
        spearman = float(spearmanr(x, y).statistic)
    bootstrap = fixed_fold_item_bootstrap(
        values,
        lambda sample: _safe_spearman(sample, left, right),
        iterations=iterations,
        seed=seed,
    )
    nonzero = [(a, b) for a, b in zip(x, y) if a != 0.0 and b != 0.0]
    return {
        "n": len(values),
        "unique_items": len({str(row["item_id"]) for row in values}),
        "pearson": pearson,
        "spearman": spearman,
        "spearman_ci95": bootstrap["ci95"],
        "spearman_bootstrap": bootstrap,
        "sign_agreement": (
            float(np.mean([(a > 0) == (b > 0) for a, b in nonzero]))
            if nonzero
            else None
        ),
    }


def prediction_summary(
    rows: Sequence[Mapping[str, Any]],
    target: str,
    prediction: str,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = SEED,
) -> dict[str, Any]:
    values = _finite_rows(rows, target, prediction)
    y = np.asarray([float(row[target]) for row in values], dtype=np.float64)
    p = np.asarray([float(row[prediction]) for row in values], dtype=np.float64)
    association = association_summary(
        values, target, prediction, iterations=iterations, seed=seed
    )
    r2 = float(r2_score(y, p)) if len(values) >= 2 and np.std(y) > 1e-12 else None
    folds: list[dict[str, Any]] = []
    for fold in range(5):
        selected = [row for row in values if int(row["fold"]) == fold]
        folds.append(
            {
                "fold": fold,
                "n": len(selected),
                "spearman": _safe_spearman(selected, target, prediction),
            }
        )
    return {
        "n": len(values),
        "r2": r2,
        "mae": float(mean_absolute_error(y, p)) if len(values) else None,
        "mse": float(mean_squared_error(y, p)) if len(values) else None,
        "association": association,
        "fold_metrics": folds,
        "positive_spearman_fold_count": sum(
            _finite(row["spearman"]) and float(row["spearman"]) > 0
            for row in folds
        ),
    }


def _analysis_outcome_value(row: Mapping[str, Any], outcome: str) -> Any:
    """Return an outcome, deriving hard-side indicators when needed.

    ``other`` is a real post-treatment answer outcome, not missing data.  Some
    resumable records may expose only the three-level ``hard_answer_side``;
    deriving its two indicators keeps all items in the secondary total-effect
    analysis and never changes a primary outcome.
    """

    value = row.get(outcome)
    if _finite(value):
        return value
    side = str(row.get("hard_answer_side", "")).lower()
    if outcome == "hard_answer_image" and side in {"text", "image", "other"}:
        return 1.0 if side == "image" else 0.0
    if outcome == "hard_answer_other" and side in {"text", "image", "other"}:
        return 1.0 if side == "other" else 0.0
    return value


def icc_consistency(two_columns: Sequence[Sequence[float]]) -> float | None:
    """ICC(3,1)-style consistency for two fixed replicate measurements."""

    matrix = np.asarray(two_columns, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 2 or matrix.shape[0] < 3:
        return None
    if not np.isfinite(matrix).all():
        return None
    n, k = matrix.shape
    row_means = matrix.mean(axis=1)
    column_means = matrix.mean(axis=0)
    grand = float(matrix.mean())
    ms_rows = float(k * np.sum((row_means - grand) ** 2) / (n - 1))
    residual = matrix - row_means[:, None] - column_means[None, :] + grand
    ms_error = float(np.sum(residual**2) / ((n - 1) * (k - 1)))
    denominator = ms_rows + (k - 1) * ms_error
    return float((ms_rows - ms_error) / denominator) if denominator > 1e-12 else None


def holm_adjust(p_values: Mapping[str, float | None]) -> dict[str, float | None]:
    """Return monotone Holm-adjusted p-values without dropping missing tests."""

    valid = sorted(
        ((name, float(value)) for name, value in p_values.items() if _finite(value)),
        key=lambda pair: pair[1],
    )
    adjusted: dict[str, float | None] = {name: None for name in p_values}
    running = 0.0
    m = len(valid)
    for rank, (name, value) in enumerate(valid):
        running = max(running, (m - rank) * value)
        adjusted[name] = min(1.0, running)
    return adjusted


def build_factorial_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    outcomes: Sequence[str] = (*PRIMARY_OUTCOMES, *SECONDARY_OUTCOMES),
) -> list[dict[str, Any]]:
    """Collapse complete branch records to one contrast row per item."""

    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    metadata: dict[str, dict[str, Any]] = {}
    for raw in records:
        row = dict(raw)
        if row.get("status", "completed") != "completed":
            continue
        case_id = str(row["case_id"])
        branch = str(row["branch"])
        if branch not in BRANCHES:
            raise ValueError(f"Unknown Stage-09 branch: {branch}")
        if branch in by_case[case_id]:
            raise ValueError(f"Duplicate completed branch {case_id}/{branch}")
        by_case[case_id][branch] = row
        if "history_match_tier" not in row or "history_ordered_pair_exact" not in row:
            raise ValueError(
                f"Branch record lacks frozen History match metadata: {case_id}/{branch}"
            )
        match_tier = str(row["history_match_tier"])
        ordered_pair_exact = row["history_ordered_pair_exact"]
        if not isinstance(ordered_pair_exact, (bool, np.bool_)):
            raise ValueError(
                f"history_ordered_pair_exact is not boolean: {case_id}/{branch}"
            )
        expected_exact = match_tier == "exact_ordered_text_image_answer_pair"
        if bool(ordered_pair_exact) != expected_exact:
            raise ValueError(
                f"History match tier/exact flag disagree: {case_id}/{branch}"
            )
        current = {
            "case_id": case_id,
            "item_id": str(row["item_id"]),
            "fold": int(row["fold"]),
            "history_match_tier": match_tier,
            "history_ordered_pair_exact": bool(ordered_pair_exact),
        }
        # These frozen donor identities make reused-donor sensitivity analyses
        # possible.  They are part of the formal Stage-09 branch contract and
        # must be present and identical in all nine branches.
        for donor_key in (
            "history_donor_item_id",
            "donor5_item_id",
            "donor6_item_id",
        ):
            if row.get(donor_key) is None or str(row[donor_key]) == "":
                raise ValueError(
                    f"Branch record lacks frozen donor identity {donor_key}: "
                    f"{case_id}/{branch}"
                )
            current[donor_key] = str(row[donor_key])
        if case_id in metadata and metadata[case_id] != current:
            raise ValueError(f"Branch metadata drift for {case_id}")
        metadata[case_id] = current
    output: list[dict[str, Any]] = []
    for case_id, branches in sorted(by_case.items()):
        if set(branches) != set(BRANCHES):
            continue
        row = dict(metadata[case_id])
        complete_outcomes: list[str] = []
        for outcome in outcomes:
            branch_values = {
                branch: _analysis_outcome_value(branches[branch], outcome)
                for branch in BRANCHES
            }
            if not all(_finite(branch_values[branch]) for branch in BRANCHES):
                continue
            complete_outcomes.append(str(outcome))
            row[f"no_history_{outcome}"] = float(branch_values["no_history"])
            for relevance in ("relevant", "irrelevant"):
                t_at = float(branch_values[f"{relevance}_text_at"])
                t_ai = float(branch_values[f"{relevance}_text_ai"])
                i_at = float(branch_values[f"{relevance}_image_at"])
                i_ai = float(branch_values[f"{relevance}_image_ai"])
                history_mean = 0.25 * (t_at + t_ai + i_at + i_ai)
                row[f"{relevance}_modality_{outcome}"] = 0.5 * (
                    i_at + i_ai - t_at - t_ai
                )
                row[f"{relevance}_replay_{outcome}"] = 0.5 * (
                    t_ai + i_ai - t_at - i_at
                )
                row[f"{relevance}_interaction_{outcome}"] = (
                    i_ai - i_at - t_ai + t_at
                )
                row[f"{relevance}_history_mean_{outcome}"] = history_mean
                row[f"{relevance}_history_vs_none_{outcome}"] = (
                    history_mean - row[f"no_history_{outcome}"]
                )
            for effect in ("modality", "replay", "interaction", "history_vs_none"):
                row[f"did_{effect}_{outcome}"] = (
                    row[f"relevant_{effect}_{outcome}"]
                    - row[f"irrelevant_{effect}_{outcome}"]
                )
        # A complete branch grid with no complete requested outcome is not an
        # analyzable item and must never inflate factorial_complete_item_n.
        if complete_outcomes:
            row["complete_outcomes"] = complete_outcomes
            output.append(row)
    return output


def _factorial_effect_summaries_core(
    contrast_rows: Sequence[Mapping[str, Any]],
    *,
    outcomes: Sequence[str],
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    seed = SEED
    for family in ("relevant", "irrelevant"):
        output[family] = {}
        for effect in ("modality", "replay", "interaction", "history_vs_none"):
            output[family][effect] = {}
            for outcome in outcomes:
                seed += 1
                output[family][effect][outcome] = paired_effect_summary(
                    contrast_rows,
                    f"{family}_{effect}_{outcome}",
                    iterations=iterations,
                    seed=seed,
                )
    bundle_did: dict[str, Any] = {}
    for effect in ("modality", "replay", "interaction", "history_vs_none"):
        bundle_did[effect] = {}
        for outcome in outcomes:
            seed += 1
            bundle_did[effect][outcome] = paired_effect_summary(
                contrast_rows,
                f"did_{effect}_{outcome}",
                iterations=iterations,
                seed=seed,
            )
    output["bundle_did"] = bundle_did
    # Compatibility alias only.  The estimand is the difference between two
    # complete History bundles; it is never labelled a pure relevance effect.
    output["relevance_did"] = bundle_did
    output["estimand_scope"] = {
        "did_label": "relevant_history_bundle_minus_irrelevant_history_bundle",
        "pure_relevance_effect": False,
        "reason": (
            "The fallback stratum changes historical item and answer identity; "
            "even exact ordered-pair matching does not remove target repetition."
        ),
    }
    return output


def factorial_match_tier_sensitivity(
    contrast_rows: Sequence[Mapping[str, Any]],
    *,
    outcomes: Sequence[str] = PRIMARY_OUTCOMES,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    """Describe factorial effects separately for exact-pair and fallback rows.

    These small strata are descriptive and never qualification-gate-bearing.
    In the frozen primary cohort the exact ordered-pair stratum is expected to
    contain only about six cases.
    """

    rows = [dict(row) for row in contrast_rows]
    exact = [row for row in rows if row.get("history_ordered_pair_exact") is True]
    fallback = [
        row for row in rows if row.get("history_ordered_pair_exact") is False
    ]
    return {
        "exact_ordered_pair": {
            "n": len(exact),
            "descriptive_only": True,
            "gate_bearing": False,
            "effects": _factorial_effect_summaries_core(
                exact, outcomes=outcomes, iterations=iterations
            ),
        },
        "fallback": {
            "n": len(fallback),
            "descriptive_only": True,
            "gate_bearing": False,
            "effects": _factorial_effect_summaries_core(
                fallback, outcomes=outcomes, iterations=iterations
            ),
        },
    }


# Backwards-compatible descriptive name retained for callers written while the
# protocol was being frozen.  The explicit public name above is what analysis
# artifacts should use.
stratified_factorial_effect_summaries = factorial_match_tier_sensitivity


def factorial_effect_summaries(
    contrast_rows: Sequence[Mapping[str, Any]],
    *,
    outcomes: Sequence[str] = PRIMARY_OUTCOMES,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    output = _factorial_effect_summaries_core(
        contrast_rows, outcomes=outcomes, iterations=iterations
    )
    output["match_strata"] = factorial_match_tier_sensitivity(
        contrast_rows, outcomes=outcomes, iterations=iterations
    )
    return output


def leave_one_reused_donor_cluster_out(
    contrast_rows: Sequence[Mapping[str, Any]],
    *,
    outcomes: Sequence[str] = PRIMARY_OUTCOMES,
) -> dict[str, Any]:
    """Describe sensitivity to repeated use of the same frozen donor item.

    This is deliberately not a new inferential unit or a gate.  For each donor
    role whose item is reused by multiple recipients, it removes the complete
    recipient cluster attached to that donor and records the remaining paired
    mean for every available primary factorial contrast.  The min/max envelope
    is conditional on this frozen cohort and must not be called a confidence
    interval.
    """

    rows = [dict(row) for row in contrast_rows]
    if len({str(row.get("item_id")) for row in rows}) != len(rows):
        raise ValueError("donor-cluster sensitivity requires one row per recipient item")
    effect_keys = tuple(
        f"{family}_{effect}_{outcome}"
        for family in ("relevant", "irrelevant", "did")
        for effect in ("modality", "replay", "interaction", "history_vs_none")
        for outcome in outcomes
    )
    role_output: dict[str, Any] = {}
    for role in ("history_donor", "donor5", "donor6"):
        field = f"{role}_item_id"
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            donor = row.get(field)
            if donor is not None and str(donor) != "":
                counts[str(donor)] += 1
        reused = sorted(donor for donor, count in counts.items() if count > 1)
        estimates: dict[str, list[dict[str, Any]]] = {
            key: [] for key in effect_keys
        }
        omissions: list[dict[str, Any]] = []
        for donor in reused:
            kept = [row for row in rows if str(row.get(field)) != donor]
            omissions.append(
                {
                    "donor_item_id": donor,
                    "removed_recipient_n": counts[donor],
                    "remaining_recipient_n": len(kept),
                }
            )
            for key in effect_keys:
                finite = _finite_rows(kept, key)
                if finite:
                    estimates[key].append(
                        {
                            "donor_item_id": donor,
                            "mean": float(np.mean([float(row[key]) for row in finite])),
                            "n": len(finite),
                        }
                    )
        effects: dict[str, Any] = {}
        for key, values in estimates.items():
            means = [float(value["mean"]) for value in values]
            ns = [int(value["n"]) for value in values]
            effects[key] = {
                "leave_cluster_out_n": len(values),
                "mean_range": [min(means), max(means)] if means else [None, None],
                "analysis_n_range": [min(ns), max(ns)] if ns else [None, None],
                "estimates": values,
            }
        role_output[role] = {
            "donor_field": field,
            "observed_donor_n": len(counts),
            "reused_donor_cluster_n": len(reused),
            "reused_donor_item_ids": reused,
            "omissions": omissions,
            "effects": effects,
        }
    return {
        "n": len(rows),
        "descriptive_only": True,
        "gate_bearing": False,
        "conditional_on_frozen_cohort": True,
        "range_is_confidence_interval": False,
        "roles": role_output,
    }


def change_reliability(
    contrast_rows: Sequence[Mapping[str, Any]],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    seed = 500
    for prefix in (
        "relevant_modality",
        "irrelevant_modality",
        "did_modality",
        "relevant_replay",
        "irrelevant_replay",
        "did_replay",
    ):
        rows = _finite_rows(
            contrast_rows,
            f"{prefix}_M5",
            f"{prefix}_M6",
            f"{prefix}_B_D",
            f"{prefix}_B_M56",
        )
        seed += 3
        output[prefix] = {
            "M5_vs_M6": association_summary(
                rows,
                f"{prefix}_M5",
                f"{prefix}_M6",
                iterations=iterations,
                seed=seed,
            ),
            "M5_M6_icc": icc_consistency(
                [[row[f"{prefix}_M5"], row[f"{prefix}_M6"]] for row in rows]
            ),
            "D_vs_M56": association_summary(
                rows,
                f"{prefix}_B_D",
                f"{prefix}_B_M56",
                iterations=iterations,
                seed=seed + 1,
            ),
        }
    return output


def _correlation_difference(
    rows: Sequence[Mapping[str, Any]],
    common: str,
    preferred: str,
    comparison: str,
) -> float:
    left = _safe_spearman(rows, common, preferred)
    right = _safe_spearman(rows, common, comparison)
    return left - right if math.isfinite(left) and math.isfinite(right) else math.nan


def shift_alignment(
    contrast_rows: Sequence[Mapping[str, Any]],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    """Report correlations and direct specialization contrasts for each shift."""

    output: dict[str, Any] = {}
    seed = 700
    pairs = (
        ("B_D", "U_prediction"),
        ("B_M56", "U_prediction"),
        ("A_prediction", "V"),
        ("B_D", "A_prediction"),
        ("B_D", "V"),
        ("B_M56", "A_prediction"),
        ("B_M56", "V"),
        ("U_prediction", "A_prediction"),
        ("U_prediction", "V"),
    )
    for prefix in (
        "relevant_modality",
        "irrelevant_modality",
        "did_modality",
        "relevant_replay",
        "irrelevant_replay",
        "did_replay",
    ):
        seed += 20
        matrix = {
            f"{left}_vs_{right}": association_summary(
                contrast_rows,
                f"{prefix}_{left}",
                f"{prefix}_{right}",
                iterations=iterations,
                seed=seed + index,
            )
            for index, (left, right) in enumerate(pairs)
        }
        direct = {}
        for index, (name, common, preferred, comparison) in enumerate(
            (
                ("B_D_U_minus_A", "B_D", "U_prediction", "A_prediction"),
                ("B_M56_U_minus_A", "B_M56", "U_prediction", "A_prediction"),
                ("V_A_minus_U", "V", "A_prediction", "U_prediction"),
            )
        ):
            keys = [
                f"{prefix}_{common}",
                f"{prefix}_{preferred}",
                f"{prefix}_{comparison}",
            ]
            values = _finite_rows(contrast_rows, *keys)
            direct[name] = fixed_fold_item_bootstrap(
                values,
                lambda sample, c=keys[0], p=keys[1], a=keys[2]: _correlation_difference(
                    sample, c, p, a
                ),
                iterations=iterations,
                seed=seed + 10 + index,
            )
        output[prefix] = {"matrix": matrix, "direct_differences": direct}
    return output


def _lower_positive(summary: Mapping[str, Any], path: Sequence[str]) -> bool:
    value: Any = summary
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return False
        value = value[key]
    return bool(isinstance(value, Sequence) and value[0] is not None and float(value[0]) > 0)


def _positive_bootstrap_ci_gate(
    summary: Mapping[str, Any],
    *,
    expected_n: int,
    iterations: int,
    ci_key: str,
    bootstrap_key: str,
) -> bool:
    """Validate a positive CI only when the bootstrap itself is sufficiently valid."""

    bootstrap = summary.get(bootstrap_key)
    minimum_valid = int(math.ceil(0.95 * int(iterations)))
    return bool(
        summary.get("n") == int(expected_n)
        and _lower_positive(summary, (ci_key,))
        and isinstance(bootstrap, Mapping)
        and bootstrap.get("iterations") == int(iterations)
        and isinstance(bootstrap.get("valid"), int)
        and int(bootstrap["valid"]) >= minimum_valid
    )


def _strict_finite_number(value: Any) -> bool:
    """Return whether ``value`` is a finite numeric scalar, excluding booleans."""

    return not isinstance(value, (bool, np.bool_)) and _finite(value)


def _qualification_technical_audit(
    values: Sequence[Mapping[str, Any]], *, expected_n: int
) -> dict[str, Any]:
    """Audit the byte-frozen 40-row Phase-1 qualification contract.

    This audit intentionally does not relax the five-fold allocation when a
    caller passes another ``expected_n``.  Stage 09 is frozen at 40 rows and
    eight recipient items in every fold; accepting a different shape would
    silently create a new protocol after outcomes exist.
    """

    rows = [dict(row) for row in values]
    case_ids = [row.get("case_id") for row in rows]
    item_ids = [row.get("item_id") for row in rows]
    case_ids_present = all(value is not None and str(value) != "" for value in case_ids)
    item_ids_present = all(value is not None and str(value) != "" for value in item_ids)
    case_ids_unique = bool(
        case_ids_present and len({str(value) for value in case_ids}) == len(rows)
    )
    item_ids_unique = bool(
        item_ids_present and len({str(value) for value in item_ids}) == len(rows)
    )

    fold_counts = {fold: 0 for fold in range(5)}
    fold_values_valid = True
    for row in rows:
        try:
            raw_fold = row.get("fold")
            fold = int(raw_fold)
            if isinstance(raw_fold, (float, np.floating)) and float(raw_fold) != fold:
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            fold_values_valid = False
            continue
        if fold not in fold_counts:
            fold_values_valid = False
            continue
        fold_counts[fold] += 1

    numeric_fields = {
        field: bool(
            len(rows) == int(expected_n)
            and all(_strict_finite_number(row.get(field)) for row in rows)
        )
        for field in QUALIFICATION_NUMERIC_FIELDS
    }
    common_structural_fields = {
        **{
            field: bool(
                len(rows) == int(expected_n)
                and all(row.get(field) is True for row in rows)
            )
            for field in QUALIFICATION_COMMON_TRUE_FIELDS
        },
        **{
            field: bool(
                len(rows) == int(expected_n)
                and all(row.get(field) is False for row in rows)
            )
            for field in QUALIFICATION_COMMON_FALSE_FIELDS
        },
    }
    behavior_structural_fields = {
        field: bool(
            len(rows) == int(expected_n)
            and all(row.get(field) is True for row in rows)
        )
        for field in QUALIFICATION_BEHAVIOR_TRUE_FIELDS
    }
    report_structural_fields = {
        field: bool(
            len(rows) == int(expected_n)
            and all(row.get(field) is True for row in rows)
        )
        for field in QUALIFICATION_REPORT_TRUE_FIELDS
    }
    structural_fields = {
        **common_structural_fields,
        **behavior_structural_fields,
        **report_structural_fields,
    }
    cohort_checks = {
        "expected_n_is_frozen_40": int(expected_n) == 40,
        "completed_n_exact": len(rows) == int(expected_n) == 40,
        "case_id_present": case_ids_present,
        "case_id_unique": case_ids_unique,
        "item_id_present": item_ids_present,
        "item_id_unique": item_ids_unique,
        "fold_values_valid": fold_values_valid,
        "fold_counts_exact": bool(
            fold_values_valid and fold_counts == {fold: 8 for fold in range(5)}
        ),
    }
    behavior_checks = {
        "cohort_contract": all(cohort_checks.values()),
        "common_structural_fields": all(common_structural_fields.values()),
        "behavior_numeric_fields_finite": all(
            numeric_fields[field]
            for field in QUALIFICATION_BEHAVIOR_NUMERIC_FIELDS
        ),
        "behavior_structural_fields": all(behavior_structural_fields.values()),
    }
    report_checks = {
        "cohort_contract": all(cohort_checks.values()),
        "common_structural_fields": all(common_structural_fields.values()),
        "report_numeric_fields_finite": all(
            numeric_fields[field] for field in QUALIFICATION_REPORT_NUMERIC_FIELDS
        ),
        "report_structural_fields": all(report_structural_fields.values()),
    }
    checks = {
        **cohort_checks,
        "required_numeric_fields_finite": all(numeric_fields.values()),
        "required_structural_fields": all(structural_fields.values()),
        "behavior_readout_technical_complete": all(behavior_checks.values()),
        "report_formation_technical_complete": all(report_checks.values()),
    }
    return {
        "passed": bool(
            checks["behavior_readout_technical_complete"]
            and checks["report_formation_technical_complete"]
        ),
        "completed_n": len(rows),
        "expected_n": int(expected_n),
        "fold_counts": {str(fold): count for fold, count in fold_counts.items()},
        "numeric_fields": numeric_fields,
        "structural_fields": structural_fields,
        "common_structural_fields": common_structural_fields,
        "tracks": {
            "behavior_readout": {
                "passed": all(behavior_checks.values()),
                "checks": behavior_checks,
            },
            "report_formation": {
                "passed": all(report_checks.values()),
                "checks": report_checks,
            },
        },
        "checks": checks,
    }


def _qualification_metrics_n_audit(
    metrics: Mapping[str, Any], *, expected_n: int
) -> dict[str, Any]:
    """Require every gate-bearing metric to use all forty frozen rows."""

    expected = int(expected_n)
    u = metrics.get("U_transport")
    nuisance = metrics.get("U_nuisance")
    improvement = metrics.get("U_squared_error_improvement")
    behavior_checks = {
        "D_vs_M56_n": metrics.get("D_vs_M56", {}).get("n") == expected,
        "M5_vs_M6_n": metrics.get("M5_vs_M6", {}).get("n") == expected,
        "M5_M6_icc_n": metrics.get("M5_M6_icc_n") == expected,
        "U_transport_n": isinstance(u, Mapping) and u.get("n") == expected,
        "U_transport_association_n": bool(
            isinstance(u, Mapping)
            and isinstance(u.get("association"), Mapping)
            and u["association"].get("n") == expected
        ),
        "U_nuisance_n": isinstance(nuisance, Mapping)
        and nuisance.get("n") == expected,
        "U_nuisance_association_n": bool(
            isinstance(nuisance, Mapping)
            and isinstance(nuisance.get("association"), Mapping)
            and nuisance["association"].get("n") == expected
        ),
        "U_squared_error_improvement_n": isinstance(improvement, Mapping)
        and improvement.get("n") == expected,
    }
    report_checks = {
        "A_V_transport_n": metrics.get("A_V_transport", {}).get("n") == expected,
    }
    return {
        "behavior_readout": behavior_checks,
        "report_formation": report_checks,
        "behavior_readout_complete": all(behavior_checks.values()),
        "report_formation_complete": all(report_checks.values()),
        "all_complete": all((*behavior_checks.values(), *report_checks.values())),
    }


def qualification_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_n: int = 40,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    """Freeze independent B/U and A/V qualification tracks before History.

    ``passed`` remains the backwards-compatible all-four-layer decision.  The
    per-track authorization fields are authoritative for downstream work: a
    failed behavior/readout track never suppresses an otherwise qualified
    report-formation track.
    """

    values = [dict(row) for row in rows if row.get("status") == "completed"]
    technical = _qualification_technical_audit(values, expected_n=expected_n)

    # Invalid recipient identity or fold metadata cannot be bootstrapped
    # safely.  Return empty metrics instead of raising before the frozen gate
    # can record why qualification failed.
    identity_ready = bool(
        technical["checks"]["case_id_present"]
        and technical["checks"]["case_id_unique"]
        and technical["checks"]["item_id_present"]
        and technical["checks"]["item_id_unique"]
        and technical["checks"]["fold_values_valid"]
    )
    metric_values = values if identity_ready else []
    d_m = association_summary(
        metric_values, "B_D", "B_M56", iterations=iterations, seed=901
    )
    donor = association_summary(
        metric_values, "M5", "M6", iterations=iterations, seed=902
    )
    donor_rows = _finite_rows(metric_values, "M5", "M6")
    donor_icc = icc_consistency(
        [[row["M5"], row["M6"]] for row in donor_rows]
    )
    u = prediction_summary(
        metric_values,
        "B_target_shared",
        "U_prediction",
        iterations=iterations,
        seed=903,
    )
    a_v = association_summary(
        metric_values, "A_prediction", "V", iterations=iterations, seed=904
    )
    nuisance = None
    improvement = None
    if metric_values and all(
        _strict_finite_number(row.get("U_nuisance_prediction"))
        for row in metric_values
    ):
        nuisance = prediction_summary(
            metric_values,
            "B_target_shared",
            "U_nuisance_prediction",
            iterations=iterations,
            seed=905,
        )
        comparison = []
        for row in _finite_rows(
            metric_values,
            "B_target_shared",
            "U_prediction",
            "U_nuisance_prediction",
        ):
            target = float(row["B_target_shared"])
            comparison.append(
                {
                    **row,
                    "squared_error_improvement": (
                        (target - float(row["U_nuisance_prediction"])) ** 2
                        - (target - float(row["U_prediction"])) ** 2
                    ),
                }
            )
        improvement = paired_effect_summary(
            comparison,
            "squared_error_improvement",
            iterations=iterations,
            seed=906,
        )
    metrics = {
        "D_vs_M56": d_m,
        "M5_vs_M6": donor,
        "M5_M6_icc": donor_icc,
        "M5_M6_icc_n": len(donor_rows),
        "U_transport": u,
        "A_V_transport": a_v,
        "U_nuisance": nuisance,
        "U_squared_error_improvement": improvement,
    }
    metric_n_checks = _qualification_metrics_n_audit(metrics, expected_n=expected_n)
    behavior_technical_complete = bool(
        technical["tracks"]["behavior_readout"]["passed"]
        and metric_n_checks["behavior_readout_complete"]
    )
    report_technical_complete = bool(
        technical["tracks"]["report_formation"]["passed"]
        and metric_n_checks["report_formation_complete"]
    )
    metrics_n_complete = bool(metric_n_checks["all_complete"])
    technical_complete = bool(
        behavior_technical_complete and report_technical_complete
    )

    components = {
        "technical_complete": technical_complete,
        "behavior_technical_complete": behavior_technical_complete,
        "report_technical_complete": report_technical_complete,
        "D_vs_M56": _positive_bootstrap_ci_gate(
            d_m,
            expected_n=expected_n,
            iterations=iterations,
            ci_key="spearman_ci95",
            bootstrap_key="spearman_bootstrap",
        ),
        "M5_vs_M6": _positive_bootstrap_ci_gate(
            donor,
            expected_n=expected_n,
            iterations=iterations,
            ci_key="spearman_ci95",
            bootstrap_key="spearman_bootstrap",
        ),
        "M5_M6_icc": donor_icc is not None and donor_icc >= 0.60,
        "U_r2_positive": u["r2"] is not None and float(u["r2"]) > 0,
        "U_rank_transport": _positive_bootstrap_ci_gate(
            u.get("association", {}),
            expected_n=expected_n,
            iterations=iterations,
            ci_key="spearman_ci95",
            bootstrap_key="spearman_bootstrap",
        ),
        "U_four_positive_folds": int(u["positive_spearman_fold_count"]) >= 4,
        "A_V_rank_transport": _positive_bootstrap_ci_gate(
            a_v,
            expected_n=expected_n,
            iterations=iterations,
            ci_key="spearman_ci95",
            bootstrap_key="spearman_bootstrap",
        ),
        "frozen_nuisance_available": nuisance is not None,
        "U_beats_frozen_nuisance": bool(
            improvement is not None
            and _positive_bootstrap_ci_gate(
                improvement,
                expected_n=expected_n,
                iterations=iterations,
                ci_key="ci95",
                bootstrap_key="bootstrap",
            )
        ),
    }
    behavior_component_names = (
        "behavior_technical_complete",
        "D_vs_M56",
        "M5_vs_M6",
        "M5_M6_icc",
        "U_r2_positive",
        "U_rank_transport",
        "U_four_positive_folds",
        "frozen_nuisance_available",
        "U_beats_frozen_nuisance",
    )
    report_component_names = (
        "report_technical_complete",
        "A_V_rank_transport",
    )
    behavior_components = {
        name: bool(components[name]) for name in behavior_component_names
    }
    report_components = {
        name: bool(components[name]) for name in report_component_names
    }
    behavior_authorized = all(behavior_components.values())
    report_authorized = all(report_components.values())
    passed = bool(behavior_authorized and report_authorized)

    if passed:
        failure_action = "run_full_B_U_A_V_history_factorial"
    elif report_authorized:
        failure_action = "skip_B_U_history_factorial; run_A_V_report_formation"
    elif behavior_authorized:
        failure_action = "run_B_U_history_factorial; skip_A_V_report_formation"
    else:
        failure_action = "skip_B_U_history_factorial; skip_A_V_report_formation"

    return {
        "n": len(values),
        "expected_n": int(expected_n),
        "technical_audit": {
            **technical,
            "metrics_n_complete": metrics_n_complete,
            "metric_n_checks": metric_n_checks,
            "bootstrap_minimum_valid": int(math.ceil(0.95 * int(iterations))),
        },
        "metrics": metrics,
        "components": components,
        "tracks": {
            "behavior_readout": {
                "authorized": behavior_authorized,
                "components": behavior_components,
            },
            "report_formation": {
                "authorized": report_authorized,
                "components": report_components,
            },
        },
        "authorizations": {
            "behavior_readout_history": behavior_authorized,
            "report_formation_history": report_authorized,
            "full_four_layer": passed,
        },
        # Compatibility aliases for the draft runner and early local artifacts.
        # New code must use ``authorizations`` above.
        "authorization": {
            "behavior_readout": behavior_authorized,
            "report_formation": report_authorized,
            "full_four_layer": passed,
        },
        "behavior_readout_authorized": behavior_authorized,
        "report_formation_authorized": report_authorized,
        "passed": passed,
        "failure_action": failure_action,
    }
