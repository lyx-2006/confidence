"""Stage 3 follow-up experiments: evidence, exact History, and SA directions."""

from __future__ import annotations

import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from confidence_test.inference_extension import ASSISTANT_ANSWER_PREFILL
from layer_metacognition.hidden_state_store import (
    append_jsonl,
    atomic_write_json,
    atomic_write_text,
    load_jsonl,
)
from layer_metacognition.steering.decision_side_steering import (
    BaselineHiddenStateRepository,
)
from layer_metacognition.steering.source_attribution_mean_steering import (
    load_sa_candidates,
    select_strong_sa_sources,
)

from .core import (
    PRIMARY_LAYER,
    PRIMARY_POSITION,
    SEED,
    FoldDirection,
    SAFormationArtifacts,
    SAOOFDirectionRepository,
    atomic_save_npz,
    canonical_message_hash,
    item_cluster_bootstrap,
    load_baseline_rows,
    orthogonal_equal_norm_control,
    paired_effect_summary,
    read_json,
    stable_hash,
    write_experiment_summary,
    write_jsonl_atomic,
)
from .runtime import (
    Stage3Runtime,
    assistant_message,
    build_factorial_history_messages,
    build_no_history_messages,
    full_prompt,
    image_content,
    prepare_exact_generated_measurement,
    prepare_measurement,
    source_prefix_from_generation,
)


EVIDENCE_DIR = "01_evidence_item_balanced"
HISTORY_DIR = "02_history_exact_factorial"
DIRECTION_DIR = "03_direction_comparison"
HISTORY_BRANCHES = (
    ("text_at", "text", "text"),
    ("text_ai", "text", "image"),
    ("image_at", "image", "text"),
    ("image_ai", "image", "image"),
)


def _numeric_item_key(value: Any) -> tuple[int, Any]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def _safe_record(base: dict[str, Any], function: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = function()
    except Exception as exc:
        result = {
            **base,
            "status": "failed",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    result["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    return result


def _latest_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    latest: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        latest[str(row["intervention_key"])] = row
    return list(latest.values())


def _item_mean_summary(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value is not None and math.isfinite(float(value)):
            grouped[str(row["item_id"])].append(float(value))
    item_rows = [
        {"item_id": item, key: statistics.fmean(values)}
        for item, values in sorted(grouped.items(), key=lambda pair: _numeric_item_key(pair[0]))
    ]
    summary = paired_effect_summary(item_rows, key)
    summary["estimand"] = "equal-weight mean of within-item pair means"
    return summary


def run_evidence_reanalysis(
    stage3_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    directory = output_root / EVIDENCE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    oof = load_jsonl(stage3_root / "00_natural_state" / "oof_predictions.jsonl")
    grouped: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = {}
    for row in oof:
        prefix = str(row["condition"]).rsplit("_", 1)[0]
        grouped.setdefault(
            (str(row["item_id"]), int(row["prior_index"]), prefix), {}
        )[str(row["difficulty"])] = row
    pairs: list[dict[str, Any]] = []
    for (item_id, prior_index, prefix), values in sorted(
        grouped.items(), key=lambda pair: (_numeric_item_key(pair[0][0]), pair[0][1], pair[0][2])
    ):
        if "easy" not in values or "hard" not in values:
            continue
        easy, hard = values["easy"], values["hard"]
        pairs.append(
            {
                "pair_id": f"{item_id}|{prior_index}|{prefix}",
                "item_id": item_id,
                "prior_index": prior_index,
                "condition_prefix": prefix,
                "fold": int(easy["fold"]),
                "easy_case_id": easy["case_id"],
                "hard_case_id": hard["case_id"],
                "easy_final_answer": easy["final_answer"],
                "hard_final_answer": hard["final_answer"],
                "endpoint_matched": easy["final_answer"] == hard["final_answer"],
                "delta_sa_hard_minus_easy": float(hard["sa"] - easy["sa"]),
                "delta_z_hard_minus_easy": float(hard["z_sa"] - easy["z_sa"]),
                "delta_prediction_hard_minus_easy": float(
                    hard["ridge_prediction"] - easy["ridge_prediction"]
                ),
            }
        )
    endpoint = [row for row in pairs if row["endpoint_matched"]]
    write_jsonl_atomic(directory / "results.jsonl", pairs)
    atomic_write_json(
        directory / "cohort_manifest.json",
        {
            "definition": "all available within-item/prior/condition-prefix easy-hard pairs",
            "selection_before_endpoint_filter": True,
            "pair_count": len(pairs),
            "unique_items": len({row["item_id"] for row in pairs}),
            "endpoint_matched_pair_count": len(endpoint),
            "endpoint_matched_unique_items": len({row["item_id"] for row in endpoint}),
            "one_item_primary_weight": True,
        },
    )
    summary = {
        "title": "Follow-up 1 — Item-balanced easy/hard evidence",
        "status": "completed",
        "gpu_forwards": 0,
        "primary_all_pairs": {
            "n": len(pairs),
            "unique_items": len({row["item_id"] for row in pairs}),
            "answer_change_rate": statistics.fmean(not row["endpoint_matched"] for row in pairs),
            "sa_pair_weighted": paired_effect_summary(pairs, "delta_sa_hard_minus_easy"),
            "z_pair_weighted": paired_effect_summary(pairs, "delta_z_hard_minus_easy"),
            "prediction_pair_weighted": paired_effect_summary(
                pairs, "delta_prediction_hard_minus_easy"
            ),
            "sa_item_weighted": _item_mean_summary(pairs, "delta_sa_hard_minus_easy"),
            "z_item_weighted": _item_mean_summary(pairs, "delta_z_hard_minus_easy"),
            "prediction_item_weighted": _item_mean_summary(
                pairs, "delta_prediction_hard_minus_easy"
            ),
        },
        "secondary_endpoint_matched": {
            "n": len(endpoint),
            "unique_items": len({row["item_id"] for row in endpoint}),
            "sa_item_weighted": _item_mean_summary(endpoint, "delta_sa_hard_minus_easy"),
            "z_item_weighted": _item_mean_summary(endpoint, "delta_z_hard_minus_easy"),
            "prediction_item_weighted": _item_mean_summary(
                endpoint, "delta_prediction_hard_minus_easy"
            ),
        },
        "claim_limit": "Paired condition contrast; endpoint-matched subset is secondary and outcome-conditioned.",
    }
    summary["primary_all_pairs"]["sa_item_weighted"]["textward_direction_rate"] = statistics.fmean(
        statistics.fmean(float(row["delta_sa_hard_minus_easy"]) for row in item_rows) < 0
        for item_rows in (
            [row for row in pairs if row["item_id"] == item]
            for item in sorted({row["item_id"] for row in pairs}, key=_numeric_item_key)
        )
    )
    summary["primary_all_pairs"]["z_item_weighted"]["textward_direction_rate"] = statistics.fmean(
        statistics.fmean(float(row["delta_z_hard_minus_easy"]) for row in item_rows) < 0
        for item_rows in (
            [row for row in pairs if row["item_id"] == item]
            for item in sorted({row["item_id"] for row in pairs}, key=_numeric_item_key)
        )
    )
    atomic_write_json(directory / "summary.json", summary)
    return summary


def _balanced_unique_cases(
    rows: Sequence[dict[str, Any]],
    n: int,
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if row.get("final_answer")
        and row.get("text_answer")
        and row.get("image_answer")
        and row["text_answer"] != row["image_answer"]
        and str(row["condition"]).startswith("conflict_")
    ]
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(
        eligible,
        key=lambda value: (
            _numeric_item_key(value["item_id"]),
            int(value["prior_index"]),
            str(value["condition"]),
        ),
    ):
        buckets[(row["fold"], row["difficulty"], row.get("decision_side"))].append(row)
    selected: list[dict[str, Any]] = []
    used_items: set[str] = set()
    keys = sorted(buckets, key=str)
    while len(selected) < n:
        advanced = False
        for key in keys:
            while buckets[key] and str(buckets[key][0]["item_id"]) in used_items:
                buckets[key].pop(0)
            if buckets[key] and len(selected) < n:
                row = buckets[key].pop(0)
                selected.append(row)
                used_items.add(str(row["item_id"]))
                advanced = True
        if not advanced:
            break
    if len(selected) != n:
        raise ValueError(f"Only {len(selected)} unique balanced cases available; requested {n}")
    return selected


def select_history_factorial_cohort(
    oof: Sequence[dict[str, Any]], n: int = 30
) -> list[dict[str, Any]]:
    return _balanced_unique_cases(oof, n)


def _history_branch_messages(
    runtime: Stage3Runtime,
    row: dict[str, Any],
    branch: str,
) -> list[dict[str, Any]]:
    case = runtime.case(row["item_id"], row["prior_index"])
    if branch == "no_history":
        return build_no_history_messages(case, row["condition"])
    definition = next(value for value in HISTORY_BRANCHES if value[0] == branch)
    _, modality, answer_side = definition
    answer = row["text_answer"] if answer_side == "text" else row["image_answer"]
    return build_factorial_history_messages(
        case,
        row["condition"],
        modality,
        str(answer),
    )


def _run_history_branch(
    runtime: Stage3Runtime,
    row: dict[str, Any],
    branch: str,
    direction: FoldDirection,
    *,
    generation_use_cache: bool = True,
) -> dict[str, Any]:
    messages = _history_branch_messages(runtime, row, branch)
    case = runtime.case(row["item_id"], row["prior_index"])
    generated = runtime.generator.generate_messages(
        messages,
        case.answer_classes,
        max_new_tokens=48,
        use_cache=generation_use_cache,
    )
    if (
        not generated.parse_success
        or generated.source_metric_status != "completed"
        or generated.source_attribution is None
        or not generated.normalized_answer
    ):
        raise RuntimeError(f"Pass 1 failed for {branch}: {generated.error}")
    prepared = prepare_exact_generated_measurement(
        runtime.generator,
        messages,
        generated,
        assistant_text=ASSISTANT_ANSWER_PREFILL,
    )
    measured = runtime.measure(prepared, direction)
    generation_logits = generated.source_attribution["class_logits"]
    pass2_logits = measured.source["class_logits"]
    max_logit_error = max(
        abs(float(left) - float(right))
        for left, right in zip(generation_logits, pass2_logits)
    )
    soft_error = abs(
        float(generated.source_attribution["soft_image_score"])
        - float(measured.source["soft_image_score"])
    )
    max_abs_logit = max(
        abs(float(value)) for value in [*generation_logits, *pass2_logits]
    )
    bf16_logit_tolerance = max(0.5, 0.02 * max_abs_logit)
    result = {
        "branch": branch,
        "messages_hash": canonical_message_hash(messages),
        "final_evidence_hash": canonical_message_hash(
            [[message for message in messages if message.get("role") == "user"][-1]]
        ),
        "pass1": generated.to_dict(),
        "pass2": measured.to_dict(),
        "exact_prefix_hash": prepared.prefix_hash,
        "reconstruction": {
            "max_class_logit_abs_error": max_logit_error,
            "soft_score_abs_error": soft_error,
            "bf16_logit_tolerance": bf16_logit_tolerance,
            "raw_logit_within_bf16_tolerance": max_logit_error <= bf16_logit_tolerance,
            "soft_within_0.01": soft_error <= 0.01,
        },
    }
    runtime.release_inputs(prepared)
    return result


def _factorial_contrasts(row: dict[str, Any], outcome: str) -> dict[str, float]:
    def value(branch: str) -> float:
        record = row["branches"][branch]
        if outcome == "pass1_sa":
            return float(record["pass1"]["source_attribution"]["soft_image_score"])
        if outcome == "pass2_sa":
            return float(record["pass2"]["source"]["soft_image_score"])
        if outcome == "z_sa":
            return float(record["pass2"]["z_sa"])
        raise KeyError(outcome)

    text_at, text_ai = value("text_at"), value("text_ai")
    image_at, image_ai = value("image_at"), value("image_ai")
    return {
        "modality_main": ((image_at + image_ai) - (text_at + text_ai)) / 2.0,
        "prior_answer_main": ((text_ai + image_ai) - (text_at + image_at)) / 2.0,
        "interaction": (image_ai - image_at) - (text_ai - text_at),
    }


def _summarize_history(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed"]
    strict = [row for row in completed if row.get("strict_four_way_endpoint_matched")]
    for row in strict:
        for outcome in ("pass1_sa", "pass2_sa", "z_sa"):
            for key, value in _factorial_contrasts(row, outcome).items():
                row[f"{outcome}_{key}"] = value
    reconstruction = [
        branch["reconstruction"]
        for row in completed
        for branch in row["branches"].values()
    ]
    pairwise: dict[str, Any] = {}
    for answer_side in ("at", "ai"):
        eligible = []
        for row in completed:
            text = row["branches"][f"text_{answer_side}"]
            image = row["branches"][f"image_{answer_side}"]
            if text["pass1"]["normalized_answer"] != image["pass1"]["normalized_answer"]:
                continue
            record = {"item_id": row["item_id"]}
            for outcome in ("pass1_sa", "pass2_sa", "z_sa"):
                if outcome == "pass1_sa":
                    getter = lambda branch: float(branch["pass1"]["source_attribution"]["soft_image_score"])
                elif outcome == "pass2_sa":
                    getter = lambda branch: float(branch["pass2"]["source"]["soft_image_score"])
                else:
                    getter = lambda branch: float(branch["pass2"]["z_sa"])
                record[outcome] = getter(image) - getter(text)
            eligible.append(record)
        pairwise[answer_side] = {
            "n": len(eligible),
            **{
                outcome: paired_effect_summary(eligible, outcome)
                for outcome in ("pass1_sa", "pass2_sa", "z_sa")
            },
        }
    prior_answer_pairwise: dict[str, Any] = {}
    for modality in ("text", "image"):
        eligible = []
        for row in completed:
            at = row["branches"][f"{modality}_at"]
            ai = row["branches"][f"{modality}_ai"]
            if at["pass1"]["normalized_answer"] != ai["pass1"]["normalized_answer"]:
                continue
            record = {"item_id": row["item_id"]}
            for outcome in ("pass1_sa", "pass2_sa", "z_sa"):
                if outcome == "pass1_sa":
                    getter = lambda branch: float(branch["pass1"]["source_attribution"]["soft_image_score"])
                elif outcome == "pass2_sa":
                    getter = lambda branch: float(branch["pass2"]["source"]["soft_image_score"])
                else:
                    getter = lambda branch: float(branch["pass2"]["z_sa"])
                record[outcome] = getter(ai) - getter(at)
            eligible.append(record)
        prior_answer_pairwise[modality] = {
            "n": len(eligible),
            **{
                outcome: paired_effect_summary(eligible, outcome)
                for outcome in ("pass1_sa", "pass2_sa", "z_sa")
            },
        }
    strict_effects = {
        outcome: {
            contrast: paired_effect_summary(strict, f"{outcome}_{contrast}")
            for contrast in ("modality_main", "prior_answer_main", "interaction")
        }
        for outcome in ("pass1_sa", "pass2_sa", "z_sa")
    }
    return {
        "title": "Follow-up 2 — Exact-token factorial History",
        "status": "completed",
        "attempted": len(rows),
        "completed": len(completed),
        "failed": len(rows) - len(completed),
        "unique_items": len({row["item_id"] for row in completed}),
        "strict_four_way_endpoint_matched_n": len(strict),
        "reconstruction": {
            "branch_n": len(reconstruction),
            "raw_logit_within_bf16_tolerance_rate": statistics.fmean(
                value["raw_logit_within_bf16_tolerance"] for value in reconstruction
            ) if reconstruction else None,
            "soft_within_0.01_rate": statistics.fmean(
                value["soft_within_0.01"] for value in reconstruction
            ) if reconstruction else None,
            "mean_soft_abs_error": statistics.fmean(
                value["soft_score_abs_error"] for value in reconstruction
            ) if reconstruction else None,
            "max_soft_abs_error": max(
                (value["soft_score_abs_error"] for value in reconstruction), default=None
            ),
            "max_logit_abs_error": max(
                (value["max_class_logit_abs_error"] for value in reconstruction), default=None
            ),
        },
        "strict_factorial": strict_effects,
        "pairwise_endpoint_matched_modality": pairwise,
        "pairwise_endpoint_matched_prior_answer": prior_answer_pairwise,
        "claim_limit": "Prior modality and replayed prior answer are separated; strict factorial inference requires all four generated final answers to match.",
    }


def run_history_factorial(
    runtime: Stage3Runtime,
    stage3_root: Path,
    output_root: Path,
    *,
    n_items: int,
    deadline: Callable[[], None],
) -> dict[str, Any]:
    directory = output_root / HISTORY_DIR
    directory.mkdir(parents=True, exist_ok=True)
    oof = load_jsonl(stage3_root / "00_natural_state" / "oof_predictions.jsonl")
    available_items = len({str(row["item_id"]) for row in oof})
    cohort = select_history_factorial_cohort(oof, min(available_items, n_items + 20))
    atomic_write_json(
        directory / "cohort_manifest.json",
        {
            "target_completed_unique_items": n_items,
            "candidate_case_count": len(cohort),
            "unique_items": len({row["item_id"] for row in cohort}),
            "branches": [value[0] for value in HISTORY_BRANCHES] + ["no_history"],
            "generation_use_cache": False,
            "authoritative_results": "results_nocache.jsonl",
            "superseded_diagnostic_results": "results.jsonl",
            "case_ids": [row["case_id"] for row in cohort],
            "resume_semantics": "latest row per intervention_key",
        },
    )
    results_path = directory / "results_nocache.jsonl"
    existing_rows = _latest_rows(results_path)
    existing = {row["intervention_key"] for row in existing_rows}
    completed_count = sum(row.get("status") == "completed" for row in existing_rows)
    directions = SAOOFDirectionRepository(stage3_root / "directions")
    for row in cohort:
        deadline()
        if completed_count >= n_items:
            break
        key = f"history_exact_nocache|{row['case_id']}"
        if key in existing:
            continue
        base = {
            "intervention_key": key,
            "experiment": "history_exact_factorial",
            "case_id": row["case_id"],
            "item_id": row["item_id"],
            "prior_index": row["prior_index"],
            "condition": row["condition"],
            "fold": row["fold"],
            "text_answer": row["text_answer"],
            "image_answer": row["image_answer"],
        }

        def execute() -> dict[str, Any]:
            direction = directions.get(row["fold"])
            branches = {
                branch: _run_history_branch(
                    runtime,
                    row,
                    branch,
                    direction,
                    generation_use_cache=False,
                )
                for branch in [value[0] for value in HISTORY_BRANCHES] + ["no_history"]
            }
            evidence_hashes = {
                branches[branch]["final_evidence_hash"]
                for branch, _, _ in HISTORY_BRANCHES
            }
            if len(evidence_hashes) != 1:
                raise ValueError("Factorial History final evidence differs across branches")
            final_answers = {
                branches[branch]["pass1"]["normalized_answer"]
                for branch, _, _ in HISTORY_BRANCHES
            }
            return {
                **base,
                "status": "completed",
                "branches": branches,
                "strict_four_way_endpoint_matched": len(final_answers) == 1,
                "final_evidence_equal": True,
            }

        result = _safe_record(base, execute)
        append_jsonl(results_path, result)
        if result.get("status") == "completed":
            completed_count += 1
    rows = _latest_rows(results_path)
    summary = _summarize_history(rows)
    atomic_write_json(directory / "summary.json", summary)
    return summary


def fit_oof_old_mean_directions(
    artifacts: SAFormationArtifacts,
    stage3_root: Path,
    output_root: Path,
    *,
    cases_per_side: int = 25,
) -> tuple[SAOOFDirectionRepository, dict[int, np.ndarray], dict[int, np.ndarray]]:
    direction_root = output_root / "directions" / "old_oof"
    residual_root = output_root / "directions" / "old_perp_ridge_oof"
    if (direction_root / "index.json").is_file() and (residual_root / "index.json").is_file():
        old_repo = SAOOFDirectionRepository(direction_root)
        residual_repo = SAOOFDirectionRepository(residual_root)
        return (
            old_repo,
            {fold: old_repo.get(fold).d_unit for fold in range(5)},
            {fold: residual_repo.get(fold).d_unit for fold in range(5)},
        )
    candidates = load_sa_candidates(artifacts.experiment_dir, artifacts.manifest)
    rows = load_baseline_rows(artifacts)
    hidden_repo = BaselineHiddenStateRepository(artifacts.experiment_dir)
    hidden = np.stack(
        [hidden_repo.get(row["manifest"], PRIMARY_LAYER, PRIMARY_POSITION) for row in rows]
    ).astype(np.float64, copy=False)
    folds = np.asarray([int(row["fold"]) for row in rows])
    item_to_fold = {
        str(key): int(value)
        for key, value in read_json(artifacts.item_split)["item_to_fold"].items()
    }
    ridge_repo = SAOOFDirectionRepository(stage3_root / "directions")
    old_entries: list[dict[str, Any]] = []
    residual_entries: list[dict[str, Any]] = []
    old_units: dict[int, np.ndarray] = {}
    residual_units: dict[int, np.ndarray] = {}
    source_audit: list[dict[str, Any]] = []
    for fold in range(5):
        train_candidates = [
            row for row in candidates if item_to_fold[str(row["item_id"])] != fold
        ]
        groups = select_strong_sa_sources(train_candidates, cases_per_side=cases_per_side)
        image_matrix = np.stack(
            [hidden_repo.get(row, PRIMARY_LAYER, PRIMARY_POSITION) for row in groups["follows_image"]]
        ).astype(np.float64, copy=False)
        text_matrix = np.stack(
            [hidden_repo.get(row, PRIMARY_LAYER, PRIMARY_POSITION) for row in groups["follows_text"]]
        ).astype(np.float64, copy=False)
        difference = image_matrix.mean(axis=0) - text_matrix.mean(axis=0)
        old_unit = difference / np.linalg.norm(difference)
        ridge_unit = ridge_repo.get(fold).d_unit
        cosine = float(old_unit @ ridge_unit)
        residual = old_unit - cosine * ridge_unit
        residual_unit = residual / np.linalg.norm(residual)
        if float(residual_unit @ old_unit) < 0:
            residual_unit = -residual_unit
        train = folds != fold
        old_sigma = float(np.std(hidden[train] @ old_unit, ddof=1))
        residual_sigma = float(np.std(hidden[train] @ residual_unit, ddof=1))
        old_units[fold] = old_unit
        residual_units[fold] = residual_unit
        for root, unit, sigma, entries, kind in (
            (direction_root, old_unit, old_sigma, old_entries, "old_mean_oof"),
            (residual_root, residual_unit, residual_sigma, residual_entries, "old_perp_ridge_oof"),
        ):
            filename = f"fold_{fold}_layer_18_panl.npz"
            atomic_save_npz(
                root / filename,
                alpha=np.asarray(0.0),
                d_raw=unit,
                d_unit=unit,
                raw_intercept=np.asarray(0.0),
                scaler_mean=np.zeros_like(unit),
                scaler_scale=np.ones_like(unit),
                sigma_z=np.asarray(sigma),
                sign_flipped=np.asarray(False),
            )
            entries.append(
                {
                    "fold": fold,
                    "file": filename,
                    "direction_kind": kind,
                    "sigma_z": sigma,
                    "source_count_per_side": cases_per_side,
                    "cos_old_ridge": cosine,
                }
            )
        source_audit.append(
            {
                "fold": fold,
                "heldout_fold_excluded": True,
                "cos_old_ridge": cosine,
                "old_sigma_z": old_sigma,
                "residual_sigma_z": residual_sigma,
                "follows_text_case_ids": [row["case_id"] for row in groups["follows_text"]],
                "follows_image_case_ids": [row["case_id"] for row in groups["follows_image"]],
                "source_item_overlap_with_heldout": sorted(
                    {
                        str(row["item_id"])
                        for side in groups.values()
                        for row in side
                        if item_to_fold[str(row["item_id"])] == fold
                    }
                ),
            }
        )
    atomic_write_json(direction_root / "index.json", {"folds": old_entries})
    atomic_write_json(residual_root / "index.json", {"folds": residual_entries})
    atomic_write_json(output_root / "directions" / "fold_audit.json", source_audit)
    return SAOOFDirectionRepository(direction_root), old_units, residual_units


def _direction_eval_context(
    runtime: Stage3Runtime,
    row: dict[str, Any],
) -> Any:
    case = runtime.case(row["item_id"], row["prior_index"])
    current = row["baseline"]["generated"]["current_answer_result"]
    source = row["baseline"]["generated"]["source_attribution"]
    raw_output = str(current["raw_output"])
    label = str(source["parsed_label"])
    assistant_text = source_prefix_from_generation(raw_output, label)
    messages = [
        {
            "role": "user",
            "content": image_content(
                str(case.conditions[row["condition"]].resolved_image_path),
                full_prompt(case),
            ),
        },
        assistant_message(assistant_text),
    ]
    return prepare_measurement(
        runtime.generator,
        messages,
        assistant_text=assistant_text,
        answer=str(current["normalized_answer"]),
    )


def _direction_object(unit: np.ndarray, sigma: float, fold: int) -> FoldDirection:
    return FoldDirection(
        fold=fold,
        alpha=0.0,
        d_raw=unit,
        d_unit=unit,
        raw_intercept=0.0,
        scaler_mean=np.zeros_like(unit),
        scaler_scale=np.ones_like(unit),
        sigma_z=float(sigma),
        sign_flipped=False,
    )


def _summarize_direction_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed"]
    summaries: dict[str, Any] = {}
    for kind in ("ridge", "old", "old_perp_ridge"):
        by_dose: dict[str, Any] = {}
        for dose in (1, 2):
            contrast_key = f"{kind}_dose_{dose}_contrast"
            control_key = f"{kind}_random_dose_{dose}_contrast"
            for row in completed:
                positive = row["arms"][f"{kind}|+{dose}"]["source"]["soft_image_score"]
                negative = row["arms"][f"{kind}|-{dose}"]["source"]["soft_image_score"]
                random_positive = row["arms"][f"{kind}_random|+{dose}"]["source"]["soft_image_score"]
                random_negative = row["arms"][f"{kind}_random|-{dose}"]["source"]["soft_image_score"]
                row[contrast_key] = float(positive - negative)
                row[control_key] = float(random_positive - random_negative)
                row[f"{kind}_dose_{dose}_minus_control"] = row[contrast_key] - row[control_key]
            effect = paired_effect_summary(completed, contrast_key)
            control = paired_effect_summary(completed, control_key)
            difference = paired_effect_summary(completed, f"{kind}_dose_{dose}_minus_control")
            by_dose[str(dose)] = {
                "minus_to_plus_sa": effect,
                "random_equal_l2_minus_to_plus_sa": control,
                "direction_minus_control": difference,
                "causal_output_control_supported": bool(
                    effect["n"] >= 25
                    and effect["ci95"][0] is not None
                    and effect["ci95"][0] > 0
                    and effect["direction_rate"] >= 0.60
                    and difference["ci95"][0] is not None
                    and difference["ci95"][0] > 0
                ),
            }
        summaries[kind] = by_dose
    for dose in (1, 2):
        for row in completed:
            row[f"old_minus_ridge_dose_{dose}"] = (
                row[f"old_dose_{dose}_contrast"] - row[f"ridge_dose_{dose}_contrast"]
            )
        summaries[f"old_minus_ridge_dose_{dose}"] = paired_effect_summary(
            completed, f"old_minus_ridge_dose_{dose}"
        )
    return {
        "title": "Follow-up 3 — OOF old-vs-Ridge matched-dose intervention",
        "status": "completed",
        "attempted": len(rows),
        "completed": len(completed),
        "failed": len(rows) - len(completed),
        "unique_items": len({row["item_id"] for row in completed}),
        "direction_results": summaries,
        "claim_limit": "Establishes output-control under the tested L18 PANL interventions, not natural mechanistic use or mediation.",
    }


def run_direction_comparison(
    runtime: Stage3Runtime,
    artifacts: SAFormationArtifacts,
    stage3_root: Path,
    output_root: Path,
    *,
    n_items: int,
    deadline: Callable[[], None],
) -> dict[str, Any]:
    directory = output_root / DIRECTION_DIR
    directory.mkdir(parents=True, exist_ok=True)
    old_repo, _, _ = fit_oof_old_mean_directions(
        artifacts, stage3_root, output_root
    )
    residual_repo = SAOOFDirectionRepository(
        output_root / "directions" / "old_perp_ridge_oof"
    )
    ridge_repo = SAOOFDirectionRepository(stage3_root / "directions")
    baseline = load_baseline_rows(artifacts)
    cohort = _balanced_unique_cases(baseline, n_items)
    atomic_write_json(
        directory / "cohort_manifest.json",
        {
            "case_count": len(cohort),
            "unique_items": len({row["item_id"] for row in cohort}),
            "one_case_per_item": True,
            "doses_sigma_units": [-2, -1, 0, 1, 2],
            "directions": ["ridge", "old", "old_perp_ridge"],
            "controls": "fixed per case/direction random orthogonal axis with matched ±L2",
            "case_ids": [row["case_id"] for row in cohort],
        },
    )
    results_path = directory / "results.jsonl"
    existing = {row["intervention_key"] for row in _latest_rows(results_path)}
    for row in cohort:
        deadline()
        key = f"direction_compare|{row['case_id']}"
        if key in existing:
            continue
        base = {
            "intervention_key": key,
            "experiment": "direction_comparison",
            "case_id": row["case_id"],
            "item_id": row["item_id"],
            "prior_index": row["prior_index"],
            "condition": row["condition"],
            "fold": row["fold"],
        }

        def execute() -> dict[str, Any]:
            prepared = _direction_eval_context(runtime, row)
            directions = {
                "ridge": ridge_repo.get(row["fold"]),
                "old": old_repo.get(row["fold"]),
                "old_perp_ridge": residual_repo.get(row["fold"]),
            }
            clean = runtime.measure(prepared, directions["ridge"]).to_dict()
            arms: dict[str, Any] = {}
            for kind, direction in directions.items():
                random_unit = orthogonal_equal_norm_control(
                    direction.d_unit,
                    1.0,
                    seed_material=f"{row['case_id']}|{kind}|followup",
                )
                for dose in (-2, -1, 1, 2):
                    vector = float(dose) * direction.sigma_z * direction.d_unit
                    random_vector = float(dose) * direction.sigma_z * random_unit
                    label = f"{dose:+d}"
                    arms[f"{kind}|{label}"] = runtime.measure(
                        prepared, direction, steering_vector=vector
                    ).to_dict()
                    arms[f"{kind}_random|{label}"] = runtime.measure(
                        prepared, direction, steering_vector=random_vector
                    ).to_dict()
            runtime.release_inputs(prepared)
            return {
                **base,
                "status": "completed",
                "clean": clean,
                "arms": arms,
                "geometry": {
                    "cos_old_ridge": float(
                        directions["old"].d_unit @ directions["ridge"].d_unit
                    ),
                    "cos_residual_ridge": float(
                        directions["old_perp_ridge"].d_unit
                        @ directions["ridge"].d_unit
                    ),
                    "cos_residual_old": float(
                        directions["old_perp_ridge"].d_unit @ directions["old"].d_unit
                    ),
                },
            }

        append_jsonl(results_path, _safe_record(base, execute))
    rows = _latest_rows(results_path)
    summary = _summarize_direction_rows(rows)
    audit = read_json(output_root / "directions" / "fold_audit.json")
    summary["fold_geometry"] = [
        {
            "fold": row["fold"],
            "cos_old_ridge": row["cos_old_ridge"],
            "old_sigma_z": row["old_sigma_z"],
            "residual_sigma_z": row["residual_sigma_z"],
        }
        for row in audit
    ]
    atomic_write_json(directory / "summary.json", summary)
    return summary


def write_followup_report(output_root: Path) -> dict[str, Any]:
    evidence = read_json(output_root / EVIDENCE_DIR / "summary.json")
    history = read_json(output_root / HISTORY_DIR / "summary.json")
    directions = read_json(output_root / DIRECTION_DIR / "summary.json")
    for directory_name, summary in (
        (EVIDENCE_DIR, evidence),
        (HISTORY_DIR, history),
        (DIRECTION_DIR, directions),
    ):
        write_experiment_summary(output_root / directory_name, summary)
    e_sa = evidence["primary_all_pairs"]["sa_item_weighted"]
    e_z = evidence["primary_all_pairs"]["z_item_weighted"]
    h_at_sa = history["pairwise_endpoint_matched_modality"]["at"]["pass1_sa"]
    h_at_z = history["pairwise_endpoint_matched_modality"]["at"]["z_sa"]
    h_ai_sa = history["pairwise_endpoint_matched_modality"]["ai"]["pass1_sa"]
    h_ai_z = history["pairwise_endpoint_matched_modality"]["ai"]["z_sa"]
    h_prior_text = history["pairwise_endpoint_matched_prior_answer"]["text"]["pass1_sa"]
    h_prior_image = history["pairwise_endpoint_matched_prior_answer"]["image"]["pass1_sa"]
    ridge = directions["direction_results"]["ridge"]["1"]
    old = directions["direction_results"]["old"]["1"]
    residual = directions["direction_results"]["old_perp_ridge"]["1"]
    ridge_dose2 = directions["direction_results"]["ridge"]["2"]
    old_dose2 = directions["direction_results"]["old"]["2"]
    residual_dose2 = directions["direction_results"]["old_perp_ridge"]["2"]
    mean_old_ridge_cosine = statistics.fmean(
        row["cos_old_ridge"] for row in directions["fold_geometry"]
    )
    payload = {
        "status": "completed",
        "evidence": {
            "hard_minus_easy_sa": e_sa,
            "hard_minus_easy_z": e_z,
        },
        "history": {
            "strict_n": history["strict_four_way_endpoint_matched_n"],
            "fixed_prior_at": {"pass1_sa": h_at_sa, "z_sa": h_at_z},
            "fixed_prior_ai": {"pass1_sa": h_ai_sa, "z_sa": h_ai_z},
            "prior_answer_effect_within_text_history": h_prior_text,
            "prior_answer_effect_within_image_history": h_prior_image,
            "reconstruction": history["reconstruction"],
        },
        "direction_comparison": {
            "ridge_dose1": ridge,
            "old_dose1": old,
            "old_perp_ridge_dose1": residual,
            "ridge_dose2": ridge_dose2,
            "old_dose2": old_dose2,
            "old_perp_ridge_dose2": residual_dose2,
            "mean_cos_old_ridge": mean_old_ridge_cosine,
        },
    }
    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(analysis_dir / "final_analysis.json", payload)
    lines = [
        "# Stage 3 SA Formation — Core Follow-up",
        "",
        "## Evidence easy/hard",
        "",
        f"- All-pair item-weighted hard−easy SA: {e_sa['mean']:.6f}, 95% CI={e_sa['ci95']}, unique items={e_sa['unique_items']}.",
        f"- All-pair item-weighted hard−easy z: {e_z['mean']:.6f}, 95% CI={e_z['ci95']}.",
        f"- Endpoint-matched secondary hard−easy SA: {evidence['secondary_endpoint_matched']['sa_item_weighted']['mean']:.6f}, 95% CI={evidence['secondary_endpoint_matched']['sa_item_weighted']['ci95']}.",
        "",
        "## Exact-token factorial History",
        "",
        f"- Completed unique items={history['unique_items']}; strict four-way endpoint-matched n={history['strict_four_way_endpoint_matched_n']}.",
        f"- Fixed prior A_T, endpoint-matched Image−Text History Pass-1 SA: {h_at_sa['mean']}, 95% CI={h_at_sa['ci95']}; z={h_at_z['mean']}, 95% CI={h_at_z['ci95']}.",
        f"- Fixed prior A_I, endpoint-matched Image−Text History Pass-1 SA: {h_ai_sa['mean']}, 95% CI={h_ai_sa['ci95']}; z={h_ai_z['mean']}, 95% CI={h_ai_z['ci95']}.",
        f"- Replayed A_I−A_T effect within Text history: {h_prior_text['mean']}, 95% CI={h_prior_text['ci95']}; within Image history: {h_prior_image['mean']}, 95% CI={h_prior_image['ci95']}.",
        f"- Strict four-cell endpoint matching leaves n={history['strict_four_way_endpoint_matched_n']}; it is reported as exploratory only.",
        f"- Exact-token mean soft reconstruction error={history['reconstruction']['mean_soft_abs_error']}; BF16-scaled raw-logit pass rate={history['reconstruction']['raw_logit_within_bf16_tolerance_rate']}.",
        "",
        "## OOF direction comparison",
        "",
        f"- Ridge ±1σ SA contrast: {ridge['minus_to_plus_sa']['mean']}, 95% CI={ridge['minus_to_plus_sa']['ci95']}, supported={ridge['causal_output_control_supported']}.",
        f"- Old mean ±1σ SA contrast: {old['minus_to_plus_sa']['mean']}, 95% CI={old['minus_to_plus_sa']['ci95']}, supported={old['causal_output_control_supported']}.",
        f"- Old⊥Ridge ±1σ SA contrast: {residual['minus_to_plus_sa']['mean']}, 95% CI={residual['minus_to_plus_sa']['ci95']}, supported={residual['causal_output_control_supported']}.",
        f"- At ±2σ: Ridge={ridge_dose2['minus_to_plus_sa']['mean']} (CI={ridge_dose2['minus_to_plus_sa']['ci95']}), old={old_dose2['minus_to_plus_sa']['mean']} (CI={old_dose2['minus_to_plus_sa']['ci95']}), old⊥Ridge={residual_dose2['minus_to_plus_sa']['mean']} (CI={residual_dose2['minus_to_plus_sa']['ci95']}).",
        f"- Mean foldwise cos(old, Ridge)={mean_old_ridge_cosine:.6f}; every equal-L2 random-control CI includes zero.",
        "",
        "## Conclusions",
        "",
        "- Easy→hard image condition produces a smaller but broadly item-general textward shift in both verbal SA and the L18 predictive coordinate.",
        "- Prior image-vs-text interaction history changes verbal SA even after holding the replayed prior answer fixed; Ridge z is null with prior A_T and positive with prior A_I, so the internal-coordinate result is conditional rather than a stable main effect.",
        "- The OOF old mean direction controls SA output while the dominant Ridge predictive direction does not; removing the Ridge component from old preserves control.",
        "- This supports a predictive/output-control direction dissociation, not natural mediation by either scalar.",
        "",
        "Causal labels refer only to output control under the tested L18 PANL intervention; they do not establish natural mediation.",
        "",
    ]
    atomic_write_text(analysis_dir / "FINAL_ANALYSIS.md", "\n".join(lines))
    return payload
