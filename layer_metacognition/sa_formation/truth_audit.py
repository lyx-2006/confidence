"""Behavior-grounded audit of verbal Source Attribution.

This follow-up separates three questions which earlier experiments left mixed:

* whether verbal SA tracks counterfactual source support beyond final-answer side;
* whether History modality and the replayed History answer have distinct effects;
* whether coarse output-protocol failures reflect granularity or token grammar.

All source artifacts are read-only.  New outputs live under
``stage3_sa_truth_audit``.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from confidence_test.answer_metrics import normalize_answer
from confidence_test.dataset_utils import load_evaluation_cases
from layer_metacognition.hidden_state_store import append_jsonl, atomic_write_json, load_jsonl
from layer_metacognition.steering.decision_side_steering import (
    BaselineHiddenStateRepository,
)
from layer_metacognition.model_adapter import run_logits_forward

from .core import (
    RIDGE_ALPHAS,
    SAFormationArtifacts,
    SAOOFDirectionRepository,
    atomic_save_npz,
    canonical_message_hash,
    item_cluster_bootstrap,
    load_baseline_rows,
    paired_effect_summary,
    ridge_raw_space,
    write_csv_atomic,
    write_experiment_summary,
    write_jsonl_atomic,
)
from .experiments import _load_unimodal
from .followup import _balanced_unique_cases
from .runtime import Stage3Runtime
from .second_order import (
    ProtocolAnalyzer,
    ProtocolSpec,
    _latest_rows,
    _protocol_context,
    _safe_record,
    build_answer_history_messages,
    generate_answer_messages,
    protocol_prompt,
    protocol_specs,
)
from .runtime import (
    ASSISTANT_ANSWER_PREFILL,
    assistant_message,
    build_factorial_history_messages,
    image_content,
    prepare_measurement,
)
from confidence_test.source_attribution_variants import get_source_prompt_variant


BEHAVIOR_DIR = "01_counterfactual_source_use"
MATCHED_GROUNDING_DIR = "02_matched_prompt_source_perturbation"
FACTORIAL_DIR = "03_history_factorial_reanalysis"
ANSWER_PROTOCOL_DIR = "04_answer_only_protocol_robustness"
GRANULARITY_DIR = "05_protocol_granularity_bridge"
GATE_DIR = "06_truth_gate"
TRACING_DIR = "07_grounded_blockwise_tracing"
SUBSPACE_DIR = "08_grounded_low_rank_subspace"


def _log_probability(value: float, epsilon: float = 1e-8) -> float:
    return math.log(max(epsilon, min(1.0, float(value))))


def counterfactual_source_use(
    p_full_answer: float,
    p_text_answer: float,
    p_image_answer: float,
) -> dict[str, float]:
    """Return source-removal effects for one fixed, naturally selected answer.

    ``remove_image_drop`` compares full evidence with text-only evidence;
    ``remove_text_drop`` compares full evidence with image-only evidence.  Their
    difference cancels the full-context term and is positive when image-only
    evidence supports the selected answer more strongly than text-only evidence.
    """

    log_full = _log_probability(p_full_answer)
    log_text = _log_probability(p_text_answer)
    log_image = _log_probability(p_image_answer)
    remove_image_drop = log_full - log_text
    remove_text_drop = log_full - log_image
    relative_log_support = remove_image_drop - remove_text_drop
    bounded = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, relative_log_support))))
    return {
        "p_full_fixed_answer": float(p_full_answer),
        "p_text_fixed_answer": float(p_text_answer),
        "p_image_fixed_answer": float(p_image_answer),
        "remove_image_drop_logp": remove_image_drop,
        "remove_text_drop_logp": remove_text_drop,
        "relative_image_use_log": relative_log_support,
        "behavior_imageward_score": bounded,
    }


def _prior_strength(case: Any) -> float:
    value = str(case.prior_bin or "")
    try:
        left, right = value.split("-", 1)
        return abs((float(left) + float(right)) / 2.0 - 0.5)
    except Exception:
        return 0.0


def build_counterfactual_rows(
    artifacts: SAFormationArtifacts,
    runtime: Stage3Runtime | None = None,
) -> list[dict[str, Any]]:
    baseline = load_baseline_rows(artifacts)
    cases, _ = load_evaluation_cases(artifacts.dataset)
    case_by_key = {(str(case.item_id), int(case.prior_index)): case for case in cases}
    text_map, image_map = _load_unimodal(artifacts)
    rows: list[dict[str, Any]] = []
    for row in baseline:
        text_answer = normalize_answer(row.get("text_answer"))
        image_answer = normalize_answer(row.get("image_answer"))
        final_answer = normalize_answer(row.get("final_answer"))
        if (
            not text_answer
            or not image_answer
            or text_answer == image_answer
            or final_answer not in {text_answer, image_answer}
        ):
            continue
        text_result = text_map[(row["item_id"], row["prior_index"])][
            "generation_result"
        ]
        image_result = image_map[(row["item_id"], row["condition"])][
            "generation_result"
        ]
        full_result = row["baseline"]["generated"]["current_answer_result"]
        text_probabilities = text_result["answer_class_probabilities"]
        image_probabilities = image_result["answer_class_probabilities"]
        full_probabilities = full_result["answer_class_probabilities"]
        if not all(
            final_answer in probabilities
            for probabilities in (text_probabilities, image_probabilities, full_probabilities)
        ):
            continue
        case = (
            runtime.case(row["item_id"], row["prior_index"])
            if runtime
            else case_by_key[(str(row["item_id"]), int(row["prior_index"]))]
        )
        use = counterfactual_source_use(
            float(full_probabilities[final_answer]),
            float(text_probabilities[final_answer]),
            float(image_probabilities[final_answer]),
        )
        rows.append(
            {
                "intervention_key": f"counterfactual_use|{row['case_id']}",
                "experiment": "counterfactual_source_use",
                "status": "completed",
                "case_id": row["case_id"],
                "item_id": row["item_id"],
                "prior_index": row["prior_index"],
                "condition": row["condition"],
                "difficulty": row["difficulty"],
                "fold": int(row["fold"]),
                "text_answer": text_answer,
                "image_answer": image_answer,
                "final_answer": final_answer,
                "final_image": int(final_answer == image_answer),
                "decision_side": (
                    "follows_image" if final_answer == image_answer else "follows_text"
                ),
                "prior_strength": _prior_strength(case),
                "verbal_sa": float(row["sa"]),
                **use,
                "manifest": row["manifest"],
            }
        )
    if not rows:
        raise ValueError("No conflict endpoint cases support counterfactual source-use scoring")
    return rows


def _association(
    rows: Sequence[dict[str, Any]], x_key: str, y_key: str
) -> dict[str, Any]:
    valid = [
        row
        for row in rows
        if row.get(x_key) is not None
        and row.get(y_key) is not None
        and np.isfinite(row[x_key])
        and np.isfinite(row[y_key])
    ]
    if len(valid) < 3:
        return {
            "n": len(valid),
            "pearson": None,
            "spearman": None,
            "spearman_item_bootstrap": None,
        }
    x = np.asarray([row[x_key] for row in valid], dtype=np.float64)
    y = np.asarray([row[y_key] for row in valid], dtype=np.float64)
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return {
            "n": len(valid),
            "pearson": None,
            "spearman": None,
            "spearman_item_bootstrap": {
                "estimate": None,
                "ci95": [None, None],
                "iterations": 1000,
                "valid": 0,
            },
        }
    bootstrap = item_cluster_bootstrap(
        valid,
        lambda sample: spearmanr(
            [row[x_key] for row in sample], [row[y_key] for row in sample]
        ).statistic,
    )
    return {
        "n": len(valid),
        "unique_items": len({str(row["item_id"]) for row in valid}),
        "pearson": float(pearsonr(x, y).statistic),
        "spearman": float(spearmanr(x, y).statistic),
        "spearman_item_bootstrap": bootstrap,
    }


def _nuisance_matrix(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            [
                1.0,
                float(row["final_image"]),
                float(row["difficulty"] == "hard"),
                float(row["prior_strength"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )


def fit_counterfactual_oof_direction(
    artifacts: SAFormationArtifacts,
    rows: list[dict[str, Any]],
    directory: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repository = BaselineHiddenStateRepository(artifacts.experiment_dir)
    hidden = np.stack(
        [repository.get(row["manifest"], 18, "panl") for row in rows], axis=0
    )
    raw_target = np.asarray(
        [row["relative_image_use_log"] for row in rows], dtype=np.float64
    )
    verbal = np.asarray([row["verbal_sa"] for row in rows], dtype=np.float64)
    folds = np.asarray([row["fold"] for row in rows], dtype=np.int64)
    nuisance = _nuisance_matrix(rows)
    target_residual = np.full(len(rows), np.nan)
    verbal_residual = np.full(len(rows), np.nan)
    predictions = np.full(len(rows), np.nan)
    coordinates = np.full(len(rows), np.nan)
    direction_dir = directory / "directions"
    entries: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for fold in sorted(set(folds.tolist())):
        train = folds != fold
        test = folds == fold
        train_items = {str(rows[index]["item_id"]) for index in np.flatnonzero(train)}
        test_items = {str(rows[index]["item_id"]) for index in np.flatnonzero(test)}
        overlap = sorted(train_items.intersection(test_items))
        if overlap:
            raise RuntimeError(f"Counterfactual OOF item leakage in fold {fold}: {overlap[:5]}")
        beta_target = np.linalg.lstsq(
            nuisance[train], raw_target[train], rcond=None
        )[0]
        beta_verbal = np.linalg.lstsq(nuisance[train], verbal[train], rcond=None)[0]
        train_target = raw_target[train] - nuisance[train] @ beta_target
        test_target = raw_target[test] - nuisance[test] @ beta_target
        target_residual[test] = test_target
        verbal_residual[test] = verbal[test] - nuisance[test] @ beta_verbal
        scaler = StandardScaler().fit(hidden[train])
        ridge = RidgeCV(
            alphas=np.asarray(RIDGE_ALPHAS), scoring="neg_mean_squared_error"
        ).fit(scaler.transform(hidden[train]), train_target)
        d_raw, raw_intercept = ridge_raw_space(scaler, ridge)
        norm = float(np.linalg.norm(d_raw))
        if norm <= 0 or not np.isfinite(norm):
            raise RuntimeError(f"Degenerate counterfactual direction in fold {fold}")
        d_unit = d_raw / norm
        sign_flipped = bool(
            np.corrcoef(hidden[train] @ d_unit, train_target)[0, 1] < 0
        )
        if sign_flipped:
            d_unit = -d_unit
        train_z = hidden[train] @ d_unit
        sigma_z = float(np.std(train_z, ddof=1))
        mean_z = float(np.mean(train_z))
        predictions[test] = hidden[test] @ d_raw + raw_intercept
        coordinates[test] = (hidden[test] @ d_unit - mean_z) / sigma_z
        filename = f"fold_{fold}_layer_18_panl.npz"
        atomic_save_npz(
            direction_dir / filename,
            alpha=np.asarray(float(ridge.alpha_)),
            d_raw=d_raw,
            d_unit=d_unit,
            raw_intercept=np.asarray(raw_intercept),
            scaler_mean=scaler.mean_,
            scaler_scale=scaler.scale_,
            sigma_z=np.asarray(sigma_z),
            train_z_mean=np.asarray(mean_z),
            nuisance_beta_target=beta_target,
            nuisance_beta_verbal=beta_verbal,
            sign_flipped=np.asarray(sign_flipped),
        )
        fold_r2 = float(r2_score(test_target, predictions[test]))
        audits.append(
            {
                "fold": fold,
                "train_n": int(train.sum()),
                "test_n": int(test.sum()),
                "train_item_count": len(train_items),
                "test_item_count": len(test_items),
                "item_overlap": overlap,
                "selected_alpha": float(ridge.alpha_),
                "sigma_z": sigma_z,
                "train_z_mean": mean_z,
                "test_r2": fold_r2,
                "test_mae": float(mean_absolute_error(test_target, predictions[test])),
            }
        )
        entries.append({"fold": fold, "file": filename, **audits[-1]})
    if not all(
        np.isfinite(values).all()
        for values in (target_residual, verbal_residual, predictions, coordinates)
    ):
        raise RuntimeError("Counterfactual OOF fit did not cover every row")
    output_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        clean = {key: value for key, value in row.items() if key != "manifest"}
        clean.update(
            {
                "behavior_use_residual": float(target_residual[index]),
                "verbal_sa_residual": float(verbal_residual[index]),
                "oof_behavior_residual_prediction": float(predictions[index]),
                "z_behavior_use_std": float(coordinates[index]),
            }
        )
        output_rows.append(clean)
    atomic_write_json(
        direction_dir / "index.json",
        {
            "format_version": 1,
            "definition": "item-OOF StandardScaler + RidgeCV direction for counterfactual relative-image log support after training-fold nuisance residualization",
            "target": "relative_image_use_log residualized on final side, difficulty, and prior strength",
            "layer": 18,
            "position": "panl",
            "alphas": list(RIDGE_ALPHAS),
            "folds": entries,
        },
    )
    return output_rows, {"fold_audits": audits}


def _paired_evidence_changes(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_key.setdefault((str(row["item_id"]), int(row["prior_index"])), {})[
            row["difficulty"]
        ] = row
    pairs: list[dict[str, Any]] = []
    for (item_id, prior_index), values in by_key.items():
        if not {"easy", "hard"}.issubset(values):
            continue
        easy, hard = values["easy"], values["hard"]
        if easy["final_answer"] != hard["final_answer"]:
            continue
        pairs.append(
            {
                "item_id": item_id,
                "prior_index": prior_index,
                "final_image": easy["final_image"],
                "delta_behavior_easy_minus_hard": easy["behavior_imageward_score"]
                - hard["behavior_imageward_score"],
                "delta_verbal_sa_easy_minus_hard": easy["verbal_sa"]
                - hard["verbal_sa"],
            }
        )
    return pairs


def run_counterfactual_source_use(
    artifacts: SAFormationArtifacts,
    output_root: Path,
) -> dict[str, Any]:
    directory = output_root / BEHAVIOR_DIR
    directory.mkdir(parents=True, exist_ok=True)
    rows = build_counterfactual_rows(artifacts)
    oof, audit = fit_counterfactual_oof_direction(artifacts, rows, directory)
    write_jsonl_atomic(directory / "results.jsonl", oof)
    atomic_write_json(directory / "fold_audit.json", audit)
    atomic_write_json(
        directory / "cohort_manifest.json",
        {
            "n": len(oof),
            "unique_items": len({row["item_id"] for row in oof}),
            "filter": "A_T != A_I and natural final answer is A_T or A_I",
            "fixed_answer": "natural full-evidence final answer",
            "counterfactual_definition": "log P(A_final|Image)-log P(A_final|Text), equivalently removal-image drop minus removal-text drop",
            "nuisance": ["final_answer_side", "difficulty", "prior_strength"],
            "nuisance_fit": "training items only within each held-out item fold",
        },
    )
    raw_alignment = _association(oof, "behavior_imageward_score", "verbal_sa")
    residual_alignment = _association(oof, "behavior_use_residual", "verbal_sa_residual")
    decoder = _association(
        oof, "oof_behavior_residual_prediction", "behavior_use_residual"
    )
    target = np.asarray([row["behavior_use_residual"] for row in oof])
    prediction = np.asarray(
        [row["oof_behavior_residual_prediction"] for row in oof]
    )
    decoder.update(
        {
            "r2": float(r2_score(target, prediction)),
            "mae": float(mean_absolute_error(target, prediction)),
        }
    )
    by_side = {
        side: _association(
            [row for row in oof if bool(row["final_image"]) == (side == "image")],
            "behavior_imageward_score",
            "verbal_sa",
        )
        for side in ("text", "image")
    }
    evidence_pairs = _paired_evidence_changes(oof)
    write_jsonl_atomic(directory / "evidence_pairs.jsonl", evidence_pairs)
    evidence = {
        "n": len(evidence_pairs),
        "behavior_change": paired_effect_summary(
            evidence_pairs, "delta_behavior_easy_minus_hard"
        ),
        "verbal_sa_change": paired_effect_summary(
            evidence_pairs, "delta_verbal_sa_easy_minus_hard"
        ),
        "change_alignment": _association(
            evidence_pairs,
            "delta_behavior_easy_minus_hard",
            "delta_verbal_sa_easy_minus_hard",
        ),
    }
    residual_ci = residual_alignment["spearman_item_bootstrap"]["ci95"]
    weak_alignment = bool(
        residual_alignment["spearman"] is not None
        and abs(float(residual_alignment["spearman"])) < 0.10
    )
    direction_effective = bool(
        decoder["r2"] > 0
        and decoder["spearman_item_bootstrap"]["ci95"][0] is not None
        and decoder["spearman_item_bootstrap"]["ci95"][0] > 0
    )
    summary = {
        "title": "Truth Audit 1 — Counterfactual Source Use",
        "status": "completed",
        "n": len(oof),
        "unique_items": len({row["item_id"] for row in oof}),
        "verbal_sa_alignment": {
            "raw": raw_alignment,
            "within_final_side": by_side,
            "training_fold_nuisance_residualized": residual_alignment,
            "weak_beyond_choice_side": weak_alignment,
            "residual_ci_excludes_zero": bool(
                residual_ci[0] is not None and residual_ci[0] > 0
            ),
        },
        "behavior_direction_oof": decoder,
        "behavior_direction_gate": {
            "passed": direction_effective,
            "rule": "OOF residual R2 > 0 and item-bootstrap Spearman CI lower > 0",
        },
        "paired_evidence_perturbation": evidence,
        "classification": (
            "verbal SA is dominated by final-choice side rather than graded counterfactual source support"
            if weak_alignment
            else "verbal SA retains nontrivial graded alignment beyond final-choice side"
        ),
    }
    write_experiment_summary(directory, summary)
    return summary


def _prompt_with_text(case: Any, text_clue: str) -> str:
    variant = get_source_prompt_variant("answer_basis_9")
    return variant.v4_joint_prompt.format(
        question=case.question,
        text_clue=text_clue,
        source_classes=variant.class_text,
    )


def canonical_leading_answer_tokens(tokenizer: Any, classes: Sequence[str]) -> dict[str, int]:
    encodings = {
        normalize_answer(label): tokenizer.encode(
            " " + str(normalize_answer(label)), add_special_tokens=False
        )
        for label in classes
    }
    invalid = {label: ids for label, ids in encodings.items() if len(ids) != 1}
    if invalid:
        raise ValueError(f"Answer labels are not canonical leading-space single tokens: {invalid}")
    token_ids = {str(label): int(ids[0]) for label, ids in encodings.items()}
    if len(set(token_ids.values())) != len(token_ids):
        raise ValueError("Canonical answer labels collide at the token-id level")
    return token_ids


def direct_fixed_answer_distribution(
    runtime: Stage3Runtime,
    *,
    prompt: str,
    image_path: str,
    answer_classes: Sequence[str],
    fixed_answer: str,
) -> dict[str, Any]:
    messages = [
        {"role": "user", "content": image_content(image_path, prompt)},
        assistant_message(ASSISTANT_ANSWER_PREFILL),
    ]
    rendered, inputs = runtime.generator.prepare_messages(
        messages, assistant_text=ASSISTANT_ANSWER_PREFILL
    )
    del rendered
    position = int(inputs.input_ids.shape[1]) - 1
    vocab_logits = run_logits_forward(
        runtime.model, inputs, [position], runtime.modules
    )[position]
    token_ids = canonical_leading_answer_tokens(
        runtime.generator.tokenizer, answer_classes
    )
    labels = [str(normalize_answer(label)) for label in answer_classes]
    class_logits = torch.stack([vocab_logits[token_ids[label]] for label in labels])
    probabilities = torch.softmax(class_logits.float(), dim=-1)
    logits_map = {
        label: float(class_logits[index].item()) for index, label in enumerate(labels)
    }
    probability_map = {
        label: float(probabilities[index].item()) for index, label in enumerate(labels)
    }
    ordered = sorted(logits_map.items(), key=lambda value: value[1], reverse=True)
    top_margin = float(ordered[0][1] - ordered[1][1]) if len(ordered) > 1 else math.inf
    fixed = str(normalize_answer(fixed_answer))
    if fixed not in probability_map:
        raise ValueError(f"Fixed answer {fixed!r} is not in the candidate set")
    del inputs
    return {
        "messages_hash": canonical_message_hash(messages),
        "fixed_answer": fixed,
        "fixed_answer_probability": probability_map[fixed],
        "predicted_answer": ordered[0][0],
        "unique_top1": bool(top_margin > 1e-6),
        "top1_top2_logit_margin": top_margin,
        "answer_class_logits": logits_map,
        "answer_class_probabilities": probability_map,
        "canonical_leading_token_ids": token_ids,
    }


def _select_grounding_donors(
    runtime: Stage3Runtime,
    cohort: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    token_rows = {
        row["case_id"]: row for row in [*candidates, *cohort]
    }
    token_length = {
        row["case_id"]: len(
            runtime.generator.tokenizer.encode(
                runtime.case(row["item_id"], row["prior_index"]).text_clue,
                add_special_tokens=False,
            )
        )
        for row in token_rows.values()
    }
    donors: dict[str, dict[str, Any]] = {}
    for target in cohort:
        target_case = runtime.case(target["item_id"], target["prior_index"])
        eligible = [
            row
            for row in candidates
            if row["item_id"] != target["item_id"]
            and int(row["fold"]) == int(target["fold"])
            and row["difficulty"] == target["difficulty"]
            and int(row["final_image"]) == int(target["final_image"])
        ]
        same_prior_bin = [
            row
            for row in eligible
            if runtime.case(row["item_id"], row["prior_index"]).prior_bin
            == target_case.prior_bin
        ]
        if same_prior_bin:
            eligible = same_prior_bin
        if not eligible:
            raise ValueError(f"No matched donor for {target['case_id']}")
        donors[target["case_id"]] = min(
            eligible,
            key=lambda row: (
                abs(token_length[row["case_id"]] - token_length[target["case_id"]]),
                int(row["item_id"]),
                int(row["prior_index"]),
            ),
        )
    return donors


def _grounding_design(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    answers = sorted({str(row["final_answer"]) for row in rows})
    reference = answers[0]
    return np.asarray(
        [
            [
                1.0,
                float(row["final_image"]),
                float(row["difficulty"] == "hard"),
                float(row.get("prior_strength", 0.0)),
                *[
                    float(row["final_answer"] == answer)
                    for answer in answers
                    if answer != reference
                ],
            ]
            for row in rows
        ],
        dtype=np.float64,
    )


def _crossfit_residual_columns(
    rows: list[dict[str, Any]], columns: Sequence[str]
) -> None:
    design = _grounding_design(rows)
    folds = np.asarray([int(row["fold"]) for row in rows])
    for column in columns:
        outcome = np.asarray([float(row[column]) for row in rows])
        residual = np.full(len(rows), np.nan)
        for fold in sorted(set(folds.tolist())):
            train = folds != fold
            test = folds == fold
            beta = np.linalg.lstsq(design[train], outcome[train], rcond=None)[0]
            residual[test] = outcome[test] - design[test] @ beta
        if not np.isfinite(residual).all():
            raise RuntimeError(f"Cross-fit residualization failed for {column}")
        for index, row in enumerate(rows):
            row[f"{column}_residual"] = float(residual[index])


def _oof_incremental_r2(
    rows: Sequence[dict[str, Any]], target: str, predictor: str
) -> dict[str, float]:
    design = _grounding_design(rows)
    added = np.asarray([float(row[predictor]) for row in rows])[:, None]
    outcome = np.asarray([float(row[target]) for row in rows])
    folds = np.asarray([int(row["fold"]) for row in rows])
    base_prediction = np.full(len(rows), np.nan)
    extended_prediction = np.full(len(rows), np.nan)
    for fold in sorted(set(folds.tolist())):
        train = folds != fold
        test = folds == fold
        beta_base = np.linalg.lstsq(design[train], outcome[train], rcond=None)[0]
        extended_design = np.column_stack([design, added])
        beta_extended = np.linalg.lstsq(
            extended_design[train], outcome[train], rcond=None
        )[0]
        base_prediction[test] = design[test] @ beta_base
        extended_prediction[test] = extended_design[test] @ beta_extended
    base_r2 = float(r2_score(outcome, base_prediction))
    extended_r2 = float(r2_score(outcome, extended_prediction))
    return {
        "base_oof_r2": base_r2,
        "extended_oof_r2": extended_r2,
        "delta_oof_r2": extended_r2 - base_r2,
    }


def run_matched_prompt_grounding(
    runtime: Stage3Runtime,
    experiment_root: Path,
    output_root: Path,
    *,
    n_items: int,
    deadline: Callable[[], None],
) -> dict[str, Any]:
    directory = output_root / MATCHED_GROUNDING_DIR
    directory.mkdir(parents=True, exist_ok=True)
    source_rows = [
        row
        for row in load_jsonl(output_root / BEHAVIOR_DIR / "results.jsonl")
        if row.get("status") == "completed"
    ]
    cohort = _balanced_unique_cases(source_rows, n_items)
    donors = _select_grounding_donors(runtime, cohort, source_rows)
    atomic_write_json(
        directory / "cohort_manifest.json",
        {
            "n": len(cohort),
            "case_ids": [row["case_id"] for row in cohort],
            "conditions": ["full", "no_text", "no_image", "replace_text", "replace_image"],
            "fixed_answer": "natural full-evidence answer A*",
            "answer_scoring": "canonical leading-space one-token label only",
            "prompt": "same answer_basis_9 joint prompt and **Answer**: causal prefix",
            "donors": {
                row["case_id"]: donors[row["case_id"]]["case_id"] for row in cohort
            },
            "matching": ["fold", "difficulty", "final answer side", "prior bin when available", "text token length"],
        },
    )
    result_path = directory / "results.jsonl"
    existing = {
        row["intervention_key"]
        for row in _latest_rows(result_path)
        if row.get("status") == "completed"
    }
    for row in cohort:
        deadline()
        key = f"matched_grounding|{row['case_id']}"
        if key in existing:
            continue
        donor = donors[row["case_id"]]
        base = {
            "intervention_key": key,
            "experiment": "matched_prompt_source_perturbation",
            "case_id": row["case_id"],
            "item_id": row["item_id"],
            "prior_index": row["prior_index"],
            "condition": row["condition"],
            "difficulty": row["difficulty"],
            "fold": int(row["fold"]),
            "text_answer": row["text_answer"],
            "image_answer": row["image_answer"],
            "final_answer": row["final_answer"],
            "final_image": int(row["final_image"]),
            "prior_strength": float(row["prior_strength"]),
            "verbal_sa": float(row["verbal_sa"]),
            "donor_case_id": donor["case_id"],
            "donor_item_id": donor["item_id"],
        }

        def execute() -> dict[str, Any]:
            case = runtime.case(row["item_id"], row["prior_index"])
            donor_case = runtime.case(donor["item_id"], donor["prior_index"])
            original_image = str(case.conditions[row["condition"]].resolved_image_path)
            null_image = str(case.conditions["null"].resolved_image_path)
            irrelevant_image = str(case.conditions["irr"].resolved_image_path)
            conditions = {
                "full": (_prompt_with_text(case, case.text_clue), original_image),
                "no_text": (
                    _prompt_with_text(case, "[No text clue available.]"),
                    original_image,
                ),
                "no_image": (_prompt_with_text(case, case.text_clue), null_image),
                "replace_text": (
                    _prompt_with_text(case, donor_case.text_clue),
                    original_image,
                ),
                "replace_image": (
                    _prompt_with_text(case, case.text_clue),
                    irrelevant_image,
                ),
            }
            measured = {
                name: direct_fixed_answer_distribution(
                    runtime,
                    prompt=prompt,
                    image_path=image_path,
                    answer_classes=case.answer_classes,
                    fixed_answer=row["final_answer"],
                )
                for name, (prompt, image_path) in conditions.items()
            }
            logp = {
                name: _log_probability(value["fixed_answer_probability"])
                for name, value in measured.items()
            }
            deletion = logp["no_text"] - logp["no_image"]
            replacement = logp["replace_text"] - logp["replace_image"]
            return {
                **base,
                "status": "completed",
                "measurements": measured,
                "remove_image_drop_logp": logp["full"] - logp["no_image"],
                "remove_text_drop_logp": logp["full"] - logp["no_text"],
                "replace_image_drop_logp": logp["full"] - logp["replace_image"],
                "replace_text_drop_logp": logp["full"] - logp["replace_text"],
                "behavior_delete_imageward": deletion,
                "behavior_replace_imageward": replacement,
                "full_endpoint_reconstructed": bool(
                    measured["full"]["predicted_answer"] == row["final_answer"]
                    and measured["full"]["unique_top1"]
                ),
            }

        append_jsonl(result_path, _safe_record(base, execute))
    latest = _latest_rows(result_path)
    completed = [row for row in latest if row.get("status") == "completed"]
    confirmatory = [row for row in completed if row["full_endpoint_reconstructed"]]
    if len(confirmatory) < 3:
        raise RuntimeError(
            f"Only {len(confirmatory)} matched-prompt cases reconstruct a unique natural endpoint"
        )
    _crossfit_residual_columns(
        confirmatory,
        ["behavior_delete_imageward", "behavior_replace_imageward", "verbal_sa"],
    )
    reliability = _association(
        confirmatory,
        "behavior_delete_imageward_residual",
        "behavior_replace_imageward_residual",
    )
    sign_agreement = statistics.fmean(
        float(
            row["behavior_delete_imageward_residual"]
            * row["behavior_replace_imageward_residual"]
            > 0
        )
        for row in confirmatory
    )
    deletion_alignment = _association(
        confirmatory, "behavior_delete_imageward_residual", "verbal_sa_residual"
    )
    replacement_alignment = _association(
        confirmatory, "behavior_replace_imageward_residual", "verbal_sa_residual"
    )
    incremental = {
        "deletion": _oof_incremental_r2(
            confirmatory, "behavior_delete_imageward", "verbal_sa"
        ),
        "replacement": _oof_incremental_r2(
            confirmatory, "behavior_replace_imageward", "verbal_sa"
        ),
    }
    write_jsonl_atomic(directory / "confirmatory_analysis.jsonl", confirmatory)
    behavior_gate = bool(
        len(confirmatory) >= 80
        and reliability["spearman_item_bootstrap"]["ci95"][0] is not None
        and reliability["spearman_item_bootstrap"]["ci95"][0] > 0
        and sign_agreement >= 0.60
    )
    sa_gate = bool(
        behavior_gate
        and deletion_alignment["spearman_item_bootstrap"]["ci95"][0] is not None
        and deletion_alignment["spearman_item_bootstrap"]["ci95"][0] > 0
        and incremental["deletion"]["delta_oof_r2"] > 0
    )
    summary = {
        "title": "Truth Audit 2 — Matched-prompt Source Perturbation",
        "status": "completed",
        "attempted": len(latest),
        "completed": len(completed),
        "failed": len(latest) - len(completed),
        "confirmatory_unique_endpoint_n": len(confirmatory),
        "full_endpoint_reconstruction_rate": len(confirmatory) / len(completed),
        "behavior_target_reliability": {
            "delete_vs_replace_residual": reliability,
            "residual_sign_agreement": sign_agreement,
            "gate_passed": behavior_gate,
            "rule": "n>=80, residual delete-vs-replace Spearman CI lower>0, sign agreement>=.60",
        },
        "verbal_sa_alignment": {
            "deletion_residual": deletion_alignment,
            "replacement_residual": replacement_alignment,
            "oof_incremental_r2": incremental,
            "gate_passed": sa_gate,
        },
        "classification": (
            "reliable behavioral source-use contrast is aligned with verbal SA"
            if sa_gate
            else "behavioral source-use and verbal self-report are not jointly validated"
        ),
    }
    write_experiment_summary(directory, summary)
    return summary


def factorial_contrasts(cells: dict[str, float]) -> dict[str, float]:
    required = {"text_at", "text_ai", "image_at", "image_ai"}
    if set(cells) != required:
        raise ValueError(f"Factorial cells must be exactly {sorted(required)}")
    modality = 0.5 * (
        cells["image_at"] + cells["image_ai"]
        - cells["text_at"] - cells["text_ai"]
    )
    prior_answer = 0.5 * (
        cells["text_ai"] + cells["image_ai"]
        - cells["text_at"] - cells["image_at"]
    )
    interaction = (
        cells["image_ai"] - cells["text_ai"]
        - cells["image_at"] + cells["text_at"]
    )
    return {
        "modality_main": float(modality),
        "prior_answer_main": float(prior_answer),
        "interaction": float(interaction),
        "congruence_main": float(interaction / 2.0),
    }


def _joint_branch_value(row: dict[str, Any], branch: str, outcome: str) -> float:
    payload = row["branches"][branch]
    if outcome == "verbal_sa":
        return float(payload["pass1"]["source_attribution"]["soft_image_score"])
    if outcome == "p_image_answer":
        return float(payload["pass1"]["answer_class_probabilities"][row["image_answer"]])
    if outcome == "answer_margin":
        logits = payload["pass1"]["answer_class_logits"]
        return float(logits[row["image_answer"]] - logits[row["text_answer"]])
    if outcome == "hard_image_answer":
        return float(payload["pass1"]["normalized_answer"] == row["image_answer"])
    if outcome == "other_answer":
        return float(
            payload["pass1"]["normalized_answer"]
            not in {row["text_answer"], row["image_answer"]}
        )
    if outcome == "follow_prior":
        prior = row["text_answer"] if branch.endswith("_at") else row["image_answer"]
        return float(payload["pass1"]["normalized_answer"] == prior)
    if outcome == "z_sa":
        return float(payload["pass2"]["z_sa"])
    raise KeyError(outcome)


def _factorial_rows(
    rows: Sequence[dict[str, Any]], outcome: str
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for row in rows:
        cells = {
            branch: _joint_branch_value(row, branch, outcome)
            for branch in ("text_at", "text_ai", "image_at", "image_ai")
        }
        values.append({"item_id": row["item_id"], **factorial_contrasts(cells)})
    return values


def summarize_factorial_values(
    rows: Sequence[dict[str, Any]], outcome: str
) -> dict[str, Any]:
    contrasts = _factorial_rows(rows, outcome)
    return {
        key: paired_effect_summary(contrasts, key)
        for key in (
            "modality_main",
            "prior_answer_main",
            "interaction",
            "congruence_main",
        )
    }


def run_history_factorial_reanalysis(
    experiment_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    directory = output_root / FACTORIAL_DIR
    directory.mkdir(parents=True, exist_ok=True)
    source = (
        experiment_root
        / "stage3_sa_formation_followup"
        / "02_history_exact_factorial"
        / "results_nocache.jsonl"
    )
    rows = [row for row in _latest_rows(source) if row.get("status") == "completed"]
    if len(rows) != 30:
        raise ValueError(f"Expected 30 authoritative factorial cases, got {len(rows)}")
    if not all(row.get("final_evidence_equal") for row in rows):
        raise ValueError("Factorial branches do not all share final evidence")
    reconstruction = [
        branch["reconstruction"]
        for row in rows
        for branch in row["branches"].values()
    ]
    if not all(
        value["raw_logit_within_bf16_tolerance"]
        and value["soft_within_0.01"]
        for value in reconstruction
    ):
        raise ValueError("Authoritative factorial reconstruction audit failed")
    compact: list[dict[str, Any]] = []
    for row in rows:
        compact.append(
            {
                "intervention_key": f"factorial_reanalysis|{row['case_id']}",
                "experiment": "history_factorial_reanalysis",
                "status": "completed",
                "case_id": row["case_id"],
                "item_id": row["item_id"],
                "prior_index": row["prior_index"],
                "condition": row["condition"],
                "fold": row["fold"],
                "text_answer": row["text_answer"],
                "image_answer": row["image_answer"],
                "strict_four_way_endpoint_matched": row[
                    "strict_four_way_endpoint_matched"
                ],
                "branches": {
                    branch: {
                        "normalized_answer": value["pass1"]["normalized_answer"],
                        "verbal_sa": value["pass1"]["source_attribution"][
                            "soft_image_score"
                        ],
                        "p_image_answer": value["pass1"][
                            "answer_class_probabilities"
                        ][row["image_answer"]],
                        "z_sa": value["pass2"]["z_sa"],
                    }
                    for branch, value in row["branches"].items()
                },
            }
        )
    write_jsonl_atomic(directory / "results.jsonl", compact)
    atomic_write_json(
        directory / "cohort_manifest.json",
        {
            "n": len(rows),
            "unique_items": len({row["item_id"] for row in rows}),
            "source": str(source),
            "source_status": "authoritative no-cache exact-token reconstruction",
            "branches": ["text_at", "text_ai", "image_at", "image_ai", "no_history"],
            "final_evidence_equal": True,
            "reconstruction_branch_n": len(reconstruction),
            "reconstruction_all_passed": True,
        },
    )
    outcomes = {
        outcome: summarize_factorial_values(rows, outcome)
        for outcome in (
            "verbal_sa",
            "answer_margin",
            "p_image_answer",
            "hard_image_answer",
            "other_answer",
            "follow_prior",
            "z_sa",
        )
    }
    summary = {
        "title": "Truth Audit 3 — History Modality × Replayed Answer",
        "status": "completed",
        "n": len(rows),
        "strict_four_way_endpoint_matched_n": sum(
            row["strict_four_way_endpoint_matched"] for row in rows
        ),
        "total_effect_factorial": outcomes,
        "interpretation": {
            "verbal_report_tracks_modality": bool(
                outcomes["verbal_sa"]["modality_main"]["ci95"][0] > 0
            ),
            "answer_behavior_tracks_replayed_answer": bool(
                outcomes["answer_margin"]["prior_answer_main"]["ci95"][0] > 0
            ),
            "total_effect_warning": "All-30 factorial effects include answer-mediated paths; existing pairwise endpoint-matched contrasts remain the direct-report evidence.",
        },
        "classification": "History modality primarily controls verbal SA while the replayed historical answer primarily controls answer behavior",
    }
    write_experiment_summary(directory, summary)
    return summary


def _completed_by_case(path: Path) -> dict[str, dict[str, Any]]:
    return {
        row["case_id"]: row
        for row in _latest_rows(path)
        if row.get("status") == "completed"
    }


def _compact_answer_generation(
    generated: dict[str, Any], image_answer: str
) -> dict[str, Any]:
    normalized = normalize_answer(generated.get("normalized_answer"))
    probabilities = generated["answer_class_probabilities"]
    logits = generated["answer_class_logits"]
    return {
        "normalized_answer": normalized,
        "p_image_answer": float(probabilities[image_answer]),
        "answer_class_probabilities": probabilities,
        "answer_class_logits": logits,
        "answer_metric_status": generated.get("answer_metric_status"),
    }


def _valid_answer_generation(generated: dict[str, Any]) -> bool:
    return bool(
        generated.get("parse_success")
        and generated.get("answer_metric_status") == "completed"
        and generated.get("normalized_answer")
    )


def _answer_protocol_value(
    row: dict[str, Any], protocol: str, branch: str, outcome: str
) -> float:
    value = row[protocol][branch]
    if outcome == "p_image_answer":
        return float(value["p_image_answer"])
    if outcome == "answer_margin":
        logits = value["answer_class_logits"]
        return float(logits[row["image_answer"]] - logits[row["text_answer"]])
    if outcome == "hard_image_answer":
        return float(value["normalized_answer"] == row["image_answer"])
    if outcome == "other_answer":
        return float(
            value["normalized_answer"]
            not in {row["text_answer"], row["image_answer"]}
        )
    if outcome == "follow_prior":
        prior = row["text_answer"] if branch.endswith("_at") else row["image_answer"]
        return float(value["normalized_answer"] == prior)
    raise KeyError(outcome)


def _summarize_answer_protocol_factorial(
    rows: Sequence[dict[str, Any]], protocol: str, outcome: str
) -> dict[str, Any]:
    contrasts: list[dict[str, Any]] = []
    for row in rows:
        cells = {
            branch: _answer_protocol_value(row, protocol, branch, outcome)
            for branch in ("text_at", "text_ai", "image_at", "image_ai")
        }
        contrasts.append({"item_id": row["item_id"], **factorial_contrasts(cells)})
    return {
        key: paired_effect_summary(contrasts, key)
        for key in (
            "modality_main",
            "prior_answer_main",
            "interaction",
            "congruence_main",
        )
    }


def run_answer_only_protocol_robustness(
    runtime: Stage3Runtime,
    experiment_root: Path,
    output_root: Path,
    *,
    deadline: Callable[[], None],
) -> dict[str, Any]:
    """Complete the answer-only 2x2 and compare it with the joint-report prompt."""

    directory = output_root / ANSWER_PROTOCOL_DIR
    directory.mkdir(parents=True, exist_ok=True)
    behavior_source = _completed_by_case(
        experiment_root
        / "stage3_sa_second_order"
        / "01_history_behavior_dissociation"
        / "results.jsonl"
    )
    joint_source = _completed_by_case(
        experiment_root
        / "stage3_sa_formation_followup"
        / "02_history_exact_factorial"
        / "results_nocache.jsonl"
    )
    cohort = sorted(
        behavior_source.values(),
        key=lambda row: (int(row["fold"]), int(row["item_id"]), int(row["prior_index"])),
    )
    if len(cohort) != 30:
        raise ValueError(f"Expected 30 answer-only source cases, got {len(cohort)}")
    atomic_write_json(
        directory / "cohort_manifest.json",
        {
            "n": len(cohort),
            "case_ids": [row["case_id"] for row in cohort],
            "branches": ["text_at", "text_ai", "image_at", "image_ai"],
            "answer_only_reuse": "Text+A_T and Image+A_T from completed stage3_sa_second_order after exact message-hash reconstruction",
            "new_answer_only": "Text+A_I and Image+A_I",
            "joint_reuse": "authoritative no-cache exact-factorial rows when available; all four branches newly generated only for a missing cohort case",
            "generation_use_cache": False,
        },
    )
    result_path = directory / "results.jsonl"
    existing = {
        row["intervention_key"]
        for row in _latest_rows(result_path)
        if row.get("status") == "completed"
    }
    for source_row in cohort:
        deadline()
        key = f"answer_protocol_factorial|{source_row['case_id']}"
        if key in existing:
            continue
        base = {
            "intervention_key": key,
            "experiment": "answer_only_protocol_robustness",
            "case_id": source_row["case_id"],
            "item_id": source_row["item_id"],
            "prior_index": source_row["prior_index"],
            "condition": source_row["condition"],
            "fold": int(source_row["fold"]),
            "text_answer": normalize_answer(source_row["text_answer"]),
            "image_answer": normalize_answer(source_row["image_answer"]),
        }

        def execute() -> dict[str, Any]:
            target = runtime.case(source_row["item_id"], source_row["prior_index"])
            answer_only: dict[str, Any] = {}
            joint: dict[str, Any] = {}
            old_joint = joint_source.get(source_row["case_id"])
            for modality in ("text", "image"):
                for answer_side in ("at", "ai"):
                    branch = f"{modality}_{answer_side}"
                    prior_answer = (
                        base["text_answer"] if answer_side == "at" else base["image_answer"]
                    )
                    answer_messages = build_answer_history_messages(
                        target,
                        source_row["condition"],
                        target,
                        source_row["condition"],
                        modality,
                        prior_answer,
                    )
                    answer_hash = canonical_message_hash(answer_messages)
                    if answer_side == "at":
                        reused = source_row["branches"][f"relevant_{modality}"]
                        if reused["messages_hash"] != answer_hash:
                            raise ValueError(
                                f"Reused answer-only hash mismatch for {source_row['case_id']} {branch}"
                            )
                        generated = reused["generated"]
                        source_kind = "reused_stage3_sa_second_order"
                    else:
                        generated_result = generate_answer_messages(
                            runtime, answer_messages, target.answer_classes
                        )
                        generated = generated_result.to_dict()
                        source_kind = "new_generation"
                    if not _valid_answer_generation(generated):
                        raise RuntimeError(
                            f"Answer-only generation failed for {branch}: {generated.get('error')}"
                        )
                    answer_only[branch] = {
                        **_compact_answer_generation(generated, base["image_answer"]),
                        "messages_hash": answer_hash,
                        "source": source_kind,
                    }

                    joint_messages = build_factorial_history_messages(
                        target,
                        source_row["condition"],
                        modality,
                        prior_answer,
                    )
                    joint_hash = canonical_message_hash(joint_messages)
                    if old_joint is not None:
                        old_branch = old_joint["branches"][branch]
                        if old_branch["messages_hash"] != joint_hash:
                            raise ValueError(
                                f"Reused joint hash mismatch for {source_row['case_id']} {branch}"
                            )
                        joint_generated = old_branch["pass1"]
                        joint_source_kind = "reused_authoritative_exact_factorial"
                    else:
                        generated_joint = runtime.generator.generate_messages(
                            joint_messages,
                            target.answer_classes,
                            max_new_tokens=48,
                            use_cache=False,
                        )
                        joint_generated = generated_joint.to_dict()
                        joint_source_kind = "new_generation_missing_cohort_case"
                    if not _valid_answer_generation(joint_generated):
                        raise RuntimeError(
                            f"Joint generation failed for {branch}: {joint_generated.get('error')}"
                        )
                    joint[branch] = {
                        **_compact_answer_generation(
                            joint_generated, base["image_answer"]
                        ),
                        "messages_hash": joint_hash,
                        "source": joint_source_kind,
                    }
            return {**base, "status": "completed", "answer_only": answer_only, "joint": joint}

        append_jsonl(result_path, _safe_record(base, execute))
    rows = [row for row in _latest_rows(result_path) if row.get("status") == "completed"]
    outcomes = (
        "answer_margin",
        "p_image_answer",
        "hard_image_answer",
        "other_answer",
        "follow_prior",
    )
    factorial = {
        protocol: {
            outcome: _summarize_answer_protocol_factorial(rows, protocol, outcome)
            for outcome in outcomes
        }
        for protocol in ("answer_only", "joint")
    }
    direct: dict[str, Any] = {}
    for outcome in outcomes:
        direct[outcome] = {}
        for branch in ("text_at", "text_ai", "image_at", "image_ai"):
            paired = [
                {
                    "item_id": row["item_id"],
                    "joint_minus_answer_only": _answer_protocol_value(
                        row, "joint", branch, outcome
                    )
                    - _answer_protocol_value(row, "answer_only", branch, outcome),
                }
                for row in rows
            ]
            direct[outcome][branch] = paired_effect_summary(
                paired, "joint_minus_answer_only"
            )
    factor_difference: dict[str, Any] = {}
    for outcome in outcomes:
        values: list[dict[str, Any]] = []
        for row in rows:
            protocol_effects = {}
            for protocol in ("answer_only", "joint"):
                cells = {
                    branch: _answer_protocol_value(row, protocol, branch, outcome)
                    for branch in ("text_at", "text_ai", "image_at", "image_ai")
                }
                protocol_effects[protocol] = factorial_contrasts(cells)
            values.append(
                {
                    "item_id": row["item_id"],
                    **{
                        key: protocol_effects["joint"][key]
                        - protocol_effects["answer_only"][key]
                        for key in protocol_effects["joint"]
                    },
                }
            )
        factor_difference[outcome] = {
            key: paired_effect_summary(values, key)
            for key in (
                "modality_main",
                "prior_answer_main",
                "interaction",
                "congruence_main",
            )
        }
    summary = {
        "title": "Truth Audit 4 — Answer-only versus Joint-report Protocol",
        "status": "completed",
        "n": len(rows),
        "failed": len(_latest_rows(result_path)) - len(rows),
        "factorial": factorial,
        "joint_minus_answer_only_by_branch": direct,
        "joint_minus_answer_only_factor_effect": factor_difference,
        "classification": "report-request effects on first-order answer behavior are estimated directly from otherwise matched History contexts",
    }
    write_experiment_summary(directory, summary)
    return summary


def common_protocol_specs() -> tuple[ProtocolSpec, ...]:
    nine = protocol_specs()[0]
    ternary_descriptions = (
        "The answer was based more on the text clue.",
        "The answer was based on the text clue and image to a similar extent.",
        "The answer was based more on the image.",
    )
    binary_descriptions = (
        "The answer was based more on the text clue.",
        "The answer was based more on the image.",
    )
    return (
        ProtocolSpec("common_9_ordered", tuple(str(i) for i in range(9)), nine.midpoints, nine.descriptions),
        ProtocolSpec("common_3_ordered", ("0", "4", "8"), (0.05, 0.5, 0.95), ternary_descriptions),
        ProtocolSpec("common_2_ordered", ("0", "8"), (0.05, 0.95), binary_descriptions),
        ProtocolSpec("common_3_reversed", ("8", "4", "0"), (0.05, 0.5, 0.95), ternary_descriptions),
        ProtocolSpec("common_2_reversed", ("8", "0"), (0.05, 0.95), binary_descriptions),
        ProtocolSpec("common_3_semantic", ("T", "B", "I"), (0.05, 0.5, 0.95), ternary_descriptions),
        ProtocolSpec("common_2_semantic", ("T", "I"), (0.05, 0.95), binary_descriptions),
    )


def common_protocol_prompt(case: Any, spec: ProtocolSpec) -> str:
    class_block = "\n".join(
        f"{label}: {description}"
        for label, description in zip(spec.labels_by_semantic, spec.descriptions)
    )
    return f"""Question:
{case.question}

Text clue:
{case.text_clue}

Answer the question using the text clue and the image.

Then report the relative contribution of the text clue and the image to that answer by choosing exactly one class listed below.

Source attribution classes:
{class_block}

Output exactly:

**Answer**: <your answer>
**Source Attribution**:<CLASS>

Do not include reasoning, confidence, or any additional text."""


def _common_protocol_context(
    runtime: Stage3Runtime, row: dict[str, Any], spec: ProtocolSpec
) -> Any:
    case = runtime.case(row["item_id"], row["prior_index"])
    answer = str(row["final_answer"])
    assistant_text = f"**Answer**: {answer}\n**Source Attribution**:"
    messages = [
        {
            "role": "user",
            "content": image_content(
                str(case.conditions[row["condition"]].resolved_image_path),
                common_protocol_prompt(case, spec),
            ),
        },
        assistant_message(assistant_text),
    ]
    return prepare_measurement(
        runtime.generator, messages, assistant_text=assistant_text, answer=answer
    )


def _legacy_protocol_context(
    runtime: Stage3Runtime, row: dict[str, Any], spec: ProtocolSpec
) -> Any:
    case = runtime.case(row["item_id"], row["prior_index"])
    answer = str(row["final_answer"])
    assistant_text = f"**Answer**: {answer}\n**Source Attribution**:"
    messages = [
        {
            "role": "user",
            "content": image_content(
                str(case.conditions[row["condition"]].resolved_image_path),
                protocol_prompt(case, spec),
            ),
        },
        assistant_message(assistant_text),
    ]
    return prepare_measurement(
        runtime.generator, messages, assistant_text=assistant_text, answer=answer
    )


def posthoc_collapse_nine(probabilities: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (9,) or not np.isfinite(values).all() or values.sum() <= 0:
        raise ValueError("Nine-class probabilities must be finite with shape (9,)")
    values = values / values.sum()
    q_text = float(values[:4].sum())
    q_both = float(values[4])
    q_image = float(values[5:].sum())
    ternary = np.asarray([q_text, q_both, q_image], dtype=np.float64)
    nonneutral = q_text + q_image
    if nonneutral <= 1e-12:
        binary_conditional = np.asarray([0.5, 0.5], dtype=np.float64)
    else:
        binary_conditional = np.asarray(
            [q_text / nonneutral, q_image / nonneutral], dtype=np.float64
        )
    binary_split = np.asarray(
        [q_text + 0.5 * q_both, q_image + 0.5 * q_both], dtype=np.float64
    )
    return {
        "ternary_probabilities": ternary.tolist(),
        "ternary_score": float(ternary @ np.asarray([0.05, 0.5, 0.95])),
        "binary_conditional_probabilities": binary_conditional.tolist(),
        "binary_conditional_score": float(
            binary_conditional @ np.asarray([0.05, 0.95])
        ),
        "binary_split_probabilities": binary_split.tolist(),
        "binary_split_score": float(binary_split @ np.asarray([0.05, 0.95])),
    }


def _js_divergence(left: Sequence[float], right: Sequence[float]) -> float:
    p = np.asarray(left, dtype=np.float64)
    q = np.asarray(right, dtype=np.float64)
    p = p / p.sum()
    q = q / q.sum()
    midpoint = 0.5 * (p + q)

    def kl(values: np.ndarray, reference: np.ndarray) -> float:
        mask = values > 0
        return float(np.sum(values[mask] * np.log(values[mask] / reference[mask])))

    return 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)


def _select_bridge_cohort(
    source_rows: Sequence[dict[str, Any]],
    previous_manifest: Path,
    n_items: int,
) -> list[dict[str, Any]]:
    previous_ids = set()
    if previous_manifest.is_file():
        import json

        previous_ids = set(json.loads(previous_manifest.read_text())["case_ids"])
    by_case = {row["case_id"]: row for row in source_rows}
    mandatory = [by_case[case_id] for case_id in sorted(previous_ids) if case_id in by_case]
    used_items = {row["item_id"] for row in mandatory}
    remaining = [row for row in source_rows if row["item_id"] not in used_items]
    needed = n_items - len(mandatory)
    if needed < 0:
        mandatory = mandatory[:n_items]
        needed = 0
    added = _balanced_unique_cases(remaining, needed) if needed else []
    cohort = mandatory + added
    if len(cohort) != n_items or len({row["item_id"] for row in cohort}) != n_items:
        raise ValueError("Could not construct unique protocol-bridge cohort")
    return cohort


def _protocol_pair_rows(
    rows: Sequence[dict[str, Any]], left: str, right: str
) -> list[dict[str, Any]]:
    return [
        {
            "item_id": row["item_id"],
            "left": float(row["protocols"][left]["semantic_imageward_score"]),
            "right": float(row["protocols"][right]["semantic_imageward_score"]),
            "difference": float(
                row["protocols"][right]["semantic_imageward_score"]
                - row["protocols"][left]["semantic_imageward_score"]
            ),
        }
        for row in rows
    ]


def _protocol_agreement(
    rows: Sequence[dict[str, Any]], left: str, right: str
) -> dict[str, Any]:
    paired = _protocol_pair_rows(rows, left, right)
    association = _association(paired, "left", "right")
    association["paired_difference"] = paired_effect_summary(paired, "difference")
    return association


def _crossfit_affine_calibration(
    rows: Sequence[dict[str, Any]], protocol: str
) -> dict[str, Any]:
    x = np.asarray(
        [row["protocols"][protocol]["ridge_sa_prediction"] for row in rows],
        dtype=np.float64,
    )
    y = np.asarray(
        [row["protocols"][protocol]["semantic_imageward_score"] for row in rows],
        dtype=np.float64,
    )
    folds = np.asarray([int(row["fold"]) for row in rows])
    prediction = np.full(len(rows), np.nan)
    for fold in sorted(set(folds.tolist())):
        train = folds != fold
        test = folds == fold
        design_train = np.column_stack([np.ones(train.sum()), x[train]])
        beta = np.linalg.lstsq(design_train, y[train], rcond=None)[0]
        prediction[test] = np.column_stack([np.ones(test.sum()), x[test]]) @ beta
    return {
        "n": len(rows),
        "oof_r2": float(r2_score(y, prediction)),
        "oof_mae": float(mean_absolute_error(y, prediction)),
    }


def paired_ci_within_equivalence_band(
    effect: dict[str, Any], threshold: float
) -> bool:
    """Return whether the full paired 95% CI lies inside ``[-threshold, threshold]``."""
    low, high = effect["paired_difference"]["ci95"]
    return bool(
        threshold > 0
        and low is not None
        and high is not None
        and max(abs(float(low)), abs(float(high))) <= threshold
    )


def run_protocol_granularity_bridge(
    runtime: Stage3Runtime,
    experiment_root: Path,
    output_root: Path,
    *,
    n_items: int,
    deadline: Callable[[], None],
) -> dict[str, Any]:
    directory = output_root / GRANULARITY_DIR
    directory.mkdir(parents=True, exist_ok=True)
    behavior_rows = [
        row
        for row in load_jsonl(output_root / BEHAVIOR_DIR / "results.jsonl")
        if row.get("status") == "completed"
    ]
    cohort = _select_bridge_cohort(
        behavior_rows,
        experiment_root
        / "stage3_sa_second_order"
        / "03_protocol_invariant_semantic_sa"
        / "cohort_manifest.json",
        n_items,
    )
    common_specs = common_protocol_specs()
    legacy_by_name = {spec.name: spec for spec in protocol_specs()}
    legacy_specs = (
        legacy_by_name["normal_numeric"],
        legacy_by_name["text_both_image"],
        legacy_by_name["binary_text_image"],
    )
    all_specs = common_specs + legacy_specs
    analyzers = {
        spec.name if spec in common_specs else f"legacy_{spec.name}": ProtocolAnalyzer(
            runtime.generator.tokenizer, spec
        )
        for spec in all_specs
    }
    protocol_names = [spec.name for spec in common_specs] + [
        f"legacy_{spec.name}" for spec in legacy_specs
    ]
    atomic_write_json(
        directory / "cohort_manifest.json",
        {
            "n": len(cohort),
            "case_ids": [row["case_id"] for row in cohort],
            "protocols": protocol_names,
            "common_protocols": [asdict(spec) for spec in common_specs],
            "legacy_anchors": [asdict(spec) for spec in legacy_specs],
            "fixed_answer": "natural baseline answer",
            "site": "L18 PANL",
            "saved": ["restricted logits/probabilities", "hidden", "legacy SA decoder", "behavior-grounded decoder"],
        },
    )
    ridge_sa = SAOOFDirectionRepository(
        experiment_root / "stage3_sa_formation" / "directions"
    )
    ridge_behavior = SAOOFDirectionRepository(
        output_root / BEHAVIOR_DIR / "directions"
    )
    result_path = directory / "results.jsonl"
    existing = {
        row["intervention_key"]
        for row in _latest_rows(result_path)
        if row.get("status") == "completed"
    }
    hidden_dir = directory / "hidden"
    for row in cohort:
        deadline()
        key = f"granularity_bridge|{row['case_id']}"
        if key in existing:
            continue
        base = {
            "intervention_key": key,
            "experiment": "protocol_granularity_bridge",
            "case_id": row["case_id"],
            "item_id": row["item_id"],
            "prior_index": row["prior_index"],
            "condition": row["condition"],
            "difficulty": row["difficulty"],
            "fold": int(row["fold"]),
            "final_answer": row["final_answer"],
            "final_image": row["final_image"],
            "behavior_use_residual": row["behavior_use_residual"],
        }

        def execute() -> dict[str, Any]:
            sa_direction = ridge_sa.get(row["fold"])
            behavior_direction = ridge_behavior.get(row["fold"])
            results: dict[str, Any] = {}
            hidden_values: list[np.ndarray] = []
            for spec in common_specs:
                name = spec.name
                prepared = _common_protocol_context(runtime, row, spec)
                measured = runtime.measure(
                    prepared, sa_direction, analyzer=analyzers[name]
                )
                hidden = measured.hidden.astype(np.float32, copy=False)
                results[name] = {
                    "semantic_imageward_score": float(measured.source["soft_image_score"]),
                    "hard_label": measured.source["hard_label"],
                    "class_logits": measured.source["class_logits"],
                    "class_probabilities": measured.source["class_probabilities"],
                    "labels_by_semantic": list(spec.labels_by_semantic),
                    "ridge_sa_prediction": sa_direction.predict(hidden),
                    "ridge_sa_coordinate": sa_direction.z(hidden),
                    "ridge_behavior_prediction": behavior_direction.predict(hidden),
                    "ridge_behavior_coordinate": behavior_direction.z(hidden),
                    "prefix_hash": prepared.prefix_hash,
                    "panl_position": prepared.panl_position,
                    "input_token_count": int(prepared.inputs.input_ids.shape[1]),
                    "hook_exactly_once": measured.applied_count == 1,
                }
                hidden_values.append(hidden)
                runtime.release_inputs(prepared)
            for spec in legacy_specs:
                name = f"legacy_{spec.name}"
                prepared = _legacy_protocol_context(runtime, row, spec)
                measured = runtime.measure(
                    prepared, sa_direction, analyzer=analyzers[name]
                )
                hidden = measured.hidden.astype(np.float32, copy=False)
                results[name] = {
                    "semantic_imageward_score": float(measured.source["soft_image_score"]),
                    "hard_label": measured.source["hard_label"],
                    "class_logits": measured.source["class_logits"],
                    "class_probabilities": measured.source["class_probabilities"],
                    "labels_by_semantic": list(spec.labels_by_semantic),
                    "ridge_sa_prediction": sa_direction.predict(hidden),
                    "ridge_sa_coordinate": sa_direction.z(hidden),
                    "ridge_behavior_prediction": behavior_direction.predict(hidden),
                    "ridge_behavior_coordinate": behavior_direction.z(hidden),
                    "prefix_hash": prepared.prefix_hash,
                    "panl_position": prepared.panl_position,
                    "input_token_count": int(prepared.inputs.input_ids.shape[1]),
                    "hook_exactly_once": measured.applied_count == 1,
                }
                hidden_values.append(hidden)
                runtime.release_inputs(prepared)
            collapse = posthoc_collapse_nine(
                results["common_9_ordered"]["class_probabilities"]
            )
            results["posthoc_from_common_9"] = collapse
            results["common_3_ordered"]["posthoc_js_divergence"] = _js_divergence(
                collapse["ternary_probabilities"],
                results["common_3_ordered"]["class_probabilities"],
            )
            results["common_2_ordered"]["posthoc_conditional_js_divergence"] = _js_divergence(
                collapse["binary_conditional_probabilities"],
                results["common_2_ordered"]["class_probabilities"],
            )
            hidden_file = hidden_dir / f"{row['case_id'].replace('/', '_')}.npz"
            atomic_save_npz(
                hidden_file,
                protocols=np.asarray(protocol_names),
                hidden=np.stack(hidden_values),
            )
            return {
                **base,
                "status": "completed",
                "protocols": results,
                "hidden_file": str(hidden_file.relative_to(output_root)),
            }

        append_jsonl(result_path, _safe_record(base, execute))
    latest = _latest_rows(result_path)
    completed = [row for row in latest if row.get("status") == "completed"]
    summary = _summarize_granularity_bridge(completed, len(latest) - len(completed))
    write_experiment_summary(directory, summary)
    return summary


def _summarize_granularity_bridge(
    rows: Sequence[dict[str, Any]], failed: int
) -> dict[str, Any]:
    expanded: list[dict[str, Any]] = []
    for row in rows:
        collapse = row["protocols"]["posthoc_from_common_9"]
        protocols = dict(row["protocols"])
        protocols["posthoc_3"] = {
            "semantic_imageward_score": collapse["ternary_score"]
        }
        protocols["posthoc_2_conditional"] = {
            "semantic_imageward_score": collapse["binary_conditional_score"]
        }
        protocols["posthoc_2_split"] = {
            "semantic_imageward_score": collapse["binary_split_score"]
        }
        expanded.append({**row, "protocols": protocols})
    bridge = {
        "ternary_prompted_vs_posthoc": _protocol_agreement(
            expanded, "posthoc_3", "common_3_ordered"
        ),
        "binary_prompted_vs_posthoc_conditional": _protocol_agreement(
            expanded, "posthoc_2_conditional", "common_2_ordered"
        ),
        "binary_prompted_vs_posthoc_split": _protocol_agreement(
            expanded, "posthoc_2_split", "common_2_ordered"
        ),
    }
    mapping = {
        "ternary_ordered_vs_reversed": _protocol_agreement(
            expanded, "common_3_ordered", "common_3_reversed"
        ),
        "ternary_ordered_vs_semantic": _protocol_agreement(
            expanded, "common_3_ordered", "common_3_semantic"
        ),
        "binary_ordered_vs_reversed": _protocol_agreement(
            expanded, "common_2_ordered", "common_2_reversed"
        ),
        "binary_ordered_vs_semantic": _protocol_agreement(
            expanded, "common_2_ordered", "common_2_semantic"
        ),
    }
    grammar = {
        "nine_legacy_vs_common": _protocol_agreement(
            expanded, "legacy_normal_numeric", "common_9_ordered"
        ),
        "ternary_legacy_vs_common": _protocol_agreement(
            expanded, "legacy_text_both_image", "common_3_semantic"
        ),
        "binary_legacy_vs_common": _protocol_agreement(
            expanded, "legacy_binary_text_image", "common_2_semantic"
        ),
    }
    primary_protocols = [
        "common_9_ordered",
        "common_3_ordered",
        "common_2_ordered",
        "common_3_reversed",
        "common_2_reversed",
        "common_3_semantic",
        "common_2_semantic",
    ]
    decoders: dict[str, Any] = {}
    behavior_decoders: dict[str, Any] = {}
    calibration: dict[str, Any] = {}
    for protocol in primary_protocols:
        values = [
            {
                "item_id": row["item_id"],
                "prediction": row["protocols"][protocol]["ridge_sa_prediction"],
                "semantic": row["protocols"][protocol]["semantic_imageward_score"],
                "behavior_prediction": row["protocols"][protocol][
                    "ridge_behavior_prediction"
                ],
                "behavior_target": row["behavior_use_residual"],
            }
            for row in expanded
        ]
        decoders[protocol] = _association(values, "prediction", "semantic")
        behavior_decoders[protocol] = _association(
            values, "behavior_prediction", "behavior_target"
        )
        calibration[protocol] = _crossfit_affine_calibration(expanded, protocol)
    js = {
        "ternary_mean": statistics.fmean(
            row["protocols"]["common_3_ordered"]["posthoc_js_divergence"]
            for row in expanded
        ),
        "binary_conditional_mean": statistics.fmean(
            row["protocols"]["common_2_ordered"][
                "posthoc_conditional_js_divergence"
            ]
            for row in expanded
        ),
    }

    def hard_semantic_index(row: dict[str, Any], protocol: str) -> int:
        value = row["protocols"][protocol]
        return list(value["labels_by_semantic"]).index(value["hard_label"])

    hard_semantic_agreement = {
        "ternary_ordered_vs_reversed": statistics.fmean(
            hard_semantic_index(row, "common_3_ordered")
            == hard_semantic_index(row, "common_3_reversed")
            for row in expanded
        ),
        "binary_ordered_vs_reversed": statistics.fmean(
            hard_semantic_index(row, "common_2_ordered")
            == hard_semantic_index(row, "common_2_reversed")
            for row in expanded
        ),
    }

    def rank_pass(values: dict[str, Any]) -> bool:
        bootstrap = values["spearman_item_bootstrap"]
        return bool(
            bootstrap
            and bootstrap["ci95"][0] is not None
            and bootstrap["ci95"][0] > 0
        )

    bridge_pass = all(rank_pass(value) for value in bridge.values())
    mapping_pass = all(rank_pass(value) for value in mapping.values())
    grammar_pass = all(rank_pass(value) for value in grammar.values())
    decoder_pass = all(
        rank_pass(decoders[name])
        for name in ("common_9_ordered", "common_3_ordered", "common_2_ordered")
    )
    calibration_pass = all(
        calibration[name]["oof_r2"] > 0
        for name in ("common_9_ordered", "common_3_ordered", "common_2_ordered")
    )
    common_nine_scores = [
        float(row["protocols"]["common_9_ordered"]["semantic_imageward_score"])
        for row in expanded
    ]
    common_nine_sd = (
        float(statistics.stdev(common_nine_scores))
        if len(common_nine_scores) > 1
        else 0.0
    )
    equivalence_threshold = 0.25 * common_nine_sd
    equivalence_checks: dict[str, Any] = {}
    for family, effects in (
        ("prompted_vs_posthoc", bridge),
        ("mapping_and_lexeme", mapping),
        ("legacy_vs_common_grammar", grammar),
    ):
        for name, effect in effects.items():
            low, high = effect["paired_difference"]["ci95"]
            equivalent = paired_ci_within_equivalence_band(
                effect, equivalence_threshold
            )
            equivalence_checks[f"{family}.{name}"] = {
                "ci95": [low, high],
                "equivalent": equivalent,
            }
    coordinate_equivalence_pass = bool(
        equivalence_threshold > 0
        and equivalence_checks
        and all(value["equivalent"] for value in equivalence_checks.values())
    )
    full = bool(
        len(expanded) >= 70
        and bridge_pass
        and mapping_pass
        and grammar_pass
        and decoder_pass
        and calibration_pass
        and coordinate_equivalence_pass
    )
    rank_only = bool(
        not full and len(expanded) >= 70 and bridge_pass and mapping_pass
    )
    level = 1 if full else 2 if rank_only else 3
    return {
        "title": "Truth Audit 5 — Common-template Protocol Granularity Bridge",
        "status": "completed",
        "n": len(expanded),
        "failed": failed,
        "prompted_vs_posthoc": bridge,
        "mapping_and_lexeme": mapping,
        "legacy_vs_common_grammar": grammar,
        "posthoc_distribution_js": js,
        "hard_semantic_agreement": hard_semantic_agreement,
        "legacy_sa_decoder_transfer": decoders,
        "behavior_decoder_transfer": behavior_decoders,
        "crossfit_affine_calibration": calibration,
        "absolute_coordinate_equivalence": {
            "status": "posthoc_sensitivity",
            "reference": "0.25 * SD(common_9_ordered semantic score)",
            "common_9_sd": common_nine_sd,
            "threshold": equivalence_threshold,
            "passed": coordinate_equivalence_pass,
            "comparisons": equivalence_checks,
        },
        "protocol_gate": {
            "level": level,
            "full_protocol_invariant_bridge": full,
            "rank_bridge_only": rank_only,
            "components": {
                "prompted_posthoc": bridge_pass,
                "mapping": mapping_pass,
                "grammar": grammar_pass,
                "decoder": decoder_pass,
                "calibration": calibration_pass,
                "coordinate_equivalence": coordinate_equivalence_pass,
            },
            "classification": (
                "full cross-protocol verbal-SA coordinate bridge"
                if full
                else "rank bridge with protocol-dependent offsets/scales"
                if rank_only
                else "report protocol materially constructs the verbal-SA measurement"
            ),
        },
    }


def write_truth_gate_and_controlled_skips(output_root: Path) -> dict[str, Any]:
    import json

    behavior = json.loads((output_root / BEHAVIOR_DIR / "summary.json").read_text())
    matched = json.loads(
        (output_root / MATCHED_GROUNDING_DIR / "summary.json").read_text()
    )
    protocol = json.loads(
        (output_root / GRANULARITY_DIR / "summary.json").read_text()
    )
    components = {
        "counterfactual_behavior_direction": bool(
            behavior["behavior_direction_gate"]["passed"]
        ),
        "matched_behavior_target": bool(
            matched["behavior_target_reliability"]["gate_passed"]
        ),
        "verbal_sa_behavior_alignment": bool(
            matched["verbal_sa_alignment"]["gate_passed"]
        ),
        "full_protocol_bridge": bool(
            protocol["protocol_gate"]["full_protocol_invariant_bridge"]
        ),
    }
    passed = all(components.values())
    gate = {
        "title": "Truth Audit 6 — Grounded Causal-target Gate",
        "status": "completed",
        "passed": passed,
        "components": components,
        "rule": "behavior direction, matched deletion/replacement target, verbal-SA incremental alignment, and full common-template protocol bridge must all pass",
        "classification": (
            "behavior-grounded protocol-invariant verbal-SA target"
            if passed
            else "no behavior-grounded protocol-invariant causal verbal-SA target"
        ),
    }
    directory = output_root / GATE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(directory / "cohort_manifest.json", {"components": components})
    write_jsonl_atomic(directory / "results.jsonl", [{"status": "completed", **gate}])
    write_experiment_summary(directory, gate)
    if passed:
        raise RuntimeError(
            "Grounded causal-target gate passed; blockwise tracing/subspace implementation is required"
        )
    reason = "Grounded causal-target gate failed; causal tracing would target an unvalidated self-report coordinate"
    for name, title in (
        (TRACING_DIR, "Grounded Blockwise Causal Tracing"),
        (SUBSPACE_DIR, "Grounded Low-rank Formation Subspace"),
    ):
        target = output_root / name
        target.mkdir(parents=True, exist_ok=True)
        atomic_write_json(target / "cohort_manifest.json", {"cases": [], "reason": reason})
        write_jsonl_atomic(target / "results.jsonl", [])
        write_experiment_summary(
            target, {"title": title, "status": "skipped", "n": 0, "reason": reason}
        )
    return gate


def write_truth_audit_report(output_root: Path) -> dict[str, Any]:
    import json

    summaries = {
        "counterfactual": json.loads(
            (output_root / BEHAVIOR_DIR / "summary.json").read_text()
        ),
        "matched_grounding": json.loads(
            (output_root / MATCHED_GROUNDING_DIR / "summary.json").read_text()
        ),
        "history_factorial": json.loads(
            (output_root / FACTORIAL_DIR / "summary.json").read_text()
        ),
        "answer_protocol": json.loads(
            (output_root / ANSWER_PROTOCOL_DIR / "summary.json").read_text()
        ),
        "granularity": json.loads(
            (output_root / GRANULARITY_DIR / "summary.json").read_text()
        ),
        "gate": json.loads((output_root / GATE_DIR / "summary.json").read_text()),
        "tracing": json.loads((output_root / TRACING_DIR / "summary.json").read_text()),
        "subspace": json.loads((output_root / SUBSPACE_DIR / "summary.json").read_text()),
    }
    history_grounding_path = (
        output_root
        / "09_history_conditioned_fixed_answer_deletion"
        / "summary.json"
    )
    if history_grounding_path.is_file():
        summaries["history_grounding"] = json.loads(
            history_grounding_path.read_text()
        )
    rows: list[dict[str, Any]] = []

    def add(name: str, kind: str, effect: dict[str, Any], key: str = "mean") -> None:
        bootstrap = effect.get("spearman_item_bootstrap")
        if bootstrap is not None:
            ci = bootstrap["ci95"]
            value = effect.get("spearman")
            metric = "spearman"
        else:
            ci = effect.get("ci95", [None, None])
            value = effect.get(key)
            metric = key
        rows.append(
            {
                "result": name,
                "type": kind,
                "metric": metric,
                "n": effect.get("n"),
                "unique_items": effect.get("unique_items"),
                "effect": value,
                "ci_low": ci[0],
                "ci_high": ci[1],
                "status": "reported",
            }
        )

    def add_scalar(
        name: str,
        kind: str,
        metric: str,
        value: Any,
        *,
        n: int | None = None,
        unique_items: int | None = None,
        status: str = "reported",
    ) -> None:
        rows.append(
            {
                "result": name,
                "type": kind,
                "metric": metric,
                "n": n,
                "unique_items": unique_items,
                "effect": value,
                "ci_low": None,
                "ci_high": None,
                "status": status,
            }
        )

    counter = summaries["counterfactual"]
    add(
        "SA vs counterfactual use (raw)",
        "behavior-report association",
        counter["verbal_sa_alignment"]["raw"],
    )
    add(
        "SA vs counterfactual use (residual)",
        "behavior-report association",
        counter["verbal_sa_alignment"]["training_fold_nuisance_residualized"],
    )
    add(
        "L18 PANL direction vs counterfactual use residual",
        "behavior direction",
        counter["behavior_direction_oof"],
    )
    add_scalar(
        "L18 PANL direction vs counterfactual use residual",
        "behavior direction",
        "oof_r2",
        counter["behavior_direction_oof"]["r2"],
        n=counter["behavior_direction_oof"]["n"],
        unique_items=counter["behavior_direction_oof"]["unique_items"],
        status="failed_gate",
    )
    add(
        "Easy-to-hard evidence change in behavior source use",
        "paired evidence perturbation",
        counter["paired_evidence_perturbation"]["behavior_change"],
    )
    add(
        "Easy-to-hard evidence change in verbal SA",
        "paired evidence perturbation",
        counter["paired_evidence_perturbation"]["verbal_sa_change"],
    )
    add(
        "Behavior-change vs SA-change alignment",
        "paired evidence perturbation",
        counter["paired_evidence_perturbation"]["change_alignment"],
    )
    matched = summaries["matched_grounding"]
    add(
        "Deletion vs replacement behavior target",
        "behavior target reliability",
        matched["behavior_target_reliability"]["delete_vs_replace_residual"],
    )
    add(
        "Deletion source use vs SA",
        "behavior-report association",
        matched["verbal_sa_alignment"]["deletion_residual"],
    )
    add(
        "Replacement source use vs SA",
        "behavior-report association",
        matched["verbal_sa_alignment"]["replacement_residual"],
    )
    for target in ("deletion", "replacement"):
        add_scalar(
            f"SA incremental prediction of {target} source use",
            "behavior-report incremental prediction",
            "delta_oof_r2",
            matched["verbal_sa_alignment"]["oof_incremental_r2"][target][
                "delta_oof_r2"
            ],
            n=matched["verbal_sa_alignment"][f"{target}_residual"]["n"],
            unique_items=matched["verbal_sa_alignment"][f"{target}_residual"][
                "unique_items"
            ],
        )
    factorial = summaries["history_factorial"]["total_effect_factorial"]
    add(
        "History modality -> verbal SA",
        "history total effect",
        factorial["verbal_sa"]["modality_main"],
    )
    add(
        "History answer side -> answer margin",
        "history total effect",
        factorial["answer_margin"]["prior_answer_main"],
    )
    for name, outcome, term in (
        ("History answer side -> verbal SA", "verbal_sa", "prior_answer_main"),
        ("History modality -> old z_SA", "z_sa", "modality_main"),
        ("History modality -> answer margin", "answer_margin", "modality_main"),
        ("History modality x answer side -> answer margin", "answer_margin", "interaction"),
    ):
        add(name, "history total effect", factorial[outcome][term])
    answer_protocol = summaries["answer_protocol"]
    add(
        "Joint-report minus answer-only prior-answer effect",
        "report-request effect on answer",
        answer_protocol["joint_minus_answer_only_factor_effect"]["answer_margin"][
            "prior_answer_main"
        ],
    )
    add(
        "Joint-report minus answer-only modality effect",
        "report-request effect on answer",
        answer_protocol["joint_minus_answer_only_factor_effect"]["answer_margin"][
            "modality_main"
        ],
    )
    add(
        "Joint-report minus answer-only interaction",
        "report-request effect on answer",
        answer_protocol["joint_minus_answer_only_factor_effect"]["answer_margin"][
            "interaction"
        ],
    )
    granularity = summaries["granularity"]
    for name, effect in granularity["prompted_vs_posthoc"].items():
        add(name, "protocol bridge", effect)
        add(
            f"{name} offset",
            "protocol coordinate shift",
            effect["paired_difference"],
        )
    for family in ("mapping_and_lexeme", "legacy_vs_common_grammar"):
        for name, effect in granularity[family].items():
            add(name, "protocol bridge", effect)
            add(
                f"{name} offset",
                "protocol coordinate shift",
                effect["paired_difference"],
            )
    add_scalar(
        "Strict four-way endpoint-matched History cohort",
        "sample diagnostic",
        "n",
        summaries["history_factorial"]["strict_four_way_endpoint_matched_n"],
        n=summaries["history_factorial"]["strict_four_way_endpoint_matched_n"],
        unique_items=summaries["history_factorial"]["strict_four_way_endpoint_matched_n"],
    )
    for component, passed in summaries["gate"]["components"].items():
        add_scalar(
            component,
            "grounded causal-target gate component",
            "passed",
            int(bool(passed)),
            status="passed" if passed else "failed",
        )
    history_grounding = summaries.get("history_grounding")
    if history_grounding is not None:
        add(
            "Endpoint-matched History -> verbal SA",
            "history grounded comparison",
            history_grounding["old_verbal_sa_history_effect"],
        )
        for protocol_name in ("joint_report", "answer_only"):
            for method in ("deletion", "replacement"):
                add(
                    f"{protocol_name} History -> {method} source sensitivity",
                    "history grounded comparison",
                    history_grounding["protocols"][protocol_name][
                        "delta_history"
                    ][method],
                )
        for method in ("deletion", "replacement"):
            add(
                f"Joint minus answer-only History effect ({method})",
                "report-protocol interaction on source sensitivity",
                history_grounding["joint_minus_answer_only_history_effect"][
                    method
                ],
            )
    analysis = output_root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(analysis / "core_table.csv", rows)
    payload = {"status": "completed", "experiments": summaries}
    atomic_write_json(analysis / "final_analysis.json", payload)

    raw = counter["verbal_sa_alignment"]["raw"]
    residual = counter["verbal_sa_alignment"]["training_fold_nuisance_residualized"]
    deletion = matched["verbal_sa_alignment"]["deletion_residual"]
    replacement = matched["verbal_sa_alignment"]["replacement_residual"]
    behavior_reliability = matched["behavior_target_reliability"]
    history = summaries["history_factorial"]["total_effect_factorial"]
    protocol_gate = granularity["protocol_gate"]
    answer_request = answer_protocol["joint_minus_answer_only_factor_effect"][
        "answer_margin"
    ]["prior_answer_main"]
    answer_request_modality = answer_protocol[
        "joint_minus_answer_only_factor_effect"
    ]["answer_margin"]["modality_main"]
    answer_request_interaction = answer_protocol[
        "joint_minus_answer_only_factor_effect"
    ]["answer_margin"]["interaction"]
    ternary_mapping = granularity["mapping_and_lexeme"][
        "ternary_ordered_vs_reversed"
    ]["paired_difference"]
    binary_mapping = granularity["mapping_and_lexeme"][
        "binary_ordered_vs_reversed"
    ]["paired_difference"]
    ternary_bridge = granularity["prompted_vs_posthoc"][
        "ternary_prompted_vs_posthoc"
    ]
    binary_bridge = granularity["prompted_vs_posthoc"][
        "binary_prompted_vs_posthoc_conditional"
    ]
    paired_evidence = counter["paired_evidence_perturbation"]

    def rho(effect: dict[str, Any]) -> str:
        low, high = effect["spearman_item_bootstrap"]["ci95"]
        return f"rho={effect['spearman']:.3f}, 95% CI [{low:.3f}, {high:.3f}], n={effect['n']}"

    def mean(effect: dict[str, Any]) -> str:
        low, high = effect["ci95"]
        return f"{effect['mean']:+.3f}, 95% CI [{low:+.3f}, {high:+.3f}], n={effect['n']}"

    lines = [
        "# Behavior-grounded Source Attribution Truth Audit",
        "",
        "## Main answer",
        "",
        "History changes both the population-level verbal report and at least one answer-only behavioral source-sensitivity measure, so the effect is not purely verbal. However, verbal-SA changes do not track behavioral changes item by item, and the absolute report is strongly protocol-dependent. The supported interpretation is therefore a context-conditioned self-report coupled to a coarse population-level source-sensitivity shift, not a validated faithful or causal per-case meter of source use.",
        "",
        f"- Counterfactual use vs verbal SA, raw: {rho(raw)}.",
        f"- Counterfactual use vs verbal SA after training-fold control for answer side, difficulty, prior strength: {rho(residual)}.",
        f"- Matched deletion vs replacement target reliability: {rho(behavior_reliability['delete_vs_replace_residual'])}; sign agreement={behavior_reliability['residual_sign_agreement']:.3f}.",
        f"- Matched deletion target vs verbal SA residual: {rho(deletion)}.",
        f"- Matched replacement target vs verbal SA residual: {rho(replacement)}.",
        f"- SA incremental OOF R²: deletion={matched['verbal_sa_alignment']['oof_incremental_r2']['deletion']['delta_oof_r2']:+.4f}; replacement={matched['verbal_sa_alignment']['oof_incremental_r2']['replacement']['delta_oof_r2']:+.4f}.",
        f"- The L18 behavior-direction readout has rho={counter['behavior_direction_oof']['spearman']:.3f} but OOF R²={counter['behavior_direction_oof']['r2']:+.3f}; it fails the joint direction gate.",
        f"- Across paired easy-to-hard evidence changes, verbal SA moves by {mean(paired_evidence['verbal_sa_change'])}, behavior source use by {mean(paired_evidence['behavior_change'])}, and their change alignment is {rho(paired_evidence['change_alignment'])}.",
        "",
        "## History construction",
        "",
        f"- History modality main effect on verbal SA: {mean(history['verbal_sa']['modality_main'])}.",
        f"- Replayed-answer-side main effect on verbal SA: {mean(history['verbal_sa']['prior_answer_main'])}.",
        f"- History modality main effect on the old L18 z_SA coordinate: {mean(history['z_sa']['modality_main'])}.",
        f"- History modality main effect on answer margin: {mean(history['answer_margin']['modality_main'])}; modality x replayed-answer interaction: {mean(history['answer_margin']['interaction'])}.",
        f"- Replayed-answer-side main effect on answer margin: {mean(history['answer_margin']['prior_answer_main'])}.",
        f"- History congruence effect on verbal SA: {mean(history['verbal_sa']['congruence_main'])}.",
        f"- The strict four-cell endpoint-matched subset contains only n={summaries['history_factorial']['strict_four_way_endpoint_matched_n']}; all-30 estimates are total effects and may include answer-mediated paths.",
        "",
        "## History-grounded behavior",
        "",
        *(
            [
                f"- Existing endpoint-matched verbal-SA IF−TF effect: {mean(history_grounding['old_verbal_sa_history_effect'])}.",
                f"- Joint-report source sensitivity: deletion {mean(history_grounding['protocols']['joint_report']['delta_history']['deletion'])}; replacement {mean(history_grounding['protocols']['joint_report']['delta_history']['replacement'])}.",
                f"- Answer-only source sensitivity (no SA request): deletion {mean(history_grounding['protocols']['answer_only']['delta_history']['deletion'])}; replacement {mean(history_grounding['protocols']['answer_only']['delta_history']['replacement'])}.",
                f"- Answer-only deletion/replacement target reliability: {rho(history_grounding['protocols']['answer_only']['behavior_target_reliability']['context_delete_vs_replace'])}; sign agreement={history_grounding['protocols']['answer_only']['behavior_target_reliability']['context_sign_agreement']:.3f}.",
                f"- Item-level answer-only behavior↔verbal-SA alignment: deletion {rho(history_grounding['protocols']['answer_only']['delta_behavior_vs_delta_sa']['deletion'])}; replacement {rho(history_grounding['protocols']['answer_only']['delta_behavior_vs_delta_sa']['replacement'])}.",
                f"- Joint−answer-only protocol interaction: deletion {mean(history_grounding['joint_minus_answer_only_history_effect']['deletion'])}; replacement {mean(history_grounding['joint_minus_answer_only_history_effect']['replacement'])}.",
                f"- Classification: {history_grounding['classification']}; strict answer-only cross-method gate={history_grounding['grounded_gate_passed']}.",
                "",
            ]
            if history_grounding is not None
            else ["- Not run.", ""]
        ),
        "## Report protocol",
        "",
        f"- Common-template protocol gate level: {protocol_gate['level']} — {protocol_gate['classification']}.",
        f"- Prompted vs post-hoc rank bridge: ternary {rho(ternary_bridge)}; binary {rho(binary_bridge)}. Their paired offsets are {mean(ternary_bridge['paired_difference'])} and {mean(binary_bridge['paired_difference'])}, respectively.",
        f"- Joint SA-report request minus answer-only prior-answer effect on answer margin: {mean(answer_request)}.",
        f"- Joint-report minus answer-only modality effect: {mean(answer_request_modality)}; interaction difference: {mean(answer_request_interaction)}. This is the effect of the complete report-request prompt bundle, not a single isolated phrase.",
        f"- Reversing the numeric label mapping shifts the mapped ternary score by {mean(ternary_mapping)} and the binary score by {mean(binary_mapping)}.",
        f"- Ordered/reversed hard semantic agreement is {granularity['hard_semantic_agreement']['ternary_ordered_vs_reversed']:.3f} (ternary) and {granularity['hard_semantic_agreement']['binary_ordered_vs_reversed']:.3f} (binary).",
        f"- Grounded causal-target gate passed: **{summaries['gate']['passed']}**.",
        f"- Gate components: {', '.join(f'{name}={value}' for name, value in summaries['gate']['components'].items())}.",
        "",
        "## Interpretation limits",
        "",
        "- Counterfactual source use is operationalized by fixed-answer deletion/replacement effects, not by an unverifiable introspective ground truth.",
        "- History total effects permit answer-mediated paths; endpoint-matched sensitivities are post-treatment selected and are not promoted to pure direct effects.",
        "- Protocol rank agreement does not imply coordinate equivalence or causal mediation.",
        f"- Grounded tracing: {summaries['tracing']['status']}; grounded subspace: {summaries['subspace']['status']}.",
        "",
    ]
    from layer_metacognition.hidden_state_store import atomic_write_text

    atomic_write_text(analysis / "FINAL_ANALYSIS.md", "\n".join(lines))
    return payload
