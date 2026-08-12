"""Natural-formation, semantic-specificity, and cue-direction experiments.

This module is deliberately downstream-only.  It reads the completed v4 /
joint / answer_basis_9 artifacts and writes a separate
``stage3_sa_mechanism`` tree without mutating Stage 1--3 inputs.
"""

from __future__ import annotations

import hashlib
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr

from confidence_test.inference_extension import ASSISTANT_ANSWER_PREFILL
from confidence_test.joint_answer_source_extension import JointAnswerSourceGenerationResult
from confidence_test.source_attribution_schema import (
    SOURCE_ATTRIBUTION_MIDPOINTS,
    gather_source_class_logits,
    source_distribution,
)
from confidence_test.source_attribution_variants import get_source_prompt_variant
from confidence_test.prompt_utils import STAGE1_TEXT_ANSWER_PROMPT
from layer_metacognition.hidden_state_store import (
    append_jsonl,
    atomic_write_json,
    atomic_write_text,
    load_jsonl,
)
from layer_metacognition.probe.prompts import IMAGE_ONLY_ANSWER_PROMPT
from layer_metacognition.steering.decision_side_steering import BaselineHiddenStateRepository

from .core import (
    FoldDirection,
    SAFormationArtifacts,
    SAOOFDirectionRepository,
    atomic_save_npz,
    canonical_message_hash,
    coordinate_delta,
    assert_policy_no_verbal_sa,
    item_cluster_bootstrap,
    load_baseline_rows,
    orthogonal_equal_norm_control,
    paired_effect_summary,
    read_json,
    write_experiment_summary,
    write_csv_atomic,
    write_jsonl_atomic,
)
from .followup import (
    _balanced_unique_cases,
    _direction_eval_context,
    _history_branch_messages,
    _item_mean_summary,
)
from .runtime import (
    SOURCE_CHOICE_PROMPT,
    Stage3Runtime,
    assistant_message,
    build_no_history_messages,
    full_prompt,
    image_content,
    prepare_exact_generated_measurement,
    prepare_measurement,
    prepare_policy_measurement,
    text_content,
)


OLD_AUDIT_DIR = "01_old_direction_natural_audit"
REMAP_DIR = "02_label_remapping"
HISTORY_RELEVANCE_DIR = "03_relevant_irrelevant_history"
CUE_DIRECTIONS_DIR = "04_natural_cue_directions"
MEDIATION_DIR = "05_old_direction_mediation"
POLICY_DIR = "06_policy_transfer"

HISTORY_BRANCHES = ("text_at", "text_ai", "image_at", "image_ai", "no_history")
SEMANTIC_MIDPOINTS = tuple(float(value) for value in SOURCE_ATTRIBUTION_MIDPOINTS)
SEMANTIC_DESCRIPTIONS = (
    "The answer was based almost entirely on the text clue.",
    "The answer was based mainly on the text clue.",
    "The answer was based more on the text clue than on the image.",
    "The answer was based slightly more on the text clue.",
    "The answer was based on the text clue and the image to a similar extent.",
    "The answer was based slightly more on the image.",
    "The answer was based more on the image than on the text clue.",
    "The answer was based mainly on the image.",
    "The answer was based almost entirely on the image.",
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


def _correlation_summary(
    rows: Sequence[dict[str, Any]], x_key: str, y_key: str
) -> dict[str, Any]:
    valid = [
        row
        for row in rows
        if row.get(x_key) is not None
        and row.get(y_key) is not None
        and np.isfinite(float(row[x_key]))
        and np.isfinite(float(row[y_key]))
    ]
    x = np.asarray([row[x_key] for row in valid], dtype=np.float64)
    y = np.asarray([row[y_key] for row in valid], dtype=np.float64)
    if len(valid) < 3:
        return {"n": len(valid), "pearson": None, "spearman": None, "spearman_item_bootstrap": None}
    spearman_boot = item_cluster_bootstrap(
        valid,
        lambda sample: spearmanr(
            [float(row[x_key]) for row in sample],
            [float(row[y_key]) for row in sample],
        ).statistic,
    )
    return {
        "n": len(valid),
        "unique_items": len({str(row["item_id"]) for row in valid}),
        "pearson": float(pearsonr(x, y).statistic),
        "spearman": float(spearmanr(x, y).statistic),
        "spearman_item_bootstrap": spearman_boot,
    }


def _quintile_summary(rows: Sequence[dict[str, Any]], x_key: str, y_key: str) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: float(row[x_key]))
    groups = np.array_split(np.arange(len(ordered)), 5)
    values = [
        {
            "quintile": index + 1,
            "n": len(indices),
            "mean_coordinate": statistics.fmean(float(ordered[i][x_key]) for i in indices),
            "mean_sa": statistics.fmean(float(ordered[i][y_key]) for i in indices),
        }
        for index, indices in enumerate(groups)
    ]
    return {
        "bins": values,
        "monotonic": all(values[i]["mean_sa"] <= values[i + 1]["mean_sa"] for i in range(4)),
    }


def _ols_old_increment(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Report old-coordinate incremental fit after fold-standardized Ridge z."""

    def fit(sample: Sequence[dict[str, Any]]) -> tuple[float, float, float]:
        y = np.asarray([row["sa"] for row in sample], dtype=np.float64)
        zr = np.asarray([row["z_ridge_std"] for row in sample], dtype=np.float64)
        zo = np.asarray([row["z_old_std"] for row in sample], dtype=np.float64)
        reduced = np.column_stack([np.ones(len(sample)), zr])
        full = np.column_stack([np.ones(len(sample)), zr, zo])
        pred_reduced = reduced @ np.linalg.lstsq(reduced, y, rcond=None)[0]
        coef = np.linalg.lstsq(full, y, rcond=None)[0]
        pred_full = full @ coef
        denominator = float(np.sum((y - y.mean()) ** 2))
        r2_reduced = 1.0 - float(np.sum((y - pred_reduced) ** 2)) / denominator
        r2_full = 1.0 - float(np.sum((y - pred_full) ** 2)) / denominator
        return float(coef[2]), r2_reduced, r2_full

    coefficient, reduced_r2, full_r2 = fit(rows)
    boot = item_cluster_bootstrap(rows, lambda sample: fit(sample)[0])
    return {
        "model": "SA ~ 1 + fold-standardized Ridge-z + fold-standardized old-z",
        "old_z_coefficient": coefficient,
        "old_z_coefficient_item_bootstrap": boot,
        "ridge_only_r2": reduced_r2,
        "ridge_plus_old_r2": full_r2,
        "incremental_r2": full_r2 - reduced_r2,
    }


@dataclass
class BaselineGeometry:
    rows: list[dict[str, Any]]
    hidden: np.ndarray
    row_by_case: dict[str, dict[str, Any]]
    means: dict[str, dict[int, float]]


def load_baseline_geometry(
    artifacts: SAFormationArtifacts,
    stage3_root: Path,
    followup_root: Path,
) -> BaselineGeometry:
    rows = load_baseline_rows(artifacts)
    hidden_repo = BaselineHiddenStateRepository(artifacts.experiment_dir)
    hidden = np.stack(
        [hidden_repo.get(row["manifest"], 18, "panl") for row in rows]
    ).astype(np.float64, copy=False)
    repositories = {
        "old": SAOOFDirectionRepository(followup_root / "directions" / "old_oof"),
        "ridge": SAOOFDirectionRepository(stage3_root / "directions"),
        "old_perp_ridge": SAOOFDirectionRepository(
            followup_root / "directions" / "old_perp_ridge_oof"
        ),
    }
    means: dict[str, dict[int, float]] = {kind: {} for kind in repositories}
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    for kind, repository in repositories.items():
        for fold in range(5):
            direction = repository.get(fold)
            means[kind][fold] = float(np.mean(hidden[folds != fold] @ direction.d_unit))
    for index, row in enumerate(rows):
        fold = int(row["fold"])
        for kind, repository in repositories.items():
            direction = repository.get(fold)
            raw = float(hidden[index] @ direction.d_unit)
            row[f"z_{kind}_raw"] = raw
            row[f"z_{kind}_std"] = (raw - means[kind][fold]) / direction.sigma_z
        row["ridge_prediction"] = repositories["ridge"].get(fold).predict(hidden[index])
    return BaselineGeometry(
        rows=rows,
        hidden=hidden,
        row_by_case={str(row["case_id"]): row for row in rows},
        means=means,
    )


def _evidence_pairs(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        prefix = str(row["condition"]).rsplit("_", 1)[0]
        grouped.setdefault((str(row["item_id"]), int(row["prior_index"]), prefix), {})[
            str(row["difficulty"])
        ] = row
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
                "endpoint_matched": easy["final_answer"] == hard["final_answer"],
                "delta_sa_easy_minus_hard": float(easy["sa"] - hard["sa"]),
                "delta_z_old_easy_minus_hard": float(
                    easy["z_old_std"] - hard["z_old_std"]
                ),
                "delta_z_ridge_easy_minus_hard": float(
                    easy["z_ridge_std"] - hard["z_ridge_std"]
                ),
            }
        )
    return pairs


def run_old_natural_cpu(
    artifacts: SAFormationArtifacts,
    stage3_root: Path,
    followup_root: Path,
    output_root: Path,
) -> tuple[BaselineGeometry, dict[str, Any]]:
    directory = output_root / OLD_AUDIT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    geometry = load_baseline_geometry(artifacts, stage3_root, followup_root)
    natural_rows = [
        {
            "case_id": row["case_id"],
            "item_id": row["item_id"],
            "prior_index": row["prior_index"],
            "condition": row["condition"],
            "difficulty": row["difficulty"],
            "fold": row["fold"],
            "sa": row["sa"],
            "z_old_raw": row["z_old_raw"],
            "z_old_std": row["z_old_std"],
            "z_ridge_raw": row["z_ridge_raw"],
            "z_ridge_std": row["z_ridge_std"],
            "z_old_perp_ridge_raw": row["z_old_perp_ridge_raw"],
            "z_old_perp_ridge_std": row["z_old_perp_ridge_std"],
            "ridge_prediction": row["ridge_prediction"],
        }
        for row in geometry.rows
    ]
    evidence = _evidence_pairs(geometry.rows)
    write_jsonl_atomic(directory / "results.jsonl", natural_rows)
    write_jsonl_atomic(directory / "evidence_pairs.jsonl", evidence)
    natural = _correlation_summary(natural_rows, "z_old_std", "sa")
    natural["quintiles"] = _quintile_summary(natural_rows, "z_old_std", "sa")
    natural["after_ridge_z"] = _ols_old_increment(natural_rows)
    natural["association_supported"] = bool(
        natural["spearman_item_bootstrap"]["ci95"][0] is not None
        and natural["spearman_item_bootstrap"]["ci95"][0] > 0
    )
    evidence_summary = {
        "all_pairs": {
            "n": len(evidence),
            "sa": _item_mean_summary(evidence, "delta_sa_easy_minus_hard"),
            "old_z": _item_mean_summary(evidence, "delta_z_old_easy_minus_hard"),
            "ridge_z": _item_mean_summary(evidence, "delta_z_ridge_easy_minus_hard"),
        },
        "endpoint_matched": {
            "n": sum(row["endpoint_matched"] for row in evidence),
            "sa": _item_mean_summary(
                [row for row in evidence if row["endpoint_matched"]],
                "delta_sa_easy_minus_hard",
            ),
            "old_z": _item_mean_summary(
                [row for row in evidence if row["endpoint_matched"]],
                "delta_z_old_easy_minus_hard",
            ),
        },
    }
    evidence_summary["cue_supported"] = bool(
        evidence_summary["all_pairs"]["sa"]["ci95"][0] > 0
        and evidence_summary["all_pairs"]["old_z"]["ci95"][0] > 0
    )
    partial = {
        "title": "Experiment 1 — Old-direction Natural Formation Audit",
        "status": "awaiting_history_replay",
        "n": len(natural_rows),
        "natural_association": natural,
        "evidence_easy_minus_hard": evidence_summary,
    }
    atomic_write_json(directory / "summary_cpu.json", partial)
    atomic_write_json(
        directory / "cohort_manifest.json",
        {
            "natural_case_count": len(natural_rows),
            "unique_items": len({row["item_id"] for row in natural_rows}),
            "history_source": str(followup_root / "02_history_exact_factorial" / "results_nocache.jsonl"),
            "fold_standardization": "for held-out fold f, mean and SD use only items outside f projected on d_old[f]",
        },
    )
    return geometry, partial


def _project_state(
    hidden: np.ndarray,
    fold: int,
    repositories: dict[str, SAOOFDirectionRepository],
    means: dict[str, dict[int, float]],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for kind, repository in repositories.items():
        direction = repository.get(fold)
        raw = float(hidden @ direction.d_unit)
        values[f"z_{kind}_raw"] = raw
        values[f"z_{kind}_std"] = (raw - means[kind][fold]) / direction.sigma_z
    return values


def run_old_history_replay(
    runtime: Stage3Runtime,
    stage3_root: Path,
    followup_root: Path,
    output_root: Path,
    geometry: BaselineGeometry,
    *,
    deadline: Callable[[], None],
) -> dict[str, Any]:
    directory = output_root / OLD_AUDIT_DIR
    history_source = followup_root / "02_history_exact_factorial" / "results_nocache.jsonl"
    source_rows = [row for row in load_jsonl(history_source) if row.get("status") == "completed"]
    result_path = directory / "history_results.jsonl"
    existing = {row["intervention_key"] for row in _latest_rows(result_path)}
    repositories = {
        "old": SAOOFDirectionRepository(followup_root / "directions" / "old_oof"),
        "ridge": SAOOFDirectionRepository(stage3_root / "directions"),
        "old_perp_ridge": SAOOFDirectionRepository(
            followup_root / "directions" / "old_perp_ridge_oof"
        ),
    }
    hidden_dir = directory / "history_hidden"
    for source_row in source_rows:
        deadline()
        key = f"old_history_exact|{source_row['case_id']}"
        if key in existing:
            continue
        base = {
            "intervention_key": key,
            "experiment": "old_history_exact_replay",
            "case_id": source_row["case_id"],
            "item_id": source_row["item_id"],
            "prior_index": source_row["prior_index"],
            "condition": source_row["condition"],
            "fold": int(source_row["fold"]),
        }

        def execute() -> dict[str, Any]:
            fold = int(source_row["fold"])
            baseline = geometry.row_by_case[str(source_row["case_id"])]
            branch_results: dict[str, Any] = {}
            matrices: list[np.ndarray] = []
            for branch in HISTORY_BRANCHES:
                messages = _history_branch_messages(runtime, baseline, branch)
                generated = JointAnswerSourceGenerationResult(**source_row["branches"][branch]["pass1"])
                prepared = prepare_exact_generated_measurement(
                    runtime.generator,
                    messages,
                    generated,
                    assistant_text=ASSISTANT_ANSWER_PREFILL,
                )
                measured = runtime.measure(prepared, repositories["old"].get(fold))
                expected_soft = float(
                    source_row["branches"][branch]["pass1"]["source_attribution"]["soft_image_score"]
                )
                branch_results[branch] = {
                    "messages_hash": canonical_message_hash(messages),
                    "normalized_answer": generated.normalized_answer,
                    "sa": float(measured.source["soft_image_score"]),
                    "sa_replay_abs_error": abs(
                        float(measured.source["soft_image_score"]) - expected_soft
                    ),
                    **_project_state(measured.hidden, fold, repositories, geometry.means),
                }
                matrices.append(measured.hidden.astype(np.float32, copy=False))
                runtime.release_inputs(prepared)
            hidden_file = hidden_dir / f"{source_row['case_id'].replace('/', '_')}.npz"
            atomic_save_npz(
                hidden_file,
                branches=np.asarray(HISTORY_BRANCHES),
                hidden=np.stack(matrices),
            )
            return {
                **base,
                "status": "completed",
                "branches": branch_results,
                "hidden_file": str(hidden_file.relative_to(output_root)),
            }

        append_jsonl(result_path, _safe_record(base, execute))
    rows = [row for row in _latest_rows(result_path) if row.get("status") == "completed"]
    return _summarize_old_history(rows)


def _paired_history_effect(
    rows: Sequence[dict[str, Any]], left: str, right: str, outcome: str
) -> dict[str, Any]:
    paired: list[dict[str, Any]] = []
    for row in rows:
        left_row, right_row = row["branches"][left], row["branches"][right]
        if left_row["normalized_answer"] != right_row["normalized_answer"]:
            continue
        paired.append(
            {
                "item_id": row["item_id"],
                "delta": float(right_row[outcome] - left_row[outcome]),
            }
        )
    return paired_effect_summary(paired, "delta")


def _summarize_old_history(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    pairs = {
        "fixed_at_image_minus_text": ("text_at", "image_at"),
        "fixed_ai_image_minus_text": ("text_ai", "image_ai"),
        "text_at_minus_no_history": ("no_history", "text_at"),
        "image_at_minus_no_history": ("no_history", "image_at"),
        "text_ai_minus_no_history": ("no_history", "text_ai"),
        "image_ai_minus_no_history": ("no_history", "image_ai"),
    }
    for name, (left, right) in pairs.items():
        comparisons[name] = {
            outcome: _paired_history_effect(rows, left, right, outcome)
            for outcome in ("sa", "z_old_std", "z_ridge_std", "z_old_perp_ridge_std")
        }
    modality_old = [
        comparisons[name]["z_old_std"]
        for name in ("fixed_at_image_minus_text", "fixed_ai_image_minus_text")
    ]
    modality_sa = [
        comparisons[name]["sa"]
        for name in ("fixed_at_image_minus_text", "fixed_ai_image_minus_text")
    ]
    stable = [
        old["n"] >= 10
        and old["ci95"][0] is not None
        and old["ci95"][0] > 0
        and sa["ci95"][0] is not None
        and sa["ci95"][0] > 0
        for old, sa in zip(modality_old, modality_sa)
    ]
    replay_errors = [
        branch["sa_replay_abs_error"]
        for row in rows
        for branch in row["branches"].values()
    ]
    return {
        "n": len(rows),
        "branch_n": 5 * len(rows),
        "comparisons": comparisons,
        "cue_supported": any(stable),
        "cue_gate_rule": "at least one fixed-prior-answer endpoint-matched Image−Text contrast has n>=10 and positive SA and old-z CI lower bounds",
        "replay": {
            "max_sa_abs_error": max(replay_errors, default=None),
            "all_within_0.01": all(value <= 0.01 for value in replay_errors),
        },
        "no_history_pattern": _classify_no_history_pattern(comparisons),
    }


def _classify_no_history_pattern(comparisons: dict[str, Any]) -> str:
    text = comparisons["text_at_minus_no_history"]["z_old_std"]
    image = comparisons["image_at_minus_no_history"]["z_old_std"]
    if text["mean"] is None or image["mean"] is None:
        return "unavailable"
    if text["mean"] < 0 < image["mean"]:
        return "Text < NoHistory < Image (point estimates)"
    if image["mean"] > 0 and text["ci95"][0] <= 0 <= text["ci95"][1]:
        return "primarily Image-side shift"
    if text["mean"] < 0 and image["ci95"][0] <= 0 <= image["ci95"][1]:
        return "primarily Text-side shift"
    return "not a simple bidirectional/one-sided pattern"


def finalize_old_audit(output_root: Path) -> dict[str, Any]:
    directory = output_root / OLD_AUDIT_DIR
    cpu = read_json(directory / "summary_cpu.json")
    history_rows = [row for row in _latest_rows(directory / "history_results.jsonl") if row.get("status") == "completed"]
    history = _summarize_old_history(history_rows)
    association = bool(cpu["natural_association"]["association_supported"])
    evidence = bool(cpu["evidence_easy_minus_hard"]["cue_supported"])
    history_supported = bool(history["cue_supported"])
    gate_passed = association and evidence and history_supported
    summary = {
        **cpu,
        "status": "completed",
        "history_exact_replay": history,
        "natural_gate": {
            "passed": gate_passed,
            "association_supported": association,
            "evidence_moves_old_with_sa": evidence,
            "history_moves_old_with_sa": history_supported,
            "rule": "association AND Evidence cue AND at least one fixed-answer History cue",
            "classification": (
                "candidate natural formation mediator"
                if gate_passed
                else "report-output actuator unless later evidence overturns the failed natural gate"
            ),
        },
    }
    write_experiment_summary(directory, summary)
    atomic_write_json(output_root / "gate_exp1_natural.json", summary["natural_gate"])
    return summary


def label_mappings() -> dict[str, tuple[str, ...]]:
    """Return token labels indexed by semantic class (textward -> imageward)."""

    return {
        "normal_numeric": tuple(str(index) for index in range(9)),
        "reversed_numeric": tuple(str(index) for index in reversed(range(9))),
        # Seed-42 fixed permutation of unordered raw single-token labels.
        "arbitrary_tokens": ("Q", "M", "Z", "B", "T", "R", "K", "V", "F"),
    }


def semantic_class_text(labels_by_semantic: Sequence[str]) -> str:
    if len(labels_by_semantic) != 9 or len(set(labels_by_semantic)) != 9:
        raise ValueError("A semantic mapping must contain nine distinct labels")
    return "Source attribution classes:\n" + "\n".join(
        f"{label}: {description}"
        for label, description in zip(labels_by_semantic, SEMANTIC_DESCRIPTIONS)
    )


def semantic_mapping_prompt(case: Any, labels_by_semantic: Sequence[str]) -> str:
    variant = get_source_prompt_variant("answer_basis_9")
    prompt = variant.v4_joint_prompt.format(
        question=case.question,
        text_clue=case.text_clue,
        source_classes=semantic_class_text(labels_by_semantic),
    )
    return prompt.replace(
        "Do not choose class 4 merely",
        f"Do not choose class {labels_by_semantic[4]} merely",
    )


class SingleTokenSemanticAnalyzer:
    """Restricted-logit analyzer for arbitrary no-space one-token labels."""

    def __init__(self, tokenizer: Any, labels_by_semantic: Sequence[str]) -> None:
        self.tokenizer = tokenizer
        self.labels = [str(label) for label in labels_by_semantic]
        encodings = {
            label: [int(value) for value in tokenizer.encode(label, add_special_tokens=False)]
            for label in self.labels
        }
        invalid = {label: ids for label, ids in encodings.items() if len(ids) != 1}
        if invalid:
            raise ValueError(f"Remapping labels are not raw single tokens: {invalid}")
        token_ids = [ids[0] for ids in encodings.values()]
        if len(set(token_ids)) != len(token_ids):
            raise ValueError("Remapping labels collide at the token-id level")
        self.encodings = encodings

    def score_vocab_logits(
        self, vocab_logits: Any, *, raw_output: str, parsed_label: str | None
    ) -> Any:
        logits = gather_source_class_logits(vocab_logits.float(), self.encodings, self.labels)
        return source_distribution(
            logits,
            class_token_ids=self.encodings,
            raw_output=raw_output,
            parsed_label=parsed_label,
            token_diagnostics={
                "protocol": "no-space raw single-token remapping",
                "raw_encodings": self.encodings,
            },
            classes=self.labels,
            midpoints=SEMANTIC_MIDPOINTS,
        )


def _remap_context(
    runtime: Stage3Runtime,
    row: dict[str, Any],
    labels_by_semantic: Sequence[str],
) -> Any:
    case = runtime.case(row["item_id"], row["prior_index"])
    answer = str(row["baseline"]["generated"]["current_answer_result"]["normalized_answer"])
    assistant_text = f"**Answer**: {answer}\n**Source Attribution**:"
    messages = [
        {
            "role": "user",
            "content": image_content(
                str(case.conditions[row["condition"]].resolved_image_path),
                semantic_mapping_prompt(case, labels_by_semantic),
            ),
        },
        assistant_message(assistant_text),
    ]
    return prepare_measurement(
        runtime.generator,
        messages,
        assistant_text=assistant_text,
        answer=answer,
    )


def run_label_remapping(
    runtime: Stage3Runtime,
    artifacts: SAFormationArtifacts,
    followup_root: Path,
    output_root: Path,
    *,
    n_items: int,
    deadline: Callable[[], None],
) -> dict[str, Any]:
    directory = output_root / REMAP_DIR
    directory.mkdir(parents=True, exist_ok=True)
    mappings = label_mappings()
    analyzers = {
        name: SingleTokenSemanticAnalyzer(runtime.generator.tokenizer, labels)
        for name, labels in mappings.items()
    }
    baseline = load_baseline_rows(artifacts)
    cohort = _balanced_unique_cases(baseline, n_items)
    atomic_write_json(
        directory / "cohort_manifest.json",
        {
            "case_count": len(cohort),
            "one_case_per_item": True,
            "case_ids": [row["case_id"] for row in cohort],
            "mappings_semantic_text_to_image": mappings,
            "doses_sigma_units": [-2, 0, 2],
            "fixed_fields": ["case", "answer", "history(no-history)", "L18 PANL site", "old OOF direction"],
            "changed_field": "SA class-label protocol in the user prompt and restricted output vocabulary",
        },
    )
    old_repo = SAOOFDirectionRepository(followup_root / "directions" / "old_oof")
    result_path = directory / "results.jsonl"
    existing = {row["intervention_key"] for row in _latest_rows(result_path)}
    for row in cohort:
        deadline()
        key = f"label_remap|{row['case_id']}"
        if key in existing:
            continue
        base = {
            "intervention_key": key,
            "experiment": "label_remapping_semantic_specificity",
            "case_id": row["case_id"],
            "item_id": row["item_id"],
            "prior_index": row["prior_index"],
            "condition": row["condition"],
            "fold": int(row["fold"]),
        }

        def execute() -> dict[str, Any]:
            direction = old_repo.get(row["fold"])
            mapping_results: dict[str, Any] = {}
            for name, labels in mappings.items():
                prepared = _remap_context(runtime, row, labels)
                arms: dict[str, Any] = {}
                for dose in (-2, 0, 2):
                    vector = None if dose == 0 else dose * direction.sigma_z * direction.d_unit
                    measured = runtime.measure(
                        prepared,
                        direction,
                        steering_vector=vector,
                        analyzer=analyzers[name],
                    )
                    probabilities = {
                        label: float(probability)
                        for label, probability in zip(
                            labels, measured.source["class_probabilities"]
                        )
                    }
                    raw_numeric = (
                        sum(probabilities[str(index)] * index / 8.0 for index in range(9))
                        if set(labels) == {str(index) for index in range(9)}
                        else None
                    )
                    arms[str(dose)] = {
                        "semantic_imageward_score": float(measured.source["soft_image_score"]),
                        "raw_numeric_score": raw_numeric,
                        "hard_label": measured.source["hard_label"],
                        "class_probabilities_by_label": probabilities,
                        "applied_delta_sigma": measured.applied_delta_z / direction.sigma_z,
                        "injection_l2": measured.injection_l2,
                    }
                mapping_results[name] = {
                    "labels_by_semantic_class": labels,
                    "prompt_hash": prepared.prefix_hash,
                    "arms": arms,
                }
                runtime.release_inputs(prepared)
            return {**base, "status": "completed", "mappings": mapping_results}

        append_jsonl(result_path, _safe_record(base, execute))
    rows = _latest_rows(result_path)
    summary = _summarize_label_remapping(rows)
    write_experiment_summary(directory, summary)
    atomic_write_json(output_root / "gate_exp2_semantic.json", summary["semantic_gate"])
    return summary


def _summarize_label_remapping(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed"]
    effects: dict[str, Any] = {}
    for name in label_mappings():
        values: list[dict[str, Any]] = []
        for row in completed:
            arms = row["mappings"][name]["arms"]
            record = {
                "item_id": row["item_id"],
                "semantic_plus2_minus_minus2": float(
                    arms["2"]["semantic_imageward_score"]
                    - arms["-2"]["semantic_imageward_score"]
                ),
            }
            if arms["2"]["raw_numeric_score"] is not None:
                record["raw_numeric_plus2_minus_minus2"] = float(
                    arms["2"]["raw_numeric_score"]
                    - arms["-2"]["raw_numeric_score"]
                )
            values.append(record)
        semantic = paired_effect_summary(values, "semantic_plus2_minus_minus2")
        effect = {"semantic_imageward": semantic}
        if values and "raw_numeric_plus2_minus_minus2" in values[0]:
            effect["raw_numeric_upward"] = paired_effect_summary(
                values, "raw_numeric_plus2_minus_minus2"
            )
        effect["semantic_supported"] = bool(
            semantic["n"] >= 25
            and semantic["ci95"][0] is not None
            and semantic["ci95"][0] > 0
            and semantic["direction_rate"] >= 0.60
        )
        effects[name] = effect
    semantic_pass = all(effect["semantic_supported"] for effect in effects.values())
    normal = effects["normal_numeric"]["semantic_imageward"]
    reversed_semantic = effects["reversed_numeric"]["semantic_imageward"]
    reversed_numeric = effects["reversed_numeric"]["raw_numeric_upward"]
    if semantic_pass:
        classification = "SA-semantic controller across mappings"
    elif (
        normal["ci95"][0] is not None
        and normal["ci95"][0] > 0
        and reversed_semantic["ci95"][1] is not None
        and reversed_semantic["ci95"][1] < 0
        and reversed_numeric["ci95"][0] is not None
        and reversed_numeric["ci95"][0] > 0
    ):
        classification = "numeric/token-geometry actuator"
    elif effects["normal_numeric"]["semantic_supported"]:
        classification = "original-prompt/report-protocol-specific actuator"
    else:
        classification = "no stable remapping-generalized output control"
    return {
        "title": "Experiment 2 — Label-Remapping Semantic Specificity",
        "status": "completed",
        "n": len(completed),
        "failed": len(rows) - len(completed),
        "effects": effects,
        "semantic_gate": {
            "passed": semantic_pass,
            "rule": "all three mappings: n>=25, semantic +2σ−(−2σ) CI lower > 0, direction rate >= 0.60",
            "classification": classification,
        },
    }


def _history_first_turn(case: Any, condition: str, modality: str) -> dict[str, Any]:
    if modality == "text":
        prompt = STAGE1_TEXT_ANSWER_PROMPT.format(
            question=case.question,
            text_clue=case.text_clue,
        )
        return {"role": "user", "content": text_content(prompt)}
    if modality == "image":
        prompt = IMAGE_ONLY_ANSWER_PROMPT.format(question=case.question)
        return {
            "role": "user",
            "content": image_content(
                str(case.conditions[condition].resolved_image_path), prompt
            ),
        }
    raise ValueError(f"Unknown History modality: {modality}")


def build_relevance_history_messages(
    target_case: Any,
    target_condition: str,
    history_case: Any,
    history_condition: str,
    modality: str,
    prior_answer: str,
    *,
    assistant_text: str = ASSISTANT_ANSWER_PREFILL,
) -> list[dict[str, Any]]:
    """Build a History branch whose historical item can be target or donor."""

    return [
        _history_first_turn(history_case, history_condition, modality),
        assistant_message(f"**Answer**: {prior_answer}"),
        {
            "role": "user",
            "content": image_content(
                str(target_case.conditions[target_condition].resolved_image_path),
                full_prompt(target_case),
            ),
        },
        assistant_message(assistant_text),
    ]


def _history_text_length(case: Any, modality: str) -> int:
    if modality == "text":
        return len(
            STAGE1_TEXT_ANSWER_PROMPT.format(
                question=case.question, text_clue=case.text_clue
            )
        )
    return len(IMAGE_ONLY_ANSWER_PROMPT.format(question=case.question))


def choose_irrelevant_donor(
    runtime: Stage3Runtime,
    target: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    target_case = runtime.case(target["item_id"], target["prior_index"])
    eligible: list[dict[str, Any]] = []
    seen_items: set[str] = set()
    for row in sorted(
        candidates,
        key=lambda value: (
            int(value["fold"]),
            _numeric_item_key(value["item_id"]),
            int(value["prior_index"]),
        ),
    ):
        item = str(row["item_id"])
        if item == str(target["item_id"]) or item in seen_items:
            continue
        if int(row["fold"]) != int(target["fold"]):
            continue
        donor_case = runtime.case(row["item_id"], row["prior_index"])
        if row["condition"] not in donor_case.conditions:
            continue
        seen_items.add(item)
        score = sum(
            abs(_history_text_length(donor_case, modality) - _history_text_length(target_case, modality))
            for modality in ("text", "image")
        )
        eligible.append({**row, "length_mismatch": score})
    if not eligible:
        raise ValueError(f"No irrelevant donor for {target['case_id']}")
    return min(
        eligible,
        key=lambda row: (
            row["length_mismatch"],
            _numeric_item_key(row["item_id"]),
            int(row["prior_index"]),
        ),
    )


def _run_relevance_branch(
    runtime: Stage3Runtime,
    target: dict[str, Any],
    donor: dict[str, Any],
    branch: str,
    direction: FoldDirection,
) -> tuple[dict[str, Any], np.ndarray]:
    target_case = runtime.case(target["item_id"], target["prior_index"])
    prior_answer = str(target["text_answer"])
    if branch == "no_history":
        messages = build_no_history_messages(target_case, target["condition"])
    else:
        relevance, modality = branch.split("_", 1)
        history_row = target if relevance == "relevant" else donor
        history_case = runtime.case(history_row["item_id"], history_row["prior_index"])
        messages = build_relevance_history_messages(
            target_case,
            target["condition"],
            history_case,
            history_row["condition"],
            modality,
            prior_answer,
        )
    generated = runtime.generator.generate_messages(
        messages,
        target_case.answer_classes,
        max_new_tokens=48,
        use_cache=False,
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
    result = {
        "branch": branch,
        "messages_hash": canonical_message_hash(messages),
        "final_evidence_hash": canonical_message_hash(
            [[message for message in messages if message.get("role") == "user"][-1]]
        ),
        "normalized_answer": generated.normalized_answer,
        "pass1_sa": float(generated.source_attribution["soft_image_score"]),
        "pass2_sa": float(measured.source["soft_image_score"]),
        "reconstruction_soft_error": abs(
            float(generated.source_attribution["soft_image_score"])
            - float(measured.source["soft_image_score"])
        ),
        "z_old_raw": measured.z_sa,
        "generated_token_ids": generated.generated_token_ids,
        "source_token_step": generated.source_token_step,
    }
    hidden = measured.hidden.astype(np.float32, copy=False)
    runtime.release_inputs(prepared)
    return result, hidden


def run_relevant_irrelevant_history(
    runtime: Stage3Runtime,
    artifacts: SAFormationArtifacts,
    followup_root: Path,
    output_root: Path,
    geometry: BaselineGeometry,
    *,
    n_items: int,
    deadline: Callable[[], None],
) -> dict[str, Any]:
    directory = output_root / HISTORY_RELEVANCE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    baseline = load_baseline_rows(artifacts)
    candidates = _balanced_unique_cases(baseline, min(n_items + 20, len({row['item_id'] for row in baseline})))
    donors = {
        row["case_id"]: choose_irrelevant_donor(runtime, row, baseline)
        for row in candidates
    }
    branches = (
        "relevant_text",
        "relevant_image",
        "irrelevant_text",
        "irrelevant_image",
        "no_history",
    )
    atomic_write_json(
        directory / "cohort_manifest.json",
        {
            "target_completed_items": n_items,
            "candidate_count": len(candidates),
            "branches": branches,
            "common_history_answer": "target A_T in every relevant/irrelevant branch",
            "matching": [
                "same target final full-evidence turn",
                "same historical assistant-answer token",
                "same prompt templates and modality counts",
                "donor selected in same fold with closest combined Text/Image prompt length",
            ],
            "targets": [
                {
                    "case_id": row["case_id"],
                    "donor_case_id": donors[row["case_id"]]["case_id"],
                    "donor_item_id": donors[row["case_id"]]["item_id"],
                    "length_mismatch": donors[row["case_id"]]["length_mismatch"],
                }
                for row in candidates
            ],
        },
    )
    old_repo = SAOOFDirectionRepository(followup_root / "directions" / "old_oof")
    result_path = directory / "results.jsonl"
    existing_rows = _latest_rows(result_path)
    existing = {
        row["intervention_key"]
        for row in existing_rows
        if row.get("status") == "completed"
    }
    completed_count = sum(row.get("status") == "completed" for row in existing_rows)
    hidden_dir = directory / "hidden"
    for row in candidates:
        deadline()
        if completed_count >= n_items:
            break
        key = f"history_relevance|{row['case_id']}"
        if key in existing:
            continue
        donor = donors[row["case_id"]]
        base = {
            "intervention_key": key,
            "experiment": "relevant_irrelevant_history",
            "case_id": row["case_id"],
            "item_id": row["item_id"],
            "prior_index": row["prior_index"],
            "condition": row["condition"],
            "fold": int(row["fold"]),
            "prior_answer": row["text_answer"],
            "donor_case_id": donor["case_id"],
            "donor_item_id": donor["item_id"],
            "donor_length_mismatch": donor["length_mismatch"],
        }

        def execute() -> dict[str, Any]:
            direction = old_repo.get(row["fold"])
            branch_results: dict[str, Any] = {}
            matrices: list[np.ndarray] = []
            for branch in branches:
                result, hidden = _run_relevance_branch(
                    runtime, row, donor, branch, direction
                )
                raw = float(hidden.astype(np.float64) @ direction.d_unit)
                result["z_old_std"] = (
                    raw - geometry.means["old"][int(row["fold"])]
                ) / direction.sigma_z
                branch_results[branch] = result
                matrices.append(hidden)
            if len({value["final_evidence_hash"] for value in branch_results.values()}) != 1:
                raise ValueError("Relevant/irrelevant branches do not share final evidence")
            relevant_hashes = {
                branch_results[name]["messages_hash"]
                for name in ("relevant_text", "relevant_image")
            }
            irrelevant_hashes = {
                branch_results[name]["messages_hash"]
                for name in ("irrelevant_text", "irrelevant_image")
            }
            if relevant_hashes == irrelevant_hashes:
                raise ValueError("Irrelevant History unexpectedly equals relevant History")
            hidden_file = hidden_dir / f"{row['case_id'].replace('/', '_')}.npz"
            atomic_save_npz(
                hidden_file,
                branches=np.asarray(branches),
                hidden=np.stack(matrices),
            )
            final_answers = {
                name: branch_results[name]["normalized_answer"] for name in branches
            }
            return {
                **base,
                "status": "completed",
                "branches": branch_results,
                "hidden_file": str(hidden_file.relative_to(output_root)),
                "strict_four_history_endpoint_matched": len(
                    {final_answers[name] for name in branches[:4]}
                ) == 1,
            }

        result = _safe_record(base, execute)
        append_jsonl(result_path, result)
        if result.get("status") == "completed":
            completed_count += 1
    rows = _latest_rows(result_path)
    summary = _summarize_history_relevance(rows)
    write_experiment_summary(directory, summary)
    return summary


def _history_relevance_pair(
    rows: Sequence[dict[str, Any]], relevance: str, outcome: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for row in rows:
        text = row["branches"][f"{relevance}_text"]
        image = row["branches"][f"{relevance}_image"]
        if text["normalized_answer"] != image["normalized_answer"]:
            continue
        pairs.append(
            {"item_id": row["item_id"], "delta": float(image[outcome] - text[outcome])}
        )
    return pairs, paired_effect_summary(pairs, "delta")


def _summarize_history_relevance(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed"]
    effects: dict[str, Any] = {}
    pair_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for outcome in ("pass1_sa", "z_old_std"):
        effects[outcome] = {}
        for relevance in ("relevant", "irrelevant"):
            pairs, summary = _history_relevance_pair(completed, relevance, outcome)
            pair_cache[(relevance, outcome)] = pairs
            effects[outcome][f"{relevance}_image_minus_text"] = summary
        strict: list[dict[str, Any]] = []
        for row in completed:
            if not row["strict_four_history_endpoint_matched"]:
                continue
            branches = row["branches"]
            relevant = branches["relevant_image"][outcome] - branches["relevant_text"][outcome]
            irrelevant = branches["irrelevant_image"][outcome] - branches["irrelevant_text"][outcome]
            strict.append(
                {
                    "item_id": row["item_id"],
                    "relevant_minus_irrelevant_modality_effect": float(relevant - irrelevant),
                }
            )
        effects[outcome]["strict_relevant_minus_irrelevant"] = paired_effect_summary(
            strict, "relevant_minus_irrelevant_modality_effect"
        )
    relevant = effects["pass1_sa"]["relevant_image_minus_text"]
    irrelevant = effects["pass1_sa"]["irrelevant_image_minus_text"]
    strict_difference = effects["pass1_sa"]["strict_relevant_minus_irrelevant"]
    if irrelevant["ci95"][0] is not None and irrelevant["ci95"][0] > 0:
        classification = (
            "generic modality priming plus task-relevance amplification"
            if strict_difference["ci95"][0] is not None
            and strict_difference["ci95"][0] > 0
            else "generic modality priming remains plausible"
        )
    elif (
        relevant["ci95"][0] is not None
        and relevant["ci95"][0] > 0
        and irrelevant["ci95"][0] is not None
        and irrelevant["ci95"][0] <= 0 <= irrelevant["ci95"][1]
        and abs(irrelevant["mean"]) <= max(0.05, 0.5 * abs(relevant["mean"]))
    ):
        classification = "task-relevant source-history retention supported"
    else:
        classification = "inconclusive between source memory and generic priming"
    errors = [
        branch["reconstruction_soft_error"]
        for row in completed
        for branch in row["branches"].values()
    ]
    return {
        "title": "Experiment 3 — Relevant vs Irrelevant History",
        "status": "completed",
        "n": len(completed),
        "failed": len(rows) - len(completed),
        "strict_four_history_endpoint_matched_n": sum(
            row["strict_four_history_endpoint_matched"] for row in completed
        ),
        "effects": effects,
        "classification": classification,
        "reconstruction": {
            "max_soft_abs_error": max(errors, default=None),
            "all_within_0.01": all(error <= 0.01 for error in errors),
        },
        "claim_limit": "Pairwise endpoint-matched contrasts are primary; the four-cell interaction is strict but may be low-powered.",
    }


def _save_unit_direction(
    root: Path,
    fold: int,
    kind: str,
    unit: np.ndarray,
    sigma: float,
    train_count: int,
) -> dict[str, Any]:
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
    return {
        "fold": fold,
        "file": filename,
        "direction_kind": kind,
        "sigma_z": sigma,
        "training_pair_count": train_count,
    }


def _load_history_hidden(output_root: Path, row: dict[str, Any]) -> dict[str, np.ndarray]:
    payload = np.load(output_root / row["hidden_file"])
    names = [str(value) for value in payload["branches"].tolist()]
    matrix = np.asarray(payload["hidden"], dtype=np.float64)
    return dict(zip(names, matrix))


def fit_natural_cue_directions(
    output_root: Path,
    geometry: BaselineGeometry,
) -> tuple[SAOOFDirectionRepository, SAOOFDirectionRepository, list[dict[str, Any]]]:
    directory = output_root / CUE_DIRECTIONS_DIR / "directions"
    h_root, e_root = directory / "history_oof", directory / "evidence_oof"
    history_rows = [
        row
        for row in _latest_rows(output_root / HISTORY_RELEVANCE_DIR / "results.jsonl")
        if row.get("status") == "completed"
        and row["branches"]["relevant_text"]["normalized_answer"]
        == row["branches"]["relevant_image"]["normalized_answer"]
    ]
    history_pairs = []
    for row in history_rows:
        hidden = _load_history_hidden(output_root, row)
        history_pairs.append(
            {
                "item_id": row["item_id"],
                "fold": int(row["fold"]),
                "difference": hidden["relevant_image"] - hidden["relevant_text"],
            }
        )
    evidence_rows = [
        row
        for row in load_jsonl(output_root / OLD_AUDIT_DIR / "evidence_pairs.jsonl")
        if row["endpoint_matched"]
    ]
    hidden_by_case = {
        str(row["case_id"]): geometry.hidden[index]
        for index, row in enumerate(geometry.rows)
    }
    evidence_pairs = [
        {
            "item_id": row["item_id"],
            "fold": int(row["fold"]),
            "difference": hidden_by_case[str(row["easy_case_id"])]
            - hidden_by_case[str(row["hard_case_id"])],
        }
        for row in evidence_rows
    ]
    h_entries: list[dict[str, Any]] = []
    e_entries: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    baseline_folds = np.asarray([int(row["fold"]) for row in geometry.rows])
    for fold in range(5):
        train_h = [row for row in history_pairs if row["fold"] != fold]
        train_e = [row for row in evidence_pairs if row["fold"] != fold]
        heldout_items = {str(row["item_id"]) for row in geometry.rows if int(row["fold"]) == fold}
        for pairs, root, kind, entries in (
            (train_h, h_root, "history_image_minus_text", h_entries),
            (train_e, e_root, "evidence_easy_minus_hard", e_entries),
        ):
            overlap = sorted({str(row["item_id"]) for row in pairs}.intersection(heldout_items))
            if overlap:
                raise RuntimeError(f"Cue direction fold leakage: {kind}, fold={fold}, items={overlap[:5]}")
            difference = np.mean(np.stack([row["difference"] for row in pairs]), axis=0)
            norm = float(np.linalg.norm(difference))
            if norm <= 1e-12:
                raise RuntimeError(f"Degenerate {kind} direction in fold {fold}")
            unit = difference / norm
            sigma = float(np.std(geometry.hidden[baseline_folds != fold] @ unit, ddof=1))
            entries.append(_save_unit_direction(root, fold, kind, unit, sigma, len(pairs)))
        audits.append(
            {
                "fold": fold,
                "heldout_items_excluded": True,
                "history_training_pairs": len(train_h),
                "evidence_training_pairs": len(train_e),
                "history_source_endpoint_matched": True,
                "evidence_source_endpoint_matched": True,
            }
        )
    atomic_write_json(h_root / "index.json", {"folds": h_entries})
    atomic_write_json(e_root / "index.json", {"folds": e_entries})
    atomic_write_json(directory / "fold_audit.json", audits)
    return SAOOFDirectionRepository(h_root), SAOOFDirectionRepository(e_root), audits


def _decision_direction(artifacts: SAFormationArtifacts, fold: int) -> np.ndarray:
    path = (
        artifacts.decision_direction_dir
        / "decision_directions"
        / f"v4_to_v4__fold_{fold}__panl__layer_18.npz"
    )
    payload = np.load(path)
    unit = np.asarray(payload["d_K"], dtype=np.float64)
    return unit / np.linalg.norm(unit)


def _cue_geometry(
    artifacts: SAFormationArtifacts,
    h_repo: SAOOFDirectionRepository,
    e_repo: SAOOFDirectionRepository,
    old_repo: SAOOFDirectionRepository,
    ridge_repo: SAOOFDirectionRepository,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in range(5):
        directions = {
            "d_H": h_repo.get(fold).d_unit,
            "d_E": e_repo.get(fold).d_unit,
            "d_old": old_repo.get(fold).d_unit,
            "d_Ridge": ridge_repo.get(fold).d_unit,
            "d_K": _decision_direction(artifacts, fold),
        }
        row: dict[str, Any] = {"fold": fold}
        names = list(directions)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                row[f"cos({left},{right})"] = float(directions[left] @ directions[right])
        rows.append(row)
    return rows


def run_natural_cue_directions(
    runtime: Stage3Runtime,
    artifacts: SAFormationArtifacts,
    stage3_root: Path,
    followup_root: Path,
    output_root: Path,
    geometry: BaselineGeometry,
    *,
    n_items: int,
    deadline: Callable[[], None],
) -> dict[str, Any]:
    directory = output_root / CUE_DIRECTIONS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    h_repo, e_repo, audits = fit_natural_cue_directions(output_root, geometry)
    old_repo = SAOOFDirectionRepository(followup_root / "directions" / "old_oof")
    ridge_repo = SAOOFDirectionRepository(stage3_root / "directions")
    geometry_rows = _cue_geometry(artifacts, h_repo, e_repo, old_repo, ridge_repo)
    atomic_write_json(directory / "geometry.json", geometry_rows)
    cohort = _balanced_unique_cases(load_baseline_rows(artifacts), n_items)
    atomic_write_json(
        directory / "cohort_manifest.json",
        {
            "case_count": len(cohort),
            "case_ids": [row["case_id"] for row in cohort],
            "directions": ["d_H", "d_E"],
            "dose_sigma_units": [-2, 2],
            "controls": "per-case equal-L2 random direction orthogonal to the tested cue direction",
            "fold_audit": audits,
        },
    )
    result_path = directory / "results.jsonl"
    existing = {row["intervention_key"] for row in _latest_rows(result_path)}
    for row in cohort:
        deadline()
        key = f"cue_direction|{row['case_id']}"
        if key in existing:
            continue
        base = {
            "intervention_key": key,
            "experiment": "natural_cue_direction_intervention",
            "case_id": row["case_id"],
            "item_id": row["item_id"],
            "fold": int(row["fold"]),
        }

        def execute() -> dict[str, Any]:
            prepared = _direction_eval_context(runtime, row)
            directions = {"d_H": h_repo.get(row["fold"]), "d_E": e_repo.get(row["fold"])}
            clean = runtime.measure(prepared, directions["d_H"]).to_dict()
            arms: dict[str, Any] = {}
            for kind, direction in directions.items():
                random_unit = orthogonal_equal_norm_control(
                    direction.d_unit,
                    1.0,
                    seed_material=f"{row['case_id']}|{kind}|cue",
                )
                for dose in (-2, 2):
                    label = f"{dose:+d}"
                    arms[f"{kind}|{label}"] = runtime.measure(
                        prepared,
                        direction,
                        steering_vector=dose * direction.sigma_z * direction.d_unit,
                    ).to_dict()
                    arms[f"{kind}_random|{label}"] = runtime.measure(
                        prepared,
                        direction,
                        steering_vector=dose * direction.sigma_z * random_unit,
                    ).to_dict()
            runtime.release_inputs(prepared)
            return {**base, "status": "completed", "clean": clean, "arms": arms}

        append_jsonl(result_path, _safe_record(base, execute))
    rows = _latest_rows(result_path)
    summary = _summarize_cue_directions(rows, geometry_rows)
    write_experiment_summary(directory, summary)
    return summary


def _summarize_cue_directions(
    rows: Sequence[dict[str, Any]], geometry_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed"]
    effects: dict[str, Any] = {}
    for kind in ("d_H", "d_E"):
        values: list[dict[str, Any]] = []
        for row in completed:
            arms = row["arms"]
            effect = (
                arms[f"{kind}|+2"]["source"]["soft_image_score"]
                - arms[f"{kind}|-2"]["source"]["soft_image_score"]
            )
            control = (
                arms[f"{kind}_random|+2"]["source"]["soft_image_score"]
                - arms[f"{kind}_random|-2"]["source"]["soft_image_score"]
            )
            values.append(
                {
                    "item_id": row["item_id"],
                    "effect": float(effect),
                    "control": float(control),
                    "effect_minus_control": float(effect - control),
                }
            )
        effect_summary = paired_effect_summary(values, "effect")
        control_summary = paired_effect_summary(values, "control")
        versus = paired_effect_summary(values, "effect_minus_control")
        effects[kind] = {
            "plus2_minus_minus2_sa": effect_summary,
            "equal_l2_random": control_summary,
            "direction_minus_control": versus,
            "causal_output_control_supported": bool(
                effect_summary["n"] >= 25
                and effect_summary["ci95"][0] is not None
                and effect_summary["ci95"][0] > 0
                and effect_summary["direction_rate"] >= 0.60
                and versus["ci95"][0] is not None
                and versus["ci95"][0] > 0
            ),
        }
    mean_geometry = {
        key: statistics.fmean(float(row[key]) for row in geometry_rows)
        for key in geometry_rows[0]
        if key != "fold"
    }
    return {
        "title": "Experiment 4 — Natural Cue Directions d_H and d_E",
        "status": "completed",
        "n": len(completed),
        "failed": len(rows) - len(completed),
        "fold_geometry": list(geometry_rows),
        "mean_geometry": mean_geometry,
        "interventions": effects,
        "claim_limit": "Cue directions are OOF fold-level mean contrasts; intervention establishes local output control, not by itself mediation.",
    }


def mechanism_gate(output_root: Path) -> dict[str, Any]:
    natural = read_json(output_root / "gate_exp1_natural.json")
    semantic = read_json(output_root / "gate_exp2_semantic.json")
    if not natural["passed"] and not semantic["passed"]:
        classification = (
            f"{semantic['classification']}; natural History-mediator gate also failed"
        )
    elif not natural["passed"]:
        classification = natural["classification"]
    else:
        classification = semantic["classification"]
    decision = {
        "natural_gate_passed": bool(natural["passed"]),
        "semantic_gate_passed": bool(semantic["passed"]),
        "allow_old_mediation": bool(natural["passed"] and semantic["passed"]),
        "allow_policy_transfer": bool(natural["passed"] and semantic["passed"]),
        "old_direction_classification": classification,
    }
    atomic_write_json(output_root / "gate_final.json", decision)
    return decision


def _write_skipped(directory: Path, title: str, reason: str) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(directory / "cohort_manifest.json", {"cases": [], "reason": reason})
    write_jsonl_atomic(directory / "results.jsonl", [])
    summary = {"title": title, "status": "skipped", "n": 0, "reason": reason}
    write_experiment_summary(directory, summary)
    return summary


def _standardized_effect(summary: dict[str, Any]) -> float:
    if summary.get("mean") is None or summary.get("sd") in (None, 0):
        return 0.0
    return abs(float(summary["mean"]) / float(summary["sd"]))


def _select_mediation_cue(output_root: Path) -> dict[str, Any]:
    audit = read_json(output_root / OLD_AUDIT_DIR / "summary.json")
    candidates: list[dict[str, Any]] = []
    evidence = audit["evidence_easy_minus_hard"]["endpoint_matched"]["old_z"]
    if evidence["ci95"][0] is not None and evidence["ci95"][0] > 0:
        candidates.append(
            {"cue": "evidence_easy_minus_hard", "effect": evidence, "fixed_answer_side": None}
        )
    comparisons = audit["history_exact_replay"]["comparisons"]
    for side in ("at", "ai"):
        value = comparisons[f"fixed_{side}_image_minus_text"]["z_old_std"]
        if value["ci95"][0] is not None and value["ci95"][0] > 0:
            candidates.append(
                {
                    "cue": "history_image_minus_text",
                    "effect": value,
                    "fixed_answer_side": side,
                }
            )
    if not candidates:
        raise RuntimeError("No stable natural cue is available for mediation")
    return max(candidates, key=lambda value: _standardized_effect(value["effect"]))


def _prepare_exact_history_context(
    runtime: Stage3Runtime,
    baseline: dict[str, Any],
    source_row: dict[str, Any],
    branch: str,
) -> Any:
    messages = _history_branch_messages(runtime, baseline, branch)
    generated = JointAnswerSourceGenerationResult(**source_row["branches"][branch]["pass1"])
    return prepare_exact_generated_measurement(
        runtime.generator,
        messages,
        generated,
        assistant_text=ASSISTANT_ANSWER_PREFILL,
    )


def _mediation_pairs(
    output_root: Path,
    geometry: BaselineGeometry,
    cue: dict[str, Any],
    n_pairs: int,
) -> list[dict[str, Any]]:
    if cue["cue"] == "evidence_easy_minus_hard":
        rows = [
            row
            for row in load_jsonl(output_root / OLD_AUDIT_DIR / "evidence_pairs.jsonl")
            if row["endpoint_matched"]
        ]
        rows.sort(key=lambda row: abs(float(row["delta_z_old_easy_minus_hard"])), reverse=True)
        selected: list[dict[str, Any]] = []
        used: set[str] = set()
        for row in rows:
            if row["item_id"] in used:
                continue
            selected.append(
                {
                    "pair_id": row["pair_id"],
                    "item_id": row["item_id"],
                    "fold": row["fold"],
                    "cue": cue["cue"],
                    "left_case_id": row["hard_case_id"],
                    "right_case_id": row["easy_case_id"],
                }
            )
            used.add(row["item_id"])
            if len(selected) >= n_pairs:
                break
        return selected
    side = str(cue["fixed_answer_side"])
    rows = [
        row
        for row in _latest_rows(output_root / OLD_AUDIT_DIR / "history_results.jsonl")
        if row.get("status") == "completed"
        and row["branches"][f"text_{side}"]["normalized_answer"]
        == row["branches"][f"image_{side}"]["normalized_answer"]
    ]
    rows.sort(
        key=lambda row: abs(
            row["branches"][f"image_{side}"]["z_old_std"]
            - row["branches"][f"text_{side}"]["z_old_std"]
        ),
        reverse=True,
    )
    return [
        {
            "pair_id": f"history|{row['case_id']}|{side}",
            "item_id": row["item_id"],
            "fold": row["fold"],
            "cue": cue["cue"],
            "fixed_answer_side": side,
            "source_case_id": row["case_id"],
            "left_branch": f"text_{side}",
            "right_branch": f"image_{side}",
        }
        for row in rows[:n_pairs]
    ]


def _mediation_contexts(
    runtime: Stage3Runtime,
    pair: dict[str, Any],
    geometry: BaselineGeometry,
    history_source: dict[str, dict[str, Any]],
) -> tuple[Any, Any]:
    if pair["cue"] == "evidence_easy_minus_hard":
        return (
            _direction_eval_context(runtime, geometry.row_by_case[pair["left_case_id"]]),
            _direction_eval_context(runtime, geometry.row_by_case[pair["right_case_id"]]),
        )
    baseline = geometry.row_by_case[pair["source_case_id"]]
    source = history_source[pair["source_case_id"]]
    return (
        _prepare_exact_history_context(runtime, baseline, source, pair["left_branch"]),
        _prepare_exact_history_context(runtime, baseline, source, pair["right_branch"]),
    )


def run_old_mediation(
    runtime: Stage3Runtime,
    stage3_root: Path,
    followup_root: Path,
    output_root: Path,
    geometry: BaselineGeometry,
    *,
    n_pairs: int,
    deadline: Callable[[], None],
) -> dict[str, Any]:
    directory = output_root / MEDIATION_DIR
    gate = mechanism_gate(output_root)
    if not gate["allow_old_mediation"]:
        return _write_skipped(
            directory,
            "Experiment 5 — Old-direction Mediation",
            "Natural and semantic gates did not both pass",
        )
    cue = _select_mediation_cue(output_root)
    pairs = _mediation_pairs(output_root, geometry, cue, n_pairs)
    atomic_write_json(
        directory / "cohort_manifest.json",
        {
            "selected_cue": cue,
            "pair_count": len(pairs),
            "pairs": pairs,
            "arms": ["clean", "old_clamp", "ridge_clamp", "zero_sham", "old_equal_l2_orthogonal"],
            "target_rule": "within-pair coordinate midpoint",
        },
    )
    old_repo = SAOOFDirectionRepository(followup_root / "directions" / "old_oof")
    ridge_repo = SAOOFDirectionRepository(stage3_root / "directions")
    history_source = {
        str(row["case_id"]): row
        for row in load_jsonl(
            followup_root / "02_history_exact_factorial" / "results_nocache.jsonl"
        )
        if row.get("status") == "completed"
    }
    result_path = directory / "results.jsonl"
    existing = {row["intervention_key"] for row in _latest_rows(result_path)}
    for pair in pairs:
        deadline()
        key = f"old_mediation|{pair['pair_id']}"
        if key in existing:
            continue
        base = {
            "intervention_key": key,
            "experiment": "old_direction_mediation",
            **pair,
        }

        def execute() -> dict[str, Any]:
            fold = int(pair["fold"])
            old = old_repo.get(fold)
            ridge = ridge_repo.get(fold)
            left_context, right_context = _mediation_contexts(
                runtime, pair, geometry, history_source
            )
            left_clean = runtime.measure(left_context, old)
            right_clean = runtime.measure(right_context, old)
            old_target = (left_clean.z_sa + right_clean.z_sa) / 2.0
            left_old_vector = coordinate_delta(left_clean.hidden, old.d_unit, old_target)
            right_old_vector = coordinate_delta(right_clean.hidden, old.d_unit, old_target)
            left_old = runtime.measure(left_context, old, steering_vector=left_old_vector)
            right_old = runtime.measure(right_context, old, steering_vector=right_old_vector)
            left_ridge_z = float(left_clean.hidden @ ridge.d_unit)
            right_ridge_z = float(right_clean.hidden @ ridge.d_unit)
            ridge_target = (left_ridge_z + right_ridge_z) / 2.0
            left_ridge_vector = coordinate_delta(left_clean.hidden, ridge.d_unit, ridge_target)
            right_ridge_vector = coordinate_delta(right_clean.hidden, ridge.d_unit, ridge_target)
            left_ridge = runtime.measure(left_context, old, steering_vector=left_ridge_vector)
            right_ridge = runtime.measure(right_context, old, steering_vector=right_ridge_vector)
            left_sham = runtime.measure(left_context, old)
            right_sham = runtime.measure(right_context, old)
            left_control = runtime.measure(
                left_context,
                old,
                steering_vector=orthogonal_equal_norm_control(
                    old.d_unit,
                    np.linalg.norm(left_old_vector),
                    seed_material=key + "|left",
                ),
            )
            right_control = runtime.measure(
                right_context,
                old,
                steering_vector=orthogonal_equal_norm_control(
                    old.d_unit,
                    np.linalg.norm(right_old_vector),
                    seed_material=key + "|right",
                ),
            )

            def effect(left: Any, right: Any) -> float:
                return float(
                    right.source["soft_image_score"] - left.source["soft_image_score"]
                )

            effects = {
                "clean": effect(left_clean, right_clean),
                "old_clamp": effect(left_old, right_old),
                "ridge_clamp": effect(left_ridge, right_ridge),
                "zero_sham": effect(left_sham, right_sham),
                "orthogonal": effect(left_control, right_control),
            }
            runtime.release_inputs(left_context, right_context)
            return {
                **base,
                "status": "completed",
                "old_target": old_target,
                "ridge_target": ridge_target,
                "effects": effects,
                "old_attenuation": effects["clean"] - effects["old_clamp"],
                "ridge_attenuation": effects["clean"] - effects["ridge_clamp"],
                "sham_attenuation": effects["clean"] - effects["zero_sham"],
                "orthogonal_attenuation": effects["clean"] - effects["orthogonal"],
                "old_minus_orthogonal_attenuation": effects["orthogonal"] - effects["old_clamp"],
                "old_minus_ridge_attenuation": effects["ridge_clamp"] - effects["old_clamp"],
            }

        append_jsonl(result_path, _safe_record(base, execute))
    rows = _latest_rows(result_path)
    summary = _summarize_mediation(rows, cue)
    write_experiment_summary(directory, summary)
    return summary


def _summarize_mediation(rows: Sequence[dict[str, Any]], cue: dict[str, Any]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed"]
    old = paired_effect_summary(completed, "old_attenuation")
    versus_orthogonal = paired_effect_summary(completed, "old_minus_orthogonal_attenuation")
    versus_ridge = paired_effect_summary(completed, "old_minus_ridge_attenuation")
    supported = bool(
        old["ci95"][0] is not None
        and old["ci95"][0] > 0
        and versus_orthogonal["ci95"][0] is not None
        and versus_orthogonal["ci95"][0] > 0
        and versus_ridge["ci95"][0] is not None
        and versus_ridge["ci95"][0] > 0
    )
    return {
        "title": "Experiment 5 — Old-direction Mediation",
        "status": "completed",
        "n": len(completed),
        "failed": len(rows) - len(completed),
        "selected_cue": cue,
        "old_clamp_attenuation": old,
        "ridge_clamp_attenuation": paired_effect_summary(completed, "ridge_attenuation"),
        "zero_sham_attenuation": paired_effect_summary(completed, "sham_attenuation"),
        "orthogonal_attenuation": paired_effect_summary(completed, "orthogonal_attenuation"),
        "old_minus_orthogonal_attenuation": versus_orthogonal,
        "old_minus_ridge_attenuation": versus_ridge,
        "causal_mediation_evidence": supported,
        "claim_limit": "causal mediation evidence, not complete mediation" if supported else "no mediation claim",
    }


def _orthogonal_to_span(vectors: Sequence[np.ndarray], seed_material: str) -> np.ndarray:
    seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    candidate = rng.standard_normal(vectors[0].shape)
    basis: list[np.ndarray] = []
    for vector in vectors:
        residual = np.asarray(vector, dtype=np.float64).copy()
        for previous in basis:
            residual -= float(residual @ previous) * previous
        norm = np.linalg.norm(residual)
        if norm > 1e-10:
            basis.append(residual / norm)
    for vector in basis:
        candidate -= float(candidate @ vector) * vector
    return candidate / np.linalg.norm(candidate)


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


def _policy_context(runtime: Stage3Runtime, row: dict[str, Any]) -> Any:
    case = runtime.case(row["item_id"], row["prior_index"])
    answer = str(row["baseline"]["generated"]["current_answer_result"]["normalized_answer"])
    messages = [
        {
            "role": "user",
            "content": image_content(
                str(case.conditions[row["condition"]].resolved_image_path),
                full_prompt(case),
            ),
        },
        assistant_message(f"**Answer**: {answer}\n\n"),
        {"role": "user", "content": text_content(SOURCE_CHOICE_PROMPT)},
        assistant_message("**Source Choice**:"),
    ]
    assert_policy_no_verbal_sa(messages)
    return prepare_policy_measurement(
        runtime.generator,
        messages,
        assistant_text="**Source Choice**:",
        fixed_answer=answer,
    )


def run_policy_transfer(
    runtime: Stage3Runtime,
    artifacts: SAFormationArtifacts,
    stage3_root: Path,
    followup_root: Path,
    output_root: Path,
    *,
    n_items: int,
    deadline: Callable[[], None],
) -> dict[str, Any]:
    directory = output_root / POLICY_DIR
    gate = mechanism_gate(output_root)
    if not gate["allow_policy_transfer"]:
        return _write_skipped(
            directory,
            "Experiment 6 — Policy Transfer",
            "Natural and semantic gates did not both pass",
        )
    old_repo = SAOOFDirectionRepository(followup_root / "directions" / "old_oof")
    residual_repo = SAOOFDirectionRepository(
        followup_root / "directions" / "old_perp_ridge_oof"
    )
    ridge_repo = SAOOFDirectionRepository(stage3_root / "directions")
    cohort = _balanced_unique_cases(load_baseline_rows(artifacts), n_items)
    atomic_write_json(
        directory / "cohort_manifest.json",
        {
            "case_count": len(cohort),
            "case_ids": [row["case_id"] for row in cohort],
            "directions": ["old", "ridge", "old_perp_ridge", "random_span_orthogonal"],
            "doses_sigma_units": [-2, -1, 0, 1, 2],
            "protocol": "0=Text, 1=Image",
            "verbal_sa_in_assistant_history": False,
        },
    )
    result_path = directory / "results.jsonl"
    existing = {row["intervention_key"] for row in _latest_rows(result_path)}
    for row in cohort:
        deadline()
        key = f"policy_transfer|{row['case_id']}"
        if key in existing:
            continue
        base = {
            "intervention_key": key,
            "experiment": "policy_transfer",
            "case_id": row["case_id"],
            "item_id": row["item_id"],
            "fold": int(row["fold"]),
        }

        def execute() -> dict[str, Any]:
            fold = int(row["fold"])
            old, ridge, residual = old_repo.get(fold), ridge_repo.get(fold), residual_repo.get(fold)
            random_unit = _orthogonal_to_span(
                [old.d_unit, ridge.d_unit, residual.d_unit],
                f"{row['case_id']}|policy",
            )
            directions = {
                "old": old,
                "ridge": ridge,
                "old_perp_ridge": residual,
                "random_span_orthogonal": _direction_object(random_unit, old.sigma_z, fold),
            }
            prepared = _policy_context(runtime, row)
            clean = runtime.measure(prepared, old, policy=True)
            arms: dict[str, Any] = {}
            for kind, direction in directions.items():
                for dose in (-2, -1, 0, 1, 2):
                    measured = (
                        clean
                        if dose == 0
                        else runtime.measure(
                            prepared,
                            direction,
                            steering_vector=dose * direction.sigma_z * direction.d_unit,
                            policy=True,
                        )
                    )
                    arms[f"{kind}|{dose:+d}"] = {
                        "p_image": float(measured.source["soft_image_score"]),
                        "entropy": float(measured.source["source_entropy"]),
                        "hard_choice": measured.source["hard_label"],
                        "applied_delta_sigma": measured.applied_delta_z / direction.sigma_z,
                    }
            runtime.release_inputs(prepared)
            return {
                **base,
                "status": "completed",
                "clean_old_z": clean.z_sa,
                "clean_p_image": float(clean.source["soft_image_score"]),
                "arms": arms,
            }

        append_jsonl(result_path, _safe_record(base, execute))
    rows = _latest_rows(result_path)
    summary = _summarize_policy(rows)
    write_experiment_summary(directory, summary)
    return summary


def _summarize_policy(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed"]
    association = _correlation_summary(
        [
            {"item_id": row["item_id"], "z": row["clean_old_z"], "p": row["clean_p_image"]}
            for row in completed
        ],
        "z",
        "p",
    )
    effects: dict[str, Any] = {}
    for kind in ("old", "ridge", "old_perp_ridge", "random_span_orthogonal"):
        by_dose: dict[str, Any] = {}
        for dose in (1, 2):
            values = [
                {
                    "item_id": row["item_id"],
                    "delta": float(
                        row["arms"][f"{kind}|+{dose}"]["p_image"]
                        - row["arms"][f"{kind}|-{dose}"]["p_image"]
                    ),
                }
                for row in completed
            ]
            by_dose[str(dose)] = paired_effect_summary(values, "delta")
        effects[kind] = by_dose
    old = effects["old"]["2"]
    supported = bool(
        old["n"] >= 15
        and old["ci95"][0] is not None
        and old["ci95"][0] > 0
        and old["direction_rate"] >= 0.60
    )
    return {
        "title": "Experiment 6 — Policy Transfer",
        "status": "completed",
        "n": len(completed),
        "failed": len(rows) - len(completed),
        "natural_old_z_policy_association": association,
        "direction_effects": effects,
        "old_direction_policy_transfer_supported": supported,
        "classification": (
            "broader downstream source-preference state"
            if supported
            else "SA report-specific actuator under the tested Policy continuation"
        ),
    }


def write_mechanism_report(output_root: Path) -> dict[str, Any]:
    summaries = {
        "exp1_old_natural": read_json(output_root / OLD_AUDIT_DIR / "summary.json"),
        "exp2_label_remapping": read_json(output_root / REMAP_DIR / "summary.json"),
        "exp3_history_relevance": read_json(
            output_root / HISTORY_RELEVANCE_DIR / "summary.json"
        ),
        "exp4_cue_directions": read_json(
            output_root / CUE_DIRECTIONS_DIR / "summary.json"
        ),
        "exp5_mediation": read_json(output_root / MEDIATION_DIR / "summary.json"),
        "exp6_policy": read_json(output_root / POLICY_DIR / "summary.json"),
    }
    gate = mechanism_gate(output_root)
    exp1 = summaries["exp1_old_natural"]
    exp2 = summaries["exp2_label_remapping"]
    exp3 = summaries["exp3_history_relevance"]
    exp4 = summaries["exp4_cue_directions"]
    core_rows = [
        {
            "experiment": "Old natural association",
            "evidence_type": "OOF correlation",
            "n": exp1["n"],
            "effect": exp1["natural_association"]["spearman"],
            "ci_low": exp1["natural_association"]["spearman_item_bootstrap"]["ci95"][0],
            "ci_high": exp1["natural_association"]["spearman_item_bootstrap"]["ci95"][1],
            "supported": exp1["natural_association"]["association_supported"],
        },
        {
            "experiment": "Evidence Easy-Hard -> old z",
            "evidence_type": "paired natural cue",
            "n": exp1["evidence_easy_minus_hard"]["all_pairs"]["old_z"]["n"],
            "effect": exp1["evidence_easy_minus_hard"]["all_pairs"]["old_z"]["mean"],
            "ci_low": exp1["evidence_easy_minus_hard"]["all_pairs"]["old_z"]["ci95"][0],
            "ci_high": exp1["evidence_easy_minus_hard"]["all_pairs"]["old_z"]["ci95"][1],
            "supported": exp1["evidence_easy_minus_hard"]["cue_supported"],
        },
    ]
    for mapping, result in exp2["effects"].items():
        effect = result["semantic_imageward"]
        core_rows.append(
            {
                "experiment": f"Label remapping: {mapping}",
                "evidence_type": "causal SA intervention",
                "n": effect["n"],
                "effect": effect["mean"],
                "ci_low": effect["ci95"][0],
                "ci_high": effect["ci95"][1],
                "supported": result["semantic_supported"],
            }
        )
    for relevance in ("relevant", "irrelevant"):
        effect = exp3["effects"]["pass1_sa"][f"{relevance}_image_minus_text"]
        core_rows.append(
            {
                "experiment": f"History {relevance} Image-Text",
                "evidence_type": "endpoint-matched paired formation",
                "n": effect["n"],
                "effect": effect["mean"],
                "ci_low": effect["ci95"][0],
                "ci_high": effect["ci95"][1],
                "supported": effect["ci95"][0] is not None and effect["ci95"][0] > 0,
            }
        )
    for direction in ("d_H", "d_E"):
        effect = exp4["interventions"][direction]["plus2_minus_minus2_sa"]
        core_rows.append(
            {
                "experiment": f"do({direction}) -> SA",
                "evidence_type": "causal SA intervention",
                "n": effect["n"],
                "effect": effect["mean"],
                "ci_low": effect["ci95"][0],
                "ci_high": effect["ci95"][1],
                "supported": exp4["interventions"][direction]["causal_output_control_supported"],
            }
        )
    analysis = output_root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(analysis / "core_table.csv", core_rows)
    payload = {"status": "completed", "gate": gate, "experiments": summaries}
    atomic_write_json(analysis / "final_analysis.json", payload)
    natural = exp1["natural_association"]
    evidence = exp1["evidence_easy_minus_hard"]["all_pairs"]["old_z"]
    history_old = exp1["history_exact_replay"]["comparisons"]
    lines = [
        "# Stage 3 SA Mechanism — Final Analysis",
        "",
        "## Identity gates",
        "",
        f"- Old natural gate: `{gate['natural_gate_passed']}`. OOF Spearman={natural['spearman']:.6f}, item-bootstrap CI={natural['spearman_item_bootstrap']['ci95']}.",
        f"- Easy−Hard old-z={evidence['mean']:.6f}, CI={evidence['ci95']}.",
        f"- Fixed A_T History Image−Text old-z={history_old['fixed_at_image_minus_text']['z_old_std']['mean']}, CI={history_old['fixed_at_image_minus_text']['z_old_std']['ci95']}.",
        f"- Fixed A_I History Image−Text old-z={history_old['fixed_ai_image_minus_text']['z_old_std']['mean']}, CI={history_old['fixed_ai_image_minus_text']['z_old_std']['ci95']}.",
        f"- Semantic remapping gate: `{gate['semantic_gate_passed']}`; classification: **{gate['old_direction_classification']}**.",
        "",
        "## History specificity",
        "",
        f"- Relevant/irrelevant classification: **{exp3['classification']}**.",
        f"- Relevant Image−Text SA: {exp3['effects']['pass1_sa']['relevant_image_minus_text']}.",
        f"- Irrelevant Image−Text SA: {exp3['effects']['pass1_sa']['irrelevant_image_minus_text']}.",
        "",
        "## Natural cue geometry",
        "",
        f"- Mean fold geometry: `{exp4['mean_geometry']}`.",
        f"- do(d_H) output control: `{exp4['interventions']['d_H']['causal_output_control_supported']}`.",
        f"- do(d_E) output control: `{exp4['interventions']['d_E']['causal_output_control_supported']}`.",
        "",
        "## Gate-controlled downstream tests",
        "",
        f"- Old mediation: status `{summaries['exp5_mediation']['status']}`; evidence={summaries['exp5_mediation'].get('causal_mediation_evidence')}",
        f"- Policy transfer: status `{summaries['exp6_policy']['status']}`; supported={summaries['exp6_policy'].get('old_direction_policy_transfer_supported')}",
        "",
        "The three identities remain distinct: natural formation association, linear predictability, and causal output control are not treated as interchangeable.",
        "",
    ]
    atomic_write_text(analysis / "FINAL_ANALYSIS.md", "\n".join(lines))
    return payload
