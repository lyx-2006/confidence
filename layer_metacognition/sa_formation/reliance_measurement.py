"""Clean behavioral measurement of Actual Source Reliance.

The experiment deliberately contains no Source Attribution request.  It first
selects the natural answer under an answer-only full-evidence prompt and then
measures how the probability of that fixed answer changes when text or image
evidence is deleted or symmetrically replaced.  Verbal SA is never used to
construct, calibrate, or gate the behavioral target.
"""

from __future__ import annotations

import hashlib
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from confidence_test.answer_metrics import normalize_answer
from confidence_test.dataset_utils import EvaluationCase, load_evaluation_cases
from confidence_test.inference_extension import ASSISTANT_ANSWER_PREFILL
from confidence_test.prompt_utils import V4_STAGE1_FULL_EVIDENCE_ANSWER_PROMPT
from layer_metacognition.hidden_state_store import (
    append_jsonl,
    atomic_write_json,
    atomic_write_text,
    load_jsonl,
)
from layer_metacognition.model_adapter import (
    HookedForwardResult,
    run_hooked_forward,
    run_logits_forward,
)

from .core import (
    SAFormationArtifacts,
    atomic_save_npz,
    canonical_message_hash,
    item_cluster_bootstrap,
    stable_hash,
    write_jsonl_atomic,
)
from .runtime import Stage3Runtime, assistant_message, image_content


BRIDGE_DIR = "stage3_sa_computational_bridge"
RELIANCE_DIR = "01_actual_source_reliance"
MEASUREMENT_METHOD_VERSION = 2
DEVELOPMENT_N = 100
CONFIRMATORY_N = 77
SEED = 42
NO_TEXT_PLACEHOLDER = "[No text clue available.]"
CORE_CONDITIONS = ("full", "no_text", "no_image", "replace_text_d1", "replace_image_d1")
PANEL_CONDITIONS = (
    *CORE_CONDITIONS,
    "replace_text_d2",
    "replace_image_d2",
)
POSITIONS = ("pre_answer", "post_answer")
NUISANCE_BASE_FEATURES = (
    "intercept",
    "choice_image",
    "choice_other",
    "difficulty_hard",
    "prior_strength",
)


class AmbiguousEndpointError(ValueError):
    """The answer-only restricted distribution has no unique natural top-1."""

    def __init__(self, message: str, *, endpoint_audit: dict[str, Any]) -> None:
        super().__init__(message)
        self.endpoint_audit = endpoint_audit


def _numeric_item_key(value: Any) -> tuple[int, str]:
    text = str(value)
    try:
        return int(text), text
    except ValueError:
        return 10**12, text


def _latest_rows(path: str | Path) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path, repair_trailing=True):
        key = str(row.get("intervention_key", row.get("case_id", "")))
        if key:
            latest[key] = row
    return list(latest.values())


def _safe_record(base: dict[str, Any], operation: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        value = operation()
        return {**value, "elapsed_seconds": time.perf_counter() - started}
    except AmbiguousEndpointError as exc:
        return {
            **base,
            "status": "excluded",
            "exclusion_reason": "tied_natural_endpoint",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "endpoint_audit": exc.endpoint_audit,
            "elapsed_seconds": time.perf_counter() - started,
        }
    except Exception as exc:
        return {
            **base,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": time.perf_counter() - started,
        }


def answer_only_prompt(case: EvaluationCase, text_clue: str) -> str:
    return V4_STAGE1_FULL_EVIDENCE_ANSWER_PROMPT.format(
        question=case.question,
        text_clue=text_clue,
    )


def contains_verbal_sa_request(messages: Sequence[dict[str, Any]]) -> bool:
    for message in messages:
        for part in message.get("content", []):
            if (
                isinstance(part, dict)
                and part.get("type") == "text"
                and "source attribution" in str(part.get("text", "")).lower()
            ):
                return True
    return False


def build_answer_only_messages(
    case: EvaluationCase,
    *,
    text_clue: str,
    image_path: str,
    assistant_text: str = ASSISTANT_ANSWER_PREFILL,
) -> list[dict[str, Any]]:
    messages = [
        {
            "role": "user",
            "content": image_content(image_path, answer_only_prompt(case, text_clue)),
        },
        assistant_message(assistant_text),
    ]
    if contains_verbal_sa_request(messages):
        raise ValueError("Answer-only Actual Reliance prompt leaks a verbal-SA request")
    return messages


def canonical_answer_token_ids(tokenizer: Any, answer_classes: Sequence[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for raw_label in answer_classes:
        label = str(normalize_answer(raw_label))
        ids = tokenizer.encode(" " + label, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Answer label is not a canonical leading-space token: {label}={ids}")
        result[label] = int(ids[0])
    if len(result) != len(answer_classes) or len(set(result.values())) != len(result):
        raise ValueError("Canonical answer labels collide after normalization/tokenization")
    return result


def restricted_distribution(
    vocab_logits: torch.Tensor,
    token_ids: dict[str, int],
) -> dict[str, Any]:
    labels = list(token_ids)
    selected = torch.stack([vocab_logits[token_ids[label]] for label in labels]).float()
    probabilities = torch.softmax(selected, dim=-1)
    logits = {label: float(selected[index].item()) for index, label in enumerate(labels)}
    probs = {label: float(probabilities[index].item()) for index, label in enumerate(labels)}
    ordered = sorted(logits.items(), key=lambda value: value[1], reverse=True)
    margin = float(ordered[0][1] - ordered[1][1]) if len(ordered) > 1 else math.inf
    return {
        "predicted_answer": ordered[0][0],
        "unique_top1": bool(margin > 1e-6),
        "top1_top2_logit_margin": margin,
        "answer_class_logits": logits,
        "answer_class_probabilities": probs,
        "canonical_leading_token_ids": token_ids,
    }


def fixed_answer_effects(measurements: dict[str, dict[str, Any]], answer: str) -> dict[str, float]:
    required = set(PANEL_CONDITIONS)
    missing = required.difference(measurements)
    if missing:
        raise ValueError(f"Measurement panel is incomplete: {sorted(missing)}")
    fixed = str(normalize_answer(answer))
    logp: dict[str, float] = {}
    for condition in PANEL_CONDITIONS:
        probabilities = measurements[condition]["answer_class_probabilities"]
        probability = float(probabilities[fixed])
        if not math.isfinite(probability) or probability <= 0.0:
            raise ValueError(f"Non-positive fixed-answer probability in {condition}: {probability}")
        logp[condition] = math.log(probability)
    deletion = logp["no_text"] - logp["no_image"]
    replacement_d1 = logp["replace_text_d1"] - logp["replace_image_d1"]
    replacement_d2 = logp["replace_text_d2"] - logp["replace_image_d2"]
    replacement = 0.5 * (replacement_d1 + replacement_d2)
    return {
        "fixed_answer_logp_full": logp["full"],
        "behavior_delete_imageward": deletion,
        "behavior_replace_imageward_d1": replacement_d1,
        "behavior_replace_imageward_d2": replacement_d2,
        "behavior_replace_imageward": replacement,
        "replacement_donor_disagreement": 0.5 * (replacement_d1 - replacement_d2),
        "remove_image_drop_logp": logp["full"] - logp["no_image"],
        "remove_text_drop_logp": logp["full"] - logp["no_text"],
    }


def _prior_bin(case: EvaluationCase) -> str:
    return str(case.prior_bin or "")


def load_source_rows(experiment_dir: str | Path) -> list[dict[str, Any]]:
    path = (
        Path(experiment_dir)
        / "stage3_sa_truth_audit"
        / "01_counterfactual_source_use"
        / "results.jsonl"
    )
    rows = [row for row in load_jsonl(path) if row.get("status") == "completed"]
    if len({str(row["item_id"]) for row in rows}) != 178:
        raise ValueError("Actual Reliance requires the authoritative 178-item source-use pool")
    return rows


def _balanced_unique_cases(rows: Sequence[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if row.get("text_answer")
        and row.get("image_answer")
        and row["text_answer"] != row["image_answer"]
        and str(row.get("condition", "")).startswith("conflict_")
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
        buckets[(int(row["fold"]), str(row["difficulty"]), int(row["final_image"]))].append(row)
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    keys = sorted(buckets, key=str)
    while len(selected) < n:
        advanced = False
        for key in keys:
            while buckets[key] and str(buckets[key][0]["item_id"]) in used:
                buckets[key].pop(0)
            if buckets[key] and len(selected) < n:
                row = buckets[key].pop(0)
                selected.append(row)
                used.add(str(row["item_id"]))
                advanced = True
        if not advanced:
            break
    if len(selected) != n:
        raise ValueError(f"Only {len(selected)} balanced unique cases are available; expected {n}")
    return selected


def select_split_cohorts(
    experiment_dir: str | Path,
    source_rows: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    experiment = Path(experiment_dir)
    development_manifest = (
        experiment
        / "stage3_sa_truth_audit"
        / "02_matched_prompt_source_perturbation"
        / "cohort_manifest.json"
    )
    payload = __import__("json").loads(development_manifest.read_text(encoding="utf-8"))
    case_map = {str(row["case_id"]): row for row in source_rows}
    development = [case_map[str(case_id)] for case_id in payload["case_ids"]]
    if len(development) != DEVELOPMENT_N or len({str(row["item_id"]) for row in development}) != DEVELOPMENT_N:
        raise ValueError("Authoritative development cohort is not 100 unique items")
    used_items = {str(row["item_id"]) for row in development}
    confirmatory_pool = [row for row in source_rows if str(row["item_id"]) not in used_items]
    confirmatory = _balanced_unique_cases(confirmatory_pool, CONFIRMATORY_N)
    if {str(row["item_id"]) for row in development}.intersection(
        str(row["item_id"]) for row in confirmatory
    ):
        raise RuntimeError("Development and confirmatory item sets overlap")
    return {"development": development, "confirmatory": confirmatory}


def select_two_donors(
    target: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    case_by_key: dict[tuple[str, int], EvaluationCase],
) -> tuple[dict[str, Any], dict[str, Any]]:
    target_case = case_by_key[(str(target["item_id"]), int(target["prior_index"]))]
    eligible = [
        row
        for row in candidates
        if str(row["item_id"]) != str(target["item_id"])
        and int(row["fold"]) == int(target["fold"])
        and str(row["difficulty"]) == str(target["difficulty"])
        and int(row["final_image"]) == int(target["final_image"])
    ]
    if not eligible:
        raise ValueError(f"No matched donor for {target['case_id']}")

    def rank(row: dict[str, Any]) -> tuple[Any, ...]:
        donor_case = case_by_key[(str(row["item_id"]), int(row["prior_index"]))]
        return (
            int(_prior_bin(donor_case) != _prior_bin(target_case)),
            int(str(row["condition"]) != str(target["condition"])),
            abs(len(donor_case.text_clue) - len(target_case.text_clue)),
            _numeric_item_key(row["item_id"]),
            int(row["prior_index"]),
            str(row["case_id"]),
        )

    ranked = sorted(eligible, key=rank)
    chosen: list[dict[str, Any]] = []
    used_items: set[str] = set()
    for row in ranked:
        item = str(row["item_id"])
        if item in used_items:
            continue
        chosen.append(row)
        used_items.add(item)
        if len(chosen) == 2:
            break
    if len(chosen) != 2:
        raise ValueError(f"Fewer than two distinct donor items for {target['case_id']}")
    return chosen[0], chosen[1]


def plan_split(
    artifacts: SAFormationArtifacts,
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, tuple[dict[str, Any], dict[str, Any]]], dict[tuple[str, int], EvaluationCase]]:
    if split not in {"development", "confirmatory"}:
        raise ValueError(f"Unknown split: {split}")
    source_rows = load_source_rows(artifacts.experiment_dir)
    cohorts = select_split_cohorts(artifacts.experiment_dir, source_rows)
    cases, _ = load_evaluation_cases(artifacts.dataset)
    case_by_key = {(str(case.item_id), int(case.prior_index)): case for case in cases}
    cohort = cohorts[split]
    # Donors come from the full authoritative source-use pool.  Requiring them
    # to be targets in the same split makes rare fold/difficulty/final-side
    # strata impossible (one confirmatory stratum has a single target).  The
    # donor itself remains a distinct item in the same held-out fold.
    donors = {
        str(row["case_id"]): select_two_donors(row, source_rows, case_by_key)
        for row in cohort
    }
    return cohort, donors, case_by_key


def condition_sources(
    target_case: EvaluationCase,
    target_row: dict[str, Any],
    donor_rows: tuple[dict[str, Any], dict[str, Any]],
    case_by_key: dict[tuple[str, int], EvaluationCase],
) -> dict[str, dict[str, Any]]:
    target_image = str(target_case.conditions[str(target_row["condition"])].resolved_image_path)
    null_image = str(target_case.conditions["null"].resolved_image_path)
    result: dict[str, dict[str, Any]] = {
        "full": {
            "text_clue": target_case.text_clue,
            "image_path": target_image,
            "text_source_item": str(target_row["item_id"]),
            "image_source_item": str(target_row["item_id"]),
        },
        "no_text": {
            "text_clue": NO_TEXT_PLACEHOLDER,
            "image_path": target_image,
            "text_source_item": None,
            "image_source_item": str(target_row["item_id"]),
        },
        "no_image": {
            "text_clue": target_case.text_clue,
            "image_path": null_image,
            "text_source_item": str(target_row["item_id"]),
            "image_source_item": None,
        },
    }
    for index, donor_row in enumerate(donor_rows, start=1):
        donor_case = case_by_key[(str(donor_row["item_id"]), int(donor_row["prior_index"]))]
        donor_image = str(donor_case.conditions[str(donor_row["condition"])].resolved_image_path)
        result[f"replace_text_d{index}"] = {
            "text_clue": donor_case.text_clue,
            "image_path": target_image,
            "text_source_item": str(donor_row["item_id"]),
            "image_source_item": str(target_row["item_id"]),
        }
        result[f"replace_image_d{index}"] = {
            "text_clue": target_case.text_clue,
            "image_path": donor_image,
            "text_source_item": str(target_row["item_id"]),
            "image_source_item": str(donor_row["item_id"]),
        }
    if tuple(result) != PANEL_CONDITIONS:
        raise RuntimeError(f"Condition order drifted: {tuple(result)}")
    for index in (1, 2):
        text = result[f"replace_text_d{index}"]
        image = result[f"replace_image_d{index}"]
        if text["text_source_item"] != image["image_source_item"]:
            raise RuntimeError(f"Donor {index} is asymmetric")
        if text["image_source_item"] != image["text_source_item"]:
            raise RuntimeError(f"Target context is asymmetric for donor {index}")
    return result


def _stack_hidden(result: HookedForwardResult, layer_count: int) -> np.ndarray:
    stacked = np.stack(
        [
            np.stack(
                [
                    result.hidden_by_name[position][layer]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    for layer in range(layer_count)
                ],
                axis=0,
            )
            for position in POSITIONS
        ],
        axis=0,
    ).astype(np.float16)
    if stacked.ndim != 3 or stacked.shape[:2] != (2, layer_count):
        raise RuntimeError(f"Unexpected hooked hidden shape: {stacked.shape}")
    if not np.isfinite(stacked).all():
        raise RuntimeError("Hooked hidden contains non-finite values")
    return stacked


def _causal_prefix_audit(pre_inputs: Any, post_inputs: Any) -> dict[str, Any]:
    """Verify that adding the fixed answer changes only the final text token.

    Qwen-VL can take a numerically different BF16 kernel path when sequence
    length changes.  Equality here is therefore defined on the causal inputs,
    not on logits recomputed with a longer sequence.
    """

    pre_length = int(pre_inputs.input_ids.shape[1])
    post_length = int(post_inputs.input_ids.shape[1])
    if post_length != pre_length + 1:
        raise ValueError(
            f"Teacher-forced answer is not exactly one token: pre={pre_length}, post={post_length}"
        )
    checks: dict[str, bool] = {}
    for key in ("input_ids", "attention_mask"):
        left = pre_inputs[key]
        right = post_inputs[key][:, :pre_length]
        checks[f"{key}_prefix_equal"] = bool(torch.equal(left, right))
    for key in (
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
        "second_per_grid_ts",
    ):
        if key not in pre_inputs and key not in post_inputs:
            continue
        equal = (
            key in pre_inputs
            and key in post_inputs
            and tuple(pre_inputs[key].shape) == tuple(post_inputs[key].shape)
            and torch.equal(pre_inputs[key], post_inputs[key])
        )
        checks[f"{key}_equal"] = bool(equal)
    passed = bool(checks and all(checks.values()))
    if not passed:
        raise RuntimeError(f"Teacher-forced causal prefix changed: {checks}")
    return {
        "passed": passed,
        "pre_token_count": pre_length,
        "post_token_count": post_length,
        "checks": checks,
    }


def _pre_answer_condition(
    runtime: Stage3Runtime,
    case: EvaluationCase,
    source: dict[str, Any],
    answer: str | None,
    token_ids: dict[str, int],
    *,
    capture_hidden: bool,
) -> tuple[dict[str, Any], np.ndarray | None]:
    pre_messages = build_answer_only_messages(
        case,
        text_clue=str(source["text_clue"]),
        image_path=str(source["image_path"]),
    )
    pre_rendered, pre_inputs = runtime.generator.prepare_messages(
        pre_messages,
        assistant_text=ASSISTANT_ANSWER_PREFILL,
    )
    pre_position = int(pre_inputs.input_ids.shape[1]) - 1
    if capture_hidden:
        pre_hooked = run_hooked_forward(
            runtime.model,
            pre_inputs,
            runtime.modules,
            {"pre_answer": pre_position},
            logits_positions=[pre_position],
        )
        vocab_logits = pre_hooked.logits_by_position[pre_position]
    else:
        pre_hooked = None
        vocab_logits = run_logits_forward(
            runtime.model,
            pre_inputs,
            [pre_position],
            runtime.modules,
        )[pre_position]
    distribution = restricted_distribution(vocab_logits, token_ids)
    fixed_answer = answer
    if fixed_answer is None:
        if not distribution["unique_top1"]:
            top_logit = max(
                float(value) for value in distribution["answer_class_logits"].values()
            )
            raise AmbiguousEndpointError(
                "Answer-only full context has a tied natural endpoint",
                endpoint_audit={
                    "selection": distribution,
                    "selection_rendered_hash": hashlib.sha256(
                        pre_rendered.encode("utf-8")
                    ).hexdigest(),
                    "tied_labels": sorted(
                        label
                        for label, value in distribution["answer_class_logits"].items()
                        if abs(float(value) - top_logit) <= 1e-6
                    ),
                    "selection_stage": "full answer-only prefix before any perturbation",
                },
            )
        fixed_answer = str(distribution["predicted_answer"])
    metadata = {
        **distribution,
        "messages_hash": canonical_message_hash(pre_messages),
        "prefix_rendered_hash": hashlib.sha256(pre_rendered.encode("utf-8")).hexdigest(),
        "pre_answer_position": pre_position,
        "post_answer_position": None,
        "input_token_count": int(pre_inputs.input_ids.shape[1]),
        "verbal_sa_leakage": contains_verbal_sa_request(pre_messages),
        "measurement_forward": "answer-only causal prefix without fixed answer appended",
    }
    hidden: np.ndarray | None = None
    if capture_hidden:
        assistant_text = f"{ASSISTANT_ANSWER_PREFILL} {fixed_answer}"
        post_messages = build_answer_only_messages(
            case,
            text_clue=str(source["text_clue"]),
            image_path=str(source["image_path"]),
            assistant_text=assistant_text,
        )
        post_rendered, post_inputs = runtime.generator.prepare_messages(
            post_messages,
            assistant_text=assistant_text,
        )
        post_position = int(post_inputs.input_ids.shape[1]) - 1
        prefix_audit = _causal_prefix_audit(pre_inputs, post_inputs)
        if int(post_inputs.input_ids[0, post_position].item()) != int(
            token_ids[fixed_answer]
        ):
            raise ValueError("Teacher-forced answer token differs from canonical leading token")
        if not post_rendered.startswith(pre_rendered):
            raise RuntimeError("Teacher-forced rendered text does not preserve the exact prefix")
        post_hooked = run_hooked_forward(
            runtime.model,
            post_inputs,
            runtime.modules,
            {"post_answer": post_position},
            logits_positions=[pre_position],
        )
        replay_logits = post_hooked.logits_by_position[pre_position]
        replay_distribution = restricted_distribution(replay_logits, token_ids)
        combined = HookedForwardResult(
            hidden_by_name={
                "pre_answer": pre_hooked.hidden_by_name["pre_answer"],
                "post_answer": post_hooked.hidden_by_name["post_answer"],
            },
            logits_by_position={},
        )
        hidden = _stack_hidden(combined, runtime.modules.num_hidden_layers)
        replay_error = max(
            abs(
                float(distribution["answer_class_logits"][label])
                - float(replay_distribution["answer_class_logits"][label])
            )
            for label in token_ids
        )
        replay_tv = 0.5 * sum(
            abs(
                float(distribution["answer_class_probabilities"][label])
                - float(replay_distribution["answer_class_probabilities"][label])
            )
            for label in token_ids
        )
        metadata.update(
            {
                "teacher_forced_messages_hash": canonical_message_hash(post_messages),
                "teacher_forced_rendered_hash": hashlib.sha256(
                    post_rendered.encode("utf-8")
                ).hexdigest(),
                "post_answer_position": post_position,
                "teacher_forced_input_token_count": int(post_inputs.input_ids.shape[1]),
                "teacher_forced_causal_prefix_audit": prefix_audit,
                "teacher_forced_length_path_max_logit_error": replay_error,
                "teacher_forced_length_path_probability_tv": replay_tv,
                "teacher_forced_length_path_predicted_answer": replay_distribution[
                    "predicted_answer"
                ],
            }
        )
        del post_inputs, replay_logits, post_hooked, combined
    del pre_inputs, vocab_logits, pre_hooked
    return metadata, hidden


def measure_case(
    runtime: Stage3Runtime,
    row: dict[str, Any],
    donor_rows: tuple[dict[str, Any], dict[str, Any]],
    case_by_key: dict[tuple[str, int], EvaluationCase],
    hidden_path: str | Path,
) -> dict[str, Any]:
    case = case_by_key[(str(row["item_id"]), int(row["prior_index"]))]
    token_ids = canonical_answer_token_ids(runtime.generator.tokenizer, case.answer_classes)
    sources = condition_sources(case, row, donor_rows, case_by_key)
    full_measurement, full_hidden = _pre_answer_condition(
        runtime,
        case,
        sources["full"],
        None,
        token_ids,
        capture_hidden=True,
    )
    selection = full_measurement
    selection_rendered_hash = str(full_measurement["prefix_rendered_hash"])
    answer = str(selection["predicted_answer"])
    measurements: dict[str, dict[str, Any]] = {"full": full_measurement}
    for condition in PANEL_CONDITIONS[1:]:
        measured, condition_hidden = _pre_answer_condition(
            runtime,
            case,
            sources[condition],
            answer,
            token_ids,
            capture_hidden=condition == "full",
        )
        measurements[condition] = measured
        if condition_hidden is not None:
            raise RuntimeError(f"Non-Full condition unexpectedly captured hidden: {condition}")
    if full_hidden is None:
        raise RuntimeError("Full answer-only hidden state is missing")
    hidden_array = full_hidden
    atomic_save_npz(
        hidden_path,
        positions=np.asarray(POSITIONS),
        layers=np.arange(runtime.modules.num_hidden_layers, dtype=np.int64),
        hidden=hidden_array,
    )
    effects = fixed_answer_effects(measurements, answer)
    answer_side = (
        "image"
        if answer == str(normalize_answer(row["image_answer"]))
        else "text"
        if answer == str(normalize_answer(row["text_answer"]))
        else "other"
    )
    return {
        "answer_star": answer,
        "answer_only_answer": answer,
        "answer_star_side": answer_side,
        "full_margin": float(selection["top1_top2_logit_margin"]),
        "selection": selection,
        "selection_rendered_hash": selection_rendered_hash,
        "measurements": measurements,
        "condition_sources": {
            condition: {
                key: value
                for key, value in sources[condition].items()
                if key != "text_clue"
            }
            for condition in PANEL_CONDITIONS
        },
        "selection_measurement_same_forward": True,
        "teacher_forced_causal_prefix_equal": bool(
            measurements["full"]["teacher_forced_causal_prefix_audit"]["passed"]
        ),
        "teacher_forced_length_path_max_logit_error": float(
            measurements["full"]["teacher_forced_length_path_max_logit_error"]
        ),
        "teacher_forced_length_path_probability_tv": float(
            measurements["full"]["teacher_forced_length_path_probability_tv"]
        ),
        "measurement_method_version": MEASUREMENT_METHOD_VERSION,
        "hidden_file": str(Path(hidden_path).name),
        "hidden_shape": list(hidden_array.shape),
        "hidden_dtype": str(hidden_array.dtype),
        "all_layer_hook_count": runtime.modules.num_hidden_layers,
        "verbal_sa_leakage": any(
            bool(measurement["verbal_sa_leakage"])
            for measurement in measurements.values()
        ),
        **effects,
        "deletion": effects["behavior_delete_imageward"],
        "replacement_d1": effects["behavior_replace_imageward_d1"],
        "replacement_d2": effects["behavior_replace_imageward_d2"],
        "replacement_mean": effects["behavior_replace_imageward"],
    }


def _prior_strength_from_row(row: dict[str, Any]) -> float:
    return float(row.get("prior_strength", 0.0))


def nuisance_feature_spec(answer_vocabulary: Sequence[str]) -> dict[str, Any]:
    vocabulary = sorted({str(normalize_answer(value)) for value in answer_vocabulary})
    if len(vocabulary) < 2:
        raise ValueError("Nuisance answer vocabulary must contain at least two labels")
    reference = vocabulary[0]
    names = [*NUISANCE_BASE_FEATURES, *[f"answer={label}" for label in vocabulary if label != reference]]
    return {"answer_vocabulary": vocabulary, "answer_reference": reference, "feature_names": names}


def nuisance_vector(row: dict[str, Any], specification: dict[str, Any]) -> np.ndarray:
    answer = str(normalize_answer(row["answer_star"]))
    vocabulary = list(specification["answer_vocabulary"])
    if answer not in vocabulary:
        raise ValueError(f"Answer {answer!r} is outside frozen nuisance vocabulary")
    side = str(row["answer_star_side"])
    reference = str(specification["answer_reference"])
    return np.asarray(
        [
            1.0,
            float(side == "image"),
            float(side == "other"),
            float(str(row["difficulty"]) == "hard"),
            _prior_strength_from_row(row),
            *[float(answer == label) for label in vocabulary if label != reference],
        ],
        dtype=np.float64,
    )


def fit_development_calibration(
    rows: Sequence[dict[str, Any]],
    answer_vocabulary: Sequence[str],
) -> dict[str, Any]:
    if len(rows) < 10:
        raise ValueError("Too few development rows for cross-fit calibration")
    specification = nuisance_feature_spec(answer_vocabulary)
    design = np.stack([nuisance_vector(row, specification) for row in rows])
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    calibration: dict[str, Any] = {
        "format_version": 1,
        "definition": "training-fold nuisance residualization and equal-family aggregation",
        "nuisance": specification,
        "method_columns": {
            "deletion": "behavior_delete_imageward",
            "replacement": "behavior_replace_imageward",
        },
        "aggregation": "replacement=(donor1+donor2)/2; shared=(z_deletion+z_replacement)/2",
        "folds": {},
    }
    for fold in sorted(set(folds.tolist())):
        train = folds != fold
        test = folds == fold
        train_items = {str(rows[index]["item_id"]) for index in np.flatnonzero(train)}
        test_items = {str(rows[index]["item_id"]) for index in np.flatnonzero(test)}
        if train_items.intersection(test_items):
            raise RuntimeError(f"Item leakage in reliance calibration fold {fold}")
        entry: dict[str, Any] = {
            "train_n": int(train.sum()),
            "test_n": int(test.sum()),
            "train_items_sha256": stable_hash(sorted(train_items, key=_numeric_item_key)),
            "test_items_sha256": stable_hash(sorted(test_items, key=_numeric_item_key)),
            "methods": {},
        }
        for method, column in calibration["method_columns"].items():
            outcome = np.asarray([float(row[column]) for row in rows], dtype=np.float64)
            beta = np.linalg.lstsq(design[train], outcome[train], rcond=None)[0]
            residual_train = outcome[train] - design[train] @ beta
            raw_sd = float(np.std(outcome[train], ddof=1))
            graded_sd = float(np.std(residual_train, ddof=1))
            if raw_sd <= 0.0 or graded_sd <= 0.0:
                raise RuntimeError(f"Degenerate {method} scale in fold {fold}")
            entry["methods"][method] = {
                "raw_mean": float(np.mean(outcome[train])),
                "raw_sd": raw_sd,
                "nuisance_beta": beta.tolist(),
                "graded_mean": float(np.mean(residual_train)),
                "graded_sd": graded_sd,
            }
        calibration["folds"][str(fold)] = entry
    calibration["calibration_fingerprint"] = stable_hash(calibration)
    return calibration


def apply_frozen_calibration(
    rows: Sequence[dict[str, Any]], calibration: dict[str, Any]
) -> list[dict[str, Any]]:
    expected = dict(calibration)
    fingerprint = str(expected.pop("calibration_fingerprint"))
    if stable_hash(expected) != fingerprint:
        raise ValueError("Frozen calibration fingerprint mismatch")
    specification = calibration["nuisance"]
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        fold = str(int(row["fold"]))
        if fold not in calibration["folds"]:
            raise ValueError(f"Frozen calibration omits fold {fold}")
        vector = nuisance_vector(row, specification)
        raw_z: dict[str, float] = {}
        graded_z: dict[str, float] = {}
        residual: dict[str, float] = {}
        for method, column in calibration["method_columns"].items():
            parameters = calibration["folds"][fold]["methods"][method]
            value = float(row[column])
            raw_z[method] = (value - float(parameters["raw_mean"])) / float(parameters["raw_sd"])
            beta = np.asarray(parameters["nuisance_beta"], dtype=np.float64)
            residual[method] = value - float(vector @ beta)
            graded_z[method] = (
                residual[method] - float(parameters["graded_mean"])
            ) / float(parameters["graded_sd"])
        row.update(
            {
                "raw_z_delete": raw_z["deletion"],
                "raw_z_replace": raw_z["replacement"],
                "reliance_raw_shared": 0.5 * (raw_z["deletion"] + raw_z["replacement"]),
                "reliance_raw_method_disagreement": 0.5 * (raw_z["deletion"] - raw_z["replacement"]),
                "graded_residual_delete": residual["deletion"],
                "graded_residual_replace": residual["replacement"],
                "graded_z_delete": graded_z["deletion"],
                "graded_z_replace": graded_z["replacement"],
                "reliance_graded_shared": 0.5 * (graded_z["deletion"] + graded_z["replacement"]),
                "reliance_graded_method_disagreement": 0.5 * (graded_z["deletion"] - graded_z["replacement"]),
                "calibration_fingerprint": fingerprint,
            }
        )
        output.append(row)
    return output


def _association(rows: Sequence[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    valid = [
        row
        for row in rows
        if row.get(left) is not None
        and row.get(right) is not None
        and np.isfinite(float(row[left]))
        and np.isfinite(float(row[right]))
    ]
    if len(valid) < 3:
        return {"n": len(valid), "pearson": None, "spearman": None, "spearman_item_bootstrap": None}
    x = np.asarray([float(row[left]) for row in valid])
    y = np.asarray([float(row[right]) for row in valid])
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return {"n": len(valid), "pearson": None, "spearman": None, "spearman_item_bootstrap": None}
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


def _sign_agreement(rows: Sequence[dict[str, Any]], left: str, right: str) -> float:
    pairs = [
        (float(row[left]), float(row[right]))
        for row in rows
        if float(row[left]) != 0.0 and float(row[right]) != 0.0
    ]
    return statistics.fmean((a > 0.0) == (b > 0.0) for a, b in pairs) if pairs else float("nan")


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
    return None if denominator <= 0 else float((ms_between - ms_error) / denominator)


def summarize_measurement(
    rows: Sequence[dict[str, Any]],
    *,
    split: str,
    failed: int,
    excluded: int = 0,
) -> dict[str, Any]:
    raw = _association(rows, "behavior_delete_imageward", "behavior_replace_imageward")
    graded = _association(rows, "graded_residual_delete", "graded_residual_replace")
    donor = _association(rows, "behavior_replace_imageward_d1", "behavior_replace_imageward_d2")
    raw_sign = _sign_agreement(rows, "behavior_delete_imageward", "behavior_replace_imageward")
    graded_sign = _sign_agreement(rows, "graded_residual_delete", "graded_residual_replace")
    donor_sign = _sign_agreement(rows, "behavior_replace_imageward_d1", "behavior_replace_imageward_d2")
    donor_icc = _icc_consistency(
        np.asarray(
            [
                [float(row["behavior_replace_imageward_d1"]), float(row["behavior_replace_imageward_d2"])]
                for row in rows
            ]
        )
    )
    fold_metrics: list[dict[str, Any]] = []
    for fold in sorted({int(row["fold"]) for row in rows}):
        selected = [row for row in rows if int(row["fold"]) == fold]
        fold_raw = _association(selected, "behavior_delete_imageward", "behavior_replace_imageward")
        fold_graded = _association(selected, "graded_residual_delete", "graded_residual_replace")
        fold_metrics.append(
            {
                "fold": fold,
                "n": len(selected),
                "raw_spearman": fold_raw["spearman"],
                "graded_spearman": fold_graded["spearman"],
            }
        )
    expected_n = DEVELOPMENT_N if split == "development" else CONFIRMATORY_N
    minimum_n = 90 if split == "development" else 70
    leakage_n = sum(bool(row.get("verbal_sa_leakage")) for row in rows)
    invalid_hidden_n = sum(
        row.get("hidden_shape", [])[0:2] != [len(POSITIONS), 28]
        or row.get("hidden_dtype") != "float16"
        for row in rows
    )
    non_source_endpoint_n = sum(row.get("answer_star_side") == "other" for row in rows)
    causal_prefix_failure_n = sum(
        not bool(row.get("teacher_forced_causal_prefix_equal")) for row in rows
    )
    selection_reuse_failure_n = sum(
        not bool(row.get("selection_measurement_same_forward")) for row in rows
    )
    max_length_path_error = max(
        (float(row["teacher_forced_length_path_max_logit_error"]) for row in rows),
        default=None,
    )
    max_length_path_probability_tv = max(
        (float(row["teacher_forced_length_path_probability_tv"]) for row in rows),
        default=None,
    )
    raw_ci = raw.get("spearman_item_bootstrap") or {"ci95": [None, None]}
    graded_ci = graded.get("spearman_item_bootstrap") or {"ci95": [None, None]}
    donor_ci = donor.get("spearman_item_bootstrap") or {"ci95": [None, None]}
    positive_raw_folds = sum(
        metric["raw_spearman"] is not None and metric["raw_spearman"] > 0.0
        for metric in fold_metrics
    )
    positive_graded_folds = sum(
        metric["graded_spearman"] is not None and metric["graded_spearman"] > 0.0
        for metric in fold_metrics
    )
    raw_alpha = _cronbach_two_indicator(raw["pearson"])
    graded_alpha = _cronbach_two_indicator(graded["pearson"])
    technical_gate = bool(
        len(rows) >= minimum_n
        and failed == 0
        and len(rows) + excluded == expected_n
        and excluded <= expected_n - minimum_n
        and leakage_n == 0
        and invalid_hidden_n == 0
        and causal_prefix_failure_n == 0
        and selection_reuse_failure_n == 0
    )
    raw_gate = bool(
        raw_ci["ci95"][0] is not None
        and raw_ci["ci95"][0] > 0.0
        and raw_sign >= 0.70
        and raw_alpha is not None
        and raw_alpha >= 0.60
        and positive_raw_folds >= 4
    )
    graded_gate = bool(
        graded_ci["ci95"][0] is not None
        and graded_ci["ci95"][0] > 0.0
        and graded_sign >= 0.60
        and graded_alpha is not None
        and graded_alpha >= 0.60
        and positive_graded_folds >= 4
    )
    donor_gate = bool(
        donor_ci["ci95"][0] is not None
        and donor_ci["ci95"][0] > 0.0
        and donor_sign >= 0.60
        and donor_icc is not None
        and donor_icc >= 0.60
    )
    measurement_gate = bool(technical_gate and raw_gate and graded_gate and donor_gate)
    return {
        "title": f"Clean Actual Source Reliance — {split}",
        "status": "completed",
        "split": split,
        "planned": expected_n,
        "attempted": len(rows) + failed + excluded,
        "completed": len(rows),
        "excluded": excluded,
        "failed": failed,
        "unique_items": len({str(row["item_id"]) for row in rows}),
        "technical": {
            "verbal_sa_leakage_n": leakage_n,
            "invalid_hidden_n": invalid_hidden_n,
            "answer_star_other_n": non_source_endpoint_n,
            "teacher_forced_causal_prefix_failure_n": causal_prefix_failure_n,
            "selection_measurement_reuse_failure_n": selection_reuse_failure_n,
            "max_teacher_forced_length_path_logit_error_diagnostic": max_length_path_error,
            "max_teacher_forced_length_path_probability_tv_diagnostic": max_length_path_probability_tv,
            "gate_passed": technical_gate,
        },
        "raw_reliability": {
            "delete_vs_replace": raw,
            "sign_agreement": raw_sign,
            "cronbach_alpha_two_indicator": raw_alpha,
            "positive_fold_count": positive_raw_folds,
            "gate_passed": raw_gate,
        },
        "graded_reliability": {
            "delete_vs_replace": graded,
            "sign_agreement": graded_sign,
            "cronbach_alpha_two_indicator": graded_alpha,
            "positive_fold_count": positive_graded_folds,
            "gate_passed": graded_gate,
        },
        "donor_replicate_reliability": {
            "donor1_vs_donor2": donor,
            "sign_agreement": donor_sign,
            "icc_consistency": donor_icc,
            "gate_passed": donor_gate,
        },
        "fold_metrics": fold_metrics,
        "measurement_gate_passed": measurement_gate,
        "gate_rule": {
            "technical": f"planned={expected_n}, terminal={expected_n}, evaluable>={minimum_n}, failures=0, no SA leakage, valid hidden, exact causal-prefix inputs, and Full selection reused as the behavior measurement",
            "raw": "Spearman bootstrap lower>0, sign>=.70, alpha>=.60, >=4/5 positive folds",
            "graded": "Spearman bootstrap lower>0, sign>=.60, alpha>=.60, >=4/5 positive folds",
            "donor": "Spearman bootstrap lower>0, sign>=.60, ICC>=.60",
        },
        "claim": (
            "cross-method Actual Source Reliance target validated"
            if measurement_gate
            else "Actual Source Reliance target remains provisional; do not fit a causal reliance mediator"
        ),
    }


def build_split_manifest(
    split: str,
    cohort: Sequence[dict[str, Any]],
    donors: dict[str, tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    rows = []
    for row in cohort:
        d1, d2 = donors[str(row["case_id"])]
        rows.append(
            {
                "case_id": row["case_id"],
                "item_id": row["item_id"],
                "prior_index": row["prior_index"],
                "condition": row["condition"],
                "fold": row["fold"],
                "donor1_case_id": d1["case_id"],
                "donor1_item_id": d1["item_id"],
                "donor2_case_id": d2["case_id"],
                "donor2_item_id": d2["item_id"],
            }
        )
    payload = {
        "format_version": 1,
        "split": split,
        "n": len(rows),
        "conditions": list(PANEL_CONDITIONS),
        "prompt": "V4 full-evidence answer-only; no Source Attribution request",
        "answer_star": "restricted natural top-1 from this answer-only full context",
        "replacement_rule": "same matched donor supplies donor text and donor image within each replicate",
        "donor_selection": "authoritative source pool; same fold/difficulty/final-side; prefer prior-bin and condition; distinct items; deterministic",
        "rows": rows,
    }
    if split == "confirmatory":
        payload["selection_audit"] = {
            "remaining_unique_after_development": 78,
            "eligible_unique": CONFIRMATORY_N,
            "excluded_unique": 1,
            "excluded_item_ids": ["34"],
            "reason": "item 34 has no conflict_* source-use row, so it fails the frozen conflict eligibility rule",
        }
    payload["manifest_fingerprint"] = stable_hash(payload)
    return payload


def write_summary_markdown(directory: Path, summary: dict[str, Any]) -> None:
    raw = summary["raw_reliability"]
    graded = summary["graded_reliability"]
    donor = summary["donor_replicate_reliability"]
    lines = [
        f"# {summary['title']}",
        "",
        f"- Evaluable: {summary['completed']}/{summary['planned']} unique-item cases; structurally excluded={summary['excluded']}; failed={summary['failed']}.",
        f"- Technical gate: {summary['technical']['gate_passed']}.",
        f"- Raw delete↔replace Spearman: {raw['delete_vs_replace']['spearman']}; sign={raw['sign_agreement']}; gate={raw['gate_passed']}.",
        f"- Graded delete↔replace Spearman: {graded['delete_vs_replace']['spearman']}; sign={graded['sign_agreement']}; gate={graded['gate_passed']}.",
        f"- Donor replicate Spearman: {donor['donor1_vs_donor2']['spearman']}; ICC={donor['icc_consistency']}; gate={donor['gate_passed']}.",
        f"- Overall measurement gate: {summary['measurement_gate_passed']}.",
        "",
        summary["claim"],
        "",
    ]
    atomic_write_text(directory / f"{summary['split']}_summary.md", "\n".join(lines))


def run_reliance_panel(
    runtime: Stage3Runtime,
    artifacts: SAFormationArtifacts,
    output: Path,
    *,
    split: str,
    deadline: Callable[[], None],
) -> dict[str, Any]:
    cohort, donors, case_by_key = plan_split(artifacts, split)
    manifest = build_split_manifest(split, cohort, donors)
    manifest_path = output / f"{split}_cohort_manifest.json"
    if manifest_path.is_file():
        existing = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("manifest_fingerprint") != manifest["manifest_fingerprint"]:
            raise ValueError(f"{split} cohort/donor manifest fingerprint changed")
    else:
        atomic_write_json(manifest_path, manifest)
    result_path = output / f"{split}_results.jsonl"
    terminal_keys = {
        str(row["intervention_key"])
        for row in _latest_rows(result_path)
        if int(row.get("measurement_method_version", -1)) == MEASUREMENT_METHOD_VERSION
        and row.get("status") in {"completed", "excluded"}
    }
    hidden_dir = output / "hidden" / split
    hidden_dir.mkdir(parents=True, exist_ok=True)
    for row in cohort:
        deadline()
        key = f"actual_reliance|{split}|{row['case_id']}"
        if key in terminal_keys:
            continue
        d1, d2 = donors[str(row["case_id"])]
        base = {
            "intervention_key": key,
            "experiment": "clean_actual_source_reliance",
            "split": split,
            "case_id": row["case_id"],
            "item_id": row["item_id"],
            "prior_index": int(row["prior_index"]),
            "condition": row["condition"],
            "difficulty": row["difficulty"],
            "fold": int(row["fold"]),
            "text_answer": row["text_answer"],
            "image_answer": row["image_answer"],
            "prior_strength": _prior_strength_from_row(row),
            "donor1_case_id": d1["case_id"],
            "donor1_item_id": d1["item_id"],
            "donor2_case_id": d2["case_id"],
            "donor2_item_id": d2["item_id"],
            "manifest_fingerprint": manifest["manifest_fingerprint"],
            "measurement_method_version": MEASUREMENT_METHOD_VERSION,
        }

        def operation() -> dict[str, Any]:
            hidden_path = hidden_dir / f"{row['case_id']}.npz"
            measured = measure_case(runtime, row, (d1, d2), case_by_key, hidden_path)
            return {
                **base,
                "status": "completed",
                **measured,
                "hidden_file": str(hidden_path.relative_to(output)),
            }

        append_jsonl(result_path, _safe_record(base, operation))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    latest = [
        row
        for row in _latest_rows(result_path)
        if int(row.get("measurement_method_version", -1)) == MEASUREMENT_METHOD_VERSION
    ]
    completed = [row for row in latest if row.get("status") == "completed"]
    excluded_rows = [row for row in latest if row.get("status") == "excluded"]
    failed_rows = [row for row in latest if row.get("status") == "failed"]
    failed = len(failed_rows)
    excluded = len(excluded_rows)
    expected = DEVELOPMENT_N if split == "development" else CONFIRMATORY_N
    if len(latest) != expected:
        raise RuntimeError(
            f"{split} panel reached {len(latest)}/{expected} terminal/latest method-v2 rows"
        )
    if failed:
        raise RuntimeError(
            f"{split} panel has {failed} retryable technical failures; resume before analysis"
        )
    answer_vocabulary = sorted(
        {
            str(normalize_answer(label))
            for case in case_by_key.values()
            for label in case.answer_classes
        }
    )
    frozen_path = output / "frozen_measurement_rule.json"
    if split == "development":
        calibration = fit_development_calibration(completed, answer_vocabulary)
    else:
        if not frozen_path.is_file():
            raise ValueError("Confirmatory scoring requires frozen development rule")
        frozen = __import__("json").loads(frozen_path.read_text(encoding="utf-8"))
        calibration = frozen["calibration"]
    analysis = apply_frozen_calibration(completed, calibration)
    write_jsonl_atomic(output / f"{split}_analysis.jsonl", analysis)
    summary = summarize_measurement(
        analysis,
        split=split,
        failed=failed,
        excluded=excluded,
    )
    atomic_write_json(output / f"{split}_summary.json", summary)
    write_summary_markdown(output, summary)
    if split == "development":
        frozen = {
            "format_version": 1,
            "development_manifest_fingerprint": manifest["manifest_fingerprint"],
            "measurement_formula": {
                "deletion": "log P(A*|no_text)-log P(A*|no_image)",
                "replacement_donor": "log P(A*|replace_text_d)-log P(A*|replace_image_d)",
                "replacement": "mean of two deterministic donor replicates",
                "shared": "equal-family average of fold-standardized deletion and replacement",
                "method_disagreement": "half difference of fold-standardized deletion and replacement",
            },
            "verbal_sa_used": False,
            "measurement_method_version": MEASUREMENT_METHOD_VERSION,
            "structural_exclusions": [
                {
                    "case_id": row["case_id"],
                    "item_id": row["item_id"],
                    "reason": row["exclusion_reason"],
                    "endpoint_audit": row.get("endpoint_audit"),
                }
                for row in excluded_rows
            ],
            "calibration": calibration,
            "development_measurement_gate_passed": summary["measurement_gate_passed"],
            "confirmatory_allowed": summary["measurement_gate_passed"],
        }
        frozen["rule_fingerprint"] = stable_hash(frozen)
        atomic_write_json(frozen_path, frozen)
    aggregate: dict[str, Any] = {}
    for name in ("development", "confirmatory"):
        path = output / f"{name}_summary.json"
        if path.is_file():
            aggregate[name] = __import__("json").loads(
                path.read_text(encoding="utf-8")
            )
    atomic_write_json(output / "summary.json", aggregate)
    return summary


def verify_confirmatory_allowed(output: str | Path) -> dict[str, Any]:
    path = Path(output) / "frozen_measurement_rule.json"
    if not path.is_file():
        raise ValueError("Confirmatory split requires completed development calibration")
    frozen = __import__("json").loads(path.read_text(encoding="utf-8"))
    fingerprint = str(frozen.get("rule_fingerprint", ""))
    payload = dict(frozen)
    payload.pop("rule_fingerprint", None)
    if stable_hash(payload) != fingerprint:
        raise ValueError("Frozen measurement rule fingerprint mismatch")
    if not frozen.get("confirmatory_allowed"):
        raise ValueError("Development measurement gate failed; confirmatory split is not allowed")
    return frozen
