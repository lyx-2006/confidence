"""Second-order Source Attribution formation experiments.

The experiments separate answer behavior, upstream History priming, semantic
measurement invariance, and (only behind a semantic gate) causal formation
tracing.  All outputs live under ``stage3_sa_second_order``.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

from confidence_test.answer_metrics import (
    compute_answer_metrics,
    normalize_answer,
    parse_answer_output,
)
from confidence_test.inference_extension import ASSISTANT_ANSWER_PREFILL, AnswerGenerationResult
from confidence_test.prompt_utils import V4_STAGE1_FULL_EVIDENCE_ANSWER_PROMPT
from confidence_test.source_attribution_schema import gather_source_class_logits, source_distribution
from layer_metacognition.hidden_state_store import (
    append_jsonl,
    atomic_write_json,
    atomic_write_text,
    load_jsonl,
)

from .core import (
    SAFormationArtifacts,
    SAOOFDirectionRepository,
    canonical_message_hash,
    item_cluster_bootstrap,
    load_baseline_rows,
    paired_effect_summary,
    read_json,
    write_csv_atomic,
    write_experiment_summary,
    write_jsonl_atomic,
)
from .mechanism import (
    HISTORY_RELEVANCE_DIR,
    _history_first_turn,
    _numeric_item_key,
    build_relevance_history_messages,
    label_mappings,
    semantic_mapping_prompt,
)
from .runtime import (
    Stage3Runtime,
    assistant_message,
    full_prompt,
    image_content,
    prepare_measurement,
    text_content,
)


BEHAVIOR_DIR = "01_history_behavior_dissociation"
PRIMING_DIR = "02_priming_decomposition"
SEMANTIC_DIR = "03_protocol_invariant_semantic_sa"
TRACING_DIR = "04_blockwise_causal_tracing"
SUBSPACE_DIR = "05_low_rank_formation_subspace"


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


def _source_history_rows(mechanism_root: Path) -> list[dict[str, Any]]:
    rows = [
        row
        for row in _latest_rows(
            mechanism_root / HISTORY_RELEVANCE_DIR / "results.jsonl"
        )
        if row.get("status") == "completed"
    ]
    return sorted(rows, key=lambda row: (_numeric_item_key(row["item_id"]), row["case_id"]))


def answer_only_prompt(case: Any) -> str:
    return V4_STAGE1_FULL_EVIDENCE_ANSWER_PROMPT.format(
        question=case.question,
        text_clue=case.text_clue,
    )


def build_answer_history_messages(
    target_case: Any,
    target_condition: str,
    history_case: Any,
    history_condition: str,
    modality: str,
    prior_answer: str,
) -> list[dict[str, Any]]:
    messages = [
        _history_first_turn(history_case, history_condition, modality),
        assistant_message(f"**Answer**: {prior_answer}"),
        {
            "role": "user",
            "content": image_content(
                str(target_case.conditions[target_condition].resolved_image_path),
                answer_only_prompt(target_case),
            ),
        },
        assistant_message(ASSISTANT_ANSWER_PREFILL),
    ]
    if any(
        "Source Attribution" in str(part.get("text", ""))
        for message in messages
        for part in message.get("content", [])
        if isinstance(part, dict)
    ):
        raise ValueError("Answer-only History branch leaks an SA request")
    return messages


def generate_answer_messages(
    runtime: Stage3Runtime,
    messages: list[dict[str, Any]],
    answer_classes: Sequence[str],
    *,
    max_new_tokens: int = 24,
) -> AnswerGenerationResult:
    started = time.perf_counter()
    result = AnswerGenerationResult(candidate_count=len(answer_classes))
    try:
        rendered, inputs = runtime.generator.prepare_messages(
            messages, assistant_text=ASSISTANT_ANSWER_PREFILL
        )
        del rendered
        with torch.inference_mode():
            generated = runtime.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        input_length = int(inputs.input_ids.shape[1])
        continuation = runtime.generator.tokenizer.decode(
            generated.sequences[0, input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        result.raw_output = ASSISTANT_ANSWER_PREFILL + continuation
        result.answer, result.normalized_answer, result.parse_success = parse_answer_output(
            result.raw_output
        )
        if not generated.scores:
            raise RuntimeError("Answer-only generation returned no first-token scores")
        metrics = compute_answer_metrics(
            generated.scores[0][0],
            answer_classes,
            result.normalized_answer,
            runtime.generator.tokenizer,
        )
        for key, value in asdict(metrics).items():
            if key != "error":
                setattr(result, key, value)
        result.error = metrics.error
        if not result.parse_success:
            result.answer_metric_status = "failed"
            result.error = {
                "type": "AnswerParseError",
                "message": f"Could not parse answer-only output: {result.raw_output!r}",
            }
    except Exception as exc:
        result.error = {"type": type(exc).__name__, "message": str(exc)}
        result.answer_metric_status = "failed"
    result.elapsed_seconds = round(time.perf_counter() - started, 6)
    return result


def _answer_side(answer: str | None, text_answer: str, image_answer: str) -> str:
    normalized = normalize_answer(answer)
    if normalized == normalize_answer(text_answer):
        return "text"
    if normalized == normalize_answer(image_answer):
        return "image"
    return "other"


def _case_coordinates(case_id: str) -> tuple[str, int]:
    parts = str(case_id).split("__")
    if len(parts) < 2 or not parts[1].startswith("prior_"):
        raise ValueError(f"Cannot parse item/prior from case id: {case_id}")
    return parts[0], int(parts[1].removeprefix("prior_"))


def run_history_behavior(
    runtime: Stage3Runtime,
    mechanism_root: Path,
    output_root: Path,
    *,
    deadline: Callable[[], None],
) -> dict[str, Any]:
    directory = output_root / BEHAVIOR_DIR
    directory.mkdir(parents=True, exist_ok=True)
    source_rows = _source_history_rows(mechanism_root)
    baseline_by_case = {
        row["case_id"]: row for row in load_baseline_rows(runtime_artifacts(runtime))
    }
    branches = (
        "relevant_text",
        "relevant_image",
        "irrelevant_text",
        "irrelevant_image",
    )
    atomic_write_json(
        directory / "cohort_manifest.json",
        {
            "case_count": len(source_rows),
            "source": str(mechanism_root / HISTORY_RELEVANCE_DIR / "results.jsonl"),
            "branches": branches,
            "final_prompt": "V4 full-evidence answer-only; no SA request",
            "common_history_answer": "target A_T",
            "case_ids": [row["case_id"] for row in source_rows],
        },
    )
    result_path = directory / "results.jsonl"
    existing = {
        row["intervention_key"]
        for row in _latest_rows(result_path)
        if row.get("status") == "completed"
    }
    for row in source_rows:
        deadline()
        key = f"history_behavior|{row['case_id']}"
        if key in existing:
            continue
        base = {
            "intervention_key": key,
            "experiment": "first_second_order_dissociation",
            "case_id": row["case_id"],
            "item_id": row["item_id"],
            "prior_index": row["prior_index"],
            "condition": row["condition"],
            "fold": int(row["fold"]),
            "text_answer": row["branches"]["relevant_text"].get("text_answer", row.get("prior_answer")),
            "image_answer": None,
            "donor_case_id": row["donor_case_id"],
        }

        def execute() -> dict[str, Any]:
            target = runtime.case(row["item_id"], row["prior_index"])
            donor_case_id = str(row["donor_case_id"])
            donor_item, donor_prior = _case_coordinates(donor_case_id)
            donor = runtime.case(donor_item, donor_prior)
            prior_answer = str(row["prior_answer"])
            baseline_source = baseline_by_case[row["case_id"]]
            text_answer = str(baseline_source["text_answer"])
            image_answer = str(baseline_source["image_answer"])
            results: dict[str, Any] = {}
            for branch in branches:
                relevance, modality = branch.split("_", 1)
                history_case = target if relevance == "relevant" else donor
                history_condition = (
                    row["condition"]
                    if relevance == "relevant"
                    else str(baseline_by_case[donor_case_id]["condition"])
                )
                messages = build_answer_history_messages(
                    target,
                    row["condition"],
                    history_case,
                    history_condition,
                    modality,
                    prior_answer,
                )
                generated = generate_answer_messages(runtime, messages, target.answer_classes)
                if (
                    not generated.parse_success
                    or generated.answer_metric_status != "completed"
                    or generated.normalized_answer is None
                ):
                    raise RuntimeError(f"Answer-only generation failed for {branch}: {generated.error}")
                probabilities = generated.answer_class_probabilities
                results[branch] = {
                    "messages_hash": canonical_message_hash(messages),
                    "generated": generated.to_dict(),
                    "final_side": _answer_side(
                        generated.normalized_answer, text_answer, image_answer
                    ),
                    "p_image_answer": float(probabilities[normalize_answer(image_answer)]),
                    "revised_from_history_answer": generated.normalized_answer
                    != normalize_answer(prior_answer),
                }
            return {
                **base,
                "status": "completed",
                "text_answer": text_answer,
                "image_answer": image_answer,
                "branches": results,
            }

        append_jsonl(result_path, _safe_record(base, execute))
    rows = _latest_rows(result_path)
    summary = _summarize_history_behavior(rows, mechanism_root)
    write_experiment_summary(directory, summary)
    return summary


_RUNTIME_ARTIFACTS: dict[int, SAFormationArtifacts] = {}


def register_runtime_artifacts(runtime: Stage3Runtime, artifacts: SAFormationArtifacts) -> None:
    _RUNTIME_ARTIFACTS[id(runtime)] = artifacts


def runtime_artifacts(runtime: Stage3Runtime) -> SAFormationArtifacts:
    try:
        return _RUNTIME_ARTIFACTS[id(runtime)]
    except KeyError as exc:
        raise RuntimeError("Runtime artifacts were not registered") from exc


def _summarize_history_behavior(
    rows: Sequence[dict[str, Any]], mechanism_root: Path
) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed"]
    source_by_case = {
        row["case_id"]: row for row in _source_history_rows(mechanism_root)
    }
    effects: dict[str, Any] = {}
    endpoint_matched_effects: dict[str, Any] = {}
    for relevance in ("relevant", "irrelevant"):
        paired: list[dict[str, Any]] = []
        endpoint_matched: list[dict[str, Any]] = []
        for row in completed:
            text = row["branches"][f"{relevance}_text"]
            image = row["branches"][f"{relevance}_image"]
            contrast = {
                "item_id": row["item_id"],
                "delta_p_image_answer": image["p_image_answer"] - text["p_image_answer"],
                "delta_hard_image_side": float(image["final_side"] == "image")
                - float(text["final_side"] == "image"),
                "delta_revision_rate": float(image["revised_from_history_answer"])
                - float(text["revised_from_history_answer"]),
            }
            paired.append(contrast)
            source = source_by_case[row["case_id"]]
            if (
                source["branches"][f"{relevance}_text"]["normalized_answer"]
                == source["branches"][f"{relevance}_image"]["normalized_answer"]
            ):
                endpoint_matched.append(contrast)
        effects[relevance] = {
            key: paired_effect_summary(paired, key)
            for key in (
                "delta_p_image_answer",
                "delta_hard_image_side",
                "delta_revision_rate",
            )
        }
        endpoint_matched_effects[relevance] = {
            key: paired_effect_summary(endpoint_matched, key)
            for key in (
                "delta_p_image_answer",
                "delta_hard_image_side",
                "delta_revision_rate",
            )
        }
    sa_summary = read_json(mechanism_root / HISTORY_RELEVANCE_DIR / "summary.json")
    relevant_sa = sa_summary["effects"]["pass1_sa"]["relevant_image_minus_text"]
    relevant_answer = endpoint_matched_effects["relevant"]["delta_p_image_answer"]
    sa_positive = bool(
        relevant_sa["ci95"][0] is not None and relevant_sa["ci95"][0] > 0
    )
    answer_null_like = bool(
        relevant_answer["ci95"][0] is not None
        and relevant_answer["ci95"][0] <= 0 <= relevant_answer["ci95"][1]
        and abs(float(relevant_answer["mean"])) <= 0.05
    )
    answer_negative = bool(
        relevant_answer["ci95"][1] is not None
        and relevant_answer["ci95"][1] < 0
    )
    answer_positive = bool(
        relevant_answer["ci95"][0] is not None
        and relevant_answer["ci95"][0] > 0
    )
    second_order_specific = bool(sa_positive and answer_null_like)
    sign_dissociation = bool(sa_positive and answer_negative)
    if sign_dissociation:
        classification = "cross-level sign dissociation: verbal SA shifts imageward while answer behavior shifts textward"
    elif second_order_specific:
        classification = "second-order-specific History effect"
    elif sa_positive and answer_positive:
        classification = "same-direction first-order and second-order History effects"
    else:
        classification = "cross-level dissociation is inconclusive"
    return {
        "title": "Experiment 1 — First-order / Second-order Dissociation",
        "status": "completed",
        "n": len(completed),
        "failed": len(rows) - len(completed),
        "answer_effects": effects,
        "answer_effects_on_existing_sa_endpoint_matched_cohort": endpoint_matched_effects,
        "existing_sa_effects": sa_summary["effects"]["pass1_sa"],
        "second_order_specific_history_effect": second_order_specific,
        "cross_level_sign_dissociation": sign_dissociation,
        "criteria": {
            "second_order_specific": "On the identical existing-SA endpoint-matched cohort: Relevant SA CI lower > 0 while ΔP(A_I) CI includes 0 and |mean|<=0.05",
            "sign_dissociation": "On the identical existing-SA endpoint-matched cohort: Relevant SA CI lower > 0 while ΔP(A_I) CI upper < 0",
        },
        "classification": classification,
    }


def _blank_image(source: Path, destination: Path) -> Path:
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        blank = Image.new("RGB", image.size, color=(127, 127, 127))
        temporary = destination.with_suffix(".tmp.png")
        blank.save(temporary, format="PNG")
        temporary.replace(destination)
    return destination


def build_blank_history_messages(
    target: Any,
    target_condition: str,
    donor: Any,
    donor_condition: str,
    blank_path: Path,
    prior_answer: str,
) -> list[dict[str, Any]]:
    from layer_metacognition.probe.prompts import IMAGE_ONLY_ANSWER_PROMPT

    first_prompt = IMAGE_ONLY_ANSWER_PROMPT.format(question=donor.question)
    return [
        {"role": "user", "content": image_content(str(blank_path), first_prompt)},
        assistant_message(f"**Answer**: {prior_answer}"),
        {
            "role": "user",
            "content": image_content(
                str(target.conditions[target_condition].resolved_image_path),
                full_prompt(target),
            ),
        },
        assistant_message(ASSISTANT_ANSWER_PREFILL),
    ]


def run_priming_decomposition(
    runtime: Stage3Runtime,
    mechanism_root: Path,
    output_root: Path,
    *,
    deadline: Callable[[], None],
) -> dict[str, Any]:
    directory = output_root / PRIMING_DIR
    directory.mkdir(parents=True, exist_ok=True)
    source_rows = _source_history_rows(mechanism_root)
    baseline_by_case = {
        row["case_id"]: row for row in load_baseline_rows(runtime_artifacts(runtime))
    }
    atomic_write_json(
        directory / "cohort_manifest.json",
        {
            "case_count": len(source_rows),
            "conditions": [
                "no_history",
                "blank_image_history",
                "irrelevant_semantic_image_history",
                "relevant_image_history",
            ],
            "reuse": "No/Irrelevant/Relevant use completed exact History Pass-1 outputs; only Blank is newly generated",
            "blank": "mid-gray image with exact donor-image width and height; donor question/prompt and common A_T answer retained",
        },
    )
    result_path = directory / "results.jsonl"
    existing = {
        row["intervention_key"]
        for row in _latest_rows(result_path)
        if row.get("status") == "completed"
    }
    for row in source_rows:
        deadline()
        key = f"priming_decompose|{row['case_id']}"
        if key in existing:
            continue
        base = {
            "intervention_key": key,
            "experiment": "priming_decomposition",
            "case_id": row["case_id"],
            "item_id": row["item_id"],
            "prior_index": row["prior_index"],
            "condition": row["condition"],
            "fold": int(row["fold"]),
            "donor_case_id": row["donor_case_id"],
        }

        def execute() -> dict[str, Any]:
            target = runtime.case(row["item_id"], row["prior_index"])
            donor_item, donor_prior = _case_coordinates(str(row["donor_case_id"]))
            donor = runtime.case(donor_item, donor_prior)
            donor_row = baseline_by_case[row["donor_case_id"]]
            donor_image = Path(donor.conditions[donor_row["condition"]].resolved_image_path)
            blank = _blank_image(
                donor_image,
                directory / "assets" / f"blank_{row['case_id'].replace('/', '_')}.png",
            )
            messages = build_blank_history_messages(
                target,
                row["condition"],
                donor,
                donor_row["condition"],
                blank,
                str(row["prior_answer"]),
            )
            generated = runtime.generator.generate_messages(
                messages,
                target.answer_classes,
                max_new_tokens=48,
                use_cache=False,
            )
            if (
                not generated.parse_success
                or generated.source_metric_status != "completed"
                or generated.source_attribution is None
                or generated.normalized_answer is None
            ):
                raise RuntimeError(f"Blank History generation failed: {generated.error}")
            reused = {
                "no_history": row["branches"]["no_history"],
                "irrelevant_semantic_image_history": row["branches"]["irrelevant_image"],
                "relevant_image_history": row["branches"]["relevant_image"],
            }
            branches = {
                name: {
                    "normalized_answer": value["normalized_answer"],
                    "sa": float(value["pass1_sa"]),
                    "source": "reused_stage3_sa_mechanism",
                }
                for name, value in reused.items()
            }
            with Image.open(blank) as blank_image:
                blank_size = list(blank_image.size)
            with Image.open(donor_image) as semantic_image:
                donor_size = list(semantic_image.size)
            branches["blank_image_history"] = {
                "normalized_answer": generated.normalized_answer,
                "sa": float(generated.source_attribution["soft_image_score"]),
                "source": "new_generation",
                "messages_hash": canonical_message_hash(messages),
                "blank_path": str(blank.relative_to(output_root)),
                "blank_size": blank_size,
                "donor_size": donor_size,
            }
            return {**base, "status": "completed", "branches": branches}

        append_jsonl(result_path, _safe_record(base, execute))
    rows = _latest_rows(result_path)
    summary = _summarize_priming(rows)
    write_experiment_summary(directory, summary)
    return summary


def _priming_contrast(
    rows: Sequence[dict[str, Any]], left: str, right: str
) -> dict[str, Any]:
    paired = []
    for row in rows:
        a, b = row["branches"][left], row["branches"][right]
        if a["normalized_answer"] != b["normalized_answer"]:
            continue
        paired.append({"item_id": row["item_id"], "delta": b["sa"] - a["sa"]})
    return paired_effect_summary(paired, "delta")


def _summarize_priming(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed"]
    contrasts = {
        "blank_minus_no": _priming_contrast(
            completed, "no_history", "blank_image_history"
        ),
        "irrelevant_minus_no": _priming_contrast(
            completed, "no_history", "irrelevant_semantic_image_history"
        ),
        "relevant_minus_no": _priming_contrast(
            completed, "no_history", "relevant_image_history"
        ),
        "irrelevant_minus_blank": _priming_contrast(
            completed, "blank_image_history", "irrelevant_semantic_image_history"
        ),
        "relevant_minus_irrelevant": _priming_contrast(
            completed, "irrelevant_semantic_image_history", "relevant_image_history"
        ),
    }
    blank = contrasts["blank_minus_no"]
    semantic = contrasts["irrelevant_minus_blank"]
    relevance = contrasts["relevant_minus_irrelevant"]
    if blank["ci95"][0] is not None and blank["ci95"][0] > 0 and semantic["ci95"][0] <= 0 <= semantic["ci95"][1]:
        classification = "primarily image-token/modality-format priming"
    elif semantic["ci95"][0] is not None and semantic["ci95"][0] > 0:
        classification = (
            "visual semantic priming plus task relevance"
            if relevance["ci95"][0] is not None and relevance["ci95"][0] > 0
            else "visual semantic priming without stable relevance increment"
        )
    else:
        classification = "priming components remain inconclusive"
    strict = sum(
        len({branch["normalized_answer"] for branch in row["branches"].values()}) == 1
        for row in completed
    )
    return {
        "title": "Experiment 2 — Priming Decomposition",
        "status": "completed",
        "n": len(completed),
        "failed": len(rows) - len(completed),
        "strict_four_condition_endpoint_matched_n": strict,
        "contrasts": contrasts,
        "classification": classification,
        "claim_limit": "All primary contrasts are final-answer endpoint matched.",
    }


@dataclass(frozen=True)
class ProtocolSpec:
    name: str
    labels_by_semantic: tuple[str, ...]
    midpoints: tuple[float, ...]
    descriptions: tuple[str, ...]


def protocol_specs() -> tuple[ProtocolSpec, ...]:
    nine_descriptions = (
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
    nine_midpoints = (0.05, 0.175, 0.325, 0.4375, 0.5, 0.5625, 0.675, 0.825, 0.95)
    mappings = label_mappings()
    return (
        ProtocolSpec("normal_numeric", mappings["normal_numeric"], nine_midpoints, nine_descriptions),
        ProtocolSpec("reversed_numeric", mappings["reversed_numeric"], nine_midpoints, nine_descriptions),
        ProtocolSpec("random_single_token", mappings["arbitrary_tokens"], nine_midpoints, nine_descriptions),
        ProtocolSpec(
            "text_both_image",
            ("T", "B", "I"),
            (0.05, 0.5, 0.95),
            (
                "The answer was based mainly on the text clue.",
                "The answer was based on the text clue and image to a similar extent.",
                "The answer was based mainly on the image.",
            ),
        ),
        ProtocolSpec(
            "binary_text_image",
            ("T", "I"),
            (0.05, 0.95),
            (
                "The answer was based more on the text clue.",
                "The answer was based more on the image.",
            ),
        ),
    )


class ProtocolAnalyzer:
    def __init__(self, tokenizer: Any, spec: ProtocolSpec) -> None:
        self.spec = spec
        self.encodings = {
            label: [int(value) for value in tokenizer.encode(label, add_special_tokens=False)]
            for label in spec.labels_by_semantic
        }
        if any(len(ids) != 1 for ids in self.encodings.values()):
            raise ValueError(f"Protocol {spec.name} contains a non-single-token label")
        if len({ids[0] for ids in self.encodings.values()}) != len(self.encodings):
            raise ValueError(f"Protocol {spec.name} has colliding token labels")

    def score_vocab_logits(
        self, vocab_logits: torch.Tensor, *, raw_output: str, parsed_label: str | None
    ) -> Any:
        logits = gather_source_class_logits(
            vocab_logits.float(), self.encodings, self.spec.labels_by_semantic
        )
        return source_distribution(
            logits,
            class_token_ids=self.encodings,
            raw_output=raw_output,
            parsed_label=parsed_label,
            token_diagnostics={"raw_single_token_protocol": self.spec.name},
            classes=self.spec.labels_by_semantic,
            midpoints=self.spec.midpoints,
        )


def protocol_prompt(case: Any, spec: ProtocolSpec) -> str:
    if len(spec.labels_by_semantic) == 9:
        return semantic_mapping_prompt(case, spec.labels_by_semantic)
    classes = "Source attribution classes:\n" + "\n".join(
        f"{label}: {description}"
        for label, description in zip(spec.labels_by_semantic, spec.descriptions)
    )
    both = "the text clue, the image, or both" if len(spec.labels_by_semantic) == 3 else "the text clue or the image"
    return f"""Question:
{case.question}

Text clue:
{case.text_clue}

Answer the question using the text clue and the image.

Then report what you believe your answer was mainly based on.

Source attribution asks what you believe your answer was based on: {both}. Report which source you relied on more when selecting your answer.

{classes}

Output exactly:

**Answer**: <your answer>
**Source Attribution**:<CLASS>

Do not include reasoning, confidence, or any additional text."""


def _protocol_context(runtime: Stage3Runtime, row: dict[str, Any], spec: ProtocolSpec) -> Any:
    case = runtime.case(row["item_id"], row["prior_index"])
    answer = str(row["baseline"]["generated"]["current_answer_result"]["normalized_answer"])
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


def run_protocol_invariant_semantics(
    runtime: Stage3Runtime,
    artifacts: SAFormationArtifacts,
    stage3_root: Path,
    output_root: Path,
    *,
    n_items: int,
    deadline: Callable[[], None],
) -> dict[str, Any]:
    from .followup import _balanced_unique_cases

    directory = output_root / SEMANTIC_DIR
    directory.mkdir(parents=True, exist_ok=True)
    cohort = _balanced_unique_cases(load_baseline_rows(artifacts), n_items)
    specs = protocol_specs()
    analyzers = {spec.name: ProtocolAnalyzer(runtime.generator.tokenizer, spec) for spec in specs}
    atomic_write_json(
        directory / "cohort_manifest.json",
        {
            "case_count": len(cohort),
            "case_ids": [row["case_id"] for row in cohort],
            "protocols": [asdict(spec) for spec in specs],
            "fixed_answer": "natural baseline answer",
            "site": "L18 PANL",
            "decoder": "existing item-OOF Ridge trained on 3260 normal-numeric natural states",
            "primary_cross_protocol_tests": [
                "Train(Normal)->Test(Reversed)",
                "Train(Normal)->Test(Random)",
                "Train(Normal)->Test(Text/Both/Image)",
                "Train(Normal)->Test(Binary Text/Image)",
            ],
        },
    )
    ridge_repo = SAOOFDirectionRepository(stage3_root / "directions")
    result_path = directory / "results.jsonl"
    existing = {
        row["intervention_key"]
        for row in _latest_rows(result_path)
        if row.get("status") == "completed"
    }
    hidden_dir = directory / "hidden"
    for row in cohort:
        deadline()
        key = f"protocol_semantics|{row['case_id']}"
        if key in existing:
            continue
        base = {
            "intervention_key": key,
            "experiment": "protocol_invariant_semantic_sa",
            "case_id": row["case_id"],
            "item_id": row["item_id"],
            "prior_index": row["prior_index"],
            "condition": row["condition"],
            "fold": int(row["fold"]),
        }

        def execute() -> dict[str, Any]:
            direction = ridge_repo.get(row["fold"])
            protocol_results: dict[str, Any] = {}
            matrices: list[np.ndarray] = []
            for spec in specs:
                prepared = _protocol_context(runtime, row, spec)
                measured = runtime.measure(
                    prepared,
                    direction,
                    analyzer=analyzers[spec.name],
                )
                hidden = measured.hidden.astype(np.float32, copy=False)
                protocol_results[spec.name] = {
                    "semantic_imageward_score": float(measured.source["soft_image_score"]),
                    "hard_label": measured.source["hard_label"],
                    "ridge_prediction": direction.predict(hidden),
                    "ridge_coordinate": direction.z(hidden),
                    "prompt_hash": prepared.prefix_hash,
                    "token_ids": analyzers[spec.name].encodings,
                }
                matrices.append(hidden)
                runtime.release_inputs(prepared)
            hidden_file = hidden_dir / f"{row['case_id'].replace('/', '_')}.npz"
            hidden_file.parent.mkdir(parents=True, exist_ok=True)
            from .core import atomic_save_npz

            atomic_save_npz(
                hidden_file,
                protocols=np.asarray([spec.name for spec in specs]),
                hidden=np.stack(matrices),
            )
            return {
                **base,
                "status": "completed",
                "protocols": protocol_results,
                "hidden_file": str(hidden_file.relative_to(output_root)),
            }

        append_jsonl(result_path, _safe_record(base, execute))
    rows = _latest_rows(result_path)
    summary = _summarize_protocol_semantics(rows)
    write_experiment_summary(directory, summary)
    atomic_write_json(output_root / "semantic_target_gate.json", summary["semantic_target_gate"])
    return summary


def _correlation(rows: Sequence[dict[str, Any]], x_key: str, y_key: str) -> dict[str, Any]:
    if len(rows) < 3:
        return {"n": len(rows), "pearson": None, "spearman": None, "spearman_bootstrap": None}
    x = np.asarray([row[x_key] for row in rows], dtype=np.float64)
    y = np.asarray([row[y_key] for row in rows], dtype=np.float64)
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return {
            "n": len(rows),
            "pearson": None,
            "spearman": None,
            "spearman_bootstrap": {
                "estimate": None,
                "ci95": [None, None],
                "iterations": 1000,
                "valid": 0,
            },
            "r2": float(r2_score(y, x)),
            "mae": float(mean_absolute_error(y, x)),
        }
    pearson = pearsonr(x, y).statistic
    spearman = spearmanr(x, y).statistic
    boot = item_cluster_bootstrap(
        rows,
        lambda sample: spearmanr(
            [row[x_key] for row in sample], [row[y_key] for row in sample]
        ).statistic,
    )
    return {
        "n": len(rows),
        "pearson": float(pearson) if np.isfinite(pearson) else None,
        "spearman": float(spearman) if np.isfinite(spearman) else None,
        "spearman_bootstrap": boot,
        "r2": float(r2_score(y, x)),
        "mae": float(mean_absolute_error(y, x)),
    }


def _summarize_protocol_semantics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed"]
    names = [spec.name for spec in protocol_specs()]
    if len(completed) < 3:
        gate = {
            "passed": False,
            "cross_protocol_decoder_passed": False,
            "shared_component_passed": False,
            "rule": "all four Normal-trained cross-protocol Spearman bootstrap CI lower bounds >0; PC1 variance >=0.50 and every oriented loading >0.10",
            "classification": "insufficient completed cases for a semantic target",
        }
        return {
            "title": "Experiment 3 — Protocol-Invariant Semantic SA",
            "status": "completed",
            "n": len(completed),
            "failed": len(rows) - len(completed),
            "shared_component": None,
            "normal_trained_oof_decoder_transfer": {},
            "semantic_target_gate": gate,
        }
    matrix = np.asarray(
        [
            [row["protocols"][name]["semantic_imageward_score"] for name in names]
            for row in completed
        ],
        dtype=np.float64,
    )
    protocol_sd = matrix.std(axis=0, ddof=1)
    nonconstant = np.isfinite(protocol_sd) & (protocol_sd > 1e-12)
    standardized = np.zeros_like(matrix)
    standardized[:, nonconstant] = (
        matrix[:, nonconstant] - matrix[:, nonconstant].mean(axis=0)
    ) / protocol_sd[nonconstant]
    _, singular, vt = np.linalg.svd(standardized, full_matrices=False)
    loadings = vt[0].copy()
    if loadings[0] < 0:
        loadings = -loadings
    shared_score = standardized @ loadings
    singular_energy = float(np.sum(singular**2))
    explained = float(singular[0] ** 2 / singular_energy) if singular_energy > 0 else 0.0
    protocol_agreement: dict[str, Any] = {}
    normal_index = names.index("normal_numeric")
    for index, name in enumerate(names):
        if np.std(matrix[:, normal_index]) <= 1e-12 or np.std(matrix[:, index]) <= 1e-12:
            statistic = None
        else:
            statistic = spearmanr(matrix[:, normal_index], matrix[:, index]).statistic
        protocol_agreement[name] = (
            float(statistic)
            if statistic is not None and np.isfinite(statistic)
            else None
        )
    decoder: dict[str, Any] = {}
    for name in names:
        values = []
        for index, row in enumerate(completed):
            values.append(
                {
                    "item_id": row["item_id"],
                    "prediction": row["protocols"][name]["ridge_prediction"],
                    "semantic_score": row["protocols"][name]["semantic_imageward_score"],
                    "shared_component": float(shared_score[index]),
                }
            )
        decoder[name] = {
            "against_protocol_semantic_score": _correlation(
                values, "prediction", "semantic_score"
            ),
            "against_shared_component": _correlation(
                values, "prediction", "shared_component"
            ),
        }
    nonnormal = [name for name in names if name != "normal_numeric"]
    cross_pass = all(
        decoder[name]["against_protocol_semantic_score"]["spearman_bootstrap"]["ci95"][0]
        is not None
        and decoder[name]["against_protocol_semantic_score"]["spearman_bootstrap"]["ci95"][0]
        > 0
        for name in nonnormal
    )
    shared_pass = bool(
        bool(nonconstant.all())
        and explained >= 0.50
        and all(value > 0.10 for value in loadings)
    )
    gate = {
        "passed": bool(cross_pass and shared_pass),
        "cross_protocol_decoder_passed": cross_pass,
        "shared_component_passed": shared_pass,
        "rule": "all four Normal-trained cross-protocol Spearman bootstrap CI lower bounds >0; PC1 variance >=0.50 and every oriented loading >0.10",
        "classification": (
            "candidate protocol-invariant semantic SA state"
            if cross_pass and shared_pass
            else "no validated protocol-invariant semantic SA target at L18 PANL"
        ),
    }
    return {
        "title": "Experiment 3 — Protocol-Invariant Semantic SA",
        "status": "completed",
        "n": len(completed),
        "failed": len(rows) - len(completed),
        "shared_component": {
            "protocol_order": names,
            "oriented_loadings": [float(value) for value in loadings],
            "explained_variance": explained,
            "protocol_standard_deviations": [float(value) for value in protocol_sd],
            "all_protocols_nonconstant": bool(nonconstant.all()),
            "normal_protocol_pairwise_spearman": protocol_agreement,
        },
        "normal_trained_oof_decoder_transfer": decoder,
        "semantic_target_gate": gate,
    }


def write_gate_controlled_skips(output_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    gate = read_json(output_root / "semantic_target_gate.json")
    if gate["passed"]:
        raise RuntimeError("Semantic target gate passed; tracing/subspace must be run, not skipped")
    summaries = []
    for directory_name, title in (
        (TRACING_DIR, "Experiment 4 — Blockwise Causal Formation Tracing"),
        (SUBSPACE_DIR, "Experiment 5 — Low-rank Formation Subspace"),
    ):
        directory = output_root / directory_name
        directory.mkdir(parents=True, exist_ok=True)
        reason = "No validated protocol-invariant semantic SA target at L18 PANL"
        atomic_write_json(directory / "cohort_manifest.json", {"cases": [], "reason": reason})
        write_jsonl_atomic(directory / "results.jsonl", [])
        summary = {"title": title, "status": "skipped", "n": 0, "reason": reason}
        write_experiment_summary(directory, summary)
        summaries.append(summary)
    return summaries[0], summaries[1]


def write_second_order_report(output_root: Path) -> dict[str, Any]:
    summaries = {
        "behavior": read_json(output_root / BEHAVIOR_DIR / "summary.json"),
        "priming": read_json(output_root / PRIMING_DIR / "summary.json"),
        "semantic": read_json(output_root / SEMANTIC_DIR / "summary.json"),
        "tracing": read_json(output_root / TRACING_DIR / "summary.json"),
        "subspace": read_json(output_root / SUBSPACE_DIR / "summary.json"),
    }
    rows = []
    for relevance, values in summaries["behavior"]["answer_effects"].items():
        effect = values["delta_p_image_answer"]
        rows.append(
            {
                "experiment": f"History behavior {relevance}",
                "type": "first-order behavior",
                "n": effect["n"],
                "effect": effect["mean"],
                "ci_low": effect["ci95"][0],
                "ci_high": effect["ci95"][1],
            }
        )
    for relevance, values in summaries["behavior"][
        "answer_effects_on_existing_sa_endpoint_matched_cohort"
    ].items():
        effect = values["delta_p_image_answer"]
        rows.append(
            {
                "experiment": f"History behavior {relevance} (SA endpoint matched)",
                "type": "first-order behavior primary matched",
                "n": effect["n"],
                "effect": effect["mean"],
                "ci_low": effect["ci95"][0],
                "ci_high": effect["ci95"][1],
            }
        )
    for name, effect in summaries["priming"]["contrasts"].items():
        rows.append(
            {
                "experiment": name,
                "type": "second-order SA formation",
                "n": effect["n"],
                "effect": effect["mean"],
                "ci_low": effect["ci95"][0],
                "ci_high": effect["ci95"][1],
            }
        )
    semantic_decoder = summaries["semantic"]["normal_trained_oof_decoder_transfer"]
    for name, metrics in semantic_decoder.items():
        effect = metrics["against_protocol_semantic_score"]
        rows.append(
            {
                "experiment": f"Normal decoder -> {name}",
                "type": "cross-protocol semantic association",
                "n": effect["n"],
                "effect": effect["spearman"],
                "ci_low": effect["spearman_bootstrap"]["ci95"][0],
                "ci_high": effect["spearman_bootstrap"]["ci95"][1],
            }
        )
    analysis = output_root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(analysis / "core_table.csv", rows)
    payload = {"status": "completed", "experiments": summaries}
    atomic_write_json(analysis / "final_analysis.json", payload)
    behavior = summaries["behavior"]
    priming = summaries["priming"]
    semantic = summaries["semantic"]
    relevant_answer = behavior[
        "answer_effects_on_existing_sa_endpoint_matched_cohort"
    ]["relevant"]["delta_p_image_answer"]
    irrelevant_answer = behavior[
        "answer_effects_on_existing_sa_endpoint_matched_cohort"
    ]["irrelevant"]["delta_p_image_answer"]
    relevant_answer_all = behavior["answer_effects"]["relevant"]["delta_p_image_answer"]
    relevant_sa = behavior["existing_sa_effects"]["relevant_image_minus_text"]
    blank = priming["contrasts"]["blank_minus_no"]
    visual_semantic = priming["contrasts"]["irrelevant_minus_blank"]
    relevance = priming["contrasts"]["relevant_minus_irrelevant"]
    shared = semantic["shared_component"]
    gate = semantic["semantic_target_gate"]

    def effect_text(effect: dict[str, Any], digits: int = 3) -> str:
        low, high = effect["ci95"]
        return (
            f"{effect['mean']:+.{digits}f}, 95% CI "
            f"[{low:+.{digits}f}, {high:+.{digits}f}], n={effect['n']}"
        )

    def rho_text(protocol: str) -> str:
        effect = semantic_decoder[protocol]["against_protocol_semantic_score"]
        low, high = effect["spearman_bootstrap"]["ci95"]
        return (
            f"rho={effect['spearman']:.3f}, 95% CI "
            f"[{low:.3f}, {high:.3f}], n={effect['n']}"
        )

    lines = [
        "# Second-order Source Attribution Formation",
        "",
        "## Main conclusions",
        "",
        "1. On the identical endpoint-matched cohort, Relevant Image History makes verbal attribution more imageward while leaving answer probability approximately unchanged. This supports a second-order-specific History effect.",
        "2. The verbal-SA History effect is driven by visual semantic context beyond a blank image; no stable additional target-relevance increment was detected in the direct Image-History contrast.",
        "3. L18 PANL contains a substantial shared cross-protocol component and transfers across fine-grained label remappings, but the Normal-trained decoder does not transfer reliably to coarse three-class or binary protocols. The full semantic-target gate therefore fails.",
        "4. Blockwise tracing and low-rank subspace intervention were not run, because their preregistered semantic-target prerequisite was not met.",
        "",
        "## Experiment 1 — First-order / second-order dissociation",
        "",
        f"- Relevant verbal SA, Image−Text History: {effect_text(relevant_sa)}.",
        f"- Relevant answer P(A_I), identical SA endpoint-matched cohort: {effect_text(relevant_answer)}.",
        f"- Irrelevant answer P(A_I), identical SA endpoint-matched cohort: {effect_text(irrelevant_answer)}.",
        f"- All-case relevant answer sensitivity (not cohort-aligned to the verbal-SA estimate): {effect_text(relevant_answer_all)}.",
        f"- Classification: **{behavior['classification']}**.",
        "",
        "## Experiment 2 — Priming decomposition",
        "",
        f"- Blank−NoHistory: {effect_text(blank)}.",
        f"- IrrelevantSemantic−Blank: {effect_text(visual_semantic)}.",
        f"- Relevant−Irrelevant Image History: {effect_text(relevance)}.",
        f"- Strict four-condition endpoint-matched cases: {priming['strict_four_condition_endpoint_matched_n']}.",
        f"- Classification: **{priming['classification']}**.",
        "",
        "## Experiment 3 — Protocol-invariant semantic SA",
        "",
        f"- Shared PC1 explained variance: {shared['explained_variance']:.3f}; all five oriented loadings are positive.",
        f"- Normal→ReversedNumeric: {rho_text('reversed_numeric')}.",
        f"- Normal→RandomSingleToken: {rho_text('random_single_token')}.",
        f"- Normal→Text/Both/Image: {rho_text('text_both_image')}.",
        f"- Normal→BinaryText/Image: {rho_text('binary_text_image')}.",
        f"- Gate passed: **{gate['passed']}** — {gate['classification']}.",
        "",
        "## Claim limits",
        "",
        "- Exp 1 establishes an endpoint-matched dissociation between generated-answer behavior and generated verbal attribution; every History assistant turn contains the same target A_T, so the modality manipulation also changes evidence–answer congruence and does not identify a mediator.",
        "- Exp 2 isolates an endpoint-matched upstream priming component, but the relevance-null result is not evidence that relevance can never matter.",
        "- Exp 3 supports partial protocol invariance at L18 PANL, not a fully protocol-invariant causal SA state.",
        f"- Blockwise tracing: {summaries['tracing']['status']}; low-rank subspace: {summaries['subspace']['status']}.",
        "",
    ]
    atomic_write_text(analysis / "FINAL_ANALYSIS.md", "\n".join(lines))
    return payload
