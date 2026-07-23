"""CLI and orchestration for four confidence-evaluation variants."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from confidence_test.confidence_extension import MultimodalConfidenceAnalyzer
from confidence_test.dataset_utils import CONDITIONS, EvaluationCase, load_evaluation_cases
from confidence_test.inference_extension import build_extended_inference_class
from confidence_test.io_utils import (
    empty_condition,
    find_prior,
    load_json,
    new_prior_record,
    upsert_prior,
    write_result_pair,
)
from confidence_test.prompt_utils import (
    EVALUATION_VARIANTS,
    STAGE1_TEXT_ANSWER_PROMPT,
    STAGE2_TEXT_CONFIDENCE_PROMPT,
    V1_STAGE3_IMAGE_CONFIDENCE_PROMPT,
    V2_STAGE3_IMAGE_CONFIDENCE_PROMPT,
    V3_STAGE3_REANSWER_PROMPT,
    V3_STAGE4_META_CONFIDENCE_PROMPT,
    V4_STAGE1_FULL_EVIDENCE_ANSWER_PROMPT,
    V4_STAGE2_FULL_EVIDENCE_CONFIDENCE_PROMPT,
)
from confidence_test.runtime_imports import DEFAULT_INFERENCE_PATH, load_runtime


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = REPOSITORY_ROOT / "qwen-2.5-vl" / "models" / "Qwen2.5-VL-7B-Instruct"
DEFAULT_DATASET_PATH = REPOSITORY_ROOT / "datasets" / "dataset_test.json"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "output"
DEFAULT_LOG_PATH = PACKAGE_ROOT / "logs" / "evaluation.log"
FALLBACK_NULL_PATH = PACKAGE_ROOT / "assets" / "null.png"

VARIANT_ALIASES = {
    "v1": "v1",
    "v1_visible_previous_confidence": "v1",
    "v2": "v2",
    "v2_hidden_previous_confidence": "v2",
    "v3": "v3",
    "v3_reanswer_then_confidence": "v3",
    "v4": "v4",
    "v4_full_evidence_baseline": "v4",
}
VARIANT_ORDER = ("v1", "v2", "v3", "v4")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _error(error_type: str, message: str) -> dict[str, str]:
    return {"type": error_type, "message": message}


def _to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return deepcopy(value)
    if hasattr(value, "to_dict"):
        return deepcopy(value.to_dict())
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Unsupported result type: {type(value)!r}")


def _answer_generation_ready(value: dict[str, Any] | None) -> bool:
    return bool(value and value.get("parse_success") and value.get("answer"))


def _answer_metrics_ready(value: dict[str, Any] | None) -> bool:
    return bool(
        _answer_generation_ready(value)
        and value.get("answer_metric_status") == "completed"
        and value.get("answer_prob") is not None
        and value.get("answer_entropy") is not None
    )


def _confidence_ready(value: dict[str, Any] | None) -> bool:
    return bool(value and value.get("status") == "completed" and value.get("soft_confidence") is not None)


def _shared_stage_ready(stage: dict[str, Any] | None) -> bool:
    return bool(
        stage
        and stage.get("status") == "completed"
        and _answer_metrics_ready(stage.get("answer_result"))
        and _confidence_ready(stage.get("confidence_result"))
    )


def _stage_fingerprint(stage: dict[str, Any]) -> str:
    return json.dumps(stage, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def validate_shared_checkpoint_consistency(
    results_by_version: dict[str, list[dict[str, Any]]],
) -> None:
    by_record: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for version in ("v1", "v2", "v3"):
        for sample in results_by_version.get(version, []):
            for prior in sample.get("priors", []):
                stage = prior.get("text_stage")
                if _shared_stage_ready(stage):
                    by_record.setdefault(str(prior.get("record_key")), []).append((version, stage))
    for record_key, candidates in by_record.items():
        fingerprints = {_stage_fingerprint(stage) for _, stage in candidates}
        if len(fingerprints) > 1:
            versions = ", ".join(version for version, _ in candidates)
            raise ValueError(
                f"Conflicting completed shared text stages for {record_key}: {versions}"
            )


def _existing_shared_stage(
    results_by_version: dict[str, list[dict[str, Any]]],
    record_key: str,
) -> dict[str, Any] | None:
    partial: dict[str, Any] | None = None
    for version in ("v1", "v2", "v3"):
        prior = find_prior(results_by_version.get(version, []), record_key)
        if prior is None or not isinstance(prior.get("text_stage"), dict):
            continue
        stage = prior["text_stage"]
        if _shared_stage_ready(stage):
            return deepcopy(stage)
        if partial is None or (
            _answer_generation_ready(stage.get("answer_result"))
            and not _answer_generation_ready(partial.get("answer_result"))
        ):
            partial = deepcopy(stage)
    return partial


def _canonical_selection(
    raw_values: Iterable[str],
    aliases: dict[str, str],
    order: tuple[str, ...],
    label: str,
) -> list[str]:
    parts = [part.strip() for value in raw_values for part in value.split(",") if part.strip()]
    if "all" in parts:
        if len(parts) != 1:
            raise ValueError(f"{label}: 'all' cannot be combined with explicit values")
        return list(order)
    invalid = [part for part in parts if part not in aliases]
    if invalid:
        raise ValueError(f"Unknown {label} value(s): {', '.join(invalid)}")
    selected = {aliases[part] for part in parts}
    if not selected:
        raise ValueError(f"At least one {label} value is required")
    return [value for value in order if value in selected]


def canonical_variants(raw_values: Iterable[str]) -> list[str]:
    return _canonical_selection(raw_values, VARIANT_ALIASES, VARIANT_ORDER, "variant")


def canonical_conditions(raw_values: Iterable[str]) -> list[str]:
    aliases = {condition: condition for condition in CONDITIONS}
    return _canonical_selection(raw_values, aliases, CONDITIONS, "condition")


class EvaluationRunner:
    """Serial evaluator with one shared model/processor and case-level commits."""

    def __init__(
        self,
        inference: Any,
        confidence_analyzer: Any,
        confidence_class_text: str,
        cases: list[EvaluationCase],
        variants: list[str],
        conditions: list[str],
        results_by_version: dict[str, list[dict[str, Any]]],
        output_dir: Path,
        run_config: dict[str, Any],
        max_answer_tokens: int = 24,
        logger: logging.Logger | None = None,
        checkpoint_writer: Any = write_result_pair,
    ):
        self.inference = inference
        self.confidence_analyzer = confidence_analyzer
        self.confidence_class_text = confidence_class_text
        self.cases = cases
        self.variants = variants
        self.conditions = conditions
        self.results_by_version = results_by_version
        self.output_dir = output_dir
        self.run_config = run_config
        self.max_answer_tokens = max_answer_tokens
        self.logger = logger or logging.getLogger("confidence_test.null")
        self.checkpoint_writer = checkpoint_writer
        self.call_counts: dict[str, int] = {
            "shared_stage1": 0,
            "shared_stage2": 0,
            "v1_stage3": 0,
            "v2_stage3": 0,
            "v3_stage3": 0,
            "v3_stage4": 0,
            "v4_stage1": 0,
            "v4_stage2": 0,
            "total": 0,
        }
        self.item_order = {case.item_id: case.item_order for case in cases}

    def _log(self, message: str, **fields: Any) -> None:
        rendered = " ".join(f"{key}={value}" for key, value in fields.items())
        self.logger.info("%s%s", message, f" {rendered}" if rendered else "")

    def _count_call(self, stage: str, case: EvaluationCase, version: str, condition: str | None) -> None:
        self.call_counts[stage] += 1
        self.call_counts["total"] += 1
        self._log(
            "model_call",
            item_id=case.item_id,
            prior_index=case.prior_index,
            version=version,
            condition=condition,
            stage=stage,
            cumulative_calls=self.call_counts["total"],
        )

    def _run_answer(
        self,
        stage: str,
        case: EvaluationCase,
        version: str,
        condition: str | None,
        prompt: str,
        image_path: str | None,
    ) -> dict[str, Any]:
        self._count_call(stage, case, version, condition)
        result = _to_dict(
            self.inference.generate_answer_with_metrics(
                prompt=prompt,
                answer_classes=case.answer_classes,
                image_path=image_path,
                max_new_tokens=self.max_answer_tokens,
            )
        )
        result["generation_status"] = "completed" if _answer_generation_ready(result) else "failed"
        result["status"] = "completed" if _answer_metrics_ready(result) else "failed"
        if case.answer_class_error and not result.get("error"):
            result["error"] = deepcopy(case.answer_class_error)
        self._log(
            "stage_completed",
            item_id=case.item_id,
            prior_index=case.prior_index,
            version=version,
            condition=condition,
            stage=stage,
            status=result["status"],
            elapsed_seconds=result.get("elapsed_seconds"),
            error_type=(result.get("error") or {}).get("type"),
            cumulative_calls=self.call_counts["total"],
        )
        return result

    def _run_confidence(
        self,
        stage: str,
        case: EvaluationCase,
        version: str,
        condition: str | None,
        prompt: str,
        image_path: str | None,
    ) -> dict[str, Any]:
        self._count_call(stage, case, version, condition)
        started = time.perf_counter()
        try:
            result = _to_dict(self.confidence_analyzer.analyze_prompt(prompt, image_path))
            result.pop("rendered_prompt", None)
            result.pop("class_token_variants", None)
            result.pop("hidden_state_collected", None)
            result["status"] = "completed"
            result["error"] = None
        except Exception as exc:
            result = {
                "confidence_label": None,
                "hard_confidence_midpoint": None,
                "soft_confidence": None,
                "class_logits": {},
                "class_probabilities": {},
                "raw_output": "",
                "hard_label_parsed": False,
                "status": "failed",
                "error": _error(type(exc).__name__, str(exc)),
            }
        result["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        self._log(
            "stage_completed",
            item_id=case.item_id,
            prior_index=case.prior_index,
            version=version,
            condition=condition,
            stage=stage,
            status=result["status"],
            elapsed_seconds=result["elapsed_seconds"],
            error_type=(result.get("error") or {}).get("type"),
            cumulative_calls=self.call_counts["total"],
        )
        return result

    def _shared_stage(self, case: EvaluationCase) -> dict[str, Any]:
        stage = _existing_shared_stage(self.results_by_version, case.record_key) or {
            "status": "pending",
            "answer_result": None,
            "confidence_result": None,
            "elapsed_seconds": 0.0,
            "error": None,
        }
        answer_result = stage.get("answer_result")
        if not _answer_metrics_ready(answer_result):
            answer_prompt = STAGE1_TEXT_ANSWER_PROMPT.format(
                question=case.question,
                text_clue=case.text_clue,
            )
            answer_result = self._run_answer(
                "shared_stage1", case, "shared", None, answer_prompt, None
            )
            stage["answer_result"] = answer_result
            stage["confidence_result"] = None
        if _answer_generation_ready(answer_result):
            confidence_result = stage.get("confidence_result")
            if not _confidence_ready(confidence_result):
                confidence_prompt = STAGE2_TEXT_CONFIDENCE_PROMPT.format(
                    question=case.question,
                    text_clue=case.text_clue,
                    answer=answer_result["answer"],
                    classes=self.confidence_class_text,
                )
                confidence_result = self._run_confidence(
                    "shared_stage2", case, "shared", None, confidence_prompt, None
                )
                stage["confidence_result"] = confidence_result
        else:
            stage["confidence_result"] = None
        confidence_result = stage.get("confidence_result")
        stage["elapsed_seconds"] = round(
            float((answer_result or {}).get("elapsed_seconds") or 0.0)
            + float((confidence_result or {}).get("elapsed_seconds") or 0.0),
            6,
        )
        if _answer_metrics_ready(answer_result) and _confidence_ready(confidence_result):
            stage["status"] = "completed"
            stage["error"] = None
        elif not _answer_generation_ready(answer_result):
            stage["status"] = "failed"
            stage["error"] = deepcopy((answer_result or {}).get("error"))
        elif not _confidence_ready(confidence_result):
            stage["status"] = "failed"
            stage["error"] = deepcopy((confidence_result or {}).get("error"))
        else:
            stage["status"] = "failed"
            stage["error"] = deepcopy((answer_result or {}).get("error"))
        return stage

    @staticmethod
    def _condition_base(case: EvaluationCase, condition: str) -> dict[str, Any]:
        value = empty_condition("pending")
        source = case.conditions[condition]
        value["relative_image_path"] = source.relative_image_path
        value["resolved_image_path"] = source.resolved_image_path
        value["error"] = deepcopy(source.error)
        if source.error:
            value["status"] = "failed"
        return value

    @staticmethod
    def _finish_condition(value: dict[str, Any]) -> None:
        answer = value.get("answer_result") or {}
        confidence = value.get("confidence_result") or {}
        value["elapsed_seconds"] = round(
            float(answer.get("elapsed_seconds") or 0.0)
            + float(confidence.get("elapsed_seconds") or 0.0),
            6,
        )
        if _answer_metrics_ready(answer) and _confidence_ready(confidence):
            value["status"] = "completed"
            value["error"] = None
        else:
            value["status"] = "failed"
            value["error"] = deepcopy(answer.get("error") or confidence.get("error"))

    def _run_v1_or_v2(
        self,
        version: str,
        case: EvaluationCase,
        record: dict[str, Any],
        shared: dict[str, Any],
        condition: str,
    ) -> None:
        existing = record["conditions"].get(condition) or {}
        if existing.get("status") == "completed":
            self._log("checkpoint_skip", record_key=case.record_key, version=version, condition=condition)
            return
        value = self._condition_base(case, condition)
        answer = deepcopy(shared.get("answer_result"))
        value["answer_result"] = answer
        if value["error"] or not _answer_generation_ready(answer):
            value["status"] = "skipped" if not value["error"] else "failed"
            value["error"] = value["error"] or _error("DependencyError", "Shared Stage 1 failed")
            record["conditions"][condition] = value
            self._log(
                "condition_not_run",
                record_key=case.record_key,
                version=version,
                condition=condition,
                status=value["status"],
                error_type=(value["error"] or {}).get("type"),
            )
            return
        shared_confidence = shared.get("confidence_result") or {}
        if not _confidence_ready(shared_confidence):
            value["status"] = "skipped"
            value["error"] = _error("DependencyError", "Shared Stage 2 failed")
            record["conditions"][condition] = value
            self._log(
                "condition_not_run",
                record_key=case.record_key,
                version=version,
                condition=condition,
                status=value["status"],
                error_type="DependencyError",
            )
            return
        previous_confidence = shared_confidence.get("confidence_label")
        if version == "v1":
            prompt = V1_STAGE3_IMAGE_CONFIDENCE_PROMPT.format(
                question=case.question,
                text_clue=case.text_clue,
                answer=answer["answer"],
                previous_confidence=previous_confidence,
                classes=self.confidence_class_text,
            )
            stage_name = "v1_stage3"
        else:
            prompt = V2_STAGE3_IMAGE_CONFIDENCE_PROMPT.format(
                question=case.question,
                text_clue=case.text_clue,
                answer=answer["answer"],
                classes=self.confidence_class_text,
            )
            stage_name = "v2_stage3"
        existing_confidence = existing.get("confidence_result")
        confidence = existing_confidence if _confidence_ready(existing_confidence) else self._run_confidence(
            stage_name,
            case,
            version,
            condition,
            prompt,
            value["resolved_image_path"],
        )
        value["confidence_result"] = deepcopy(confidence)
        if version == "v1" and _confidence_ready(confidence):
            value["delta_soft_confidence"] = float(confidence["soft_confidence"]) - float(
                shared_confidence["soft_confidence"]
            )
        self._finish_condition(value)
        record["conditions"][condition] = value

    def _run_v3(
        self,
        case: EvaluationCase,
        record: dict[str, Any],
        shared: dict[str, Any],
        condition: str,
    ) -> None:
        existing = record["conditions"].get(condition) or {}
        if existing.get("status") == "completed":
            self._log("checkpoint_skip", record_key=case.record_key, version="v3", condition=condition)
            return
        value = self._condition_base(case, condition)
        initial_answer = shared.get("answer_result") or {}
        initial_confidence = shared.get("confidence_result") or {}
        if value["error"]:
            record["conditions"][condition] = value
            self._log(
                "condition_not_run",
                record_key=case.record_key,
                version="v3",
                condition=condition,
                status="failed",
                error_type=(value["error"] or {}).get("type"),
            )
            return
        if not _answer_generation_ready(initial_answer) or not _confidence_ready(initial_confidence):
            value["status"] = "skipped"
            value["error"] = _error("DependencyError", "Shared text stages failed")
            record["conditions"][condition] = value
            self._log(
                "condition_not_run",
                record_key=case.record_key,
                version="v3",
                condition=condition,
                status="skipped",
                error_type="DependencyError",
            )
            return
        answer_result = existing.get("answer_result")
        if not _answer_metrics_ready(answer_result):
            prompt = V3_STAGE3_REANSWER_PROMPT.format(
                question=case.question,
                text_clue=case.text_clue,
                previous_answer=initial_answer["answer"],
                previous_confidence=initial_confidence["confidence_label"],
            )
            answer_result = self._run_answer(
                "v3_stage3",
                case,
                "v3",
                condition,
                prompt,
                value["resolved_image_path"],
            )
        value["answer_result"] = deepcopy(answer_result)
        if not _answer_generation_ready(answer_result):
            value["status"] = "failed"
            value["error"] = deepcopy(answer_result.get("error"))
            record["conditions"][condition] = value
            return
        confidence_result = existing.get("confidence_result")
        if not _confidence_ready(confidence_result):
            prompt = V3_STAGE4_META_CONFIDENCE_PROMPT.format(
                question=case.question,
                text_clue=case.text_clue,
                initial_answer=initial_answer["answer"],
                initial_confidence=initial_confidence["confidence_label"],
                stage3_answer=answer_result["answer"],
                classes=self.confidence_class_text,
            )
            confidence_result = self._run_confidence(
                "v3_stage4",
                case,
                "v3",
                condition,
                prompt,
                value["resolved_image_path"],
            )
        value["confidence_result"] = deepcopy(confidence_result)
        self._finish_condition(value)
        record["conditions"][condition] = value

    def _run_v4(
        self,
        case: EvaluationCase,
        record: dict[str, Any],
        condition: str,
    ) -> None:
        existing = record["conditions"].get(condition) or {}
        if existing.get("status") == "completed":
            self._log("checkpoint_skip", record_key=case.record_key, version="v4", condition=condition)
            return
        value = self._condition_base(case, condition)
        if value["error"]:
            record["conditions"][condition] = value
            self._log(
                "condition_not_run",
                record_key=case.record_key,
                version="v4",
                condition=condition,
                status="failed",
                error_type=(value["error"] or {}).get("type"),
            )
            return
        answer_result = existing.get("answer_result")
        if not _answer_metrics_ready(answer_result):
            prompt = V4_STAGE1_FULL_EVIDENCE_ANSWER_PROMPT.format(
                question=case.question,
                text_clue=case.text_clue,
            )
            answer_result = self._run_answer(
                "v4_stage1",
                case,
                "v4",
                condition,
                prompt,
                value["resolved_image_path"],
            )
        value["answer_result"] = deepcopy(answer_result)
        if not _answer_generation_ready(answer_result):
            value["status"] = "failed"
            value["error"] = deepcopy(answer_result.get("error"))
            record["conditions"][condition] = value
            return
        confidence_result = existing.get("confidence_result")
        if not _confidence_ready(confidence_result):
            prompt = V4_STAGE2_FULL_EVIDENCE_CONFIDENCE_PROMPT.format(
                question=case.question,
                text_clue=case.text_clue,
                answer=answer_result["answer"],
                classes=self.confidence_class_text,
            )
            confidence_result = self._run_confidence(
                "v4_stage2",
                case,
                "v4",
                condition,
                prompt,
                value["resolved_image_path"],
            )
        value["confidence_result"] = deepcopy(confidence_result)
        self._finish_condition(value)
        record["conditions"][condition] = value

    def process_case(self, case: EvaluationCase) -> None:
        self._log("case_started", item_id=case.item_id, prior_index=case.prior_index, record_key=case.record_key)
        records: dict[str, dict[str, Any]] = {}
        for version in self.variants:
            existing = find_prior(self.results_by_version[version], case.record_key)
            record = deepcopy(existing) if existing is not None else new_prior_record(case, version)
            record.setdefault("conditions", {})
            for condition in CONDITIONS:
                record["conditions"].setdefault(condition, empty_condition())
            records[version] = record

        shared: dict[str, Any] | None = None
        if any(version in self.variants for version in ("v1", "v2", "v3")):
            shared = self._shared_stage(case)
            for version in ("v1", "v2", "v3"):
                if version in records:
                    records[version]["text_stage"] = deepcopy(shared)
                    records[version]["text_answer"] = (shared.get("answer_result") or {}).get("answer")
                    records[version]["text_conf"] = (shared.get("confidence_result") or {}).get("soft_confidence")

        for version in self.variants:
            record = records[version]
            if version == "v4":
                record["text_stage"] = None
                record["text_answer"] = None
                record["text_conf"] = None
            for condition in self.conditions:
                if version in ("v1", "v2"):
                    assert shared is not None
                    self._run_v1_or_v2(version, case, record, shared, condition)
                elif version == "v3":
                    assert shared is not None
                    self._run_v3(case, record, shared, condition)
                else:
                    self._run_v4(case, record, condition)

        # The only checkpoint boundary: all selected version/condition work for
        # this item×prior is complete in memory before any result file is touched.
        for version in self.variants:
            upsert_prior(
                self.results_by_version[version],
                case,
                records[version],
                self.run_config,
                self.item_order,
            )
        for version in self.variants:
            full_path, simplified_path = self.checkpoint_writer(
                self.output_dir,
                version,
                self.results_by_version[version],
            )
            self._log(
                "case_checkpoint",
                record_key=case.record_key,
                version=version,
                full_path=full_path,
                simplified_path=simplified_path,
                cumulative_calls=self.call_counts["total"],
            )

    def run(self) -> dict[str, int]:
        for case in self.cases:
            self.process_case(case)
        self._log("evaluation_completed", **self.call_counts)
        return dict(self.call_counts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--inference-path", default=str(DEFAULT_INFERENCE_PATH))
    parser.add_argument("--variants", nargs="+", default=["all"])
    parser.add_argument("--conditions", nargs="+", default=["all"])
    parser.add_argument("--max-answer-tokens", type=int, default=24)
    parser.add_argument("--max-confidence-tokens", type=int, default=12)
    parser.add_argument("--item-limit", type=int)
    parser.add_argument("--prior-limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _inside_package(path: Path) -> bool:
    try:
        path.resolve().relative_to(PACKAGE_ROOT.resolve())
        return True
    except ValueError:
        return False


def validate_args(args: argparse.Namespace) -> tuple[list[str], list[str], Path, Path, Path, Path]:
    variants = canonical_variants(args.variants)
    conditions = canonical_conditions(args.conditions)
    model_path = Path(args.model_path).resolve()
    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    inference_path = Path(args.inference_path).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")
    if not inference_path.is_file():
        raise FileNotFoundError(f"Inference source does not exist: {inference_path}")
    if not _inside_package(output_dir):
        raise ValueError(f"--output-dir must be inside {PACKAGE_ROOT}: {output_dir}")
    for name in ("max_answer_tokens", "max_confidence_tokens", "item_limit", "prior_limit"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    return variants, conditions, model_path, dataset_path, output_dir, inference_path


def configure_logger() -> logging.Logger:
    DEFAULT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("confidence_test.evaluation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(DEFAULT_LOG_PATH, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        variants, conditions, model_path, dataset_path, output_dir, inference_path = validate_args(args)
        output_dir.mkdir(parents=True, exist_ok=True)
        all_results: dict[str, list[dict[str, Any]]] = {}
        for version in VARIANT_ORDER:
            path = output_dir / f"{version}_results.json"
            loaded = load_json(path, [])
            if not isinstance(loaded, list):
                raise ValueError(f"Full result root must be an array: {path}")
            all_results[version] = loaded
        if args.overwrite:
            for version in variants:
                all_results[version] = []
                for suffix in ("results.json", "simplified.json"):
                    path = output_dir / f"{version}_{suffix}"
                    if path.exists():
                        path.unlink()
        validate_shared_checkpoint_consistency(all_results)
        cases, dataset_metadata = load_evaluation_cases(
            dataset_path,
            item_limit=args.item_limit,
            prior_limit=args.prior_limit,
            fallback_null_path=FALLBACK_NULL_PATH,
        )
        runtime = load_runtime(inference_path)
    except Exception as exc:
        parser.error(str(exc))

    logger = configure_logger()
    logger.info(
        "run_started variants=%s conditions=%s dataset=%s output_dir=%s",
        variants,
        conditions,
        dataset_path,
        output_dir,
    )
    run_config = {
        "started_at": utc_now(),
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "inference_path": str(inference_path),
        "output_dir": str(output_dir),
        "variants": variants,
        "conditions": conditions,
        "max_answer_tokens": args.max_answer_tokens,
        "max_confidence_tokens": args.max_confidence_tokens,
        "item_limit": args.item_limit,
        "prior_limit": args.prior_limit,
        "checkpoint_policy": "once_per_completed_item_prior_case",
        "null_image": dataset_metadata["null_image"],
        "evaluation_variants": [entry["version"] for entry in EVALUATION_VARIANTS],
    }
    try:
        extended_class = build_extended_inference_class(runtime.QwenVLInference)
        inference = extended_class(model_path=str(model_path))
        logger.info("model_loaded model_instances=1 processor_instances=1")
        base_confidence = runtime.ConfidenceAnalyzer(
            inference,
            max_new_tokens=args.max_confidence_tokens,
        )
        confidence = MultimodalConfidenceAnalyzer(base_confidence, inference)
        runner = EvaluationRunner(
            inference=inference,
            confidence_analyzer=confidence,
            confidence_class_text=runtime.CONFIDENCE_CLASS_TEXT,
            cases=cases,
            variants=variants,
            conditions=conditions,
            results_by_version=all_results,
            output_dir=output_dir,
            run_config=run_config,
            max_answer_tokens=args.max_answer_tokens,
            logger=logger,
        )
        counts = runner.run()
        print(json.dumps({"status": "completed", "call_counts": counts}, ensure_ascii=False))
        return 0
    except KeyboardInterrupt:
        logger.warning("interrupted current_case_not_checkpointed=true")
        print("[WARN] Interrupted; the current item×prior case was not checkpointed.", file=sys.stderr)
        return 130
    except Exception as exc:
        logger.exception("evaluation_failed error_type=%s", type(exc).__name__)
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
