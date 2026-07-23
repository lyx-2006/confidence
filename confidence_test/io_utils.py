"""Atomic result persistence and deterministic simplified JSON rendering."""

from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

from confidence_test.dataset_utils import CONDITIONS, EvaluationCase


def atomic_write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_json(path: str | Path, default: Any = None) -> Any:
    source = Path(path)
    if not source.exists():
        return deepcopy(default)
    with source.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    atomic_write_text(target, json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    with target.open("r", encoding="utf-8") as handle:
        json.load(handle)


def empty_condition(status: str = "not_selected") -> dict[str, Any]:
    return {
        "status": status,
        "relative_image_path": None,
        "resolved_image_path": None,
        "answer_result": None,
        "confidence_result": None,
        "delta_soft_confidence": None,
        "elapsed_seconds": 0.0,
        "error": None,
    }


def new_prior_record(case: EvaluationCase, version: str) -> dict[str, Any]:
    return {
        "record_key": case.record_key,
        "prior_index": case.prior_index,
        "prior_bin": case.prior_bin,
        "text_clue": case.text_clue,
        "text_answer": None,
        "text_conf": None,
        "text_stage": None if version == "v4" else {
            "status": "pending",
            "answer_result": None,
            "confidence_result": None,
            "elapsed_seconds": 0.0,
            "error": None,
        },
        "conditions": {condition: empty_condition() for condition in CONDITIONS},
    }


def find_prior(results: list[dict[str, Any]], record_key: str) -> dict[str, Any] | None:
    for sample in results:
        for prior in sample.get("priors", []):
            if prior.get("record_key") == record_key:
                return prior
    return None


def upsert_prior(
    results: list[dict[str, Any]],
    case: EvaluationCase,
    prior_record: dict[str, Any],
    run_config: dict[str, Any],
    item_order: dict[str, int],
) -> None:
    sample = next((entry for entry in results if str(entry.get("id")) == case.item_id), None)
    if sample is None:
        sample = {
            "id": case.item_id,
            "ground_truth_answer": case.ground_truth_answer,
            "run_config": deepcopy(run_config),
            "priors": [],
        }
        results.append(sample)
    sample["ground_truth_answer"] = case.ground_truth_answer
    sample["run_config"] = deepcopy(run_config)
    priors = sample.setdefault("priors", [])
    existing_index = next(
        (index for index, value in enumerate(priors) if value.get("record_key") == case.record_key),
        None,
    )
    if existing_index is None:
        priors.append(deepcopy(prior_record))
    else:
        priors[existing_index] = deepcopy(prior_record)
    priors.sort(key=lambda value: int(value.get("prior_index", 0)))
    # Existing sample order is authoritative on resume. Fresh samples are
    # appended while cases are traversed in dataset order.
    for value in results:
        value["run_config"] = deepcopy(run_config)


def _simplified_values(condition: dict[str, Any]) -> list[Any]:
    if condition.get("status") != "completed":
        return [None, None, None, None]
    answer = condition.get("answer_result") or {}
    confidence = condition.get("confidence_result") or {}
    values = [
        answer.get("answer"),
        answer.get("answer_prob"),
        answer.get("answer_entropy"),
        confidence.get("soft_confidence"),
    ]
    return values if all(value is not None for value in values) else [None, None, None, None]


def full_to_simplified(results: list[dict[str, Any]], version: str) -> list[dict[str, Any]]:
    simplified: list[dict[str, Any]] = []
    for sample in results:
        sample_value = {
            "id": str(sample.get("id")),
            "ground_truth_answer": sample.get("ground_truth_answer"),
            "priors": [],
        }
        for prior in sample.get("priors", []):
            text_stage = prior.get("text_stage") or {}
            text_answer_result = text_stage.get("answer_result") or {}
            text_confidence_result = text_stage.get("confidence_result") or {}
            prior_value = {
                "prior_index": prior.get("prior_index"),
                "prior_bin": prior.get("prior_bin"),
                "text_answer": None if version == "v4" else text_answer_result.get("answer"),
                "text_conf": None if version == "v4" else text_confidence_result.get("soft_confidence"),
                "conditions": {
                    name: _simplified_values((prior.get("conditions") or {}).get(name, {}))
                    for name in CONDITIONS
                },
            }
            sample_value["priors"].append(prior_value)
        simplified.append(sample_value)
    return simplified


def write_compact_simplified_json(path: Path, data: list[dict[str, Any]]) -> None:
    lines = ["["]
    for sample_index, sample in enumerate(data):
        lines.append("  {")
        lines.append(f"    \"id\": {json.dumps(sample.get('id'), ensure_ascii=False)},")
        lines.append(
            "    \"ground_truth_answer\": "
            + json.dumps(sample.get("ground_truth_answer"), ensure_ascii=False)
            + ","
        )
        lines.append("    \"priors\": [")
        priors = sample.get("priors", [])
        for prior_index, prior in enumerate(priors):
            lines.append("      {")
            lines.append(f"        \"prior_index\": {json.dumps(prior.get('prior_index'))},")
            lines.append(
                "        \"prior_bin\": "
                + json.dumps(prior.get("prior_bin"), ensure_ascii=False)
                + ","
            )
            lines.append(
                "        \"text_answer\": "
                + json.dumps(prior.get("text_answer"), ensure_ascii=False)
                + ","
            )
            lines.append(
                "        \"text_conf\": " + json.dumps(prior.get("text_conf")) + ","
            )
            lines.append("        \"conditions\": {")
            for condition_index, condition in enumerate(CONDITIONS):
                comma = "," if condition_index < len(CONDITIONS) - 1 else ""
                values = prior.get("conditions", {}).get(condition, [None, None, None, None])
                lines.append(
                    f"          {json.dumps(condition)}: "
                    f"{json.dumps(values, ensure_ascii=False)}{comma}"
                )
            lines.append("        }")
            prior_comma = "," if prior_index < len(priors) - 1 else ""
            lines.append(f"      }}{prior_comma}")
        lines.append("    ]")
        sample_comma = "," if sample_index < len(data) - 1 else ""
        lines.append(f"  }}{sample_comma}")
    lines.append("]")
    target = Path(path)
    atomic_write_text(target, "\n".join(lines) + "\n")
    with target.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if loaded != data:
        raise ValueError(f"Compact simplified JSON round-trip mismatch: {target}")


def write_result_pair(
    output_dir: Path,
    version: str,
    results: list[dict[str, Any]],
) -> tuple[Path, Path]:
    full_path = output_dir / f"{version}_results.json"
    simplified_path = output_dir / f"{version}_simplified.json"
    atomic_write_json(full_path, results)
    write_compact_simplified_json(
        simplified_path,
        full_to_simplified(results, version),
    )
    return full_path, simplified_path
