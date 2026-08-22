"""Deterministic disjoint cohorts and mean-embedding artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from confidence_test.source_attribution_variants import SourcePromptVariant
from layer_metacognition.sa_steering.artifacts import (
    select_evaluation_cases,
    select_extreme_sources,
)
from layer_metacognition.sa_steering.runner import RuntimeCase, build_runtime_cases

from .protocol import prepare_fixed_prefix


@dataclass(frozen=True)
class EmbeddingArtifacts:
    image: torch.Tensor
    text: torch.Tensor
    text_position_counts: torch.Tensor
    answer: torch.Tensor
    metadata: dict[str, Any]


def _case_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    item = str(record["item_id"])
    item_key: tuple[int, Any] = (0, int(item)) if item.isdigit() else (1, item)
    return (
        *item_key,
        int(record["prior_index"]),
        str(record["condition"]),
        str(record["case_id"]),
    )


def _sample_unique_items(
    candidates: Sequence[dict[str, Any]],
    *,
    count: int,
    seed: int,
    excluded_items: set[str] | None = None,
) -> list[dict[str, Any]]:
    shuffled = sorted(candidates, key=_case_sort_key)
    random.Random(seed).shuffle(shuffled)
    used = set(excluded_items or ())
    output: list[dict[str, Any]] = []
    for record in shuffled:
        item = str(record["item_id"])
        if item in used:
            continue
        output.append(record)
        used.add(item)
        if len(output) == count:
            return output
    raise ValueError(f"Could not select {count} unique item-disjoint sources")


def filter_evaluation_records(
    records: Sequence[dict[str, Any]],
    conditions: Sequence[str],
) -> list[dict[str, Any]]:
    """Restrict only the evaluation cohort to the requested conditions."""

    selected_conditions = {str(value) for value in conditions}
    if not selected_conditions:
        raise ValueError("Evaluation conditions must be non-empty")
    output = [
        record
        for record in records
        if str(record.get("condition")) in selected_conditions
    ]
    if not output:
        raise ValueError(
            "No baseline records match evaluation conditions: "
            f"{sorted(selected_conditions)}"
        )
    return output


def select_cohorts(
    records: Sequence[dict[str, Any]],
    item_to_fold: dict[str, int],
    *,
    test_fold: int,
    eval_cases: int,
    seed: int,
    evaluation_conditions: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    evaluation_candidates = filter_evaluation_records(
        records,
        evaluation_conditions,
    )
    evaluation = select_evaluation_cases(
        evaluation_candidates,
        item_to_fold,
        test_fold=test_fold,
        eval_cases=eval_cases,
        seed=seed,
    )
    evaluation_items = {str(record["item_id"]) for record in evaluation}
    train = [
        record
        for record in records
        if item_to_fold[str(record["item_id"])] != test_fold
        and str(record["item_id"]) not in evaluation_items
    ]
    easy = _sample_unique_items(
        [record for record in train if str(record["condition"]).endswith("easy")],
        count=50,
        seed=seed + 101,
    )
    image_used = {str(record["item_id"]) for record in easy}
    hard = _sample_unique_items(
        [record for record in train if str(record["condition"]).endswith("hard")],
        count=50,
        seed=seed + 102,
        excluded_items=image_used,
    )
    text = _sample_unique_items(train, count=100, seed=seed + 103)
    answer_groups = select_extreme_sources(
        records,
        item_to_fold,
        test_fold=test_fold,
        cases_per_side=50,
    )
    answer = [*answer_groups["low"], *answer_groups["high"]]
    sources = {"image": [*easy, *hard], "text": text, "answer": answer}
    for name, group in sources.items():
        overlap = evaluation_items.intersection(str(record["item_id"]) for record in group)
        if overlap:
            raise ValueError(f"{name} embedding sources leak evaluation items: {sorted(overlap)}")
    metadata = {
        "test_fold": int(test_fold),
        "seed": int(seed),
        "evaluation_case_count": len(evaluation),
        "evaluation_conditions": list(evaluation_conditions),
        "evaluation_condition_counts": {
            condition: sum(record["condition"] == condition for record in evaluation)
            for condition in evaluation_conditions
        },
        "evaluation_item_count": len(evaluation_items),
        "evaluation_group_counts": {
            name: sum(record["baseline_sa_group"] == name for record in evaluation)
            for name in ("low", "high")
        },
        "source_evaluation_item_overlap": False,
        "evaluation_cases": [
            {
                "case_id": record["case_id"],
                "item_id": record["item_id"],
                "baseline_sa_group": record["baseline_sa_group"],
                "condition": record["condition"],
            }
            for record in evaluation
        ],
        "embedding_sources": {
            name: [
                {
                    "case_id": record["case_id"],
                    "item_id": record["item_id"],
                    "condition": record["condition"],
                    "baseline_label": record["baseline"]["generated_label"],
                }
                for record in group
            ]
            for name, group in sources.items()
        },
        "image_difficulty_counts": {"easy": len(easy), "hard": len(hard)},
        "answer_group_counts": {
            "low": len(answer_groups["low"]),
            "high": len(answer_groups["high"]),
        },
    }
    return evaluation, sources, metadata


def cohort_fingerprint(metadata: dict[str, Any]) -> str:
    encoded = json.dumps(
        metadata,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ragged_position_mean(vectors: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if not vectors:
        raise ValueError("Ragged mean requires at least one tensor")
    hidden_size = int(vectors[0].shape[1])
    max_tokens = max(int(vector.shape[0]) for vector in vectors)
    sums = torch.zeros(max_tokens, hidden_size, dtype=torch.float64)
    counts = torch.zeros(max_tokens, dtype=torch.int64)
    for vector in vectors:
        if vector.ndim != 2 or int(vector.shape[1]) != hidden_size:
            raise ValueError("Ragged mean hidden shape mismatch")
        if not bool(torch.isfinite(vector).all()):
            raise ValueError("Ragged mean received non-finite values")
        length = int(vector.shape[0])
        sums[:length] += vector.detach().double().cpu()
        counts[:length] += 1
    if bool((counts == 0).any()):
        raise ValueError("Ragged mean has an uncovered position")
    return (sums / counts.unsqueeze(1)).float(), counts


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _runtime_sources(
    sources: Sequence[dict[str, Any]],
    *,
    dataset: str | Path,
    output_dir: str | Path,
    source_variant: SourcePromptVariant,
    image_root: str | Path | None,
) -> list[RuntimeCase]:
    runtime, _metadata = build_runtime_cases(
        sources,
        dataset=dataset,
        output_dir=output_dir,
        source_variant=source_variant,
        image_root=image_root,
    )
    return runtime


def build_or_load_embedding_artifacts(
    *,
    output_dir: str | Path,
    sources: dict[str, list[dict[str, Any]]],
    metadata: dict[str, Any],
    dataset: str | Path,
    image_root: str | Path | None,
    source_variant: SourcePromptVariant,
    inference: Any,
    joint_generator: Any,
    hidden_size: int,
) -> EmbeddingArtifacts:
    output_dir = Path(output_dir).resolve()
    artifact_dir = output_dir / "corruption_embeddings"
    paths = {
        "image": artifact_dir / "image_mean_embedding.pt",
        "text": artifact_dir / "text_mean_embedding.pt",
        "answer": artifact_dir / "answer_mean_embedding.pt",
    }
    fingerprint = cohort_fingerprint(metadata)
    if all(path.is_file() for path in paths.values()):
        payloads = {
            name: torch.load(path, map_location="cpu", weights_only=False)
            for name, path in paths.items()
        }
        if any(payload.get("cohort_fingerprint") != fingerprint for payload in payloads.values()):
            raise ValueError("Existing corruption embedding cohort differs from this run")
        image = payloads["image"]["mean_embedding"].float()
        text = payloads["text"]["mean_embedding"].float()
        counts = payloads["text"]["position_counts"].long()
        answer = payloads["answer"]["mean_embedding"].float()
    else:
        if any(path.exists() for path in paths.values()):
            raise ValueError("Corruption embedding artifacts are only partially present")
        runtime = {
            name: _runtime_sources(
                group,
                dataset=dataset,
                output_dir=output_dir,
                source_variant=source_variant,
                image_root=image_root,
            )
            for name, group in sources.items()
        }
        image_vectors: list[torch.Tensor] = []
        for case in runtime["image"]:
            prepared = prepare_fixed_prefix(
                joint_generator,
                case,
                case.record["fixed_answer"],
                positions=("ac",),
            )
            inputs = prepared.inputs
            pixel_values = inputs.get("pixel_values")
            image_grid = inputs.get("image_grid_thw")
            if pixel_values is None or image_grid is None:
                raise ValueError("Image embedding source is missing vision tensors")
            with torch.inference_mode():
                features = inference.model.get_image_features(pixel_values, image_grid)
            vector = torch.cat(list(features), dim=0).detach().float().cpu()
            if vector.shape != (len(prepared.spans["image"]), hidden_size):
                raise ValueError(
                    f"Image embedding shape mismatch: {tuple(vector.shape)} vs "
                    f"{(len(prepared.spans['image']), hidden_size)}"
                )
            image_vectors.append(vector)
            del prepared, inputs, features
        shapes = {tuple(vector.shape) for vector in image_vectors}
        if len(shapes) != 1:
            raise ValueError(f"Image embedding sources have inconsistent shapes: {shapes}")
        image = torch.stack(image_vectors).mean(dim=0)

        text_vectors: list[torch.Tensor] = []
        for case in runtime["text"]:
            prepared = prepare_fixed_prefix(
                joint_generator,
                case,
                case.record["fixed_answer"],
                positions=("ac",),
            )
            with torch.inference_mode():
                embeddings = inference.model.get_input_embeddings()(
                    prepared.inputs.input_ids
                )
            text_vectors.append(
                embeddings[0, list(prepared.spans["text"]), :].detach().float().cpu()
            )
            del prepared, embeddings
        text, counts = ragged_position_mean(text_vectors)

        answer_vectors: list[torch.Tensor] = []
        for case in runtime["answer"]:
            prepared = prepare_fixed_prefix(
                joint_generator,
                case,
                case.record["fixed_answer"],
                positions=("ac",),
            )
            with torch.inference_mode():
                embeddings = inference.model.get_input_embeddings()(
                    prepared.inputs.input_ids
                )
            answer_vectors.append(
                embeddings[0, list(prepared.spans["answer"]), :].detach().float().cpu()
            )
            del prepared, embeddings
        answer_shapes = {tuple(vector.shape) for vector in answer_vectors}
        if len(answer_shapes) != 1:
            raise ValueError(f"Answer embedding sources have inconsistent shapes: {answer_shapes}")
        answer = torch.stack(answer_vectors).mean(dim=0)

        common = {
            "format_version": 1,
            "cohort_fingerprint": fingerprint,
            "hidden_size": int(hidden_size),
            "dtype": "float32",
        }
        _atomic_torch_save(
            paths["image"],
            {**common, "mean_embedding": image, "source_count": 100},
        )
        _atomic_torch_save(
            paths["text"],
            {
                **common,
                "mean_embedding": text,
                "position_counts": counts,
                "source_count": 100,
                "aggregation": "ragged_position_mean",
            },
        )
        _atomic_torch_save(
            paths["answer"],
            {**common, "mean_embedding": answer, "source_count": 100},
        )
    if image.ndim != 2 or int(image.shape[1]) != hidden_size:
        raise ValueError("Invalid image mean embedding shape")
    if text.ndim != 2 or int(text.shape[1]) != hidden_size or len(counts) != len(text):
        raise ValueError("Invalid text mean embedding shape")
    if answer.ndim != 2 or int(answer.shape[1]) != hidden_size:
        raise ValueError("Invalid answer mean embedding shape")
    if not all(bool(torch.isfinite(value).all()) for value in (image, text, answer)):
        raise ValueError("Mean embedding artifact contains non-finite values")
    return EmbeddingArtifacts(
        image=image.contiguous(),
        text=text.contiguous(),
        text_position_counts=counts.contiguous(),
        answer=answer.contiguous(),
        metadata={
            "cohort_fingerprint": fingerprint,
            "paths": {name: str(path) for name, path in paths.items()},
            "shapes": {
                "image": list(image.shape),
                "text": list(text.shape),
                "answer": list(answer.shape),
            },
            "text_position_counts": counts.tolist(),
        },
    )
