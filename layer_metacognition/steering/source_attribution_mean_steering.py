"""Mean-difference Source Attribution steering utilities.

The positive direction is always imageward::

    mean(strong-image hidden states) - mean(strong-text hidden states)

Source exemplars and evaluation cases are separated by item_id.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from layer_metacognition.hidden_state_store import atomic_write_json
from layer_metacognition.steering.decision_side_steering import (
    CONFLICT_CONDITIONS,
    DecisionDirection,
    SteeringCase,
)


SOURCE_POSITIONS = ("ac", "panl")
SOURCE_GROUP_ORDER = ("follows_text", "follows_image")
SOURCE_SCORE_FIELD = "SA_soft_image_score"
MEAN_DIRECTION_KIND = "strong_sa_mean_difference"


def _case_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    raw_item = str(record["item_id"])
    item_key: tuple[int, Any] = (
        (0, int(raw_item)) if raw_item.isdigit() else (1, raw_item)
    )
    return (*item_key, int(record["prior_index"]), str(record["condition"]))


def load_sa_candidates(
    experiment_dir: str | Path,
    manifest_path: str | Path,
    *,
    conditions: Sequence[str] = CONFLICT_CONDITIONS,
) -> list[dict[str, Any]]:
    """Join eligible V4 manifests with completed baseline SA scores."""
    experiment_dir = Path(experiment_dir).resolve()
    manifest_path = Path(manifest_path).resolve()
    allowed_conditions = {str(value) for value in conditions}
    manifests: dict[str, dict[str, Any]] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Manifest row {line_number} is not an object")
            if not (
                record.get("eligible_decision_side_probe") is True
                and record.get("version") == "v4"
                and record.get("condition") in allowed_conditions
                and record.get("decision_side") in SOURCE_GROUP_ORDER
            ):
                continue
            case_id = str(record.get("case_id", ""))
            if not case_id or case_id in manifests:
                raise ValueError(f"Duplicate or empty eligible case_id: {case_id!r}")
            if not isinstance(record.get("hidden_state_reference"), dict):
                raise ValueError(f"Candidate omits hidden-state reference: {case_id}")
            manifests[case_id] = record
    if not manifests:
        raise ValueError("No eligible V4 conflict candidates were found")

    candidates: list[dict[str, Any]] = []
    seen_results: set[str] = set()
    results_path = experiment_dir / "results.jsonl"
    with results_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            result = json.loads(line)
            case_id = str(result.get("case_id", ""))
            if case_id not in manifests:
                continue
            if case_id in seen_results:
                raise ValueError(f"Duplicate baseline case at line {line_number}: {case_id}")
            seen_results.add(case_id)
            if result.get("status") != "completed":
                continue
            generated = result.get("generated")
            source = generated.get("source_attribution") if isinstance(generated, dict) else None
            if not isinstance(source, dict):
                raise ValueError(f"Baseline source attribution is missing: {case_id}")
            score = float(source.get("soft_image_score"))
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"Invalid SA soft image score for {case_id}: {score}")
            candidates.append(
                {
                    **manifests[case_id],
                    SOURCE_SCORE_FIELD: score,
                    "SA_hard_label": str(source.get("hard_label")),
                    "SA_parsed_label": (
                        str(source.get("parsed_label"))
                        if source.get("parsed_label") is not None
                        else None
                    ),
                }
            )
    if not candidates:
        raise ValueError("No completed baseline SA candidates were found")
    return candidates


def select_strong_sa_sources(
    candidates: Sequence[dict[str, Any]],
    cases_per_side: int = 25,
) -> dict[str, list[dict[str, Any]]]:
    """Select disjoint-item strongest text and image SA exemplars.

    The scarcer follows_text group is selected first. Text strength is a low
    soft-image score; image strength is a high soft-image score.
    """
    if cases_per_side < 1:
        raise ValueError("cases_per_side must be positive")
    selected: dict[str, list[dict[str, Any]]] = {}
    used_items: set[str] = set()
    for side in SOURCE_GROUP_ORDER:
        side_candidates = [
            dict(record) for record in candidates if record.get("decision_side") == side
        ]
        side_candidates.sort(
            key=lambda record: (
                float(record[SOURCE_SCORE_FIELD])
                if side == "follows_text"
                else -float(record[SOURCE_SCORE_FIELD]),
                _case_sort_key(record),
                str(record["case_id"]),
            )
        )
        group: list[dict[str, Any]] = []
        for record in side_candidates:
            item_id = str(record["item_id"])
            if item_id in used_items:
                continue
            used_items.add(item_id)
            group.append(record)
            if len(group) == cases_per_side:
                break
        if len(group) != cases_per_side:
            raise ValueError(
                f"Only {len(group)} disjoint-item {side} candidates are available; "
                f"requested {cases_per_side}"
            )
        selected[side] = group
    return selected


def source_manifest_payload(
    groups: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    all_items = [
        str(record["item_id"])
        for side in SOURCE_GROUP_ORDER
        for record in groups[side]
    ]
    if len(all_items) != len(set(all_items)):
        raise ValueError("Source groups reuse an item_id")
    output_groups: dict[str, Any] = {}
    for side in SOURCE_GROUP_ORDER:
        records = groups[side]
        scores = [float(record[SOURCE_SCORE_FIELD]) for record in records]
        output_groups[side] = {
            "count": len(records),
            "score_mean": float(np.mean(scores)),
            "score_min": min(scores),
            "score_max": max(scores),
            "cases": [
                {
                    "rank": rank,
                    "case_id": str(record["case_id"]),
                    "item_id": str(record["item_id"]),
                    "prior_index": int(record["prior_index"]),
                    "condition": str(record["condition"]),
                    "decision_side": str(record["decision_side"]),
                    SOURCE_SCORE_FIELD: float(record[SOURCE_SCORE_FIELD]),
                    "SA_hard_label": str(record["SA_hard_label"]),
                    "SA_parsed_label": record["SA_parsed_label"],
                }
                for rank, record in enumerate(records, 1)
            ],
        }
    return {
        "format_version": 1,
        "selection_metric": SOURCE_SCORE_FIELD,
        "positive_semantics": "higher_is_more_image",
        "follows_text_order": "ascending_SA_soft_image_score",
        "follows_image_order": "descending_SA_soft_image_score",
        "selection_group_order": list(SOURCE_GROUP_ORDER),
        "one_case_per_item": True,
        "cross_group_item_reuse": False,
        "groups": output_groups,
    }


def build_mean_directions(
    *,
    groups: dict[str, list[dict[str, Any]]],
    hidden_states: Any,
    layers: Sequence[int],
    positions: Sequence[str],
) -> tuple[dict[tuple[int, str], DecisionDirection], list[dict[str, Any]]]:
    directions: dict[tuple[int, str], DecisionDirection] = {}
    metadata: list[dict[str, Any]] = []
    for layer in layers:
        for position in positions:
            image_matrix = np.stack(
                [
                    hidden_states.get(record, int(layer), str(position))
                    for record in groups["follows_image"]
                ]
            ).astype(np.float64, copy=False)
            text_matrix = np.stack(
                [
                    hidden_states.get(record, int(layer), str(position))
                    for record in groups["follows_text"]
                ]
            ).astype(np.float64, copy=False)
            image_mean = np.mean(image_matrix, axis=0, dtype=np.float64)
            text_mean = np.mean(text_matrix, axis=0, dtype=np.float64)
            difference = image_mean - text_mean
            norm = float(np.linalg.norm(difference))
            if not math.isfinite(norm) or norm <= 0.0:
                raise ValueError(
                    f"Mean SA direction is zero/non-finite: layer={layer} "
                    f"position={position}"
                )
            unit = difference / norm
            midpoint = (image_mean + text_mean) / 2.0
            file_name = f"layer_{int(layer):02d}_{position}.npz"
            direction = DecisionDirection(
                file=file_name,
                fold=0,
                layer=int(layer),
                position=str(position),
                d_raw=unit,
                d_K=unit,
                raw_intercept=-float(np.dot(unit, midpoint)),
                steering_vector=difference,
                direction_kind=MEAN_DIRECTION_KIND,
            )
            directions[(int(layer), str(position))] = direction
            metadata.append(
                {
                    "file": file_name,
                    "layer": int(layer),
                    "position": str(position),
                    "hidden_size": int(difference.size),
                    "source_count_per_side": len(groups["follows_image"]),
                    "difference_l2": norm,
                    "image_mean_l2": float(np.linalg.norm(image_mean)),
                    "text_mean_l2": float(np.linalg.norm(text_mean)),
                    "positive_direction": "+alpha -> imageward",
                    "vector_definition": "mean_image_minus_mean_text",
                    "injection_vector": "alpha_times_raw_mean_difference",
                    "image_mean": image_mean,
                    "text_mean": text_mean,
                    "difference": difference,
                    "unit_direction": unit,
                    "midpoint": midpoint,
                }
            )
    return directions, metadata


class MeanSADirectionRepository:
    """In-memory repository matching the Decision Steering runner protocol."""

    def __init__(
        self,
        *,
        directions: dict[tuple[int, str], DecisionDirection],
        manifest_path: str | Path,
    ) -> None:
        if not directions:
            raise ValueError("Mean SA direction repository is empty")
        self._directions = dict(directions)
        self.manifest_path = Path(manifest_path).resolve()

    def fold_for_item(self, _item_id: str) -> int:
        return 0

    def available_layers(self, position: str) -> list[int]:
        return sorted(
            layer
            for layer, entry_position in self._directions
            if entry_position == str(position)
        )

    def trajectory_layers(self, injection_layer: int, position: str) -> list[int]:
        layers = [
            layer
            for layer in self.available_layers(position)
            if layer >= int(injection_layer)
        ]
        if int(injection_layer) not in layers:
            raise ValueError(
                f"No mean SA direction at layer {injection_layer} for {position}"
            )
        return layers

    def validate_requested_grid(
        self,
        layers: Sequence[int],
        positions: Sequence[str],
    ) -> None:
        missing = [
            (int(layer), str(position))
            for position in positions
            for layer in layers
            if (int(layer), str(position)) not in self._directions
        ]
        if missing:
            raise ValueError(f"Mean SA direction grid is missing: {missing}")

    def get(
        self,
        fold: int,
        layer: int,
        position: str,
        _version_setting: str | None = None,
    ) -> DecisionDirection:
        if int(fold) != 0:
            raise ValueError(f"Mean SA directions use a single fold; received {fold}")
        try:
            return self._directions[(int(layer), str(position))]
        except KeyError as exc:
            raise KeyError(f"No mean SA direction for layer={layer} {position}") from exc


def select_heldout_evaluation_cases(
    cases: Sequence[SteeringCase],
    *,
    excluded_item_ids: set[str],
    cases_per_side: int,
    max_cases: int | None = None,
) -> list[SteeringCase]:
    """Select item-disjoint evaluation cases, balanced by side and difficulty."""
    if cases_per_side < 1:
        raise ValueError("cases_per_side must be positive")
    if max_cases is not None and max_cases < 1:
        raise ValueError("max_cases must be positive")
    selected_by_side: dict[str, list[SteeringCase]] = {}
    used_items = set(str(value) for value in excluded_item_ids)
    for side_index, side in enumerate(SOURCE_GROUP_ORDER):
        easy_target = (cases_per_side + (1 if side_index == 0 else 0)) // 2
        hard_target = cases_per_side - easy_target
        targets = {"conflict_easy": easy_target, "conflict_hard": hard_target}
        selected: list[SteeringCase] = []
        for condition in CONFLICT_CONDITIONS:
            candidates = sorted(
                (
                    case
                    for case in cases
                    if case.manifest["decision_side"] == side
                    and case.manifest["condition"] == condition
                    and str(case.manifest["item_id"]) not in used_items
                ),
                key=lambda case: _case_sort_key(case.manifest),
            )
            condition_selected: list[SteeringCase] = []
            for case in candidates:
                item_id = str(case.manifest["item_id"])
                if item_id in used_items:
                    continue
                used_items.add(item_id)
                condition_selected.append(case)
                if len(condition_selected) == targets[condition]:
                    break
            if len(condition_selected) != targets[condition]:
                raise ValueError(
                    f"Only {len(condition_selected)} held-out {side}/{condition} "
                    f"cases are available; requested {targets[condition]}"
                )
            selected.extend(condition_selected)
        selected_by_side[side] = selected

    if max_cases is None:
        output = [
            case for side in SOURCE_GROUP_ORDER for case in selected_by_side[side]
        ]
    else:
        text_count = (max_cases + 1) // 2
        image_count = max_cases - text_count
        counts = {"follows_text": text_count, "follows_image": image_count}
        output = []
        for side_index, side in enumerate(SOURCE_GROUP_ORDER):
            count = min(counts[side], len(selected_by_side[side]))
            easy_count = (count + (1 if side_index == 0 else 0)) // 2
            hard_count = count - easy_count
            easy = [
                case
                for case in selected_by_side[side]
                if case.manifest["condition"] == "conflict_easy"
            ]
            hard = [
                case
                for case in selected_by_side[side]
                if case.manifest["condition"] == "conflict_hard"
            ]
            output.extend(easy[:easy_count])
            output.extend(hard[:hard_count])
    return sorted(output, key=lambda case: _case_sort_key(case.manifest))


def persist_direction_artifacts(
    output_dir: str | Path,
    *,
    source_manifest: dict[str, Any],
    direction_metadata: Sequence[dict[str, Any]],
) -> tuple[Path, Path]:
    """Persist reproducible source selections and all mean vectors."""
    output_dir = Path(output_dir).resolve()
    source_path = output_dir / "source_cohort_manifest.json"
    direction_dir = output_dir / "directions"
    index_path = direction_dir / "index.json"
    direction_dir.mkdir(parents=True, exist_ok=True)

    if source_path.exists():
        saved = json.loads(source_path.read_text(encoding="utf-8"))
        if saved != source_manifest:
            raise ValueError("Existing source cohort manifest differs")
    else:
        atomic_write_json(source_path, source_manifest)

    index_entries: list[dict[str, Any]] = []
    for entry in direction_metadata:
        public = {
            key: value
            for key, value in entry.items()
            if not isinstance(value, np.ndarray)
        }
        path = direction_dir / str(entry["file"])
        arrays = {
            key: np.asarray(entry[key], dtype=np.float32)
            for key in (
                "image_mean",
                "text_mean",
                "difference",
                "unit_direction",
                "midpoint",
            )
        }
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if set(saved.files) != set(arrays) or any(
                    not np.array_equal(saved[key], value)
                    for key, value in arrays.items()
                ):
                    raise ValueError(f"Existing direction artifact differs: {path}")
        else:
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **arrays)
            temporary.replace(path)
        index_entries.append(public)
    index = {
        "format_version": 1,
        "direction_kind": MEAN_DIRECTION_KIND,
        "positive_direction": "+alpha -> imageward",
        "directions": index_entries,
    }
    if index_path.exists():
        saved = json.loads(index_path.read_text(encoding="utf-8"))
        if saved != index:
            raise ValueError("Existing mean direction index differs")
    else:
        atomic_write_json(index_path, index)
    return source_path, index_path
