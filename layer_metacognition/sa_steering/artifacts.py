"""Input validation, deterministic cohorts, and SA steering vectors."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from layer_metacognition.hidden_state_store import atomic_write_json
from layer_metacognition.probe.hidden_state_loader import HiddenStateLoader

from . import (
    HIGH_SA_CLASSES,
    HIDDEN_STATE_DEFINITION,
    LOW_SA_CLASSES,
    VECTOR_NORM_FRACTION,
)


@dataclass(frozen=True)
class SteeringVector:
    method: str
    position: str
    layer: int
    vector: np.ndarray
    raw_direction: np.ndarray
    vector_norm: float
    raw_direction_norm: float
    hidden_norm_mean: float
    target_norm: float
    metadata: dict[str, Any]


def read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {source}")
    return value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configuration_fingerprint(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object: {source}:{line_number}")
            yield value


def _finite_number(value: Any, name: str, case_id: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{case_id}: invalid {name}: {value!r}")
    return float(value)


def load_baseline_records(
    experiment_dir: str | Path,
    *,
    layers: Sequence[int],
    positions: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load completed V4 joint cases and the exact reusable SA baseline."""

    experiment_dir = Path(experiment_dir).resolve()
    config_path = experiment_dir / "config.json"
    results_path = experiment_dir / "results.jsonl"
    index_path = experiment_dir / "hidden_states" / "index.json"
    for path in (config_path, results_path, index_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required source artifact is missing: {path}")
    config = read_json(config_path)
    if config.get("versions") != ["v4"] or config.get("attribution_mode") != "joint":
        raise ValueError("SA steering requires the completed V4 joint source experiment")
    index = read_json(index_path)
    indexed_cases = index.get("cases")
    if not isinstance(indexed_cases, dict):
        raise ValueError("Hidden-state index has no cases object")
    available_layers = [int(value) for value in index.get("layer_indices", [])]
    available_positions = [str(value) for value in index.get("position_names", [])]
    missing_layers = sorted(set(map(int, layers)) - set(available_layers))
    missing_positions = sorted(set(map(str, positions)) - set(available_positions))
    if missing_layers or missing_positions:
        raise ValueError(
            "Hidden-state schema does not cover the steering grid: "
            f"missing_layers={missing_layers}, missing_positions={missing_positions}"
        )

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    skipped = 0
    for source in iter_jsonl(results_path):
        if not (
            source.get("status") == "completed"
            and source.get("version") == "v4"
            and source.get("attribution_mode") == "joint"
        ):
            skipped += 1
            continue
        case_id = str(source.get("case_id", ""))
        if not case_id or case_id in seen:
            raise ValueError(f"Duplicate or empty completed case_id: {case_id!r}")
        generated = source.get("generated")
        sa = generated.get("source_attribution") if isinstance(generated, dict) else None
        answer = generated.get("current_answer") if isinstance(generated, dict) else None
        reference = source.get("hidden_state_reference")
        if not isinstance(sa, dict) or not isinstance(reference, dict):
            raise ValueError(f"{case_id}: reusable baseline or hidden reference is missing")
        if not isinstance(answer, str) or not answer.strip() or "\n" in answer:
            raise ValueError(f"{case_id}: reusable generated answer is invalid: {answer!r}")
        parsed_label = sa.get("parsed_label")
        hard_label = sa.get("hard_label")
        if str(parsed_label) not in {str(index) for index in range(9)}:
            raise ValueError(f"{case_id}: invalid parsed SA label: {parsed_label!r}")
        if str(hard_label) not in {str(index) for index in range(9)}:
            raise ValueError(f"{case_id}: invalid scored SA label: {hard_label!r}")
        logits = sa.get("class_logits")
        probabilities = sa.get("class_probabilities")
        if not (
            isinstance(logits, list)
            and isinstance(probabilities, list)
            and len(logits) == 9
            and len(probabilities) == 9
        ):
            raise ValueError(f"{case_id}: baseline SA logits/probabilities are incomplete")
        indexed = indexed_cases.get(case_id)
        if not isinstance(indexed, dict):
            raise ValueError(f"{case_id}: absent from hidden-state index")
        for key in ("shard_path", "offset", "hidden_size", "hidden_state_definition"):
            if reference.get(key) != indexed.get(key):
                raise ValueError(f"{case_id}: hidden reference/index mismatch for {key}")
        if reference.get("hidden_state_definition") != HIDDEN_STATE_DEFINITION:
            raise ValueError(f"{case_id}: unexpected hidden-state definition")
        record = {
            "case_id": case_id,
            "item_id": str(source["item_id"]),
            "prior_index": int(source["prior_index"]),
            "condition": str(source["condition"]),
            "version": "v4",
            "fixed_answer": answer.strip(),
            "baseline": {
                "raw_output": sa.get("raw_output"),
                "generated_label": str(parsed_label),
                "hard_label": str(hard_label),
                "soft_score": _finite_number(sa.get("soft_image_score"), "soft score", case_id),
                "class_logits": [
                    _finite_number(value, "class logit", case_id) for value in logits
                ],
                "class_probabilities": [
                    _finite_number(value, "class probability", case_id)
                    for value in probabilities
                ],
                "entropy": _finite_number(sa.get("source_entropy"), "entropy", case_id),
            },
            "hidden_state_reference": dict(reference),
        }
        records.append(record)
        seen.add(case_id)
    if not records:
        raise ValueError("No completed reusable SA baselines were found")
    return records, {
        "source_config_path": str(config_path),
        "source_results_path": str(results_path),
        "hidden_state_index_path": str(index_path),
        "source_config_sha256": sha256_file(config_path),
        "source_results_sha256": sha256_file(results_path),
        "hidden_state_index_sha256": sha256_file(index_path),
        "eligible_case_count": len(records),
        "skipped_source_record_count": skipped,
        "available_layers": available_layers,
        "available_positions": available_positions,
    }


def load_item_folds(probe_dir: str | Path, records: Sequence[dict[str, Any]]) -> dict[str, int]:
    path = Path(probe_dir).resolve() / "split_assignments.json"
    assignment = read_json(path)
    raw = assignment.get("item_to_fold")
    if assignment.get("group_key") != "item_id" or not isinstance(raw, dict):
        raise ValueError("SA probe split assignments are not item-grouped")
    mapping = {str(key): int(value) for key, value in raw.items()}
    missing = sorted({record["item_id"] for record in records} - set(mapping))
    if missing:
        raise ValueError(f"Split assignments omit source items: {missing[:10]}")
    return mapping


def _case_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    item = str(record["item_id"])
    item_key: tuple[int, Any] = (0, int(item)) if item.isdigit() else (1, item)
    return (*item_key, int(record["prior_index"]), str(record["condition"]), str(record["case_id"]))


def select_evaluation_cases(
    records: Sequence[dict[str, Any]],
    item_to_fold: dict[str, int],
    *,
    test_fold: int,
    eval_cases: int,
    seed: int,
) -> list[dict[str, Any]]:
    if eval_cases < 2 or eval_cases % 2:
        raise ValueError("--eval-cases must be a positive even number")
    half = eval_cases // 2
    fold_records = [
        record for record in records if item_to_fold[record["item_id"]] == test_fold
    ]
    groups = {
        "low": [
            record
            for record in fold_records
            if record["baseline"]["generated_label"] in LOW_SA_CLASSES
        ],
        "high": [
            record
            for record in fold_records
            if record["baseline"]["generated_label"] in HIGH_SA_CLASSES
        ],
    }
    selected: list[dict[str, Any]] = []
    for offset, group_name in enumerate(("low", "high")):
        candidates = sorted(groups[group_name], key=_case_sort_key)
        if len(candidates) < half:
            raise ValueError(
                f"Only {len(candidates)} fold-{test_fold} {group_name}-SA cases; need {half}"
            )
        random.Random(seed + offset).shuffle(candidates)
        for record in candidates[:half]:
            selected.append({**record, "baseline_sa_group": group_name})
    return sorted(selected, key=_case_sort_key)


def select_extreme_sources(
    records: Sequence[dict[str, Any]],
    item_to_fold: dict[str, int],
    *,
    test_fold: int,
    cases_per_side: int,
) -> dict[str, list[dict[str, Any]]]:
    if cases_per_side < 1:
        raise ValueError("--source-cases-per-side must be positive")
    train = [record for record in records if item_to_fold[record["item_id"]] != test_fold]
    used_items: set[str] = set()
    output: dict[str, list[dict[str, Any]]] = {}
    for name, reverse in (("low", False), ("high", True)):
        candidates = sorted(
            train,
            key=lambda record: (
                -float(record["baseline"]["soft_score"])
                if reverse
                else float(record["baseline"]["soft_score"]),
                _case_sort_key(record),
            ),
        )
        group: list[dict[str, Any]] = []
        for record in candidates:
            if record["item_id"] in used_items:
                continue
            group.append(record)
            used_items.add(record["item_id"])
            if len(group) == cases_per_side:
                break
        if len(group) != cases_per_side:
            raise ValueError(f"Could not select {cases_per_side} item-disjoint {name} sources")
        output[name] = group
    return output


def cohort_manifest(
    evaluation: Sequence[dict[str, Any]],
    sources: dict[str, list[dict[str, Any]]],
    item_to_fold: dict[str, int],
    *,
    test_fold: int,
) -> dict[str, Any]:
    train_items = {
        record["item_id"] for group in sources.values() for record in group
    }
    evaluation_items = {record["item_id"] for record in evaluation}
    if train_items.intersection(evaluation_items):
        raise ValueError("Steering source and evaluation cohorts overlap by item_id")
    return {
        "test_fold": int(test_fold),
        "split_mode": "item",
        "source_evaluation_item_overlap": False,
        "evaluation_case_count": len(evaluation),
        "evaluation_item_count": len(evaluation_items),
        "evaluation_group_counts": {
            name: sum(record["baseline_sa_group"] == name for record in evaluation)
            for name in ("low", "high")
        },
        "evaluation_label_counts": {
            label: sum(record["baseline"]["generated_label"] == label for record in evaluation)
            for label in sorted({record["baseline"]["generated_label"] for record in evaluation})
        },
        "evaluation_cases": [
            {
                "case_id": record["case_id"],
                "item_id": record["item_id"],
                "fold": item_to_fold[record["item_id"]],
                "baseline_sa_group": record["baseline_sa_group"],
                "baseline_generated_label": record["baseline"]["generated_label"],
                "baseline_soft_score": record["baseline"]["soft_score"],
            }
            for record in evaluation
        ],
        "source_groups": {
            name: [
                {
                    "rank": index + 1,
                    "case_id": record["case_id"],
                    "item_id": record["item_id"],
                    "fold": item_to_fold[record["item_id"]],
                    "soft_score": record["baseline"]["soft_score"],
                }
                for index, record in enumerate(group)
            ]
            for name, group in sources.items()
        },
    }


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _normalize_direction(
    direction: np.ndarray,
    hidden_norms: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    raw = np.asarray(direction, dtype=np.float64).reshape(-1)
    raw_norm = float(np.linalg.norm(raw))
    hidden_norm_mean = float(np.mean(np.asarray(hidden_norms, dtype=np.float64)))
    target_norm = VECTOR_NORM_FRACTION * hidden_norm_mean
    if not all(math.isfinite(value) and value > 0 for value in (raw_norm, hidden_norm_mean, target_norm)):
        raise ValueError(
            "Cannot normalize non-finite/zero steering direction or hidden norm: "
            f"raw={raw_norm}, hidden={hidden_norm_mean}, target={target_norm}"
        )
    vector = raw / raw_norm * target_norm
    return vector.astype(np.float32), raw_norm, hidden_norm_mean, target_norm


def _load_oof_soft_predictions(
    probe_dir: Path,
    layers: Sequence[int],
    positions: Sequence[str],
    test_fold: int,
) -> dict[tuple[str, int, str], float]:
    wanted_layers = set(map(int, layers))
    wanted_positions = set(map(str, positions))
    output: dict[tuple[str, int, str], float] = {}
    path = probe_dir / "predictions" / "oof_predictions.jsonl"
    for record in iter_jsonl(path):
        if not (
            record.get("task") == "soft_score"
            and int(record.get("fold", -1)) == test_fold
            and int(record.get("layer", -1)) in wanted_layers
            and str(record.get("position")) in wanted_positions
        ):
            continue
        key = (str(record["position"]), int(record["layer"]), str(record["case_id"]))
        if key in output:
            raise ValueError(f"Duplicate OOF soft prediction: {key}")
        output[key] = float(record["predicted_score"])
    return output


class SteeringVectorRepository:
    def __init__(self, vector_dir: str | Path):
        self.vector_dir = Path(vector_dir).resolve()
        self.index_path = self.vector_dir / "index.json"
        self.index = read_json(self.index_path)
        entries = self.index.get("vectors")
        if not isinstance(entries, list) or not entries:
            raise ValueError("Steering vector index is empty")
        self._entries = {
            (str(entry["method"]), str(entry["position"]), int(entry["layer"])): entry
            for entry in entries
        }
        if len(self._entries) != len(entries):
            raise ValueError("Steering vector index contains duplicate cells")
        self._cache: dict[tuple[str, str, int], SteeringVector] = {}

    def get(self, method: str, position: str, layer: int) -> SteeringVector:
        key = (str(method), str(position), int(layer))
        if key in self._cache:
            return self._cache[key]
        if key not in self._entries:
            raise KeyError(f"No steering vector for {key}")
        entry = self._entries[key]
        path = self.vector_dir / str(entry["file"])
        with np.load(path) as payload:
            vector = np.asarray(payload["vector"], dtype=np.float32)
            raw = np.asarray(payload["raw_direction"], dtype=np.float64)
        if vector.ndim != 1 or raw.shape != vector.shape or not np.isfinite(vector).all():
            raise ValueError(f"Invalid steering vector artifact: {path}")
        result = SteeringVector(
            method=key[0],
            position=key[1],
            layer=key[2],
            vector=vector,
            raw_direction=raw,
            vector_norm=float(entry["vector_norm"]),
            raw_direction_norm=float(entry["raw_direction_norm"]),
            hidden_norm_mean=float(entry["hidden_norm_mean"]),
            target_norm=float(entry["target_norm"]),
            metadata=dict(entry),
        )
        self._cache[key] = result
        return result


def build_or_load_vectors(
    *,
    vector_dir: str | Path,
    records: Sequence[dict[str, Any]],
    item_to_fold: dict[str, int],
    sources: dict[str, list[dict[str, Any]]],
    experiment_dir: str | Path,
    probe_dir: str | Path,
    layers: Sequence[int],
    positions: Sequence[str],
    test_fold: int,
    source_fingerprint: str,
) -> SteeringVectorRepository:
    vector_dir = Path(vector_dir).resolve()
    index_path = vector_dir / "index.json"
    expected = {
        "format_version": 1,
        "source_fingerprint": source_fingerprint,
        "layers": list(map(int, layers)),
        "positions": list(map(str, positions)),
        "methods": ["mean_difference", "probe_weight"],
        "test_fold": int(test_fold),
        "normalization_fraction": VECTOR_NORM_FRACTION,
    }
    if index_path.is_file():
        existing = read_json(index_path)
        for key, value in expected.items():
            if existing.get(key) != value:
                raise ValueError(f"Existing steering vector index differs for {key}")
        repository = SteeringVectorRepository(vector_dir)
        for position in positions:
            for layer in layers:
                for method in expected["methods"]:
                    repository.get(method, position, layer)
        return repository

    vector_dir.mkdir(parents=True, exist_ok=True)
    loader = HiddenStateLoader(experiment_dir, cache_size=4)
    probe_dir = Path(probe_dir).resolve()
    oof = _load_oof_soft_predictions(probe_dir, layers, positions, test_fold)
    train_records = [
        record for record in records if item_to_fold[record["item_id"]] != test_fold
    ]
    test_records = [
        record for record in records if item_to_fold[record["item_id"]] == test_fold
    ]
    record_index = {record["case_id"]: index for index, record in enumerate(records)}
    source_low_indices = [record_index[record["case_id"]] for record in sources["low"]]
    source_high_indices = [record_index[record["case_id"]] for record in sources["high"]]
    train_indices = [record_index[record["case_id"]] for record in train_records]
    test_indices = [record_index[record["case_id"]] for record in test_records]
    entries: list[dict[str, Any]] = []

    for position in positions:
        for layer in layers:
            matrix = np.stack(
                [loader.load_vector(record, int(layer), str(position)) for record in records]
            ).astype(np.float32, copy=False)
            if matrix.ndim != 2 or not np.isfinite(matrix).all():
                raise ValueError(f"Invalid hidden matrix: position={position}, layer={layer}")

            low = matrix[source_low_indices].astype(np.float64, copy=False)
            high = matrix[source_high_indices].astype(np.float64, copy=False)
            mean_raw = np.mean(high, axis=0) - np.mean(low, axis=0)
            mean_norms = np.linalg.norm(np.concatenate([low, high], axis=0), axis=1)
            mean_vector, raw_norm, hidden_mean, target_norm = _normalize_direction(
                mean_raw, mean_norms
            )
            mean_file = f"mean_difference__{position}__layer_{int(layer):02d}.npz"
            _atomic_save_npz(
                vector_dir / mean_file,
                vector=mean_vector,
                raw_direction=np.asarray(mean_raw, dtype=np.float64),
            )
            entries.append(
                {
                    "method": "mean_difference",
                    "position": str(position),
                    "layer": int(layer),
                    "file": mean_file,
                    "raw_direction_norm": raw_norm,
                    "hidden_norm_mean": hidden_mean,
                    "target_norm": target_norm,
                    "vector_norm": float(np.linalg.norm(mean_vector.astype(np.float64))),
                    "source_count_low": len(low),
                    "source_count_high": len(high),
                    "positive_semantics": "higher_image_source_attribution",
                }
            )

            train_X = matrix[train_indices]
            test_X = matrix[test_indices]
            train_y = np.asarray(
                [record["baseline"]["soft_score"] for record in train_records],
                dtype=np.float64,
            )
            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("regressor", Ridge(alpha=1.0, solver="lsqr")),
                ]
            )
            pipeline.fit(train_X, train_y)
            predicted = np.asarray(pipeline.predict(test_X), dtype=np.float64)
            expected_predictions = np.asarray(
                [oof[(str(position), int(layer), record["case_id"])] for record in test_records],
                dtype=np.float64,
            )
            absolute_errors = np.abs(predicted - expected_predictions)
            max_abs_error = float(np.max(absolute_errors))
            mean_abs_error = float(np.mean(absolute_errors))
            correlation = float(np.corrcoef(predicted, expected_predictions)[0, 1])
            # Ridge(solver="lsqr") is iterative and the original run did not
            # persist coefficients.  Re-fitting with the recorded sklearn
            # version can differ slightly with BLAS/OpenMP execution while
            # retaining effectively identical predictions.  Require all three
            # strict behavioral checks instead of pretending bitwise replay is
            # possible without the original weights.
            if not (
                math.isfinite(max_abs_error)
                and math.isfinite(mean_abs_error)
                and math.isfinite(correlation)
                and max_abs_error <= 0.02
                and mean_abs_error <= 0.003
                and correlation >= 0.9995
            ):
                raise ValueError(
                    "Retrained Ridge probe does not reproduce existing OOF predictions: "
                    f"position={position}, layer={layer}, max_abs_error={max_abs_error}, "
                    f"mean_abs_error={mean_abs_error}, correlation={correlation}"
                )
            scaler = pipeline.named_steps["scaler"]
            regressor = pipeline.named_steps["regressor"]
            coef = np.asarray(regressor.coef_, dtype=np.float64).reshape(-1)
            scale = np.asarray(scaler.scale_, dtype=np.float64).reshape(-1)
            if coef.shape != (matrix.shape[1],) or scale.shape != coef.shape:
                raise ValueError("Unexpected Ridge/StandardScaler coefficient shape")
            probe_raw = coef / scale
            train_norms = np.linalg.norm(train_X.astype(np.float64, copy=False), axis=1)
            probe_vector, raw_norm, hidden_mean, target_norm = _normalize_direction(
                probe_raw, train_norms
            )
            probe_file = f"probe_weight__{position}__layer_{int(layer):02d}.npz"
            _atomic_save_npz(
                vector_dir / probe_file,
                vector=probe_vector,
                raw_direction=probe_raw,
                standardized_coefficient=coef,
                scaler_scale=scale,
                scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
            )
            entries.append(
                {
                    "method": "probe_weight",
                    "position": str(position),
                    "layer": int(layer),
                    "file": probe_file,
                    "raw_direction_norm": raw_norm,
                    "hidden_norm_mean": hidden_mean,
                    "target_norm": target_norm,
                    "vector_norm": float(np.linalg.norm(probe_vector.astype(np.float64))),
                    "train_sample_count": len(train_records),
                    "test_sample_count": len(test_records),
                    "oof_reproduction_max_abs_error": max_abs_error,
                    "oof_reproduction_mean_abs_error": mean_abs_error,
                    "oof_reproduction_pearson": correlation,
                    "oof_reproduction_thresholds": {
                        "max_abs_error": 0.02,
                        "mean_abs_error": 0.003,
                        "pearson": 0.9995,
                    },
                    "probe": "StandardScaler + Ridge(alpha=1.0, solver=lsqr)",
                    "raw_space_direction": "regressor.coef_ / scaler.scale_",
                    "positive_semantics": "higher_image_source_attribution",
                }
            )
            del matrix, train_X, test_X

    index_payload = {
        **expected,
        "hidden_state_definition": HIDDEN_STATE_DEFINITION,
        "vector_count": len(entries),
        "hidden_state_shard_load_count": loader.shard_load_count,
        "vectors": entries,
    }
    atomic_write_json(index_path, index_payload)
    return SteeringVectorRepository(vector_dir)


def direction_cosines(repository: SteeringVectorRepository) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for position in repository.index["positions"]:
        for layer in repository.index["layers"]:
            mean = repository.get("mean_difference", position, int(layer)).raw_direction
            probe = repository.get("probe_weight", position, int(layer)).raw_direction
            denominator = float(np.linalg.norm(mean) * np.linalg.norm(probe))
            cosine = float(np.dot(mean, probe) / denominator)
            cells.append({"position": position, "layer": int(layer), "cosine": cosine})
    return cells
