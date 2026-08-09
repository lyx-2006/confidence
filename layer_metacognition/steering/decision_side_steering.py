"""OOF Decision-Side activation steering for V4 joint answer + attribution."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from confidence_test.answer_metrics import normalize_answer
from confidence_test.dataset_utils import ConditionInput, EvaluationCase, load_evaluation_cases
from confidence_test.inference_extension import ASSISTANT_ANSWER_PREFILL
from confidence_test.joint_answer_source_extension import JointAnswerSourceGenerator
from confidence_test.source_attribution_analyzer import SourceAttributionAnalyzer
from confidence_test.source_attribution_schema import ASSISTANT_SOURCE_ATTRIBUTION_PREFILL
from confidence_test.source_attribution_variants import get_source_prompt_variant
from layer_metacognition.hidden_state_store import (
    append_jsonl,
    atomic_write_json,
)
from layer_metacognition.model_adapter import (
    AdditiveActivationHook,
    LanguageModules,
    ReinjectingActivationHook,
    run_logits_forward,
)
from layer_metacognition.token_positions import (
    locate_marker_in_assistant,
    locate_token_after_field,
    locate_text_clue_save_positions,
)
from layer_metacognition.token_spans import build_rendered_alignment


STEERING_POSITIONS = ("ptnl", "ac", "panl")
STEERING_SCALES = ("probe_logit", "unit")
INJECTION_SITES = ("block_output", "block_input")
INTERVENTION_MODES = ("single", "reinject")
DECISION_MAPPING = {"follows_text": 0, "follows_image": 1}
VERSION_SETTING = "v4_to_v4"
CONFLICT_CONDITIONS = ("conflict_easy", "conflict_hard")
MANIPULATION_ABS_TOLERANCE = 0.1
MANIPULATION_REL_TOLERANCE = 0.1
BF16_MANIPULATION_ABS_TOLERANCE = 0.25
BF16_MANIPULATION_REL_TOLERANCE = 0.25
BASELINE_SOFT_TOLERANCE = 0.05
PROBABILITY_EPSILON = 1e-12
DEFAULT_CASES_PER_DECISION_SIDE = 150


class SteeringInvariantError(RuntimeError):
    """An error that invalidates the intervention rather than one case output."""


@dataclass(frozen=True)
class DecisionDirection:
    file: str
    fold: int
    layer: int
    position: str
    d_raw: np.ndarray
    d_K: np.ndarray
    raw_intercept: float


@dataclass(frozen=True)
class SteeringCase:
    manifest: dict[str, Any]
    evaluation: EvaluationCase
    baseline: dict[str, Any]
    fold: int


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configuration_fingerprint(configuration: dict[str, Any]) -> str:
    payload = json.dumps(
        configuration,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_grid(
    layers: Sequence[int],
    positions: Sequence[str],
    alphas: Sequence[float],
) -> tuple[list[int], list[str], list[float]]:
    normalized_layers = [int(value) for value in layers]
    normalized_positions = [str(value).casefold() for value in positions]
    normalized_alphas = [float(value) for value in alphas]
    if not normalized_layers or len(normalized_layers) != len(set(normalized_layers)):
        raise ValueError("--layers must contain distinct layer indices")
    if any(layer < 0 for layer in normalized_layers):
        raise ValueError("--layers must be non-negative")
    if not normalized_positions or len(normalized_positions) != len(
        set(normalized_positions)
    ):
        raise ValueError("--positions must contain distinct values")
    invalid_positions = [
        value for value in normalized_positions if value not in STEERING_POSITIONS
    ]
    if invalid_positions:
        raise ValueError(
            "--positions only supports ptnl/ac/panl; invalid: "
            + ", ".join(invalid_positions)
        )
    if not normalized_alphas or not all(math.isfinite(value) for value in normalized_alphas):
        raise ValueError("--alphas must contain finite values")
    if len(normalized_alphas) != len(set(normalized_alphas)):
        raise ValueError("--alphas must contain distinct values")
    if sum(value == 0.0 for value in normalized_alphas) != 1:
        raise ValueError("--alphas must contain alpha=0 exactly once")
    return normalized_layers, normalized_positions, normalized_alphas


def build_steering_vector(
    direction: DecisionDirection,
    alpha: float,
    steering_scale: str,
) -> np.ndarray:
    if steering_scale == "probe_logit":
        squared_norm = float(np.dot(direction.d_raw, direction.d_raw))
        if not math.isfinite(squared_norm) or squared_norm <= 0:
            raise ValueError("d_raw has a non-positive squared norm")
        vector = float(alpha) * direction.d_raw / squared_norm
    elif steering_scale == "unit":
        vector = float(alpha) * direction.d_K
    else:
        raise ValueError(f"Unknown steering scale: {steering_scale}")
    if not np.isfinite(vector).all():
        raise ValueError("Computed steering vector contains non-finite values")
    return np.asarray(vector, dtype=np.float64)


def intervention_key(
    case_id: str,
    layer: int,
    position: str,
    alpha: float,
    steering_scale: str,
    injection_site: str = "block_output",
    intervention_mode: str = "single",
) -> str:
    normalized_alpha = 0.0 if float(alpha) == 0.0 else float(alpha)
    parts = [
            str(case_id),
            f"layer={int(layer)}",
            f"position={position}",
            f"alpha={format(normalized_alpha, '.17g')}",
            f"scale={steering_scale}",
    ]
    if injection_site != "block_output":
        parts.append(f"site={injection_site}")
    if intervention_mode != "single":
        parts.append(f"mode={intervention_mode}")
    return "|".join(parts)


class DirectionRepository:
    """Strict reader for Stage 1 item-split OOF direction artifacts."""

    def __init__(self, probe_run_dir: str | Path) -> None:
        self.probe_run_dir = Path(probe_run_dir).resolve()
        self.run_config_path = self.probe_run_dir / "run_config.json"
        self.split_path = self.probe_run_dir / "split_assignments.json"
        self.direction_dir = self.probe_run_dir / "decision_directions"
        self.index_path = self.direction_dir / "index.json"
        for path in (self.run_config_path, self.split_path, self.index_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required Stage 1 artifact is missing: {path}")

        self.run_config = _read_json(self.run_config_path)
        self.split_assignments = _read_json(self.split_path)
        self.direction_index = _read_json(self.index_path)
        self._validate_metadata()
        self.item_to_fold = {
            str(item_id): int(fold)
            for item_id, fold in self.split_assignments["item_to_fold"].items()
        }
        entries = self.direction_index.get("directions")
        assert isinstance(entries, list)
        self._entries: dict[tuple[int, int, str, str], dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Direction index contains a non-object entry")
            key = (
                int(entry.get("fold")),
                int(entry.get("layer")),
                str(entry.get("position")),
                str(entry.get("version_setting")),
            )
            if key in self._entries:
                raise ValueError(f"Duplicate direction index entry: {key}")
            self._entries[key] = entry
        self._cache: dict[tuple[int, int, str, str], DecisionDirection] = {}

    def _validate_metadata(self) -> None:
        if self.run_config.get("status") != "complete":
            raise ValueError("Stage 1 run_config status is not complete")
        if self.run_config.get("split_mode") != "item":
            raise ValueError("Steering requires Stage 1 split_mode=item")
        if self.run_config.get("decision_side_label_mapping") != DECISION_MAPPING:
            raise ValueError("Stage 1 run_config has the wrong Decision-Side mapping")
        if self.direction_index.get("split_mode") != "item":
            raise ValueError("Direction index is not an item-split index")
        if self.direction_index.get("class_mapping") != DECISION_MAPPING:
            raise ValueError("Direction index has the wrong Decision-Side mapping")
        entries = self.direction_index.get("directions")
        if not isinstance(entries, list) or not entries:
            raise ValueError("Direction index has no directions")
        if int(self.direction_index.get("direction_count", -1)) != len(entries):
            raise ValueError("Direction index count does not match its entries")
        if self.split_assignments.get("group_key") != "item_id":
            raise ValueError("Stage 1 split assignments are not grouped by item_id")
        if not isinstance(self.split_assignments.get("item_to_fold"), dict):
            raise ValueError("Stage 1 split assignments have no item_to_fold mapping")

    @property
    def manifest_path(self) -> Path:
        raw = self.run_config.get("manifest_path")
        if not isinstance(raw, str) or not raw:
            raise ValueError("Stage 1 run_config has no manifest_path")
        path = Path(raw).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Stage 1 manifest does not exist: {path}")
        return path

    def fold_for_item(self, item_id: str) -> int:
        try:
            return self.item_to_fold[str(item_id)]
        except KeyError as exc:
            raise KeyError(f"No OOF fold assignment for item {item_id!r}") from exc

    def validate_requested_grid(
        self,
        layers: Sequence[int],
        positions: Sequence[str],
    ) -> None:
        missing = [
            (fold, int(layer), str(position), VERSION_SETTING)
            for fold in range(int(self.split_assignments["n_splits"]))
            for position in positions
            for layer in layers
            if (fold, int(layer), str(position), VERSION_SETTING) not in self._entries
        ]
        if missing:
            raise ValueError(f"Direction index is missing requested cells: {missing[:10]}")
        for fold, layer, position, version in [
            (fold, int(layer), str(position), VERSION_SETTING)
            for fold in range(int(self.split_assignments["n_splits"]))
            for position in positions
            for layer in layers
        ]:
            self.get(fold, layer, position, version)

    def available_layers(self, position: str) -> list[int]:
        folds = range(int(self.split_assignments["n_splits"]))
        layers = sorted(
            {
                layer
                for fold, layer, entry_position, version in self._entries
                if entry_position == str(position) and version == VERSION_SETTING
            }
        )
        return [
            layer
            for layer in layers
            if all(
                (fold, layer, str(position), VERSION_SETTING) in self._entries
                for fold in folds
            )
        ]

    def trajectory_layers(self, injection_layer: int, position: str) -> list[int]:
        layers = [
            layer
            for layer in self.available_layers(position)
            if layer >= int(injection_layer)
        ]
        if int(injection_layer) not in layers:
            raise ValueError(
                f"No OOF trajectory probe at injection layer {injection_layer} "
                f"for {position}"
            )
        return layers

    def get(
        self,
        fold: int,
        layer: int,
        position: str,
        version_setting: str = VERSION_SETTING,
    ) -> DecisionDirection:
        key = (int(fold), int(layer), str(position), str(version_setting))
        if key in self._cache:
            return self._cache[key]
        try:
            entry = self._entries[key]
        except KeyError as exc:
            raise KeyError(f"No direction for {key}") from exc
        if entry.get("class0") != "follows_text" or entry.get("class1") != "follows_image":
            raise ValueError(f"Direction {key} has reversed class semantics")
        if entry.get("positive_direction") != "+d_K -> follows_image":
            raise ValueError(f"Direction {key} does not declare image-positive semantics")
        raw_file = entry.get("file")
        if not isinstance(raw_file, str) or not raw_file:
            raise ValueError(f"Direction {key} has no file")
        path = (self.direction_dir / raw_file).resolve()
        if path.parent != self.direction_dir.resolve():
            raise ValueError(f"Direction file escapes its directory: {raw_file}")
        if not path.is_file():
            raise FileNotFoundError(f"Direction file does not exist: {path}")
        with np.load(path, allow_pickle=False) as payload:
            required = {
                "scaler_mean",
                "scaler_scale",
                "weight",
                "intercept",
                "d_raw",
                "d_K",
                "raw_intercept",
            }
            missing = required.difference(payload.files)
            if missing:
                raise ValueError(f"Direction {path} is missing arrays: {sorted(missing)}")
            for name in required:
                if not np.isfinite(payload[name]).all():
                    raise ValueError(f"Direction {path} contains non-finite {name}")
            d_raw = np.asarray(payload["d_raw"], dtype=np.float64).reshape(-1)
            d_K = np.asarray(payload["d_K"], dtype=np.float64).reshape(-1)
            raw_intercept_values = np.asarray(
                payload["raw_intercept"], dtype=np.float64
            ).reshape(-1)
        if d_raw.size == 0 or d_raw.shape != d_K.shape:
            raise ValueError(f"Direction {path} has incompatible d_raw/d_K shapes")
        norm = float(np.linalg.norm(d_raw))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError(f"Direction {path} has a zero d_raw norm")
        if not np.allclose(d_K, d_raw / norm, rtol=1e-6, atol=1e-8):
            raise ValueError(f"Direction {path} has an invalid d_K")
        if raw_intercept_values.size != 1:
            raise ValueError(f"Direction {path} has a non-scalar raw_intercept")
        direction = DecisionDirection(
            file=raw_file,
            fold=int(fold),
            layer=int(layer),
            position=str(position),
            d_raw=d_raw,
            d_K=d_K,
            raw_intercept=float(raw_intercept_values[0]),
        )
        self._cache[key] = direction
        return direction


class BaselineHiddenStateRepository:
    """Read baseline decoder outputs saved by the original experiment."""

    def __init__(self, experiment_dir: str | Path) -> None:
        self.experiment_dir = Path(experiment_dir).resolve()
        self.index_path = self.experiment_dir / "hidden_states" / "index.json"
        if not self.index_path.is_file():
            raise FileNotFoundError(
                f"Baseline hidden-state index is missing: {self.index_path}"
            )
        self.index = _read_json(self.index_path)
        self._last_path: Path | None = None
        self._last_payload: dict[str, Any] | None = None

    def _payload(self, path: Path) -> dict[str, Any]:
        if self._last_path == path and self._last_payload is not None:
            return self._last_payload
        value = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(value, dict):
            raise ValueError(f"Hidden-state shard is not a mapping: {path}")
        self._last_path = path
        self._last_payload = value
        return value

    def get(
        self,
        manifest: dict[str, Any],
        layer: int,
        position: str,
    ) -> np.ndarray:
        case_id = str(manifest["case_id"])
        reference = manifest.get("hidden_state_reference")
        if not isinstance(reference, dict):
            raise ValueError(f"Manifest case has no hidden-state reference: {case_id}")
        indexed_cases = self.index.get("cases")
        if not isinstance(indexed_cases, dict) or case_id not in indexed_cases:
            raise ValueError(f"Hidden-state index has no case: {case_id}")
        indexed_reference = indexed_cases[case_id]
        if not isinstance(indexed_reference, dict) or any(
            indexed_reference.get(key) != reference.get(key)
            for key in ("shard_path", "offset")
        ):
            raise ValueError(
                f"Manifest/index hidden-state reference differs for {case_id}"
            )
        raw_path = reference.get("shard_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"Manifest hidden-state shard is missing: {case_id}")
        path = (self.experiment_dir / raw_path).resolve()
        if not path.is_relative_to(self.experiment_dir):
            raise ValueError(f"Hidden-state shard escapes experiment directory: {raw_path}")
        if not path.is_file():
            raise FileNotFoundError(f"Hidden-state shard is missing: {path}")
        payload = self._payload(path)
        case_ids = [str(value) for value in payload.get("case_ids", [])]
        offset = int(reference.get("offset", -1))
        if offset < 0 or offset >= len(case_ids) or case_ids[offset] != case_id:
            raise ValueError(
                f"Hidden-state offset does not resolve to {case_id}: {path}:{offset}"
            )
        layer_indices = [int(value) for value in payload.get("layer_indices", [])]
        position_names = [str(value) for value in payload.get("position_names", [])]
        if int(layer) not in layer_indices:
            raise ValueError(f"Baseline hidden state omits layer {layer}: {case_id}")
        if str(position) not in position_names:
            raise ValueError(
                f"Baseline hidden state omits position {position}: {case_id}"
            )
        hidden = payload.get("hidden_states")
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 4:
            raise ValueError(f"Invalid hidden-state tensor in {path}")
        vector = hidden[
            offset,
            layer_indices.index(int(layer)),
            position_names.index(str(position)),
            :,
        ].detach().float().numpy().astype(np.float64, copy=False)
        if not np.isfinite(vector).all():
            raise ValueError(
                f"Baseline hidden state contains non-finite values: {case_id} "
                f"layer={layer} position={position}"
            )
        return vector

    def validate_cases(
        self,
        cases: Sequence[SteeringCase],
        repository: DirectionRepository,
        injection_layers: Sequence[int],
        positions: Sequence[str],
    ) -> dict[str, list[int]]:
        trajectory_layers_by_position: dict[str, list[int]] = {}
        for position in positions:
            readout_layers = sorted(
                {
                    readout_layer
                    for injection_layer in injection_layers
                    for readout_layer in repository.trajectory_layers(
                        injection_layer, position
                    )
                }
            )
            repository.validate_requested_grid(readout_layers, [position])
            trajectory_layers_by_position[position] = readout_layers
        for case in cases:
            for position, readout_layers in trajectory_layers_by_position.items():
                for layer in readout_layers:
                    self.get(case.manifest, layer, position)
        return trajectory_layers_by_position


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Manifest row {line_number} is not an object")
            if not (
                record.get("eligible_decision_side_probe") is True
                and record.get("version") == "v4"
                and record.get("condition") in CONFLICT_CONDITIONS
                and str(record.get("case_id", "")).endswith("__v4__joint")
            ):
                continue
            case_id = str(record["case_id"])
            if case_id in seen:
                raise ValueError(f"Duplicate eligible manifest case_id: {case_id}")
            if record.get("decision_side") not in DECISION_MAPPING:
                raise ValueError(f"Eligible case has invalid decision_side: {case_id}")
            seen.add(case_id)
            records.append(record)
    if not records:
        raise ValueError("Manifest has no eligible V4 Decision-Side cases")
    return records


def _load_selected_baselines(
    path: Path,
    selected_case_ids: set[str],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            case_id = str(record.get("case_id", ""))
            if case_id not in selected_case_ids:
                continue
            if case_id in output:
                raise ValueError(f"Duplicate baseline case_id at line {line_number}: {case_id}")
            if record.get("status") != "completed":
                raise ValueError(f"Baseline case is not completed: {case_id}")
            generated = record.get("generated")
            if not isinstance(generated, dict):
                raise ValueError(f"Baseline case has no generated payload: {case_id}")
            source = generated.get("source_attribution")
            if not isinstance(source, dict):
                raise ValueError(f"Baseline case has no Source Attribution: {case_id}")
            answer_result = generated.get("current_answer_result")
            if not isinstance(answer_result, dict):
                raise ValueError(f"Baseline case has no answer metrics: {case_id}")
            probabilities = answer_result.get("answer_class_probabilities")
            if not isinstance(probabilities, dict):
                raise ValueError(
                    f"Baseline case has no answer class probabilities: {case_id}"
                )
            output[case_id] = {
                "generated": {
                    "current_answer": generated.get("current_answer"),
                    "current_answer_result": {
                        "raw_output": answer_result.get("raw_output"),
                        "source_label": answer_result.get("source_label"),
                        "answer_class_probabilities": probabilities,
                    },
                    "source_attribution": {
                        "hard_label": source.get("hard_label"),
                        "parsed_label": source.get("parsed_label"),
                        "soft_image_score": source.get("soft_image_score"),
                        "class_probabilities": source.get("class_probabilities"),
                        "source_entropy": source.get("source_entropy"),
                    },
                }
            }
    missing = selected_case_ids.difference(output)
    if missing:
        raise ValueError(f"Baseline results are missing selected cases: {sorted(missing)[:10]}")
    return output


def _rebase_case_images(
    cases: Iterable[EvaluationCase],
    image_root: Path | None,
) -> list[EvaluationCase]:
    if image_root is None:
        return list(cases)
    output: list[EvaluationCase] = []
    for case in cases:
        conditions: dict[str, ConditionInput] = {}
        for name, condition in case.conditions.items():
            raw = condition.relative_image_path
            if raw is None:
                conditions[name] = condition
                continue
            raw_path = Path(raw)
            resolved = (
                raw_path.resolve()
                if raw_path.is_absolute()
                else (image_root / raw_path).resolve()
            )
            error = None
            if not resolved.is_file():
                error = {
                    "type": "FileNotFoundError",
                    "message": f"Image does not exist under --image-root: {resolved}",
                }
            conditions[name] = replace(
                condition,
                resolved_image_path=str(resolved),
                error=error,
            )
        output.append(replace(case, conditions=conditions))
    return output


def _case_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    raw_item = str(record["item_id"])
    item_key: tuple[int, Any] = (
        (0, int(raw_item)) if raw_item.isdigit() else (1, raw_item)
    )
    return (*item_key, int(record["prior_index"]), str(record["condition"]))


def select_balanced_decision_side_cases(
    records: Sequence[dict[str, Any]],
    cases_per_decision_side: int,
) -> list[dict[str, Any]]:
    if cases_per_decision_side <= 0:
        raise ValueError("--cases-per-decision-side must be positive")
    ordered = sorted(records, key=_case_sort_key)
    balanced: list[dict[str, Any]] = []
    for decision_side in DECISION_MAPPING:
        side_records = [
            row for row in ordered if row["decision_side"] == decision_side
        ]
        if len(side_records) < cases_per_decision_side:
            raise ValueError(
                f"Requested {cases_per_decision_side} {decision_side} cases, "
                f"but only {len(side_records)} remain after filters"
            )
        balanced.extend(side_records[:cases_per_decision_side])
    return sorted(balanced, key=_case_sort_key)


def prepare_cases(
    *,
    repository: DirectionRepository,
    experiment_dir: Path,
    dataset: Path,
    image_root: Path | None,
    conditions: Sequence[str],
    item_ids: set[str] | None,
    prior_indices: set[int] | None,
    cases_per_decision_side: int | None,
    max_cases: int | None,
    max_baseline_abs_answer_margin: float | None = None,
    fallback_null_path: Path,
) -> tuple[list[SteeringCase], dict[str, Any]]:
    manifest = _load_manifest(repository.manifest_path)
    selected = [row for row in manifest if row["condition"] in set(conditions)]
    if item_ids is not None:
        selected = [row for row in selected if str(row["item_id"]) in item_ids]
    if prior_indices is not None:
        selected = [row for row in selected if int(row["prior_index"]) in prior_indices]
    baseline_path = experiment_dir / "results.jsonl"
    baselines = _load_selected_baselines(
        baseline_path,
        {str(row["case_id"]) for row in selected},
    )
    if max_baseline_abs_answer_margin is not None:
        if not math.isfinite(max_baseline_abs_answer_margin) or (
            max_baseline_abs_answer_margin <= 0.0
        ):
            raise ValueError("--max-baseline-abs-answer-margin must be positive")

        def inside_margin(row: dict[str, Any]) -> bool:
            probabilities = baselines[str(row["case_id"])]["generated"][
                "current_answer_result"
            ]["answer_class_probabilities"]
            normalized = {
                normalize_answer(key): float(value)
                for key, value in probabilities.items()
            }
            p_text = normalized[str(row["text_only_answer"])]
            p_image = normalized[str(row["image_only_answer"])]
            margin = math.log(p_image + PROBABILITY_EPSILON) - math.log(
                p_text + PROBABILITY_EPSILON
            )
            return abs(margin) < max_baseline_abs_answer_margin

        selected = [row for row in selected if inside_margin(row)]
    selected.sort(key=_case_sort_key)
    if cases_per_decision_side is not None:
        selected = select_balanced_decision_side_cases(
            selected, cases_per_decision_side
        )
    if max_cases is not None:
        selected = selected[:max_cases]
    if not selected:
        raise ValueError("No eligible cases remain after Steering filters")

    evaluation_cases, dataset_metadata = load_evaluation_cases(
        dataset,
        fallback_null_path=fallback_null_path,
    )
    evaluation_cases = _rebase_case_images(evaluation_cases, image_root)
    by_key = {(case.item_id, case.prior_index): case for case in evaluation_cases}
    baselines = {
        str(row["case_id"]): baselines[str(row["case_id"])] for row in selected
    }
    output: list[SteeringCase] = []
    for row in selected:
        case_id = str(row["case_id"])
        key = (str(row["item_id"]), int(row["prior_index"]))
        if key not in by_key:
            raise ValueError(f"Dataset has no case for manifest row {case_id}")
        case = by_key[key]
        condition = str(row["condition"])
        condition_input = case.conditions[condition]
        if condition_input.error or not condition_input.resolved_image_path:
            raise ValueError(
                f"Selected case has no usable {condition} image: {case_id}: "
                f"{condition_input.error}"
            )
        if [normalize_answer(value) for value in case.answer_classes] != list(
            row["answer_classes"]
        ):
            raise ValueError(f"Dataset/manifest answer classes differ for {case_id}")
        output.append(
            SteeringCase(
                manifest=row,
                evaluation=case,
                baseline=baselines[case_id],
                fold=repository.fold_for_item(str(row["item_id"])),
            )
        )
    return output, dataset_metadata


def locate_steering_position(
    *,
    tokenizer: Any,
    rendered: str,
    inputs: Any,
    assistant_text: str,
    text_clue: str,
    position: str,
    answer: str | None = None,
    assistant_occurrence: str = "unique",
) -> tuple[int, dict[str, Any]]:
    processed_ids = [int(value) for value in inputs.input_ids[0].tolist()]
    alignment = build_rendered_alignment(
        tokenizer,
        rendered,
        inputs.input_ids,
        inputs.attention_mask,
    )
    if position == "ac":
        detail = locate_marker_in_assistant(
            tokenizer,
            alignment.rendered_ids,
            assistant_text,
            "**Answer**:",
            name="ac",
            position_map=alignment.rendered_to_processed,
            processed_ids=processed_ids,
            assistant_occurrence=assistant_occurrence,
        )
    elif position == "ptnl":
        detail = locate_text_clue_save_positions(
            tokenizer,
            alignment,
            text_clue,
        )["ptnl"]
    elif position == "panl":
        if answer is None:
            raise ValueError("PANL Steering requires the generated answer")
        detail = locate_token_after_field(
            tokenizer,
            alignment.rendered_ids,
            "**Answer**:",
            answer,
            name="panl",
            position_map=alignment.rendered_to_processed,
            processed_ids=processed_ids,
        )
    else:
        raise ValueError(f"Unsupported Steering position: {position}")
    return int(detail["position"]), detail


def _locate_sac(
    *,
    tokenizer: Any,
    rendered: str,
    inputs: Any,
    assistant_text: str,
) -> tuple[int, dict[str, Any]]:
    processed_ids = [int(value) for value in inputs.input_ids[0].tolist()]
    alignment = build_rendered_alignment(
        tokenizer,
        rendered,
        inputs.input_ids,
        inputs.attention_mask,
    )
    detail = locate_marker_in_assistant(
        tokenizer,
        alignment.rendered_ids,
        assistant_text,
        ASSISTANT_SOURCE_ATTRIBUTION_PREFILL,
        name="sac",
        position_map=alignment.rendered_to_processed,
        processed_ids=processed_ids,
    )
    return int(detail["position"]), detail


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def teacher_forced_assistant_text(
    steered_answer: str,
    parsed_source_label: str,
) -> str:
    """Use the generated steered answer in the joint teacher-forced wire format."""
    return (
        f"{ASSISTANT_ANSWER_PREFILL} {steered_answer}\n"
        f"{ASSISTANT_SOURCE_ATTRIBUTION_PREFILL}{parsed_source_label}"
    )


def manipulation_diagnostics(
    hook: AdditiveActivationHook,
    direction: DecisionDirection,
    *,
    alpha: float,
    steering_scale: str,
) -> dict[str, Any]:
    base = hook.diagnostics()
    assert hook.h_before is not None and hook.h_after is not None
    before = hook.h_before.numpy().astype(np.float64, copy=False)
    after = hook.h_after.numpy().astype(np.float64, copy=False)
    logit_before = float(np.dot(direction.d_raw, before) + direction.raw_intercept)
    logit_after = float(np.dot(direction.d_raw, after) + direction.raw_intercept)
    actual_delta = logit_after - logit_before
    expected_delta = (
        float(alpha)
        if steering_scale == "probe_logit"
        else float(alpha) * float(np.linalg.norm(direction.d_raw))
    )
    requested_delta = float(
        np.dot(
            direction.d_raw,
            hook.steering_vector.detach().cpu().numpy().astype(np.float64, copy=False),
        )
    )
    requested_tolerance = max(1e-8, 1e-8 * abs(expected_delta))
    requested_passed = abs(requested_delta - expected_delta) <= requested_tolerance
    if hook.activation_dtype == "bfloat16":
        tolerance = max(
            BF16_MANIPULATION_ABS_TOLERANCE,
            BF16_MANIPULATION_REL_TOLERANCE * abs(expected_delta),
        )
    else:
        tolerance = max(
            MANIPULATION_ABS_TOLERANCE,
            MANIPULATION_REL_TOLERANCE * abs(expected_delta),
        )
    realized_passed = abs(actual_delta - expected_delta) <= tolerance
    direction_passed = (
        expected_delta == 0.0 or actual_delta * expected_delta > 0.0
    )
    passed = requested_passed and realized_passed and direction_passed
    if float(alpha) == 0.0:
        passed = passed and base["injection_l2"] == 0.0 and actual_delta == 0.0
    return {
        **base,
        "decision_logit_before": logit_before,
        "decision_logit_after": logit_after,
        "decision_logit_delta": actual_delta,
        "requested_decision_logit_delta": requested_delta,
        "expected_decision_logit_delta": expected_delta,
        "requested_decision_logit_delta_tolerance": requested_tolerance,
        "decision_logit_delta_tolerance": tolerance,
        "realized_quantization_error": actual_delta - requested_delta,
        "requested_delta_passed": bool(requested_passed),
        "realized_delta_passed": bool(realized_passed),
        "direction_passed": bool(direction_passed),
        "K_before": _sigmoid(logit_before),
        "K_after": _sigmoid(logit_after),
        "passed": bool(passed),
    }


def _direction_readout(
    direction: DecisionDirection,
    hidden: np.ndarray | torch.Tensor,
) -> tuple[float, float]:
    if isinstance(hidden, torch.Tensor):
        vector = hidden.detach().cpu().numpy().astype(np.float64, copy=False)
    else:
        vector = np.asarray(hidden, dtype=np.float64)
    vector = vector.reshape(-1)
    if vector.shape != direction.d_raw.shape or not np.isfinite(vector).all():
        raise ValueError(
            f"Hidden vector is incompatible with direction layer={direction.layer} "
            f"position={direction.position}: {vector.shape}"
        )
    logit = float(np.dot(direction.d_raw, vector) + direction.raw_intercept)
    return logit, _sigmoid(logit)


def _hidden_delta_diagnostics(
    *,
    baseline_hidden: np.ndarray | torch.Tensor,
    steered_hidden: np.ndarray | torch.Tensor,
    direction: DecisionDirection,
) -> dict[str, float | None]:
    """Describe how much of a downstream hidden-state change remains on d_K."""
    baseline = np.asarray(
        baseline_hidden.detach().cpu().numpy()
        if isinstance(baseline_hidden, torch.Tensor)
        else baseline_hidden,
        dtype=np.float64,
    ).reshape(-1)
    steered = np.asarray(
        steered_hidden.detach().cpu().numpy()
        if isinstance(steered_hidden, torch.Tensor)
        else steered_hidden,
        dtype=np.float64,
    ).reshape(-1)
    if baseline.shape != direction.d_K.shape or steered.shape != direction.d_K.shape:
        raise ValueError(
            "Hidden delta is incompatible with Decision direction: "
            f"baseline={baseline.shape} steered={steered.shape} "
            f"direction={direction.d_K.shape}"
        )
    if not np.isfinite(baseline).all() or not np.isfinite(steered).all():
        raise ValueError("Hidden delta contains non-finite values")
    delta = steered - baseline
    delta_l2 = float(np.linalg.norm(delta))
    projection = float(np.dot(delta, direction.d_K))
    orthogonal_squared = max(0.0, delta_l2 * delta_l2 - projection * projection)
    cosine = projection / delta_l2 if delta_l2 > 0.0 else None
    return {
        "delta_hidden_l2": delta_l2,
        "delta_hidden_projection_on_d_K": projection,
        "delta_hidden_cosine_with_d_K": cosine,
        "delta_hidden_orthogonal_l2": math.sqrt(orthogonal_squared),
        "directional_energy_fraction": (
            projection * projection / (delta_l2 * delta_l2)
            if delta_l2 > 0.0
            else None
        ),
    }


def build_layer_trajectory(
    *,
    steering_case: SteeringCase,
    repository: DirectionRepository,
    baseline_hidden_states: BaselineHiddenStateRepository,
    hook: AdditiveActivationHook | ReinjectingActivationHook | None,
    injection_layer: int,
    position: str,
    teacher_forced_answer: str | None = None,
    intervention_mode: str = "single",
) -> list[dict[str, Any]]:
    readout_layers = repository.trajectory_layers(injection_layer, position)
    steered_hidden = hook.trajectory_hidden() if hook is not None else {}
    rows: list[dict[str, Any]] = []
    in_run_target_delta: float | None = None
    in_run_target_before: float | None = None
    if hook is not None:
        hook.validate_applied_once()
        assert hook.h_before is not None and hook.h_after is not None
        target_direction = repository.get(
            steering_case.fold,
            injection_layer,
            position,
        )
        in_run_target_before, _ = _direction_readout(
            target_direction, hook.h_before
        )
        in_run_target_after, _ = _direction_readout(target_direction, hook.h_after)
        in_run_target_delta = in_run_target_after - in_run_target_before
    for readout_layer in readout_layers:
        direction = repository.get(
            steering_case.fold,
            readout_layer,
            position,
        )
        baseline_hidden = baseline_hidden_states.get(
            steering_case.manifest,
            readout_layer,
            position,
        )
        baseline_logit, baseline_K = _direction_readout(direction, baseline_hidden)
        if hook is None:
            diagnostic_baseline = baseline_hidden
            diagnostic_steered = baseline_hidden
            steered_logit, steered_K = baseline_logit, baseline_K
        else:
            diagnostic_baseline = (
                hook.h_before
                if readout_layer == injection_layer
                and hook.injection_site == "block_output"
                else baseline_hidden
            )
            diagnostic_steered = steered_hidden[readout_layer]
            steered_logit, steered_K = _direction_readout(
                direction, steered_hidden[readout_layer]
            )
        hidden_delta = _hidden_delta_diagnostics(
            baseline_hidden=diagnostic_baseline,
            steered_hidden=diagnostic_steered,
            direction=direction,
        )
        baseline_answer = normalize_answer(
            steering_case.baseline["generated"]["current_answer"]
        )
        answer_context_matches = (
            True
            if teacher_forced_answer is None
            else normalize_answer(teacher_forced_answer) == baseline_answer
        )
        rows.append(
            {
                "injection_layer": int(injection_layer),
                "readout_layer": int(readout_layer),
                "layer_offset": int(readout_layer - injection_layer),
                "position": str(position),
                "direction_file": direction.file,
                "baseline_logit": baseline_logit,
                "steered_logit": steered_logit,
                "delta_logit": steered_logit - baseline_logit,
                "baseline_K": baseline_K,
                "steered_K": steered_K,
                "delta_K": steered_K - baseline_K,
                **hidden_delta,
                "delta_hidden_baseline_source": (
                    "same_forward_pre_injection"
                    if hook is not None
                    and readout_layer == injection_layer
                    and hook.injection_site == "block_output"
                    else "original_experiment_hidden_states"
                ),
                "injection_site": (
                    hook.injection_site if hook is not None else "baseline_reuse"
                ),
                "intervention_mode": (
                    hook.intervention_mode
                    if hook is not None and hasattr(hook, "intervention_mode")
                    else intervention_mode
                ),
                "trajectory_semantics": (
                    "cumulative_reinjection_response"
                    if (
                        getattr(hook, "intervention_mode", intervention_mode)
                        if hook is not None
                        else intervention_mode
                    )
                    == "reinject"
                    else "single_injection_propagation"
                ),
                "answer_context_matches_saved_baseline": answer_context_matches,
                "retention_fraction": None,
                "retention_denominator_logit_delta": in_run_target_delta,
                "saved_baseline_alignment_logit_error": (
                    in_run_target_before - baseline_logit
                    if readout_layer == injection_layer
                    and in_run_target_before is not None
                    and hook is not None
                    and hook.injection_site == "block_output"
                    else None
                ),
            }
        )
    if in_run_target_delta is not None and abs(in_run_target_delta) > 1e-12:
        for row in rows:
            row["retention_fraction"] = (
                float(row["delta_logit"]) / in_run_target_delta
            )
    return rows


def _base_record(
    steering_case: SteeringCase,
    direction: DecisionDirection,
    *,
    layer: int,
    position: str,
    alpha: float,
    steering_scale: str,
    injection_site: str = "block_output",
    intervention_mode: str = "single",
) -> dict[str, Any]:
    manifest = steering_case.manifest
    baseline_generated = steering_case.baseline["generated"]
    baseline_answer = baseline_generated.get("current_answer")
    return {
        "intervention_key": intervention_key(
            str(manifest["case_id"]),
            layer,
            position,
            alpha,
            steering_scale,
            injection_site,
            intervention_mode,
        ),
        "case_id": str(manifest["case_id"]),
        "item_id": str(manifest["item_id"]),
        "prior_index": int(manifest["prior_index"]),
        "condition": str(manifest["condition"]),
        "version": "v4",
        "baseline_decision_side": str(manifest["decision_side"]),
        "fold": int(steering_case.fold),
        "layer": int(layer),
        "position": str(position),
        "alpha": float(alpha),
        "steering_scale": steering_scale,
        "injection_site": injection_site,
        "intervention_mode": intervention_mode,
        "direction_file": direction.file,
        "baseline_answer": baseline_answer,
        "baseline_normalized_answer": normalize_answer(baseline_answer),
        "text_only_answer": str(manifest["text_only_answer"]),
        "image_only_answer": str(manifest["image_only_answer"]),
        "status": "running",
        "error": None,
    }


def _failed(record: dict[str, Any], stage: str, exc: Exception) -> dict[str, Any]:
    record["status"] = "failed"
    record["error"] = {
        "stage": stage,
        "type": type(exc).__name__,
        "message": str(exc),
    }
    return record


def _validate_manipulation(diagnostics: dict[str, Any], pass_name: str) -> None:
    if not diagnostics["passed"]:
        raise SteeringInvariantError(
            f"{pass_name} manipulation check failed: actual="
            f"{diagnostics['decision_logit_delta']}, expected="
            f"{diagnostics['expected_decision_logit_delta']}, tolerance="
            f"{diagnostics['decision_logit_delta_tolerance']}"
        )


def build_baseline_validation(
    *,
    answer_match: bool,
    hard_source_match: bool,
    generated_source_match: bool,
    soft_source_abs_error: float,
    generation_diagnostics: dict[str, Any],
    teacher_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    generation_zero_delta = bool(
        generation_diagnostics["injection_l2"] == 0.0
        and generation_diagnostics["decision_logit_delta"] == 0.0
    )
    teacher_forced_zero_delta = bool(
        teacher_diagnostics["injection_l2"] == 0.0
        and teacher_diagnostics["decision_logit_delta"] == 0.0
    )
    markers: list[str] = []
    if not answer_match:
        markers.append("answer_mismatch")
    if not hard_source_match:
        markers.append("hard_source_mismatch")
    if not generated_source_match:
        markers.append("generated_source_mismatch")
    if soft_source_abs_error > BASELINE_SOFT_TOLERANCE:
        markers.append("soft_source_gap_gt_0.05")
    if not generation_zero_delta:
        markers.append("generation_nonzero_delta_at_alpha_zero")
    if not teacher_forced_zero_delta:
        markers.append("teacher_forced_nonzero_delta_at_alpha_zero")
    return {
        "answer_match": bool(answer_match),
        "hard_source_match": bool(hard_source_match),
        "generated_source_match": bool(generated_source_match),
        "soft_source_abs_error": float(soft_source_abs_error),
        "soft_source_tolerance": BASELINE_SOFT_TOLERANCE,
        "soft_source_gap_exceeded": bool(
            soft_source_abs_error > BASELINE_SOFT_TOLERANCE
        ),
        "generation_zero_delta": generation_zero_delta,
        "teacher_forced_zero_delta": teacher_forced_zero_delta,
        "markers": markers,
        "marked": bool(markers),
        "passed": not markers,
        "used_as_paired_baseline": True,
    }


def build_reused_baseline_record(
    *,
    steering_case: SteeringCase,
    direction: DecisionDirection,
    layer: int,
    position: str,
    steering_scale: str,
    injection_site: str = "block_output",
    intervention_mode: str = "single",
    repository: DirectionRepository | None = None,
    baseline_hidden_states: BaselineHiddenStateRepository | None = None,
) -> dict[str, Any]:
    """Build alpha=0 directly from the completed pre-Steering result."""
    record = _base_record(
        steering_case,
        direction,
        layer=layer,
        position=position,
        alpha=0.0,
        steering_scale=steering_scale,
        injection_site=injection_site,
        intervention_mode=intervention_mode,
    )
    manifest = steering_case.manifest
    generated = steering_case.baseline["generated"]
    answer_result = generated["current_answer_result"]
    source = generated["source_attribution"]
    normalized_probabilities = {
        normalize_answer(key): float(value)
        for key, value in answer_result["answer_class_probabilities"].items()
    }
    text_answer = str(manifest["text_only_answer"])
    image_answer = str(manifest["image_only_answer"])
    if text_answer not in normalized_probabilities or image_answer not in normalized_probabilities:
        raise ValueError(
            f"Pre-Steering answer metrics omit a unimodal answer: {manifest['case_id']}"
        )
    p_text = normalized_probabilities[text_answer]
    p_image = normalized_probabilities[image_answer]
    denominator = p_image + p_text
    if denominator <= 0:
        raise ValueError(
            f"Pre-Steering unimodal probability denominator is zero: {manifest['case_id']}"
        )
    answer = str(generated["current_answer"])
    normalized_answer = normalize_answer(answer)
    if normalized_answer == image_answer:
        decision_side = "follows_image"
    elif normalized_answer == text_answer:
        decision_side = "follows_text"
    else:
        decision_side = "follows_neither"
    class_probabilities = source.get("class_probabilities")
    if not isinstance(class_probabilities, list):
        raise ValueError(
            f"Pre-Steering Source Attribution probabilities are missing: {manifest['case_id']}"
        )
    reused_diagnostics = {
        "executed": False,
        "source": "pre_steering_results",
        "reason": "alpha_zero_reuses_baseline_without_model_forward",
        "hook_call_count": 0,
        "steering_applied_count": 0,
        "injection_l2": 0.0,
        "decision_logit_delta": 0.0,
        "passed": True,
    }
    record.update(
        {
            "steered_answer": answer,
            "normalized_answer": normalized_answer,
            "steered_decision_side": decision_side,
            "P_text_answer": p_text,
            "P_image_answer": p_image,
            "pair_image_prob": p_image / denominator,
            "answer_margin": math.log(p_image + PROBABILITY_EPSILON)
            - math.log(p_text + PROBABILITY_EPSILON),
            "answer_changed": False,
            "answer_intervention_applicable": position != "panl",
            "generated_raw_output": str(answer_result.get("raw_output") or ""),
            "generated_source_label": str(
                answer_result.get("source_label")
                if answer_result.get("source_label") is not None
                else source["hard_label"]
            ),
            "generation_diagnostics": dict(reused_diagnostics),
            "generation_reinjection_diagnostics": [],
            "teacher_forced_diagnostics": dict(reused_diagnostics),
            "teacher_forced_reinjection_diagnostics": [],
            "generation_target_position": None,
            "teacher_forced_target_position": None,
            "sac_position": None,
            "decision_logit_before": None,
            "decision_logit_after": None,
            "K_before": None,
            "K_after": None,
            "injection_l2": 0.0,
            "SA_soft_image_score": float(source["soft_image_score"]),
            "SA_hard_label": str(source["hard_label"]),
            "SA_class_probabilities": [float(value) for value in class_probabilities],
            "SA_entropy": float(source["source_entropy"]),
            "baseline_validation": {
                "source": "pre_steering_results",
                "markers": [],
                "marked": False,
                "passed": True,
                "used_as_paired_baseline": True,
            },
            "baseline_validation_marked": False,
            "layer_trajectory": (
                build_layer_trajectory(
                    steering_case=steering_case,
                    repository=repository,
                    baseline_hidden_states=baseline_hidden_states,
                    hook=None,
                    injection_layer=layer,
                    position=position,
                    teacher_forced_answer=answer,
                    intervention_mode=intervention_mode,
                )
                if repository is not None and baseline_hidden_states is not None
                else []
            ),
            "status": "completed",
        }
    )
    if record["layer_trajectory"]:
        target = record["layer_trajectory"][0]
        record.update(
            {
                "decision_logit_before": target["baseline_logit"],
                "decision_logit_after": target["steered_logit"],
                "K_before": target["baseline_K"],
                "K_after": target["steered_K"],
            }
        )
    return record


def run_intervention(
    *,
    steering_case: SteeringCase,
    repository: DirectionRepository,
    joint_generator: JointAnswerSourceGenerator,
    source_analyzer: SourceAttributionAnalyzer,
    modules: LanguageModules,
    layer: int,
    position: str,
    alpha: float,
    steering_scale: str,
    injection_site: str = "block_output",
    intervention_mode: str = "single",
    max_answer_tokens: int,
    baseline_hidden_states: BaselineHiddenStateRepository | None = None,
) -> tuple[dict[str, Any], bool]:
    if injection_site not in INJECTION_SITES:
        raise ValueError(f"Unknown injection_site: {injection_site}")
    if intervention_mode not in INTERVENTION_MODES:
        raise ValueError(f"Unknown intervention_mode: {intervention_mode}")
    if intervention_mode == "reinject" and injection_site != "block_output":
        raise ValueError("reinject only supports injection_site=block_output")
    direction = repository.get(steering_case.fold, layer, position)
    if float(alpha) == 0.0:
        try:
            return (
                build_reused_baseline_record(
                    steering_case=steering_case,
                    direction=direction,
                    layer=layer,
                    position=position,
                    steering_scale=steering_scale,
                    injection_site=injection_site,
                    intervention_mode=intervention_mode,
                    repository=repository,
                    baseline_hidden_states=baseline_hidden_states,
                ),
                False,
            )
        except Exception as exc:
            record = _base_record(
                steering_case,
                direction,
                layer=layer,
                position=position,
                alpha=alpha,
                steering_scale=steering_scale,
                injection_site=injection_site,
                intervention_mode=intervention_mode,
            )
            return _failed(record, "baseline_reuse", exc), False
    record = _base_record(
        steering_case,
        direction,
        layer=layer,
        position=position,
        alpha=alpha,
        steering_scale=steering_scale,
        injection_site=injection_site,
        intervention_mode=intervention_mode,
    )
    stage = "initialization"
    fatal = False
    try:
        case = steering_case.evaluation
        manifest = steering_case.manifest
        condition_input = case.conditions[str(manifest["condition"])]
        image_path = str(condition_input.resolved_image_path)
        source_variant = get_source_prompt_variant("answer_basis_9")
        prompt = source_variant.v4_joint_prompt.format(
            question=case.question,
            text_clue=case.text_clue,
            source_classes=source_variant.class_text,
        )
        vector = build_steering_vector(direction, alpha, steering_scale)
        vector_tensor = torch.from_numpy(vector)
        trajectory_layers = repository.trajectory_layers(layer, position)
        reinjection_vectors = {
            readout_layer: torch.from_numpy(
                build_steering_vector(
                    repository.get(steering_case.fold, readout_layer, position),
                    alpha,
                    steering_scale,
                )
            )
            for readout_layer in trajectory_layers
        }

        def make_hook(target_position: int, prefill_length: int):
            if intervention_mode == "reinject":
                return ReinjectingActivationHook(
                    modules,
                    primary_layer_index=layer,
                    target_position=target_position,
                    steering_vectors=reinjection_vectors,
                    prefill_sequence_length=prefill_length,
                )
            return AdditiveActivationHook(
                modules,
                layer_index=layer,
                target_position=target_position,
                steering_vector=vector_tensor,
                prefill_sequence_length=prefill_length,
                injection_site=injection_site,
            )

        def reinjection_diagnostics(hook: Any) -> list[dict[str, Any]]:
            if intervention_mode != "reinject":
                return []
            diagnostics: list[dict[str, Any]] = []
            for readout_layer in trajectory_layers:
                layer_direction = repository.get(
                    steering_case.fold,
                    readout_layer,
                    position,
                )
                layer_diagnostics = manipulation_diagnostics(
                    hook.layer_view(readout_layer),
                    layer_direction,
                    alpha=alpha,
                    steering_scale=steering_scale,
                )
                _validate_manipulation(
                    layer_diagnostics,
                    f"reinject_layer_{readout_layer}",
                )
                diagnostics.append(
                    {"layer": int(readout_layer), **layer_diagnostics}
                )
            return diagnostics

        stage = "answer_generation"
        generation_holder: dict[str, Any] = {}

        def generation_context_factory(
            inputs: Any,
            rendered: str,
        ) -> AdditiveActivationHook | ReinjectingActivationHook:
            target_position, position_detail = locate_steering_position(
                tokenizer=joint_generator.tokenizer,
                rendered=rendered,
                inputs=inputs,
                assistant_text=ASSISTANT_ANSWER_PREFILL,
                text_clue=case.text_clue,
                position=position,
                assistant_occurrence="final_suffix",
            )
            hook = make_hook(target_position, int(inputs.input_ids.shape[1]))
            generation_holder.update(
                {"hook": hook, "target_position": target_position, "detail": position_detail}
            )
            return hook

        answer_result_object = joint_generator.generate(
            prompt,
            list(case.answer_classes),
            image_path,
            max_new_tokens=max(max_answer_tokens + 8, 32),
            source_classes=source_variant.classes,
            generation_context_factory=(
                None if position == "panl" else generation_context_factory
            ),
        )
        answer_result = answer_result_object.to_dict()
        if position == "panl":
            generation_diagnostics = {
                "executed": False,
                "reason": "panl_is_post_answer_and_is_steered_teacher_forced_only",
                "hook_call_count": 0,
                "steering_applied_count": 0,
                "passed": True,
            }
            record["generation_diagnostics"] = generation_diagnostics
            record["generation_reinjection_diagnostics"] = []
            record["generation_target_position"] = None
        else:
            if "hook" not in generation_holder:
                raise RuntimeError(
                    "Joint generation did not prepare a steering hook: "
                    f"{answer_result.get('error')}"
                )
            try:
                generation_diagnostics = manipulation_diagnostics(
                    generation_holder["hook"],
                    direction,
                    alpha=alpha,
                    steering_scale=steering_scale,
                )
            except Exception as exc:
                raise SteeringInvariantError(
                    f"Generation steering hook validation failed: {exc}"
                ) from exc
            record["generation_diagnostics"] = generation_diagnostics
            record["generation_target_position"] = generation_holder[
                "target_position"
            ]
            for key in (
                "decision_logit_before",
                "decision_logit_after",
                "K_before",
                "K_after",
                "injection_l2",
            ):
                record[key] = generation_diagnostics[key]
            _validate_manipulation(generation_diagnostics, "generation")
            record["generation_reinjection_diagnostics"] = reinjection_diagnostics(
                generation_holder["hook"]
            )

        if not (
            answer_result.get("parse_success")
            and answer_result.get("answer_metric_status") == "completed"
            and answer_result.get("answer")
            and answer_result.get("source_label") is not None
        ):
            raise RuntimeError(f"Joint answer generation failed: {answer_result.get('error')}")
        steered_answer = str(answer_result["answer"])
        normalized_answer = normalize_answer(steered_answer)
        probabilities = {
            str(key): float(value)
            for key, value in answer_result["answer_class_probabilities"].items()
        }
        text_answer = str(manifest["text_only_answer"])
        image_answer = str(manifest["image_only_answer"])
        if text_answer not in probabilities or image_answer not in probabilities:
            raise RuntimeError("Answer metrics omitted one of the two unimodal answers")
        p_text = probabilities[text_answer]
        p_image = probabilities[image_answer]
        denominator = p_image + p_text
        if denominator <= 0:
            raise RuntimeError("Unimodal answer probability denominator is zero")
        if normalized_answer == image_answer:
            steered_side = "follows_image"
        elif normalized_answer == text_answer:
            steered_side = "follows_text"
        else:
            steered_side = "follows_neither"

        record.update(
            {
                "steered_answer": steered_answer,
                "normalized_answer": normalized_answer,
                "steered_decision_side": steered_side,
                "P_text_answer": p_text,
                "P_image_answer": p_image,
                "pair_image_prob": p_image / denominator,
                "answer_margin": math.log(p_image + PROBABILITY_EPSILON)
                - math.log(p_text + PROBABILITY_EPSILON),
                "answer_changed": normalized_answer
                != record["baseline_normalized_answer"],
                "answer_intervention_applicable": position != "panl",
                "generated_raw_output": str(answer_result["raw_output"]),
                "generated_source_label": str(answer_result["source_label"]),
            }
        )

        stage = "teacher_forced_source_scoring"
        parsed_source_label = str(answer_result["source_label"])
        assistant_text = teacher_forced_assistant_text(
            steered_answer,
            parsed_source_label,
        )
        _messages, rendered, teacher_inputs = joint_generator.prepare_inputs(
            prompt,
            image_path,
            assistant_text=assistant_text,
        )
        try:
            target_position, _teacher_position_detail = locate_steering_position(
                tokenizer=joint_generator.tokenizer,
                rendered=rendered,
                inputs=teacher_inputs,
                assistant_text=assistant_text,
                text_clue=case.text_clue,
                position=position,
                answer=steered_answer,
            )
            sac_position, _sac_detail = _locate_sac(
                tokenizer=joint_generator.tokenizer,
                rendered=rendered,
                inputs=teacher_inputs,
                assistant_text=assistant_text,
            )
            if intervention_mode == "reinject":
                teacher_hook = make_hook(
                    target_position,
                    int(teacher_inputs.input_ids.shape[1]),
                )
            else:
                teacher_hook = AdditiveActivationHook(
                    modules,
                    layer_index=layer,
                    target_position=target_position,
                    steering_vector=vector_tensor,
                    prefill_sequence_length=int(teacher_inputs.input_ids.shape[1]),
                    capture_layer_indices=trajectory_layers,
                    injection_site=injection_site,
                )
            with teacher_hook:
                logits_by_position = run_logits_forward(
                    joint_generator.model,
                    teacher_inputs,
                    [sac_position],
                    modules,
                )
            try:
                teacher_diagnostics = manipulation_diagnostics(
                    teacher_hook,
                    direction,
                    alpha=alpha,
                    steering_scale=steering_scale,
                )
            except Exception as exc:
                raise SteeringInvariantError(
                    f"Teacher-forced steering hook validation failed: {exc}"
                ) from exc
            _validate_manipulation(teacher_diagnostics, "teacher_forced")
            teacher_reinjection_diagnostics = reinjection_diagnostics(teacher_hook)
            layer_trajectory = (
                build_layer_trajectory(
                    steering_case=steering_case,
                    repository=repository,
                    baseline_hidden_states=baseline_hidden_states,
                    hook=teacher_hook,
                    injection_layer=layer,
                    position=position,
                    teacher_forced_answer=steered_answer,
                    intervention_mode=intervention_mode,
                )
                if baseline_hidden_states is not None
                else []
            )
            scored = source_analyzer.score_vocab_logits(
                logits_by_position[sac_position],
                raw_output=str(answer_result["raw_output"]),
                parsed_label=parsed_source_label,
            ).to_dict()
        finally:
            del teacher_inputs
        record.update(
            {
                "teacher_forced_diagnostics": teacher_diagnostics,
                "teacher_forced_reinjection_diagnostics": (
                    teacher_reinjection_diagnostics
                ),
                "teacher_forced_target_position": target_position,
                "layer_trajectory": layer_trajectory,
                "sac_position": sac_position,
                "SA_soft_image_score": float(scored["soft_image_score"]),
                "SA_hard_label": str(scored["hard_label"]),
                "SA_class_probabilities": [
                    float(value) for value in scored["class_probabilities"]
                ],
                "SA_entropy": float(scored["source_entropy"]),
            }
        )
        if position == "panl":
            for key in (
                "decision_logit_before",
                "decision_logit_after",
                "K_before",
                "K_after",
                "injection_l2",
            ):
                record[key] = teacher_diagnostics[key]

        record["status"] = "completed"
        return record, False
    except SteeringInvariantError as exc:
        fatal = True
        return _failed(record, stage, exc), fatal
    except Exception as exc:
        return _failed(record, stage, exc), fatal


def _numeric_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "std": None, "sem": None}
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "std": float(std),
        "sem": float(std / math.sqrt(len(values))),
    }


def build_paired_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    completed = [record for record in records if record.get("status") == "completed"]
    baselines = {
        (record["case_id"], int(record["layer"]), str(record["position"])): record
        for record in completed
        if float(record["alpha"]) == 0.0
    }

    def baseline_abs_margin(record: dict[str, Any]) -> float:
        baseline = baselines[
            (record["case_id"], int(record["layer"]), str(record["position"]))
        ]
        return abs(float(baseline["answer_margin"]))

    subgroup_predicates = {
        "pooled": lambda _record: True,
        "follows_text": lambda record: record["baseline_decision_side"] == "follows_text",
        "follows_image": lambda record: record["baseline_decision_side"] == "follows_image",
        "conflict_easy": lambda record: record["condition"] == "conflict_easy",
        "conflict_hard": lambda record: record["condition"] == "conflict_hard",
        "baseline_abs_answer_margin_lt_0.5": (
            lambda record: baseline_abs_margin(record) < 0.5
        ),
        "baseline_abs_answer_margin_lt_1": (
            lambda record: baseline_abs_margin(record) < 1.0
        ),
        "baseline_abs_answer_margin_ge_1": (
            lambda record: baseline_abs_margin(record) >= 1.0
        ),
    }
    cells: list[dict[str, Any]] = []
    cell_keys = sorted(
        {
            (int(record["layer"]), str(record["position"]), float(record["alpha"]))
            for record in completed
        },
        key=lambda value: (value[0], value[1], value[2]),
    )
    for layer, position, alpha in cell_keys:
        cell_records = [
            record
            for record in completed
            if int(record["layer"]) == layer
            and str(record["position"]) == position
            and float(record["alpha"]) == alpha
            and (record["case_id"], layer, position) in baselines
        ]
        for subgroup, predicate in subgroup_predicates.items():
            selected = [record for record in cell_records if predicate(record)]
            answer_deltas: list[float] = []
            source_deltas: list[float] = []
            directional_answer = 0
            directional_source = 0
            text_to_image = 0
            image_to_text = 0
            baseline_text_count = 0
            baseline_image_count = 0
            for record in selected:
                baseline = baselines[(record["case_id"], layer, position)]
                answer_delta = float(record["answer_margin"]) - float(
                    baseline["answer_margin"]
                )
                source_delta = float(record["SA_soft_image_score"]) - float(
                    baseline["SA_soft_image_score"]
                )
                if position != "panl":
                    answer_deltas.append(answer_delta)
                source_deltas.append(source_delta)
                if position != "panl" and (
                    (alpha > 0 and answer_delta > 0)
                    or (alpha < 0 and answer_delta < 0)
                    or (alpha == 0 and answer_delta == 0)
                ):
                    directional_answer += 1
                if (alpha > 0 and source_delta > 0) or (alpha < 0 and source_delta < 0) or (
                    alpha == 0 and source_delta == 0
                ):
                    directional_source += 1
                if position != "panl" and (
                    record["baseline_decision_side"] == "follows_text"
                    and record["steered_decision_side"] == "follows_image"
                ):
                    text_to_image += 1
                if position != "panl" and (
                    record["baseline_decision_side"] == "follows_image"
                    and record["steered_decision_side"] == "follows_text"
                ):
                    image_to_text += 1
                if record["baseline_decision_side"] == "follows_text":
                    baseline_text_count += 1
                elif record["baseline_decision_side"] == "follows_image":
                    baseline_image_count += 1
            n = len(selected)
            cells.append(
                {
                    "layer": layer,
                    "position": position,
                    "alpha": alpha,
                    "subgroup": subgroup,
                    "injection_site": (
                        selected[0].get("injection_site", "block_output")
                        if selected
                        else None
                    ),
                    "intervention_mode": (
                        selected[0].get("intervention_mode", "single")
                        if selected
                        else None
                    ),
                    "answer_intervention_applicable": position != "panl",
                    "delta_answer_margin": _numeric_stats(answer_deltas),
                    "delta_SA_soft_image_score": _numeric_stats(source_deltas),
                    "answer_directional_success_rate": (
                        directional_answer / n if n and position != "panl" else None
                    ),
                    "SA_directional_success_rate": directional_source / n if n else None,
                    "baseline_follows_text_count": baseline_text_count,
                    "text_to_image_flip_count": text_to_image,
                    "text_to_image_flip_rate": (
                        text_to_image / baseline_text_count
                        if baseline_text_count and position != "panl"
                        else None
                    ),
                    "baseline_follows_image_count": baseline_image_count,
                    "image_to_text_flip_count": image_to_text,
                    "image_to_text_flip_rate": (
                        image_to_text / baseline_image_count
                        if baseline_image_count and position != "panl"
                        else None
                    ),
                }
            )
    trajectory_cells: list[dict[str, Any]] = []
    trajectory_keys = sorted(
        {
            (
                int(record["layer"]),
                int(row["readout_layer"]),
                str(record["position"]),
                float(record["alpha"]),
            )
            for record in completed
            for row in record.get("layer_trajectory", [])
        }
    )
    for injection_layer, readout_layer, position, alpha in trajectory_keys:
        matching = [
            (record, row)
            for record in completed
            for row in record.get("layer_trajectory", [])
            if int(record["layer"]) == injection_layer
            and int(row["readout_layer"]) == readout_layer
            and str(record["position"]) == position
            and float(record["alpha"]) == alpha
        ]
        for subgroup, predicate in subgroup_predicates.items():
            selected = [(record, row) for record, row in matching if predicate(record)]
            trajectory_cells.append(
                {
                    "injection_layer": injection_layer,
                    "readout_layer": readout_layer,
                    "layer_offset": readout_layer - injection_layer,
                    "position": position,
                    "alpha": alpha,
                    "subgroup": subgroup,
                    "injection_site": (
                        selected[0][0].get("injection_site", "block_output")
                        if selected
                        else None
                    ),
                    "intervention_mode": (
                        selected[0][0].get("intervention_mode", "single")
                        if selected
                        else None
                    ),
                    "delta_logit": _numeric_stats(
                        [float(row["delta_logit"]) for _record, row in selected]
                    ),
                    "delta_K": _numeric_stats(
                        [float(row["delta_K"]) for _record, row in selected]
                    ),
                    "retention_fraction": _numeric_stats(
                        [
                            float(row["retention_fraction"])
                            for _record, row in selected
                            if row.get("retention_fraction") is not None
                        ]
                    ),
                    "delta_hidden_l2": _numeric_stats(
                        [
                            float(row["delta_hidden_l2"])
                            for _record, row in selected
                            if row.get("delta_hidden_l2") is not None
                        ]
                    ),
                    "delta_hidden_projection_on_d_K": _numeric_stats(
                        [
                            float(row["delta_hidden_projection_on_d_K"])
                            for _record, row in selected
                            if row.get("delta_hidden_projection_on_d_K") is not None
                        ]
                    ),
                    "delta_hidden_cosine_with_d_K": _numeric_stats(
                        [
                            float(row["delta_hidden_cosine_with_d_K"])
                            for _record, row in selected
                            if row.get("delta_hidden_cosine_with_d_K") is not None
                        ]
                    ),
                    "delta_hidden_orthogonal_l2": _numeric_stats(
                        [
                            float(row["delta_hidden_orthogonal_l2"])
                            for _record, row in selected
                            if row.get("delta_hidden_orthogonal_l2") is not None
                        ]
                    ),
                    "directional_energy_fraction": _numeric_stats(
                        [
                            float(row["directional_energy_fraction"])
                            for _record, row in selected
                            if row.get("directional_energy_fraction") is not None
                        ]
                    ),
                    "answer_context_match_count": sum(
                        row.get("answer_context_matches_saved_baseline") is True
                        for _record, row in selected
                    ),
                    "answer_context_mismatch_count": sum(
                        row.get("answer_context_matches_saved_baseline") is False
                        for _record, row in selected
                    ),
                    "saved_baseline_alignment_logit_error": _numeric_stats(
                        [
                            float(row["saved_baseline_alignment_logit_error"])
                            for _record, row in selected
                            if row.get("saved_baseline_alignment_logit_error")
                            is not None
                        ]
                    ),
                }
            )
    return {
        "format_version": 2,
        "record_count": len(records),
        "completed_count": len(completed),
        "failed_count": len(records) - len(completed),
        "baseline_marked_count": sum(
            bool((record.get("baseline_validation") or {}).get("marked"))
            for record in records
        ),
        "paired_cells": cells,
        "trajectory_cells": trajectory_cells,
    }


def progress_payload(
    records: Sequence[dict[str, Any]],
    *,
    expected_count: int,
    status: str,
) -> dict[str, Any]:
    completed = sum(record.get("status") == "completed" for record in records)
    failed = sum(record.get("status") == "failed" for record in records)
    baseline_marked = sum(
        bool((record.get("baseline_validation") or {}).get("marked"))
        for record in records
    )
    return {
        "format_version": 1,
        "status": status,
        "expected_count": int(expected_count),
        "recorded_count": len(records),
        "completed_count": completed,
        "failed_count": failed,
        "baseline_marked_count": baseline_marked,
        "remaining_count": max(0, expected_count - len(records)),
    }


def _compact_runtime_record(record: dict[str, Any]) -> dict[str, Any]:
    """Keep only resume/progress/paired-analysis fields in memory."""
    keys = (
        "intervention_key",
        "case_id",
        "layer",
        "position",
        "alpha",
        "condition",
        "baseline_decision_side",
        "steered_decision_side",
        "answer_margin",
        "SA_soft_image_score",
        "status",
        "error",
        "baseline_validation",
        "answer_intervention_applicable",
        "injection_site",
        "intervention_mode",
        "layer_trajectory",
    )
    return {key: record.get(key) for key in keys if key in record}


def _load_existing_compact(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    valid_end = 0
    with path.open("rb") as handle:
        while True:
            line_start = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                valid_end = handle.tell()
                continue
            try:
                value = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                remainder = handle.read()
                if remainder.strip():
                    raise ValueError(
                        f"Invalid non-trailing JSONL record at byte {line_start}: {path}"
                    ) from exc
                break
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record at byte {line_start} is not an object")
            records.append(_compact_runtime_record(value))
            valid_end = handle.tell()
    if valid_end != path.stat().st_size:
        with path.open("r+b") as handle:
            handle.truncate(valid_end)
    return records


def initialize_output(
    output_dir: Path,
    configuration: dict[str, Any],
    *,
    resume: bool,
) -> tuple[list[dict[str, Any]], Path, Path, Path, Path]:
    config_path = output_dir / "run_config.json"
    results_path = output_dir / "results.jsonl"
    progress_path = output_dir / "progress.json"
    summary_path = output_dir / "summary.json"
    existing_paths = [
        path for path in (config_path, results_path, progress_path, summary_path) if path.exists()
    ]
    fingerprint = configuration_fingerprint(configuration)
    if existing_paths and not resume:
        raise ValueError(
            "Steering output already exists; pass --resume or choose a new directory: "
            + ", ".join(str(path) for path in existing_paths)
        )
    if resume:
        if not config_path.is_file():
            raise ValueError("--resume requires an existing run_config.json")
        saved = _read_json(config_path)
        if saved.get("config_fingerprint") != fingerprint:
            raise ValueError("Resume configuration differs from saved run_config.json")
        existing = _load_existing_compact(results_path)
    else:
        existing = []
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            config_path,
            {
                **configuration,
                "config_fingerprint": fingerprint,
                "status": "running",
            },
        )
    keys = [record.get("intervention_key") for record in existing]
    if any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("Existing Steering results contain a missing intervention_key")
    if len(keys) != len(set(keys)):
        raise ValueError("Existing Steering results contain duplicate intervention keys")
    return existing, config_path, results_path, progress_path, summary_path


def assert_cuda_only_model(model: torch.nn.Module, modules: LanguageModules) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU Steering is forbidden")
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        offloaded = {
            str(name): str(device)
            for name, device in device_map.items()
            if str(device).casefold() in {"cpu", "disk"}
        }
        if offloaded:
            raise RuntimeError(f"CPU/disk model offload is forbidden: {offloaded}")
    non_cuda_parameters = sorted(
        {
            str(parameter.device)
            for parameter in model.parameters()
            if parameter.device.type != "cuda"
        }
    )
    if non_cuda_parameters:
        raise RuntimeError(
            "All Qwen parameters must reside on CUDA; found: "
            + ", ".join(non_cuda_parameters)
        )
    for index, layer in enumerate(modules.language_layers):
        try:
            device = next(layer.parameters()).device
        except StopIteration:
            continue
        if device.type != "cuda":
            raise RuntimeError(f"Decoder layer {index} is not on CUDA: {device}")


def execute_run(
    *,
    cases: Sequence[SteeringCase],
    repository: DirectionRepository,
    joint_generator: JointAnswerSourceGenerator,
    source_analyzer: SourceAttributionAnalyzer,
    modules: LanguageModules,
    baseline_hidden_states: BaselineHiddenStateRepository | None,
    layers: Sequence[int],
    positions: Sequence[str],
    alphas: Sequence[float],
    steering_scale: str,
    injection_site: str = "block_output",
    intervention_mode: str = "single",
    max_answer_tokens: int,
    existing: list[dict[str, Any]],
    results_path: Path,
    progress_path: Path,
    summary_path: Path,
) -> list[dict[str, Any]]:
    expected_count = len(cases) * len(layers) * len(positions) * len(alphas)
    existing_keys = {str(record["intervention_key"]) for record in existing}
    expected_keys = {
        intervention_key(
            str(case.manifest["case_id"]),
            layer,
            position,
            alpha,
            steering_scale,
            injection_site,
            intervention_mode,
        )
        for case in cases
        for position in positions
        for layer in layers
        for alpha in alphas
    }
    extra_keys = existing_keys.difference(expected_keys)
    if extra_keys:
        raise ValueError(
            "Existing Steering results contain interventions outside the saved "
            f"configuration: {sorted(extra_keys)[:10]}"
        )
    ordered_alphas = [0.0] + [float(alpha) for alpha in alphas if float(alpha) != 0.0]
    atomic_write_json(
        progress_path,
        progress_payload(existing, expected_count=expected_count, status="running"),
    )
    for alpha in ordered_alphas:
        for case in cases:
            for position in positions:
                for layer in layers:
                    key = intervention_key(
                        str(case.manifest["case_id"]),
                        layer,
                        position,
                        alpha,
                        steering_scale,
                        injection_site,
                        intervention_mode,
                    )
                    if key in existing_keys:
                        continue
                    record, _invariant_failure = run_intervention(
                        steering_case=case,
                        repository=repository,
                        joint_generator=joint_generator,
                        source_analyzer=source_analyzer,
                        modules=modules,
                        layer=layer,
                        position=position,
                        alpha=alpha,
                        steering_scale=steering_scale,
                        injection_site=injection_site,
                        intervention_mode=intervention_mode,
                        max_answer_tokens=max_answer_tokens,
                        baseline_hidden_states=baseline_hidden_states,
                    )
                    append_jsonl(results_path, record, fsync=True)
                    existing.append(_compact_runtime_record(record))
                    existing_keys.add(key)
                    atomic_write_json(
                        progress_path,
                        progress_payload(
                            existing,
                            expected_count=expected_count,
                            status="running",
                        ),
                    )
    failed_count = sum(record.get("status") == "failed" for record in existing)
    if len(existing) == expected_count:
        status = "complete_with_failures" if failed_count else "complete"
    else:
        status = "incomplete"
    atomic_write_json(summary_path, build_paired_summary(existing))
    atomic_write_json(
        progress_path,
        progress_payload(existing, expected_count=expected_count, status=status),
    )
    return existing
