"""Frozen confirmatory panel for the protocol-shared attribution candidate.

The panel is deliberately external to the 80-item development screen in
``stage3_sa_truth_audit/10_protocol_shared_attribution_component``.  It uses
the 76 completed method-v2 confirmatory endpoints, fixes the answer to that
experiment's ``answer_star``, and applies the Stage-10 fold transforms without
refitting or protocol-specific calibration.

This is a report/readout validation experiment.  It contains no activation
intervention and can never authorize a causal-mediator claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

from confidence_test.dataset_utils import EvaluationCase
from layer_metacognition.hidden_state_store import (
    atomic_write_json,
    atomic_write_text,
    load_jsonl,
)

from .attribution_component import absolute_agreement_icc
from .core import (
    SAFormationArtifacts,
    atomic_save_npz,
    canonical_message_hash,
    sha256_file,
    stable_hash,
    write_jsonl_atomic,
)
from .reliance_measurement import (
    MEASUREMENT_METHOD_VERSION,
    build_answer_only_messages,
)
from .runtime import (
    PreparedMeasurement,
    Stage3Runtime,
    assistant_message,
    image_content,
    prepare_measurement,
    text_content,
)
from .second_order import ProtocolAnalyzer, ProtocolSpec
from .truth_audit import common_protocol_prompt, common_protocol_specs


BRIDGE_DIR = "stage3_sa_computational_bridge"
PANEL_DIR = "06_confirmatory_attribution_panel"
STAGE10_RELATIVE = Path(
    "stage3_sa_truth_audit/10_protocol_shared_attribution_component"
)
METHOD_V2_RELATIVE = Path(
    "stage3_sa_computational_bridge/01_actual_source_reliance"
)
EXPECTED_COMPLETED_ITEMS = 76
EXPECTED_PRESELECTED_ITEMS = 77
PRIMARY_LAYER = 18
PRIMARY_POSITION = "panl"
BOOTSTRAP_ITERATIONS = 1000
SEED = 42
FROZEN_EQUIVALENCE_BAND = 0.2120674481

CORE_PROTOCOL_NAMES = (
    "common_9_ordered",
    "common_3_ordered",
    "common_2_ordered",
    "common_3_reversed",
    "common_2_reversed",
    "common_3_semantic",
    "common_2_semantic",
)
RANDOM_LABEL_MAPPINGS: dict[int, tuple[str, ...]] = {
    4242: ("B", "T", "Q", "F", "R", "M", "Z", "V", "K"),
    314159: ("R", "Z", "K", "F", "T", "B", "M", "Q", "V"),
    20260811: ("T", "B", "F", "R", "M", "V", "Z", "K", "Q"),
}
ROW_ORDERS: dict[int, tuple[int, ...]] = {
    4242: (3, 4, 0, 8, 5, 1, 2, 7, 6),
    314159: (5, 2, 6, 8, 4, 3, 1, 0, 7),
}
RANDOM_PROTOCOL_NAMES = tuple(
    f"random_labels_seed_{seed}" for seed in RANDOM_LABEL_MAPPINGS
)
ROW_ORDER_PROTOCOL_NAMES = tuple(f"row_order_seed_{seed}" for seed in ROW_ORDERS)
HOLDOUT_PROTOCOL_NAMES = RANDOM_PROTOCOL_NAMES + ROW_ORDER_PROTOCOL_NAMES
JOINT_PROTOCOL_NAMES = CORE_PROTOCOL_NAMES + HOLDOUT_PROTOCOL_NAMES
POSTQUERY_PROTOCOL_NAME = "postquery_common_9_ordered"
ALL_PROTOCOL_NAMES = JOINT_PROTOCOL_NAMES + (POSTQUERY_PROTOCOL_NAME,)


@dataclass(frozen=True)
class PanelProtocol:
    name: str
    spec: ProtocolSpec
    display_order: tuple[int, ...]
    role: str


@dataclass(frozen=True)
class FrozenFoldDirection:
    fold: int
    d_raw: np.ndarray
    d_unit: np.ndarray
    raw_intercept: float
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    train_z_mean: float
    train_z_sd: float
    target_mean: np.ndarray
    target_scale: np.ndarray
    target_loading: np.ndarray

    def predict(self, hidden: np.ndarray) -> float:
        value = np.asarray(hidden, dtype=np.float64)
        return float(value @ self.d_raw + self.raw_intercept)

    def coordinate(self, hidden: np.ndarray) -> float:
        value = np.asarray(hidden, dtype=np.float64)
        return float((value @ self.d_unit - self.train_z_mean) / self.train_z_sd)

    def transform_target(self, common_scores: Sequence[float]) -> float:
        values = np.asarray(common_scores, dtype=np.float64)
        if values.shape != self.target_mean.shape:
            raise ValueError(
                f"Fold {self.fold} target shape mismatch: {values.shape} != "
                f"{self.target_mean.shape}"
            )
        return float(((values - self.target_mean) / self.target_scale) @ self.target_loading)


class FrozenDirectionRepository:
    """Read the byte-frozen Stage-10 directions copied under the new output."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir).resolve()
        rule_path = self.output_dir / "frozen_rule.json"
        if not rule_path.is_file():
            raise FileNotFoundError(f"Frozen rule is missing: {rule_path}")
        self.rule = json.loads(rule_path.read_text(encoding="utf-8"))
        candidate = dict(self.rule)
        fingerprint = str(candidate.pop("rule_fingerprint"))
        if stable_hash(candidate) != fingerprint:
            raise ValueError("Frozen attribution rule fingerprint mismatch")
        self._cache: dict[int, FrozenFoldDirection] = {}

    def get(self, fold: int) -> FrozenFoldDirection:
        key = int(fold)
        if key in self._cache:
            return self._cache[key]
        entry = next(
            (value for value in self.rule["folds"] if int(value["fold"]) == key),
            None,
        )
        if entry is None:
            raise KeyError(f"Frozen attribution rule omits fold {key}")
        path = self.output_dir / str(entry["frozen_file"])
        if sha256_file(path) != str(entry["sha256"]):
            raise ValueError(f"Frozen direction checksum mismatch for fold {key}")
        with np.load(path, allow_pickle=False) as payload:
            direction = FrozenFoldDirection(
                fold=key,
                d_raw=np.asarray(payload["d_raw"], dtype=np.float64),
                d_unit=np.asarray(payload["d_unit"], dtype=np.float64),
                raw_intercept=float(payload["raw_intercept"]),
                scaler_mean=np.asarray(payload["scaler_mean"], dtype=np.float64),
                scaler_scale=np.asarray(payload["scaler_scale"], dtype=np.float64),
                train_z_mean=float(payload["train_z_mean"]),
                train_z_sd=float(payload["train_z_sd"]),
                target_mean=np.asarray(payload["target_mean"], dtype=np.float64),
                target_scale=np.asarray(payload["target_scale"], dtype=np.float64),
                target_loading=np.asarray(payload["target_loading"], dtype=np.float64),
            )
        if not math.isclose(
            float(np.linalg.norm(direction.d_unit)), 1.0, rel_tol=1e-6, abs_tol=1e-6
        ):
            raise ValueError(f"Frozen fold {key} direction is not unit norm")
        arrays = (
            direction.d_raw,
            direction.d_unit,
            direction.scaler_mean,
            direction.scaler_scale,
            direction.target_mean,
            direction.target_scale,
            direction.target_loading,
        )
        if any(not np.isfinite(value).all() for value in arrays):
            raise ValueError(f"Frozen fold {key} contains non-finite arrays")
        if direction.train_z_sd <= 0 or np.any(direction.target_scale <= 0):
            raise ValueError(f"Frozen fold {key} contains an invalid training scale")
        self._cache[key] = direction
        return direction


def panel_root(experiment_dir: str | Path) -> Path:
    return Path(experiment_dir).resolve() / BRIDGE_DIR / PANEL_DIR


def stage10_root(experiment_dir: str | Path) -> Path:
    return Path(experiment_dir).resolve() / STAGE10_RELATIVE


def method_v2_root(experiment_dir: str | Path) -> Path:
    return Path(experiment_dir).resolve() / METHOD_V2_RELATIVE


def _numeric_key(value: Any) -> tuple[int, str]:
    text = str(value)
    try:
        return int(text), text
    except ValueError:
        return 10**12, text


def _latest_rows(path: str | Path) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path, repair_trailing=True):
        key = str(row.get("intervention_key") or row.get("case_id") or "")
        if key:
            latest[key] = row
    return list(latest.values())


def _descriptions_and_midpoints() -> tuple[tuple[str, ...], tuple[float, ...]]:
    nine = common_protocol_specs()[0]
    return tuple(nine.descriptions), tuple(float(value) for value in nine.midpoints)


def panel_protocols() -> tuple[PanelProtocol, ...]:
    """Return the exact frozen joint protocol order used by every item."""

    core_specs = common_protocol_specs()
    if tuple(spec.name for spec in core_specs) != CORE_PROTOCOL_NAMES:
        raise ValueError("Core common protocol definitions drifted from Stage 10")
    result: list[PanelProtocol] = [
        PanelProtocol(
            name=spec.name,
            spec=spec,
            display_order=tuple(range(len(spec.labels_by_semantic))),
            role="core_common",
        )
        for spec in core_specs
    ]
    descriptions, midpoints = _descriptions_and_midpoints()
    for seed, labels in RANDOM_LABEL_MAPPINGS.items():
        name = f"random_labels_seed_{seed}"
        result.append(
            PanelProtocol(
                name=name,
                spec=ProtocolSpec(name, labels, midpoints, descriptions),
                display_order=tuple(range(9)),
                role="frozen_random_label_holdout",
            )
        )
    normal_labels = tuple(str(index) for index in range(9))
    for seed, order in ROW_ORDERS.items():
        if tuple(sorted(order)) != tuple(range(9)):
            raise ValueError(f"Row order seed {seed} is not a permutation of 0..8")
        name = f"row_order_seed_{seed}"
        result.append(
            PanelProtocol(
                name=name,
                spec=ProtocolSpec(name, normal_labels, midpoints, descriptions),
                display_order=order,
                role="frozen_row_order_holdout",
            )
        )
    if tuple(value.name for value in result) != JOINT_PROTOCOL_NAMES:
        raise RuntimeError("Frozen joint protocol order construction failed")
    return tuple(result)


def protocol_freeze_payload() -> dict[str, Any]:
    values = panel_protocols()
    return {
        "core": [
            {
                "name": value.name,
                "labels_by_semantic": list(value.spec.labels_by_semantic),
                "midpoints": list(value.spec.midpoints),
                "descriptions": list(value.spec.descriptions),
                "display_order": list(value.display_order),
                "role": value.role,
            }
            for value in values
            if value.name in CORE_PROTOCOL_NAMES
        ],
        "random_label_mappings": {
            str(seed): list(labels) for seed, labels in RANDOM_LABEL_MAPPINGS.items()
        },
        "row_orders": {str(seed): list(order) for seed, order in ROW_ORDERS.items()},
        "joint_protocol_order": list(JOINT_PROTOCOL_NAMES),
        "postquery_protocol": POSTQUERY_PROTOCOL_NAME,
        "postquery_mapping": list(common_protocol_specs()[0].labels_by_semantic),
    }


def _joint_prompt_with_order(case: EvaluationCase, protocol: PanelProtocol) -> str:
    if protocol.name in CORE_PROTOCOL_NAMES:
        return common_protocol_prompt(case, protocol.spec)
    rows = [
        f"{protocol.spec.labels_by_semantic[index]}: {protocol.spec.descriptions[index]}"
        for index in protocol.display_order
    ]
    class_block = "\n".join(rows)
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


def postquery_prompt(spec: ProtocolSpec | None = None) -> str:
    selected = spec or common_protocol_specs()[0]
    class_block = "\n".join(
        f"{label}: {description}"
        for label, description in zip(
            selected.labels_by_semantic, selected.descriptions, strict=True
        )
    )
    return f"""Report the relative contribution of the text clue and the image to the fixed answer already given above by choosing exactly one class listed below.

Source attribution classes:
{class_block}

Output exactly:

**Source Attribution**:<CLASS>

Do not repeat the answer. Do not include reasoning, confidence, or any additional text."""


def build_joint_messages(
    case: EvaluationCase,
    condition: str,
    protocol: PanelProtocol,
    *,
    answer_star: str,
) -> tuple[list[dict[str, Any]], str]:
    """Build a joint context using only method-v2 ``answer_star`` as endpoint."""

    answer = str(answer_star)
    if not answer:
        raise ValueError("method-v2 answer_star is empty")
    assistant_text = f"**Answer**: {answer}\n**Source Attribution**:"
    messages = [
        {
            "role": "user",
            "content": image_content(
                str(case.conditions[condition].resolved_image_path),
                _joint_prompt_with_order(case, protocol),
            ),
        },
        assistant_message(assistant_text),
    ]
    return messages, assistant_text


def build_postquery_messages(
    case: EvaluationCase,
    condition: str,
    *,
    answer_star: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Build the exact answer-only base followed by a new SA-report turn."""

    answer = str(answer_star)
    if not answer:
        raise ValueError("method-v2 answer_star is empty")
    image_path = str(case.conditions[condition].resolved_image_path)
    base_assistant = f"**Answer**: {answer}"
    base = build_answer_only_messages(
        case,
        text_clue=case.text_clue,
        image_path=image_path,
        assistant_text=base_assistant,
    )
    assistant_text = "**Source Attribution**:"
    branch = [
        *base,
        {"role": "user", "content": text_content(postquery_prompt())},
        assistant_message(assistant_text),
    ]
    return base, branch, assistant_text


def audit_answer_star_token(tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    """Match the current tokenizer's one-token A* encoding to method-v2."""

    answer = str(row["answer_star"])
    expected = int(row["method_v2_answer_star_token_id"])
    ids = [int(value) for value in tokenizer.encode(" " + answer, add_special_tokens=False)]
    return {
        "answer_star": answer,
        "encoding_text": " " + answer,
        "current_token_ids": ids,
        "method_v2_token_id": expected,
        "single_token": len(ids) == 1,
        "token_id_equal": len(ids) == 1 and ids[0] == expected,
        "passed": len(ids) == 1 and ids[0] == expected,
    }


def _input_ids(inputs: Any) -> np.ndarray:
    values = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[0] != 1:
        raise ValueError(f"Expected singleton batched input ids, got {array.shape}")
    return array[0]


def audit_prefix_through_answer(
    *,
    base_messages: Sequence[dict[str, Any]],
    branch_messages: Sequence[dict[str, Any]],
    base_rendered: str,
    branch_rendered: str,
    base_inputs: Any,
    branch_inputs: Any,
    answer_star: str,
) -> dict[str, Any]:
    """Verify structural identity through A* without comparing BF16 hidden values."""

    base_hash = canonical_message_hash(base_messages)
    branch_base_hash = canonical_message_hash(branch_messages[: len(base_messages)])
    marker = f"**Answer**: {answer_star}"
    base_start = base_rendered.rfind(marker)
    branch_start = branch_rendered.find(marker)
    if base_start < 0 or branch_start < 0:
        raise ValueError("Rendered prefix does not contain the exact method-v2 A*")
    base_text = base_rendered[: base_start + len(marker)]
    branch_text = branch_rendered[: branch_start + len(marker)]
    base_ids = _input_ids(base_inputs)
    branch_ids = _input_ids(branch_inputs)
    token_prefix_equal = bool(
        len(branch_ids) >= len(base_ids)
        and np.array_equal(branch_ids[: len(base_ids)], base_ids)
    )
    checks = {
        "canonical_base_message_hash_equal": base_hash == branch_base_hash,
        "rendered_prefix_through_answer_equal": base_text == branch_text,
        "token_prefix_through_answer_equal": token_prefix_equal,
    }
    return {
        "passed": all(checks.values()),
        "answer_star": str(answer_star),
        "canonical_base_message_hash": base_hash,
        "branch_base_message_hash": branch_base_hash,
        "base_input_token_count": int(len(base_ids)),
        "branch_input_token_count": int(len(branch_ids)),
        "checks": checks,
        "hidden_numeric_equality_tested": False,
        "hidden_numeric_equality_claimed": False,
        "bf16_shape_drift_note": (
            "Exact causal-prefix text/tokens do not imply bitwise hidden equality when "
            "the total prefill shape differs under BF16 kernels."
        ),
    }


def load_confirmatory_cohort(
    experiment_dir: str | Path,
    *,
    expected_completed: int = EXPECTED_COMPLETED_ITEMS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load exactly the terminal method-v2 confirmatory endpoints."""

    root = Path(experiment_dir).resolve()
    source_path = method_v2_root(root) / "confirmatory_results.jsonl"
    rows = _latest_rows(source_path)
    if len(rows) != EXPECTED_PRESELECTED_ITEMS:
        raise ValueError(
            f"Expected {EXPECTED_PRESELECTED_ITEMS} terminal method-v2 rows, found {len(rows)}"
        )
    completed = [row for row in rows if row.get("status") == "completed"]
    excluded = [row for row in rows if row.get("status") == "excluded"]
    failed = [row for row in rows if row.get("status") == "failed"]
    if len(completed) != expected_completed or failed:
        raise ValueError(
            f"Expected {expected_completed} completed and zero failed rows; "
            f"found completed={len(completed)} failed={len(failed)}"
        )
    if len(excluded) != EXPECTED_PRESELECTED_ITEMS - expected_completed or any(
        row.get("exclusion_reason") != "tied_natural_endpoint" for row in excluded
    ):
        raise ValueError("The sole pre-registered endpoint exclusion has drifted")
    item_ids = [str(row["item_id"]) for row in completed]
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("Confirmatory panel must contain one endpoint per item")
    stage10_manifest_path = stage10_root(root) / "cohort_manifest.json"
    stage10_manifest = json.loads(stage10_manifest_path.read_text(encoding="utf-8"))
    stage10_items = {str(value) for value in stage10_manifest["item_ids"]}
    overlap = sorted(set(item_ids).intersection(stage10_items), key=_numeric_key)
    if overlap:
        raise RuntimeError(f"Confirmatory/Stage-10 development item overlap: {overlap}")

    selected: list[dict[str, Any]] = []
    for source in completed:
        if int(source.get("measurement_method_version", -1)) != MEASUREMENT_METHOD_VERSION:
            raise ValueError(f"Non-method-v2 endpoint: {source.get('case_id')}")
        if source.get("split") != "confirmatory":
            raise ValueError(f"Non-confirmatory endpoint: {source.get('case_id')}")
        answer = source.get("answer_star")
        if not isinstance(answer, str) or not answer:
            raise ValueError(f"Missing answer_star: {source.get('case_id')}")
        if answer != source.get("answer_only_answer"):
            raise ValueError(f"answer_star/answer_only_answer drift: {source.get('case_id')}")
        if not bool(source.get("selection_measurement_same_forward")):
            raise ValueError(f"A* was not selected in the measurement forward: {source.get('case_id')}")
        if bool(source.get("verbal_sa_leakage")):
            raise ValueError(f"Method-v2 endpoint contains verbal-SA leakage: {source.get('case_id')}")
        teacher_hash = source.get("selection", {}).get("teacher_forced_messages_hash")
        if not teacher_hash:
            raise ValueError(f"Missing teacher-forced base hash: {source.get('case_id')}")
        token_map = source.get("selection", {}).get("canonical_leading_token_ids")
        if not isinstance(token_map, dict) or answer not in token_map:
            raise ValueError(f"Missing method-v2 A* token id: {source.get('case_id')}")
        selected.append(
            {
                "case_id": str(source["case_id"]),
                "item_id": str(source["item_id"]),
                "prior_index": int(source["prior_index"]),
                "condition": str(source["condition"]),
                "difficulty": str(source["difficulty"]),
                "fold": int(source["fold"]),
                "answer_star": answer,
                "answer_star_side": str(source["answer_star_side"]),
                "text_answer": source.get("text_answer"),
                "image_answer": source.get("image_answer"),
                "method_v2_intervention_key": str(source["intervention_key"]),
                "method_v2_manifest_fingerprint": str(source["manifest_fingerprint"]),
                "method_v2_selection_messages_hash": str(
                    source["selection"]["messages_hash"]
                ),
                "method_v2_teacher_forced_messages_hash": str(teacher_hash),
                "method_v2_selection_rendered_hash": str(
                    source["selection_rendered_hash"]
                ),
                "method_v2_answer_star_token_id": int(token_map[answer]),
                "answer_source": "method_v2_confirmatory_results.answer_star",
            }
        )
    selected.sort(key=lambda row: (_numeric_key(row["item_id"]), row["case_id"]))
    audit = {
        "status": "passed",
        "preselected_n": len(rows),
        "completed_n": len(selected),
        "excluded_n": len(excluded),
        "failed_n": len(failed),
        "excluded_case_ids": [str(row["case_id"]) for row in excluded],
        "excluded_reasons": [str(row["exclusion_reason"]) for row in excluded],
        "unique_item_n": len(set(item_ids)),
        "stage10_development_item_n": len(stage10_items),
        "stage10_item_overlap": overlap,
        "item_isolation_passed": not overlap,
        "answer_field": "answer_star",
        "fallback_answer_field": None,
        "method_version": MEASUREMENT_METHOD_VERSION,
        "source_results": str(source_path),
        "source_sha256": sha256_file(source_path),
    }
    return selected, audit


def build_cohort_manifest(
    rows: Sequence[dict[str, Any]], endpoint_audit: dict[str, Any]
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        fold = str(int(row["fold"]))
        counts[fold] = counts.get(fold, 0) + 1
    payload = {
        "format_version": 1,
        "n": len(rows),
        "unique_items": len({str(row["item_id"]) for row in rows}),
        "fold_counts": counts,
        "answer_endpoint": "method-v2 confirmatory answer_star (never old joint final_answer)",
        "item_isolation": "zero overlap with Stage-10 80-item development panel",
        "forwards_per_item": len(ALL_PROTOCOL_NAMES),
        "expected_formal_forward_count": len(rows) * len(ALL_PROTOCOL_NAMES),
        "joint_protocol_order": list(JOINT_PROTOCOL_NAMES),
        "postquery_protocol": POSTQUERY_PROTOCOL_NAME,
        "endpoint_audit": endpoint_audit,
        "rows": [dict(row) for row in rows],
    }
    payload["manifest_fingerprint"] = stable_hash(payload)
    return payload


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_sha = sha256_file(source)
    if destination.exists():
        if sha256_file(destination) != source_sha:
            raise FileExistsError(f"Frozen copy differs from source: {destination}")
        return
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        with source.open("rb") as reader, open(temporary, "wb") as writer:
            shutil.copyfileobj(reader, writer)
            writer.flush()
            os.fsync(writer.fileno())
        if sha256_file(temporary) != source_sha:
            raise IOError(f"Atomic frozen copy checksum failed: {source}")
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def inspect_stage10_rule(experiment_dir: str | Path) -> dict[str, Any]:
    source = stage10_root(experiment_dir)
    summary_path = source / "summary.json"
    index_path = source / "directions" / "index.json"
    if not summary_path.is_file() or not index_path.is_file():
        raise FileNotFoundError("Completed Stage-10 attribution component is required")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if summary.get("status") != "completed" or not summary.get("rank_gate", {}).get("passed"):
        raise ValueError("Stage-10 frozen source did not pass its rank gate")
    if not summary.get("coordinate_metrics", {}).get(
        "common_pairwise_equivalence_passed"
    ):
        raise ValueError("Stage-10 common-template coordinate component did not pass")
    folds: list[dict[str, Any]] = []
    expected_folds = {0, 1, 2, 3, 4}
    observed_folds = {int(value["fold"]) for value in index["folds"]}
    if observed_folds != expected_folds:
        raise ValueError(f"Stage-10 direction folds drifted: {observed_folds}")
    required_arrays = {
        "d_raw",
        "d_unit",
        "raw_intercept",
        "scaler_mean",
        "scaler_scale",
        "train_z_mean",
        "train_z_sd",
        "target_mean",
        "target_scale",
        "target_loading",
    }
    for entry in sorted(index["folds"], key=lambda value: int(value["fold"])):
        path = source / "directions" / str(entry["file"])
        with np.load(path, allow_pickle=False) as payload:
            missing = required_arrays.difference(payload.files)
            if missing:
                raise ValueError(f"Stage-10 fold {entry['fold']} lacks arrays: {sorted(missing)}")
            array_fingerprint = stable_hash(
                {
                    name: {
                        "shape": list(np.asarray(payload[name]).shape),
                        "dtype": str(np.asarray(payload[name]).dtype),
                        "sha256": hashlib.sha256(
                            np.asarray(payload[name]).tobytes(order="C")
                        ).hexdigest(),
                    }
                    for name in sorted(required_arrays)
                }
            )
        folds.append(
            {
                "fold": int(entry["fold"]),
                "source_file": str(path),
                "source_file_name": path.name,
                "source_sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "array_fingerprint": array_fingerprint,
                "selected_alpha": float(entry["selected_alpha"]),
                "train_z_mean": float(entry["train_z_mean"]),
                "train_z_sd": float(entry["train_z_sd"]),
                "target_mean": list(entry["target_mean"]),
                "target_scale": list(entry["target_scale"]),
                "target_loading": list(entry["target_loading"]),
                "train_items": list(entry["train_items"]),
                "test_items": list(entry["test_items"]),
            }
        )
    return {
        "format_version": 1,
        "source_stage10": str(source),
        "source_summary_sha256": sha256_file(summary_path),
        "source_direction_index_sha256": sha256_file(index_path),
        "source_rank_gate_passed": True,
        "source_common_coordinate_component_passed": True,
        "layer": PRIMARY_LAYER,
        "position": PRIMARY_POSITION,
        "fit_policy": "no refit, no recalibration, no sign change, no protocol indicator",
        "prediction": "h @ d_raw + raw_intercept",
        "coordinate": "(h @ d_unit - train_z_mean) / train_z_sd",
        "target_transform": "((seven_common_scores-target_mean)/target_scale) @ target_loading",
        "protocols": protocol_freeze_payload(),
        "rank_gate": {
            "target_r2_minimum_exclusive": 0.0,
            "target_spearman_ci_lower_minimum_exclusive": 0.0,
            "every_core_protocol_spearman_ci_lower_minimum_exclusive": 0.0,
            "every_random_or_order_holdout_spearman_ci_lower_minimum_exclusive": 0.0,
        },
        "common_coordinate_gate": {
            "equivalence_band": FROZEN_EQUIVALENCE_BAND,
            "icc_point_minimum": 0.75,
            "icc_ci_lower_minimum": 0.60,
            "within_between_sd_ratio_maximum": 0.50,
            "slope_interval": [0.80, 1.25],
            "reference_protocol": "common_9_ordered",
            "protocol_specific_calibration": False,
        },
        "postquery_gate": {
            "report_spearman_ci_lower_minimum_exclusive": 0.0,
            "core_target_spearman_ci_lower_minimum_exclusive": 0.0,
            "joint_postquery_report_spearman_ci_lower_minimum_exclusive": 0.0,
            "coordinate_gate_bearing": False,
        },
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_unit": "confirmatory item",
        "seed": SEED,
        "causal_intervention": False,
        "causal_mediator_authorized": False,
        "folds": folds,
    }


def freeze_stage10_rule(
    experiment_dir: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Byte-copy and fingerprint all Stage-10 fold artifacts under output 06."""

    output = Path(output_dir).resolve()
    source = inspect_stage10_rule(experiment_dir)
    frozen_folds: list[dict[str, Any]] = []
    for entry in source["folds"]:
        source_path = Path(entry["source_file"])
        destination = output / "frozen_directions" / source_path.name
        _atomic_copy(source_path, destination)
        copied = dict(entry)
        copied["frozen_file"] = str(destination.relative_to(output))
        copied["sha256"] = sha256_file(destination)
        frozen_folds.append(copied)
    rule = dict(source)
    rule["folds"] = frozen_folds
    rule["rule_fingerprint"] = stable_hash(rule)
    path = output / "frozen_rule.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != rule:
            raise ValueError("Existing frozen rule differs; refusing to overwrite")
    else:
        atomic_write_json(path, rule)
    return rule


def _measurement_payload(
    measured: Any,
    prepared: Any,
    direction: FrozenFoldDirection,
    protocol: PanelProtocol,
) -> tuple[dict[str, Any], np.ndarray]:
    hidden = np.asarray(measured.hidden, dtype=np.float32)
    if hidden.ndim != 1 or not np.isfinite(hidden).all():
        raise ValueError(f"Invalid L18 hidden state for {protocol.name}: {hidden.shape}")
    source = measured.source
    return (
        {
            "role": protocol.role,
            "labels_by_semantic": list(protocol.spec.labels_by_semantic),
            "midpoints": list(protocol.spec.midpoints),
            "display_order": list(protocol.display_order),
            "semantic_imageward_score": float(source["soft_image_score"]),
            "hard_label": source["hard_label"],
            "class_logits": source["class_logits"],
            "class_probabilities": source["class_probabilities"],
            "frozen_prediction": direction.predict(hidden),
            "frozen_coordinate": direction.coordinate(hidden),
            "prefix_hash": prepared.prefix_hash,
            "panl_position": int(prepared.panl_position),
            "target_position": int(prepared.target_position),
            "input_token_count": int(prepared.inputs.input_ids.shape[1]),
            "hook_call_count": int(measured.hook_call_count),
            "hook_applied_count": int(measured.applied_count),
            "hook_exactly_once": bool(measured.applied_count == 1),
            "steering_applied": False,
            "injection_l2": float(measured.injection_l2),
        },
        hidden,
    )


def measure_confirmatory_case(
    runtime: Stage3Runtime,
    row: dict[str, Any],
    direction: FrozenFoldDirection,
    hidden_path: str | Path,
    *,
    deadline: Callable[[], None],
) -> dict[str, Any]:
    """Run the frozen 12-joint + one postquery panel for one endpoint."""

    case = runtime.case(str(row["item_id"]), int(row["prior_index"]))
    answer_star = str(row["answer_star"])
    answer_token_audit = audit_answer_star_token(runtime.generator.tokenizer, row)
    if not answer_token_audit["passed"]:
        raise ValueError(
            f"Current tokenizer A* mapping differs from method-v2: {answer_token_audit}"
        )
    protocol_results: dict[str, Any] = {}
    hidden_values: list[np.ndarray] = []
    analyzers: dict[str, ProtocolAnalyzer] = {}
    for protocol in panel_protocols():
        analyzers[protocol.name] = ProtocolAnalyzer(
            runtime.generator.tokenizer, protocol.spec
        )

    for protocol in panel_protocols():
        deadline()
        messages, assistant_text = build_joint_messages(
            case,
            str(row["condition"]),
            protocol,
            answer_star=answer_star,
        )
        prepared = prepare_measurement(
            runtime.generator,
            messages,
            assistant_text=assistant_text,
            answer=answer_star,
        )
        measured = runtime.measure(
            prepared,
            direction,  # type: ignore[arg-type]
            analyzer=analyzers[protocol.name],
        )
        payload, hidden = _measurement_payload(
            measured, prepared, direction, protocol
        )
        protocol_results[protocol.name] = payload
        hidden_values.append(hidden)
        runtime.release_inputs(prepared)

    deadline()
    common9 = common_protocol_specs()[0]
    post_protocol = PanelProtocol(
        POSTQUERY_PROTOCOL_NAME,
        ProtocolSpec(
            POSTQUERY_PROTOCOL_NAME,
            tuple(common9.labels_by_semantic),
            tuple(common9.midpoints),
            tuple(common9.descriptions),
        ),
        tuple(range(9)),
        "branched_post_answer_report",
    )
    base_messages, branch_messages, branch_assistant = build_postquery_messages(
        case, str(row["condition"]), answer_star=answer_star
    )
    base_assistant = f"**Answer**: {answer_star}"
    base_rendered, base_inputs = runtime.generator.prepare_messages(
        base_messages, assistant_text=base_assistant
    )
    branch_rendered, branch_inputs = runtime.generator.prepare_messages(
        branch_messages, assistant_text=branch_assistant
    )
    prefix_audit = audit_prefix_through_answer(
        base_messages=base_messages,
        branch_messages=branch_messages,
        base_rendered=base_rendered,
        branch_rendered=branch_rendered,
        base_inputs=base_inputs,
        branch_inputs=branch_inputs,
        answer_star=answer_star,
    )
    prefix_audit["method_v2_teacher_forced_messages_hash"] = str(
        row["method_v2_teacher_forced_messages_hash"]
    )
    prefix_audit["matches_method_v2_teacher_forced_messages_hash"] = bool(
        prefix_audit["canonical_base_message_hash"]
        == row["method_v2_teacher_forced_messages_hash"]
    )
    prefix_audit["passed"] = bool(
        prefix_audit["passed"]
        and prefix_audit["matches_method_v2_teacher_forced_messages_hash"]
    )
    if not prefix_audit["passed"]:
        raise ValueError(f"Postquery causal-prefix reconstruction failed: {prefix_audit}")
    base_token_count = int(len(_input_ids(base_inputs)))
    branch = PreparedMeasurement(
        messages=branch_messages,
        rendered=branch_rendered,
        inputs=branch_inputs,
        assistant_text=branch_assistant,
        answer=answer_star,
        panl_position=base_token_count - 1,
        target_position=int(len(_input_ids(branch_inputs))) - 1,
        prefix_hash=canonical_message_hash(branch_messages),
    )
    del base_inputs
    post_analyzer = ProtocolAnalyzer(runtime.generator.tokenizer, post_protocol.spec)
    measured = runtime.measure(
        branch,
        direction,  # type: ignore[arg-type]
        analyzer=post_analyzer,
    )
    payload, hidden = _measurement_payload(
        measured, branch, direction, post_protocol
    )
    payload["prefix_through_answer_audit"] = prefix_audit
    payload["hidden_site"] = (
        "answer-prefix state in the longer postquery branch; analyzed separately "
        "from the original joint-PANL coordinate gate"
    )
    protocol_results[POSTQUERY_PROTOCOL_NAME] = payload
    hidden_values.append(hidden)
    runtime.release_inputs(branch)

    hidden_array = np.stack(hidden_values).astype(np.float32, copy=False)
    if hidden_array.shape[0] != len(ALL_PROTOCOL_NAMES):
        raise RuntimeError(f"Incomplete hidden panel: {hidden_array.shape}")
    destination = Path(hidden_path)
    atomic_save_npz(
        destination,
        protocols=np.asarray(ALL_PROTOCOL_NAMES),
        joint_protocols=np.asarray(JOINT_PROTOCOL_NAMES),
        hidden=hidden_array,
        layer=np.asarray(PRIMARY_LAYER, dtype=np.int64),
    )
    return {
        "status": "completed",
        "case_id": str(row["case_id"]),
        "item_id": str(row["item_id"]),
        "prior_index": int(row["prior_index"]),
        "condition": str(row["condition"]),
        "difficulty": str(row["difficulty"]),
        "fold": int(row["fold"]),
        "answer_star": answer_star,
        "answer_star_side": str(row["answer_star_side"]),
        "answer_source": "method_v2_confirmatory_results.answer_star",
        "answer_star_token_audit": answer_token_audit,
        "method_v2_intervention_key": str(row["method_v2_intervention_key"]),
        "method_v2_teacher_forced_messages_hash": str(
            row["method_v2_teacher_forced_messages_hash"]
        ),
        "protocol_order": list(ALL_PROTOCOL_NAMES),
        "joint_protocol_order": list(JOINT_PROTOCOL_NAMES),
        "postquery_protocol": POSTQUERY_PROTOCOL_NAME,
        "protocols": protocol_results,
        "hidden_file": str(destination.name),
        "hidden_sha256": sha256_file(destination),
        "hidden_shape": list(hidden_array.shape),
        "hidden_dtype": str(hidden_array.dtype),
        "formal_forward_count": len(ALL_PROTOCOL_NAMES),
        "causal_intervention": False,
    }


def _append_record_atomic(path: Path, row: dict[str, Any]) -> None:
    existing = load_jsonl(path, repair_trailing=True)
    write_jsonl_atomic(path, [*existing, row])


def run_confirmatory_panel(
    runtime: Stage3Runtime,
    rows: Sequence[dict[str, Any]],
    output_dir: str | Path,
    *,
    deadline: Callable[[], None],
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    repository = FrozenDirectionRepository(output)
    result_path = output / "results.jsonl"
    latest = {
        str(row.get("intervention_key") or row.get("case_id")): row
        for row in _latest_rows(result_path)
    }
    for source in rows:
        deadline()
        key = f"confirmatory_attribution|{source['case_id']}"
        if latest.get(key, {}).get("status") == "completed":
            hidden = output / "hidden" / str(latest[key]["hidden_file"])
            if not hidden.is_file() or sha256_file(hidden) != latest[key].get("hidden_sha256"):
                raise ValueError(f"Completed result has a missing/tampered hidden file: {key}")
            continue
        base = {
            "intervention_key": key,
            "experiment": "confirmatory_attribution_panel",
            "case_id": str(source["case_id"]),
            "item_id": str(source["item_id"]),
            "fold": int(source["fold"]),
        }
        started = time.perf_counter()
        try:
            hidden_path = output / "hidden" / f"{str(source['case_id']).replace('/', '_')}.npz"
            measured = measure_confirmatory_case(
                runtime,
                dict(source),
                repository.get(int(source["fold"])),
                hidden_path,
                deadline=deadline,
            )
            record = {
                **base,
                **measured,
                "hidden_file": str(hidden_path.relative_to(output / "hidden")),
                "elapsed_seconds": time.perf_counter() - started,
            }
        except Exception as exc:
            if type(exc).__name__ == "TimeBudgetExceeded":
                raise
            record = {
                **base,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": time.perf_counter() - started,
            }
        _append_record_atomic(result_path, record)
        latest[key] = record
        atomic_write_json(
            output / "progress.json",
            {
                "status": "running",
                "completed_n": sum(
                    value.get("status") == "completed" for value in latest.values()
                ),
                "failed_n": sum(
                    value.get("status") == "failed" for value in latest.values()
                ),
                "expected_n": len(rows),
                "formal_forward_count_expected": len(rows)
                * len(ALL_PROTOCOL_NAMES),
                "last_case_id": str(source["case_id"]),
            },
        )
    terminal = _latest_rows(result_path)
    return {
        "terminal_n": len(terminal),
        "completed_n": sum(row.get("status") == "completed" for row in terminal),
        "failed_n": sum(row.get("status") == "failed" for row in terminal),
    }


def _safe_corr(x: np.ndarray, y: np.ndarray, *, rank: bool) -> float | None:
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    if len(left) < 3 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    value = spearmanr(left, right).statistic if rank else pearsonr(left, right).statistic
    return float(value) if np.isfinite(value) else None


def bootstrap_association(
    x: Sequence[float],
    y: Sequence[float],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = SEED,
) -> dict[str, Any]:
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("Association arrays must be aligned one-dimensional arrays")
    pearson = _safe_corr(left, right, rank=False)
    spearman = _safe_corr(left, right, rank=True)
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(iterations):
        indices = rng.integers(0, len(left), size=len(left))
        value = _safe_corr(left[indices], right[indices], rank=True)
        if value is not None:
            samples.append(value)
    ci = [None, None]
    if samples:
        ci = [float(value) for value in np.quantile(samples, [0.025, 0.975])]
    return {
        "n": int(len(left)),
        "pearson": pearson,
        "spearman": spearman,
        "spearman_ci95": ci,
        "bootstrap_iterations": iterations,
        "bootstrap_valid": len(samples),
    }


def _mean_ci(values: np.ndarray, *, iterations: int, seed: int) -> list[float | None]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = [
        float(np.mean(array[rng.integers(0, len(array), size=len(array))]))
        for _ in range(iterations)
    ]
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def _slope_ci(
    reference: np.ndarray, other: np.ndarray, *, iterations: int, seed: int
) -> dict[str, Any]:
    x = np.asarray(reference, dtype=np.float64)
    y = np.asarray(other, dtype=np.float64)

    def fit(indices: np.ndarray) -> tuple[float, float] | None:
        selected = x[indices]
        if np.std(selected) <= 1e-12:
            return None
        slope, intercept = np.polyfit(selected, y[indices], 1)
        return float(slope), float(intercept)

    point = fit(np.arange(len(x)))
    rng = np.random.default_rng(seed)
    values: list[tuple[float, float]] = []
    for _ in range(iterations):
        result = fit(rng.integers(0, len(x), size=len(x)))
        if result is not None and all(np.isfinite(result)):
            values.append(result)
    if point is None or not values:
        return {
            "slope": None,
            "intercept": None,
            "slope_ci95": [None, None],
            "intercept_ci95": [None, None],
        }
    array = np.asarray(values)
    return {
        "slope": point[0],
        "intercept": point[1],
        "slope_ci95": [float(value) for value in np.quantile(array[:, 0], [0.025, 0.975])],
        "intercept_ci95": [float(value) for value in np.quantile(array[:, 1], [0.025, 0.975])],
    }


def frozen_common_coordinate_metrics(
    coordinates: np.ndarray,
    *,
    protocols: Sequence[str] = CORE_PROTOCOL_NAMES,
    equivalence_band: float = FROZEN_EQUIVALENCE_BAND,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = SEED,
) -> dict[str, Any]:
    matrix = np.asarray(coordinates, dtype=np.float64)
    names = tuple(protocols)
    if names != CORE_PROTOCOL_NAMES:
        raise ValueError("Frozen common coordinate gate requires the exact core-7 order")
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError(f"Coordinate matrix has the wrong shape: {matrix.shape}")
    icc = absolute_agreement_icc(matrix)
    rng = np.random.default_rng(seed + 700)
    icc_samples: list[float] = []
    for _ in range(iterations):
        value = absolute_agreement_icc(
            matrix[rng.integers(0, len(matrix), size=len(matrix))]
        )
        if value is not None and np.isfinite(value):
            icc_samples.append(float(value))
    icc_ci: list[float | None] = [None, None]
    if icc_samples:
        icc_ci = [float(value) for value in np.quantile(icc_samples, [0.025, 0.975])]
    between_sd = float(np.std(matrix.mean(axis=1), ddof=1))
    within_sd = float(np.sqrt(np.mean(np.var(matrix, axis=1, ddof=1))))
    ratio = float(within_sd / between_sd) if between_sd > 1e-12 else math.inf
    reference = matrix[:, 0]
    comparisons: dict[str, Any] = {}
    for index, name in enumerate(names[1:], start=1):
        other = matrix[:, index]
        difference = other - reference
        mean_ci = _mean_ci(difference, iterations=iterations, seed=seed + 1000 + index)
        regression = _slope_ci(
            reference, other, iterations=iterations, seed=seed + 2000 + index
        )
        mean_pass = bool(
            mean_ci[0] is not None
            and max(abs(float(mean_ci[0])), abs(float(mean_ci[1])))
            <= equivalence_band
        )
        slope_ci = regression["slope_ci95"]
        slope_pass = bool(
            slope_ci[0] is not None
            and float(slope_ci[0]) >= 0.80
            and float(slope_ci[1]) <= 1.25
        )
        comparisons[name] = {
            "mean_difference": float(np.mean(difference)),
            "mean_difference_ci95": mean_ci,
            "mean_equivalent": mean_pass,
            **regression,
            "slope_equivalent": slope_pass,
            "coordinate_equivalent": bool(mean_pass and slope_pass),
        }
    components = {
        "icc_point": bool(icc is not None and icc >= 0.75),
        "icc_lower_ci": bool(icc_ci[0] is not None and float(icc_ci[0]) >= 0.60),
        "within_between_ratio": bool(np.isfinite(ratio) and ratio <= 0.50),
        "all_common_mean_equivalent": all(
            value["mean_equivalent"] for value in comparisons.values()
        ),
        "all_common_slope_equivalent": all(
            value["slope_equivalent"] for value in comparisons.values()
        ),
    }
    return {
        "reference_protocol": names[0],
        "equivalence_band": float(equivalence_band),
        "equivalence_band_source": "frozen Stage-10 value; not recomputed on confirmatory items",
        "common_icc_a1": icc,
        "common_icc_bootstrap_ci95": icc_ci,
        "between_item_sd": between_sd,
        "within_item_protocol_sd": within_sd,
        "within_between_sd_ratio": ratio if np.isfinite(ratio) else None,
        "comparisons": comparisons,
        "components": components,
        "passed": all(components.values()),
        "protocol_specific_calibration": False,
    }


def _resolve_hidden(output: Path, row: dict[str, Any]) -> Path:
    path = output / "hidden" / str(row["hidden_file"])
    if not path.is_file():
        raise FileNotFoundError(f"Confirmatory hidden file is missing: {path}")
    if sha256_file(path) != str(row["hidden_sha256"]):
        raise ValueError(f"Confirmatory hidden checksum mismatch: {path}")
    return path


def _completed_results(output: Path) -> tuple[list[dict[str, Any]], int]:
    terminal = _latest_rows(output / "results.jsonl")
    completed = [row for row in terminal if row.get("status") == "completed"]
    failed = len(terminal) - len(completed)
    return sorted(completed, key=lambda row: (_numeric_key(row["item_id"]), row["case_id"])), failed


def _lower_positive(metric: dict[str, Any]) -> bool:
    value = metric.get("spearman_ci95", [None, None])[0]
    return bool(value is not None and float(value) > 0.0)


def frozen_rank_gate_components(
    *,
    technical_passed: bool,
    source_rank_passed: bool,
    target_r2: float,
    target_association: dict[str, Any],
    protocol_rank: dict[str, dict[str, Any]],
) -> dict[str, bool]:
    """Apply the fully frozen confirmatory rank rule without data-dependent tuning."""

    missing = [name for name in JOINT_PROTOCOL_NAMES if name not in protocol_rank]
    if missing:
        raise ValueError(f"Frozen rank gate lacks protocols: {missing}")
    return {
        "technical_gate": bool(technical_passed),
        "source_stage10_rank_gate": bool(source_rank_passed),
        "shared_target_r2_positive": float(target_r2) > 0.0,
        "shared_target_spearman_lower_positive": _lower_positive(target_association),
        "all_core_protocol_rank_lower_positive": all(
            _lower_positive(protocol_rank[name]) for name in CORE_PROTOCOL_NAMES
        ),
        "all_random_order_holdout_rank_lower_positive": all(
            _lower_positive(protocol_rank[name]) for name in HOLDOUT_PROTOCOL_NAMES
        ),
    }


def _technical_audit(
    output: Path,
    rows: Sequence[dict[str, Any]],
    failed: int,
    cohort_manifest: dict[str, Any],
) -> dict[str, Any]:
    problems: list[str] = []
    if len(rows) != EXPECTED_COMPLETED_ITEMS:
        problems.append(f"completed_n={len(rows)}")
    if failed:
        problems.append(f"failed_n={failed}")
    if not cohort_manifest["endpoint_audit"]["item_isolation_passed"]:
        problems.append("development_item_overlap")
    total_forwards = 0
    prefix_failures: list[str] = []
    for row in rows:
        if row.get("answer_source") != "method_v2_confirmatory_results.answer_star":
            problems.append(f"answer_source:{row['case_id']}")
        if not row.get("answer_star_token_audit", {}).get("passed"):
            problems.append(f"answer_star_token_mapping:{row['case_id']}")
        if tuple(row.get("protocol_order", [])) != ALL_PROTOCOL_NAMES:
            problems.append(f"protocol_order:{row['case_id']}")
            continue
        protocols = row.get("protocols", {})
        if tuple(protocols) != ALL_PROTOCOL_NAMES:
            problems.append(f"protocol_payload:{row['case_id']}")
            continue
        total_forwards += int(row.get("formal_forward_count", 0))
        if any(not bool(protocols[name].get("hook_exactly_once")) for name in ALL_PROTOCOL_NAMES):
            problems.append(f"hook_count:{row['case_id']}")
        if any(float(protocols[name].get("injection_l2", math.inf)) != 0.0 for name in ALL_PROTOCOL_NAMES):
            problems.append(f"unexpected_injection:{row['case_id']}")
        prefix = protocols[POSTQUERY_PROTOCOL_NAME].get("prefix_through_answer_audit", {})
        if not prefix.get("passed"):
            prefix_failures.append(str(row["case_id"]))
        hidden_path = _resolve_hidden(output, row)
        with np.load(hidden_path, allow_pickle=False) as payload:
            names = tuple(str(value) for value in payload["protocols"].tolist())
            hidden = np.asarray(payload["hidden"])
        if names != ALL_PROTOCOL_NAMES or hidden.ndim != 2 or hidden.shape[0] != len(names):
            problems.append(f"hidden_schema:{row['case_id']}")
        if hidden.dtype != np.float32 or not np.isfinite(hidden).all():
            problems.append(f"hidden_values:{row['case_id']}")
    if prefix_failures:
        problems.append(f"postquery_prefix_failures={len(prefix_failures)}")
    expected_forwards = EXPECTED_COMPLETED_ITEMS * len(ALL_PROTOCOL_NAMES)
    if total_forwards != expected_forwards:
        problems.append(f"formal_forward_count={total_forwards}")
    return {
        "passed": not problems,
        "completed_n": len(rows),
        "failed_n": failed,
        "expected_n": EXPECTED_COMPLETED_ITEMS,
        "formal_forward_count": total_forwards,
        "expected_formal_forward_count": expected_forwards,
        "protocols_per_item": len(ALL_PROTOCOL_NAMES),
        "postquery_prefix_failure_case_ids": prefix_failures,
        "item_isolation_passed": cohort_manifest["endpoint_audit"]["item_isolation_passed"],
        "problems": problems,
    }


def build_endpoint_audit(
    cohort_manifest: dict[str, Any],
    results: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = list(results or [])
    prefix = [
        row["protocols"][POSTQUERY_PROTOCOL_NAME]["prefix_through_answer_audit"]
        for row in rows
        if row.get("status") == "completed"
        and POSTQUERY_PROTOCOL_NAME in row.get("protocols", {})
    ]
    token_audits = [
        row.get("answer_star_token_audit", {})
        for row in rows
        if row.get("status") == "completed"
    ]
    return {
        "status": "passed" if rows and all(value.get("passed") for value in prefix) else "preflight_passed" if not rows else "failed",
        "preflight": cohort_manifest["endpoint_audit"],
        "answer_field": "answer_star",
        "old_joint_final_answer_used": False,
        "completed_result_n": len(rows),
        "canonical_prefix_audit_n": len(prefix),
        "canonical_prefix_all_passed": bool(prefix) and all(value.get("passed") for value in prefix),
        "rendered_prefix_all_passed": bool(prefix) and all(
            value.get("checks", {}).get("rendered_prefix_through_answer_equal")
            for value in prefix
        ),
        "token_prefix_all_passed": bool(prefix) and all(
            value.get("checks", {}).get("token_prefix_through_answer_equal")
            for value in prefix
        ),
        "method_v2_hash_all_passed": bool(prefix) and all(
            value.get("matches_method_v2_teacher_forced_messages_hash")
            for value in prefix
        ),
        "answer_star_token_audit_n": len(token_audits),
        "answer_star_token_mapping_all_passed": bool(token_audits)
        and all(value.get("passed") for value in token_audits),
        "hidden_numeric_equality_tested": False,
        "hidden_numeric_equality_claimed": False,
        "bf16_shape_drift_note": (
            "The postquery branch has a longer total sequence. Exact message/rendered/token "
            "prefix equality through A* is audited, but BF16 kernel shape drift precludes a "
            "claim of numerically identical hidden states."
        ),
    }


def _artifact_manifest(output: Path, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    paths = [
        output / "run_config.json",
        output / "frozen_rule.json",
        output / "cohort_manifest.json",
        output / "provenance.json",
        output / "results.jsonl",
        output / "analysis.jsonl",
        output / "endpoint_audit.json",
        *sorted((output / "frozen_directions").glob("*.npz")),
        *[_resolve_hidden(output, row) for row in rows],
    ]
    smoke = output / "gpu_smoke.json"
    if smoke.is_file():
        paths.append(smoke)
    smoke_hidden = output / "gpu_smoke_hidden.npz"
    if smoke_hidden.is_file():
        paths.append(smoke_hidden)
    analysis_provenance = output / "analysis_rerun_provenance.json"
    if analysis_provenance.is_file():
        paths.append(analysis_provenance)
    entries = [
        {
            "path": str(path.relative_to(output)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in paths
    ]
    payload = {
        "format_version": 1,
        "files": entries,
        "aggregate_definition": "SHA256 of canonical JSON file/sha256/bytes entries",
        "aggregate_sha256": stable_hash(entries),
    }
    atomic_write_json(output / "artifact_manifest.json", payload)
    return payload


def _markdown(summary: dict[str, Any]) -> str:
    def number(value: Any, digits: int = 3) -> str:
        return "NA" if value is None else f"{float(value):.{digits}f}"

    rank = summary["frozen_rank_gate"]
    coordinate = summary["frozen_common_coordinate_gate"]
    post = summary["postquery_report_transfer"]
    lines = [
        "# Confirmatory Attribution Panel",
        "",
        f"- Status: `{summary['status']}`",
        f"- Confirmatory items: {summary['n']} (Stage-10 development overlap: 0).",
        f"- Formal forwards: {summary['technical_gate']['formal_forward_count']} / 988.",
        f"- Technical gate: **{'PASS' if summary['technical_gate']['passed'] else 'FAIL'}**.",
        f"- Frozen rank gate: **{'PASS' if rank['passed'] else 'FAIL'}**.",
        f"- Frozen common-coordinate gate: **{'PASS' if coordinate['passed'] else 'FAIL'}**.",
        f"- Postquery report transfer: **{'PASS' if post['passed'] else 'FAIL'}**.",
        "- Causal intervention: no; causal-mediator authorization: no.",
        "",
        "## Frozen rank transfer",
        "",
        f"Frozen target R² = {number(rank['shared_target']['r2'])}; Spearman = "
        f"{number(rank['shared_target']['association']['spearman'])} "
        f"[{number(rank['shared_target']['association']['spearman_ci95'][0])}, "
        f"{number(rank['shared_target']['association']['spearman_ci95'][1])}].",
        "",
        "| Protocol | Role | Spearman | 95% item-bootstrap CI |",
        "|---|---|---:|---:|",
    ]
    for name in JOINT_PROTOCOL_NAMES:
        value = rank["protocol_rank_transfer"][name]
        role = "core" if name in CORE_PROTOCOL_NAMES else "unseen mapping/order holdout"
        lines.append(
            f"| {name} | {role} | {number(value['spearman'])} | "
            f"[{number(value['spearman_ci95'][0])}, {number(value['spearman_ci95'][1])}] |"
        )
    metric = coordinate["metrics"]
    lines.extend(
        [
            "",
            "## Frozen common coordinate",
            "",
            f"ICC(A,1) = {number(metric['common_icc_a1'])} "
            f"[{number(metric['common_icc_bootstrap_ci95'][0])}, "
            f"{number(metric['common_icc_bootstrap_ci95'][1])}]; within/between SD ratio = "
            f"{number(metric['within_between_sd_ratio'])}; frozen equivalence band = "
            f"±{metric['equivalence_band']:.10f}.",
            "",
            "The band, directions, origins, scales, target transforms, and Ridge intercepts "
            "were copied from Stage 10 before these confirmatory outcomes were measured.",
            "",
            "## Branched post-answer continuation",
            "",
            f"Frozen answer-prefix coordinate ↔ later postquery report Spearman = "
            f"{number(post['frozen_prediction_vs_postquery_report']['spearman'])} "
            f"[{number(post['frozen_prediction_vs_postquery_report']['spearman_ci95'][0])}, "
            f"{number(post['frozen_prediction_vs_postquery_report']['spearman_ci95'][1])}].",
            "",
            "The canonical messages plus rendered/token prefix through A* are checked exactly. "
            "No numerical hidden-state equality is claimed because the longer branch changes "
            "BF16 kernel shape.",
            "",
            "## Interpretation limit",
            "",
            summary["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)


def analyze_confirmatory_panel(output_dir: str | Path) -> dict[str, Any]:
    """Recompute every confirmatory statistic from hidden/logit artifacts."""

    output = Path(output_dir).resolve()
    cohort_manifest = json.loads(
        (output / "cohort_manifest.json").read_text(encoding="utf-8")
    )
    frozen = FrozenDirectionRepository(output)
    rows, failed = _completed_results(output)
    technical = _technical_audit(output, rows, failed, cohort_manifest)
    if not rows:
        raise ValueError("No completed confirmatory attribution rows are available")

    n = len(rows)
    hidden = np.empty((n, len(ALL_PROTOCOL_NAMES), 0), dtype=np.float64)
    hidden_values: list[np.ndarray] = []
    semantic = np.full((n, len(ALL_PROTOCOL_NAMES)), np.nan, dtype=np.float64)
    predictions = np.full_like(semantic, np.nan)
    coordinates = np.full_like(semantic, np.nan)
    targets = np.full(n, np.nan, dtype=np.float64)
    analysis_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        path = _resolve_hidden(output, row)
        with np.load(path, allow_pickle=False) as payload:
            names = tuple(str(value) for value in payload["protocols"].tolist())
            values = np.asarray(payload["hidden"], dtype=np.float64)
        if names != ALL_PROTOCOL_NAMES:
            raise ValueError(f"Hidden protocol order drift: {row['case_id']}")
        hidden_values.append(values)
        direction = frozen.get(int(row["fold"]))
        for protocol_index, name in enumerate(names):
            semantic[row_index, protocol_index] = float(
                row["protocols"][name]["semantic_imageward_score"]
            )
            predictions[row_index, protocol_index] = direction.predict(
                values[protocol_index]
            )
            coordinates[row_index, protocol_index] = direction.coordinate(
                values[protocol_index]
            )
        targets[row_index] = direction.transform_target(
            semantic[row_index, : len(CORE_PROTOCOL_NAMES)]
        )
        analysis_rows.append(
            {
                "case_id": row["case_id"],
                "item_id": row["item_id"],
                "fold": int(row["fold"]),
                "answer_star": row["answer_star"],
                "frozen_shared_target": float(targets[row_index]),
                "core_consensus_prediction": float(
                    np.mean(predictions[row_index, : len(CORE_PROTOCOL_NAMES)])
                ),
                "protocols": {
                    name: {
                        "semantic_imageward_score": float(semantic[row_index, index]),
                        "frozen_prediction": float(predictions[row_index, index]),
                        "frozen_coordinate": float(coordinates[row_index, index]),
                    }
                    for index, name in enumerate(ALL_PROTOCOL_NAMES)
                },
            }
        )
    hidden = np.stack(hidden_values)
    if hidden.shape[:2] != (n, len(ALL_PROTOCOL_NAMES)):
        raise ValueError(f"Stacked hidden panel is malformed: {hidden.shape}")
    write_jsonl_atomic(output / "analysis.jsonl", analysis_rows)

    core_count = len(CORE_PROTOCOL_NAMES)
    core_consensus = predictions[:, :core_count].mean(axis=1)
    target_association = bootstrap_association(
        core_consensus, targets, seed=SEED + 10
    )
    protocol_rank = {
        name: bootstrap_association(
            predictions[:, index], semantic[:, index], seed=SEED + 100 + index
        )
        for index, name in enumerate(JOINT_PROTOCOL_NAMES)
    }
    target_r2 = float(r2_score(targets, core_consensus))
    rank_components = frozen_rank_gate_components(
        technical_passed=technical["passed"],
        source_rank_passed=bool(frozen.rule["source_rank_gate_passed"]),
        target_r2=target_r2,
        target_association=target_association,
        protocol_rank=protocol_rank,
    )
    rank_gate = {
        "passed": all(rank_components.values()),
        "components": rank_components,
        "shared_target": {
            "r2": target_r2,
            "mae": float(mean_absolute_error(targets, core_consensus)),
            "association": target_association,
        },
        "protocol_rank_transfer": protocol_rank,
        "fit": "fully frozen Stage-10 fold model; no confirmatory fitting or calibration",
    }

    coordinate_metrics = frozen_common_coordinate_metrics(
        coordinates[:, :core_count],
        equivalence_band=float(
            frozen.rule["common_coordinate_gate"]["equivalence_band"]
        ),
    )
    coordinate_components = {
        "technical_gate": technical["passed"],
        "frozen_rank_gate": rank_gate["passed"],
        "source_stage10_common_component": bool(
            frozen.rule["source_common_coordinate_component_passed"]
        ),
        **coordinate_metrics["components"],
    }
    coordinate_gate = {
        "passed": all(coordinate_components.values()),
        "components": coordinate_components,
        "metrics": coordinate_metrics,
        "scope": "seven core common protocols only",
        "random_order_coordinates_gate_bearing": False,
    }

    post_index = ALL_PROTOCOL_NAMES.index(POSTQUERY_PROTOCOL_NAME)
    common9_index = ALL_PROTOCOL_NAMES.index("common_9_ordered")
    post_prediction_report = bootstrap_association(
        predictions[:, post_index], semantic[:, post_index], seed=SEED + 400
    )
    post_prediction_target = bootstrap_association(
        predictions[:, post_index], targets, seed=SEED + 401
    )
    joint_post_report = bootstrap_association(
        semantic[:, common9_index], semantic[:, post_index], seed=SEED + 402
    )
    post_components = {
        "technical_gate": technical["passed"],
        "canonical_prefix_audit": not technical["postquery_prefix_failure_case_ids"],
        "frozen_prediction_vs_report_lower_positive": _lower_positive(
            post_prediction_report
        ),
        "frozen_prediction_vs_core_target_lower_positive": _lower_positive(
            post_prediction_target
        ),
        "joint_vs_postquery_report_lower_positive": _lower_positive(joint_post_report),
    }
    postquery = {
        "passed": all(post_components.values()),
        "components": post_components,
        "frozen_prediction_vs_postquery_report": post_prediction_report,
        "frozen_prediction_vs_core_target": post_prediction_target,
        "joint_common9_report_vs_postquery_report": joint_post_report,
        "paired_coordinate_shift_postquery_minus_joint_common9": {
            "mean": float(np.mean(coordinates[:, post_index] - coordinates[:, common9_index])),
            "ci95": _mean_ci(
                coordinates[:, post_index] - coordinates[:, common9_index],
                iterations=BOOTSTRAP_ITERATIONS,
                seed=SEED + 403,
            ),
        },
        "coordinate_gate_bearing": False,
        "hidden_site": (
            "answer-prefix state inside the longer postquery branch; semantic report logits "
            "are read at the later Source Attribution continuation"
        ),
        "hidden_numeric_equality_claimed": False,
    }

    endpoint = build_endpoint_audit(cohort_manifest, rows)
    atomic_write_json(output / "endpoint_audit.json", endpoint)
    artifacts = _artifact_manifest(output, rows)
    if rank_gate["passed"] and coordinate_gate["passed"] and postquery["passed"]:
        classification = "frozen_rank_common_coordinate_and_postquery_transfer_confirmed"
    elif rank_gate["passed"] and coordinate_gate["passed"]:
        classification = "frozen_rank_and_common_coordinate_confirmed_postquery_not_confirmed"
    elif rank_gate["passed"]:
        classification = "frozen_rank_transfer_only"
    else:
        classification = "frozen_attribution_candidate_not_confirmed"
    interpretation = (
        "This panel tests whether the Stage-10 attribution readout transfers to wholly held-out "
        "method-v2 items, unseen label mappings, and unseen row orders. A pass supports a "
        "protocol-transportable report/readout component at the stated scope. It does not show "
        "that the coordinate causally mediates Actual Source Reliance or that verbal SA is a "
        "faithful instance-wise behavioral readout."
    )
    summary = {
        "title": "Frozen Confirmatory Attribution Panel",
        "status": "completed" if technical["passed"] else "technical_gate_failed",
        "n": n,
        "failed": failed,
        "classification": classification,
        "technical_gate": technical,
        "frozen_rank_gate": rank_gate,
        "frozen_common_coordinate_gate": coordinate_gate,
        "postquery_report_transfer": postquery,
        "endpoint_audit": endpoint,
        "artifact_aggregate_sha256": artifacts["aggregate_sha256"],
        "formal_forward_count": technical["formal_forward_count"],
        "development_item_overlap": [],
        "causal_intervention": False,
        "causal_mediator_authorized": False,
        "claim_scope": "confirmatory report/readout transport; not causal mediation",
        "interpretation": interpretation,
        "frozen_rule_fingerprint": frozen.rule["rule_fingerprint"],
    }
    atomic_write_json(output / "summary.json", summary)
    atomic_write_text(output / "summary.md", _markdown(summary))
    return summary


def gpu_smoke(
    runtime: Stage3Runtime,
    row: dict[str, Any],
    output_dir: str | Path,
    *,
    deadline: Callable[[], None],
) -> dict[str, Any]:
    """Run one complete protocol panel without adding a formal result row."""

    output = Path(output_dir).resolve()
    direction = FrozenDirectionRepository(output).get(int(row["fold"]))
    hidden = output / "gpu_smoke_hidden.npz"
    measured = measure_confirmatory_case(
        runtime, row, direction, hidden, deadline=deadline
    )
    protocols = measured["protocols"]
    prefix = protocols[POSTQUERY_PROTOCOL_NAME]["prefix_through_answer_audit"]
    return {
        "status": "passed",
        "case_id": row["case_id"],
        "answer_star": row["answer_star"],
        "answer_source": measured["answer_source"],
        "forward_count": measured["formal_forward_count"],
        "protocol_order": measured["protocol_order"],
        "answer_star_token_audit": measured["answer_star_token_audit"],
        "tokenizer_mapping_validated": bool(
            measured["answer_star_token_audit"]["passed"]
        ),
        "all_hooks_exactly_once": all(
            protocols[name]["hook_exactly_once"] for name in ALL_PROTOCOL_NAMES
        ),
        "all_zero_injection": all(
            float(protocols[name]["injection_l2"]) == 0.0 for name in ALL_PROTOCOL_NAMES
        ),
        "postquery_prefix_audit": prefix,
        "hidden_file": hidden.name,
        "hidden_sha256": sha256_file(hidden),
        "hidden_shape": measured["hidden_shape"],
        "causal_intervention": False,
    }


def immutable_json(path: str | Path, value: dict[str, Any]) -> None:
    destination = Path(path)
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != value:
            raise ValueError(f"Immutable artifact differs; refusing overwrite: {destination}")
        return
    atomic_write_json(destination, value)
