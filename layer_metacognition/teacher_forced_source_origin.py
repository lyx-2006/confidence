"""Teacher-forced source-attribution causal experiments.

The module intentionally keeps cohort construction, pairing, metrics, state
lifecycle, and model execution in one experiment-specific implementation.  It
reuses the repository's prompt, parsing, multimodal preparation, token locators,
and restricted SA scorer rather than introducing parallel definitions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from confidence_test.answer_metrics import normalize_answer
from confidence_test.dataset_utils import load_evaluation_cases
from confidence_test.source_attribution_analyzer import (
    SourceAttributionAnalyzer,
    parse_parallel_source_output,
)
from confidence_test.source_attribution_schema import (
    ASSISTANT_SOURCE_ATTRIBUTION_PREFILL,
)
from confidence_test.source_attribution_variants import SourcePromptVariant
from layer_metacognition.hidden_state_store import (
    append_jsonl,
    atomic_write_json,
    load_jsonl,
)
from layer_metacognition.model_adapter import (
    HiddenStateReplacement,
    HiddenStateReplacementHook,
    LanguageModules,
)
from layer_metacognition.token_positions import (
    locate_field_value_span,
    locate_image_pad_span,
    locate_marker_in_assistant,
    locate_token_after_field,
)
from layer_metacognition.token_spans import build_rendered_alignment


FORMAT_VERSION = 1
CONFLICT_CONDITIONS = ("conflict_easy", "conflict_hard")
INTERVENTION_LAYERS = (12, 16, 20, 24, 26)
EVIDENCE_TARGETS = ("image", "text_clue", "both")
STATE_INTERVENTIONS = (
    "ac_natural",
    "panl_only",
    "ac_panl_clamp_clean",
)
DECISION_SIDES = ("follows_text", "follows_image")
SELF_SWAP_TOLERANCE = 1e-4


@dataclass(frozen=True)
class PreparedTeacherContext:
    context_id: str
    prompt: str
    image_path: str
    forced_answer: str
    assistant_text: str
    rendered: str
    inputs: Any
    positions: dict[str, Any]


def configuration_fingerprint(configuration: dict[str, Any]) -> str:
    payload = json.dumps(
        configuration,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _item_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    raw = str(record["item_id"])
    item = (0, int(raw)) if raw.isdigit() else (1, raw)
    return (*item, int(record["prior_index"]), str(record["condition"]))


def load_causal_candidates(
    *,
    experiment_dir: str | Path,
    dataset: str | Path,
    fallback_null_path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join completed V4 conflict baselines to dataset cases without SA filtering."""

    experiment_path = Path(experiment_dir).resolve()
    baseline_path = experiment_path / "results.jsonl"
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Baseline results do not exist: {baseline_path}")
    cases, dataset_metadata = load_evaluation_cases(
        dataset,
        fallback_null_path=fallback_null_path,
    )
    by_key = {(case.item_id, case.prior_index): case for case in cases}
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    with baseline_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            condition = str(row.get("condition", ""))
            case_id = str(row.get("case_id", ""))
            if not (
                row.get("status") == "completed"
                and row.get("version") == "v4"
                and row.get("attribution_mode") == "joint"
                and condition in CONFLICT_CONDITIONS
                and case_id.endswith("__v4__joint")
            ):
                continue
            if case_id in seen:
                raise ValueError(
                    f"Duplicate eligible baseline case at line {line_number}: {case_id}"
                )
            item_id = str(row.get("item_id", ""))
            prior_index = int(row.get("prior_index", -1))
            evaluation = by_key.get((item_id, prior_index))
            if evaluation is None:
                raise ValueError(f"Dataset has no item/prior for baseline {case_id}")
            generated = row.get("generated")
            if not isinstance(generated, dict):
                continue
            exact_answer = generated.get("current_answer")
            normalized = normalize_answer(exact_answer)
            if normalized == evaluation.text_answer:
                decision_side = "follows_text"
            elif normalized == evaluation.conflict_answer:
                decision_side = "follows_image"
            else:
                continue
            source = generated.get("source_attribution")
            answer_result = generated.get("current_answer_result")
            if not isinstance(source, dict) or not isinstance(answer_result, dict):
                continue
            conflict_input = evaluation.conditions[condition]
            difficulty = condition.removeprefix("conflict_")
            donor_condition = f"consistent_{difficulty}"
            donor_input = evaluation.conditions[donor_condition]
            if (
                conflict_input.error
                or donor_input.error
                or not conflict_input.resolved_image_path
                or not donor_input.resolved_image_path
            ):
                continue
            if not isinstance(exact_answer, str) or normalized is None:
                continue
            free_soft = source.get("soft_image_score")
            if free_soft is None or not math.isfinite(float(free_soft)):
                continue
            output.append(
                {
                    "case_id": case_id,
                    "item_id": item_id,
                    "item_order": int(evaluation.item_order),
                    "prior_index": prior_index,
                    "prior_bin": evaluation.prior_bin,
                    "condition": condition,
                    "difficulty": difficulty,
                    "decision_side": decision_side,
                    "question": evaluation.question,
                    "text_clue": evaluation.text_clue,
                    "answer_classes": list(evaluation.answer_classes),
                    "forced_answer": exact_answer,
                    "normalized_answer": normalized,
                    "text_answer": evaluation.text_answer,
                    "image_answer": evaluation.conflict_answer,
                    "recipient_image_path": str(conflict_input.resolved_image_path),
                    "donor_condition": donor_condition,
                    "donor_image_path": str(donor_input.resolved_image_path),
                    "free_generation": {
                        "raw_output": answer_result.get("raw_output"),
                        "hard_label": source.get("hard_label"),
                        "parsed_label": source.get("parsed_label"),
                        "soft_image_score": float(free_soft),
                        "class_probabilities": source.get("class_probabilities"),
                        "entropy": source.get("source_entropy"),
                    },
                }
            )
            seen.add(case_id)
    if not output:
        raise ValueError("No completed eligible V4 conflict baselines were found")
    output.sort(key=_item_sort_key)
    return output, dataset_metadata


def select_balanced_cohort(
    candidates: Sequence[dict[str, Any]],
    *,
    cases_per_cell: int = 25,
) -> list[dict[str, Any]]:
    """Select side/difficulty cells while greedily maximizing item diversity."""

    if cases_per_cell < 1:
        raise ValueError("cases_per_cell must be positive")
    selected: list[dict[str, Any]] = []
    used_items: set[str] = set()
    cells = [
        (side, condition)
        for side in DECISION_SIDES
        for condition in CONFLICT_CONDITIONS
    ]
    for side, condition in cells:
        available = [
            dict(row)
            for row in candidates
            if row["decision_side"] == side and row["condition"] == condition
        ]
        if len(available) < cases_per_cell:
            raise ValueError(
                f"Need {cases_per_cell} cases for {side}/{condition}, "
                f"found {len(available)}"
            )
        cell: list[dict[str, Any]] = []
        cell_items: set[str] = set()
        while len(cell) < cases_per_cell:
            remaining = [row for row in available if row not in cell]
            remaining.sort(
                key=lambda row: (
                    str(row["item_id"]) in used_items,
                    str(row["item_id"]) in cell_items,
                    _item_sort_key(row),
                )
            )
            chosen = remaining[0]
            cell.append(chosen)
            cell_items.add(str(chosen["item_id"]))
            used_items.add(str(chosen["item_id"]))
        selected.extend(cell)
    selected.sort(key=_item_sort_key)
    return selected


def cohort_manifest_payload(
    cohort: Sequence[dict[str, Any]],
    *,
    source_candidate_count: int,
    cases_per_cell: int,
    selection_profile: str,
) -> dict[str, Any]:
    counts = Counter(
        (str(row["decision_side"]), str(row["condition"])) for row in cohort
    )
    return {
        "format_version": FORMAT_VERSION,
        "selection_profile": selection_profile,
        "selection_uses_source_attribution": False,
        "source_candidate_count": int(source_candidate_count),
        "case_count": len(cohort),
        "unique_item_count": len({str(row["item_id"]) for row in cohort}),
        "cases_per_cell": int(cases_per_cell),
        "counts": {
            f"{side}|{condition}": counts[(side, condition)]
            for side in DECISION_SIDES
            for condition in CONFLICT_CONDITIONS
        },
        "cases": list(cohort),
    }


def state_pair_tier(left: dict[str, Any], right: dict[str, Any]) -> int:
    same_item = str(left["item_id"]) == str(right["item_id"])
    same_condition = str(left["condition"]) == str(right["condition"])
    if same_item and same_condition:
        return 0
    if same_item:
        return 1
    if same_condition:
        return 2
    return 3


def select_state_pairs(
    clean_cases: Sequence[dict[str, Any]],
    *,
    min_sa_gap: float = 0.15,
    max_pairs: int = 30,
) -> list[dict[str, Any]]:
    """Deterministically greedily match disjoint same-answer/same-side pairs."""

    if min_sa_gap < 0 or not math.isfinite(min_sa_gap):
        raise ValueError("min_sa_gap must be finite and non-negative")
    if max_pairs < 0:
        raise ValueError("max_pairs must be non-negative")
    edges: list[tuple[Any, ...]] = []
    ordered = sorted(clean_cases, key=_item_sort_key)
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if (
                left["normalized_answer"] != right["normalized_answer"]
                or left["decision_side"] != right["decision_side"]
            ):
                continue
            left_sa = float(left["clean_teacher_forced"]["soft_image_score"])
            right_sa = float(right["clean_teacher_forced"]["soft_image_score"])
            gap = abs(left_sa - right_sa)
            if gap + 1e-12 < min_sa_gap:
                continue
            low, high = (left, right) if left_sa <= right_sa else (right, left)
            tier = state_pair_tier(left, right)
            edges.append(
                (
                    tier,
                    -gap,
                    str(low["case_id"]),
                    str(high["case_id"]),
                    low,
                    high,
                    gap,
                )
            )
    edges.sort(key=lambda edge: edge[:4])
    used: set[str] = set()
    pairs: list[dict[str, Any]] = []
    for tier, _negative_gap, _low_id, _high_id, low, high, gap in edges:
        if len(pairs) >= max_pairs:
            break
        if low["case_id"] in used or high["case_id"] in used:
            continue
        pair_id = f"state_pair_{len(pairs):03d}"
        pairs.append(
            {
                "pair_id": pair_id,
                "match_tier": int(tier),
                "match_tier_name": (
                    "same_item_same_condition"
                    if tier == 0
                    else "same_item"
                    if tier == 1
                    else "same_condition"
                    if tier == 2
                    else "relaxed_condition"
                ),
                "decision_side": low["decision_side"],
                "normalized_answer": low["normalized_answer"],
                "sa_gap": float(gap),
                "low_case_id": low["case_id"],
                "low_sa": float(
                    low["clean_teacher_forced"]["soft_image_score"]
                ),
                "high_case_id": high["case_id"],
                "high_sa": float(
                    high["clean_teacher_forced"]["soft_image_score"]
                ),
            }
        )
        used.update((str(low["case_id"]), str(high["case_id"])))
    return pairs


def aligned_answer_force_delta(
    raw_delta: float,
    opposite_forced_side: str,
) -> float:
    if opposite_forced_side == "image":
        return float(raw_delta)
    if opposite_forced_side == "text":
        return -float(raw_delta)
    raise ValueError(f"Unknown forced side: {opposite_forced_side!r}")


def donor_metrics(
    *,
    recipient_sa: float,
    patched_sa: float,
    donor_sa: float,
) -> dict[str, Any]:
    delta = float(patched_sa) - float(recipient_sa)
    desired = float(donor_sa) - float(recipient_sa)
    return {
        "delta_sa": delta,
        "aligned_delta_sa": delta * (1.0 if desired > 0 else -1.0 if desired < 0 else 0.0),
        "directional": bool(delta * desired > 0.0),
        "donor_pull": bool(
            abs(float(patched_sa) - float(donor_sa))
            < abs(float(recipient_sa) - float(donor_sa)) - 1e-12
        ),
        "donor_gap": desired,
    }


def self_swap_validation(
    clean: dict[str, Any],
    patched: dict[str, Any],
    *,
    tolerance: float = SELF_SWAP_TOLERANCE,
) -> dict[str, Any]:
    clean_probs = [float(value) for value in clean["class_probabilities"]]
    patched_probs = [float(value) for value in patched["class_probabilities"]]
    if len(clean_probs) != len(patched_probs):
        raise ValueError("Self-swap class probability lengths differ")
    soft_error = abs(
        float(clean["soft_image_score"]) - float(patched["soft_image_score"])
    )
    probability_error = max(
        (abs(left - right) for left, right in zip(clean_probs, patched_probs)),
        default=0.0,
    )
    hard_match = str(clean["hard_label"]) == str(patched["hard_label"])
    return {
        "passed": bool(
            hard_match
            and soft_error <= tolerance
            and probability_error <= tolerance
        ),
        "tolerance": float(tolerance),
        "hard_match": hard_match,
        "soft_abs_error": soft_error,
        "class_probability_max_abs_error": probability_error,
    }


def intervention_key(*parts: Any) -> str:
    return "|".join(str(value) for value in parts)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class StreamingStateStore:
    """Transient per-context CPU-FP16 tensors with a persistent lifecycle index."""

    def __init__(self, output_dir: str | Path) -> None:
        self.root = Path(output_dir) / "precomputed_states"
        self.core_dir = self.root / "core"
        self.span_dir = self.root / "spans"
        self.index_path = self.root / "index.json"
        self.root.mkdir(parents=True, exist_ok=True)
        if self.index_path.is_file():
            self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
        else:
            self.index = {
                "format_version": FORMAT_VERSION,
                "retention_policy": "stream_delete",
                "contexts": {},
            }
            self._commit()

    @staticmethod
    def _safe_name(context_id: str) -> str:
        digest = hashlib.sha256(context_id.encode("utf-8")).hexdigest()[:16]
        return f"context_{digest}"

    def _commit(self) -> None:
        atomic_write_json(self.index_path, self.index)

    def save(
        self,
        *,
        context_id: str,
        metadata: dict[str, Any],
        core: dict[str, torch.Tensor],
        spans: dict[str, torch.Tensor],
    ) -> None:
        name = self._safe_name(context_id)
        core_path = self.core_dir / f"{name}.pt"
        span_path = self.span_dir / f"{name}.pt"
        core_payload = {
            "format_version": FORMAT_VERSION,
            "context_id": context_id,
            "states": {
                key: value.detach().to(device="cpu", dtype=torch.float16).contiguous()
                for key, value in core.items()
            },
        }
        span_payload = {
            "format_version": FORMAT_VERSION,
            "context_id": context_id,
            "states": {
                key: value.detach().to(device="cpu", dtype=torch.float16).contiguous()
                for key, value in spans.items()
            },
        }
        _atomic_torch_save(core_path, core_payload)
        _atomic_torch_save(span_path, span_payload)
        entry = {
            "context_id": context_id,
            "metadata": metadata,
            "core": {
                "path": str(core_path.relative_to(self.root)),
                "deleted": False,
                "shapes": {
                    key: list(value.shape) for key, value in core_payload["states"].items()
                },
            },
            "spans": {
                "path": str(span_path.relative_to(self.root)),
                "deleted": False,
                "shapes": {
                    key: list(value.shape) for key, value in span_payload["states"].items()
                },
            },
        }
        self.index["contexts"][context_id] = entry
        self._commit()

    def exists(self, context_id: str, kind: str) -> bool:
        entry = self.index["contexts"].get(context_id, {}).get(kind)
        return bool(
            isinstance(entry, dict)
            and not entry.get("deleted")
            and (self.root / str(entry.get("path"))).is_file()
        )

    def load(self, context_id: str, kind: str) -> dict[str, torch.Tensor]:
        if kind not in {"core", "spans"}:
            raise ValueError(f"Unknown state kind: {kind}")
        entry = self.index["contexts"].get(context_id)
        if not isinstance(entry, dict) or not self.exists(context_id, kind):
            raise KeyError(f"No live {kind} state for {context_id!r}")
        payload = _torch_load(self.root / entry[kind]["path"])
        if payload.get("context_id") != context_id:
            raise ValueError(f"State context mismatch for {context_id!r}")
        return {str(key): value for key, value in payload["states"].items()}

    def delete(self, context_id: str, kind: str, *, consumed_by: str) -> None:
        entry = self.index["contexts"].get(context_id)
        if not isinstance(entry, dict) or kind not in entry:
            return
        path = self.root / str(entry[kind]["path"])
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        entry[kind]["deleted"] = True
        entry[kind]["consumed_by"] = consumed_by
        self._commit()

    def live_tensor_files(self) -> list[str]:
        return [
            str(path.relative_to(self.root))
            for path in sorted(self.root.rglob("*.pt"))
        ]


def _stack_captured(
    captured: dict[str, dict[int, torch.Tensor]],
    num_layers: int,
) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for name, by_layer in captured.items():
        if set(by_layer) != set(range(num_layers)):
            raise RuntimeError(
                f"Clean capture for {name!r} missed decoder layers: "
                f"captured={sorted(by_layer)}"
            )
        output[name] = torch.stack(
            [by_layer[layer] for layer in range(num_layers)], dim=0
        ).contiguous()
    return output


class CleanStateCaptureHook:
    """Capture only named token/span slices from every decoder block."""

    def __init__(
        self,
        modules: LanguageModules,
        *,
        core_positions: dict[str, tuple[int, ...]],
        span_positions: dict[str, tuple[int, ...]],
        prefill_sequence_length: int,
    ) -> None:
        self.modules = modules
        self.core_positions = core_positions
        self.span_positions = span_positions
        self.prefill_sequence_length = int(prefill_sequence_length)
        self.core: dict[str, dict[int, torch.Tensor]] = {
            name: {} for name in core_positions
        }
        self.spans: dict[str, dict[int, torch.Tensor]] = {
            name: {} for name in span_positions
        }
        self._handles: list[Any] = []

    def _capture(self, layer_index: int, output: Any) -> None:
        tensor = output[0] if isinstance(output, tuple) and output else output
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            raise TypeError("Decoder output is not a rank-3 hidden tensor")
        if int(tensor.shape[1]) != self.prefill_sequence_length:
            return
        for name, positions in self.core_positions.items():
            if layer_index not in self.core[name]:
                self.core[name][layer_index] = tensor[
                    0, list(positions), :
                ].detach().to(device="cpu", dtype=torch.float16).contiguous()
        for name, positions in self.span_positions.items():
            if layer_index not in self.spans[name]:
                self.spans[name][layer_index] = tensor[
                    0, list(positions), :
                ].detach().to(device="cpu", dtype=torch.float16).contiguous()

    def __enter__(self) -> "CleanStateCaptureHook":
        for layer_index, layer in enumerate(self.modules.language_layers):
            self._handles.append(
                layer.register_forward_hook(
                    lambda _module, _args, output, index=layer_index: self._capture(
                        index, output
                    )
                )
            )
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def stacked(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        return (
            _stack_captured(self.core, self.modules.num_hidden_layers),
            _stack_captured(self.spans, self.modules.num_hidden_layers),
        )


def build_v4_prompt(case: dict[str, Any], variant: SourcePromptVariant) -> str:
    return variant.v4_joint_prompt.format(
        question=case["question"],
        text_clue=case["text_clue"],
        source_classes=variant.class_text,
    )


def forced_assistant_text(answer: str) -> str:
    return f"**Answer**: {answer}\n{ASSISTANT_SOURCE_ATTRIBUTION_PREFILL}"


def prepare_teacher_context(
    *,
    joint_generator: Any,
    context_id: str,
    prompt: str,
    image_path: str,
    forced_answer: str,
    text_clue: str,
) -> PreparedTeacherContext:
    assistant_text = forced_assistant_text(forced_answer)
    _messages, rendered, inputs = joint_generator.prepare_inputs(
        prompt,
        image_path,
        assistant_text=assistant_text,
    )
    tokenizer = joint_generator.tokenizer
    processed_ids = [int(value) for value in inputs.input_ids[0].tolist()]
    alignment = build_rendered_alignment(
        tokenizer,
        rendered,
        inputs.input_ids,
        inputs.attention_mask,
    )
    common = {
        "position_map": alignment.rendered_to_processed,
        "processed_ids": processed_ids,
    }
    ac = locate_marker_in_assistant(
        tokenizer,
        alignment.rendered_ids,
        assistant_text,
        "**Answer**:",
        name="ac",
        assistant_occurrence="final_suffix",
        **common,
    )
    sac = locate_marker_in_assistant(
        tokenizer,
        alignment.rendered_ids,
        assistant_text,
        ASSISTANT_SOURCE_ATTRIBUTION_PREFILL,
        name="sac",
        assistant_occurrence="final_suffix",
        **common,
    )
    panl = locate_token_after_field(
        tokenizer,
        alignment.rendered_ids,
        "**Answer**:",
        forced_answer,
        separator=" ",
        name="panl",
        **common,
    )
    text = locate_field_value_span(
        tokenizer,
        alignment.rendered_ids,
        "Text clue:",
        text_clue,
        separator="\n",
        name="text_clue",
        **common,
    )
    image = locate_image_pad_span(tokenizer, processed_ids)
    return PreparedTeacherContext(
        context_id=context_id,
        prompt=prompt,
        image_path=image_path,
        forced_answer=forced_answer,
        assistant_text=assistant_text,
        rendered=rendered,
        inputs=inputs,
        positions={
            "ac": ac,
            "panl": panl,
            "sac": sac,
            "image": image,
            "text_clue": text,
        },
    )


def _position_tuple(record: dict[str, Any]) -> tuple[int, ...]:
    if "position" in record:
        return (int(record["position"]),)
    span = record.get("span")
    if not isinstance(span, list) or len(span) != 2:
        raise ValueError(f"Position record has no position/span: {record}")
    start, end = (int(span[0]), int(span[1]))
    if end <= start:
        raise ValueError(f"Position record has an empty span: {record}")
    return tuple(range(start, end))


def _compact_positions(positions: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, record in positions.items():
        if "position" in record:
            output[name] = {
                "position": int(record["position"]),
                "token_id": record.get("token_id"),
                "token_text": record.get("token_text"),
            }
        else:
            output[name] = {
                "span": [int(value) for value in record["span"]],
                "token_count": int(record["span"][1]) - int(record["span"][0]),
                "token_text": record.get("token_text"),
            }
    return output


def run_teacher_forced_source(
    *,
    joint_generator: Any,
    source_analyzer: SourceAttributionAnalyzer,
    modules: LanguageModules,
    context: PreparedTeacherContext,
    replacements: Sequence[HiddenStateReplacement] | None = None,
    capture_clean: bool = False,
    max_new_tokens: int = 4,
) -> tuple[dict[str, Any], dict[str, torch.Tensor] | None, dict[str, torch.Tensor] | None, dict[str, Any] | None]:
    """Generate SA from the forced prefix, optionally capturing or replacing state."""

    prefill_length = int(context.inputs.input_ids.shape[1])
    capture = None
    if capture_clean:
        capture = CleanStateCaptureHook(
            modules,
            core_positions={
                name: _position_tuple(context.positions[name])
                for name in ("ac", "panl", "sac")
            },
            span_positions={
                name: _position_tuple(context.positions[name])
                for name in ("image", "text_clue")
            },
            prefill_sequence_length=prefill_length,
        )
    replacement_hook = (
        HiddenStateReplacementHook(
            modules,
            replacements=list(replacements),
            prefill_sequence_length=prefill_length,
        )
        if replacements
        else None
    )
    started = time.perf_counter()
    with (
        capture if capture is not None else nullcontext()
    ), (
        replacement_hook if replacement_hook is not None else nullcontext()
    ), torch.inference_mode():
        generated = joint_generator.model.generate(
            **context.inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
    if not generated.scores:
        raise RuntimeError("Teacher-forced Source Attribution returned no logits")
    input_length = int(context.inputs.input_ids.shape[1])
    continuation = joint_generator.tokenizer.decode(
        generated.sequences[0, input_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    raw_output = ASSISTANT_SOURCE_ATTRIBUTION_PREFILL + continuation
    parsed_label = parse_parallel_source_output(
        raw_output,
        source_analyzer.source_classes,
    )
    scored = source_analyzer.score_vocab_logits(
        generated.scores[0][0],
        raw_output=raw_output,
        parsed_label=parsed_label,
    ).to_dict()
    score = {
        "raw_output": raw_output,
        "parsed_label": parsed_label,
        "parse_success": parsed_label is not None,
        "hard_label": str(scored["hard_label"]),
        "hard_image_score": float(scored["hard_image_score"]),
        "soft_image_score": float(scored["soft_image_score"]),
        "class_probabilities": [
            float(value) for value in scored["class_probabilities"]
        ],
        "entropy": float(scored["source_entropy"]),
        "normalized_entropy": float(scored["normalized_source_entropy"]),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    core = spans = None
    if capture is not None:
        core, spans = capture.stacked()
    diagnostics = (
        replacement_hook.diagnostics() if replacement_hook is not None else None
    )
    del generated
    return score, core, spans, diagnostics


def context_metadata(
    context: PreparedTeacherContext,
    modules: LanguageModules,
) -> dict[str, Any]:
    return {
        "prompt_sha256": hashlib.sha256(context.prompt.encode("utf-8")).hexdigest(),
        "image_path": context.image_path,
        "forced_answer": context.forced_answer,
        "prefill_sequence_length": int(context.inputs.input_ids.shape[1]),
        "positions": _compact_positions(context.positions),
        "layer_indices": list(range(modules.num_hidden_layers)),
        "num_hidden_layers": modules.num_hidden_layers,
        "hidden_size": modules.hidden_size,
        "dtype": "float16",
        "hidden_state_definition": "decoder_block_output_pre_final_norm",
    }


def evidence_replacements(
    *,
    layer: int,
    target: str,
    recipient_positions: dict[str, Any],
    source_spans: dict[str, torch.Tensor],
) -> list[HiddenStateReplacement]:
    names = ("image", "text_clue") if target == "both" else (target,)
    if any(name not in {"image", "text_clue"} for name in names):
        raise ValueError(f"Unknown evidence target: {target}")
    replacements: list[HiddenStateReplacement] = []
    for name in names:
        target_positions = _position_tuple(recipient_positions[name])
        source = source_spans[name][int(layer)]
        if int(source.shape[0]) != len(target_positions):
            raise ValueError(
                f"{name} donor/recipient span length mismatch: "
                f"donor={source.shape[0]} recipient={len(target_positions)}"
            )
        replacements.append(
            HiddenStateReplacement(
                name=f"{target}:{name}:L{layer}",
                layer_index=int(layer),
                target_positions=target_positions,
                source_hidden=source,
            )
        )
    return replacements


def state_replacements(
    *,
    layer: int,
    intervention: str,
    recipient_positions: dict[str, Any],
    source_core: dict[str, torch.Tensor],
    recipient_core: dict[str, torch.Tensor],
    num_hidden_layers: int,
) -> list[HiddenStateReplacement]:
    layer = int(layer)
    if intervention not in STATE_INTERVENTIONS:
        raise ValueError(f"Unknown state intervention: {intervention}")
    output: list[HiddenStateReplacement] = []
    if intervention in {"ac_natural", "ac_panl_clamp_clean"}:
        output.append(
            HiddenStateReplacement(
                name=f"{intervention}:ac:L{layer}",
                layer_index=layer,
                target_positions=_position_tuple(recipient_positions["ac"]),
                source_hidden=source_core["ac"][layer],
            )
        )
    if intervention == "panl_only":
        output.append(
            HiddenStateReplacement(
                name=f"{intervention}:panl:L{layer}",
                layer_index=layer,
                target_positions=_position_tuple(recipient_positions["panl"]),
                source_hidden=source_core["panl"][layer],
            )
        )
    if intervention == "ac_panl_clamp_clean":
        for downstream in range(layer, num_hidden_layers):
            output.append(
                HiddenStateReplacement(
                    name=f"{intervention}:panl:L{downstream}",
                    layer_index=downstream,
                    target_positions=_position_tuple(recipient_positions["panl"]),
                    source_hidden=recipient_core["panl"][downstream],
                )
            )
    return output


def _numeric_stats(values: Iterable[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(finite),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "min": min(finite),
        "max": max(finite),
    }


def _fraction(values: Iterable[bool]) -> dict[str, Any]:
    flags = [bool(value) for value in values]
    return {
        "n": len(flags),
        "count": sum(flags),
        "fraction": (sum(flags) / len(flags)) if flags else None,
    }


def _group_summary(
    records: Sequence[dict[str, Any]],
    keys: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(record.get(key) for key in keys)].append(record)
    output: list[dict[str, Any]] = []
    for group_key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        entry = {key: value for key, value in zip(keys, group_key)}
        entry.update(
            {
                "count": len(rows),
                "delta_sa": _numeric_stats(
                    row["delta_sa"] for row in rows if row.get("delta_sa") is not None
                ),
                "aligned_delta_sa": _numeric_stats(
                    row["aligned_delta_sa"]
                    for row in rows
                    if row.get("aligned_delta_sa") is not None
                ),
                "directional": _fraction(
                    row["directional"]
                    for row in rows
                    if row.get("directional") is not None
                ),
                "hard_flip": _fraction(
                    row["hard_flip"]
                    for row in rows
                    if row.get("hard_flip") is not None
                ),
                "donor_pull": _fraction(
                    row["donor_pull"]
                    for row in rows
                    if row.get("donor_pull") is not None
                ),
            }
        )
        output.append(entry)
    return output


def build_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    answer = [row for row in records if row.get("experiment") == "answer_force"]
    evidence = [
        row
        for row in records
        if row.get("experiment") == "evidence_swap" and not row.get("is_control")
    ]
    state = [
        row
        for row in records
        if row.get("experiment") == "state_swap" and not row.get("is_control")
    ]
    controls = [row for row in records if row.get("is_control")]
    alignment = [row for row in records if row.get("status") == "alignment_failed"]
    return {
        "format_version": FORMAT_VERSION,
        "record_count": len(records),
        "answer_force": {
            "groups": _group_summary(answer, ("decision_side", "difficulty")),
            "teacher_forcing_calibration": _numeric_stats(
                row["teacher_forcing_calibration_delta"] for row in answer
            ),
        },
        "evidence_swap": {
            "groups": _group_summary(
                evidence,
                ("decision_side", "difficulty", "layer", "intervention"),
            )
        },
        "state_swap": {
            "interpretation": "mediation-style/path-blocking evidence only",
            "groups": _group_summary(
                state,
                (
                    "direction",
                    "decision_side",
                    "difficulty",
                    "layer",
                    "intervention",
                ),
            ),
        },
        "controls": {
            "count": len(controls),
            "pass_fraction": _fraction(
                row.get("self_swap_validation", {}).get("passed", False)
                for row in controls
            ),
            "soft_abs_error": _numeric_stats(
                row.get("self_swap_validation", {}).get("soft_abs_error", math.nan)
                for row in controls
            ),
            "class_probability_max_abs_error": _numeric_stats(
                row.get("self_swap_validation", {}).get(
                    "class_probability_max_abs_error", math.nan
                )
                for row in controls
            ),
        },
        "alignment_failures": {
            "count": len(alignment),
            "records": [
                {
                    "case_id": row.get("case_id"),
                    "layer": row.get("layer"),
                    "intervention": row.get("intervention"),
                    "error": row.get("error"),
                }
                for row in alignment
            ],
        },
    }


class TeacherForcedSourceOriginRunner:
    """Serial, resumable executor with streamed clean-state retention."""

    def __init__(
        self,
        *,
        inference: Any,
        modules: LanguageModules,
        joint_generator: Any,
        source_analyzer: SourceAttributionAnalyzer,
        source_variant: SourcePromptVariant,
        output_dir: str | Path,
        configuration: dict[str, Any],
        cohort_manifest: dict[str, Any],
        layers: Sequence[int] = INTERVENTION_LAYERS,
        state_pair_min_gap: float = 0.15,
        max_state_pairs: int = 30,
        self_swap_tolerance: float = SELF_SWAP_TOLERANCE,
        resume: bool = False,
    ) -> None:
        self.inference = inference
        self.modules = modules
        self.joint_generator = joint_generator
        self.source_analyzer = source_analyzer
        self.source_variant = source_variant
        self.output_dir = Path(output_dir).resolve()
        self.layers = tuple(int(value) for value in layers)
        self.state_pair_min_gap = float(state_pair_min_gap)
        self.max_state_pairs = int(max_state_pairs)
        self.self_swap_tolerance = float(self_swap_tolerance)
        self.resume = bool(resume)
        self.results_path = self.output_dir / "results.jsonl"
        self.config_path = self.output_dir / "run_config.json"
        self.cohort_path = self.output_dir / "cohort_manifest.json"
        self.pair_path = self.output_dir / "pair_manifest.json"
        self.progress_path = self.output_dir / "progress.json"
        self.summary_path = self.output_dir / "summary.json"
        self.configuration = dict(configuration)
        self.fingerprint = configuration_fingerprint(self.configuration)
        self.cohort_manifest = cohort_manifest
        self.cases = self.cohort_manifest["cases"]
        self.by_case_id = {str(case["case_id"]): case for case in self.cases}
        self.state_store: StreamingStateStore
        self.records: list[dict[str, Any]] = []
        self.record_by_key: dict[str, dict[str, Any]] = {}
        self.pair_manifest = {
            "format_version": FORMAT_VERSION,
            "evidence_swap": [],
            "state_swap": [],
        }
        self.already_complete = False
        self._initialize_output()

    def _initialize_output(self) -> None:
        occupied = [
            path
            for path in (
                self.config_path,
                self.cohort_path,
                self.results_path,
                self.progress_path,
                self.summary_path,
            )
            if path.exists()
        ]
        if occupied and not self.resume:
            raise ValueError(
                "Output already exists; pass --resume or choose another directory: "
                + ", ".join(str(path) for path in occupied)
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.resume:
            if not self.config_path.is_file() or not self.cohort_path.is_file():
                raise ValueError("--resume requires run_config.json and cohort_manifest.json")
            saved_config = json.loads(self.config_path.read_text(encoding="utf-8"))
            if saved_config.get("config_fingerprint") != self.fingerprint:
                raise ValueError("Resume configuration differs from saved run_config.json")
            self.already_complete = saved_config.get("status") == "complete"
            self.cohort_manifest = json.loads(
                self.cohort_path.read_text(encoding="utf-8")
            )
            self.cases = self.cohort_manifest["cases"]
            self.by_case_id = {str(case["case_id"]): case for case in self.cases}
            if self.pair_path.is_file():
                self.pair_manifest = json.loads(
                    self.pair_path.read_text(encoding="utf-8")
                )
            self.records = load_jsonl(self.results_path, repair_trailing=True)
            for record in self.records:
                key = record.get("intervention_key")
                if isinstance(key, str):
                    self.record_by_key[key] = record
        else:
            atomic_write_json(
                self.config_path,
                self.configuration
                | {
                    "config_fingerprint": self.fingerprint,
                    "status": "running",
                },
            )
            atomic_write_json(self.cohort_path, self.cohort_manifest)
            atomic_write_json(self.pair_path, self.pair_manifest)
        self.state_store = StreamingStateStore(self.output_dir)
        self._write_progress("complete" if self.already_complete else "initialized")

    def _write_progress(self, stage: str) -> None:
        atomic_write_json(
            self.progress_path,
            {
                "format_version": FORMAT_VERSION,
                "stage": stage,
                "record_count": len(self.records),
                "completed_intervention_count": len(self.record_by_key),
                "case_count": len(self.cases),
                "live_state_tensor_files": self.state_store.live_tensor_files()
                if hasattr(self, "state_store")
                else [],
            },
        )

    def _append(self, record: dict[str, Any]) -> dict[str, Any]:
        key = str(record["intervention_key"])
        if key in self.record_by_key:
            return self.record_by_key[key]
        append_jsonl(self.results_path, record)
        self.records.append(record)
        self.record_by_key[key] = record
        return record

    def _save_cohort(self) -> None:
        atomic_write_json(self.cohort_path, self.cohort_manifest)

    def _prepare_recipient(self, case: dict[str, Any]) -> PreparedTeacherContext:
        return prepare_teacher_context(
            joint_generator=self.joint_generator,
            context_id=str(case["recipient_context_id"]),
            prompt=build_v4_prompt(case, self.source_variant),
            image_path=str(case["recipient_image_path"]),
            forced_answer=str(case["forced_answer"]),
            text_clue=str(case["text_clue"]),
        )

    def _capture_context(
        self,
        *,
        context: PreparedTeacherContext,
    ) -> dict[str, Any]:
        score, core, spans, diagnostics = run_teacher_forced_source(
            joint_generator=self.joint_generator,
            source_analyzer=self.source_analyzer,
            modules=self.modules,
            context=context,
            capture_clean=True,
        )
        assert core is not None and spans is not None and diagnostics is None
        self.state_store.save(
            context_id=context.context_id,
            metadata=context_metadata(context, self.modules),
            core=core,
            spans=spans,
        )
        return score

    def _ensure_recipient_clean(
        self,
        case: dict[str, Any],
        *,
        require_spans: bool,
    ) -> PreparedTeacherContext:
        context = self._prepare_recipient(case)
        has_core = self.state_store.exists(context.context_id, "core")
        has_spans = self.state_store.exists(context.context_id, "spans")
        if (
            case.get("clean_teacher_forced") is None
            or not has_core
            or (require_spans and not has_spans)
        ):
            case["clean_teacher_forced"] = self._capture_context(context=context)
            self._save_cohort()
        return context

    def _answer_force(self, case: dict[str, Any], context: PreparedTeacherContext) -> None:
        key = intervention_key("answer_force", case["case_id"])
        if key in self.record_by_key:
            return
        clean = case["clean_teacher_forced"]
        if case["decision_side"] == "follows_text":
            opposite_answer = case["image_answer"]
            opposite_side = "image"
        else:
            opposite_answer = case["text_answer"]
            opposite_side = "text"
        if not isinstance(opposite_answer, str) or not opposite_answer:
            raise ValueError(f"Case has no opposite answer: {case['case_id']}")
        opposite = prepare_teacher_context(
            joint_generator=self.joint_generator,
            context_id=f"{case['case_id']}|answer_force|{opposite_side}",
            prompt=context.prompt,
            image_path=context.image_path,
            forced_answer=opposite_answer,
            text_clue=str(case["text_clue"]),
        )
        opposite_score, _core, _spans, _diag = run_teacher_forced_source(
            joint_generator=self.joint_generator,
            source_analyzer=self.source_analyzer,
            modules=self.modules,
            context=opposite,
        )
        raw_delta = float(opposite_score["soft_image_score"]) - float(
            clean["soft_image_score"]
        )
        free_sa = float(case["free_generation"]["soft_image_score"])
        self._append(
            {
                "format_version": FORMAT_VERSION,
                "intervention_key": key,
                "experiment": "answer_force",
                "status": "completed",
                "case_id": case["case_id"],
                "item_id": case["item_id"],
                "prior_index": case["prior_index"],
                "condition": case["condition"],
                "difficulty": case["difficulty"],
                "decision_side": case["decision_side"],
                "self_forced_answer": case["forced_answer"],
                "opposite_forced_answer": opposite_answer,
                "opposite_forced_side": opposite_side,
                "clean_teacher_forced": clean,
                "opposite_teacher_forced": opposite_score,
                "free_generation_sa": free_sa,
                "teacher_forcing_calibration_delta": float(
                    clean["soft_image_score"]
                )
                - free_sa,
                "delta_sa": raw_delta,
                "aligned_delta_sa": aligned_answer_force_delta(
                    raw_delta, opposite_side
                ),
                "directional": aligned_answer_force_delta(raw_delta, opposite_side)
                > 0,
                "hard_flip": str(clean["hard_label"])
                != str(opposite_score["hard_label"]),
                "donor_pull": None,
            }
        )
        del opposite

    def _evidence_complete(self, case: dict[str, Any]) -> bool:
        expected = [
            intervention_key(
                "evidence_swap",
                case["case_id"],
                layer,
                target,
                kind,
            )
            for layer in self.layers
            for target in EVIDENCE_TARGETS
            for kind in ("self", "donor")
        ]
        return all(key in self.record_by_key for key in expected)

    def _evidence_swap(
        self,
        case: dict[str, Any],
        recipient: PreparedTeacherContext,
    ) -> None:
        if self._evidence_complete(case):
            self.state_store.delete(
                recipient.context_id, "spans", consumed_by="evidence_swap"
            )
            return
        donor_context_id = str(case["donor_context_id"])
        donor = prepare_teacher_context(
            joint_generator=self.joint_generator,
            context_id=donor_context_id,
            prompt=recipient.prompt,
            image_path=str(case["donor_image_path"]),
            forced_answer=str(case["forced_answer"]),
            text_clue=str(case["text_clue"]),
        )
        if not (
            self.state_store.exists(donor_context_id, "core")
            and self.state_store.exists(donor_context_id, "spans")
        ):
            case["evidence_donor_clean"] = self._capture_context(context=donor)
            self._save_cohort()
        donor_clean = case.get("evidence_donor_clean")
        if not isinstance(donor_clean, dict):
            donor_clean, _a, _b, _c = run_teacher_forced_source(
                joint_generator=self.joint_generator,
                source_analyzer=self.source_analyzer,
                modules=self.modules,
                context=donor,
            )
            case["evidence_donor_clean"] = donor_clean
            self._save_cohort()
        recipient_spans = self.state_store.load(recipient.context_id, "spans")
        donor_spans = self.state_store.load(donor_context_id, "spans")
        clean = case["clean_teacher_forced"]
        for layer in self.layers:
            for target in EVIDENCE_TARGETS:
                self_key = intervention_key(
                    "evidence_swap", case["case_id"], layer, target, "self"
                )
                if self_key in self.record_by_key:
                    previous_validation = self.record_by_key[self_key].get(
                        "self_swap_validation", {}
                    )
                    if not previous_validation.get("passed"):
                        raise RuntimeError(
                            f"Saved Evidence self-swap is invalid for "
                            f"{case['case_id']} L{layer}/{target}: "
                            f"{previous_validation}"
                        )
                else:
                    replacements = evidence_replacements(
                        layer=layer,
                        target=target,
                        recipient_positions=recipient.positions,
                        source_spans=recipient_spans,
                    )
                    patched, _a, _b, diagnostics = run_teacher_forced_source(
                        joint_generator=self.joint_generator,
                        source_analyzer=self.source_analyzer,
                        modules=self.modules,
                        context=recipient,
                        replacements=replacements,
                    )
                    validation = self_swap_validation(
                        clean,
                        patched,
                        tolerance=self.self_swap_tolerance,
                    )
                    record = {
                        "format_version": FORMAT_VERSION,
                        "intervention_key": self_key,
                        "experiment": "evidence_swap",
                        "status": "completed",
                        "is_control": True,
                        "case_id": case["case_id"],
                        "item_id": case["item_id"],
                        "prior_index": case["prior_index"],
                        "condition": case["condition"],
                        "difficulty": case["difficulty"],
                        "decision_side": case["decision_side"],
                        "layer": layer,
                        "intervention": target,
                        "clean_sa": clean["soft_image_score"],
                        "patched_sa": patched["soft_image_score"],
                        "clean_teacher_forced": clean,
                        "patched_teacher_forced": patched,
                        "delta_sa": float(patched["soft_image_score"])
                        - float(clean["soft_image_score"]),
                        "aligned_delta_sa": None,
                        "directional": None,
                        "hard_flip": str(clean["hard_label"])
                        != str(patched["hard_label"]),
                        "donor_pull": None,
                        "self_swap_validation": validation,
                        "hook_diagnostics": diagnostics,
                    }
                    self._append(record)
                    if not validation["passed"]:
                        raise RuntimeError(
                            f"Evidence self-swap failed for {case['case_id']} "
                            f"L{layer}/{target}: {validation}"
                        )

                donor_key = intervention_key(
                    "evidence_swap", case["case_id"], layer, target, "donor"
                )
                if donor_key in self.record_by_key:
                    continue
                try:
                    replacements = evidence_replacements(
                        layer=layer,
                        target=target,
                        recipient_positions=recipient.positions,
                        source_spans=donor_spans,
                    )
                except ValueError as exc:
                    self._append(
                        {
                            "format_version": FORMAT_VERSION,
                            "intervention_key": donor_key,
                            "experiment": "evidence_swap",
                            "status": "alignment_failed",
                            "is_control": False,
                            "case_id": case["case_id"],
                            "item_id": case["item_id"],
                            "prior_index": case["prior_index"],
                            "condition": case["condition"],
                            "difficulty": case["difficulty"],
                            "decision_side": case["decision_side"],
                            "layer": layer,
                            "intervention": target,
                            "clean_teacher_forced": clean,
                            "donor_teacher_forced": donor_clean,
                            "error": {"type": type(exc).__name__, "message": str(exc)},
                        }
                    )
                    continue
                patched, _a, _b, diagnostics = run_teacher_forced_source(
                    joint_generator=self.joint_generator,
                    source_analyzer=self.source_analyzer,
                    modules=self.modules,
                    context=recipient,
                    replacements=replacements,
                )
                metrics = donor_metrics(
                    recipient_sa=float(clean["soft_image_score"]),
                    patched_sa=float(patched["soft_image_score"]),
                    donor_sa=float(donor_clean["soft_image_score"]),
                )
                self._append(
                    {
                        "format_version": FORMAT_VERSION,
                        "intervention_key": donor_key,
                        "experiment": "evidence_swap",
                        "status": "completed",
                        "is_control": False,
                        "case_id": case["case_id"],
                        "item_id": case["item_id"],
                        "prior_index": case["prior_index"],
                        "condition": case["condition"],
                        "difficulty": case["difficulty"],
                        "decision_side": case["decision_side"],
                        "layer": layer,
                        "intervention": target,
                        "clean_sa": clean["soft_image_score"],
                        "donor_clean_sa": donor_clean["soft_image_score"],
                        "patched_sa": patched["soft_image_score"],
                        "clean_teacher_forced": clean,
                        "donor_teacher_forced": donor_clean,
                        "patched_teacher_forced": patched,
                        **metrics,
                        "hard_flip": str(clean["hard_label"])
                        != str(patched["hard_label"]),
                        "hook_diagnostics": diagnostics,
                    }
                )
        self.state_store.delete(
            recipient.context_id, "spans", consumed_by="evidence_swap"
        )
        self.state_store.delete(donor_context_id, "spans", consumed_by="evidence_swap")
        self.state_store.delete(donor_context_id, "core", consumed_by="evidence_swap")
        del recipient_spans, donor_spans, donor

    def _build_pair_manifest(self) -> None:
        evidence_pairs = [
            {
                "recipient_case_id": case["case_id"],
                "recipient_condition": case["condition"],
                "donor_condition": case["donor_condition"],
                "item_id": case["item_id"],
                "prior_index": case["prior_index"],
                "difficulty": case["difficulty"],
                "forced_answer": case["forced_answer"],
                "recipient_clean_sa": case.get("clean_teacher_forced", {}).get(
                    "soft_image_score"
                ),
                "donor_clean_sa": case.get("evidence_donor_clean", {}).get(
                    "soft_image_score"
                ),
            }
            for case in self.cases
        ]
        state_pairs = select_state_pairs(
            self.cases,
            min_sa_gap=self.state_pair_min_gap,
            max_pairs=self.max_state_pairs,
        )
        self.pair_manifest = {
            "format_version": FORMAT_VERSION,
            "evidence_swap": evidence_pairs,
            "state_swap": state_pairs,
            "state_pair_policy": {
                "pool": "fixed_cohort_only",
                "same_normalized_answer": True,
                "same_decision_side": True,
                "min_sa_gap": self.state_pair_min_gap,
                "max_pairs": self.max_state_pairs,
                "case_reuse": False,
            },
        }
        atomic_write_json(self.pair_path, self.pair_manifest)

    def _ensure_core_context(
        self, case: dict[str, Any]
    ) -> PreparedTeacherContext:
        context = self._prepare_recipient(case)
        if not self.state_store.exists(context.context_id, "core"):
            case["clean_teacher_forced"] = self._capture_context(context=context)
            self.state_store.delete(
                context.context_id, "spans", consumed_by="state_swap_not_needed"
            )
            self._save_cohort()
        return context

    def _state_swap(self) -> None:
        for pair in self.pair_manifest["state_swap"]:
            low = self.by_case_id[str(pair["low_case_id"])]
            high = self.by_case_id[str(pair["high_case_id"])]
            contexts = {
                low["case_id"]: self._ensure_core_context(low),
                high["case_id"]: self._ensure_core_context(high),
            }
            cores = {
                low["case_id"]: self.state_store.load(
                    str(low["recipient_context_id"]), "core"
                ),
                high["case_id"]: self.state_store.load(
                    str(high["recipient_context_id"]), "core"
                ),
            }
            directions = (
                ("high_to_low", low, high),
                ("low_to_high", high, low),
            )
            for direction, recipient, donor in directions:
                recipient_context = contexts[recipient["case_id"]]
                recipient_core = cores[recipient["case_id"]]
                donor_core = cores[donor["case_id"]]
                clean = recipient["clean_teacher_forced"]
                donor_clean = donor["clean_teacher_forced"]
                for layer in self.layers:
                    for intervention in STATE_INTERVENTIONS:
                        self_key = intervention_key(
                            "state_swap",
                            pair["pair_id"],
                            direction,
                            layer,
                            intervention,
                            "self",
                        )
                        if self_key in self.record_by_key:
                            previous_validation = self.record_by_key[self_key].get(
                                "self_swap_validation", {}
                            )
                            if not previous_validation.get("passed"):
                                raise RuntimeError(
                                    f"Saved State self-swap is invalid for "
                                    f"{pair['pair_id']} {direction} "
                                    f"L{layer}/{intervention}: {previous_validation}"
                                )
                        else:
                            replacements = state_replacements(
                                layer=layer,
                                intervention=intervention,
                                recipient_positions=recipient_context.positions,
                                source_core=recipient_core,
                                recipient_core=recipient_core,
                                num_hidden_layers=self.modules.num_hidden_layers,
                            )
                            patched, _a, _b, diagnostics = run_teacher_forced_source(
                                joint_generator=self.joint_generator,
                                source_analyzer=self.source_analyzer,
                                modules=self.modules,
                                context=recipient_context,
                                replacements=replacements,
                            )
                            validation = self_swap_validation(
                                clean,
                                patched,
                                tolerance=self.self_swap_tolerance,
                            )
                            self._append(
                                {
                                    "format_version": FORMAT_VERSION,
                                    "intervention_key": self_key,
                                    "experiment": "state_swap",
                                    "status": "completed",
                                    "is_control": True,
                                    "pair_id": pair["pair_id"],
                                    "direction": direction,
                                    "case_id": recipient["case_id"],
                                    "donor_case_id": recipient["case_id"],
                                    "item_id": recipient["item_id"],
                                    "prior_index": recipient["prior_index"],
                                    "condition": recipient["condition"],
                                    "difficulty": recipient["difficulty"],
                                    "decision_side": recipient["decision_side"],
                                    "layer": layer,
                                    "intervention": intervention,
                                    "clean_sa": clean["soft_image_score"],
                                    "patched_sa": patched["soft_image_score"],
                                    "clean_teacher_forced": clean,
                                    "patched_teacher_forced": patched,
                                    "delta_sa": float(patched["soft_image_score"])
                                    - float(clean["soft_image_score"]),
                                    "aligned_delta_sa": None,
                                    "directional": None,
                                    "hard_flip": str(clean["hard_label"])
                                    != str(patched["hard_label"]),
                                    "donor_pull": None,
                                    "self_swap_validation": validation,
                                    "hook_diagnostics": diagnostics,
                                }
                            )
                            if not validation["passed"]:
                                raise RuntimeError(
                                    f"State self-swap failed for {pair['pair_id']} "
                                    f"{direction} L{layer}/{intervention}: {validation}"
                                )
                        donor_key = intervention_key(
                            "state_swap",
                            pair["pair_id"],
                            direction,
                            layer,
                            intervention,
                            "donor",
                        )
                        if donor_key in self.record_by_key:
                            continue
                        replacements = state_replacements(
                            layer=layer,
                            intervention=intervention,
                            recipient_positions=recipient_context.positions,
                            source_core=donor_core,
                            recipient_core=recipient_core,
                            num_hidden_layers=self.modules.num_hidden_layers,
                        )
                        patched, _a, _b, diagnostics = run_teacher_forced_source(
                            joint_generator=self.joint_generator,
                            source_analyzer=self.source_analyzer,
                            modules=self.modules,
                            context=recipient_context,
                            replacements=replacements,
                        )
                        metrics = donor_metrics(
                            recipient_sa=float(clean["soft_image_score"]),
                            patched_sa=float(patched["soft_image_score"]),
                            donor_sa=float(donor_clean["soft_image_score"]),
                        )
                        self._append(
                            {
                                "format_version": FORMAT_VERSION,
                                "intervention_key": donor_key,
                                "experiment": "state_swap",
                                "status": "completed",
                                "is_control": False,
                                "pair_id": pair["pair_id"],
                                "match_tier": pair["match_tier"],
                                "direction": direction,
                                "case_id": recipient["case_id"],
                                "donor_case_id": donor["case_id"],
                                "item_id": recipient["item_id"],
                                "prior_index": recipient["prior_index"],
                                "condition": recipient["condition"],
                                "difficulty": recipient["difficulty"],
                                "decision_side": recipient["decision_side"],
                                "normalized_answer": recipient["normalized_answer"],
                                "layer": layer,
                                "intervention": intervention,
                                "clean_sa": clean["soft_image_score"],
                                "donor_clean_sa": donor_clean["soft_image_score"],
                                "patched_sa": patched["soft_image_score"],
                                "clean_teacher_forced": clean,
                                "donor_teacher_forced": donor_clean,
                                "patched_teacher_forced": patched,
                                **metrics,
                                "hard_flip": str(clean["hard_label"])
                                != str(patched["hard_label"]),
                                "hook_diagnostics": diagnostics,
                            }
                        )
            del contexts, cores

    def execute(self) -> dict[str, Any]:
        if self.already_complete:
            if not self.summary_path.is_file():
                raise ValueError("Completed run has no summary.json")
            return json.loads(self.summary_path.read_text(encoding="utf-8"))
        try:
            for index, case in enumerate(self.cases):
                case.setdefault(
                    "recipient_context_id",
                    f"{case['case_id']}|recipient|{case['normalized_answer']}",
                )
                case.setdefault(
                    "donor_context_id",
                    f"{case['case_id']}|{case['donor_condition']}|"
                    f"{case['normalized_answer']}",
                )
                recipient = self._ensure_recipient_clean(
                    case,
                    require_spans=not self._evidence_complete(case),
                )
                self._answer_force(case, recipient)
                self._evidence_swap(case, recipient)
                self._save_cohort()
                self._write_progress(f"evidence_case_{index + 1}_of_{len(self.cases)}")
                del recipient

            self._build_pair_manifest()
            self._write_progress("state_swap")
            self._state_swap()

            for case in self.cases:
                context_id = str(case["recipient_context_id"])
                self.state_store.delete(
                    context_id, "spans", consumed_by="run_complete"
                )
                self.state_store.delete(context_id, "core", consumed_by="state_swap")
            summary = build_summary(self.records)
            summary["state_storage"] = {
                "retention_policy": "stream_delete",
                "live_tensor_files": self.state_store.live_tensor_files(),
                "all_decoder_layers_captured": list(range(self.modules.num_hidden_layers)),
            }
            atomic_write_json(self.summary_path, summary)
            atomic_write_json(
                self.config_path,
                self.configuration
                | {
                    "config_fingerprint": self.fingerprint,
                    "status": "complete",
                },
            )
            self._write_progress("complete")
            return summary
        except Exception:
            atomic_write_json(
                self.config_path,
                self.configuration
                | {
                    "config_fingerprint": self.fingerprint,
                    "status": "failed",
                },
            )
            self._write_progress("failed")
            raise
