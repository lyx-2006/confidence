from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from dp_SA.soft_score import class_token_ids
from dp_SA.SA_trajectory.LAT2PANL.run import load_explicit_fast_runtime
from layer_metacognition.model_adapter import LanguageModules, resolve_language_modules, run_logits_forward

from .config import CLE_LAYER, OUTPUT_PARENT, require_output_root
from .hooks import SingleHiddenCapture
from .io_utils import (
    atomic_csv,
    atomic_json,
    bf16_to_uint16,
    bits_hash,
    canonical_hash,
    load_bf16_npz,
    load_jsonl,
    sha256_file,
)
from .run import _context, _margin, _probe, _probe_value, _safe, _score
from .statistics import bh_fdr


FORMAL_ROOT = OUTPUT_PARENT / "formal"
STAGE1_ROOT = OUTPUT_PARENT / "layer_trajectory_stage1"
STAGE_ROOT = OUTPUT_PARENT / "window_bisection_stage2"
SELECTED_CELLS = (("W2", 17), ("W5", 16))
SEGMENTS = {"left4": (0, 4), "right4": (4, 8)}
EXPECTED_NEW_PATCHED = 400
EXPECTED_REUSED_FULL = 200
BOOTSTRAP_REPEATS = 2000
RETENTION_DOMINANT = 0.60
RETENTION_NEAR_ZERO = 0.20
RETENTION_DISTRIBUTED = 0.35


def _tensor_output(output: Any) -> tuple[torch.Tensor, tuple[Any, ...] | None]:
    if isinstance(output, torch.Tensor):
        return output, None
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0], output[1:]
    raise TypeError(f"Unsupported decoder output: {type(output)!r}")


class SegmentSwapHook:
    """Losslessly replace one contiguous token segment at one decoder block output."""

    def __init__(self, modules: LanguageModules, *, layer: int, positions: Sequence[int], source: torch.Tensor,
                 prefill_length: int, capture_position: int | None = None):
        self.modules = modules
        self.layer = int(layer)
        self.positions = list(map(int, positions))
        self.source = source.detach().cpu()
        self.prefill_length = int(prefill_length)
        self.handle: Any | None = None
        self.hook_count = 0
        self.applied_count = 0
        self.target_exact = False
        self.outside_exact = False
        self.patch_l2: float | None = None
        self.capture_position = None if capture_position is None else int(capture_position)
        self.captured_position_hidden: torch.Tensor | None = None
        if not self.positions or self.positions != list(range(self.positions[0], self.positions[0] + len(self.positions))):
            raise ValueError("Segment positions must be non-empty and contiguous")
        expected = (len(self.positions), modules.hidden_size)
        if self.source.dtype != torch.bfloat16 or tuple(self.source.shape) != expected:
            raise ValueError(f"Segment source must be {expected} BF16")

    def _hook(self, _module: Any, _args: Any, output: Any) -> Any:
        self.hook_count += 1
        tensor, trailing = _tensor_output(output)
        if self.applied_count or int(tensor.shape[1]) != self.prefill_length:
            return output
        if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[2] != self.modules.hidden_size:
            raise ValueError("Invalid recipient block output")
        source = self.source.to(tensor.device)
        if source.dtype != tensor.dtype:
            raise TypeError(f"Lossless swap dtype mismatch: {source.dtype} != {tensor.dtype}")
        before = tensor[0, self.positions, :].detach()
        patched = tensor.clone()
        patched[0, self.positions, :] = source
        self.target_exact = bool(torch.equal(patched[0, self.positions, :], source))
        mask = torch.ones(tensor.shape[1], dtype=torch.bool, device=tensor.device)
        mask[self.positions] = False
        self.outside_exact = bool(torch.equal(patched[:, mask, :], tensor[:, mask, :]))
        self.patch_l2 = float(torch.linalg.vector_norm(source.float() - before.float()).item())
        if self.capture_position is not None:
            if not (0 <= self.capture_position < int(patched.shape[1])):
                raise ValueError("Probe capture position outside block output")
            self.captured_position_hidden = patched[0, self.capture_position, :].detach().float().cpu()
        self.applied_count += 1
        if not self.target_exact or not self.outside_exact:
            raise RuntimeError("Segment-only replacement invariant failed")
        return patched if trailing is None else (patched, *trailing)

    def __enter__(self) -> "SegmentSwapHook":
        self.handle = self.modules.language_layers[self.layer].register_forward_hook(self._hook)
        return self

    def __exit__(self, *_args: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def diagnostics(self) -> dict[str, Any]:
        if self.applied_count != 1:
            raise RuntimeError(f"Segment swap applied {self.applied_count} times")
        return {
            "layer": self.layer,
            "positions": self.positions,
            "segment_length": len(self.positions),
            "hook_count": self.hook_count,
            "applied_count": self.applied_count,
            "target_exact": self.target_exact,
            "outside_exact": self.outside_exact,
            "patch_l2": self.patch_l2,
            "capture_position": self.capture_position,
            "captured_position": self.captured_position_hidden is not None,
            "site": "decoder_block_output_post_mlp_residual",
        }

    def validate_captured_position(self) -> torch.Tensor:
        if self.capture_position is None or self.captured_position_hidden is None:
            raise RuntimeError("No patched position capture was requested")
        return self.captured_position_hidden


def split_window(positions: Sequence[int], segment: str) -> list[int]:
    values = list(map(int, positions))
    if len(values) != 8 or values != list(range(values[0], values[0] + 8)):
        raise ValueError("Bisection requires one contiguous 8-token window")
    if segment not in SEGMENTS:
        raise KeyError(segment)
    start, stop = SEGMENTS[segment]
    return values[start:stop]


def _files() -> dict[str, Path]:
    return {
        "formal_completion": FORMAL_ROOT / "completion.json",
        "formal_config": FORMAL_ROOT / "run_config.json",
        "recipients": FORMAL_ROOT / "artifacts/manifests/recipient_manifest.jsonl",
        "donors": FORMAL_ROOT / "artifacts/manifests/donor_manifest.jsonl",
        "pairs": FORMAL_ROOT / "artifacts/manifests/donor_matching.jsonl",
        "probe": FORMAL_ROOT / "artifacts/diagnostics/cle_probe.json",
        "stage1_completion": STAGE1_ROOT / "completion.json",
        "stage1_config": STAGE1_ROOT / "run_config.json",
        "stage1_selection": STAGE1_ROOT / "selection.json",
    }


def _prepare(*, resume: bool) -> dict[str, Any]:
    root = require_output_root(STAGE_ROOT)
    for relative in ("artifacts/trials", "tables", "figures", "progress", "logs"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    sources = _files()
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    for completion_name in ("formal_completion", "stage1_completion"):
        completion = json.loads(sources[completion_name].read_text())
        if completion.get("status") != "complete" or not all(completion.get("gates", {}).values()):
            raise RuntimeError(f"Frozen parent is incomplete: {completion_name}")
    selected = json.loads(sources["stage1_selection"].read_text())
    frozen_selected = tuple((window, int(selected["windows"][window]["recommended_layer"])) for window, _ in SELECTED_CELLS)
    if frozen_selected != SELECTED_CELLS:
        raise RuntimeError(f"Selected cells changed: {frozen_selected} != {SELECTED_CELLS}")
    payload = {
        "experiment": "PANL2CLE_window_bisection_stage2",
        "selected_cells": [list(cell) for cell in SELECTED_CELLS],
        "segments": {name: list(bounds) for name, bounds in SEGMENTS.items()},
        "expected_new_patched_forwards": EXPECTED_NEW_PATCHED,
        "reused_full_window_trials": EXPECTED_REUSED_FULL,
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "classification_thresholds": {
            "dominant_retention": RETENTION_DOMINANT,
            "near_zero_retention": RETENTION_NEAR_ZERO,
            "distributed_retention": RETENTION_DISTRIBUTED,
        },
        "source_sha256": {name: sha256_file(path) for name, path in sources.items()},
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    payload["fingerprint"] = canonical_hash(payload)
    destination = root / "run_config.json"
    if destination.exists():
        old = json.loads(destination.read_text())
        if old.get("fingerprint") != payload["fingerprint"]:
            raise RuntimeError("Stage-2 resume fingerprint mismatch")
        if not resume:
            raise FileExistsError(f"Stage output exists; use --resume: {root}")
    else:
        atomic_json(destination, payload)
    return payload


def _load_parent() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    recipients = load_jsonl(_files()["recipients"])
    donors = load_jsonl(_files()["donors"])
    pairs = load_jsonl(_files()["pairs"])
    if len(recipients) != 50 or len(donors) != 100 or len(pairs) != 100:
        raise RuntimeError("Frozen parent manifest cardinality changed")
    return recipients, donors, pairs


def _donor_path(case_id: str) -> Path:
    return STAGE1_ROOT / "artifacts/donor_hidden" / f"{_safe(case_id)}.npz"


def _trial_path(case_id: str, condition: str, window: str, layer: int, segment: str) -> Path:
    return STAGE_ROOT / "artifacts/trials" / f"{_safe(case_id)}__{condition}__{window}__L{layer}__{segment}.json"


def _run_trials(runtime: Any, modules: Any, recipients: Sequence[dict[str, Any]], donors: Sequence[dict[str, Any]],
                pairs: Sequence[dict[str, Any]], token_ids: Sequence[int], *, resume: bool) -> int:
    pair_index = {(str(row["recipient_case_id"]), str(row["donor_side"])): row for row in pairs}
    donor_index = {str(row["case_id"]): row for row in donors}
    probe, _probe_record = _probe(FORMAL_ROOT)
    new_forwards = 0
    for recipient_number, row in enumerate(recipients, 1):
        case_id = str(row["case_id"])
        clean_path = FORMAL_ROOT / "artifacts/trials" / f"{_safe(case_id)}__clean.json"
        if not clean_path.is_file():
            raise FileNotFoundError(clean_path)
        clean = json.loads(clean_path.read_text())
        inputs, windows, phase = _context(runtime, row)
        sac = int(phase["P1_SAC"]["processed_index"])
        cle = int(phase["P1_CLASS_LIST_END"]["processed_index"])
        for window, layer in SELECTED_CELLS:
            full_positions = windows[window]["processed_indices"]
            for segment in SEGMENTS:
                positions = split_window(full_positions, segment)
                start, stop = SEGMENTS[segment]
                for donor_side in ("high_image", "high_text"):
                    pair = pair_index[case_id, donor_side]
                    donor = donor_index[str(pair["donor_case_id"])]
                    condition = ("H" if row["sa_side"] == "high_image" else "L") + "_from_" + ("H" if donor_side == "high_image" else "L")
                    destination = _trial_path(case_id, condition, window, layer, segment)
                    if destination.exists():
                        if resume:
                            continue
                        raise FileExistsError(destination)
                    full_source, source_meta = load_bf16_npz(_donor_path(str(donor["case_id"])), f"{window}__L{layer}")
                    source = full_source[start:stop].contiguous()
                    source_hash = bits_hash(bf16_to_uint16(source))
                    swap = SegmentSwapHook(modules, layer=layer, positions=positions, source=source,
                                           prefill_length=int(inputs.input_ids.shape[1]))
                    capture = SingleHiddenCapture(modules, layer=CLE_LAYER, position=cle,
                                                  prefill_length=int(inputs.input_ids.shape[1]))
                    started = time.perf_counter()
                    with swap, capture:
                        logits = run_logits_forward(runtime.model, inputs, [sac], modules)[sac]
                    elapsed = time.perf_counter() - started
                    score = _score(logits, token_ids)
                    downstream = capture.validate()
                    patched_cle = _probe_value(downstream, probe)
                    delta = float(score["soft_sa_image_score"]) - float(clean["soft_sa"])
                    donor_gap = float(pair["donor_clean_sa"]) - float(clean["soft_sa"])
                    direction = int(donor_gap > 0) - int(donor_gap < 0)
                    clean_distance = abs(float(clean["soft_sa"]) - float(pair["donor_clean_sa"]))
                    patched_distance = abs(float(score["soft_sa_image_score"]) - float(pair["donor_clean_sa"]))
                    clean_class = int(clean["hard_sa_class"])
                    patched_margin = _margin(score["class_logits"], clean_class)
                    trial = {
                        "status": "completed", "source": "new_stage2", "case_id": case_id,
                        "family_id": row["family_id"], "item_id": row["item_id"], "answer": row["phase0_raw_answer"],
                        "recipient_side": row["sa_side"], "donor_case_id": donor["case_id"], "donor_side": donor_side,
                        "condition": condition, "window": window, "layer": layer, "segment": segment,
                        "segment_offsets": [start, stop], "full_window_positions": full_positions,
                        "segment_positions": positions, "clean_soft_sa": clean["soft_sa"],
                        "patched_soft_sa": score["soft_sa_image_score"], "delta_soft_sa": delta,
                        "abs_delta_soft_sa": abs(delta), "donor_clean_sa": pair["donor_clean_sa"], "donor_gap": donor_gap,
                        "toward_score": delta * direction,
                        "toward": None if direction == 0 else bool(delta * direction > 0), "zero_donor_gap": direction == 0,
                        "clean_donor_distance": clean_distance, "patched_donor_distance": patched_distance,
                        "donor_distance_reduction": clean_distance - patched_distance,
                        "clean_hard_sa_class": clean_class, "patched_hard_sa_class": score["argmax_hard_class"],
                        "hard_changed": int(score["argmax_hard_class"]) != clean_class,
                        "class_logits": score["class_logits"], "class_probabilities": score["class_probabilities"],
                        "clean_margin": clean["clean_margin"], "patched_clean_class_margin": patched_margin,
                        "margin_change": patched_margin - float(clean["clean_margin"]),
                        "cle_probe_eligible": clean["cle_probe_eligible"], "clean_cle_probe_sa": clean["cle_probe_sa"],
                        "patched_cle_probe_sa": patched_cle, "cle_probe_delta": patched_cle - float(clean["cle_probe_sa"]),
                        "hook": swap.diagnostics(), "full_source_bits_sha256": source_meta["bits_sha256"],
                        "segment_source_bits_sha256": source_hash, "matching": pair, "elapsed_seconds": elapsed,
                    }
                    atomic_json(destination, trial)
                    new_forwards += 1
                    atomic_json(STAGE_ROOT / "progress/trials.json", {
                        "status": "running", "new_gpu_forwards": new_forwards,
                        "total_completed": len(list((STAGE_ROOT / "artifacts/trials").glob("*.json"))),
                        "expected": EXPECTED_NEW_PATCHED, "recipient": recipient_number, "last_trial": destination.name,
                    })
        del inputs
    total = len(list((STAGE_ROOT / "artifacts/trials").glob("*.json")))
    atomic_json(STAGE_ROOT / "progress/trials.json", {
        "status": "complete", "new_gpu_forwards": new_forwards, "total_completed": total,
        "expected": EXPECTED_NEW_PATCHED,
    })
    return new_forwards


def _load_trials() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    halves = [json.loads(path.read_text()) for path in sorted((STAGE_ROOT / "artifacts/trials").glob("*.json"))]
    full = []
    selected = set(SELECTED_CELLS)
    for path in sorted((STAGE1_ROOT / "artifacts/trials").glob("*.json")):
        row = json.loads(path.read_text())
        if (str(row.get("window")), int(row.get("layer", -1))) in selected:
            full.append({**row, "segment": "full8", "source": "reused_stage1"})
    if len(halves) != EXPECTED_NEW_PATCHED or len(full) != EXPECTED_REUSED_FULL:
        raise RuntimeError(f"Bisection trial count mismatch: halves={len(halves)}, full={len(full)}")
    return halves, full


def _paired_effects(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    paired: dict[str, dict[str, Any]] = {}
    for row in rows:
        case = str(row["case_id"])
        slot = paired.setdefault(case, {"row": row})
        slot[str(row["donor_side"])] = float(row["patched_soft_sa"])
    output = {}
    for case, values in paired.items():
        if "high_image" not in values or "high_text" not in values:
            raise RuntimeError(f"Unpaired donor trial: {case}")
        row = values["row"]
        output[case] = {
            "case_id": case, "family_id": str(row["family_id"]), "recipient_side": str(row["recipient_side"]),
            "effect": values["high_image"] - values["high_text"],
        }
    return output


def _draw_equal_side(records: Sequence[dict[str, Any]], rng: np.random.Generator, repeats: int) -> np.ndarray:
    draws = np.empty(repeats, dtype=float)
    by_side = {side: [row for row in records if row["recipient_side"] == side] for side in ("high_image", "high_text")}
    if any(not rows for rows in by_side.values()):
        raise RuntimeError("Both recipient strata are required")
    for repeat in range(repeats):
        means = []
        for side in ("high_image", "high_text"):
            families: dict[str, list[float]] = defaultdict(list)
            for row in by_side[side]:
                families[row["family_id"]].append(float(row["effect"]))
            keys = list(families)
            sampled = rng.choice(keys, size=len(keys), replace=True)
            means.append(float(np.mean([value for family in sampled for value in families[str(family)]])))
        draws[repeat] = float(np.mean(means))
    return draws


def _summary(point: float, draws: np.ndarray) -> dict[str, Any]:
    return {
        "estimate": float(point), "sem": float(np.std(draws, ddof=1)),
        "ci_low": float(np.quantile(draws, .025)), "ci_high": float(np.quantile(draws, .975)),
        "bootstrap_p_two_sided": float(min(1.0, 2 * min(np.mean(draws <= 0), np.mean(draws >= 0)))),
        "valid_bootstrap_repeats": int(len(draws)),
    }


def _point_equal_side(records: Sequence[dict[str, Any]]) -> float:
    values = []
    for side in ("high_image", "high_text"):
        side_values = [float(row["effect"]) for row in records if row["recipient_side"] == side]
        if not side_values:
            raise RuntimeError(f"Missing recipient stratum: {side}")
        values.append(float(np.mean(side_values)))
    return float(np.mean(values))


def _classify(full: dict[str, Any], left: dict[str, Any], right: dict[str, Any], interaction: dict[str, Any]) -> dict[str, Any]:
    clear = (float(full["ci_low"]) > 0) or (float(full["ci_high"]) < 0)
    result: dict[str, Any] = {"full_effect_clear": clear, "retention_left": None, "retention_right": None}
    if not clear or math.isclose(float(full["estimate"]), 0.0, abs_tol=1e-12):
        return {**result, "category": "full8_not_clearly_nonzero", "continue_bisection": False,
                "reason": "E8 is not clearly separated from zero; retention ratios are intentionally omitted."}
    e8 = float(full["estimate"])
    rl = float(left["estimate"]) / e8
    rr = float(right["estimate"]) / e8
    result.update({"retention_left": rl, "retention_right": rr})
    opposite = float(left["estimate"]) * float(right["estimate"]) < 0 and max(abs(rl), abs(rr)) >= RETENTION_NEAR_ZERO
    left_dominant = rl >= RETENTION_DOMINANT and abs(rr) <= RETENTION_NEAR_ZERO
    right_dominant = rr >= RETENTION_DOMINANT and abs(rl) <= RETENTION_NEAR_ZERO
    both_distributed = rl >= RETENTION_DISTRIBUTED and rr >= RETENTION_DISTRIBUTED
    both_weak = abs(rl) < RETENTION_DISTRIBUTED and abs(rr) < RETENTION_DISTRIBUTED
    if opposite:
        category, proceed, target = "opposite_directions", True, ["left4", "right4"]
    elif left_dominant or right_dominant:
        target = ["left4" if left_dominant else "right4"]
        category, proceed = "one_half_dominant", True
    elif both_distributed:
        category, proceed, target = "both_halves_distributed", True, ["left4", "right4"]
    elif both_weak:
        category, proceed, target = "weak_halves_full_effect_interaction", False, []
    else:
        category, proceed, target = "mixed_or_uncertain", False, []
    return {
        **result, "category": category, "continue_bisection": proceed, "suggested_segments": target,
        "interaction_estimate": float(interaction["estimate"]),
        "note": "Classification is heuristic; effect estimates and confidence intervals remain primary.",
    }


def analyze() -> dict[str, Any]:
    halves, full = _load_trials()
    effect_rows: list[dict[str, Any]] = []
    classifications: dict[str, Any] = {}
    rng = np.random.default_rng(42)
    for window, layer in SELECTED_CELLS:
        sources = {
            "full8": [row for row in full if row["window"] == window and int(row["layer"]) == layer],
            "left4": [row for row in halves if row["window"] == window and int(row["layer"]) == layer and row["segment"] == "left4"],
            "right4": [row for row in halves if row["window"] == window and int(row["layer"]) == layer and row["segment"] == "right4"],
        }
        paired = {segment: _paired_effects(rows) for segment, rows in sources.items()}
        cases = set(paired["full8"])
        if any(set(values) != cases for values in paired.values()):
            raise RuntimeError(f"Recipient sets differ across segments: {window}/L{layer}")
        records: dict[str, list[dict[str, Any]]] = {}
        for segment in ("full8", "left4", "right4"):
            records[segment] = [paired[segment][case] for case in sorted(cases)]
        records["interaction"] = [{**paired["full8"][case], "effect": paired["full8"][case]["effect"] - paired["left4"][case]["effect"] - paired["right4"][case]["effect"]} for case in sorted(cases)]
        summaries: dict[str, dict[str, Any]] = {}
        for segment in ("full8", "left4", "right4", "interaction"):
            draws = _draw_equal_side(records[segment], rng, BOOTSTRAP_REPEATS)
            summary = _summary(_point_equal_side(records[segment]), draws)
            summaries[segment] = summary
            effect_rows.append({"window": window, "layer": layer, "component": segment, "n_recipients": len(records[segment]), **summary})
        classification = _classify(summaries["full8"], summaries["left4"], summaries["right4"], summaries["interaction"])
        classifications[f"{window}__L{layer}"] = {"window": window, "layer": layer, "effects": summaries, **classification}
    p_indices = [i for i, row in enumerate(effect_rows) if row["component"] in ("left4", "right4", "interaction")]
    adjusted = bh_fdr([effect_rows[i]["bootstrap_p_two_sided"] for i in p_indices])
    for index, q in zip(p_indices, adjusted):
        effect_rows[index]["bh_fdr_q"] = q
    atomic_csv(STAGE_ROOT / "tables/bisection_effects.csv", effect_rows)
    flat_keys = ("case_id", "family_id", "recipient_side", "donor_case_id", "donor_side", "condition", "window", "layer", "segment",
                 "clean_soft_sa", "patched_soft_sa", "delta_soft_sa", "abs_delta_soft_sa", "toward_score", "toward", "hard_changed", "cle_probe_delta")
    atomic_csv(STAGE_ROOT / "tables/half_window_trials.csv", [{key: row.get(key) for key in flat_keys} for row in halves])
    atomic_json(STAGE_ROOT / "classification.json", {
        "thresholds": {"dominant": RETENTION_DOMINANT, "near_zero": RETENTION_NEAR_ZERO, "distributed": RETENTION_DISTRIBUTED},
        "interpretation_boundary": "Locates natural-hidden segments sensitive to paired donor differences; does not prove a token stores SA.",
        "cells": classifications,
    })
    _plot(effect_rows)
    summary = {
        "status": "complete", "new_patched_trials": len(halves), "reused_full_window_trials": len(full),
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "classifications": {cell: {key: value[key] for key in ("category", "continue_bisection", "suggested_segments", "retention_left", "retention_right", "interaction_estimate")} for cell, value in classifications.items()},
    }
    atomic_json(STAGE_ROOT / "summary.json", summary)
    return summary


def _plot(rows: Sequence[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (window, layer) in zip(axes, SELECTED_CELLS):
        values = [row for row in rows if row["window"] == window and int(row["layer"]) == layer]
        order = ("full8", "left4", "right4", "interaction")
        lookup = {row["component"]: row for row in values}
        points = np.asarray([lookup[name]["estimate"] for name in order])
        lows = np.asarray([lookup[name]["ci_low"] for name in order])
        highs = np.asarray([lookup[name]["ci_high"] for name in order])
        ax.errorbar(order, points, yerr=[points - lows, highs - points], fmt="o", capsize=4)
        ax.axhline(0, color="black", lw=.8)
        ax.set_title(f"{window} / L{layer}")
        ax.tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("paired high-minus-low donor effect")
    fig.tight_layout()
    fig.savefig(STAGE_ROOT / "figures/bisection_effects.png", dpi=180)
    plt.close(fig)


def verify() -> dict[str, bool]:
    halves, full = _load_trials()
    required = [
        STAGE_ROOT / "summary.json", STAGE_ROOT / "classification.json",
        STAGE_ROOT / "tables/bisection_effects.csv", STAGE_ROOT / "tables/half_window_trials.csv",
        STAGE_ROOT / "figures/bisection_effects.png",
    ]
    gates = {
        "new_trial_count": len(halves) == EXPECTED_NEW_PATCHED,
        "reused_full_count": len(full) == EXPECTED_REUSED_FULL,
        "selected_cells": {(row["window"], int(row["layer"])) for row in halves} == set(SELECTED_CELLS),
        "two_segments": {row["segment"] for row in halves} == set(SEGMENTS),
        "same_frozen_donors": all(str(row["donor_case_id"]) == str(row["matching"]["donor_case_id"]) for row in halves),
        "four_token_hooks": all(row["hook"]["segment_length"] == 4 and len(row["hook"]["positions"]) == 4 for row in halves),
        "hook_invariants": all(row["hook"]["target_exact"] and row["hook"]["outside_exact"] and row["hook"]["applied_count"] == 1 for row in halves),
        "outputs": all(path.is_file() and path.stat().st_size > 0 for path in required),
    }
    if not all(gates.values()):
        raise RuntimeError(f"Stage-2 completion gates failed: {gates}")
    return gates


def run(*, resume: bool) -> dict[str, Any]:
    started = time.time()
    config = _prepare(resume=resume)
    recipients, donors, pairs = _load_parent()
    runtime = load_explicit_fast_runtime()
    modules = resolve_language_modules(runtime.model)
    token_ids = class_token_ids(runtime.processor.tokenizer)
    try:
        patched_forwards = _run_trials(runtime, modules, recipients, donors, pairs, token_ids, resume=resume)
    finally:
        del runtime
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = analyze()
    gates = verify()
    result = {
        "status": "complete", "fingerprint": config["fingerprint"], "new_patched_forwards": patched_forwards,
        "donor_cache_forwards": 0, "summary": summary, "gates": gates, "elapsed_seconds": time.time() - started,
    }
    atomic_json(STAGE_ROOT / "completion.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PANL2CLE selected-window bisection stage 2")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = analyze() if args.analyze_only else run(resume=args.resume)
    except Exception as exc:
        STAGE_ROOT.mkdir(parents=True, exist_ok=True)
        atomic_json(STAGE_ROOT / "failure.json", {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
