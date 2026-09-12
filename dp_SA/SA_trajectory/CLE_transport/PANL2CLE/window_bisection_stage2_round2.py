from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import joblib

from dp_SA.soft_score import class_token_ids
from dp_SA.SA_trajectory.LAT2PANL.run import load_explicit_fast_runtime
from layer_metacognition.model_adapter import resolve_language_modules, run_logits_forward

from .config import CLE_LAYER, CLE_PROBE_INDEX, CLE_PROBE_ROOT, OUTPUT_PARENT
from .hooks import SingleHiddenCapture
from .io_utils import atomic_csv, atomic_json, bf16_to_uint16, bits_hash, canonical_hash, load_bf16_npz, load_jsonl, sha256_file
from .run import _context, _margin, _parity, _probe, _probe_value, _safe, _score
from .statistics import bh_fdr
from .window_bisection_stage2 import (
    FORMAL_ROOT, STAGE1_ROOT, STAGE_ROOT as ROUND1_ROOT, SegmentSwapHook,
    _draw_equal_side, _paired_effects, _point_equal_side, _summary,
)


STAGE_ROOT = OUTPUT_PARENT / "window_bisection_stage2_round2_cle_validation_v2"
SELECTED_CELLS = (("W2", 17), ("W5", 16))
# Absolute offsets within the frozen 8-token window. Names identify each 2-token child.
SPLITS = {
    ("W2", 17): {
        "right4_left2": (4, 6),
        "right4_right2": (6, 8),
    },
    ("W5", 16): {
        "left4_left2": (0, 2),
        "left4_right2": (2, 4),
        "right4_left2": (4, 6),
        "right4_right2": (6, 8),
    },
}
EXPECTED_NEW_PATCHED = 600
EXPECTED_NEW_CLEAN = 50
EXPECTED_PARENT_TRIALS = 400
EXPECTED_FULL_TRIALS = 200
BOOTSTRAP_REPEATS = 2000


def split_positions(window_positions: Sequence[int], offsets: tuple[int, int]) -> list[int]:
    values = list(map(int, window_positions))
    if len(values) != 8 or values != list(range(values[0], values[0] + 8)):
        raise ValueError("Round-2 splitting requires a contiguous frozen 8-token window")
    start, stop = offsets
    if not (0 <= start < stop <= 8 and stop - start == 2):
        raise ValueError("Round-2 segment must contain exactly two tokens")
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
        "stage2_completion": ROUND1_ROOT / "completion.json",
        "stage2_config": ROUND1_ROOT / "run_config.json",
        "probe_index": CLE_PROBE_INDEX,
        "layer16_cle_probe": CLE_PROBE_ROOT / "artifacts/probes/final_soft_sa__P1_CLASS_LIST_END__L16.joblib",
        "layer17_cle_probe": CLE_PROBE_ROOT / "artifacts/probes/final_soft_sa__P1_CLASS_LIST_END__L17.joblib",
    }


def _prepare(*, resume: bool) -> dict[str, Any]:
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)
    for relative in ("artifacts/trials", "tables", "figures", "progress", "logs"):
        (STAGE_ROOT / relative).mkdir(parents=True, exist_ok=True)
    sources = _files()
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    for key in ("formal_completion", "stage1_completion", "stage2_completion"):
        record = json.loads(sources[key].read_text())
        if record.get("status") != "complete" or not all(record.get("gates", {}).values()):
            raise RuntimeError(f"Frozen parent is incomplete: {key}")
    payload = {
        "experiment": "PANL2CLE_window_bisection_stage2_round2",
        "selected_cells": [list(cell) for cell in SELECTED_CELLS],
        "splits": {f"{window}__L{layer}": {name: list(bounds) for name, bounds in children.items()} for (window, layer), children in SPLITS.items()},
        "expected_new_patched_forwards": EXPECTED_NEW_PATCHED,
        "expected_new_clean_forwards": EXPECTED_NEW_CLEAN,
        "parent_4token_trials": EXPECTED_PARENT_TRIALS,
        "full_8token_trials": EXPECTED_FULL_TRIALS,
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "source_sha256": {name: sha256_file(path) for name, path in sources.items()},
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    payload["fingerprint"] = canonical_hash(payload)
    destination = STAGE_ROOT / "run_config.json"
    if destination.exists():
        old = json.loads(destination.read_text())
        if old.get("fingerprint") != payload["fingerprint"]:
            raise RuntimeError("Round-2 resume fingerprint mismatch")
        if not resume:
            raise FileExistsError(f"Round-2 output exists; use --resume: {STAGE_ROOT}")
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


def load_layer_cle_probes() -> dict[int, tuple[Any, dict[str, Any]]]:
    selected = {}
    for row in load_jsonl(CLE_PROBE_INDEX):
        if row.get("target") == "final_soft_sa" and row.get("position") == "P1_CLASS_LIST_END" and int(row.get("layer", -1)) in (16, 17):
            layer = int(row["layer"])
            if not row.get("readout_reliable") or not row.get("raw_expression_strict_pass"):
                raise RuntimeError(f"Unreliable layer-specific CLE probe: L{layer}")
            path = CLE_PROBE_ROOT / str(row["probe_file"])
            if not path.is_file() or sha256_file(path) != row["probe_sha256"]:
                raise RuntimeError(f"Layer-specific CLE probe fingerprint mismatch: L{layer}")
            selected[layer] = (joblib.load(path), row)
    if set(selected) != {16, 17}:
        raise RuntimeError(f"Missing layer-specific CLE probes: {set(selected)}")
    return selected


def _donor_path(case_id: str) -> Path:
    return STAGE1_ROOT / "artifacts/donor_hidden" / f"{_safe(case_id)}.npz"


def _trial_path(case_id: str, condition: str, window: str, layer: int, segment: str) -> Path:
    return STAGE_ROOT / "artifacts/trials" / f"{_safe(case_id)}__{condition}__{window}__L{layer}__{segment}.json"


def _run_trials(runtime: Any, modules: Any, recipients: Sequence[dict[str, Any]], donors: Sequence[dict[str, Any]],
                pairs: Sequence[dict[str, Any]], token_ids: Sequence[int], *, resume: bool) -> tuple[int, int]:
    pair_index = {(str(row["recipient_case_id"]), str(row["donor_side"])): row for row in pairs}
    donor_index = {str(row["case_id"]): row for row in donors}
    probe, _ = _probe(FORMAL_ROOT)
    layer_probes = load_layer_cle_probes()
    patched_forwards = 0
    clean_forwards = 0
    for recipient_number, row in enumerate(recipients, 1):
        case_id = str(row["case_id"])
        clean = json.loads((FORMAL_ROOT / "artifacts/trials" / f"{_safe(case_id)}__clean.json").read_text())
        inputs, windows, phase = _context(runtime, row)
        sac = int(phase["P1_SAC"]["processed_index"])
        cle = int(phase["P1_CLASS_LIST_END"]["processed_index"])
        local_clean_path = STAGE_ROOT / "artifacts/clean_layer_cle" / f"{_safe(case_id)}.json"
        if local_clean_path.exists():
            if not resume:
                raise FileExistsError(local_clean_path)
            local_clean = json.loads(local_clean_path.read_text())
        else:
            captures = {layer: SingleHiddenCapture(modules, layer=layer, position=cle, prefill_length=int(inputs.input_ids.shape[1])) for layer in (16, 17)}
            with captures[16], captures[17]:
                clean_logits = run_logits_forward(runtime.model, inputs, [sac], modules)[sac]
            clean_score = _score(clean_logits, token_ids)
            parity = _parity(row, clean_score)
            local_clean = {
                "status": "completed", "case_id": case_id, "cle_processed_index": cle, "parity": parity,
                "layer_cle_probe_sa": {str(layer): _probe_value(captures[layer].validate(), layer_probes[layer][0]) for layer in (16, 17)},
                "probe_sha256": {str(layer): layer_probes[layer][1]["probe_sha256"] for layer in (16, 17)},
            }
            atomic_json(local_clean_path, local_clean)
            clean_forwards += 1
        for window, layer in SELECTED_CELLS:
            for segment, offsets in SPLITS[(window, layer)].items():
                positions = split_positions(windows[window]["processed_indices"], offsets)
                start, stop = offsets
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
                    swap = SegmentSwapHook(modules, layer=layer, positions=positions, source=source,
                                           prefill_length=int(inputs.input_ids.shape[1]))
                    capture = SingleHiddenCapture(modules, layer=CLE_LAYER, position=cle, prefill_length=int(inputs.input_ids.shape[1]))
                    swap.capture_position = cle
                    started = time.perf_counter()
                    # Registration order is intentional: the same-layer capture observes the output returned by the swap hook.
                    with swap, capture:
                        logits = run_logits_forward(runtime.model, inputs, [sac], modules)[sac]
                    score = _score(logits, token_ids)
                    downstream = capture.validate()
                    patched_cle = _probe_value(downstream, probe)
                    patched_layer_cle = _probe_value(swap.validate_captured_position(), layer_probes[layer][0])
                    clean_layer_cle = float(local_clean["layer_cle_probe_sa"][str(layer)])
                    delta = float(score["soft_sa_image_score"]) - float(clean["soft_sa"])
                    donor_gap = float(pair["donor_clean_sa"]) - float(clean["soft_sa"])
                    direction = int(donor_gap > 0) - int(donor_gap < 0)
                    clean_distance = abs(float(clean["soft_sa"]) - float(pair["donor_clean_sa"]))
                    patched_distance = abs(float(score["soft_sa_image_score"]) - float(pair["donor_clean_sa"]))
                    clean_class = int(clean["hard_sa_class"])
                    patched_margin = _margin(score["class_logits"], clean_class)
                    atomic_json(destination, {
                        "status": "completed", "source": "new_stage2_round2", "case_id": case_id,
                        "family_id": row["family_id"], "item_id": row["item_id"], "answer": row["phase0_raw_answer"],
                        "recipient_side": row["sa_side"], "donor_case_id": donor["case_id"], "donor_side": donor_side,
                        "condition": condition, "window": window, "layer": layer, "segment": segment,
                        "segment_offsets": [start, stop], "full_window_positions": windows[window]["processed_indices"],
                        "segment_positions": positions, "clean_soft_sa": clean["soft_sa"],
                        "patched_soft_sa": score["soft_sa_image_score"], "delta_soft_sa": delta,
                        "abs_delta_soft_sa": abs(delta), "donor_clean_sa": pair["donor_clean_sa"], "donor_gap": donor_gap,
                        "toward_score": delta * direction, "toward": None if direction == 0 else bool(delta * direction > 0),
                        "zero_donor_gap": direction == 0, "clean_donor_distance": clean_distance,
                        "patched_donor_distance": patched_distance, "donor_distance_reduction": clean_distance - patched_distance,
                        "clean_hard_sa_class": clean_class, "patched_hard_sa_class": score["argmax_hard_class"],
                        "hard_changed": int(score["argmax_hard_class"]) != clean_class,
                        "class_logits": score["class_logits"], "class_probabilities": score["class_probabilities"],
                        "clean_margin": clean["clean_margin"], "patched_clean_class_margin": patched_margin,
                        "margin_change": patched_margin - float(clean["clean_margin"]), "cle_probe_eligible": clean["cle_probe_eligible"],
                        "clean_cle_probe_sa": clean["cle_probe_sa"], "patched_cle_probe_sa": patched_cle,
                        "cle_probe_delta": patched_cle - float(clean["cle_probe_sa"]), "hook": swap.diagnostics(),
                        "layer_cle_probe_layer": layer, "layer_cle_probe_position": "P1_CLASS_LIST_END",
                        "layer_cle_probe_sha256": layer_probes[layer][1]["probe_sha256"],
                        "clean_layer_cle_probe_sa": clean_layer_cle, "patched_layer_cle_probe_sa": patched_layer_cle,
                        "layer_cle_probe_delta": patched_layer_cle - clean_layer_cle,
                        "full_source_bits_sha256": source_meta["bits_sha256"], "segment_source_bits_sha256": bits_hash(bf16_to_uint16(source)),
                        "matching": pair, "elapsed_seconds": time.perf_counter() - started,
                    })
                    patched_forwards += 1
                    atomic_json(STAGE_ROOT / "progress/trials.json", {
                        "status": "running", "new_patched_forwards": patched_forwards, "new_clean_forwards": clean_forwards,
                        "total_completed": len(list((STAGE_ROOT / "artifacts/trials").glob("*.json"))),
                        "expected": EXPECTED_NEW_PATCHED, "recipient": recipient_number, "last_trial": destination.name,
                    })
        del inputs
    atomic_json(STAGE_ROOT / "progress/trials.json", {
        "status": "complete", "new_patched_forwards": patched_forwards, "new_clean_forwards": clean_forwards,
        "total_completed": len(list((STAGE_ROOT / "artifacts/trials").glob("*.json"))), "expected": EXPECTED_NEW_PATCHED,
    })
    return patched_forwards, clean_forwards


def _load_trials() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    new = [json.loads(path.read_text()) for path in sorted((STAGE_ROOT / "artifacts/trials").glob("*.json"))]
    parent = [json.loads(path.read_text()) for path in sorted((ROUND1_ROOT / "artifacts/trials").glob("*.json"))]
    full = []
    selected = set(SELECTED_CELLS)
    for path in sorted((STAGE1_ROOT / "artifacts/trials").glob("*.json")):
        row = json.loads(path.read_text())
        if (str(row.get("window")), int(row.get("layer", -1))) in selected:
            full.append({**row, "segment": "full8", "source": "reused_stage1"})
    if len(new) != EXPECTED_NEW_PATCHED or len(parent) != EXPECTED_PARENT_TRIALS or len(full) != EXPECTED_FULL_TRIALS:
        raise RuntimeError(f"Round-2 count mismatch: new={len(new)} parent={len(parent)} full={len(full)}")
    return new, parent, full


def _summarize_cell(window: str, layer: int, new: Sequence[dict[str, Any]], parent: Sequence[dict[str, Any]], full: Sequence[dict[str, Any]], rng: np.random.Generator) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = [row for row in new if row["window"] == window and int(row["layer"]) == layer]
    parent_selected = [row for row in parent if row["window"] == window and int(row["layer"]) == layer]
    full_selected = [row for row in full if row["window"] == window and int(row["layer"]) == layer]
    pieces: dict[str, list[dict[str, Any]]] = {"full8": full_selected}
    for segment in SPLITS[(window, layer)]:
        pieces[segment] = [row for row in selected if row["segment"] == segment]
    # Parent 4-token effects are reused from round 1, while each 2-token child is new.
    for parent_segment in ("left4", "right4"):
        if any(name.startswith(parent_segment + "_") for name in SPLITS[(window, layer)]):
            pieces[parent_segment] = [row for row in parent_selected if row["segment"] == parent_segment]
    paired = {name: _paired_effects(rows) for name, rows in pieces.items()}
    cases = set(paired["full8"])
    if any(set(values) != cases for values in paired.values()):
        raise RuntimeError(f"Recipient sets differ at {window}/L{layer}")
    records = {name: [paired[name][case] for case in sorted(cases)] for name in paired}
    rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for name, values in records.items():
        draws = _draw_equal_side(values, rng, BOOTSTRAP_REPEATS)
        point = _point_equal_side(values)
        summary = _summary(point, draws)
        summaries[name] = summary
        rows.append({"window": window, "layer": layer, "component": name, "n_recipients": len(values), **summary})
    for parent_segment in ("left4", "right4"):
        child_names = [name for name in SPLITS[(window, layer)] if name.startswith(parent_segment + "_")]
        if not child_names:
            continue
        interaction_records = []
        for case in sorted(cases):
            interaction_records.append({**paired[parent_segment][case], "effect": paired[parent_segment][case]["effect"] - sum(paired[name][case]["effect"] for name in child_names)})
        draws = _draw_equal_side(interaction_records, rng, BOOTSTRAP_REPEATS)
        summary = _summary(_point_equal_side(interaction_records), draws)
        name = parent_segment + "__interaction"
        summaries[name] = summary
        rows.append({"window": window, "layer": layer, "component": name, "n_recipients": len(interaction_records), **summary})
    full_effect = summaries["full8"]["estimate"]
    for name, summary in summaries.items():
        summary["retention_vs_full8"] = None if math.isclose(full_effect, 0.0, abs_tol=1e-12) else summary["estimate"] / full_effect
    return rows, {"window": window, "layer": layer, "effects": summaries}


def analyze() -> dict[str, Any]:
    new, parent, full = _load_trials()
    rng = np.random.default_rng(42)
    rows: list[dict[str, Any]] = []
    cells: dict[str, Any] = {}
    for window, layer in SELECTED_CELLS:
        cell_rows, cell = _summarize_cell(window, layer, new, parent, full, rng)
        rows.extend(cell_rows)
        cells[f"{window}__L{layer}"] = cell
    p_indices = [i for i, row in enumerate(rows) if row["component"] != "full8"]
    adjusted = bh_fdr([rows[i]["bootstrap_p_two_sided"] for i in p_indices])
    for index, value in zip(p_indices, adjusted):
        rows[index]["bh_fdr_q"] = value
    layer_probe_rows = []
    for window, layer in SELECTED_CELLS:
        for segment in SPLITS[(window, layer)]:
            values = [row for row in new if row["window"] == window and int(row["layer"]) == layer and row["segment"] == segment]
            probe_values = [{**row, "patched_soft_sa": row["patched_layer_cle_probe_sa"]} for row in values]
            records = list(_paired_effects(probe_values).values())
            draws = _draw_equal_side(records, rng, BOOTSTRAP_REPEATS)
            summary = _summary(_point_equal_side(records), draws)
            final_row = next(row for row in rows if row["window"] == window and int(row["layer"]) == layer and row["component"] == segment)
            layer_probe_rows.append({
                "window": window, "layer": layer, "segment": segment, "n_recipients": len(records),
                "final_sa_donor_contrast": final_row["estimate"], "layer_cle_probe_donor_contrast": summary["estimate"],
                "layer_cle_probe_sem": summary["sem"], "layer_cle_probe_ci_low": summary["ci_low"],
                "layer_cle_probe_ci_high": summary["ci_high"], "layer_cle_probe_p": summary["bootstrap_p_two_sided"],
                "direction_agrees_with_final_sa": int(np.sign(summary["estimate"])) == int(np.sign(final_row["estimate"])),
            })
    probe_q = bh_fdr([row["layer_cle_probe_p"] for row in layer_probe_rows])
    for row, value in zip(layer_probe_rows, probe_q):
        row["layer_cle_probe_bh_fdr_q"] = value
    atomic_csv(STAGE_ROOT / "tables/round2_effects.csv", rows)
    atomic_csv(STAGE_ROOT / "tables/layer_cle_probe_validation.csv", layer_probe_rows)
    atomic_csv(STAGE_ROOT / "tables/two_token_trials.csv", [{key: row.get(key) for key in ("case_id", "family_id", "recipient_side", "donor_case_id", "donor_side", "window", "layer", "segment", "delta_soft_sa", "toward_score", "toward", "hard_changed", "cle_probe_delta", "clean_layer_cle_probe_sa", "patched_layer_cle_probe_sa", "layer_cle_probe_delta", "layer_cle_probe_sha256")} for row in new])
    atomic_json(STAGE_ROOT / "classification.json", {
        "interpretation_boundary": "Two-token perturbations locate natural-hidden segments sensitive to paired donor differences; they do not prove a token stores SA.",
        "cells": cells,
    })
    _plot(rows)
    summary = {"status": "complete", "new_patched_trials": len(new), "parent_4token_trials": len(parent), "full_8token_trials": len(full), "bootstrap_repeats": BOOTSTRAP_REPEATS,
               "cells": {key: {"components": list(value["effects"])} for key, value in cells.items()},
               "layer_cle_probe_validation": layer_probe_rows}
    atomic_json(STAGE_ROOT / "summary.json", summary)
    return summary


def _plot(rows: Sequence[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, (window, layer) in zip(axes, SELECTED_CELLS):
        values = [row for row in rows if row["window"] == window and int(row["layer"]) == layer]
        labels = [row["component"] for row in values]
        points = np.asarray([row["estimate"] for row in values]); lows = np.asarray([row["ci_low"] for row in values]); highs = np.asarray([row["ci_high"] for row in values])
        ax.errorbar(labels, points, yerr=[points - lows, highs - points], fmt="o", capsize=4)
        ax.axhline(0, color="black", lw=.8); ax.set_title(f"{window} / L{layer}"); ax.tick_params(axis="x", rotation=35)
    axes[0].set_ylabel("paired high-minus-low donor effect")
    fig.tight_layout(); fig.savefig(STAGE_ROOT / "figures/round2_effects.png", dpi=180); plt.close(fig)


def verify() -> dict[str, bool]:
    new, parent, full = _load_trials()
    local_clean = [json.loads(path.read_text()) for path in sorted((STAGE_ROOT / "artifacts/clean_layer_cle").glob("*.json"))]
    required = [STAGE_ROOT / "summary.json", STAGE_ROOT / "classification.json", STAGE_ROOT / "tables/round2_effects.csv", STAGE_ROOT / "tables/two_token_trials.csv", STAGE_ROOT / "tables/layer_cle_probe_validation.csv", STAGE_ROOT / "figures/round2_effects.png"]
    gates = {
        "new_trial_count": len(new) == EXPECTED_NEW_PATCHED,
        "parent_count": len(parent) == EXPECTED_PARENT_TRIALS,
        "full_count": len(full) == EXPECTED_FULL_TRIALS,
        "local_clean_count": len(local_clean) == EXPECTED_NEW_CLEAN,
        "local_clean_layers": all(set(row["layer_cle_probe_sa"]) == {"16", "17"} and row["parity"]["passed"] for row in local_clean),
        "selected_cells": {(row["window"], int(row["layer"])) for row in new} == set(SELECTED_CELLS),
        "six_segments": {row["segment"] for row in new} == {name for children in SPLITS.values() for name in children},
        "two_token_hooks": all(row["hook"]["segment_length"] == 2 and len(row["hook"]["positions"]) == 2 for row in new),
        "hook_invariants": all(row["hook"]["target_exact"] and row["hook"]["outside_exact"] and row["hook"]["applied_count"] == 1 for row in new),
        "same_frozen_donors": all(str(row["donor_case_id"]) == str(row["matching"]["donor_case_id"]) for row in new),
        "same_layer_cle_probe": all(row["layer_cle_probe_layer"] == row["layer"] and row["layer_cle_probe_position"] == "P1_CLASS_LIST_END" and math.isfinite(row["patched_layer_cle_probe_sa"]) for row in new),
        "outputs": all(path.is_file() and path.stat().st_size > 0 for path in required),
    }
    if not all(gates.values()):
        raise RuntimeError(f"Round-2 completion gates failed: {gates}")
    return gates


def run(*, resume: bool) -> dict[str, Any]:
    started = time.time(); config = _prepare(resume=resume)
    recipients, donors, pairs = _load_parent()
    runtime = load_explicit_fast_runtime(); modules = resolve_language_modules(runtime.model); token_ids = class_token_ids(runtime.processor.tokenizer)
    try:
        patched_forwards, clean_forwards = _run_trials(runtime, modules, recipients, donors, pairs, token_ids, resume=resume)
    finally:
        del runtime
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    summary = analyze(); gates = verify()
    result = {"status": "complete", "fingerprint": config["fingerprint"], "new_patched_forwards": patched_forwards,
              "new_clean_forwards": clean_forwards, "total_new_gpu_forwards": patched_forwards + clean_forwards,
              "summary": summary, "gates": gates, "elapsed_seconds": time.time() - started}
    atomic_json(STAGE_ROOT / "completion.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PANL2CLE selected-window two-token bisection")
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = analyze() if args.analyze_only else run(resume=args.resume)
    except Exception as exc:
        STAGE_ROOT.mkdir(parents=True, exist_ok=True)
        atomic_json(STAGE_ROOT / "failure.json", {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
