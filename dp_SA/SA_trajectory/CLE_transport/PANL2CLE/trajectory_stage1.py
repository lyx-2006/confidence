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

from dp_SA.positions import locate_phase1_positions
from dp_SA.soft_score import class_token_ids
from dp_SA.SA_trajectory.LAT2PANL.run import load_explicit_fast_runtime
from layer_metacognition.model_adapter import resolve_language_modules, run_logits_forward

from .config import CLE_LAYER, OUTPUT_PARENT, require_output_root
from .hooks import SingleHiddenCapture, WindowCaptureHook, WindowSwapHook
from .io_utils import (
    atomic_bf16_npz,
    atomic_csv,
    atomic_json,
    canonical_hash,
    load_bf16_npz,
    load_jsonl,
    sha256_file,
)
from .run import _context, _margin, _parity, _probe, _probe_value, _safe, _score
from .statistics import donor_contrast


FORMAL_ROOT = OUTPUT_PARENT / "formal"
STAGE_ROOT = OUTPUT_PARENT / "layer_trajectory_stage1"
WINDOWS = ("W2", "W5")
NEW_LAYERS = (13, 14, 16, 17, 19)
REUSED_LAYERS = (15, 18)
ALL_LAYERS = tuple(range(13, 20))
NEW_CELLS = tuple((window, layer) for window in WINDOWS for layer in NEW_LAYERS)
EXPECTED_NEW_PATCHED = 1000
EXPECTED_REUSED_PATCHED = 400
BOOTSTRAP_REPEATS = 2000


def _files() -> dict[str, Path]:
    return {
        "formal_completion": FORMAL_ROOT / "completion.json",
        "formal_config": FORMAL_ROOT / "run_config.json",
        "recipients": FORMAL_ROOT / "artifacts/manifests/recipient_manifest.jsonl",
        "donors": FORMAL_ROOT / "artifacts/manifests/donor_manifest.jsonl",
        "pairs": FORMAL_ROOT / "artifacts/manifests/donor_matching.jsonl",
        "probe": FORMAL_ROOT / "artifacts/diagnostics/cle_probe.json",
    }


def _prepare(*, resume: bool) -> dict[str, Any]:
    root = require_output_root(STAGE_ROOT)
    for relative in ("artifacts/donor_hidden", "artifacts/trials", "artifacts/diagnostics", "tables", "figures", "progress", "logs"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    sources = _files()
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    completion = json.loads(sources["formal_completion"].read_text())
    if completion.get("status") != "complete" or not all(completion.get("gates", {}).values()):
        raise RuntimeError("The frozen formal parent is not complete")
    payload = {
        "experiment": "PANL2CLE_layer_trajectory_stage1",
        "windows": list(WINDOWS),
        "new_layers": list(NEW_LAYERS),
        "reused_layers": list(REUSED_LAYERS),
        "new_cells": [list(cell) for cell in NEW_CELLS],
        "expected_new_patched_forwards": EXPECTED_NEW_PATCHED,
        "source_sha256": {name: sha256_file(path) for name, path in sources.items()},
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    payload["fingerprint"] = canonical_hash(payload)
    destination = root / "run_config.json"
    if destination.exists():
        old = json.loads(destination.read_text())
        if old.get("fingerprint") != payload["fingerprint"]:
            raise RuntimeError("Stage-1 resume fingerprint mismatch")
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
    return STAGE_ROOT / "artifacts/donor_hidden" / f"{_safe(case_id)}.npz"


def _trial_path(case_id: str, condition: str, window: str, layer: int) -> Path:
    return STAGE_ROOT / "artifacts/trials" / f"{_safe(case_id)}__{condition}__{window}__L{layer}.json"


def _capture_donors(runtime: Any, modules: Any, donors: Sequence[dict[str, Any]], token_ids: Sequence[int], *, resume: bool) -> int:
    completed = 0
    for donor_index, row in enumerate(donors, 1):
        destination = _donor_path(str(row["case_id"]))
        if destination.exists():
            if not resume:
                raise FileExistsError(destination)
            continue
        inputs, windows, phase = _context(runtime, row)
        targets = {name: windows[name]["processed_indices"] for name in WINDOWS}
        hook = WindowCaptureHook(modules, windows=targets, layers=NEW_LAYERS, prefill_length=int(inputs.input_ids.shape[1]))
        sac = int(phase["P1_SAC"]["processed_index"])
        with hook:
            logits = run_logits_forward(runtime.model, inputs, [sac], modules)[sac]
        hook.validate()
        parity = _parity(row, _score(logits, token_ids))
        metadata = atomic_bf16_npz(destination, hook.values)
        for key, value in hook.values.items():
            restored, meta = load_bf16_npz(destination, key)
            if not torch.equal(restored, value.cpu()) or meta["bits_sha256"] != metadata[key]["bits_sha256"]:
                raise RuntimeError(f"BF16 donor cache round-trip failed: {row['case_id']} {key}")
        atomic_json(destination.with_suffix(".json"), {
            "case_id": row["case_id"], "parity": parity, "metadata": metadata,
            "stage_fingerprint": json.loads((STAGE_ROOT / "run_config.json").read_text())["fingerprint"],
        })
        completed += 1
        atomic_json(STAGE_ROOT / "progress/donors.json", {
            "status": "running", "new_gpu_forwards": completed, "seen": donor_index,
            "expected": len(donors), "last_case_id": row["case_id"],
        })
        del inputs
    atomic_json(STAGE_ROOT / "progress/donors.json", {
        "status": "complete", "new_gpu_forwards": completed, "expected": len(donors),
        "cache_count": len(list((STAGE_ROOT / "artifacts/donor_hidden").glob("*.npz"))),
    })
    return completed


def _run_trials(runtime: Any, modules: Any, recipients: Sequence[dict[str, Any]], donors: Sequence[dict[str, Any]],
                pairs: Sequence[dict[str, Any]], token_ids: Sequence[int], *, resume: bool) -> int:
    pair_index = {(row["recipient_case_id"], row["donor_side"]): row for row in pairs}
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
        for window, layer in NEW_CELLS:
            for donor_side in ("high_image", "high_text"):
                pair = pair_index[case_id, donor_side]
                donor = donor_index[str(pair["donor_case_id"])]
                condition = ("H" if row["sa_side"] == "high_image" else "L") + "_from_" + ("H" if donor_side == "high_image" else "L")
                destination = _trial_path(case_id, condition, window, layer)
                if destination.exists():
                    if resume:
                        continue
                    raise FileExistsError(destination)
                source, source_meta = load_bf16_npz(_donor_path(str(donor["case_id"])), f"{window}__L{layer}")
                swap = WindowSwapHook(modules, layer=layer, positions=windows[window]["processed_indices"], source=source,
                                      prefill_length=int(inputs.input_ids.shape[1]))
                capture = SingleHiddenCapture(modules, layer=CLE_LAYER, position=cle, prefill_length=int(inputs.input_ids.shape[1]))
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
                    "status": "completed", "source": "new_stage1", "case_id": case_id,
                    "family_id": row["family_id"], "item_id": row["item_id"], "answer": row["phase0_raw_answer"],
                    "recipient_side": row["sa_side"], "donor_case_id": donor["case_id"], "donor_side": donor_side,
                    "condition": condition, "window": window, "layer": layer,
                    "clean_soft_sa": clean["soft_sa"], "patched_soft_sa": score["soft_sa_image_score"],
                    "delta_soft_sa": delta, "abs_delta_soft_sa": abs(delta),
                    "donor_clean_sa": pair["donor_clean_sa"], "donor_gap": donor_gap,
                    "toward_score": delta * direction, "toward": None if direction == 0 else bool(delta * direction > 0),
                    "zero_donor_gap": direction == 0,
                    "clean_donor_distance": clean_distance, "patched_donor_distance": patched_distance,
                    "donor_distance_reduction": clean_distance - patched_distance,
                    "clean_hard_sa_class": clean_class, "patched_hard_sa_class": score["argmax_hard_class"],
                    "hard_changed": int(score["argmax_hard_class"]) != clean_class,
                    "class_logits": score["class_logits"], "class_probabilities": score["class_probabilities"],
                    "clean_margin": clean["clean_margin"], "patched_clean_class_margin": patched_margin,
                    "margin_change": patched_margin - float(clean["clean_margin"]),
                    "cle_probe_eligible": True, "clean_cle_probe_sa": clean["cle_probe_sa"],
                    "patched_cle_probe_sa": patched_cle, "cle_probe_delta": patched_cle - float(clean["cle_probe_sa"]),
                    "hook": swap.diagnostics(), "source_bits_sha256": source_meta["bits_sha256"],
                    "matching": pair, "elapsed_seconds": elapsed,
                }
                atomic_json(destination, trial)
                new_forwards += 1
                atomic_json(STAGE_ROOT / "progress/trials.json", {
                    "status": "running", "new_gpu_forwards": new_forwards,
                    "total_completed": len(list((STAGE_ROOT / "artifacts/trials").glob("*.json"))),
                    "expected": EXPECTED_NEW_PATCHED, "recipient": recipient_number,
                    "last_trial": destination.name,
                })
        del inputs
    total = len(list((STAGE_ROOT / "artifacts/trials").glob("*.json")))
    atomic_json(STAGE_ROOT / "progress/trials.json", {
        "status": "complete", "new_gpu_forwards": new_forwards, "total_completed": total,
        "expected": EXPECTED_NEW_PATCHED,
    })
    return new_forwards


def _load_trials() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    new = [json.loads(path.read_text()) for path in sorted((STAGE_ROOT / "artifacts/trials").glob("*.json"))]
    reused = []
    for path in sorted((FORMAL_ROOT / "artifacts/trials").glob("*.json")):
        row = json.loads(path.read_text())
        if row.get("window") in WINDOWS and int(row.get("layer", -1)) in REUSED_LAYERS:
            reused.append({**row, "source": "reused_formal"})
    if len(new) != EXPECTED_NEW_PATCHED or len(reused) != EXPECTED_REUSED_PATCHED:
        raise RuntimeError(f"Trajectory trial count mismatch: new={len(new)}, reused={len(reused)}")
    return new, reused


def _groups(rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    output: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output[tuple(row[field] for field in fields)].append(row)
    return output


def select_layers(metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"rule": {
        "positive_contrast": "combined high-minus-low donor contrast > 0",
        "strata_preference": "both recipient strata positive is preferred; same nonzero sign is secondary",
        "continuity": "member of a contiguous run of at least 2 layers with positive contrast and positive mean toward score",
        "not_isolated": True,
    }, "windows": {}}
    for window in WINDOWS:
        rows = sorted([dict(row) for row in metrics if row["window"] == window], key=lambda row: int(row["layer"]))
        for row in rows:
            row["positive_contrast"] = float(row["donor_contrast"]) > 0
            row["positive_toward_score"] = float(row["mean_toward_score"]) > 0
            row["both_strata_positive"] = float(row["delta_high_recipient"]) > 0 and float(row["delta_low_recipient"]) > 0
            row["strata_same_sign"] = float(row["delta_high_recipient"]) * float(row["delta_low_recipient"]) > 0
            row["core_positive"] = row["positive_contrast"] and row["positive_toward_score"]
            row["run_length"] = 0
        runs: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for row in rows:
            if row["core_positive"] and (not current or int(row["layer"]) == int(current[-1]["layer"]) + 1):
                current.append(row)
            else:
                if current:
                    runs.append(current)
                current = [row] if row["core_positive"] else []
        if current:
            runs.append(current)
        for run in runs:
            for row in run:
                row["run_length"] = len(run)
        qualifying = [row for row in rows if row["run_length"] >= 2]
        preferred = [row for row in qualifying if row["both_strata_positive"]] or [row for row in qualifying if row["strata_same_sign"]] or qualifying
        recommended = max(preferred, key=lambda row: (row["run_length"], min(float(row["delta_high_recipient"]), float(row["delta_low_recipient"])), float(row["donor_contrast"]))) if preferred else None
        result["windows"][window] = {
            "layers": rows,
            "qualifying_layers": [int(row["layer"]) for row in qualifying],
            "recommended_layer": None if recommended is None else int(recommended["layer"]),
            "recommendation_basis": None if recommended is None else {
                key: recommended[key] for key in ("donor_contrast", "delta_high_recipient", "delta_low_recipient", "mean_toward_score", "run_length", "both_strata_positive")
            },
        }
    return result


def analyze() -> dict[str, Any]:
    new, reused = _load_trials()
    rows = [*new, *reused]
    contrasts = donor_contrast(rows, repeats=BOOTSTRAP_REPEATS, seed=42)
    contrast_lookup = {(r["window"], int(r["layer"]), r["stratum"]): r for r in contrasts}
    condition_rows = []
    for (window, layer, condition), values in sorted(_groups(rows, ("window", "layer", "condition")).items()):
        condition_rows.append({
            "window": window, "layer": layer, "condition": condition, "n": len(values),
            "mean_delta_soft_sa": float(np.mean([r["delta_soft_sa"] for r in values])),
            "mean_abs_delta_soft_sa": float(np.mean([r["abs_delta_soft_sa"] for r in values])),
            "hard_change_rate": float(np.mean([r["hard_changed"] for r in values])),
            "mean_toward_score": float(np.mean([r["toward_score"] for r in values])),
            "toward_rate": float(np.mean([r["toward"] for r in values if not r["zero_donor_gap"]])),
        })
    node_rows = []
    for window in WINDOWS:
        for layer in ALL_LAYERS:
            values = [r for r in rows if r["window"] == window and int(r["layer"]) == layer]
            combined = contrast_lookup[window, layer, "combined_equal_side"]
            high = contrast_lookup[window, layer, "high_image"]
            low = contrast_lookup[window, layer, "high_text"]
            node_rows.append({
                "window": window, "layer": layer, "source": "reused_formal" if layer in REUSED_LAYERS else "new_stage1",
                "n_recipients": 50, "n_trials": len(values),
                "donor_contrast": combined["estimate"], "contrast_sem": combined["sem"],
                "contrast_ci_low": combined["ci_low"], "contrast_ci_high": combined["ci_high"],
                "contrast_p": combined["bootstrap_p_two_sided"], "contrast_bh_fdr_q": combined.get("bh_fdr_q"),
                "delta_high_recipient": high["estimate"], "delta_low_recipient": low["estimate"],
                "mean_toward_score": float(np.mean([r["toward_score"] for r in values])),
                "toward_rate": float(np.mean([r["toward"] for r in values if not r["zero_donor_gap"]])),
                "mean_abs_delta_soft_sa": float(np.mean([r["abs_delta_soft_sa"] for r in values])),
                "hard_change_rate": float(np.mean([r["hard_changed"] for r in values])),
                "mean_cle_probe_delta": float(np.mean([r["cle_probe_delta"] for r in values])),
            })
    selection = select_layers(node_rows)
    flat_keys = ("source", "case_id", "family_id", "answer", "recipient_side", "donor_case_id", "donor_side", "condition", "window", "layer",
                 "clean_soft_sa", "patched_soft_sa", "delta_soft_sa", "abs_delta_soft_sa", "donor_gap", "toward_score", "toward", "hard_changed", "cle_probe_delta")
    atomic_csv(STAGE_ROOT / "tables/combined_trials.csv", [{key: row.get(key) for key in flat_keys} for row in rows])
    atomic_csv(STAGE_ROOT / "tables/layer_trajectory.csv", node_rows)
    atomic_csv(STAGE_ROOT / "tables/donor_contrasts.csv", contrasts)
    atomic_csv(STAGE_ROOT / "tables/condition_summary.csv", condition_rows)
    atomic_json(STAGE_ROOT / "selection.json", selection)
    _plot(node_rows, condition_rows)
    summary = {
        "status": "complete", "new_patched_trials": len(new), "reused_formal_trials": len(reused),
        "combined_trials": len(rows), "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "selected_layers": {window: selection["windows"][window]["recommended_layer"] for window in WINDOWS},
        "qualifying_layers": {window: selection["windows"][window]["qualifying_layers"] for window in WINDOWS},
    }
    atomic_json(STAGE_ROOT / "summary.json", summary)
    return summary


def _plot(nodes: Sequence[dict[str, Any]], conditions: Sequence[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, window in zip(axes, WINDOWS):
        values = sorted([r for r in nodes if r["window"] == window], key=lambda r: r["layer"])
        layers = [r["layer"] for r in values]
        estimate = np.asarray([r["donor_contrast"] for r in values])
        low = np.asarray([r["contrast_ci_low"] for r in values]); high = np.asarray([r["contrast_ci_high"] for r in values])
        ax.errorbar(layers, estimate, yerr=[estimate - low, high - estimate], marker="o", capsize=3, label="combined")
        ax.plot(layers, [r["delta_high_recipient"] for r in values], marker=".", label="high recipient")
        ax.plot(layers, [r["delta_low_recipient"] for r in values], marker=".", label="low recipient")
        ax.axhline(0, color="black", lw=.8); ax.set_title(window); ax.set_xlabel("decoder layer"); ax.legend(fontsize=8)
    axes[0].set_ylabel("high-minus-low donor contrast")
    fig.tight_layout(); fig.savefig(STAGE_ROOT / "figures/layer_trajectory_contrast.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, window in zip(axes, WINDOWS):
        values = sorted([r for r in nodes if r["window"] == window], key=lambda r: r["layer"])
        ax.plot([r["layer"] for r in values], [r["mean_toward_score"] for r in values], marker="o")
        ax.axhline(0, color="black", lw=.8); ax.set_title(window); ax.set_xlabel("decoder layer")
    axes[0].set_ylabel("mean toward score")
    fig.tight_layout(); fig.savefig(STAGE_ROOT / "figures/layer_trajectory_toward.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, window in zip(axes, WINDOWS):
        subset = [r for r in conditions if r["window"] == window]
        for condition in sorted({r["condition"] for r in subset}):
            values = sorted([r for r in subset if r["condition"] == condition], key=lambda r: r["layer"])
            ax.plot([r["layer"] for r in values], [r["mean_delta_soft_sa"] for r in values], marker="o", label=condition)
        ax.axhline(0, color="black", lw=.8); ax.set_title(window); ax.set_xlabel("decoder layer"); ax.legend(fontsize=8)
    axes[0].set_ylabel("mean ΔSA")
    fig.tight_layout(); fig.savefig(STAGE_ROOT / "figures/layer_trajectory_conditions.png", dpi=180); plt.close(fig)


def verify() -> dict[str, bool]:
    new, reused = _load_trials()
    donor_files = list((STAGE_ROOT / "artifacts/donor_hidden").glob("*.npz"))
    nodes = load_jsonl(STAGE_ROOT / "artifacts/diagnostics/node_records.jsonl") if (STAGE_ROOT / "artifacts/diagnostics/node_records.jsonl").exists() else []
    required = [STAGE_ROOT / "summary.json", STAGE_ROOT / "selection.json",
                *[STAGE_ROOT / f"tables/{name}.csv" for name in ("combined_trials", "layer_trajectory", "donor_contrasts", "condition_summary")],
                *[STAGE_ROOT / f"figures/{name}.png" for name in ("layer_trajectory_contrast", "layer_trajectory_toward", "layer_trajectory_conditions")]]
    gates = {
        "new_trial_count": len(new) == EXPECTED_NEW_PATCHED,
        "reused_trial_count": len(reused) == EXPECTED_REUSED_PATCHED,
        "donor_cache_count": len(donor_files) == 100,
        "new_cells": {(r["window"], int(r["layer"])) for r in new} == set(NEW_CELLS),
        "hook_invariants": all(r["hook"]["target_exact"] and r["hook"]["outside_exact"] and r["hook"]["applied_count"] == 1 for r in new),
        "outputs": all(path.is_file() and path.stat().st_size > 0 for path in required),
    }
    if not all(gates.values()):
        raise RuntimeError(f"Stage-1 completion gates failed: {gates}")
    return gates


def run(*, resume: bool) -> dict[str, Any]:
    started = time.time(); config = _prepare(resume=resume)
    recipients, donors, pairs = _load_parent()
    runtime = load_explicit_fast_runtime(); modules = resolve_language_modules(runtime.model)
    token_ids = class_token_ids(runtime.processor.tokenizer)
    try:
        donor_forwards = _capture_donors(runtime, modules, donors, token_ids, resume=resume)
        patched_forwards = _run_trials(runtime, modules, recipients, donors, pairs, token_ids, resume=resume)
    finally:
        del runtime
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = analyze(); gates = verify()
    result = {
        "status": "complete", "fingerprint": config["fingerprint"], "donor_cache_forwards": donor_forwards,
        "new_patched_forwards": patched_forwards, "total_new_gpu_forwards": donor_forwards + patched_forwards,
        "summary": summary, "gates": gates, "elapsed_seconds": time.time() - started,
    }
    atomic_json(STAGE_ROOT / "completion.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PANL2CLE W2/W5 layer-trajectory stage 1")
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

