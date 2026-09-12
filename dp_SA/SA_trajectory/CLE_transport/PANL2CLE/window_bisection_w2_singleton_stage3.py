from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch

from dp_SA.soft_score import class_token_ids
from dp_SA.SA_trajectory.LAT2PANL.run import load_explicit_fast_runtime
from layer_metacognition.model_adapter import resolve_language_modules, run_logits_forward

from .config import CLE_LAYER, CLE_PROBE_INDEX, CLE_PROBE_ROOT, OUTPUT_PARENT
from .hooks import SingleHiddenCapture
from .io_utils import atomic_csv, atomic_json, bf16_to_uint16, bits_hash, canonical_hash, load_bf16_npz, load_jsonl, sha256_file
from .run import _context, _margin, _parity, _probe, _probe_value, _safe, _score
from .statistics import bh_fdr
from .window_bisection_stage2 import FORMAL_ROOT, STAGE1_ROOT, SegmentSwapHook, _draw_equal_side, _paired_effects, _point_equal_side, _summary
from .window_bisection_stage2_round2 import STAGE_ROOT as ROUND2_ROOT


STAGE_ROOT = OUTPUT_PARENT / "window_bisection_w2_singleton_stage3"
WINDOW = "W2"
SWAP_LAYER = 17
NEXT_CLE_LAYER = 18
PARENT_SEGMENT = "right4_left2"
PARENT_OFFSETS = (4, 6)
SINGLETONS = {
    "right4_left2_token1": (4, 5),
    "right4_left2_token2": (5, 6),
}
EXPECTED_NEW_CLEAN = 50
EXPECTED_NEW_PATCHED = 200
EXPECTED_PARENT_TRIALS = 100
BOOTSTRAP_REPEATS = 2000


def singleton_positions(window_positions: Sequence[int], offsets: tuple[int, int]) -> list[int]:
    values = list(map(int, window_positions))
    if len(values) != 8 or values != list(range(values[0], values[0] + 8)):
        raise ValueError("Singleton splitting requires a contiguous frozen 8-token window")
    start, stop = offsets
    if not (0 <= start < stop <= 8 and stop - start == 1):
        raise ValueError("Singleton segment must contain exactly one token")
    if not (PARENT_OFFSETS[0] <= start and stop <= PARENT_OFFSETS[1]):
        raise ValueError("Singleton must remain inside the selected W2 parent segment")
    return values[start:stop]


def load_next_layer_cle_probe() -> tuple[Any, dict[str, Any]]:
    matches = [row for row in load_jsonl(CLE_PROBE_INDEX)
               if row.get("target") == "final_soft_sa" and row.get("position") == "P1_CLASS_LIST_END"
               and int(row.get("layer", -1)) == NEXT_CLE_LAYER]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one L{NEXT_CLE_LAYER} CLE probe, found {len(matches)}")
    record = matches[0]
    if not record.get("readout_reliable") or not record.get("raw_expression_strict_pass"):
        raise RuntimeError("Next-layer CLE probe is not reliable")
    path = CLE_PROBE_ROOT / str(record["probe_file"])
    if not path.is_file() or sha256_file(path) != record["probe_sha256"]:
        raise RuntimeError("Next-layer CLE probe fingerprint mismatch")
    return joblib.load(path), record


def _files() -> dict[str, Path]:
    return {
        "formal_completion": FORMAL_ROOT / "completion.json",
        "formal_config": FORMAL_ROOT / "run_config.json",
        "recipients": FORMAL_ROOT / "artifacts/manifests/recipient_manifest.jsonl",
        "donors": FORMAL_ROOT / "artifacts/manifests/donor_manifest.jsonl",
        "pairs": FORMAL_ROOT / "artifacts/manifests/donor_matching.jsonl",
        "stage1_completion": STAGE1_ROOT / "completion.json",
        "round2_completion": ROUND2_ROOT / "completion.json",
        "round2_config": ROUND2_ROOT / "run_config.json",
        "probe_index": CLE_PROBE_INDEX,
        "next_layer_cle_probe": CLE_PROBE_ROOT / "artifacts/probes/final_soft_sa__P1_CLASS_LIST_END__L18.joblib",
    }


def _prepare(*, resume: bool) -> dict[str, Any]:
    for relative in ("artifacts/trials", "artifacts/clean_next_layer_cle", "tables", "figures", "progress", "logs"):
        (STAGE_ROOT / relative).mkdir(parents=True, exist_ok=True)
    sources = _files()
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    for name in ("formal_completion", "stage1_completion", "round2_completion"):
        completion = json.loads(sources[name].read_text())
        if completion.get("status") != "complete" or not all(completion.get("gates", {}).values()):
            raise RuntimeError(f"Frozen parent is incomplete: {name}")
    payload = {
        "experiment": "PANL2CLE_W2_singleton_stage3",
        "window": WINDOW, "swap_layer": SWAP_LAYER, "next_cle_layer": NEXT_CLE_LAYER,
        "parent_segment": PARENT_SEGMENT, "parent_offsets": list(PARENT_OFFSETS),
        "singletons": {name: list(offsets) for name, offsets in SINGLETONS.items()},
        "expected_new_clean_forwards": EXPECTED_NEW_CLEAN,
        "expected_new_patched_forwards": EXPECTED_NEW_PATCHED,
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "source_sha256": {name: sha256_file(path) for name, path in sources.items()},
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    payload["fingerprint"] = canonical_hash(payload)
    path = STAGE_ROOT / "run_config.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("fingerprint") != payload["fingerprint"]:
            raise RuntimeError("Stage-3 resume fingerprint mismatch")
        if not resume:
            raise FileExistsError(f"Stage-3 output exists; use --resume: {STAGE_ROOT}")
    else:
        atomic_json(path, payload)
    return payload


def _load_manifests() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    recipients = load_jsonl(_files()["recipients"]); donors = load_jsonl(_files()["donors"]); pairs = load_jsonl(_files()["pairs"])
    if len(recipients) != 50 or len(donors) != 100 or len(pairs) != 100:
        raise RuntimeError("Frozen manifest cardinality changed")
    return recipients, donors, pairs


def _trial_path(case_id: str, condition: str, segment: str) -> Path:
    return STAGE_ROOT / "artifacts/trials" / f"{_safe(case_id)}__{condition}__{WINDOW}__L{SWAP_LAYER}__{segment}.json"


def _run_trials(runtime: Any, modules: Any, recipients: Sequence[dict[str, Any]], donors: Sequence[dict[str, Any]],
                pairs: Sequence[dict[str, Any]], token_ids: Sequence[int], *, resume: bool) -> tuple[int, int]:
    pair_index = {(str(row["recipient_case_id"]), str(row["donor_side"])): row for row in pairs}
    donor_index = {str(row["case_id"]): row for row in donors}
    next_probe, next_probe_record = load_next_layer_cle_probe()
    final_probe, _ = _probe(FORMAL_ROOT)
    clean_forwards = 0; patched_forwards = 0
    for recipient_number, row in enumerate(recipients, 1):
        case_id = str(row["case_id"])
        clean = json.loads((FORMAL_ROOT / "artifacts/trials" / f"{_safe(case_id)}__clean.json").read_text())
        inputs, windows, phase = _context(runtime, row)
        sac = int(phase["P1_SAC"]["processed_index"]); cle = int(phase["P1_CLASS_LIST_END"]["processed_index"])
        clean_path = STAGE_ROOT / "artifacts/clean_next_layer_cle" / f"{_safe(case_id)}.json"
        if clean_path.exists():
            if not resume: raise FileExistsError(clean_path)
            next_clean = json.loads(clean_path.read_text())
        else:
            capture = SingleHiddenCapture(modules, layer=NEXT_CLE_LAYER, position=cle, prefill_length=int(inputs.input_ids.shape[1]))
            with capture:
                logits = run_logits_forward(runtime.model, inputs, [sac], modules)[sac]
            score = _score(logits, token_ids); parity = _parity(row, score)
            next_clean = {
                "status": "completed", "case_id": case_id, "cle_processed_index": cle,
                "probe_layer": NEXT_CLE_LAYER, "probe_position": "P1_CLASS_LIST_END",
                "probe_sha256": next_probe_record["probe_sha256"], "probe_sa": _probe_value(capture.validate(), next_probe),
                "parity": parity,
            }
            atomic_json(clean_path, next_clean); clean_forwards += 1
        for segment, offsets in SINGLETONS.items():
            positions = singleton_positions(windows[WINDOW]["processed_indices"], offsets)
            start, stop = offsets
            for donor_side in ("high_image", "high_text"):
                pair = pair_index[case_id, donor_side]; donor = donor_index[str(pair["donor_case_id"])]
                condition = ("H" if row["sa_side"] == "high_image" else "L") + "_from_" + ("H" if donor_side == "high_image" else "L")
                destination = _trial_path(case_id, condition, segment)
                if destination.exists():
                    if resume: continue
                    raise FileExistsError(destination)
                full_source, source_meta = load_bf16_npz(STAGE1_ROOT / "artifacts/donor_hidden" / f"{_safe(donor['case_id'])}.npz", f"{WINDOW}__L{SWAP_LAYER}")
                source = full_source[start:stop].contiguous()
                swap = SegmentSwapHook(modules, layer=SWAP_LAYER, positions=positions, source=source, prefill_length=int(inputs.input_ids.shape[1]))
                next_capture = SingleHiddenCapture(modules, layer=NEXT_CLE_LAYER, position=cle, prefill_length=int(inputs.input_ids.shape[1]))
                final_capture = SingleHiddenCapture(modules, layer=CLE_LAYER, position=cle, prefill_length=int(inputs.input_ids.shape[1]))
                started = time.perf_counter()
                with swap, next_capture, final_capture:
                    logits = run_logits_forward(runtime.model, inputs, [sac], modules)[sac]
                score = _score(logits, token_ids)
                next_sa = _probe_value(next_capture.validate(), next_probe); final_sa_probe = _probe_value(final_capture.validate(), final_probe)
                delta = float(score["soft_sa_image_score"]) - float(clean["soft_sa"])
                donor_gap = float(pair["donor_clean_sa"]) - float(clean["soft_sa"]); direction = int(donor_gap > 0) - int(donor_gap < 0)
                clean_class = int(clean["hard_sa_class"]); margin = _margin(score["class_logits"], clean_class)
                patched_distance = abs(float(score["soft_sa_image_score"]) - float(pair["donor_clean_sa"])); clean_distance = abs(float(clean["soft_sa"]) - float(pair["donor_clean_sa"]))
                atomic_json(destination, {
                    "status": "completed", "case_id": case_id, "family_id": row["family_id"], "item_id": row["item_id"],
                    "answer": row["phase0_raw_answer"], "recipient_side": row["sa_side"], "donor_case_id": donor["case_id"],
                    "donor_side": donor_side, "condition": condition, "window": WINDOW, "layer": SWAP_LAYER,
                    "segment": segment, "segment_offsets": [start, stop], "segment_positions": positions,
                    "clean_soft_sa": clean["soft_sa"], "patched_soft_sa": score["soft_sa_image_score"],
                    "delta_soft_sa": delta, "abs_delta_soft_sa": abs(delta), "donor_clean_sa": pair["donor_clean_sa"],
                    "donor_gap": donor_gap, "toward_score": delta * direction,
                    "toward": None if direction == 0 else bool(delta * direction > 0), "zero_donor_gap": direction == 0,
                    "clean_donor_distance": clean_distance, "patched_donor_distance": patched_distance,
                    "donor_distance_reduction": clean_distance - patched_distance,
                    "clean_hard_sa_class": clean_class, "patched_hard_sa_class": score["argmax_hard_class"],
                    "hard_changed": int(score["argmax_hard_class"]) != clean_class,
                    "class_logits": score["class_logits"], "class_probabilities": score["class_probabilities"],
                    "clean_margin": clean["clean_margin"], "patched_clean_class_margin": margin,
                    "margin_change": margin - float(clean["clean_margin"]),
                    "next_layer_cle_probe_layer": NEXT_CLE_LAYER, "next_layer_cle_probe_position": "P1_CLASS_LIST_END",
                    "next_layer_cle_probe_sha256": next_probe_record["probe_sha256"],
                    "clean_next_layer_cle_probe_sa": next_clean["probe_sa"], "patched_next_layer_cle_probe_sa": next_sa,
                    "next_layer_cle_probe_delta": next_sa - float(next_clean["probe_sa"]),
                    "clean_l22_cle_probe_sa": clean["cle_probe_sa"], "patched_l22_cle_probe_sa": final_sa_probe,
                    "l22_cle_probe_delta": final_sa_probe - float(clean["cle_probe_sa"]),
                    "hook": swap.diagnostics(), "full_source_bits_sha256": source_meta["bits_sha256"],
                    "segment_source_bits_sha256": bits_hash(bf16_to_uint16(source)), "matching": pair,
                    "elapsed_seconds": time.perf_counter() - started,
                })
                patched_forwards += 1
                atomic_json(STAGE_ROOT / "progress/trials.json", {
                    "status": "running", "new_clean_forwards": clean_forwards, "new_patched_forwards": patched_forwards,
                    "total_completed": len(list((STAGE_ROOT / "artifacts/trials").glob("*.json"))),
                    "expected": EXPECTED_NEW_PATCHED, "recipient": recipient_number, "last_trial": destination.name,
                })
        del inputs
    atomic_json(STAGE_ROOT / "progress/trials.json", {
        "status": "complete", "new_clean_forwards": clean_forwards, "new_patched_forwards": patched_forwards,
        "total_completed": len(list((STAGE_ROOT / "artifacts/trials").glob("*.json"))), "expected": EXPECTED_NEW_PATCHED,
    })
    return patched_forwards, clean_forwards


def _load_trials() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    new = [json.loads(path.read_text()) for path in sorted((STAGE_ROOT / "artifacts/trials").glob("*.json"))]
    parent = []
    for path in sorted((ROUND2_ROOT / "artifacts/trials").glob("*.json")):
        row = json.loads(path.read_text())
        if row.get("window") == WINDOW and int(row.get("layer", -1)) == SWAP_LAYER and row.get("segment") == PARENT_SEGMENT:
            parent.append(row)
    if len(new) != EXPECTED_NEW_PATCHED or len(parent) != EXPECTED_PARENT_TRIALS:
        raise RuntimeError(f"Stage-3 count mismatch: new={len(new)} parent={len(parent)}")
    return new, parent


def _effect_summary(rows: Sequence[dict[str, Any]], rng: np.random.Generator, *, value_field: str) -> dict[str, Any]:
    converted = [{**row, "patched_soft_sa": row[value_field]} for row in rows]
    records = list(_paired_effects(converted).values())
    draws = _draw_equal_side(records, rng, BOOTSTRAP_REPEATS)
    return {"n_recipients": len(records), **_summary(_point_equal_side(records), draws)}


def analyze() -> dict[str, Any]:
    new, parent = _load_trials(); rng = np.random.default_rng(42)
    final_rows = []
    parent_effect = _effect_summary(parent, rng, value_field="patched_soft_sa")
    final_rows.append({"component": PARENT_SEGMENT, **parent_effect})
    child_effects = {}
    for segment in SINGLETONS:
        values = [row for row in new if row["segment"] == segment]
        summary = _effect_summary(values, rng, value_field="patched_soft_sa")
        child_effects[segment] = summary
        final_rows.append({"component": segment, **summary})
    parent_paired = _paired_effects(parent); child_paired = {name: _paired_effects([row for row in new if row["segment"] == name]) for name in SINGLETONS}
    interaction_records = []
    for case in sorted(parent_paired):
        interaction_records.append({**parent_paired[case], "effect": parent_paired[case]["effect"] - sum(child_paired[name][case]["effect"] for name in SINGLETONS)})
    interaction_draws = _draw_equal_side(interaction_records, rng, BOOTSTRAP_REPEATS)
    interaction = {"n_recipients": len(interaction_records), **_summary(_point_equal_side(interaction_records), interaction_draws)}
    final_rows.append({"component": PARENT_SEGMENT + "__interaction", **interaction})
    p_indices = list(range(1, len(final_rows))); q = bh_fdr([final_rows[i]["bootstrap_p_two_sided"] for i in p_indices])
    for index, value in zip(p_indices, q): final_rows[index]["bh_fdr_q"] = value
    next_probe_rows = []
    for segment in SINGLETONS:
        values = [row for row in new if row["segment"] == segment]
        summary = _effect_summary(values, rng, value_field="patched_next_layer_cle_probe_sa")
        next_probe_rows.append({"segment": segment, **summary})
    q = bh_fdr([row["bootstrap_p_two_sided"] for row in next_probe_rows])
    for row, value in zip(next_probe_rows, q): row["bh_fdr_q"] = value
    parent_value = float(parent_effect["estimate"])
    result = {
        "status": "complete", "window": WINDOW, "swap_layer": SWAP_LAYER, "next_cle_probe_layer": NEXT_CLE_LAYER,
        "parent_effect": parent_effect,
        "singletons": {name: {**summary, "retention_vs_parent": None if math.isclose(parent_value, 0.0, abs_tol=1e-12) else float(summary["estimate"]) / parent_value} for name, summary in child_effects.items()},
        "interaction": interaction, "next_layer_cle_probe": next_probe_rows,
    }
    atomic_csv(STAGE_ROOT / "tables/singleton_final_sa_effects.csv", final_rows)
    atomic_csv(STAGE_ROOT / "tables/next_layer_cle_probe_effects.csv", next_probe_rows)
    atomic_csv(STAGE_ROOT / "tables/singleton_trials.csv", [{key: row.get(key) for key in ("case_id", "family_id", "recipient_side", "donor_case_id", "donor_side", "segment", "delta_soft_sa", "toward_score", "toward", "next_layer_cle_probe_delta", "patched_next_layer_cle_probe_sa", "l22_cle_probe_delta")} for row in new])
    atomic_json(STAGE_ROOT / "summary.json", result); _plot(final_rows, next_probe_rows)
    return result


def _plot(final_rows: Sequence[dict[str, Any]], probe_rows: Sequence[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, rows, title in ((axes[0], final_rows, "Final soft-SA"), (axes[1], probe_rows, "L18 CLE probe")):
        labels = [row.get("component", row.get("segment")) for row in rows]
        points = np.asarray([row["estimate"] for row in rows]); lows = np.asarray([row["ci_low"] for row in rows]); highs = np.asarray([row["ci_high"] for row in rows])
        ax.errorbar(labels, points, yerr=[points - lows, highs - points], fmt="o", capsize=4); ax.axhline(0, color="black", lw=.8)
        ax.set_title(title); ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("paired high-minus-low donor effect")
    fig.tight_layout(); fig.savefig(STAGE_ROOT / "figures/singleton_and_next_layer_cle.png", dpi=180); plt.close(fig)


def verify() -> dict[str, bool]:
    new, parent = _load_trials()
    clean = [json.loads(path.read_text()) for path in sorted((STAGE_ROOT / "artifacts/clean_next_layer_cle").glob("*.json"))]
    required = [STAGE_ROOT / "summary.json", STAGE_ROOT / "tables/singleton_final_sa_effects.csv", STAGE_ROOT / "tables/next_layer_cle_probe_effects.csv", STAGE_ROOT / "tables/singleton_trials.csv", STAGE_ROOT / "figures/singleton_and_next_layer_cle.png"]
    gates = {
        "new_trial_count": len(new) == EXPECTED_NEW_PATCHED, "parent_count": len(parent) == EXPECTED_PARENT_TRIALS,
        "clean_count": len(clean) == EXPECTED_NEW_CLEAN,
        "clean_probe_contract": all(row["probe_layer"] == NEXT_CLE_LAYER and row["probe_position"] == "P1_CLASS_LIST_END" and row["parity"]["passed"] for row in clean),
        "singleton_segments": {row["segment"] for row in new} == set(SINGLETONS),
        "one_token_hooks": all(row["hook"]["segment_length"] == 1 and len(row["hook"]["positions"]) == 1 for row in new),
        "hook_invariants": all(row["hook"]["target_exact"] and row["hook"]["outside_exact"] and row["hook"]["applied_count"] == 1 for row in new),
        "same_frozen_donors": all(str(row["donor_case_id"]) == str(row["matching"]["donor_case_id"]) for row in new),
        "next_layer_probe_contract": all(row["next_layer_cle_probe_layer"] == NEXT_CLE_LAYER and row["next_layer_cle_probe_position"] == "P1_CLASS_LIST_END" and math.isfinite(row["patched_next_layer_cle_probe_sa"]) for row in new),
        "outputs": all(path.is_file() and path.stat().st_size > 0 for path in required),
    }
    if not all(gates.values()): raise RuntimeError(f"Stage-3 completion gates failed: {gates}")
    return gates


def run(*, resume: bool) -> dict[str, Any]:
    started = time.time(); config = _prepare(resume=resume); recipients, donors, pairs = _load_manifests()
    runtime = load_explicit_fast_runtime(); modules = resolve_language_modules(runtime.model); token_ids = class_token_ids(runtime.processor.tokenizer)
    try: patched, clean = _run_trials(runtime, modules, recipients, donors, pairs, token_ids, resume=resume)
    finally:
        del runtime
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    summary = analyze(); gates = verify()
    result = {"status": "complete", "fingerprint": config["fingerprint"], "new_clean_forwards": clean, "new_patched_forwards": patched,
              "total_new_gpu_forwards": clean + patched, "summary": summary, "gates": gates, "elapsed_seconds": time.time() - started}
    atomic_json(STAGE_ROOT / "completion.json", result); return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PANL2CLE W2 two-token parent to singleton stage")
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--analyze-only", action="store_true"); args = parser.parse_args(argv)
    try: result = analyze() if args.analyze_only else run(resume=args.resume)
    except Exception as exc:
        STAGE_ROOT.mkdir(parents=True, exist_ok=True); atomic_json(STAGE_ROOT / "failure.json", {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}); raise
    print(json.dumps(result, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
