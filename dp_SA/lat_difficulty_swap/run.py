from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from dp_SA.patching.protocol import prepare_delayed_case
from dp_SA.soft_score import class_token_ids
from layer_metacognition.model_adapter import load_qwen_inference, resolve_language_modules, run_hooked_forward

from .build_pairs import build_pair_artifacts
from .config import (
    BOOTSTRAP_REPEATS, JOINED_PATH, LOGIT_PARITY_TOLERANCE, MODEL_PATH,
    PANL_ARTIFACTS, PANL_CAPTURE_PATH, PANL_READOUT_BY_SWAP_LAYER,
    POSITIONS, PROBE_PARITY_TOLERANCE, RESULTS_ROOT, SMOKE_BOOTSTRAP_REPEATS,
    SOFT_SA_PARITY_TOLERANCE, SWAP_LAYERS,
)
from .hooks import LATSwapHook
from .io_utils import append_jsonl, atomic_json, atomic_jsonl, canonical_hash, load_jsonl, sha256_file, stage_update
from .metrics import directional_metrics, hard_direction, score_logits
from .probe_runtime import ProbeBank, prepare_probe_models


def direction_specs(pair: dict[str, Any]) -> list[dict[str, Any]]:
    if pair["arm"] == "A":
        return [
            {"condition": "A_E2H", "recipient_case_id": pair["hard_case_id"], "donor_case_id": pair["easy_case_id"], "target_sign": 1, "recipient_level": "hard", "donor_level": "easy", "clean_condition": "A_HARD_CLEAN", "self_condition": "A_HARD_SELF"},
            {"condition": "A_H2E", "recipient_case_id": pair["easy_case_id"], "donor_case_id": pair["hard_case_id"], "target_sign": -1, "recipient_level": "easy", "donor_level": "hard", "clean_condition": "A_EASY_CLEAN", "self_condition": "A_EASY_SELF"},
        ]
    return [
        {"condition": "B_TE2TH", "recipient_case_id": pair["hard_case_id"], "donor_case_id": pair["easy_case_id"], "target_sign": -1, "recipient_level": "hard", "donor_level": "easy", "clean_condition": "B_TEXT_HARD_CLEAN", "self_condition": "B_TEXT_HARD_SELF"},
        {"condition": "B_TH2TE", "recipient_case_id": pair["easy_case_id"], "donor_case_id": pair["hard_case_id"], "target_sign": 1, "recipient_level": "easy", "donor_level": "hard", "clean_condition": "B_TEXT_EASY_CLEAN", "self_condition": "B_TEXT_EASY_SELF"},
    ]


def self_parity(clean_score: dict[str, Any], swap_score: dict[str, Any], clean_probe: dict[str, float], swap_probe: dict[str, float]) -> dict[str, Any]:
    soft = abs(float(swap_score["soft_sa"]) - float(clean_score["soft_sa"])); logits = float(np.max(np.abs(np.asarray(swap_score["class_logits"], dtype=float) - np.asarray(clean_score["class_logits"], dtype=float))))
    probe = max(abs(float(swap_probe[key]) - float(clean_probe[key])) for key in swap_probe)
    result = {"soft_sa_abs_difference": soft, "class_logit_max_abs_difference": logits, "hard_class_equal": int(swap_score["hard_class"]) == int(clean_score["hard_class"]), "probe_max_abs_difference": probe}
    result["passed"] = bool(soft <= SOFT_SA_PARITY_TOLERANCE and logits <= LOGIT_PARITY_TOLERANCE and result["hard_class_equal"] and probe <= PROBE_PARITY_TOLERANCE)
    return result


def _prepare(inference: Any, row: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    _rendered, inputs, details = prepare_delayed_case(inference, row)
    located = details["located"]
    for name in POSITIONS:
        old, new = row["positions"][name], located[name]
        for field in ("processed_index", "rendered_index", "token_id", "token_text"):
            if old[field] != new[field]:
                raise ValueError(f"Position parity failed: {row['case_id']} {name} {field}")
    if list(row["phase1_answer_token_ids"]) != list(located["phase1_answer_token_ids"]):
        raise ValueError(f"Answer token parity failed: {row['case_id']}")
    return inputs, details


def _historical_hidden(case_id: str, capture: dict[str, dict[str, Any]], key: str) -> np.ndarray:
    record = capture[case_id]
    with np.load(PANL_ARTIFACTS.parent / str(record["hidden_file"])) as payload:
        return np.asarray(payload[key])


def _cache_path(root: Path, case_id: str) -> Path:
    return root / "artifacts" / "hidden" / f"clean__{case_id}.pt"


def _atomic_torch(path: Path, payload: Any) -> None:
    from dp_SA.patching.io import atomic_torch_save
    atomic_torch_save(path, payload)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent); os.close(fd)
    try:
        with open(temporary, "wb") as handle: np.savez(handle, **arrays); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def _clean_one(inference: Any, modules: Any, row: dict[str, Any], root: Path, token_ids: Sequence[int], bank: ProbeBank, layers: Sequence[int], capture: dict[str, dict[str, Any]], fingerprint: str) -> tuple[dict[str, Any], int]:
    case_id = str(row["case_id"]); path = _cache_path(root, case_id)
    existing = {str(item["case_id"]): item for item in load_jsonl(root / "artifacts" / "clean_results.jsonl", repair_trailing=True)}
    if case_id in existing:
        if not path.is_file() or existing[case_id].get("cache_sha256") != sha256_file(path):
            raise ValueError(f"Clean cache missing or changed: {case_id}")
        return existing[case_id], 0
    inputs, details = _prepare(inference, row); located = details["located"]
    positions = {"P1_LAT": int(located["P1_LAT"]["processed_index"]), "P1_PANL": int(located["P1_PANL"]["processed_index"])}
    sac = int(located["P1_SAC"]["processed_index"])
    forward = run_hooked_forward(inference.model, inputs, modules, positions, logits_positions=[sac])
    logits = [float(forward.logits_by_position[sac][int(token)].item()) for token in token_ids]; score = score_logits(logits)
    logit_diff = float(np.max(np.abs(np.asarray(logits) - np.asarray(row["class_logits"], dtype=float))))
    soft_diff = abs(score["soft_sa"] - float(row["soft_sa_image_score"]))
    if logit_diff > LOGIT_PARITY_TOLERANCE or soft_diff > SOFT_SA_PARITY_TOLERANCE or score["hard_class"] != int(row["argmax_hard_class"]):
        raise ValueError(f"Clean SA parity failed: {case_id} logits={logit_diff} soft={soft_diff}")
    lat = {}; panl = {}; probes = {}; hidden_checks = []
    for layer in layers:
        lat[layer] = forward.hidden_by_name["P1_LAT"][layer].detach().float().cpu()
        expected = _historical_hidden(case_id, capture, f"P1_LAT__L{layer}")
        equal = bool(np.array_equal(lat[layer].numpy().astype(np.float16), expected)); hidden_checks.append({"position": "P1_LAT", "layer": layer, "equal_fp16": equal})
        if not equal: raise ValueError(f"Clean LAT hidden parity failed: {case_id} L{layer}")
        readout = PANL_READOUT_BY_SWAP_LAYER[layer]
        panl[layer] = forward.hidden_by_name["P1_PANL"][readout].detach().float().cpu()
        expected = _historical_hidden(case_id, capture, f"P1_PANL__L{readout}")
        equal = bool(np.array_equal(panl[layer].numpy().astype(np.float16), expected)); hidden_checks.append({"position": "P1_PANL", "layer": readout, "equal_fp16": equal})
        if not equal: raise ValueError(f"Clean PANL hidden parity failed: {case_id} L{readout}")
        predicted = bank.predict(panl[layer].numpy(), case_id=case_id, layer=readout, fold=int(row["outer_fold"])); historical = bank.historical(case_id, readout)
        for key, value in predicted.items():
            old = historical.get(key)
            if old is not None and abs(value - float(old)) > PROBE_PARITY_TOLERANCE:
                raise ValueError(f"Clean probe parity failed: {case_id} {key} L{readout}")
        probes[str(layer)] = {**predicted, "panl_readout_layer": readout, "historical": historical}
    payload = {"format_version": 1, "run_fingerprint": fingerprint, "case_id": case_id, "lat_by_swap_layer": lat, "panl_by_swap_layer": panl}
    _atomic_torch(path, payload); cache_hash = sha256_file(path)
    record = {"status": "completed", "case_id": case_id, "item_id": str(row["item_id"]), "outer_fold": int(row["outer_fold"]), "positions": located, "score": score, "probes_by_swap_layer": probes, "historical_logit_max_abs_difference": logit_diff, "historical_soft_sa_abs_difference": soft_diff, "hidden_parity": hidden_checks, "cache_file": str(path.relative_to(root)), "cache_sha256": cache_hash}
    append_jsonl(root / "artifacts" / "clean_results.jsonl", record)
    append_jsonl(root / "artifacts" / "probe_predictions" / "clean.jsonl", {"case_id": case_id, "item_id": row["item_id"], "outer_fold": row["outer_fold"], "probes_by_swap_layer": probes})
    del inputs, forward
    return record, 1


def _load_cache(root: Path, clean: dict[str, Any], fingerprint: str) -> dict[str, Any]:
    path = root / str(clean["cache_file"])
    if sha256_file(path) != clean["cache_sha256"]: raise ValueError("Clean cache hash changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["run_fingerprint"] != fingerprint or payload["case_id"] != clean["case_id"]: raise ValueError("Clean cache identity changed")
    return payload


def _trial(inference: Any, modules: Any, recipient: dict[str, Any], donor_hidden: torch.Tensor, clean: dict[str, Any], bank: ProbeBank, token_ids: Sequence[int], root: Path, *, layer: int, kind: str, spec: dict[str, Any], pair: dict[str, Any]) -> dict[str, Any]:
    inputs, details = _prepare(inference, recipient); located = details["located"]
    lat_position = int(located["P1_LAT"]["processed_index"]); panl_position = int(located["P1_PANL"]["processed_index"]); sac = int(located["P1_SAC"]["processed_index"])
    readout = PANL_READOUT_BY_SWAP_LAYER[layer]
    hook = LATSwapHook(modules, layer=layer, recipient_position=lat_position, donor_hidden=donor_hidden, prefill_length=int(inputs.input_ids.shape[1]))
    with hook:
        forward = run_hooked_forward(inference.model, inputs, modules, {"P1_PANL": panl_position}, logits_positions=[sac])
    diagnostics = hook.diagnostics(); logits = [float(forward.logits_by_position[sac][int(token)].item()) for token in token_ids]
    score = score_logits(logits, clean_class=int(clean["score"]["hard_class"])); probe = bank.predict(forward.hidden_by_name["P1_PANL"][readout].detach().float().cpu().numpy(), case_id=str(recipient["case_id"]), layer=readout, fold=int(recipient["outer_fold"]))
    delta = score["soft_sa"] - float(clean["score"]["soft_sa"]); direction = directional_metrics(delta, int(spec["target_sign"])); hard = hard_direction(int(clean["score"]["hard_class"]), int(score["hard_class"]), int(spec["target_sign"]))
    clean_probe = clean["probes_by_swap_layer"][str(layer)]
    probe_delta = {f"{key}_delta": float(value - clean_probe[key]) for key, value in probe.items()}
    condition = spec["self_condition"] if kind == "self" else spec["condition"]
    trial_key = f"{pair['pair_id']}|{condition}|L{layer}"
    hidden_path = root / "artifacts" / "hidden" / f"trial__{hashlib.sha256(trial_key.encode()).hexdigest()[:20]}.npz"
    _atomic_npz(hidden_path, panl_readout=forward.hidden_by_name["P1_PANL"][readout].detach().float().cpu().numpy().astype(np.float16))
    output = {"status": "completed", "trial_key": trial_key, "pair_id": pair["pair_id"], "arm": pair["arm"], "item_id": pair["item_id"], "condition": condition, "swap_kind": kind, "layer": layer, "panl_readout_layer": readout, "target_sign": int(spec["target_sign"]), "recipient_case_id": recipient["case_id"], "donor_case_id": recipient["case_id"] if kind == "self" else spec["donor_case_id"], "recipient_level": spec["recipient_level"], "donor_level": spec["recipient_level"] if kind == "self" else spec["donor_level"], "difficulty_gap_pair": float(pair["difficulty_gap"]), "donor_recipient_difficulty_gap": 0.0 if kind == "self" else (float(pair["difficulty_gap"]) if spec["donor_level"] == "hard" else -float(pair["difficulty_gap"])), "clean_score": clean["score"], "swap_score": score, "clean_probes": {key: clean_probe[key] for key in probe}, "swap_probes": probe, **probe_delta, **direction, **hard, "logit_diff_change": float(score["fixed_clean_class_margin"] - clean["score"]["fixed_clean_class_margin"]), "logit_disruption": float(clean["score"]["fixed_clean_class_margin"] - score["fixed_clean_class_margin"]), "activation_diagnostics": diagnostics, "hidden_file": str(hidden_path.relative_to(root)), "hidden_sha256": sha256_file(hidden_path)}
    if kind == "self":
        parity = self_parity(clean["score"], score, {key: clean_probe[key] for key in probe}, probe)
        output["self_parity"] = parity
        if not parity["passed"]: raise ValueError(f"Self-swap parity failed: {trial_key} {parity}")
    del inputs, forward
    return output


def _select_pairs(root: Path, smoke: bool) -> list[dict[str, Any]]:
    image = load_jsonl(root / "artifacts" / "image_pair_manifest.jsonl"); text = load_jsonl(root / "artifacts" / "text_pair_manifest.jsonl")
    if not smoke: return [*image, *text]
    selected = image[:2]; used = {case for pair in selected for case in (pair["easy_case_id"], pair["hard_case_id"])}
    for pair in text:
        cases = {pair["easy_case_id"], pair["hard_case_id"]}
        if not cases & used:
            selected.append(pair); used |= cases
            if len(selected) == 4: break
    if len(selected) != 4: raise RuntimeError("Could not select four disjoint smoke pairs")
    return selected


def run_experiment(root: Path, *, resume: bool, smoke: bool = False, layers: Sequence[int] | None = None) -> dict[str, Any]:
    selected_layers = tuple(map(int, layers or ((14, 18) if smoke else SWAP_LAYERS)))
    if not selected_layers or any(layer not in SWAP_LAYERS for layer in selected_layers): raise ValueError("Invalid swap layer selection")
    pairs = _select_pairs(root, smoke); expected_trials = {f"{pair['pair_id']}|{spec['self_condition'] if kind == 'self' else spec['condition']}|L{layer}" for pair in pairs for spec in direction_specs(pair) for kind in ("self", "cross") for layer in selected_layers}
    probe_audit = prepare_probe_models(root, resume=True)
    pair_fingerprint = json.loads((root / "progress" / "run_config.json").read_text())["fingerprint"]
    execution = {"pair_fingerprint": pair_fingerprint, "probe_audit_fingerprint": canonical_hash(probe_audit), "smoke": smoke, "layers": list(selected_layers), "pairs": [pair["pair_id"] for pair in pairs], "expected_trials": sorted(expected_trials)}; execution["fingerprint"] = canonical_hash(execution)
    execution_path = root / "progress" / "execution_config.json"
    if execution_path.exists():
        old = json.loads(execution_path.read_text())
        if old.get("fingerprint") != execution["fingerprint"]: raise ValueError("Execution fingerprint mismatch")
        if not resume: raise FileExistsError("Execution already exists; pass --resume")
    else: atomic_json(execution_path, execution)
    completed_rows = load_jsonl(root / "artifacts" / "swap_results.jsonl", repair_trailing=resume); completed = {str(row["trial_key"]): row for row in completed_rows}
    if not set(completed).issubset(expected_trials): raise ValueError("Completed trials fall outside frozen grid")
    completion_path = root / "progress" / "completion.json"
    if completion_path.exists() and set(completed) == expected_trials:
        result = json.loads(completion_path.read_text()); return {**result, "resumed_noop": True, "new_gpu_forwards": 0}
    bank = ProbeBank(root)
    joined = {str(row["case_id"]): row for row in load_jsonl(JOINED_PATH)}; capture = {str(row["case_id"]): row for row in load_jsonl(PANL_CAPTURE_PATH) if row.get("status") == "completed"}
    case_ids = sorted({case for pair in pairs for case in (pair["easy_case_id"], pair["hard_case_id"])})
    inference = load_qwen_inference(str(MODEL_PATH)); modules = resolve_language_modules(inference.model); tokenizer = getattr(inference.processor, "tokenizer", inference.processor); token_ids = class_token_ids(tokenizer)
    clean_by_case = {}; clean_forwards = 0; started = time.time(); stage_update(root, "gpu_run", "running", smoke=smoke, expected_trials=len(expected_trials))
    for case_id in case_ids:
        clean, added = _clean_one(inference, modules, joined[case_id], root, token_ids, bank, selected_layers, capture, execution["fingerprint"]); clean_by_case[case_id] = clean; clean_forwards += added
    position_audit = {"status": "passed", "case_count": len(case_ids), "positions_rebuilt_per_condition": True, "answer_span_and_tokens_exact": True}
    atomic_json(root / "artifacts" / "position_audit.json", position_audit)
    new_trials = 0
    for pair in pairs:
        for spec in direction_specs(pair):
            recipient = joined[str(spec["recipient_case_id"])]; clean = clean_by_case[str(spec["recipient_case_id"])]
            cross_cache = _load_cache(root, clean_by_case[str(spec["donor_case_id"])], execution["fingerprint"]); self_cache = _load_cache(root, clean, execution["fingerprint"])
            for kind, cache in (("self", self_cache), ("cross", cross_cache)):
                for layer in selected_layers:
                    trial_key = f"{pair['pair_id']}|{spec['self_condition'] if kind == 'self' else spec['condition']}|L{layer}"
                    if trial_key in completed: continue
                    row = _trial(inference, modules, recipient, cache["lat_by_swap_layer"][layer], clean, bank, token_ids, root, layer=layer, kind=kind, spec=spec, pair=pair)
                    completed[trial_key] = row; new_trials += 1; atomic_jsonl(root / "artifacts" / "swap_results.jsonl", list(completed.values()))
                    append_jsonl(root / "artifacts" / "probe_predictions" / f"{kind}.jsonl", {"trial_key": trial_key, "pair_id": pair["pair_id"], "item_id": pair["item_id"], "condition": row["condition"], "layer": layer, "panl_readout_layer": row["panl_readout_layer"], "outer_fold": recipient["outer_fold"], "clean": row["clean_probes"], "swap": row["swap_probes"]})
                    stage_update(root, "gpu_run", "running", completed_trials=len(completed), expected_trials=len(expected_trials), clean_forwards=clean_forwards, elapsed_seconds=time.time() - started)
    if set(completed) != expected_trials: raise RuntimeError(f"Incomplete trial grid: {len(completed)}/{len(expected_trials)}")
    self_rows = [row for row in completed.values() if row["swap_kind"] == "self"]
    swap_audit = {"status": "passed", "trial_count": len(completed), "all_hooks_once": all(row["activation_diagnostics"]["applied_count"] == 1 for row in completed.values()), "all_lat_only": all(row["activation_diagnostics"]["other_tokens_equal"] for row in completed.values()), "all_donor_exact": all(row["activation_diagnostics"]["target_exact_after_cast"] for row in completed.values()), "all_finite": all(row["activation_diagnostics"]["finite"] for row in completed.values())}
    self_audit = {"status": "passed", "self_trial_count": len(self_rows), "all_passed": all(row["self_parity"]["passed"] for row in self_rows), "max_soft_difference": max(row["self_parity"]["soft_sa_abs_difference"] for row in self_rows), "max_logit_difference": max(row["self_parity"]["class_logit_max_abs_difference"] for row in self_rows), "max_probe_difference": max(row["self_parity"]["probe_max_abs_difference"] for row in self_rows)}
    atomic_json(root / "artifacts" / "swap_audit.json", swap_audit); atomic_json(root / "artifacts" / "clean_parity_audit.json", {"status": "passed", "case_count": len(case_ids), "soft_tolerance": SOFT_SA_PARITY_TOLERANCE, "logit_tolerance": LOGIT_PARITY_TOLERANCE, "hard_class_exact": True, "self_swap": self_audit})
    result = {"status": "run_complete", "run_fingerprint": execution["fingerprint"], "smoke": smoke, "pair_count": len(pairs), "clean_forward_count": clean_forwards, "swap_forward_count": new_trials, "total_clean_records": len(case_ids), "total_swap_records": len(completed), "elapsed_seconds": time.time() - started, "resumed_noop": False, "new_gpu_forwards": clean_forwards + new_trials}
    atomic_json(completion_path, result); stage_update(root, "gpu_run", "complete", **{key: value for key, value in result.items() if key != "status"})
    del inference, modules; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run delayed-SA LAT difficulty swaps")
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--smoke", action="store_true"); parser.add_argument("--output-root", type=Path); parser.add_argument("--layers", nargs="+", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv); root = (args.output_root or RESULTS_ROOT).resolve()
    if not (root / "artifacts" / "image_pair_manifest.jsonl").is_file(): build_pair_artifacts(root, resume=args.resume)
    try: result = run_experiment(root, resume=args.resume, smoke=args.smoke, layers=args.layers)
    except Exception as exc:
        append_jsonl(root / "progress" / "failures.jsonl", {"stage": "run", "error_type": type(exc).__name__, "error": str(exc), "timestamp": time.time()}); raise
    print(json.dumps(result, ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
