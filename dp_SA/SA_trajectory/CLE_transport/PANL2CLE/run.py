from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import torch

from dp_SA.positions import locate_phase1_positions
from dp_SA.prompts import SA_PREFILL
from dp_SA.soft_score import class_token_ids, soft_sa_from_logits
from dp_SA.SA_trajectory.LAT2PANL.run import load_explicit_fast_runtime
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.model_adapter import model_input_device, resolve_language_modules, run_logits_forward

from .config import CLE_LAYER, LAYERS, PARITY_ATOL, SMOKE_CELLS, WINDOWS, require_output_root
from .hooks import EmptyHook, SingleHiddenCapture, WindowCaptureHook, WindowSwapHook
from .io_utils import atomic_bf16_npz, atomic_json, canonical_hash, load_bf16_npz, load_jsonl, sha256_file
from .prepare import _messages
from .windows import locate_swap_windows


def _safe(case_id: str) -> str: return str(case_id).replace("/", "_")
def _shard(value: str, count: int) -> int: return int(hashlib.sha256(value.encode()).hexdigest(), 16) % count


def _smoke_recipient_ids(root: Path) -> set[str]:
    recipients = load_jsonl(root / "artifacts/manifests/recipient_manifest.jsonl")
    pairs = load_jsonl(root / "artifacts/manifests/donor_matching.jsonl")
    totals: dict[str, int] = {}
    for pair in pairs:
        totals[pair["recipient_case_id"]] = totals.get(pair["recipient_case_id"], 0) + int(pair["position_distance"]) + int(pair["image_token_distance"]) + int(pair["sequence_length_distance"])
    chosen = []
    for side in ("high_image", "high_text"):
        options = [row for row in recipients if row["sa_side"] == side]
        chosen.append(max(options, key=lambda row: (totals.get(str(row["case_id"]), 0), str(row["case_id"]))))
    return {str(row["case_id"]) for row in chosen}


def selected_rows(root: Path, *, smoke: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    recipients = load_jsonl(root / "artifacts/manifests/recipient_manifest.jsonl")
    pairs = load_jsonl(root / "artifacts/manifests/donor_matching.jsonl")
    if smoke:
        ids = _smoke_recipient_ids(root); recipients = [row for row in recipients if str(row["case_id"]) in ids]
        pairs = [row for row in pairs if str(row["recipient_case_id"]) in ids]
    donor_ids = {str(row["donor_case_id"]) for row in pairs}
    donors = [row for row in load_jsonl(root / "artifacts/manifests/donor_manifest.jsonl") if str(row["case_id"]) in donor_ids]
    return recipients, donors, pairs


def _context(runtime: Any, row: dict[str, Any]) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    messages = _messages(row); rendered = render_continued_assistant(runtime.processor, messages, SA_PREFILL)
    if hashlib.sha256(rendered.encode()).hexdigest() != row["rendered_prompt_sha256"]: raise RuntimeError("Rendered prompt fingerprint changed")
    inputs = prepare_multimodal_inputs(runtime.processor, messages, rendered, device=model_input_device(runtime))
    located_windows = locate_swap_windows(runtime.processor.tokenizer, rendered, inputs)
    for name in WINDOWS:
        if located_windows[name]["processed_indices"] != row["windows"][name]["processed_indices"] or located_windows[name]["token_ids"] != row["windows"][name]["token_ids"]:
            raise RuntimeError(f"Window parity failed: {row['case_id']} {name}")
    phase = locate_phase1_positions(runtime.processor.tokenizer, rendered, inputs, str(row["phase0_raw_answer"]))
    return inputs, located_windows, phase


def _score(logits: torch.Tensor, ids: Sequence[int]) -> dict[str, Any]:
    score = soft_sa_from_logits(logits, ids)
    if len(score["class_logits"]) != 9 or not math.isclose(score["probability_sum"], 1.0, abs_tol=1e-9): raise RuntimeError("Invalid nine-class output")
    return score


def _parity(row: dict[str, Any], score: dict[str, Any]) -> dict[str, Any]:
    expected = np.asarray(row["class_logits"], dtype=np.float64); actual = np.asarray(score["class_logits"], dtype=np.float64)
    error = float(np.max(np.abs(expected - actual))); soft_error = abs(float(row["soft_sa_image_score"]) - float(score["soft_sa_image_score"]))
    hard = int(row["argmax_hard_class"]) == int(score["argmax_hard_class"])
    passed = error <= PARITY_ATOL and soft_error <= PARITY_ATOL and hard
    if not passed: raise RuntimeError(f"Clean parity failed for {row['case_id']}: logits={error} soft={soft_error} hard={hard}")
    return {"passed": True, "logits_max_abs_error": error, "soft_sa_abs_error": soft_error, "hard_equal": hard}


def _probe(root: Path) -> tuple[Any, dict[str, Any]]:
    record = json.loads((root / "artifacts/diagnostics/cle_probe.json").read_text()); path = Path(record["absolute_path"])
    if sha256_file(path) != record["probe_sha256"]: raise RuntimeError("CLE probe changed after prepare")
    return joblib.load(path), record


def _probe_value(hidden: torch.Tensor, payload: dict[str, Any]) -> float:
    array = hidden.numpy().reshape(1, -1)
    predicted = float(payload["model"].predict(array)[0])
    raw = float(array.reshape(-1) @ np.asarray(payload["raw_weight"]) + float(payload["raw_intercept"]))
    if abs(predicted - raw) > 1e-6: raise RuntimeError("CLE probe expression mismatch")
    return predicted


def _donor_stage(root: Path, *, smoke: bool, rank: int, world: int, resume: bool) -> dict[str, Any]:
    _recipients, donors, _pairs = selected_rows(root, smoke=smoke); donors = [row for row in donors if _shard(str(row["case_id"]), world) == rank]
    pending = []
    for row in donors:
        destination = root / "artifacts/donor_hidden" / f"{_safe(row['case_id'])}.npz"
        if destination.exists():
            if not resume: raise FileExistsError(destination)
        else: pending.append(row)
    if not pending: return {"stage": "donor", "rank": rank, "new_gpu_forwards": 0, "resumed_noop": True}
    runtime = load_explicit_fast_runtime(); modules = resolve_language_modules(runtime.model); ids = class_token_ids(runtime.processor.tokenizer)
    forwards = 0
    for row in pending:
        inputs, windows, phase = _context(runtime, row)
        targets = {name: window["processed_indices"] for name, window in windows.items()}
        hook = WindowCaptureHook(modules, windows=targets, layers=LAYERS, prefill_length=int(inputs.input_ids.shape[1]))
        with hook: logits = run_logits_forward(runtime.model, inputs, [int(phase["P1_SAC"]["processed_index"])], modules)[int(phase["P1_SAC"]["processed_index"])]
        hook.validate(); parity = _parity(row, _score(logits, ids))
        destination = root / "artifacts/donor_hidden" / f"{_safe(row['case_id'])}.npz"
        metadata = atomic_bf16_npz(destination, hook.values)
        for key, value in hook.values.items():
            restored, meta = load_bf16_npz(destination, key)
            if not torch.equal(restored, value.cpu()) or meta["bits_sha256"] != metadata[key]["bits_sha256"]: raise RuntimeError("BF16 round-trip failed")
        atomic_json(root / "artifacts/donor_hidden" / f"{_safe(row['case_id'])}.json", {"case_id": row["case_id"], "parity": parity, "metadata": metadata, "config_fingerprint": json.loads((root / "run_config.json").read_text())["fingerprint"]})
        forwards += 1; del inputs
    return {"stage": "donor", "rank": rank, "new_gpu_forwards": forwards, "resumed_noop": False}


def _margin(logits: Sequence[float], clean_class: int) -> float:
    values = np.asarray(logits, dtype=float); return float(values[clean_class] - np.max(np.delete(values, clean_class)))


def _trial_stage(root: Path, *, smoke: bool, rank: int, world: int, resume: bool) -> dict[str, Any]:
    recipients, _donors, pairs = selected_rows(root, smoke=smoke); recipients = [r for r in recipients if _shard(str(r["case_id"]), world) == rank]
    pending_any = False
    for row in recipients:
        if not (root / "artifacts/trials" / f"{_safe(row['case_id'])}__clean.json").exists(): pending_any = True
    cells = SMOKE_CELLS if smoke else tuple((window, layer) for window in WINDOWS for layer in LAYERS)
    expected = len(recipients) * (1 + (1 if smoke else 0) + len(cells) * 2)
    existing = len([p for p in (root / "artifacts/trials").glob("*.json") if any(p.name.startswith(_safe(r["case_id"]) + "__") for r in recipients)])
    if not pending_any and existing == expected and resume: return {"stage": "trials", "rank": rank, "new_gpu_forwards": 0, "resumed_noop": True}
    runtime = load_explicit_fast_runtime(); modules = resolve_language_modules(runtime.model); ids = class_token_ids(runtime.processor.tokenizer)
    probe, _probe_record = _probe(root); eligible = {r["case_id"]: r for r in load_jsonl(root / "artifacts/manifests/cle_probe_eligibility.jsonl")}
    pair_index = {(p["recipient_case_id"], p["donor_side"]): p for p in pairs}; donor_rows = {r["case_id"]: r for r in load_jsonl(root / "artifacts/manifests/donor_manifest.jsonl")}
    forwards = 0
    for row in recipients:
        case = str(row["case_id"]); inputs, windows, phase = _context(runtime, row); sac = int(phase["P1_SAC"]["processed_index"]); cle = int(phase["P1_CLASS_LIST_END"]["processed_index"])
        clean_path = root / "artifacts/trials" / f"{_safe(case)}__clean.json"
        if clean_path.exists():
            if not resume: raise FileExistsError(clean_path)
            clean = json.loads(clean_path.read_text())
        else:
            capture = SingleHiddenCapture(modules, layer=CLE_LAYER, position=cle, prefill_length=int(inputs.input_ids.shape[1]))
            with capture: clean_logits = run_logits_forward(runtime.model, inputs, [sac], modules)[sac]
            clean_score = _score(clean_logits, ids); parity = _parity(row, clean_score); clean_hidden = capture.validate()
            clean = {"status": "completed", "case_id": case, "family_id": row["family_id"], "item_id": row["item_id"],
                     "answer": row["phase0_raw_answer"], "recipient_side": row["sa_side"], "condition": "clean",
                     "soft_sa": clean_score["soft_sa_image_score"], "hard_sa_class": clean_score["argmax_hard_class"],
                     "class_logits": clean_score["class_logits"], "class_probabilities": clean_score["class_probabilities"],
                     "clean_margin": _margin(clean_score["class_logits"], clean_score["argmax_hard_class"]), "parity": parity,
                     "cle_probe_eligible": eligible[case]["eligible"], "cle_probe_exclusion_reasons": eligible[case]["exclusion_reasons"],
                     "cle_probe_sa": _probe_value(clean_hidden, probe) if eligible[case]["eligible"] else None}
            atomic_json(clean_path, clean); forwards += 1
        if smoke:
            noop_path = root / "artifacts/trials" / f"{_safe(case)}__noop.json"
            if noop_path.exists():
                if not resume: raise FileExistsError(noop_path)
            else:
                hook = EmptyHook(modules, layer=18)
                with hook: noop_logits = run_logits_forward(runtime.model, inputs, [sac], modules)[sac]
                noop_score = _score(noop_logits, ids); exact = noop_score["class_logits"] == clean["class_logits"]
                if not exact or noop_score["soft_sa_image_score"] != clean["soft_sa"] or noop_score["argmax_hard_class"] != clean["hard_sa_class"]:
                    raise RuntimeError(f"Disabled-swap no-op parity failed: {case}")
                atomic_json(noop_path, {"status": "completed", "case_id": case, "condition": "swap_disabled", "hook_count": hook.count, "bitwise_class_logits_equal": exact})
                forwards += 1
        for window, layer in cells:
            for donor_side in ("high_image", "high_text"):
                pair = pair_index[case, donor_side]; donor = donor_rows[pair["donor_case_id"]]
                condition = ("H" if row["sa_side"] == "high_image" else "L") + "_from_" + ("H" if donor_side == "high_image" else "L")
                path = root / "artifacts/trials" / f"{_safe(case)}__{condition}__{window}__L{layer}.json"
                if path.exists():
                    if resume: continue
                    raise FileExistsError(path)
                source, source_meta = load_bf16_npz(root / "artifacts/donor_hidden" / f"{_safe(donor['case_id'])}.npz", f"{window}__L{layer}")
                swap = WindowSwapHook(modules, layer=layer, positions=windows[window]["processed_indices"], source=source, prefill_length=int(inputs.input_ids.shape[1]))
                capture = SingleHiddenCapture(modules, layer=CLE_LAYER, position=cle, prefill_length=int(inputs.input_ids.shape[1]))
                started = time.perf_counter()
                with swap, capture: logits = run_logits_forward(runtime.model, inputs, [sac], modules)[sac]
                elapsed = time.perf_counter() - started; score = _score(logits, ids); downstream = capture.validate()
                delta = float(score["soft_sa_image_score"]) - float(clean["soft_sa"]); donor_gap = float(pair["donor_clean_sa"]) - float(clean["soft_sa"])
                direction = int(donor_gap > 0) - int(donor_gap < 0); patched_distance = abs(float(score["soft_sa_image_score"]) - float(pair["donor_clean_sa"]))
                clean_distance = abs(float(clean["soft_sa"]) - float(pair["donor_clean_sa"]))
                clean_class = int(clean["hard_sa_class"]); margin = _margin(score["class_logits"], clean_class)
                trial = {"status": "completed", "case_id": case, "family_id": row["family_id"], "item_id": row["item_id"],
                         "answer": row["phase0_raw_answer"], "recipient_side": row["sa_side"], "donor_case_id": donor["case_id"],
                         "donor_side": donor_side, "condition": condition, "window": window, "layer": layer,
                         "clean_soft_sa": clean["soft_sa"], "patched_soft_sa": score["soft_sa_image_score"], "delta_soft_sa": delta,
                         "abs_delta_soft_sa": abs(delta), "donor_clean_sa": pair["donor_clean_sa"], "donor_gap": donor_gap,
                         "toward_score": delta * direction, "toward": None if direction == 0 else bool(delta * direction > 0),
                         "zero_donor_gap": direction == 0, "clean_donor_distance": clean_distance, "patched_donor_distance": patched_distance,
                         "donor_distance_reduction": clean_distance - patched_distance, "clean_hard_sa_class": clean_class,
                         "patched_hard_sa_class": score["argmax_hard_class"], "hard_changed": int(score["argmax_hard_class"]) != clean_class,
                         "class_logits": score["class_logits"], "class_probabilities": score["class_probabilities"],
                         "clean_margin": clean["clean_margin"], "patched_clean_class_margin": margin, "margin_change": margin - float(clean["clean_margin"]),
                         "cle_probe_eligible": eligible[case]["eligible"], "clean_cle_probe_sa": clean["cle_probe_sa"],
                         "patched_cle_probe_sa": _probe_value(downstream, probe) if eligible[case]["eligible"] else None,
                         "cle_probe_delta": (_probe_value(downstream, probe) - float(clean["cle_probe_sa"])) if eligible[case]["eligible"] else None,
                         "hook": swap.diagnostics(), "source_bits_sha256": source_meta["bits_sha256"], "matching": pair,
                         "elapsed_seconds": elapsed}
                atomic_json(path, trial); forwards += 1
        del inputs
    return {"stage": "trials", "rank": rank, "new_gpu_forwards": forwards, "resumed_noop": forwards == 0}


def worker(root: Path, *, stage: str, smoke: bool, rank: int, world: int, resume: bool) -> dict[str, Any]:
    result = _donor_stage(root, smoke=smoke, rank=rank, world=world, resume=resume) if stage == "donor" else _trial_stage(root, smoke=smoke, rank=rank, world=world, resume=resume)
    atomic_json(root / f"progress/{stage}_rank{rank}.json", result); return result


def _visible_gpus() -> list[str]:
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    return [x.strip() for x in configured.split(",") if x.strip()] if configured else [str(i) for i in range(torch.cuda.device_count())]


def spawn_stage(root: Path, *, stage: str, smoke: bool, num_gpus: int, resume: bool) -> list[dict[str, Any]]:
    tokens = _visible_gpus()
    if len(tokens) < num_gpus: raise RuntimeError(f"Requested {num_gpus} GPUs, visible={tokens}")
    processes = []
    for rank in range(num_gpus):
        command = [sys.executable, "-m", "dp_SA.SA_trajectory.CLE_transport.PANL2CLE.run", "--worker", "--root", str(root), "--stage", stage, "--rank", str(rank), "--world", str(num_gpus)]
        if smoke: command.append("--smoke")
        if resume: command.append("--resume")
        env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = tokens[rank]
        log = root / f"logs/{stage}_gpu{rank}.log"; handle = log.open("a", encoding="utf-8")
        processes.append((subprocess.Popen(command, cwd=Path(__file__).resolve().parents[4], env=env, stdout=handle, stderr=subprocess.STDOUT), handle, log))
    failures = []
    for process, handle, log in processes:
        code = process.wait(); handle.close()
        if code: failures.append(f"{log}: exit {code}")
    if failures: raise RuntimeError("GPU worker failure: " + "; ".join(failures))
    return [json.loads((root / f"progress/{stage}_rank{rank}.json").read_text()) for rank in range(num_gpus)]


def execute(root: str | Path, *, smoke: bool, num_gpus: int, resume: bool) -> dict[str, Any]:
    root = require_output_root(root); donor = spawn_stage(root, stage="donor", smoke=smoke, num_gpus=num_gpus, resume=resume)
    trials = spawn_stage(root, stage="trials", smoke=smoke, num_gpus=num_gpus, resume=resume)
    result = {"new_gpu_forwards": sum(r["new_gpu_forwards"] for r in donor + trials), "donor": donor, "trials": trials}
    atomic_json(root / "progress/execute.json", result); return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--worker", action="store_true"); parser.add_argument("--root", required=True)
    parser.add_argument("--stage", choices=("donor", "trials")); parser.add_argument("--rank", type=int, default=0); parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--smoke", action="store_true"); parser.add_argument("--resume", action="store_true"); args = parser.parse_args(argv)
    if not args.worker or not args.stage: parser.error("run.py is an internal worker entrypoint")
    print(json.dumps(worker(Path(args.root), stage=args.stage, smoke=args.smoke, rank=args.rank, world=args.world, resume=args.resume), ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())

