from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    short_root = Path(__file__).resolve().parents[2]
    for candidate in (short_root, short_root.parent, short_root.parent.parent):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    __package__ = "sa_trajectory.panl2cle"

import numpy as np
import torch

from SA_trajectory.PANL2CLE.hooks import PANLCLEMediationHook
from dp_SA.soft_score import class_token_ids, soft_sa_from_logits
from layer_metacognition.model_adapter import resolve_language_modules, run_logits_forward
from Steering.runtime import load_qwen3_inference
from attention_block.run import _case_context

from .config import ALPHAS, CLE_LAYERS, HIDDEN_SIZE, LOGIT_PARITY_ATOL, MODEL_PATH, NUM_LAYERS, PAIRS, PANL_LAYERS
from .contracts import (
    atomic_bf16_npz, atomic_json, atomic_jsonl, expected_physical_count,
    load_bf16, load_jsonl, physical_trial_key, sha256_file,
)
from .probes import load_probes, probe_value


def _trial_path(root: Path, case_id: str, condition: str, panl: int | None, cle: int | None, alpha: float) -> Path:
    p = "none" if panl is None else f"P{panl}"
    c = "none" if cle is None else f"C{cle}"
    a = f"m{abs(alpha):g}" if alpha < 0 else f"p{alpha:g}"
    return root / "artifacts/trials" / f"{case_id}__{condition}__{p}__{c}__a{a}.json"


def _hidden_path(root: Path, case_id: str, condition: str, panl: int | None = None, alpha: float = 0) -> Path:
    suffix = "C0" if condition == "C0" else f"C1__P{panl}__a{'m' if alpha < 0 else 'p'}{abs(alpha):g}"
    return root / "artifacts/hidden" / f"{case_id}__{suffix}.npz"


def _context(runtime: Any, row: dict[str, Any]):
    inputs, located, all_positions = _case_context(runtime, row)
    positions = {
        "PANL": int(all_positions["P1_PANL"]),
        "CLE": int(all_positions["P1_CLASS_LIST_END"]),
        "SAC": int(all_positions["P1_SAC"]),
    }
    if not positions["PANL"] < positions["CLE"] < positions["SAC"]:
        raise RuntimeError("Short PANL/CLE/SAC causal order failed")
    return inputs, located, positions


def _vectors(root: Path) -> dict[int, torch.Tensor]:
    payload = torch.load(root / "artifacts/vectors/panl_vectors.pt", map_location="cpu", weights_only=False)
    return {layer: payload[f"PANL__L{layer}"]["scaled_vector"].float() for layer in PANL_LAYERS}


def _forward(
    runtime: Any, modules: Any, inputs: Any, positions: dict[str, int], ids: Sequence[int], *,
    panl_layer: int | None, vector: torch.Tensor | None,
    patch_layer: int | None, patch_source: torch.Tensor | None,
    capture_layers: Sequence[int],
):
    hook = PANLCLEMediationHook(
        modules, prefill_sequence_length=int(inputs.input_ids.shape[1]),
        panl_position=positions["PANL"], cle_position=positions["CLE"],
        panl_layer=panl_layer, steering_vector=vector,
        patch_layer=patch_layer, patch_source=patch_source,
        capture_cle_layers=capture_layers,
    )
    with hook:
        logits = run_logits_forward(runtime.model, inputs, [positions["SAC"]], modules)[positions["SAC"]]
    hook.validate()
    scored = soft_sa_from_logits(logits, ids)
    if not all(math.isfinite(float(x)) for x in scored["class_logits"]):
        raise RuntimeError("Non-finite short trajectory logits")
    return scored, hook


def _probe_readouts(hook: PANLCLEMediationHook, probes: dict[int, dict[str, Any]]):
    output = {}
    for layer, hidden in hook.cle_hidden.items():
        value, error = probe_value(hidden, probes[layer])
        output[str(layer)] = {"predicted_soft_sa": value, "raw_expression_error": error}
    return output


def _base(
    row: dict[str, Any], condition: str, panl: int | None, cle: int | None, alpha: float,
    score: dict[str, Any], hook: PANLCLEMediationHook, probes: dict[int, dict[str, Any]],
    positions: dict[str, Any], fingerprint: str,
):
    return {
        "status": "completed", "case_id": row["case_id"], "item_id": str(row["item_id"]),
        "answer": row["phase0_normalized_answer"], "test_side": row["test_side"],
        "condition": condition, "panl_layer": panl, "cle_layer": cle, "alpha": float(alpha),
        "final_soft_sa": float(score["soft_sa_image_score"]),
        "hard_sa_class": int(score["argmax_hard_class"]),
        "class_logits": score["class_logits"], "class_probabilities": score["class_probabilities"],
        "cle_probe": _probe_readouts(hook, probes), "positions": positions,
        "hook": hook.diagnostics(), "fingerprint": fingerprint,
    }


def alpha_zero_gate(root: Path, *, model_path: Path = MODEL_PATH, resume: bool = False):
    path = root / "progress/alpha_zero_gate.json"
    if resume and path.is_file() and json.loads(path.read_text()).get("status") == "passed":
        return {**json.loads(path.read_text()), "resumed_noop": True}
    rows = load_jsonl(root / "artifacts/manifests/test_manifest.jsonl")
    selected = [next(r for r in rows if r["test_side"] == side) for side in ("image_side", "text_side")]
    runtime = load_qwen3_inference(model_path, attn_implementation="sdpa")
    modules = resolve_language_modules(runtime.model)
    ids = class_token_ids(runtime.processor.tokenizer)
    vectors = _vectors(root)
    audits = []
    for row in selected:
        inputs, located, positions = _context(runtime, row)
        clean, clean_hook = _forward(
            runtime, modules, inputs, positions, ids, panl_layer=None, vector=None,
            patch_layer=None, patch_source=None, capture_layers=CLE_LAYERS,
        )
        for panl, cle in PAIRS:
            zero, hook = _forward(
                runtime, modules, inputs, positions, ids, panl_layer=panl,
                vector=vectors[panl] * 0, patch_layer=None, patch_source=None,
                capture_layers=(cle,),
            )
            logit_error = float(np.max(np.abs(np.asarray(clean["class_logits"]) - np.asarray(zero["class_logits"]))))
            probability_error = float(np.max(np.abs(np.asarray(clean["class_probabilities"]) - np.asarray(zero["class_probabilities"]))))
            soft_error = abs(float(clean["soft_sa_image_score"]) - float(zero["soft_sa_image_score"]))
            hidden_equal = torch.equal(
                clean_hook.cle_hidden[cle].view(torch.uint16), hook.cle_hidden[cle].view(torch.uint16)
            )
            passed = (
                max(logit_error, probability_error, soft_error) <= LOGIT_PARITY_ATOL
                and int(clean["argmax_hard_class"]) == int(zero["argmax_hard_class"])
                and hidden_equal and hook.injection_count == 1
            )
            audits.append({
                "case_id": row["case_id"], "test_side": row["test_side"],
                "panl_layer": panl, "cle_layer": cle,
                "logit_max_abs_error": logit_error, "probability_max_abs_error": probability_error,
                "soft_sa_abs_error": soft_error, "cle_hidden_bitwise_equal": hidden_equal,
                "hook": hook.diagnostics(), "passed": passed,
            })
    result = {
        "status": "passed" if all(r["passed"] for r in audits) else "failed",
        "checks": len(audits), "audits": audits, "resumed_noop": False,
    }
    atomic_json(path, result)
    del runtime
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if result["status"] != "passed":
        raise RuntimeError("Short trajectory alpha-zero equivalence gate failed")
    return result


def _run_shard(
    root: Path, *, model_path: Path, resume: bool, rank: int,
    world_size: int, smoke: bool,
):
    config = json.loads((root / "fingerprint.json").read_text())
    fingerprint = config["fingerprint"]
    all_rows = load_jsonl(root / "artifacts/manifests/test_manifest.jsonl")
    if smoke:
        all_rows = [next(r for r in all_rows if r["test_side"] == side) for side in ("image_side", "text_side")]
    rows = [r for i, r in enumerate(sorted(all_rows, key=lambda x: x["case_id"])) if i % world_size == rank]
    index = load_jsonl(root / "artifacts/probes/probe_index.jsonl")
    if len(index) != len(CLE_LAYERS) or not all(r["readout_reliable"] for r in index):
        raise RuntimeError("All three reliable short CLE probes are required")
    probes = load_probes(root)
    vectors = _vectors(root)
    runtime = load_qwen3_inference(model_path, attn_implementation="sdpa")
    modules = resolve_language_modules(runtime.model)
    if (modules.num_hidden_layers, modules.hidden_size) != (NUM_LAYERS, HIDDEN_SIZE):
        raise RuntimeError("Unexpected Qwen3 architecture")
    ids = class_token_ids(runtime.processor.tokenizer)
    new = 0
    started = time.time()
    for row in rows:
        inputs, located, positions = _context(runtime, row)
        case = row["case_id"]
        c0_path = _trial_path(root, case, "C0", None, None, 0)
        c0_hidden = _hidden_path(root, case, "C0")
        if not (resume and c0_path.is_file() and c0_hidden.is_file()):
            score, hook = _forward(
                runtime, modules, inputs, positions, ids, panl_layer=None, vector=None,
                patch_layer=None, patch_source=None, capture_layers=CLE_LAYERS,
            )
            parity = float(np.max(np.abs(np.asarray(score["class_logits"]) - np.asarray(row["class_logits"]))))
            if parity > LOGIT_PARITY_ATOL:
                raise RuntimeError(f"Short trajectory C0 capture parity failed: {case}: {parity}")
            atomic_bf16_npz(c0_hidden, {f"CLE_L{x}": hook.cle_hidden[x] for x in CLE_LAYERS})
            trial = _base(row, "C0", None, None, 0, score, hook, probes, located, fingerprint)
            trial["captured_hidden_file"] = str(c0_hidden.relative_to(root))
            trial["captured_hidden_sha256"] = sha256_file(c0_hidden)
            atomic_json(c0_path, trial)
            new += 1
        elif not resume:
            raise FileExistsError(c0_path)
        for panl, cle in PAIRS:
            for alpha in ALPHAS:
                c1_path = _trial_path(root, case, "C1", panl, cle, alpha)
                c1_hidden = _hidden_path(root, case, "C1", panl, alpha)
                if not (resume and c1_path.is_file() and c1_hidden.is_file()):
                    score, hook = _forward(
                        runtime, modules, inputs, positions, ids, panl_layer=panl,
                        vector=vectors[panl] * alpha, patch_layer=None, patch_source=None,
                        capture_layers=(cle,),
                    )
                    atomic_bf16_npz(c1_hidden, {f"CLE_L{cle}": hook.cle_hidden[cle]})
                    trial = _base(row, "C1", panl, cle, alpha, score, hook, probes, located, fingerprint)
                    trial["captured_hidden_file"] = str(c1_hidden.relative_to(root))
                    trial["captured_hidden_sha256"] = sha256_file(c1_hidden)
                    atomic_json(c1_path, trial)
                    new += 1
                elif not resume:
                    raise FileExistsError(c1_path)
                for condition, source, use_vector in (("C2", c0_hidden, True), ("C3", c1_hidden, False)):
                    destination = _trial_path(root, case, condition, panl, cle, alpha)
                    if resume and destination.is_file():
                        continue
                    if destination.exists():
                        raise FileExistsError(destination)
                    patch = load_bf16(source, f"CLE_L{cle}")
                    score, hook = _forward(
                        runtime, modules, inputs, positions, ids,
                        panl_layer=panl if use_vector else None,
                        vector=vectors[panl] * alpha if use_vector else None,
                        patch_layer=cle, patch_source=patch, capture_layers=(cle,),
                    )
                    trial = _base(row, condition, panl, cle, alpha, score, hook, probes, located, fingerprint)
                    trial["source_hidden_file"] = str(source.relative_to(root))
                    trial["source_hidden_sha256"] = sha256_file(source)
                    atomic_json(destination, trial)
                    new += 1
        atomic_json(root / f"progress/run_rank{rank}.json", {
            "status": "running", "new_gpu_forwards": new,
            "last_case_id": case, "elapsed_seconds": time.time() - started,
        })
    result = {
        "status": "complete", "rank": rank, "world_size": world_size,
        "new_gpu_forwards": new, "elapsed_seconds": time.time() - started,
    }
    atomic_json(root / f"progress/run_rank{rank}.json", result)
    return result


def merge(root: Path, case_count: int):
    rows = [json.loads(p.read_text()) for p in sorted((root / "artifacts/trials").glob("*.json"))]
    fingerprint = json.loads((root / "fingerprint.json").read_text())["fingerprint"]
    if any(row.get("fingerprint") != fingerprint for row in rows):
        raise RuntimeError("Short trajectory trial fingerprint mismatch")
    keys = {physical_trial_key(r) for r in rows}
    expected_count = expected_physical_count(case_count)
    if len(keys) != len(rows) or len(rows) != expected_count:
        raise RuntimeError(f"Physical trial grid incomplete/duplicated: {len(rows)}/{expected_count}")
    expected = {
        "C0": case_count,
        "C1": case_count * len(PAIRS) * len(ALPHAS),
        "C2": case_count * len(PAIRS) * len(ALPHAS),
        "C3": case_count * len(PAIRS) * len(ALPHAS),
    }
    actual = {condition: sum(r["condition"] == condition for r in rows) for condition in expected}
    if actual != expected:
        raise RuntimeError(f"Condition cardinality mismatch: {actual} != {expected}")
    for row in rows:
        relative = row.get("captured_hidden_file") or row.get("source_hidden_file")
        recorded = row.get("captured_hidden_sha256") or row.get("source_hidden_sha256")
        if relative and (not (root / relative).is_file() or sha256_file(root / relative) != recorded):
            raise RuntimeError(f"Hidden source mismatch: {relative}")
    atomic_jsonl(root / "artifacts/trials.jsonl", sorted(rows, key=physical_trial_key))
    return rows


def run(
    *, output_root: Path, model_path: Path = MODEL_PATH, resume: bool = False,
    num_gpus: int = 1, smoke: bool = False,
):
    root = Path(output_root).resolve()
    model_path = Path(model_path).resolve()
    if num_gpus not in (1, 2):
        raise ValueError("--num-gpus must be 1 or 2")
    alpha_zero_gate(root, model_path=model_path, resume=resume)
    if num_gpus == 1:
        workers = [_run_shard(root, model_path=model_path, resume=resume, rank=0, world_size=1, smoke=smoke)]
    else:
        processes = []
        for rank in range(num_gpus):
            command = [
                sys.executable, str(Path(__file__).resolve()), "--output-root", str(root),
                "--model-path", str(model_path), "--worker-rank", str(rank),
                "--world-size", str(num_gpus),
            ] + (["--resume"] if resume else []) + (["--smoke"] if smoke else [])
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(rank)
            log = root / f"progress/gpu{rank}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("w")
            processes.append((rank, subprocess.Popen(command, cwd=Path(__file__).resolve().parents[4], env=env, stdout=handle, stderr=subprocess.STDOUT), handle, log))
        workers = []
        for rank, process, handle, log in processes:
            code = process.wait()
            handle.close()
            if code:
                raise RuntimeError(f"GPU worker {rank} failed: {log}")
            workers.append(json.loads((root / f"progress/run_rank{rank}.json").read_text()))
    case_count = 2 if smoke else 80
    rows = merge(root, case_count)
    result = {
        "status": "complete", "smoke": smoke, "trial_count": len(rows),
        "new_gpu_forwards": sum(x["new_gpu_forwards"] for x in workers),
        "resumed_noop": sum(x["new_gpu_forwards"] for x in workers) == 0,
    }
    atomic_json(root / "progress/run.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--worker-rank", type=int)
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args(argv)
    result = (
        _run_shard(args.output_root, model_path=args.model_path, resume=args.resume,
                   rank=args.worker_rank, world_size=args.world_size, smoke=args.smoke)
        if args.worker_rank is not None else
        run(output_root=args.output_root, model_path=args.model_path, resume=args.resume,
            num_gpus=args.num_gpus, smoke=args.smoke)
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
