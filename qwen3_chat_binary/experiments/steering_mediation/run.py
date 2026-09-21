from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from dp_SA.io_utils import atomic_json, atomic_jsonl, load_jsonl
from layer_metacognition.model_adapter import model_input_device, resolve_language_modules, run_logits_forward
from qwen3_chat_binary.config import MODEL_PATH
from qwen3_chat_binary.conversation import prepare_multimodal_inputs, render_stage2, stage2_messages
from qwen3_chat_binary.positions import locate_positions
from qwen3_chat_binary.prompts import LABELS
from qwen3_chat_binary.runtime import load_qwen3_inference
from qwen3_chat_binary.scoring import attribution_score, label_token_ids
from qwen3_chat_binary.steering import save_torch_atomic

from .config import ALPHAS, CHAINS, LOGIT_PARITY_TOLERANCE, PAIRS, VARIANT, default_output
from .hooks import MediationHook


def _context(runtime: Any, row: dict[str, Any]):
    messages = stage2_messages(row["phase0_prompt"], row["image_path"], row["phase0_raw_output"], VARIANT)
    rendered = render_stage2(runtime.processor, messages)
    inputs = prepare_multimodal_inputs(runtime.processor, messages, rendered, device=model_input_device(runtime))
    located = locate_positions(runtime.processor.tokenizer, rendered, inputs, row["phase0_raw_output"], VARIANT)
    for name in ("LAT", "PANL", "PANL+1", "CLE", "SAC"):
        old, new = row["positions"][name], located["positions"][name]
        if (int(old["processed_index"]), int(old["token_id"])) != (int(new["processed_index"]), int(new["token_id"])):
            raise RuntimeError(f"Position drift for {row['case_id']} at {name}")
    return inputs, located, {k: int(v) for k, v in located["indices"].items()}


def _forward(runtime: Any, modules: Any, inputs: Any, positions: dict[str, int], ids: tuple[int, ...], *,
             upstream: str, downstream: str, steering_layer: int | None = None,
             steering_vector: torch.Tensor | None = None, patch_layer: int | None = None,
             patch_source: torch.Tensor | None = None,
             capture_targets: dict[str, tuple[int, int]] | None = None):
    hook = MediationHook(
        modules, prefill_sequence_length=int(inputs.input_ids.shape[1]),
        upstream_position=positions[upstream], downstream_position=positions[downstream],
        steering_layer=steering_layer, steering_vector=steering_vector,
        patch_layer=patch_layer, patch_source=patch_source,
        capture_targets=capture_targets,
    )
    with hook:
        logits = run_logits_forward(runtime.model, inputs, [positions["SAC"]], modules)[positions["SAC"]]
    hook.validate(); score = attribution_score(logits, ids)
    values = [float(score["label_logits"][label]) for label in LABELS]
    if not all(math.isfinite(x) for x in values): raise RuntimeError("Non-finite mediation logits")
    return score, values, hook


def _trial_path(root: Path, case: str, condition: str, chain: str | None = None,
                upstream_layer: int | None = None, downstream_layer: int | None = None,
                alpha: float = 0) -> Path:
    if condition == "C0": suffix = "C0"
    else:
        sign = "m" if alpha < 0 else "p"
        suffix = f"{condition}__{chain}__U{upstream_layer}__D{downstream_layer}__a{sign}{abs(alpha):g}"
    return root / "artifacts/trials" / f"{case}__{suffix}.json"


def _hidden_path(root: Path, case: str, condition: str, chain: str | None = None,
                 upstream_layer: int | None = None, downstream_layer: int | None = None,
                 alpha: float = 0) -> Path:
    if condition == "C0": suffix = "C0"
    else:
        sign = "m" if alpha < 0 else "p"
        suffix = f"C1__{chain}__U{upstream_layer}__D{downstream_layer}__a{sign}{abs(alpha):g}"
    return root / "artifacts/hidden" / f"{case}__{suffix}.pt"


def _base(row: dict[str, Any], condition: str, score: dict[str, Any], logits: list[float],
          hook: MediationHook, located: dict[str, Any], fingerprint: str, **fields: Any) -> dict[str, Any]:
    return {
        "status": "completed", "case_id": row["case_id"], "test_side": row["test_side"],
        "answer": row["phase0_normalized_answer"], "condition": condition,
        "final_sa": float(score["image_attribution_score"]), "hard_class": score["predicted_label"],
        "label_logits": logits, "label_probabilities": score["label_probabilities"],
        "positions": located, "hook": hook.diagnostics(), "fingerprint": fingerprint, **fields,
    }


def _load_vectors(root: Path) -> dict[tuple[str, int], torch.Tensor]:
    payload = torch.load(root / "artifacts/vectors.pt", map_location="cpu", weights_only=False)
    return {(position, layer): payload[f"{position}__L{layer}"]["scaled_vector"].float()
            for position in {x[0] for x in CHAINS.values()} for layer, _ in PAIRS}


def run(*, output_root: Path, model_path: Path = MODEL_PATH, resume: bool = False) -> dict[str, Any]:
    root = Path(output_root).resolve(); config = json.loads((root / "run_config.json").read_text())
    if Path(config["model"]).resolve() != Path(model_path).resolve():
        raise ValueError("Runtime model path differs from the prepared experiment config")
    fingerprint = config["fingerprint"]; rows = load_jsonl(root / "artifacts/manifests/test.jsonl")
    vectors = _load_vectors(root); runtime = load_qwen3_inference(model_path, attn_implementation="sdpa")
    modules = resolve_language_modules(runtime.model)
    if (modules.num_hidden_layers, modules.hidden_size) != (36, 4096): raise RuntimeError("Unexpected Qwen3 architecture")
    ids = label_token_ids(runtime.processor.tokenizer); new_forwards = 0; started = time.time()
    try:
        for ordinal, row in enumerate(rows, 1):
            inputs, located, positions = _context(runtime, row); case = row["case_id"]
            c0_path, c0_hidden_path = _trial_path(root, case, "C0"), _hidden_path(root, case, "C0")
            capture_targets = {f"{downstream}__L{layer}": (layer, positions[downstream])
                               for downstream in {x[1] for x in CHAINS.values()} for _, layer in PAIRS}
            if resume and c0_path.is_file() and c0_hidden_path.is_file():
                c0 = json.loads(c0_path.read_text()); clean_hidden = torch.load(c0_hidden_path, map_location="cpu", weights_only=False)
            else:
                score, logits, hook = _forward(runtime, modules, inputs, positions, ids,
                    upstream="LAT", downstream="PANL", capture_targets=capture_targets)
                saved = np.asarray([row["label_logits"][label] for label in LABELS]); error = float(np.max(np.abs(saved - logits)))
                if error > LOGIT_PARITY_TOLERANCE: raise RuntimeError(f"C0 parity failed for {case}: {error}")
                clean_hidden = hook.captured; save_torch_atomic(c0_hidden_path, clean_hidden)
                c0 = _base(row, "C0", score, logits, hook, located, fingerprint,
                           chain=None, upstream_position=None, downstream_position=None,
                           upstream_layer=None, downstream_layer=None, alpha=0.0,
                           clean_logit_max_abs_error=error, captured_hidden_file=str(c0_hidden_path.relative_to(root)))
                atomic_json(c0_path, c0); new_forwards += 1
            if c0.get("fingerprint") != fingerprint: raise RuntimeError("C0 fingerprint mismatch")
            for chain, (upstream, downstream) in CHAINS.items():
                for upstream_layer, downstream_layer in PAIRS:
                    clean_donor = clean_hidden[f"{downstream}__L{downstream_layer}"]
                    for alpha in ALPHAS:
                        common = {"chain": chain, "upstream_position": upstream, "downstream_position": downstream,
                                  "upstream_layer": upstream_layer, "downstream_layer": downstream_layer, "alpha": float(alpha)}
                        c1_path = _trial_path(root, case, "C1", chain, upstream_layer, downstream_layer, alpha)
                        c1_hidden_path = _hidden_path(root, case, "C1", chain, upstream_layer, downstream_layer, alpha)
                        if resume and c1_path.is_file() and c1_hidden_path.is_file():
                            corrupted = torch.load(c1_hidden_path, map_location="cpu", weights_only=False)["downstream"]
                        else:
                            score, logits, hook = _forward(runtime, modules, inputs, positions, ids,
                                upstream=upstream, downstream=downstream, steering_layer=upstream_layer,
                                steering_vector=vectors[(upstream, upstream_layer)] * alpha,
                                capture_targets={"downstream": (downstream_layer, positions[downstream])})
                            corrupted = hook.captured["downstream"]; save_torch_atomic(c1_hidden_path, {"downstream": corrupted})
                            atomic_json(c1_path, _base(row, "C1", score, logits, hook, located, fingerprint,
                                        **common, captured_hidden_file=str(c1_hidden_path.relative_to(root))))
                            new_forwards += 1
                        for condition, donor, use_steering, donor_condition in (
                            ("C2", clean_donor, True, "C0"), ("C3", corrupted, False, "C1")):
                            destination = _trial_path(root, case, condition, chain, upstream_layer, downstream_layer, alpha)
                            if resume and destination.is_file(): continue
                            score, logits, hook = _forward(runtime, modules, inputs, positions, ids,
                                upstream=upstream, downstream=downstream,
                                steering_layer=upstream_layer if use_steering else None,
                                steering_vector=vectors[(upstream, upstream_layer)] * alpha if use_steering else None,
                                patch_layer=downstream_layer, patch_source=donor,
                                capture_targets={"downstream": (downstream_layer, positions[downstream])})
                            atomic_json(destination, _base(row, condition, score, logits, hook, located, fingerprint,
                                        **common, donor_condition=donor_condition,
                                        donor_dtype=str(donor.dtype).replace("torch.", "")))
                            new_forwards += 1
            atomic_json(root / "progress/run.json", {"status": "running", "completed_cases": ordinal,
                        "total_cases": len(rows), "new_gpu_forwards": new_forwards})
        trials = [json.loads(p.read_text()) for p in sorted((root / "artifacts/trials").glob("*.json"))]
        expected = len(rows) * (1 + len(CHAINS) * len(PAIRS) * len(ALPHAS) * 3)
        if len(trials) != expected or any(r.get("fingerprint") != fingerprint for r in trials):
            raise RuntimeError(f"Incomplete or mixed mediation grid: {len(trials)}/{expected}")
        atomic_jsonl(root / "artifacts/trials.jsonl", trials)
        result = {"status": "complete", "case_count": len(rows), "trial_count": len(trials),
                  "new_gpu_forwards": new_forwards, "elapsed_seconds": time.time() - started}
        atomic_json(root / "progress/run.json", result); return result
    finally:
        del runtime
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run five-class steering mediation (GPU)")
    parser.add_argument("--output-root", type=Path); parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--smoke", action="store_true"); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run(output_root=args.output_root or default_output(args.smoke), model_path=args.model_path,
                         resume=args.resume), ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
