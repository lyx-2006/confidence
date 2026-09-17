from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[2]
for candidate in (REPOSITORY_ROOT, HERE):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from dp_SA.io_utils import append_jsonl, atomic_json, canonical_hash, load_jsonl
from dp_SA.prompts import SA_PREFILL, phase1_prompt
from dp_SA.soft_score import soft_sa_from_logits
from config import MODEL_PATH, OUTPUT_ROOT
from core import signed_soft_sa
from runtime import FaithfulQwenRunner
from Steering.capture import _generate


def run(root: Path, output_root: Path, resume: bool) -> dict[str, Any]:
    root, output_root = root.resolve(), output_root.resolve(); output_root.mkdir(parents=True, exist_ok=True)
    trials = [r for r in load_jsonl(root / "trials.jsonl") if r.get("status") == "completed"]
    manifests = {r["case_id"]: r for r in load_jsonl(root / "manifest.jsonl")}
    if len(trials) != 110 or any(r["case_id"] not in manifests for r in trials):
        raise ValueError("Full Soft-SA requires exactly 110 valid balanced-subset trials and manifests")
    destination = output_root / "full_softsa.jsonl"
    existing = {r["case_id"]: r for r in load_jsonl(destination) if r.get("status") == "completed"}
    if existing and not resume: raise FileExistsError(f"Output exists: {destination}; use --resume")
    runner = None
    for ordinal, trial in enumerate(sorted(trials, key=lambda r: (r["difficulty"], r["case_id"])), 1):
        if trial["case_id"] in existing: continue
        if runner is None: runner = FaithfulQwenRunner(MODEL_PATH)
        manifest = manifests[trial["case_id"]]; fixed = trial["fixed_answer"]
        prompt = phase1_prompt(trial["question"], manifest["original_text"]["text_clue"], fixed)
        _messages, rendered, inputs = runner._inputs(prompt, manifest["source_image"], SA_PREFILL)
        logits = runner._vocab_logits(inputs); score = soft_sa_from_logits(logits, runner.sa_class_ids)
        generated_ids, generated_text, _eos = _generate(runner.inference, inputs, 1, runner.sa_class_ids)
        if generated_text not in set(map(str, range(9))) or int(generated_text) != int(score["argmax_hard_class"]):
            raise ValueError(f"Full Soft-SA constrained generation mismatch for {trial['case_id']}")
        record = {
            "status": "completed", "case_id": trial["case_id"], "item_id": trial["item_id"],
            "difficulty": trial["difficulty"], "question": trial["question"], "fixed_answer": fixed,
            "text_clue": manifest["original_text"]["text_clue"], "image_path": manifest["source_image"],
            "prompt": prompt, "prompt_hash": canonical_hash(prompt), "rendered_hash": canonical_hash(rendered),
            "model_fingerprint": runner.model_fingerprint, "source_cma_signed": trial["cma_logit"]["cma_signed"],
            "source_cma_log_probability_signed": trial["cma_log_probability"]["cma_signed"],
            **score, "soft_sa_signed": signed_soft_sa(score["soft_sa_image_score"]),
            "raw_generated_class": generated_text, "generated_token_ids": generated_ids,
        }
        append_jsonl(destination, record); existing[trial["case_id"]] = record
        if ordinal % 10 == 0: atomic_json(output_root / "progress.json", {"status": "running", "completed": len(existing), "target": 110})
    summary = {"status": "complete", "case_count": len(existing), "prompt": "standard_full_phase1", "score_orientation": "higher_is_image"}
    atomic_json(output_root / "summary.json", summary); atomic_json(output_root / "progress.json", summary); return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standard full-prompt Soft-SA on 110 balanced cases")
    parser.add_argument("--root", type=Path, default=OUTPUT_ROOT.parent / "faithful_check_extended" / "balanced_subset")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT.parent / "faithful_check_extended" / "balanced_subset" / "full_softsa")
    parser.add_argument("--resume", action="store_true"); return parser.parse_args()


if __name__ == "__main__":
    args = parse_args(); print(json.dumps(run(args.root, args.output_root, args.resume), ensure_ascii=False, indent=2))
