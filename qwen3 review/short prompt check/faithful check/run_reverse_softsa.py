from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[2]
for candidate in (REPOSITORY_ROOT, HERE):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from dp_SA.io_utils import append_jsonl, atomic_json, canonical_hash, load_jsonl
from config import MODEL_PATH, OUTPUT_ROOT
from core import signed_soft_sa
from prompts import ANSWER_PREFILL
from runtime import FaithfulQwenRunner
from capture.short_prompt import SA_PREFILL
from capture_reverse.reverse_prompt import phase1_prompt_short_reverse
from capture_reverse.scoring import reverse_soft_sa_from_logits
from Steering.capture import _generate


def run(root: Path, output_root: Path, resume: bool) -> dict[str, Any]:
    root = root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    trials = [row for row in load_jsonl(root / "trials.jsonl") if row.get("status") == "completed"]
    manifests = {row["case_id"]: row for row in load_jsonl(root / "manifest.jsonl")}
    if len(trials) != 110:
        raise ValueError(f"Expected exactly 110 balanced-subset trials, found {len(trials)}")
    if any(row["case_id"] not in manifests for row in trials):
        raise ValueError("A reverse Soft-SA trial has no matching manifest")
    destination = output_root / "reverse_softsa.jsonl"
    existing = {row["case_id"]: row for row in load_jsonl(destination) if row.get("status") == "completed"}
    if existing and not resume:
        raise FileExistsError(f"Reverse output exists: {destination}; use --resume")
    runner = None
    for ordinal, trial in enumerate(sorted(trials, key=lambda row: (row["difficulty"], row["case_id"])), 1):
        if trial["case_id"] in existing:
            continue
        manifest = manifests[trial["case_id"]]
        if runner is None:
            runner = FaithfulQwenRunner(MODEL_PATH)
        fixed = trial["fixed_answer"]
        clue = manifest["original_text"]["text_clue"]
        image = manifest["source_image"]
        prompt = phase1_prompt_short_reverse(trial["question"], clue, fixed)
        _messages, rendered, inputs = runner._inputs(prompt, image, SA_PREFILL)
        logits = runner._vocab_logits(inputs)
        score = reverse_soft_sa_from_logits(logits, runner.sa_class_ids)
        generated_ids, generated_text, _eos = _generate(
            runner.inference, inputs, 1, runner.sa_class_ids,
        )
        if generated_text not in set(map(str, range(9))):
            raise ValueError(f"Invalid constrained reverse Soft-SA output: {generated_text!r}")
        if int(generated_text) != int(score["raw_argmax_class"]):
            raise ValueError("Reverse Soft-SA forward argmax differs from constrained generation")
        record = {
            "status": "completed", "case_id": trial["case_id"], "item_id": trial["item_id"],
            "difficulty": trial["difficulty"], "question": trial["question"],
            "fixed_answer": fixed, "text_clue": clue, "image_path": image,
            "prompt": prompt, "prompt_hash": canonical_hash(prompt),
            "rendered_hash": canonical_hash(rendered),
            "model_fingerprint": runner.model_fingerprint,
            "source_cma_signed": trial["cma_logit"]["cma_signed"],
            "source_cma_log_probability_signed": trial["cma_log_probability"]["cma_signed"],
            **score, "soft_sa_signed": signed_soft_sa(score["soft_sa_image_score"]),
            "raw_generated_class": generated_text, "generated_token_ids": generated_ids,
        }
        append_jsonl(destination, record)
        existing[trial["case_id"]] = record
        if ordinal % 10 == 0:
            atomic_json(output_root / "progress.json", {
                "status": "running", "completed": len(existing), "target": len(trials),
            })
    summary = {
        "status": "complete", "case_count": len(existing),
        "source_root": str(root), "output_root": str(output_root),
        "prompt_orientation": "raw_0_strong_image_to_raw_8_strong_text",
        "canonical_score_orientation": "higher_soft_sa_signed_is_stronger_image",
    }
    atomic_json(output_root / "summary.json", summary)
    atomic_json(output_root / "progress.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reverse-prompt Soft-SA on the 110-case balanced subset")
    parser.add_argument("--root", type=Path, default=OUTPUT_ROOT.parent / "faithful_check_extended" / "balanced_subset")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT.parent / "faithful_check_extended" / "balanced_subset" / "reverse_softsa")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args.root, args.output_root, args.resume), ensure_ascii=False, indent=2))
