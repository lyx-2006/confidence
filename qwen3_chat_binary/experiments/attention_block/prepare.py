from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from dp_SA.io_utils import atomic_json, atomic_jsonl, canonical_hash, load_jsonl, sha256_file
from qwen3_chat_binary.config import CAPTURE_ROOT, MODEL_PATH
from qwen3_chat_binary.contracts import ensure_fingerprinted_config
from qwen3_chat_binary.conversation import prepare_multimodal_inputs, render_stage2, stage2_messages
from qwen3_chat_binary.layout import capture_config_path, capture_results_path
from qwen3_chat_binary.positions import locate_positions
from qwen3_chat_binary.scoring import label_token_ids

from .config import CONDITIONS, SEED, TEST_PER_SIDE, VARIANT, WINDOWS, default_output
from .core import add_cle_plus_1


def select_extremes(rows: Sequence[dict[str, Any]], per_side: int = TEST_PER_SIDE) -> list[dict[str, Any]]:
    completed = [r for r in rows if r.get("status") == "completed"
                 and float(r["image_attribution_score"]) != 0.5]
    selected: list[dict[str, Any]] = []
    for side, predicate, reverse in (
        ("text_side", lambda x: x < .5, False),
        ("image_side", lambda x: x > .5, True),
    ):
        candidates = [r for r in completed if predicate(float(r["image_attribution_score"]))]
        candidates.sort(key=lambda r: ((-1 if reverse else 1) * float(r["image_attribution_score"]),
                                       str(r["case_id"])))
        if len(candidates) < per_side:
            raise ValueError(f"Need {per_side} {side} cases, found {len(candidates)}")
        selected.extend({**r, "test_side": side, "selection_rank": rank,
                         "selection_rule": "lowest_score" if side == "text_side" else "highest_score"}
                        for rank, r in enumerate(candidates[:per_side], 1))
    if len({str(r["case_id"]) for r in selected}) != 2 * per_side:
        raise AssertionError("Attention test manifest has duplicate case_id values")
    return sorted(selected, key=lambda r: str(r["case_id"]))


def validate_processor(rows: Sequence[dict[str, Any]], model_path: Path) -> dict[str, Any]:
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    tokenizer = processor.tokenizer
    label_token_ids(tokenizer)
    checked = 0
    for row in rows:
        messages = stage2_messages(row["phase0_prompt"], row["image_path"],
                                  row["phase0_raw_output"], VARIANT)
        rendered = render_stage2(processor, messages)
        inputs = prepare_multimodal_inputs(processor, messages, rendered)
        located = add_cle_plus_1(
            locate_positions(tokenizer, rendered, inputs, row["phase0_raw_output"], VARIANT),
            tokenizer, inputs.input_ids,
        )
        for name in ("LAT", "PANL", "PANL+1", "CLE", "SAC"):
            old, new = row["positions"][name], located["positions"][name]
            if (int(old["processed_index"]), int(old["token_id"])) != (
                int(new["processed_index"]), int(new["token_id"])):
                raise RuntimeError(f"Position drift for {row['case_id']} at {name}")
        checked += 1
    return {"checked_cases": checked, "labels_are_single_tokens": True,
            "cle_plus_1_before_sac": True}


def prepare(*, output_root: Path, capture_root: Path = CAPTURE_ROOT,
            model_path: Path = MODEL_PATH, smoke: bool = False,
            resume: bool = False, validate_tokens: bool = True) -> dict[str, Any]:
    root, capture_root, model_path = map(lambda p: Path(p).resolve(),
                                         (output_root, capture_root, model_path))
    existing_path = root / "run_config.json"
    if resume and existing_path.is_file():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        if existing.get("experiment") == "qwen3_chat_fiveway_attention_block":
            if (Path(existing["model"]).resolve() != model_path or
                    Path(existing["capture_root"]).resolve() != capture_root or
                    bool(existing["smoke"]) != bool(smoke)):
                raise ValueError("Legacy attention-block resume arguments differ from frozen config")
            rows = load_jsonl(root / "artifacts/manifests/test_manifest.jsonl")
            return {"status": "complete", "fingerprint": existing["fingerprint"],
                    "case_count": len(rows), "formal_case_count": 100,
                    "side_counts": dict(Counter(r["test_side"] for r in rows)),
                    "resumed_frozen_legacy_config": True}
    capture_config = json.loads(capture_config_path(capture_root).read_text(encoding="utf-8"))
    formal = select_extremes(load_jsonl(capture_results_path(capture_root, VARIANT)))
    selected = ([next(r for r in formal if r["test_side"] == side)
                 for side in ("text_side", "image_side")] if smoke else formal)
    processor_audit = validate_processor(selected, model_path) if validate_tokens else {"skipped": True}
    payload = {
        "format_version": 1, "experiment": "qwen3_chat_fiveway_attention_block",
        "variant": VARIANT, "model": str(model_path), "capture_root": str(capture_root),
        "capture_fingerprint": capture_config["fingerprint"], "smoke": smoke,
        "conditions": list(CONDITIONS), "windows": [list(x) for x in (WINDOWS[:1] if smoke else WINDOWS)],
        "window_semantics": "inclusive_zero_based", "seed": SEED,
        "manifest_fingerprint": canonical_hash(selected),
        "implementation_hashes": {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
    }
    config = ensure_fingerprinted_config(root / "run_config.json", payload, resume=resume,
                                         label="Fiveway attention block")
    atomic_jsonl(root / "artifacts/manifests/test_manifest.jsonl", selected)
    summary = {"case_count": len(selected), "formal_case_count": len(formal),
               "side_counts": dict(Counter(r["test_side"] for r in selected)),
               "processor_audit": processor_audit}
    atomic_json(root / "artifacts/manifests/selection_summary.json", summary)
    return {"status": "complete", "fingerprint": config["fingerprint"], **summary}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare five-class attention blocking")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--capture-root", type=Path, default=CAPTURE_ROOT)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--steered-block", action="store_true")
    parser.add_argument("--skip-processor-validation", action="store_true")
    args = parser.parse_args(argv)
    if args.steered_block:
        from .steered_block import default_output as enhanced_output, prepare as enhanced_prepare
        result = enhanced_prepare(output_root=args.output_root or enhanced_output(args.smoke),
            capture_root=args.capture_root, model_path=args.model_path, smoke=args.smoke,
            resume=args.resume, validate_tokens=not args.skip_processor_validation)
    else:
        result = prepare(output_root=args.output_root or default_output(args.smoke),
                         capture_root=args.capture_root, model_path=args.model_path,
                         smoke=args.smoke, resume=args.resume,
                         validate_tokens=not args.skip_processor_validation)
    print(json.dumps(result, ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
