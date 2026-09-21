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
from qwen3_chat_binary.steering import build_vectors, save_torch_atomic

from .config import (ALPHAS, CHAINS, CONSTRUCTION_PER_SIDE, PAIRS, SEED,
                     TEST_PER_SIDE, VARIANT, VECTOR_NORM_FRACTION, default_output)


def select_manifests(rows: Sequence[dict[str, Any]], *, construction_per_side: int = CONSTRUCTION_PER_SIDE,
                     test_per_side: int = TEST_PER_SIDE) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible = [r for r in rows if r.get("status") == "completed" and float(r["image_attribution_score"]) != .5]
    used: set[str] = set(); construction: list[dict[str, Any]] = []
    for side, predicate, reverse, label in (
        ("text", lambda x: x < .5, False, "high_text"),
        ("image", lambda x: x > .5, True, "high_image"),
    ):
        candidates = [r for r in eligible if predicate(float(r["image_attribution_score"]))]
        candidates.sort(key=lambda r: ((-1 if reverse else 1) * float(r["image_attribution_score"]), str(r["case_id"])))
        chosen = [r for r in candidates if str(r["case_id"]) not in used][:construction_per_side]
        if len(chosen) != construction_per_side: raise ValueError(f"Insufficient {side} construction cases")
        for rank, row in enumerate(chosen, 1):
            used.add(str(row["case_id"])); construction.append({**row, "construction_side": label, "selection_rank": rank})
    test: list[dict[str, Any]] = []
    for side, predicate in (("text_side", lambda x: x < .5), ("image_side", lambda x: x > .5)):
        candidates = [r for r in eligible if predicate(float(r["image_attribution_score"])) and str(r["case_id"]) not in used]
        candidates.sort(key=lambda r: (abs(float(r["image_attribution_score"]) - .5), str(r["case_id"])))
        chosen = candidates[:test_per_side]
        if len(chosen) != test_per_side: raise ValueError(f"Insufficient {side} central cases")
        test.extend({**r, "test_side": side, "selection_rank": rank,
                     "distance_from_midpoint": abs(float(r["image_attribution_score"]) - .5)}
                    for rank, r in enumerate(chosen, 1))
    if {r["case_id"] for r in construction} & {r["case_id"] for r in test}: raise AssertionError("Construction/test leakage")
    return sorted(construction, key=lambda r: str(r["case_id"])), sorted(test, key=lambda r: str(r["case_id"]))


def validate_processor(rows: Sequence[dict[str, Any]], model_path: Path) -> dict[str, Any]:
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True); tokenizer = processor.tokenizer
    label_token_ids(tokenizer)
    for row in rows:
        messages = stage2_messages(row["phase0_prompt"], row["image_path"], row["phase0_raw_output"], VARIANT)
        rendered = render_stage2(processor, messages); inputs = prepare_multimodal_inputs(processor, messages, rendered)
        located = locate_positions(tokenizer, rendered, inputs, row["phase0_raw_output"], VARIANT)
        for name in ("LAT", "PANL", "PANL+1", "CLE", "SAC"):
            old, new = row["positions"][name], located["positions"][name]
            if (int(old["processed_index"]), int(old["token_id"])) != (int(new["processed_index"]), int(new["token_id"])):
                raise RuntimeError(f"Position drift for {row['case_id']} at {name}")
    return {"checked_cases": len(rows), "labels_are_single_tokens": True}


def prepare(*, output_root: Path, capture_root: Path = CAPTURE_ROOT, model_path: Path = MODEL_PATH,
            smoke: bool = False, resume: bool = False, validate_tokens: bool = True) -> dict[str, Any]:
    root, capture_root, model_path = (Path(x).resolve() for x in (output_root, capture_root, model_path))
    capture_config = json.loads(capture_config_path(capture_root).read_text(encoding="utf-8"))
    all_rows = load_jsonl(capture_results_path(capture_root, VARIANT)); construction, formal_test = select_manifests(all_rows)
    test = ([next(r for r in formal_test if r["test_side"] == side) for side in ("text_side", "image_side")]
            if smoke else formal_test)
    rows_by_id = {r["case_id"]: r for r in all_rows if r.get("status") == "completed"}
    upstream_positions = tuple(dict.fromkeys(upstream for upstream, _ in CHAINS.values()))
    steering_layers = tuple(upstream for upstream, _ in PAIRS)
    vectors, vector_metadata, artifacts = build_vectors(
        capture_root, rows_by_id, construction, variant=VARIANT,
        positions=upstream_positions, layers=steering_layers,
    )
    expected = len(upstream_positions) * len(steering_layers)
    if len(vectors) != expected: raise RuntimeError(f"Vector grid incomplete: {len(vectors)}/{expected}")
    processor_audit = validate_processor(test, model_path) if validate_tokens else {"skipped": True}
    payload = {
        "format_version": 1, "experiment": "qwen3_chat_fiveway_steering_mediation",
        "variant": VARIANT, "model": str(model_path), "capture_root": str(capture_root),
        "capture_fingerprint": capture_config["fingerprint"],
        "chains": {name: list(value) for name, value in CHAINS.items()},
        "pairs": [list(x) for x in PAIRS], "alphas": list(ALPHAS), "smoke": smoke,
        "normalization_fraction": VECTOR_NORM_FRACTION, "seed": SEED,
        "construction_fingerprint": canonical_hash(construction), "test_fingerprint": canonical_hash(test),
        "implementation_hashes": {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
    }
    config = ensure_fingerprinted_config(root / "run_config.json", payload, resume=resume,
                                         label="Fiveway steering mediation")
    atomic_jsonl(root / "artifacts/manifests/construction.jsonl", construction)
    atomic_jsonl(root / "artifacts/manifests/test.jsonl", test)
    save_torch_atomic(root / "artifacts/vectors.pt", artifacts)
    atomic_json(root / "artifacts/vector_metadata.json", vector_metadata)
    summary = {"construction_counts": dict(Counter(r["construction_side"] for r in construction)),
               "test_counts": dict(Counter(r["test_side"] for r in test)), "construction_test_overlap": 0,
               "processor_audit": processor_audit, "vector_count": len(vectors)}
    atomic_json(root / "artifacts/manifests/selection_summary.json", summary)
    return {"status": "complete", "fingerprint": config["fingerprint"], **summary}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare five-class steering mediation")
    parser.add_argument("--output-root", type=Path); parser.add_argument("--capture-root", type=Path, default=CAPTURE_ROOT)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--skip-processor-validation", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(prepare(output_root=args.output_root or default_output(args.smoke), capture_root=args.capture_root,
                             model_path=args.model_path, smoke=args.smoke, resume=args.resume,
                             validate_tokens=not args.skip_processor_validation), ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
