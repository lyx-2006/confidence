from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from dp_SA.io_utils import append_jsonl, atomic_json, canonical_hash, load_jsonl, sha256_file

from .capture import acquire_pid, model_fingerprint
from .config import COUNTERFACTUAL_OUTPUT_ROOT, DATASET_PATH, MODEL_PATH, VARIANTS
from .contracts import ensure_fingerprinted_config
from .counterfactual_common import (
    CAPTURE_PHASE0_PATH, TEXT_ENTROPY_TOLERANCE, TEXT_POOL_PATH,
    TEXT_PROBABILITY_TOLERANCE, BehaviorRunner, load_text_pool,
    render_counterfactual, stable_key,
)
from .dataset import COLORS, ConflictCase, load_conflict_cases


def ensure_layout(root: Path) -> None:
    for path in (
        root / "progress", root / "tables", root / "images",
        root / "figures" / "prob", root / "figures" / "logit",
    ):
        path.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        for section in ("progress", "figures", "tables"):
            (root / variant / section).mkdir(parents=True, exist_ok=True)


def same_band(case: ConflictCase, entropy: float) -> bool:
    if case.pair_type == "hard_text_easy_image":
        return entropy >= 0.5
    if case.pair_type == "hard_image_easy_text":
        return entropy < 0.1
    return 0.3 <= entropy < 0.4


def prepare_config(model_path: Path, case_ids: Sequence[str] | None) -> dict[str, Any]:
    return {
        "format_version": 2, "experiment": "qwen3_counterfactual_prepare",
        "model": str(model_path.resolve()), "model_fingerprint": model_fingerprint(model_path),
        "dataset": str(DATASET_PATH.resolve()), "dataset_sha256": sha256_file(DATASET_PATH),
        "text_pool": str(TEXT_POOL_PATH.resolve()), "text_pool_sha256": sha256_file(TEXT_POOL_PATH),
        "phase0": str(CAPTURE_PHASE0_PATH.resolve()), "phase0_sha256": sha256_file(CAPTURE_PHASE0_PATH),
        "case_ids": list(case_ids) if case_ids else None,
        "third_color_excludes": ["text_answer", "image_answer", "fixed_answer"],
        "allow_distractor_color_collision": True,
        "cached_entropy_tolerance": TEXT_ENTROPY_TOLERANCE,
        "measured_entropy_tolerance": TEXT_ENTROPY_TOLERANCE,
        "measured_probability_tolerance": TEXT_PROBABILITY_TOLERANCE,
        "selection_policy": "strict_then_measured_entropy_only_fallback",
    }


def run_prepare(
    *, output_root: Path = COUNTERFACTUAL_OUTPUT_ROOT, model_path: Path = MODEL_PATH,
    case_ids: Sequence[str] | None = None, resume: bool = False,
) -> dict[str, Any]:
    root, model_path = output_root.resolve(), model_path.resolve()
    ensure_layout(root)
    pid = root / "progress" / "prepare.pid"
    acquire_pid(pid, "Counterfactual prepare")
    try:
        config = ensure_fingerprinted_config(
            root / "progress" / "prepare_config.json",
            prepare_config(model_path, case_ids), resume=resume, label="Counterfactual prepare",
        )
        cases = load_conflict_cases(DATASET_PATH)
        if case_ids:
            requested = set(case_ids)
            cases = [case for case in cases if case.case_id in requested]
            if {case.case_id for case in cases} != requested:
                raise ValueError("Unknown case ID requested")
        phase0 = {row["case_id"]: row for row in load_jsonl(CAPTURE_PHASE0_PATH)}
        manifest_path = root / "tables" / "construction_manifest.jsonl"
        existing = {row["case_id"]: row for row in load_jsonl(manifest_path)}
        cache_path = root / "tables" / "text_score_cache.jsonl"
        cache = {row["cache_key"]: row["score"] for row in load_jsonl(cache_path)}
        text_pool = load_text_pool()
        runner = BehaviorRunner(model_path)
        started = time.time()

        def text_score(case: ConflictCase, clue: str, expected: str) -> dict[str, Any]:
            key = canonical_hash({"question": case.question, "clue": clue, "expected": expected})
            if key not in cache:
                cache[key] = runner.text_only(case, clue, expected)
                append_jsonl(cache_path, {"cache_key": key, "score": cache[key]})
            return cache[key]

        for ordinal, case in enumerate(cases, 1):
            if case.case_id in existing:
                continue
            base = {
                "case_id": case.case_id, "source_index": case.source_index,
                "shape": case.shape, "pair_type": case.pair_type,
                "question": case.question, "original_text": case.text_clue,
                "original_image": str(case.image_path), "text_answer": case.text_answer,
                "image_answer": case.image_answer, "text_entropy_cached": case.text_entropy,
                "image_entropy_cached": case.image_entropy,
                "fixed_answer": phase0[case.case_id]["phase0_normalized_answer"],
                "phase0_answer_fingerprint": phase0[case.case_id]["phase0_answer_fingerprint"],
            }
            attempts: list[dict[str, Any]] = []
            candidates: list[dict[str, Any]] = []
            try:
                fixed = base["fixed_answer"]
                allowed = [color for color in COLORS if color not in {case.text_answer, case.image_answer, fixed}]
                candidates = [
                    {"color": color, "clue": row["clue"], "cached_entropy": float(row["Entropy"])}
                    for color in allowed for row in text_pool[color]
                    if same_band(case, float(row["Entropy"]))
                    and abs(float(row["Entropy"]) - case.text_entropy) <= TEXT_ENTROPY_TOLERANCE
                ]
                candidates.sort(key=lambda row: (
                    abs(row["cached_entropy"] - case.text_entropy),
                    stable_key(case.case_id, row["color"], row["clue"]),
                ))
                original_score = text_score(case, case.text_clue, case.text_answer)
                selected = None
                entropy_fallbacks: list[dict[str, Any]] = []
                for candidate in candidates:
                    score = text_score(case, candidate["clue"], candidate["color"])
                    deltas = {
                        "entropy_delta": abs(score["normalized_entropy"] - original_score["normalized_entropy"]),
                        "target_probability_delta": abs(score["target_probability"] - original_score["target_probability"]),
                        "target_margin_delta": abs(score["target_margin"] - original_score["target_margin"]),
                        "length_delta": abs(len(candidate["clue"]) - len(case.text_clue)),
                    }
                    valid = bool(
                        score["normalized_answer"] == candidate["color"]
                        and score["restricted_top1"] == candidate["color"]
                        and deltas["entropy_delta"] <= TEXT_ENTROPY_TOLERANCE
                        and deltas["target_probability_delta"] <= TEXT_PROBABILITY_TOLERANCE
                    )
                    attempts.append({
                        "color": candidate["color"], "clue_hash": canonical_hash(candidate["clue"]),
                        "cached_entropy": candidate["cached_entropy"], "deltas": deltas, "valid": valid,
                    })
                    if deltas["entropy_delta"] <= TEXT_ENTROPY_TOLERANCE:
                        entropy_fallbacks.append({**candidate, "score": score, "deltas": deltas})
                    if valid:
                        selected = {
                            **candidate, "score": score, "deltas": deltas,
                            "selection_rule": "strict",
                        }
                        break
                if selected is None:
                    if not entropy_fallbacks:
                        raise ValueError("no_measured_entropy_matched_counterfactual_text")
                    entropy_fallbacks.sort(key=lambda row: (
                        row["deltas"]["entropy_delta"],
                        row["deltas"]["target_probability_delta"],
                        row["deltas"]["target_margin_delta"],
                        row["deltas"]["length_delta"],
                        stable_key(case.case_id, row["color"], row["clue"]),
                    ))
                    selected = {**entropy_fallbacks[0], "selection_rule": "entropy_only_fallback"}
                render = render_counterfactual(
                    case, selected["color"], root / "images" / case.case_id
                )
                original_image_score = runner.image_only(case, str(case.image_path), case.image_answer)
                counterfactual_image_score = runner.image_only(
                    case, render["counterfactual_image"], selected["color"]
                )
                row = {
                    "status": "completed", **base, "third_color": selected["color"],
                    "counterfactual_text": selected["clue"],
                    "selection_rule": selected["selection_rule"],
                    "text_match_deltas": selected["deltas"],
                    "original_text_score": original_score,
                    "counterfactual_text_score": selected["score"],
                    "original_text_gate_passed": bool(
                        original_score["normalized_answer"] == case.text_answer
                        and original_score["restricted_top1"] == case.text_answer
                    ),
                    "counterfactual_text_gate_passed": bool(
                        selected["score"]["normalized_answer"] == selected["color"]
                        and selected["score"]["restricted_top1"] == selected["color"]
                    ),
                    "counterfactual_text_probability_tolerance_passed": bool(
                        selected["deltas"]["target_probability_delta"]
                        <= TEXT_PROBABILITY_TOLERANCE
                    ),
                    "render_audit": render,
                    "original_image_score": original_image_score,
                    "counterfactual_image_score": counterfactual_image_score,
                    "candidate_count": len(candidates), "attempt_count": len(attempts),
                    "attempts": attempts,
                }
            except Exception as exc:
                row = {
                    "status": "failed", **base, "candidate_count": len(candidates),
                    "attempts": attempts,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            append_jsonl(manifest_path, row)
            existing[case.case_id] = row
            atomic_json(root / "progress" / "prepare_progress.json", {
                "total": len(cases), "processed": len(existing),
                "completed": sum(r["status"] == "completed" for r in existing.values()),
                "failed": sum(r["status"] == "failed" for r in existing.values()),
                "elapsed_seconds": time.time() - started,
            })
        counts = Counter(row["status"] for row in existing.values())
        summary = {
            "status": "complete", "total": len(cases), "counts": dict(counts),
            "config_fingerprint": config["fingerprint"], "elapsed_seconds": time.time() - started,
        }
        atomic_json(root / "progress" / "prepare_summary.json", summary)
        return summary
    finally:
        if pid.exists() and pid.read_text().strip() == str(__import__("os").getpid()):
            pid.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare matched counterfactual cases")
    parser.add_argument("--output-root", type=Path, default=COUNTERFACTUAL_OUTPUT_ROOT)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    print(run_prepare(output_root=args.output_root, model_path=args.model_path, case_ids=args.case_ids, resume=args.resume))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
