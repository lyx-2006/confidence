from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from layer_metacognition.model_adapter import resolve_language_modules

from .analyze import analyze
from .artifacts import audit_coverage, build_or_load_means
from .config import (
    BOOTSTRAP_REPEATS, CONDITIONS, MAX_PIXELS, MIN_PIXELS, MODEL_PATH, OUTPUT_ROOT, SEED,
)
from .data import FrozenCohort, load_frozen_cohort, write_manifests
from .io_utils import atomic_json, canonical_hash, ensure_layout, sha256_file, validate_fingerprint
from .runtime import load_strict_inference
from .scoring import run_scores, tokenization_preflight


def smoke_records(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for condition in ("conflict_easy", "conflict_hard"):
        members = [row for row in rows if row["condition"] == condition]
        if not members:
            raise ValueError(f"No test records for {condition}")
        selected.append(min(members, key=lambda row: (float(row["soft_sa_image_score"]), str(row["case_id"]))))
        selected.append(max(members, key=lambda row: (float(row["soft_sa_image_score"]), str(row["case_id"]))))
    if len({row["case_id"] for row in selected}) != 4:
        raise ValueError("Smoke selection did not produce four distinct cases")
    if {row["answer_side"] for row in selected} != {"follow_image", "follow_text"}:
        raise ValueError("Smoke extremes do not cover both answer sides")
    return sorted(selected, key=lambda row: str(row["case_id"]))


def _code_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    return {path.name: sha256_file(path) for path in sorted(package.glob("*.py"))}


def _run_payload(cohort: FrozenCohort, runtime_identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "format_version": 1, "experiment": "perturbation_based_answer_reliance",
        "seed": SEED, "bootstrap_repeats": BOOTSTRAP_REPEATS, "temperature": 1.0,
        "conditions": list(CONDITIONS), "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS,
        "model_path": str(MODEL_PATH.resolve()),
        "runtime_identity_fingerprint": runtime_identity["fingerprint"],
        "cohort_audit_fingerprint": canonical_hash(cohort.audit),
        "source_sha256": cohort.audit["source_sha256"], "implementation_sha256": _code_hashes(),
        "embedding_site": "language_model_inputs_embeds_after_vision_replacement",
        "image_mean_site": "vision_projection_before_language_model",
        "text_mean_site": "input_embedding_of_text_clue_content_tokens",
        "text_donor_context_policy": "sha256_rank_alternating_exact_100_easy_100_hard",
    }


def run(mode: str, output_root: Path, *, resume: bool) -> dict[str, Any]:
    if mode not in {"smoke", "formal"}:
        raise ValueError("mode must be smoke or formal")
    root = ensure_layout(output_root)
    cohort = load_frozen_cohort()
    write_manifests(root, cohort)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        failure = {
            "status": "blocked", "mode": mode, "reason": "GPU_REQUIRED",
            "cuda_available": torch.cuda.is_available(), "cuda_device_count": torch.cuda.device_count(),
        }
        atomic_json(root / "progress" / f"{mode}_environment_gate.json", failure)
        raise RuntimeError("GPU execution requires at least one visible CUDA device")
    inference, runtime_identity = load_strict_inference()
    atomic_json(root / "progress" / "processor_audit.json", runtime_identity)
    payload = _run_payload(cohort, runtime_identity)
    fingerprint = validate_fingerprint(root / "progress" / "run_config.json", payload)
    if mode == "formal":
        smoke_path = root / "progress" / "smoke" / "smoke_report.json"
        if not smoke_path.is_file():
            raise RuntimeError("Formal run requires a completed smoke report")
        smoke = json.loads(smoke_path.read_text())
        if smoke.get("status") != "passed" or smoke.get("run_fingerprint") != fingerprint:
            raise RuntimeError("Formal run requires a passing smoke with the same run fingerprint")
    preflight = tokenization_preflight(inference.processor, cohort.tests)
    atomic_json(root / "progress" / "forward_budget.json", preflight)
    modules = resolve_language_modules(inference.model)
    mean_path = root / "artifacts" / "mean_embeddings" / "manifest.json"
    cold_means = not mean_path.exists()
    artifacts = build_or_load_means(root, cohort, inference, runtime_identity, modules.hidden_size)
    rows = smoke_records(cohort.tests) if mode == "smoke" else cohort.tests
    coverage = audit_coverage(rows, inference, artifacts, modules.hidden_size)
    coverage_path = root / "progress" / ("smoke/coverage_audit.json" if mode == "smoke" else "coverage_audit.json")
    atomic_json(coverage_path, coverage)
    if coverage.get("status") != "passed":
        raise RuntimeError(
            f"{mode} coverage gate failed; see {coverage_path} (no corruption scores were produced)"
        )
    score_dir = root / "progress" / "smoke" if mode == "smoke" else root / "artifacts"
    started = time.time()
    first = run_scores(inference, rows, artifacts, preflight, score_dir, fingerprint, smoke=mode == "smoke")
    result: dict[str, Any] = {
        "status": "passed" if mode == "smoke" else "scores_complete", "mode": mode,
        "run_fingerprint": fingerprint, "case_count": len(rows), "cold_mean_build": cold_means,
        "image_feature_donor_calls": 200 if cold_means else 0, "coverage": coverage,
        "scoring": first, "elapsed_seconds": time.time() - started,
    }
    if mode == "smoke":
        second = run_scores(inference, rows, artifacts, preflight, score_dir, fingerprint, smoke=True)
        if not second["resumed_noop"] or second["new_internal_model_forwards"] != 0:
            raise RuntimeError("Smoke resume repeated a model forward")
        result["resume"] = second
        atomic_json(root / "progress" / "smoke" / "smoke_report.json", result)
    else:
        result["analysis"] = analyze(root, cohort)
        result["status"] = "complete"
        atomic_json(root / "progress" / "formal_report.json", result)
    del artifacts, inference
    gc.collect()
    torch.cuda.empty_cache()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Perturbation-based answer reliance")
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true", help="Resume only when all fingerprints match")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args.mode, args.output_root, resume=args.resume), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
