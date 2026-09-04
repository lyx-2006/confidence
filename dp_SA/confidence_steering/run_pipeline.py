from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from .analyze import analyze
from .config import (
    EXPECTED_TEST_FAMILIES, EXPECTED_TEST_ROWS, EXPECTED_TEST_SHA256,
    FORMAL_ROOT, MAX_SMOKE_ROUNDS, NULL_INITIAL_REPEATS, NULL_MAX_REPEATS,
    PROTOCOL_VERSION, SEALED_TEST_MANIFEST, SMOKE_PARENT, TRAIN_MANIFEST,
)
from .io_utils import atomic_json, atomic_jsonl, canonical_hash, create_output_root, load_jsonl, sha256_file
from .prepare import run_prepare
from .run import run_steering
from .run_spec import add_run_spec_arguments, normalize_run_spec, run_spec_from_args

TEST_PATHS = (
    "dp_SA/confidence_steering/tests",
    "dp_SA/answer_matched_lat_steering/tests/test_position_hook.py",
    "dp_SA/answer_matched_lat_steering/tests/test_vectors.py",
    "dp_SA/unimodal_logit_confidence/tests/test_explanatory_comparison.py",
)


def run_cpu_tests() -> dict[str, Any]:
    started = time.time(); completed = subprocess.run([sys.executable, "-m", "pytest", "-q", *TEST_PATHS], cwd=Path(__file__).resolve().parents[2], check=False)
    if completed.returncode: raise RuntimeError(f"CPU tests failed with exit code {completed.returncode}")
    return {"status": "passed", "elapsed_seconds": time.time() - started, "paths": list(TEST_PATHS)}


def _next_smoke_round(run_spec: dict[str, Any]) -> tuple[int, Path]:
    parent = SMOKE_PARENT if run_spec["is_default"] else SMOKE_PARENT / f'config_{run_spec["fingerprint"][:12]}'
    parent.mkdir(parents=True, exist_ok=True)
    for number in range(1, MAX_SMOKE_ROUNDS + 1):
        path = parent / f"round_{number}"
        if not path.exists(): return number, path
    raise RuntimeError(f"All {MAX_SMOKE_ROUNDS} smoke rounds have been used")


def _latest_smoke_lock(run_spec: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    candidates = []
    if SMOKE_PARENT.is_dir():
        for path in SMOKE_PARENT.rglob("round_*/progress/experiment_lock.json"):
            value = json.loads(path.read_text())
            if value.get("status") == "locked" and int(value.get("protocol_version", -1)) == PROTOCOL_VERSION and value.get("run_spec") == run_spec: candidates.append((path, value))
    if not candidates: raise RuntimeError("Formal test remains sealed: no successful matching orthogonal smoke lock")
    return sorted(candidates, key=lambda item: item[0].stat().st_mtime)[-1]


def _validate_zero_overlap(train: Sequence[dict[str, Any]], test: Sequence[dict[str, Any]]) -> dict[str, int]:
    definitions = {
        "case": lambda r: str(r["case_id"]), "family": lambda r: str(r["family_id"]), "item": lambda r: str(r["item_id"]), "image_hash": lambda r: str(r["image_sha256"]),
        "text_unique_key": lambda r: (str(r["item_id"]), int(r["prior_index"])), "image_unique_key": lambda r: (str(r["item_id"]), str(r["condition"]), str(r["image_sha256"])),
    }
    overlaps = {name: len({fn(r) for r in train} & {fn(r) for r in test}) for name, fn in definitions.items()}
    if any(overlaps.values()): raise ValueError(f"Formal test leakage: {overlaps}")
    return overlaps


def unseal_formal_test(root: Path, lock: dict[str, Any]) -> dict[str, Any]:
    # This is the sole function in the package that opens the formal test manifest.
    if lock.get("status") != "locked": raise RuntimeError("Formal test cannot be opened before experiment lock")
    if sha256_file(SEALED_TEST_MANIFEST) != EXPECTED_TEST_SHA256: raise ValueError("Sealed formal test hash mismatch")
    test = load_jsonl(SEALED_TEST_MANIFEST); train = load_jsonl(TRAIN_MANIFEST)
    if len(test) != EXPECTED_TEST_ROWS or len({r["family_id"] for r in test}) != EXPECTED_TEST_FAMILIES: raise ValueError("Formal test 100/50 cardinality changed")
    overlaps = _validate_zero_overlap(train, test); atomic_jsonl(root / "artifacts/manifests/runtime_manifest.jsonl", test)
    audit = {"status": "unsealed_for_locked_formal_runtime", "test_sha256": EXPECTED_TEST_SHA256, "records": len(test), "families": EXPECTED_TEST_FAMILIES, "overlaps": overlaps, "experiment_lock_fingerprint": lock["fingerprint"]}; atomic_json(root / "artifacts/diagnostics/formal_test_unseal_audit.json", audit); return audit


def _make_lock(root: Path, *, smoke_lock: dict[str, Any] | None, smoke_lock_path: Path | None) -> dict[str, Any]:
    material = json.loads((root / "artifacts/diagnostics/prelock_material.json").read_text()); verdict = material["verdict"]
    if not verdict["formal_eligible"]: status = "audit_failed"
    elif smoke_lock is not None and (smoke_lock.get("code_hashes") != material["code_hashes"] or smoke_lock.get("vector_fingerprint") != material["vector_fingerprint"] or smoke_lock.get("probe_files") != material["probe_files"]): status = "smoke_lock_mismatch"
    else: status = "locked"
    run_spec = material["run_spec"]; null_initial = NULL_INITIAL_REPEATS if run_spec["shuffle_requested"] else 0
    lock = {"status": status, "protocol_version": PROTOCOL_VERSION, "test_state": "sealed", "prelock_fingerprint": material["fingerprint"], "code_hashes": material["code_hashes"], "vector_fingerprint": material["vector_fingerprint"], "probe_files": material["probe_files"], "layers": material["layers"], "alphas": material["alphas"], "directions": material["directions"], "run_spec": run_spec, "seed": material["seed"], "output_schema": material["output_schema"], "thresholds_frozen": True, "null_vectors_prebuilt": 99 if run_spec["shuffle_requested"] else 0, "null_initial": null_initial, "null_expand_to": 99 if run_spec["analysis_kind"] == "legacy_confirmatory" else null_initial, "null_expand_rule": material["null_rule"], "source_smoke_lock": str(smoke_lock_path) if smoke_lock_path else None}
    lock["fingerprint"] = canonical_hash(lock); return lock


def run_pipeline(*, output_root: Path | None = None, smoke: bool = False, resume: bool = False, num_gpus: int = 1, run_spec: dict[str, Any] | None = None) -> dict[str, Any]:
    run_spec = normalize_run_spec() if run_spec is None else run_spec
    tests = run_cpu_tests()
    if smoke:
        failures = []
        for _ in range(MAX_SMOKE_ROUNDS):
            if output_root is None: number, root = _next_smoke_round(run_spec)
            else: number, root = 1, Path(output_root)
            try:
                create_output_root(root, resume=resume); prepared = run_prepare(output_root=root, smoke=True, resume=resume, run_spec=run_spec)
                if not prepared["formal_eligible"]: raise RuntimeError("Pre-lock audit gates failed; GPU smoke not started")
                smoke_manifest = load_jsonl(root / "artifacts/manifests/smoke_manifest.jsonl"); atomic_jsonl(root / "artifacts/manifests/runtime_manifest.jsonl", smoke_manifest)
                ran = run_steering(output_root=root, smoke=True, resume=resume, num_gpus=num_gpus, desired_null=0, run_spec=run_spec); analyzed = analyze(output_root=root, smoke=True, resume=resume, run_spec=run_spec)
                resumed = run_steering(output_root=root, smoke=True, resume=True, num_gpus=num_gpus, desired_null=0, run_spec=run_spec)
                if not resumed["resumed_noop"] or resumed["new_gpu_forwards"] != 0: raise RuntimeError("Smoke resume performed new GPU forwards")
                lock = _make_lock(root, smoke_lock=None, smoke_lock_path=None); lock["status"] = "locked" if ran["status"] == "complete" and analyzed["status"] == "complete" else "failed"; lock["smoke_round"] = number; lock["fingerprint"] = canonical_hash({k: v for k, v in lock.items() if k != "fingerprint"}); atomic_json(root / "progress/experiment_lock.json", lock)
                report = {"status": "passed", "round": number, "output_root": str(root.resolve()), "formal_test_opened": False, "cpu_tests": tests, "prepare": prepared, "run": ran, "analyze": analyzed, "resume": resumed, "failures": failures, "experiment_lock": lock}; atomic_json(root / "progress/smoke_report.json", report); return report
            except Exception as exc:
                failures.append({"round": number, "type": type(exc).__name__, "message": str(exc), "output_root": str(root.resolve())})
                atomic_json(root / "progress/smoke_failure.json", failures[-1])
                if output_root is not None: break
        raise RuntimeError(f"Smoke failed after {len(failures)} round(s): {failures}")

    root = Path(output_root or FORMAL_ROOT); create_output_root(root, resume=resume); prepared = run_prepare(output_root=root, smoke=False, resume=resume, run_spec=run_spec)
    smoke_path, smoke_lock = _latest_smoke_lock(run_spec); lock = _make_lock(root, smoke_lock=smoke_lock, smoke_lock_path=smoke_path); atomic_json(root / "progress/experiment_lock.json", lock)
    if lock["status"] != "locked": raise RuntimeError(f"Formal test remains sealed: {lock['status']}")
    smoke_report = smoke_path.parent / "smoke_report.json"
    atomic_json(root / "artifacts/diagnostics/smoke_reference.json", {"report": json.loads(smoke_report.read_text()), "sha256": sha256_file(smoke_report), "lock_sha256": sha256_file(smoke_path)})
    initial_null = NULL_INITIAL_REPEATS if run_spec["shuffle_requested"] else 0
    unseal = unseal_formal_test(root, lock); ran20 = run_steering(output_root=root, smoke=False, resume=resume, num_gpus=num_gpus, desired_null=initial_null, run_spec=run_spec); analyzed20 = analyze(output_root=root, smoke=False, resume=resume, run_spec=run_spec)
    ran99 = analyzed99 = None
    if run_spec["analysis_kind"] == "legacy_confirmatory" and analyzed20["expand_null_to_99"]:
        ran99 = run_steering(output_root=root, smoke=False, resume=True, num_gpus=num_gpus, desired_null=NULL_MAX_REPEATS, run_spec=run_spec)
        # Null file changed, so analysis config is intentionally replaced only for the locked extension.
        (root / "progress/analyze_config.json").unlink(); analyzed99 = analyze(output_root=root, smoke=False, resume=False, run_spec=run_spec)
    return {"status": "complete", "formal_test_opened": True, "cpu_tests": tests, "prepare": prepared, "lock": lock, "unseal": unseal, "run20": ran20, "analyze20": analyzed20, "run99": ran99, "analyze99": analyzed99}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root"); parser.add_argument("--resume", action="store_true"); parser.add_argument("--smoke", action="store_true"); parser.add_argument("--num-gpus", type=int, choices=(1, 2), default=1)
    parser.add_argument("--random-sa-null-repeats", type=int)
    parser.add_argument("--random-sa-null-layer", type=int, default=14)
    parser.add_argument("--random-sa-null-dose", type=float, default=2.0)
    add_run_spec_arguments(parser); args = parser.parse_args(argv)
    if args.random_sa_null_repeats is not None:
        if args.random_sa_null_layer != 14 or args.random_sa_null_dose != 2.0:
            raise ValueError("The locked random-SA null protocol requires --random-sa-null-layer 14 and --random-sa-null-dose 2")
        if any(value is not None for value in (args.directions, args.layers, args.alphas)):
            raise ValueError("Random-null mode is independent of the main --directions/--layers/--alphas options")
        from .random_sa_null import run_random_null_pipeline
        tests = run_cpu_tests()
        result = run_random_null_pipeline(
            repeats=args.random_sa_null_repeats, smoke=args.smoke, resume=args.resume,
            num_gpus=args.num_gpus,
            output_root=Path(args.output_root) if args.output_root else None,
        )
        result["cpu_tests"] = tests
        print(json.dumps(result, ensure_ascii=False)); return 0
    print(json.dumps(run_pipeline(output_root=Path(args.output_root) if args.output_root else None, smoke=args.smoke, resume=args.resume, num_gpus=args.num_gpus, run_spec=run_spec_from_args(args)), ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
