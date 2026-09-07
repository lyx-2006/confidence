from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import sklearn
import transformers

from dp_SA.confidence_steering.processor import PROCESSOR_MODE, load_fast_processor, processor_identity

from .analyze import analyze
from .config import (
    BOOTSTRAP_REPEATS, DIRECTIONS, FAST_REFERENCE_ROOT, FORMAL_MANIFEST,
    FROZEN_LAT_PROBE, FROZEN_PANL_SA_PROBE, FROZEN_VECTOR_METADATA,
    FROZEN_VECTORS, MAX_SMOKE_ROUNDS, MODEL_PATH, OUTPUT_ROOT,
    PACKAGE_ROOT, PROTOCOL_VERSION, RESULTS_ROOT, SEEDS, SMOKE_BOOTSTRAP_REPEATS,
    SMOKE_SEED, TRAIN_MANIFEST, CONFIDENCE_JOINED, UNIMODAL_SCORES,
    HIDDEN_REUSE, HIDDEN_CAPTURE,
)
from .io_utils import (
    atomic_json, atomic_jsonl, canonical_hash, ensure_layout, inventory_hashes,
    load_jsonl, sha256_file, verify_inventory,
)
from .rebuild import build_seed, enrich_train_rows, load_hidden
from .run import run_steering
from .split import (
    build_all_assignments, load_formal, load_frozen_train, select_smoke_families,
    sklearn_audit, write_seed_split,
)


def code_paths() -> list[Path]:
    repo = PACKAGE_ROOT.parents[2]
    return sorted(PACKAGE_ROOT.glob("*.py")) + [
        PACKAGE_ROOT.parent / "core.py", PACKAGE_ROOT.parent / "processor.py",
        PACKAGE_ROOT.parent / "run.py", PACKAGE_ROOT.parent / "analyze.py",
        PACKAGE_ROOT.parent / "gradient_validation.py",
        repo / "dp_SA/positions.py", repo / "dp_SA/soft_score.py",
        repo / "layer_metacognition/model_adapter.py",
        repo / "layer_metacognition/conversation_builder.py",
        Path(str(__import__("dp_SA.config", fromlist=["INFERENCE_PATH"]).INFERENCE_PATH)),
    ]


def reused_paths(train_rows: Sequence[dict[str, Any]], *, formal: bool) -> list[Path]:
    paths = code_paths() + [
        TRAIN_MANIFEST, CONFIDENCE_JOINED, UNIMODAL_SCORES, HIDDEN_REUSE,
        HIDDEN_CAPTURE, FROZEN_LAT_PROBE, FROZEN_PANL_SA_PROBE,
        FROZEN_VECTORS, FROZEN_VECTOR_METADATA,
        MODEL_PATH / "preprocessor_config.json", MODEL_PATH / "config.json",
        MODEL_PATH / "tokenizer.json", MODEL_PATH / "tokenizer_config.json",
        MODEL_PATH / "model.safetensors.index.json", MODEL_PATH / "chat_template.json",
        *[MODEL_PATH / f"model-{index:05d}-of-00005.safetensors" for index in range(1, 6)],
        *sorted((FROZEN_VECTORS.parents[2] / "tables/random_sa_subspace_null").glob("*.csv")),
    ]
    if formal:
        paths.append(FORMAL_MANIFEST)
    allowed = {str(row["case_id"]) for row in train_rows}
    for row in load_jsonl(HIDDEN_REUSE):
        if str(row["case_id"]) not in allowed:
            continue
        for key in ("P1_LAT__L14", "P1_PANL__L18"):
            source = row.get("cell_sources", {}).get(key)
            if source:
                paths.append(Path(source["path"]))
    return paths


def protocol_material(before: dict[str, dict[str, Any]], *, formal: bool, seeds: Sequence[int]) -> dict[str, Any]:
    processor = load_fast_processor()
    identity = processor_identity(processor)
    if not identity["is_fast"] or not identity["image_processor_class"].endswith("Qwen2VLImageProcessorFast"):
        raise RuntimeError(f"Explicit Fast processor preflight failed: {identity}")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "analysis": "fixed-evaluation-set split-seed stability analysis",
        "formal": formal, "seeds": list(seeds), "directions": list(DIRECTIONS),
        "position": "P1_LAT", "layer": 14, "alphas": [-0.5, 0.0, 0.5],
        "processor_mode": PROCESSOR_MODE,
        "processor_requirement": "transformers.Qwen2VLImageProcessorFast",
        "processor_identity": identity,
        "image_processor_config": json.loads(json.dumps(processor.image_processor.to_dict(), sort_keys=True, default=str)),
        "transformers_version": transformers.__version__,
        "scikit_learn": sklearn_audit(),
        "bootstrap_repeats": BOOTSTRAP_REPEATS if formal else SMOKE_BOOTSTRAP_REPEATS,
        "source_inventory_sha256": canonical_hash(before),
    }


def _test_command() -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", str(PACKAGE_ROOT / "tests")]
    completed = subprocess.run(command, cwd=PACKAGE_ROOT.parents[2], text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(f"CPU tests failed:\n{completed.stdout}\n{completed.stderr}")
    return {"status": "passed", "command": command, "output": completed.stdout.strip()}


def _choose_smoke_root(fingerprint: str, resume: bool) -> Path:
    parent = OUTPUT_ROOT / "smoke" / f"config_{fingerprint}"
    existing = sorted(path for path in parent.glob("round_*") if path.is_dir()) if parent.exists() else []
    if resume and existing:
        return existing[-1]
    for round_number in range(1, MAX_SMOKE_ROUNDS + 1):
        candidate = parent / f"round_{round_number}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Smoke exceeded {MAX_SMOKE_ROUNDS} rounds")


def _validate_config(root: Path, material: dict[str, Any], resume: bool) -> str:
    fingerprint = canonical_hash(material)
    path = root / "progress/config.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("fingerprint") != fingerprint:
            raise RuntimeError("Resume config fingerprint mismatch")
        if not resume:
            raise FileExistsError(f"Output exists; pass --resume: {root}")
    else:
        atomic_json(path, {**material, "fingerprint": fingerprint})
    return fingerprint


def prepare(root: Path, *, formal: bool, seeds: Sequence[int], before: dict[str, dict[str, Any]]) -> dict[str, Any]:
    raw_train = load_frozen_train()
    enriched = enrich_train_rows(raw_train)
    assignments = build_all_assignments(enriched)
    formal_rows = load_formal() if formal else None
    split_results = {}
    for seed in SEEDS:
        split_results[seed] = write_seed_split(root, enriched, seed, assignments[seed], formal_rows)
    hidden, hidden_audit = load_hidden(enriched, root)
    build_results = []
    repeats = BOOTSTRAP_REPEATS if formal else SMOKE_BOOTSTRAP_REPEATS
    for seed in seeds:
        build_results.append(build_seed(root, seed, split_results[seed]["construction"], split_results[seed]["audit"], hidden, bootstrap_repeats=repeats))
    if formal:
        runtime = sorted(formal_rows or [], key=lambda row: str(row["case_id"]))
        selection = {"source": "frozen_formal_manifest", "records": len(runtime), "families": len({row["family_id"] for row in runtime}), "formal_manifest_opened": True}
    else:
        runtime, selection = select_smoke_families(split_results[SMOKE_SEED]["audit"])
    atomic_jsonl(root / "artifacts/trials/runtime_manifest.jsonl", runtime)
    atomic_json(root / "artifacts/diagnostics/runtime_selection.json", selection)
    result = {
        "status": "complete", "seeds_built": list(seeds), "runtime_records": len(runtime),
        "runtime_families": len({row["family_id"] for row in runtime}),
        "formal_manifest_opened": formal, "hidden_records_validated": len(hidden_audit),
        "split_hashes": {str(seed): split_results[seed]["audit_record"]["assignment_sha256"] for seed in SEEDS},
        "build": build_results,
    }
    atomic_json(root / "progress/prepare.json", result)
    return result


def run_pipeline(*, formal: bool = False, num_gpus: int = 1, resume: bool = False, run_tests: bool = True) -> dict[str, Any]:
    tests = _test_command() if run_tests else {"status": "skipped"}
    raw_train = load_frozen_train()
    before = inventory_hashes(reused_paths(raw_train, formal=formal))
    material = protocol_material(before, formal=formal, seeds=SEEDS if formal else (SMOKE_SEED,))
    fingerprint = canonical_hash(material)
    root = RESULTS_ROOT if formal else _choose_smoke_root(fingerprint, resume)
    ensure_layout(root)
    _validate_config(root, material, resume)
    atomic_json(root / "artifacts/source_hashes/before.json", before)
    seeds = SEEDS if formal else (SMOKE_SEED,)
    prepared = prepare(root, formal=formal, seeds=seeds, before=before)
    ran = run_steering(root, seeds, num_gpus, fingerprint)
    expected = len(seeds) * prepared["runtime_records"] * 7
    if ran["expected_physical_forwards"] != expected or ran["trial_rows"] != expected:
        raise RuntimeError(f"Forward/trial budget mismatch: {ran} expected={expected}")
    analyzed = analyze(root, seeds, bootstrap_repeats=BOOTSTRAP_REPEATS if formal else SMOKE_BOOTSTRAP_REPEATS)
    resumed = run_steering(root, seeds, num_gpus, fingerprint)
    if not resumed["resumed_noop"] or resumed["new_gpu_forwards"] != 0:
        raise RuntimeError("Complete resume performed new GPU forwards")
    verify_inventory(before)
    after = inventory_hashes(Path(path) for path in before)
    atomic_json(root / "artifacts/source_hashes/after.json", after)
    parity_rows = []
    for path in sorted((root / "artifacts/diagnostics").glob("parity.shard_*.jsonl")):
        parity_rows.extend(load_jsonl(path))
    if len(parity_rows) != len(seeds) * prepared["runtime_records"] or not all(row["passed"] for row in parity_rows):
        raise RuntimeError("Processor/clean parity coverage failed")
    report = {
        "status": "complete", "formal": formal, "output_root": str(root),
        "fingerprint": fingerprint, "tests": tests, "prepare": prepared,
        "run": ran, "resume": resumed, "analysis": analyzed,
        "processor_parity_cases": len(parity_rows),
        "historical_hashes_unchanged": before == after,
        "formal_started": formal,
    }
    if formal:
        if ran["trial_rows"] != 2800 or prepared["runtime_records"] != 100:
            raise RuntimeError("Formal completion cardinality failed")
        required = [
            *(root / "tables" / name for name in (
                "split_audit.csv", "probe_metrics_by_seed.csv", "vector_stability.csv",
                "subspace_principal_angles.csv", "steering_effects_by_seed.csv",
                "casewise_effect_reproducibility.csv", "component_additivity.csv",
            )),
            *(root / "figures" / name for name in (
                "vector_stability.png", "final_sa_stability.png", "panl_sa_stability.png",
            )),
            root / "README_RESULTS_zh.md",
        ]
        for seed in SEEDS:
            required.extend((
                root / f"artifacts/directions/seed_{seed}/P1_LAT__L14.npz",
                root / f"artifacts/directions/seed_{seed}/vector_metadata.json",
                root / f"artifacts/probes/seed_{seed}/confidence_gap__P1_LAT__L14__full.joblib",
                root / f"artifacts/probes/seed_{seed}/final_sa__P1_PANL__L18__full.joblib",
            ))
        missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
        if missing:
            raise RuntimeError(f"Formal completion artifacts missing/empty: {missing}")
        atomic_json(root / "completion.json", report)
    else:
        atomic_json(root / "progress/smoke_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Confidence steering split-seed stability analysis")
    parser.add_argument("--formal", action="store_true", help="Run the frozen 100-case evaluation; never implied")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--num-gpus", type=int, choices=(1, 2), default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-tests", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.cpu_only:
        result = _test_command()
    else:
        result = run_pipeline(formal=args.formal, num_gpus=args.num_gpus, resume=args.resume, run_tests=not args.skip_tests)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
