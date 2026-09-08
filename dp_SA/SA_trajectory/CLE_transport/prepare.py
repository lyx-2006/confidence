from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

from dp_SA.io_utils import atomic_json, atomic_jsonl, canonical_hash, load_jsonl, sha256_file

from .config import (
    BOOTSTRAP_REPEATS, EXPECTED_FORMAL_CASES, EXPECTED_HISTORICAL_CLEAN_SHA256,
    EXPECTED_MANIFEST_SHA256, EXPECTED_PREPROCESSOR_SHA256, EXPERIMENTS,
    HISTORICAL_CLEAN_PATH, MANIFEST_PATH, MODEL_PATH, PREPROCESSOR_CONFIG_PATH,
    SEED, SMOKE_CASES, WINDOWS, default_output,
)


def select_smoke_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: str(row["case_id"]))
    selected = [next(row for row in ordered if row["test_side"] == side)
                for side in ("high_image", "high_text")]
    if len(selected) != SMOKE_CASES or len({row["case_id"] for row in selected}) != SMOKE_CASES:
        raise ValueError("Smoke selection must contain one distinct case from each side")
    return selected


def validate_frozen_inputs() -> tuple[list[dict[str, Any]], dict[str, str]]:
    hashes = {
        "manifest": sha256_file(MANIFEST_PATH),
        "historical_clean": sha256_file(HISTORICAL_CLEAN_PATH),
        "preprocessor_config": sha256_file(PREPROCESSOR_CONFIG_PATH),
    }
    expected = {
        "manifest": EXPECTED_MANIFEST_SHA256,
        "historical_clean": EXPECTED_HISTORICAL_CLEAN_SHA256,
        "preprocessor_config": EXPECTED_PREPROCESSOR_SHA256,
    }
    if hashes != expected:
        raise ValueError(f"Frozen input hash mismatch: expected={expected}, actual={hashes}")
    rows = load_jsonl(MANIFEST_PATH)
    unique = {key: len({str(row[key]) for row in rows}) for key in ("case_id", "family_id", "item_id")}
    if len(rows) != EXPECTED_FORMAL_CASES or any(value != EXPECTED_FORMAL_CASES for value in unique.values()):
        raise ValueError(f"Formal manifest is not 174 independent cases/families/items: n={len(rows)}, unique={unique}")
    history = {str(row["case_id"]): row for row in load_jsonl(HISTORICAL_CLEAN_PATH)}
    missing = sorted(str(row["case_id"]) for row in rows if str(row["case_id"]) not in history)
    if missing:
        raise ValueError(f"Historical clean capture misses formal cases: {missing[:5]}")
    return rows, hashes


def _implementation_hashes() -> dict[str, str]:
    names = ("config.py", "prepare.py", "run.py", "analyze.py", "run_pipeline.py", "README_zh.md")
    root = Path(__file__).resolve().parent
    return {name: sha256_file(root / name) for name in names if (root / name).is_file()}


def prepare(*, experiment: str, output_root: Path | None = None, smoke: bool = False,
            resume: bool = False) -> dict[str, Any]:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment}")
    formal, frozen_hashes = validate_frozen_inputs()
    selected = select_smoke_cases(formal) if smoke else formal
    root = Path(output_root or default_output(experiment)).resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "format_version": 1, "experiment": experiment, "smoke": bool(smoke),
        "model_path": str(MODEL_PATH.resolve()), "model_config_sha256": sha256_file(MODEL_PATH / "config.json"),
        "frozen_hashes": frozen_hashes, "formal_case_count": len(formal),
        "selected_case_count": len(selected), "selected_case_hash": canonical_hash(selected),
        "windows": [list(window) for window in WINDOWS], "seed": SEED,
        "bootstrap_repeats": BOOTSTRAP_REPEATS, "spec": EXPERIMENTS[experiment].__dict__,
        "implementation_sha256": _implementation_hashes(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3], text=True).strip(),
    }
    config["fingerprint"] = canonical_hash(config)
    path = root / "run_config.json"
    existed = path.exists()
    if existed:
        previous = json.loads(path.read_text())
        if previous.get("fingerprint") != config["fingerprint"]:
            raise ValueError("Existing output uses a different immutable run fingerprint")
        if not resume:
            raise FileExistsError(f"Output exists; use --resume: {root}")
    else:
        atomic_json(path, config)
        atomic_jsonl(root / "artifacts" / "manifests" / "test_manifest.jsonl", selected)
    result = {"status": "complete", "experiment": experiment, "smoke": smoke,
              "case_count": len(selected), "formal_case_count": len(formal),
              "fingerprint": config["fingerprint"], "resumed": existed}
    atomic_json(root / "progress" / "prepare.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS), required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(prepare(experiment=args.experiment, output_root=args.output_root,
                             smoke=args.smoke, resume=args.resume), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
