from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from dp_SA.io_utils import atomic_json

from .config import DIRECTIONS, EXPECTED_CASES, LAYERS, OUTPUT_ROOT


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finalize(root: Path = OUTPUT_ROOT):
    root = Path(root).resolve()
    manifest = [json.loads(x) for x in (root / "artifacts/manifests/test_manifest.jsonl").read_text().splitlines() if x]
    trials = [json.loads(x) for x in (root / "artifacts/trials.jsonl").read_text().splitlines() if x]
    if len(manifest) != EXPECTED_CASES or len({r["item_id"] for r in manifest}) != EXPECTED_CASES:
        raise RuntimeError("50-case item-disjoint manifest incomplete")
    expected = EXPECTED_CASES * len(LAYERS) * len(DIRECTIONS)
    if len(trials) != expected:
        raise RuntimeError(f"Expected {expected} swap trials, found {len(trials)}")
    if any(r.get("status") != "completed" for r in trials):
        raise RuntimeError("Incomplete swap trial")
    hook_bad = [r["case_id"] for r in trials if not (
        r["hook"].get("replacement_bitwise_equal") and r["hook"].get("non_target_unchanged")
        and int(r["hook"].get("cle_patch_count", 0)) == 1
    )]
    if hook_bad:
        raise RuntimeError(f"Hook audit failures: {hook_bad[:3]}")
    clean_count = len(list((root / "artifacts/trials").glob("*__clean.json")))
    hidden_count = len(list((root / "artifacts/hidden").glob("*.npz")))
    required_figures = sorted((root / "figures").glob("*.png"))
    if len(required_figures) < 4 or any(p.stat().st_size == 0 for p in required_figures):
        raise RuntimeError("Expected four non-empty figures")
    completion = {
        "status": "complete", "case_count": EXPECTED_CASES, "clean_trial_count": clean_count,
        "swap_trial_count": len(trials), "hidden_source_file_count": hidden_count,
        "layers": list(LAYERS), "directions": list(DIRECTIONS), "bootstrap_repeats": 2000,
        "hook_audit_failures": len(hook_bad),
        "artifacts": {str(p.relative_to(root)): sha256(p) for p in required_figures},
    }
    atomic_json(root / "completion.json", completion)
    return completion


def main(argv=None):
    p = argparse.ArgumentParser(); p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    print(json.dumps(finalize(p.parse_args(argv).output_root), ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())

