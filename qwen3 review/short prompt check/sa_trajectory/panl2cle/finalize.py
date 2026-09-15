"""Validate completed short trajectory artifacts and write a completion marker."""
from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    short_root = Path(__file__).resolve().parents[2]
    for candidate in (short_root, short_root.parent, short_root.parent.parent):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    __package__ = "sa_trajectory.panl2cle"

from .config import CLE_LAYERS, PAIRS
from .contracts import atomic_json, expected_logical_count, expected_physical_count, load_jsonl, physical_trial_key, sha256_file


def finalize(root: Path):
    root = Path(root)
    physical = load_jsonl(root / "artifacts/trials.jsonl")
    logical = load_jsonl(root / "artifacts/logical_four_cell.jsonl")
    probes = load_jsonl(root / "artifacts/probes/probe_index.jsonl")
    run = json.loads((root / "progress/run.json").read_text())
    analysis = json.loads((root / "analysis_summary.json").read_text())
    prepared = json.loads((root / "fingerprint.json").read_text())
    if run.get("status") != "complete" or len(physical) != expected_physical_count(80):
        raise RuntimeError("Physical trajectory grid is incomplete")
    if len({physical_trial_key(r) for r in physical}) != len(physical):
        raise RuntimeError("Duplicate physical trajectory trial")
    if len(logical) != expected_logical_count(80):
        raise RuntimeError("Logical four-cell grid is incomplete")
    if len(probes) != len(CLE_LAYERS) or not all(r["readout_reliable"] for r in probes):
        raise RuntimeError("Probe reliability gate is incomplete")
    if analysis.get("status") != "complete" or analysis.get("bootstrap_repeats") != 2000:
        raise RuntimeError("Analysis/bootstrap is incomplete")
    result = {
        "status": "complete", "case_count": 80, "physical_trial_count": len(physical),
        "logical_row_count": len(logical), "probe_layers": list(CLE_LAYERS),
        "pairs": [list(pair) for pair in PAIRS], "alphas": [-5.0, 5.0],
        "bootstrap_repeats": 2000, "run": run, "analysis": analysis,
        "postrun_code_hashes": {
            str(path): sha256_file(path)
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "prepared_code_hashes": prepared.get("code_hashes", {}),
    }
    atomic_json(root / "completion.json", result)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.output_root), ensure_ascii=False, indent=2))
