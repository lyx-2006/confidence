from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

from .analyze import load_trials, summarize
from .config import FORMAL_ROOT, OUTPUT_PARENT, SMOKE_CELLS, WINDOWS, require_output_root
from .io_utils import atomic_json, load_jsonl
from .prepare import prepare
from .run import execute, selected_rows


def _required(root: Path) -> list[Path]:
    return [root / "README_zh.md", root / "summary.json", root / "run_config.json",
            *[root / f"tables/{name}.csv" for name in ("condition_summary", "donor_contrasts", "toward_donor", "answer_sensitivity", "leave_one_donor_out")],
            *[root / f"figures/{name}.png" for name in ("fig1_donor_contrast_heatmap", "fig2_conditions_by_layer", "fig3_donor_gap_scatter", "fig4_toward_rate_heatmap", "fig5_disruption_diagnostics")]]


def verify(root: Path, *, smoke: bool) -> dict[str, bool]:
    recipients, donors, pairs = selected_rows(root, smoke=smoke); clean, trials = load_trials(root)
    expected_trials = len(recipients) * 2 * (len(SMOKE_CELLS) if smoke else 24)
    noop = list((root / "artifacts/trials").glob("*__noop.json"))
    hooks = [row.get("hook", {}) for row in trials]
    gates = {
        "recipient_count": len(recipients) == (2 if smoke else 50), "recipient_sides": {r["sa_side"] for r in recipients} == {"high_image", "high_text"},
        "pair_count": len(pairs) == 2 * len(recipients), "clean_count": len(clean) == len(recipients), "trial_count": len(trials) == expected_trials,
        "noop_count": len(noop) == (2 if smoke else 0), "hook_invariants": all(h.get("target_exact") and h.get("outside_exact") and h.get("applied_count") == 1 for h in hooks),
        "layers": {int(r["layer"]) for r in trials} == ({12, 15, 18, 21} if smoke else {12, 15, 18, 21}),
        "windows": {r["window"] for r in trials} == set(WINDOWS),
        "donor_cache": all((root / f"artifacts/donor_hidden/{str(r['case_id']).replace('/', '_')}.npz").is_file() for r in donors),
        "outputs": all(path.is_file() and path.stat().st_size > 0 for path in _required(root)),
    }
    if not all(gates.values()): raise RuntimeError(f"Completion gates failed: {gates}")
    return gates


def _next_smoke_root() -> Path | None:
    parent = OUTPUT_PARENT / "smoke"
    first = parent / "round_1"; second = parent / "round_2"
    if first.exists():
        report = first / "smoke_report.json"
        if report.exists() and json.loads(report.read_text()).get("status") == "passed": return None
        if second.exists():
            report = second / "smoke_report.json"
            if report.exists() and json.loads(report.read_text()).get("status") == "passed": return None
            raise RuntimeError("Two smoke rounds already exist; refusing to start a third")
        return second
    return first


def run_smoke(*, num_gpus: int) -> dict[str, Any]:
    root = _next_smoke_root()
    if root is None:
        reports = sorted((OUTPUT_PARENT / "smoke").glob("round_*/smoke_report.json"))
        return json.loads(reports[-1].read_text())
    root.mkdir(parents=True, exist_ok=False); started = time.time()
    try:
        prepared = prepare(root, resume=False); first = execute(root, smoke=True, num_gpus=num_gpus, resume=False)
        if int(first["new_gpu_forwards"]) > 44: raise RuntimeError(f"Smoke exceeded 44-forward cap: {first}")
        summary = summarize(root, smoke=True); gates = verify(root, smoke=True)
        second = execute(root, smoke=True, num_gpus=num_gpus, resume=True)
        if second["new_gpu_forwards"] != 0: raise RuntimeError("Resume repeated GPU forwards")
        result = {"status": "passed", "smoke_only": True, "root": str(root.resolve()), "num_gpus": num_gpus,
                  "first_new_gpu_forwards": first["new_gpu_forwards"], "resume_new_gpu_forwards": 0,
                  "formal_started": False, "fingerprint": prepared["fingerprint"], "gates": gates,
                  "summary": summary, "elapsed_seconds": time.time() - started}
        atomic_json(root / "smoke_report.json", result); return result
    except Exception as exc:
        atomic_json(root / "smoke_report.json", {"status": "failed", "smoke_only": True, "formal_started": False,
                                                  "error_type": type(exc).__name__, "error": str(exc), "elapsed_seconds": time.time() - started})
        raise


def _matching_smoke(fingerprint: str) -> Path:
    for path in sorted((OUTPUT_PARENT / "smoke").glob("round_*/smoke_report.json"), reverse=True):
        report = json.loads(path.read_text())
        if report.get("status") == "passed" and report.get("fingerprint") == fingerprint and not report.get("formal_started"): return path
    raise RuntimeError("No successful smoke matches current source/code fingerprint")


def run_formal(*, num_gpus: int, resume: bool) -> dict[str, Any]:
    if not resume: raise ValueError("Formal execution requires --resume")
    prepared = prepare(FORMAL_ROOT, resume=FORMAL_ROOT.exists()); smoke = _matching_smoke(prepared["fingerprint"])
    execution = execute(FORMAL_ROOT, smoke=False, num_gpus=num_gpus, resume=FORMAL_ROOT.exists())
    summary = summarize(FORMAL_ROOT, smoke=False); gates = verify(FORMAL_ROOT, smoke=False)
    result = {"status": "complete", "smoke_gate": str(smoke), "execution": execution, "summary": summary, "gates": gates}
    atomic_json(FORMAL_ROOT / "completion.json", result); return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PANL-to-CLE answer-matched activation-window swap")
    mode = parser.add_mutually_exclusive_group(required=True); mode.add_argument("--smoke", action="store_true"); mode.add_argument("--formal", action="store_true")
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--num-gpus", type=int, choices=(1, 2), default=1); args = parser.parse_args(argv)
    if args.smoke and args.resume: parser.error("Smoke performs its own resume-noop check")
    result = run_smoke(num_gpus=args.num_gpus) if args.smoke else run_formal(num_gpus=args.num_gpus, resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())

