from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from dp_SA.io_utils import atomic_json, atomic_jsonl, load_jsonl
from .config import ALPHAS, BOOTSTRAP_REPEATS, CHAINS, PAIRS, SEED, default_output


def effect_values(cells: dict[str, float]) -> dict[str, float]:
    if set(cells) != {"C0", "C1", "C2", "C3"}: raise ValueError("Four complete conditions are required")
    return {"total_effect": cells["C1"] - cells["C0"],
            "restore_residual": cells["C2"] - cells["C0"],
            "restored_amount": cells["C1"] - cells["C2"],
            "transplant_effect": cells["C3"] - cells["C0"]}


def summarize(values: Sequence[float], repeats: int, seed: int) -> dict[str, float]:
    vector = np.asarray(values, dtype=float)
    if not len(vector) or not np.isfinite(vector).all(): raise ValueError("Cannot summarize empty/non-finite values")
    rng = np.random.default_rng(seed); boot = vector[rng.integers(0, len(vector), (repeats, len(vector)))].mean(1)
    low, high = np.quantile(boot, [.025, .975])
    return {"mean": float(vector.mean()), "ci95_low": float(low), "ci95_high": float(high), "count": len(vector)}


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def expand_four_cell(trials: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    clean = {r["case_id"]: r for r in trials if r["condition"] == "C0"}
    expanded = []
    for row in trials:
        if row["condition"] == "C0": continue
        baseline = clean[row["case_id"]]
        expanded.append({**row, "clean_final_sa": float(baseline["final_sa"]),
                         "delta_final_sa": float(row["final_sa"]) - float(baseline["final_sa"])})
    for case, baseline in clean.items():
        template = next(r for r in expanded if r["case_id"] == case)
        for chain in CHAINS:
            for upstream, downstream in PAIRS:
                for alpha in ALPHAS:
                    expanded.append({**baseline, "chain": chain,
                        "upstream_position": CHAINS[chain][0], "downstream_position": CHAINS[chain][1],
                        "upstream_layer": upstream, "downstream_layer": downstream, "alpha": alpha,
                        "clean_final_sa": float(baseline["final_sa"]), "delta_final_sa": 0.0,
                        "test_side": template["test_side"]})
    return sorted(expanded, key=lambda r: (r["case_id"], r["chain"], r["upstream_layer"], r["alpha"], r["condition"]))


def analyze(*, output_root: Path, repeats: int = BOOTSTRAP_REPEATS) -> dict[str, Any]:
    root = Path(output_root).resolve(); trials = load_jsonl(root / "artifacts/trials.jsonl")
    logical = expand_four_cell(trials); manifest = load_jsonl(root / "artifacts/manifests/test.jsonl")
    expected = len(manifest) * len(CHAINS) * len(PAIRS) * len(ALPHAS) * 4
    if len(logical) != expected: raise RuntimeError(f"Incomplete logical four-cell grid: {len(logical)}/{expected}")
    conditions, effects = [], []; seed = SEED
    for chain in CHAINS:
        for upstream, downstream in PAIRS:
            for alpha in ALPHAS:
                cell = [r for r in logical if r["chain"] == chain and r["upstream_layer"] == upstream
                        and r["downstream_layer"] == downstream and float(r["alpha"]) == alpha]
                per_case: dict[str, dict[str, float]] = {}
                for row in cell: per_case.setdefault(row["case_id"], {})[row["condition"]] = float(row["final_sa"])
                for condition in ("C0", "C1", "C2", "C3"):
                    for group in ("overall", "text_side", "image_side"):
                        rows = [r for r in cell if r["condition"] == condition and (group == "overall" or r["test_side"] == group)]
                        conditions.append({"chain": chain, "upstream_layer": upstream, "downstream_layer": downstream,
                                           "alpha": alpha, "condition": condition, "group": group,
                                           **summarize([float(r["final_sa"]) for r in rows], repeats, seed)})
                        seed += 1
                case_rows = [{"case_id": case, "test_side": next(r["test_side"] for r in cell if r["case_id"] == case),
                              **effect_values(values)} for case, values in per_case.items()]
                for effect in ("total_effect", "restore_residual", "restored_amount", "transplant_effect"):
                    for group in ("overall", "text_side", "image_side"):
                        rows = case_rows if group == "overall" else [r for r in case_rows if r["test_side"] == group]
                        effects.append({"chain": chain, "upstream_layer": upstream, "downstream_layer": downstream,
                                        "alpha": alpha, "effect": effect, "group": group,
                                        **summarize([float(r[effect]) for r in rows], repeats, seed)})
                        seed += 1
    atomic_jsonl(root / "artifacts/logical_four_cell.jsonl", logical)
    _csv(root / "tables/condition_summary.csv", conditions); _csv(root / "tables/effect_summary.csv", effects)
    summary = {"status": "complete", "case_count": len(manifest), "physical_trial_count": len(trials),
               "logical_row_count": len(logical), "side_counts": dict(Counter(r["test_side"] for r in manifest)),
               "bootstrap_repeats": repeats}
    atomic_json(root / "tables/summary.json", summary); return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze five-class steering mediation")
    parser.add_argument("--output-root", type=Path, default=default_output(False))
    parser.add_argument("--bootstrap-repeats", type=int, default=BOOTSTRAP_REPEATS)
    args = parser.parse_args(argv); print(json.dumps(analyze(output_root=args.output_root,
                                                              repeats=args.bootstrap_repeats), ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
