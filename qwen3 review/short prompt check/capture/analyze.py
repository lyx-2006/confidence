from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
SHORT_ROOT = HERE.parent
REVIEW_ROOT = SHORT_ROOT.parent
REPOSITORY_ROOT = REVIEW_ROOT.parent
for candidate in (REPOSITORY_ROOT, REVIEW_ROOT, SHORT_ROOT):
    if str(candidate) not in sys.path: sys.path.insert(0, str(candidate))

import numpy as np
from scipy.stats import pearsonr, spearmanr

from dp_SA.io_utils import atomic_json, load_jsonl


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fields = sorted({key for row in rows for key in row})
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def analyze(output_root: Path = SHORT_ROOT / "output") -> dict[str, Any]:
    long_rows = {row["case_id"]: row for row in load_jsonl(REVIEW_ROOT / "capture" / "results.jsonl") if row.get("status") == "completed"}
    short_rows = {row["case_id"]: row for row in load_jsonl(output_root / "capture" / "results.jsonl") if row.get("status") == "completed"}
    if set(long_rows) != set(short_rows) or len(short_rows) != 500:
        raise ValueError(f"Clean comparison requires the same 500 cases: long={len(long_rows)} short={len(short_rows)}")
    rows = []
    for case_id in sorted(short_rows):
        long, short = long_rows[case_id], short_rows[case_id]
        if long["phase0_answer_fingerprint"] != short["phase0_answer_fingerprint"]:
            raise ValueError(f"Fixed answer drift: {case_id}")
        left, right = float(long["soft_sa_image_score"]), float(short["soft_sa_image_score"])
        rows.append({
            "case_id": case_id, "item_id": short["item_id"], "condition": short["condition"],
            "answer": short["phase0_normalized_answer"], "long_soft_sa": left, "short_soft_sa": right,
            "short_minus_long": right - left, "absolute_error": abs(right - left),
            "long_hard_class": int(long["argmax_hard_class"]), "short_hard_class": int(short["argmax_hard_class"]),
            "direction_agrees": int(np.sign(left - .5) == np.sign(right - .5)),
        })
    x = np.asarray([row["long_soft_sa"] for row in rows]); y = np.asarray([row["short_soft_sa"] for row in rows])
    metrics = {
        "case_count": len(rows), "family_count": len({str(row["item_id"]) for row in rows}),
        "pearson": float(pearsonr(x, y).statistic), "spearman": float(spearmanr(x, y).statistic),
        "mae": float(np.mean(np.abs(y - x))), "direction_agreement_rate": float(np.mean(np.sign(x - .5) == np.sign(y - .5))),
        "mean_short_minus_long": float(np.mean(y - x)),
        "hard_class_exact_agreement": float(np.mean([row["long_hard_class"] == row["short_hard_class"] for row in rows])),
    }
    target = output_root / "comparison"; _csv(target / "tables" / "clean_case_level.csv", rows)
    _csv(target / "tables" / "clean_long_short_metrics.csv", [metrics]); atomic_json(target / "clean_summary.json", metrics)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (target / "figures").mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6)); ax.scatter(x, y, s=15, alpha=.5); ax.plot([.05, .95], [.05, .95], color="black", lw=1)
    ax.set(xlabel="long clean soft-SA", ylabel="short clean soft-SA", xlim=(.03, .97), ylim=(.03, .97))
    ax.text(.05, .95, f"Pearson={metrics['pearson']:.3f}\nSpearman={metrics['spearman']:.3f}\nMAE={metrics['mae']:.3f}\nDirection={metrics['direction_agreement_rate']:.3f}", transform=ax.transAxes, va="top")
    ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(target / "figures" / "clean_soft_sa_scatter.png", dpi=220); plt.close(fig)
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root", type=Path, default=SHORT_ROOT / "output")
    args = parser.parse_args(argv); print(json.dumps(analyze(args.output_root), ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())

