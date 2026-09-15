from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
SHORT_ROOT = HERE.parent
REPOSITORY_ROOT = SHORT_ROOT.parent.parent
for candidate in (REPOSITORY_ROOT, SHORT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np
from scipy.stats import pearsonr, spearmanr

from dp_SA.io_utils import atomic_json, load_jsonl, sha256_file


OUTPUT_ROOT = SHORT_ROOT / "output" / "capture_reverse"
SHORT_CAPTURE_ROOT = SHORT_ROOT / "output" / "capture"


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _completed(path: Path) -> dict[str, dict[str, Any]]:
    rows = [row for row in load_jsonl(path) if row.get("status") == "completed"]
    if len(rows) != len({row["case_id"] for row in rows}):
        raise ValueError(f"Duplicate completed cases in {path}")
    return {row["case_id"]: row for row in rows}


def analyze(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    root = Path(output_root)
    short = _completed(SHORT_CAPTURE_ROOT / "results.jsonl")
    reverse = _completed(root / "results.jsonl")
    if len(short) != 500 or set(short) != set(reverse):
        raise ValueError(f"Reverse comparison requires the same 500 cases: short={len(short)} reverse={len(reverse)}")

    case_rows: list[dict[str, Any]] = []
    for case_id in sorted(short):
        ordinary, flipped = short[case_id], reverse[case_id]
        if ordinary["phase0_answer_fingerprint"] != flipped["phase0_answer_fingerprint"]:
            raise ValueError(f"Fixed answer drift: {case_id}")
        hidden_path = root / flipped["hidden_file"]
        if not hidden_path.is_file():
            raise FileNotFoundError(hidden_path)
        ordinary_sa = float(ordinary["soft_sa_image_score"])
        reverse_sa = float(flipped["soft_sa_image_score"])
        ordinary_raw = int(ordinary["argmax_hard_class"])
        reverse_raw = int(flipped["raw_argmax_class"])
        reverse_canonical = int(flipped["argmax_hard_class"])
        case_rows.append({
            "case_id": case_id,
            "item_id": flipped["item_id"],
            "condition": flipped["condition"],
            "answer": flipped["phase0_normalized_answer"],
            "short_soft_sa": ordinary_sa,
            "reverse_soft_sa": reverse_sa,
            "reverse_minus_short": reverse_sa - ordinary_sa,
            "absolute_error": abs(reverse_sa - ordinary_sa),
            "short_raw_hard_class": ordinary_raw,
            "reverse_raw_hard_class": reverse_raw,
            "reverse_canonical_hard_class": reverse_canonical,
            "raw_label_sum": ordinary_raw + reverse_raw,
            "raw_label_exact_reversal": int(reverse_raw == 8 - ordinary_raw),
            "canonical_hard_class_agreement": int(reverse_canonical == ordinary_raw),
            "canonical_direction_agreement": int(np.sign(reverse_sa - .5) == np.sign(ordinary_sa - .5)),
            "reverse_hidden_sha256": sha256_file(hidden_path),
        })

    x = np.asarray([row["short_soft_sa"] for row in case_rows], dtype=float)
    y = np.asarray([row["reverse_soft_sa"] for row in case_rows], dtype=float)
    raw_pairs = Counter((row["short_raw_hard_class"], row["reverse_raw_hard_class"]) for row in case_rows)
    metrics = {
        "status": "complete",
        "case_count": len(case_rows),
        "family_count": len({str(row["item_id"]) for row in case_rows}),
        "canonical_score_definition": "higher means stronger image contribution in both columns",
        "pearson": float(pearsonr(x, y).statistic),
        "spearman": float(spearmanr(x, y).statistic),
        "mae": float(np.mean(np.abs(y - x))),
        "mean_short_soft_sa": float(x.mean()),
        "mean_reverse_soft_sa": float(y.mean()),
        "mean_reverse_minus_short": float(np.mean(y - x)),
        "canonical_direction_agreement_rate": float(np.mean(np.sign(x - .5) == np.sign(y - .5))),
        "raw_label_exact_reversal_rate": float(np.mean([row["raw_label_exact_reversal"] for row in case_rows])),
        "canonical_hard_class_agreement_rate": float(np.mean([row["canonical_hard_class_agreement"] for row in case_rows])),
        "short_raw_class_counts": dict(sorted(Counter(row["short_raw_hard_class"] for row in case_rows).items())),
        "reverse_raw_class_counts": dict(sorted(Counter(row["reverse_raw_hard_class"] for row in case_rows).items())),
        "raw_class_pair_counts": {f"short_{a}__reverse_{b}": n for (a, b), n in sorted(raw_pairs.items())},
    }
    _csv(root / "tables" / "short_reverse_case_comparison.csv", case_rows)
    _csv(root / "tables" / "short_reverse_metrics.csv", [metrics])
    atomic_json(root / "comparison_summary.json", metrics)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 6.2))
    ax.scatter(x, y, s=18, alpha=.48, edgecolors="none")
    low = min(float(x.min()), float(y.min())) - .02
    high = max(float(x.max()), float(y.max())) + .02
    ax.plot([low, high], [low, high], color="#333333", lw=1, linestyle="--")
    ax.axvline(.5, color="#999999", lw=.8)
    ax.axhline(.5, color="#999999", lw=.8)
    ax.set(xlabel="Short prompt canonical image-side soft-SA",
           ylabel="Reverse prompt canonical image-side soft-SA",
           xlim=(low, high), ylim=(low, high))
    ax.text(.04, .96,
            f"Pearson={metrics['pearson']:.3f}\nSpearman={metrics['spearman']:.3f}\n"
            f"MAE={metrics['mae']:.3f}\nDirection={metrics['canonical_direction_agreement_rate']:.3f}",
            transform=ax.transAxes, va="top")
    ax.grid(alpha=.18)
    fig.tight_layout()
    fig.savefig(figures / "short_reverse_canonical_sa_scatter.png", dpi=220)
    plt.close(fig)
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)
    print(json.dumps(analyze(args.output_root), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

