from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def summarize(path: Path) -> dict:
    groups: dict[tuple[str, int, float], list[dict]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "completed":
            groups[(row["position"], int(row["layer"]), float(row["alpha"]))].append(row)

    cells = []
    for (position, layer, alpha), rows in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        deltas = [float(row["delta_soft_sa"]) for row in rows]
        steered = [float(row["steered_soft_sa"]) for row in rows]
        changes = [bool(row["hard_class_changed"]) for row in rows]
        hooks = [int(row["hook_diagnostics"].get("applied_count", -1)) for row in rows]
        cells.append({
            "position": position, "layer": layer, "alpha": alpha, "n": len(rows),
            "delta_soft_sa_mean": sum(deltas) / len(deltas),
            "delta_soft_sa_std": math.sqrt(sum((x - sum(deltas) / len(deltas)) ** 2 for x in deltas) / len(deltas)),
            "steered_soft_sa_mean": sum(steered) / len(steered),
            "hard_class_change_rate": sum(changes) / len(changes),
            "hook_applied_counts": sorted(set(hooks)),
            "alpha_zero_max_abs_delta": max((abs(x) for x in deltas), default=0.0) if alpha == 0.0 else None,
        })
    return {"source": str(path), "completed_cells": sum(x["n"] for x in cells), "cell_count": len(cells), "cells": cells}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("completed_cells", "cell_count")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
