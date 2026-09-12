from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np


def bh_fdr(pvalues: Sequence[float]) -> list[float]:
    values = np.asarray(pvalues, dtype=float); order = np.argsort(values); adjusted = np.empty(len(values)); running = 1.0
    for reverse_rank in range(len(values) - 1, -1, -1):
        index = order[reverse_rank]; rank = reverse_rank + 1
        running = min(running, float(values[index]) * len(values) / rank); adjusted[index] = running
    return adjusted.tolist()


def _summary(values: Sequence[float], draws: np.ndarray) -> dict[str, Any]:
    point = float(np.mean(values))
    return {"estimate": point, "sem": float(np.std(draws, ddof=1)), "ci_low": float(np.quantile(draws, .025)),
            "ci_high": float(np.quantile(draws, .975)), "bootstrap_p_two_sided": float(min(1.0, 2 * min(np.mean(draws <= 0), np.mean(draws >= 0)))),
            "n": len(values), "valid_bootstrap_repeats": int(len(draws))}


def donor_contrast(rows: Sequence[dict[str, Any]], *, repeats: int, seed: int) -> list[dict[str, Any]]:
    cells = sorted({(str(r["window"]), int(r["layer"])) for r in rows}); rng = np.random.default_rng(seed); output = []
    for window, layer in cells:
        subset = [r for r in rows if r["window"] == window and int(r["layer"]) == layer]
        paired: dict[str, dict[str, Any]] = {}
        for row in subset: paired.setdefault(str(row["case_id"]), {"row": row})[str(row["donor_side"])] = float(row["patched_soft_sa"])
        items = []
        for case, values in paired.items():
            if "high_image" not in values or "high_text" not in values: continue
            row = values["row"]; items.append({"case_id": case, "family_id": str(row["family_id"]), "side": str(row["recipient_side"]),
                                               "d": values["high_image"] - values["high_text"]})
        by_side = {side: [x for x in items if x["side"] == side] for side in ("high_image", "high_text")}
        side_draws: dict[str, np.ndarray] = {}
        for side, values in by_side.items():
            array = np.asarray([x["d"] for x in values]); side_draws[side] = np.asarray([np.mean(rng.choice(array, size=len(array), replace=True)) for _ in range(repeats)])
            output.append({"window": window, "layer": layer, "stratum": side, **_summary(array.tolist(), side_draws[side])})
        combined_draws = (side_draws["high_image"] + side_draws["high_text"]) / 2
        combined_point = (np.mean([x["d"] for x in by_side["high_image"]]) + np.mean([x["d"] for x in by_side["high_text"]])) / 2
        combined = _summary([combined_point], combined_draws); combined["estimate"] = float(combined_point); combined["n"] = len(items)
        output.append({"window": window, "layer": layer, "stratum": "combined_equal_side", **combined})
    combined_indices = [i for i, row in enumerate(output) if row["stratum"] == "combined_equal_side"]
    adjusted = bh_fdr([output[i]["bootstrap_p_two_sided"] for i in combined_indices])
    for index, value in zip(combined_indices, adjusted): output[index]["bh_fdr_q"] = value
    return output

