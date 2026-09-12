from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .config import RECIPIENTS_PER_SIDE, SEED

SIDES = ("high_image", "high_text")


def side_from_class(value: int) -> str:
    if int(value) in (0, 1, 2, 3): return "high_text"
    if int(value) in (5, 6, 7, 8): return "high_image"
    if int(value) == 4: return "balanced"
    raise ValueError(f"Invalid SA class: {value}")


def stable_fraction(seed: int, *parts: Any) -> float:
    digest = hashlib.sha256("|".join(map(str, (seed, *parts))).encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _candidate_rows(candidates: Sequence[dict[str, Any]], audit_ids: set[str], construction: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    blocked = {
        "case_id": {str(r["case_id"]) for r in construction},
        "family_id": {str(r["family_id"]) for r in construction},
        "item_id": {str(r["item_id"]) for r in construction},
    }
    best: dict[tuple[str, str, str], dict[str, Any]] = {}
    for source in candidates:
        if source.get("status") != "completed" or not source.get("valid_class", True): continue
        side = str(source.get("sa_side") or side_from_class(int(source["argmax_hard_class"])))
        if side not in SIDES: continue
        if any(str(source[field]) in blocked[field] for field in blocked): continue
        answer = str(source.get("canonical_answer") or source.get("phase0_normalized_answer") or "")
        if not answer: continue
        row = {**source, "sa_side": side, "canonical_answer": answer,
               "recipient_source": "audit" if str(source["case_id"]) in audit_ids else "fallback"}
        key = (str(row["family_id"]), side, answer)
        if key not in best or stable_fraction(SEED, row["case_id"]) < stable_fraction(SEED, best[key]["case_id"]):
            best[key] = row
    return sorted(best.values(), key=lambda r: str(r["case_id"]))


def select_recipients(candidates: Sequence[dict[str, Any]], audit: Sequence[dict[str, Any]],
                      construction: Sequence[dict[str, Any]], *, per_side: int = RECIPIENTS_PER_SIDE,
                      seed: int = SEED, allowed_answers: set[str] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audit_ids = {str(row["case_id"]) for row in audit}
    rows = _candidate_rows(candidates, audit_ids, construction)
    if allowed_answers is not None:
        rows = [row for row in rows if str(row["canonical_answer"]) in allowed_answers]
    answers = sorted({str(row["canonical_answer"]) for row in rows})
    cells = [(side, answer) for side in SIDES for answer in answers]
    n, m = len(rows), len(cells)
    # Variables: selected rows x, covered answer-side cells y, absolute cell deviations d.
    size = n + m + m; objective = np.zeros(size)
    objective[:n] = [(-1_000_000.0 if row["recipient_source"] == "audit" else 0.0)
                     + stable_fraction(seed, "recipient", row["case_id"]) for row in rows]
    objective[n:n + m] = -10_000.0
    objective[n + m:] = 10.0
    constraints: list[np.ndarray] = []; lower: list[float] = []; upper: list[float] = []
    for side in SIDES:
        vector = np.zeros(size); vector[:n] = [int(row["sa_side"] == side) for row in rows]
        constraints.append(vector); lower.append(per_side); upper.append(per_side)
    for field in ("family_id", "item_id"):
        for value in sorted({str(row[field]) for row in rows}):
            vector = np.zeros(size); vector[:n] = [int(str(row[field]) == value) for row in rows]
            constraints.append(vector); lower.append(-np.inf); upper.append(1)
    target = per_side / max(len(answers), 1)
    for cell_index, (side, answer) in enumerate(cells):
        indices = [index for index, row in enumerate(rows) if row["sa_side"] == side and row["canonical_answer"] == answer]
        # y <= sum(x in cell)
        vector = np.zeros(size); vector[n + cell_index] = 1; vector[indices] = -1
        constraints.append(vector); lower.append(-np.inf); upper.append(0)
        # d >= |sum(x in cell) - target|
        first = np.zeros(size); first[indices] = 1; first[n + m + cell_index] = -1
        constraints.append(first); lower.append(-np.inf); upper.append(target)
        second = np.zeros(size); second[indices] = -1; second[n + m + cell_index] = -1
        constraints.append(second); lower.append(-np.inf); upper.append(-target)
    matrix = np.vstack(constraints)
    result = milp(objective, integrality=np.r_[np.ones(n + m), np.zeros(m)],
                  bounds=Bounds(np.zeros(size), np.r_[np.ones(n + m), np.full(m, np.inf)]),
                  constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
                  options={"time_limit": 120})
    if not result.success or result.x is None: raise ValueError(f"Recipient allocation failed: {result.message}")
    selected = [{**rows[i], "recipient_rank": 0} for i in range(n) if result.x[i] > .5]
    selected.sort(key=lambda row: (SIDES.index(row["sa_side"]), row["canonical_answer"], str(row["family_id"])))
    ranks = Counter()
    for row in selected: ranks[row["sa_side"]] += 1; row["recipient_rank"] = ranks[row["sa_side"]]
    if ranks != Counter({side: per_side for side in SIDES}): raise AssertionError(f"Bad recipient counts: {ranks}")
    for field in ("case_id", "family_id", "item_id"):
        if len({str(row[field]) for row in selected}) != len(selected): raise AssertionError(f"Recipient {field} is not unique")
    diagnostics = {
        "seed": seed, "donor_feasible_answers": answers, "counts": dict(ranks), "audit_count": sum(r["recipient_source"] == "audit" for r in selected),
        "fallback_count": sum(r["recipient_source"] == "fallback" for r in selected),
        "answer_counts": {side: dict(Counter(r["canonical_answer"] for r in selected if r["sa_side"] == side)) for side in SIDES},
        "unique": {field: len({str(r[field]) for r in selected}) for field in ("case_id", "family_id", "item_id")},
    }
    return selected, diagnostics
