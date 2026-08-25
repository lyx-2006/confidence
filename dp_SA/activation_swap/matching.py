from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any, Iterable, Sequence

import numpy as np

from .utils import stable_key, stable_seed


def quantile_edges(values: Sequence[int], bins: int = 10) -> list[float]:
    if not values:
        raise ValueError("cannot build quantile bins from an empty cohort")
    if bins < 1:
        raise ValueError("bins must be positive")
    # Token lengths are discrete.  Nearest-rank cut points avoid inventing
    # fractional boundaries (e.g. 35.4 tokens) and make duplicate-boundary
    # folding deterministic.
    raw = np.quantile(np.asarray(values, dtype=np.float64), np.linspace(0, 1, bins + 1), method="nearest").tolist()
    output: list[float] = []
    for edge in raw:
        if not output or edge > output[-1]:
            output.append(float(edge))
    return output


def quantile_bin(value: int, edges: Sequence[float]) -> int:
    if len(edges) < 2:
        return 0
    return min(len(edges) - 2, max(0, int(np.searchsorted(np.asarray(edges), float(value), side="right") - 1)))


def add_length_bins(rows: Sequence[dict[str, Any]], *, bins: int = 10) -> dict[str, list[float]]:
    question_edges = quantile_edges([int(row["question_token_length"]) for row in rows], bins)
    answer_edges = quantile_edges([int(row["answer_token_length"]) for row in rows], bins)
    for row in rows:
        row["question_quantile_bin"] = quantile_bin(int(row["question_token_length"]), question_edges)
        row["answer_quantile_bin"] = quantile_bin(int(row["answer_token_length"]), answer_edges)
    return {"question": question_edges, "answer": answer_edges}


def _candidate_score(recipient: dict[str, Any], donor: dict[str, Any], usage: int, seed: int) -> tuple[Any, ...]:
    answer_count_mismatch = int(int(recipient["answer_token_length"]) != int(donor["answer_token_length"]))
    question_bin_mismatch = int(recipient["question_quantile_bin"] != donor["question_quantile_bin"])
    answer_bin_mismatch = int(recipient["answer_quantile_bin"] != donor["answer_quantile_bin"])
    normalized_mismatch = int(recipient.get("phase0_normalized_answer") != donor.get("phase0_normalized_answer"))
    question_delta = abs(int(recipient["question_token_length"]) - int(donor["question_token_length"]))
    answer_delta = abs(int(recipient["answer_token_length"]) - int(donor["answer_token_length"]))
    tie = stable_seed(seed, "donor-tie", recipient["case_id"], donor["case_id"])
    return (answer_count_mismatch, question_bin_mismatch, answer_bin_mismatch,
            normalized_mismatch, question_delta, answer_delta, int(usage), tie)


def _assign(
    recipients: Sequence[dict[str, Any]], donors: Sequence[dict[str, Any]], *, donor_side: str,
    recipient_side: str | None, swap_kind: str | None, seed: int, usage: Counter[str] | None = None,
    demands: Sequence[tuple[dict[str, Any], str, str]] | None = None,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    pool = [row for row in donors if row.get("construction_side") == donor_side]
    if not pool:
        raise ValueError(f"empty donor pool: {donor_side}")
    if demands is None:
        if recipient_side is None or swap_kind is None:
            raise ValueError("recipient_side and swap_kind are required without explicit demands")
        demand_rows = [(row, recipient_side, swap_kind) for row in recipients]
    else:
        demand_rows = list(demands)
    demand_rows.sort(key=lambda item: stable_key(item[0]))
    random.Random(stable_seed(seed, "recipient-demand", donor_side)).shuffle(demand_rows)
    usage = usage if usage is not None else Counter()
    output: list[dict[str, Any]] = []
    for recipient, recipient_side, swap_kind in demand_rows:
        donor = min(pool, key=lambda item: _candidate_score(recipient, item, usage[str(item["case_id"])], seed))
        usage[str(donor["case_id"])] += 1
        output.append({
            "recipient_case_id": str(recipient["case_id"]),
            "recipient_item_id": str(recipient["item_id"]),
            "recipient_side": recipient_side,
            "donor_case_id": str(donor["case_id"]),
            "donor_item_id": str(donor["item_id"]),
            "donor_side": "image" if donor_side == "high_image" else "text",
            "swap_kind": swap_kind,
            "condition": ("I_from_I" if recipient_side == "image_side" and donor_side == "high_image" else
                           "I_from_T" if recipient_side == "image_side" else
                           "T_from_T" if donor_side == "high_text" else "T_from_I"),
            "answer_token_length_equal": int(recipient["answer_token_length"]) == int(donor["answer_token_length"]),
            "question_quantile_bin_equal": recipient["question_quantile_bin"] == donor["question_quantile_bin"],
            "answer_quantile_bin_equal": recipient["answer_quantile_bin"] == donor["answer_quantile_bin"],
            "normalized_answer_match": recipient.get("phase0_normalized_answer") == donor.get("phase0_normalized_answer"),
            "question_token_length_delta": abs(int(recipient["question_token_length"]) - int(donor["question_token_length"])),
            "answer_token_length_delta": abs(int(recipient["answer_token_length"]) - int(donor["answer_token_length"])),
            "donor_use_index": int(usage[str(donor["case_id"])]),
        })
    return output, usage


def build_swap_pairs(
    recipients: Sequence[dict[str, Any]], donors: Sequence[dict[str, Any]], *, seed: int, bins: int = 10,
    bin_reference_rows: Sequence[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # The frozen 150-row cohort defines the common decile cutpoints.  Callers
    # may pass a larger immutable reference cohort when running a smoke subset;
    # selected recipient/donor rows still receive the resulting bins below.
    all_rows = list(bin_reference_rows) if bin_reference_rows is not None else [*recipients, *donors]
    edges = add_length_bins(all_rows, bins=bins)
    for row in [*recipients, *donors]:
        row["question_quantile_bin"] = quantile_bin(int(row["question_token_length"]), edges["question"])
        row["answer_quantile_bin"] = quantile_bin(int(row["answer_token_length"]), edges["answer"])
    recipient_items = {str(row["item_id"]) for row in recipients}
    donor_items = {str(row["item_id"]) for row in donors}
    if recipient_items & donor_items:
        raise ValueError("donor and recipient item IDs overlap")
    pairs: list[dict[str, Any]] = []
    usage: Counter[str] = Counter()
    # Same/cross demands sharing a donor pool are jointly shuffled and
    # assigned.  This prevents the first condition from consuming the
    # balanced reuse budget before the opposite recipient side is considered.
    for donor_side, demand_rows in (
        ("high_image", [(row, "image_side", "same") for row in recipients if row.get("test_side") == "image_side"]
                        + [(row, "text_side", "cross") for row in recipients if row.get("test_side") == "text_side"]),
        ("high_text", [(row, "text_side", "same") for row in recipients if row.get("test_side") == "text_side"]
                       + [(row, "image_side", "cross") for row in recipients if row.get("test_side") == "image_side"]),
    ):
        assigned, usage = _assign(recipients, donors, donor_side=donor_side, recipient_side=None,
                                  swap_kind=None, seed=seed, usage=usage, demands=demand_rows)
        pairs.extend(assigned)
    by_recipient: dict[str, list[dict[str, Any]]] = {}
    for pair in pairs:
        by_recipient.setdefault(pair["recipient_case_id"], []).append(pair)
    if set(by_recipient) != {str(row["case_id"]) for row in recipients}:
        raise ValueError("every recipient must receive exactly two mappings")
    for case_id, rows in by_recipient.items():
        if len(rows) != 2 or {row["swap_kind"] for row in rows} != {"same", "cross"}:
            raise ValueError(f"invalid same/cross mapping for {case_id}")
    pairs.sort(key=lambda row: (stable_key(next(item for item in recipients if str(item["case_id"]) == row["recipient_case_id"])), row["swap_kind"]))
    diagnostics = {
        "seed": int(seed), "quantile_bins_requested": int(bins),
        "question_quantile_edges": edges["question"], "answer_quantile_edges": edges["answer"],
        "effective_question_bins": max(1, len(edges["question"]) - 1),
        "effective_answer_bins": max(1, len(edges["answer"]) - 1),
        "recipient_count": len(recipients), "donor_count": len(donors),
        "donor_reuse_counts": dict(sorted(usage.items())),
        "max_donor_reuse": max(usage.values(), default=0),
        "min_donor_reuse": min(usage.values(), default=0),
        "pair_count": len(pairs),
        "normalized_answer_match_rate": float(np.mean([bool(row["normalized_answer_match"]) for row in pairs])) if pairs else None,
        "question_token_length_delta_mean": float(np.mean([row["question_token_length_delta"] for row in pairs])) if pairs else None,
        "answer_token_length_delta_mean": float(np.mean([row["answer_token_length_delta"] for row in pairs])) if pairs else None,
        "by_condition": {},
    }
    for condition in ("I_from_I", "I_from_T", "T_from_T", "T_from_I"):
        subset = [row for row in pairs if row["condition"] == condition]
        diagnostics["by_condition"][condition] = {
            "count": len(subset),
            "normalized_answer_match_rate": float(np.mean([bool(row["normalized_answer_match"]) for row in subset])) if subset else None,
            "question_token_length_delta_mean": float(np.mean([row["question_token_length_delta"] for row in subset])) if subset else None,
            "answer_token_length_delta_mean": float(np.mean([row["answer_token_length_delta"] for row in subset])) if subset else None,
        }
    return pairs, diagnostics


__all__ = ["add_length_bins", "build_swap_pairs", "quantile_bin", "quantile_edges"]
