from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from confidence_test.answer_metrics import normalize_answer

from .config import CELLS, COLOR_HEX, DELTA_E76_THRESHOLD, DIFFICULTIES, ORIGINS


def stable_seed(seed: int, *parts: Any) -> int:
    raw = ":".join([str(int(seed)), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def stable_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    item = str(row.get("item_id", ""))
    item_key = (0, int(item)) if item.isdigit() else (1, item)
    return (*item_key, int(row.get("prior_index", 0)), str(row.get("condition", "")), str(row.get("case_id", "")))


def answer_type(value: Any) -> str:
    text = normalize_answer(value)
    if not text:
        return "empty"
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:%|\s*percent)?", text):
        return "numeric"
    if re.search(r"\s", text):
        return "phrase"
    return "single_word"


def _srgb_to_lab(hex_value: str) -> np.ndarray:
    rgb = np.asarray([int(hex_value[i : i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=float)
    rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    matrix = np.asarray(
        [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]],
        dtype=float,
    )
    x, y, z = matrix @ rgb
    xyz = np.asarray([x / 0.95047, y, z / 1.08883], dtype=float)
    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(xyz > epsilon, xyz ** (1.0 / 3.0), (kappa * xyz + 16.0) / 116.0)
    return np.asarray([116.0 * f[1] - 16.0, 500.0 * (f[0] - f[1]), 200.0 * (f[1] - f[2])])


_COLOR_LAB = {name: _srgb_to_lab(value) for name, value in COLOR_HEX.items()}


def delta_e76(left: Any, right: Any) -> float | None:
    a, b = normalize_answer(left), normalize_answer(right)
    if a not in _COLOR_LAB or b not in _COLOR_LAB:
        return None
    return float(np.linalg.norm(_COLOR_LAB[a] - _COLOR_LAB[b]))


def _contains_answer(text: Any, answer: Any) -> bool:
    value, candidate = normalize_answer(text), normalize_answer(answer)
    if not value or not candidate:
        return False
    pattern = r"(?<!\w)" + re.escape(candidate).replace(r"\ ", r"\s+") + r"(?!\w)"
    return bool(re.search(pattern, value, flags=re.IGNORECASE))


def question_stem(question: str) -> str:
    return re.split(r"choose\s+from\s*:", str(question), maxsplit=1, flags=re.IGNORECASE)[0]


@dataclass(frozen=True)
class EligibleCase:
    row: dict[str, Any]
    origin: str
    difficulty: str
    text_answer: str
    image_answer: str
    normalized_answer: str

    @property
    def cell(self) -> tuple[str, str]:
        return self.origin, self.difficulty


def classify_origin(row: Mapping[str, Any], text_answer: Any, image_answer: Any) -> str | None:
    normalized = normalize_answer(row.get("phase0_raw_answer"))
    text, image = normalize_answer(text_answer), normalize_answer(image_answer)
    if not normalized or not text or not image or text == image:
        return None
    text_match, image_match = normalized == text, normalized == image
    if text_match == image_match:
        return None
    return "text" if text_match else "image"


def _row_reasons(row: Mapping[str, Any], case: Any, split: Mapping[str, int], tokenizer: Any | None) -> list[str]:
    reasons: list[str] = []
    if row.get("status") != "completed":
        reasons.append("capture_not_completed")
    if str(row.get("item_id")) not in split:
        reasons.append("missing_split_assignment")
    text_answer = normalize_answer(getattr(case, "text_answer", None))
    image_answer = normalize_answer(getattr(case, "conflict_answer", None))
    normalized = normalize_answer(row.get("phase0_raw_answer"))
    if not normalized:
        reasons.append("phase0_answer_empty_or_parse_failed")
    if isinstance(row.get("phase0_raw_answer"), str) and any(c in row["phase0_raw_answer"] for c in "\r\n"):
        reasons.append("phase0_answer_multiline")
    if not text_answer or not image_answer or text_answer == image_answer:
        reasons.append("candidate_answers_invalid_or_equal")
    if normalized and text_answer and image_answer and ((normalized == text_answer) == (normalized == image_answer)):
        reasons.append("phase0_matches_both_or_neither")
    if tokenizer is not None and normalized:
        try:
            token_ids = tokenizer.encode(str(row.get("phase0_raw_answer")), add_special_tokens=False)
            if not token_ids:
                reasons.append("phase0_answer_unstable_tokenization")
        except Exception:
            reasons.append("phase0_answer_unstable_tokenization")
    if not str(row.get("question", "")).strip() or not str(row.get("text_clue", "")).strip():
        reasons.append("missing_question_or_text_clue")
    return reasons


def eligible_cases(
    rows: Sequence[Mapping[str, Any]],
    cases_by_key: Mapping[tuple[str, int], Any],
    split: Mapping[str, int],
    *,
    tokenizer: Any | None = None,
) -> tuple[list[EligibleCase], list[dict[str, Any]]]:
    eligible: list[EligibleCase] = []
    excluded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        row = dict(raw)
        case_id = str(row.get("case_id", ""))
        reasons = _row_reasons(row, cases_by_key.get((str(row.get("item_id")), int(row.get("prior_index", -1)))), split, tokenizer) if cases_by_key.get((str(row.get("item_id")), int(row.get("prior_index", -1)))) is not None else ["dataset_case_missing"]
        case = cases_by_key.get((str(row.get("item_id")), int(row.get("prior_index", -1))))
        if case_id in seen:
            reasons.append("duplicate_case_id")
        seen.add(case_id)
        if not reasons and case is not None:
            text_answer = str(case.text_answer)
            image_answer = str(case.conflict_answer)
            origin = classify_origin(row, text_answer, image_answer)
            if origin is None:
                reasons.append("phase0_matches_both_or_neither")
            else:
                eligible.append(
                    EligibleCase(
                        row=row,
                        origin=origin,
                        difficulty=str(row.get("condition", "")).removeprefix("conflict_"),
                        text_answer=text_answer,
                        image_answer=image_answer,
                        normalized_answer=str(normalize_answer(row["phase0_raw_answer"])),
                    )
                )
        if reasons:
            excluded.append({"case_id": case_id, "item_id": row.get("item_id"), "prior_index": row.get("prior_index"), "condition": row.get("condition"), "reasons": sorted(set(reasons))})
    return eligible, excluded


def _case_priority(item: EligibleCase, seed: int) -> tuple[int, tuple[Any, ...]]:
    return stable_seed(seed, "case", item.row.get("case_id")), stable_key(item.row)


def _maximum_slot_matching(edges: Mapping[str, set[tuple[str, str]]], quota: int, seed: int) -> dict[tuple[str, str, int], str]:
    slots = [(cell[0], cell[1], index) for cell in CELLS for index in range(quota)]
    candidates = {slot: sorted([item for item, cells in edges.items() if (slot[0], slot[1]) in cells], key=lambda item: (stable_seed(seed, "item", item), item)) for slot in slots}
    slots.sort(key=lambda slot: (len(candidates[slot]), stable_seed(seed, "slot", *slot)))
    match_item: dict[str, tuple[str, str, int]] = {}

    def visit(slot: tuple[str, str, int], seen: set[str]) -> bool:
        for item in candidates[slot]:
            if item in seen:
                continue
            seen.add(item)
            previous = match_item.get(item)
            if previous is None or visit(previous, seen):
                match_item[item] = slot
                return True
        return False

    for slot in slots:
        if not visit(slot, set()):
            raise ValueError(f"Unable to satisfy item-disjoint recipient quotas; matched {len(match_item)}/{len(slots)} slots")
    return {slot: item for item, slot in match_item.items()}


def select_recipients(
    eligible: Sequence[EligibleCase],
    *,
    seed: int = 42,
    per_cell: int = 25,
    smoke: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if per_cell < 1:
        raise ValueError("per_cell must be positive")
    by_item_cell: dict[str, dict[tuple[str, str], EligibleCase]] = defaultdict(dict)
    for item in eligible:
        key = str(item.row["item_id"])
        current = by_item_cell[key].get(item.cell)
        if current is None or _case_priority(item, seed) < _case_priority(current, seed):
            by_item_cell[key][item.cell] = item
    edges = {item: set(cells) for item, cells in by_item_cell.items()}
    selected_slots = _maximum_slot_matching(edges, per_cell, seed)
    selected: list[dict[str, Any]] = []
    for (origin, difficulty, _index), item_id in sorted(selected_slots.items(), key=lambda pair: (pair[0][0], pair[0][1], pair[0][2])):
        item = by_item_cell[item_id][(origin, difficulty)]
        row = dict(item.row)
        row.update({
            "origin": origin,
            "difficulty": difficulty,
            "text_answer": item.text_answer,
            "image_answer": item.image_answer,
            "normalized_answer": item.normalized_answer,
            "forced_direction": 1 if origin == "text" else -1,
            "split_fold": int(row.get("split_fold", -1)),
            "selection_cell_index": int(_index),
        })
        row["forced_opposite_answer"] = item.image_answer if origin == "text" else item.text_answer
        selected.append(row)
    if smoke:
        selected = [row for row in selected if int(row["selection_cell_index"]) == 0]
        # One easy and one hard per origin is the deterministic 2+2 smoke.
        selected = sorted(selected, key=stable_key)
    counts = Counter((str(row["origin"]), str(row["difficulty"])) for row in selected)
    expected = 4 if smoke else 4 * per_cell
    if len(selected) != expected or len({str(row["item_id"]) for row in selected}) != len(selected):
        raise ValueError(f"recipient selection is not item-disjoint/balanced: {len(selected)} rows")
    summary = {
        "seed": int(seed), "smoke": bool(smoke), "requested_per_cell": int(per_cell),
        "selected_count": len(selected), "unique_item_count": len({str(row["item_id"]) for row in selected}),
        "cell_counts": {f"{origin}|{difficulty}": int(counts[(origin, difficulty)]) for origin, difficulty in CELLS},
        "selection_uses_clean_sa": False, "algorithm": "seeded_item_disjoint_slot_matching",
    }
    return sorted(selected, key=stable_key), summary


def _token_ids(tokenizer: Any, answer: str) -> list[int]:
    ids = tokenizer.encode(str(answer), add_special_tokens=False)
    if not ids:
        raise ValueError(f"Answer has no tokenizer tokens: {answer!r}")
    return [int(value) for value in ids]


def build_unrelated_manifest(
    recipients: Sequence[Mapping[str, Any]],
    canonical_pool: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    seed: int = 42,
    delta_threshold: float = DELTA_E76_THRESHOLD,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pool = [dict(row) for row in canonical_pool]
    usage_item: Counter[str] = Counter()
    usage_entry: Counter[str] = Counter()
    output: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {"seed": int(seed), "delta_e76_threshold": float(delta_threshold), "recipient_count": len(recipients), "candidate_counts": {}, "unmatched": []}
    for recipient in sorted(recipients, key=stable_key):
        rec_id, rec_item = str(recipient["case_id"]), str(recipient["item_id"])
        original = normalize_answer(recipient.get("phase0_raw_answer"))
        text_answer = normalize_answer(recipient.get("text_answer"))
        image_answer = normalize_answer(recipient.get("image_answer"))
        opposite = normalize_answer(recipient.get("forced_opposite_answer"))
        opposite_length = len(_token_ids(tokenizer, str(recipient["forced_opposite_answer"])))
        stem = question_stem(str(recipient.get("question", "")))
        clue = str(recipient.get("text_clue", ""))
        rec_type = answer_type(original)
        candidates: list[tuple[tuple[Any, ...], dict[str, Any], list[int], float | None, float | None]] = []
        for donor in pool:
            donor_item, donor_answer = str(donor["item_id"]), str(donor["answer"])
            donor_norm = normalize_answer(donor_answer)
            if donor_item == rec_item or donor_norm in {original, text_answer, image_answer}:
                continue
            if answer_type(donor_answer) != rec_type:
                continue
            # The option list is intentionally ignored; stem and clue remain
            # leakage checks.  This is necessary because every question lists
            # the complete 12-color answer set.
            if _contains_answer(stem, donor_answer) or _contains_answer(clue, donor_answer):
                continue
            d_text, d_image = delta_e76(donor_answer, text_answer), delta_e76(donor_answer, image_answer)
            if d_text is not None and (d_text < delta_threshold or d_image is None or d_image < delta_threshold):
                continue
            try:
                ids = _token_ids(tokenizer, donor_answer)
            except Exception:
                continue
            token_delta = abs(len(ids) - opposite_length)
            char_delta = abs(len(donor_answer) - len(str(recipient["forced_opposite_answer"])))
            entry_id = str(donor.get("entry_id", f"{donor_item}|{donor.get('role', '')}|{donor_norm}"))
            score = (int(len(ids) != opposite_length), token_delta, char_delta, int(usage_item[donor_item]), int(usage_entry[entry_id]), stable_seed(seed, "donor", rec_id, entry_id))
            candidates.append((score, donor, ids, d_text, d_image))
        diagnostics["candidate_counts"][rec_id] = len(candidates)
        if not candidates:
            diagnostics["unmatched"].append(rec_id)
            raise ValueError(f"No unrelated donor for {rec_id}")
        _score, donor, ids, d_text, d_image = min(candidates, key=lambda value: value[0])
        entry_id = str(donor.get("entry_id", f"{donor['item_id']}|{donor.get('role', '')}|{normalize_answer(donor['answer'])}"))
        usage_item[str(donor["item_id"])] += 1
        usage_entry[entry_id] += 1
        output.append({
            "recipient_case_id": rec_id, "recipient_item_id": rec_item,
            "donor_case_id": donor.get("case_id"), "donor_item_id": str(donor["item_id"]),
            "donor_role": donor.get("role"), "donor_entry_id": entry_id,
            "forced_answer": str(donor["answer"]), "normalized_answer": normalize_answer(donor["answer"]),
            "answer_type": rec_type, "answer_token_ids": ids, "answer_token_length": len(ids),
            "opposite_token_length": opposite_length, "answer_token_length_equal": len(ids) == opposite_length,
            "answer_token_length_delta": abs(len(ids) - opposite_length),
            "character_length_delta": char_delta, "delta_e76_to_text": d_text, "delta_e76_to_image": d_image,
            "donor_item_reuse_index": int(usage_item[str(donor["item_id"])]),
            "donor_entry_reuse_index": int(usage_entry[entry_id]),
        })
    diagnostics.update({
        "exact_token_length_match_rate": float(np.mean([r["answer_token_length_equal"] for r in output])) if output else None,
        "answer_type_match_rate": 1.0 if output else None,
        "answer_token_length_delta_mean": float(np.mean([r["answer_token_length_delta"] for r in output])) if output else None,
        "character_length_delta_mean": float(np.mean([r["character_length_delta"] for r in output])) if output else None,
        "donor_item_reuse": dict(sorted(usage_item.items())),
        "donor_entry_reuse": dict(sorted(usage_entry.items())),
        "max_donor_item_reuse": max(usage_item.values(), default=0),
        "min_donor_item_reuse": min(usage_item.values(), default=0),
    })
    return sorted(output, key=lambda row: str(row["recipient_case_id"])), diagnostics


def unrelated_candidate_count(
    recipient: Mapping[str, Any], canonical_pool: Sequence[Mapping[str, Any]], tokenizer: Any,
    *, delta_threshold: float = DELTA_E76_THRESHOLD, token_lengths: Mapping[str, int] | None = None,
) -> int:
    """Count donors before reuse balancing, for recipient preflight filtering."""
    original = normalize_answer(recipient.get("phase0_raw_answer"))
    text_answer = normalize_answer(recipient.get("text_answer"))
    image_answer = normalize_answer(recipient.get("image_answer"))
    opposite = normalize_answer(recipient.get("forced_opposite_answer"))
    if not original or not text_answer or not image_answer or not opposite:
        return 0
    stem, clue, rec_item, rec_type = question_stem(str(recipient.get("question", ""))), str(recipient.get("text_clue", "")), str(recipient["item_id"]), answer_type(original)
    count = 0
    for donor in canonical_pool:
        donor_norm = normalize_answer(donor.get("answer"))
        if str(donor.get("item_id")) == rec_item or donor_norm in {original, text_answer, image_answer} or answer_type(donor.get("answer")) != rec_type:
            continue
        if _contains_answer(stem, donor.get("answer")) or _contains_answer(clue, donor.get("answer")):
            continue
        d_text, d_image = delta_e76(donor.get("answer"), text_answer), delta_e76(donor.get("answer"), image_answer)
        if d_text is not None and (d_text < delta_threshold or d_image is None or d_image < delta_threshold):
            continue
        try:
            # A same-length donor is preferred by build_unrelated_manifest,
            # but a valid different-length donor still satisfies the design.
            # Preflight therefore counts every finite-token candidate rather
            # than silently treating length mismatch as infeasibility.
            _donor_length = int(token_lengths[str(normalize_answer(donor["answer"]))]) if token_lengths is not None else len(_token_ids(tokenizer, str(donor["answer"])))
            count += 1
        except Exception:
            continue
    return count


def canonical_answer_pool(cases: Iterable[Any]) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for case in cases:
        for role, answer in (("text", case.text_answer), ("image", case.conflict_answer)):
            normalized = normalize_answer(answer)
            key = (str(case.item_id), role, str(normalized))
            if not normalized or key in seen:
                continue
            seen.add(key)
            pool.append({"entry_id": f"{case.item_id}|{role}|{normalized}", "case_id": None, "item_id": str(case.item_id), "role": role, "answer": str(answer)})
    return sorted(pool, key=lambda row: ((0, int(row["item_id"])) if str(row["item_id"]).isdigit() else (1, str(row["item_id"])), row["role"], row["answer"]))


__all__ = [
    "EligibleCase", "answer_type", "canonical_answer_pool", "classify_origin", "build_unrelated_manifest",
    "delta_e76", "eligible_cases", "question_stem", "select_recipients", "stable_key", "stable_seed", "unrelated_candidate_count",
]
