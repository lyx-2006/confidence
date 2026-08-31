from __future__ import annotations

import collections
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from confidence_test.answer_metrics import normalize_answer

from .config import (
    CANONICAL_ANSWERS,
    CANDIDATE_CASE_FINGERPRINT,
    CANDIDATE_IDENTITY_FINGERPRINT,
    CONSTRUCTION_MIN_FAMILIES,
    FAMILY_FINGERPRINT,
    FINAL_FOLD_COUNT,
    FOLD_COUNTS,
    FOLD_SEARCH_REPEATS,
    HISTORICAL_CAPTURE,
    SEED,
    SMOKE_CONSTRUCTION_MIN,
    TEST_MAX_FAMILIES,
    TEST_MIN_FAMILIES,
)
from .io_utils import canonical_hash, load_jsonl


def record_key(row: dict[str, Any]) -> tuple[Any, ...]:
    item = str(row["item_id"])
    return (int(item) if item.isdigit() else math.inf, item, int(row.get("prior_index", 0)), str(row.get("condition", "")), str(row.get("version", "")), str(row["case_id"]))


def sa_side(row: dict[str, Any]) -> str:
    hard = int(row["argmax_hard_class"])
    if hard in (0, 1, 2, 3): return "high_text"
    if hard == 4: return "balanced"
    if hard in (5, 6, 7, 8): return "high_image"
    raise ValueError(f"Invalid hard SA class: {hard}")


def load_clean_source() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw = load_jsonl(HISTORICAL_CAPTURE)
    canonical = []
    for row in raw:
        if row.get("status") != "completed" or not row.get("valid_class", True):
            continue
        answer = normalize_answer(str(row.get("phase0_normalized_answer") or row.get("phase0_raw_answer") or ""))
        if answer not in CANONICAL_ANSWERS:
            continue
        canonical.append({**row, "canonical_answer": answer, "sa_side": sa_side(row)})
    canonical.sort(key=record_key)
    candidates = [row for row in canonical if row["sa_side"] != "balanced"]
    return canonical, candidates


class _UnionFind:
    def __init__(self, values: Iterable[str]): self.parent = {value: value for value in values}
    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]; value = self.parent[value]
        return value
    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b: self.parent[max(a, b)] = min(a, b)


def build_families(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    items = sorted({str(row["item_id"]) for row in rows}, key=lambda x: (int(x) if x.isdigit() else math.inf, x))
    union = _UnionFind(items)
    by_hash: dict[str, set[str]] = collections.defaultdict(set)
    for row in rows: by_hash[str(row["image_sha256"])].add(str(row["item_id"]))
    shared_hashes = {}
    for image_hash, linked in by_hash.items():
        ordered = sorted(linked)
        if len(ordered) > 1: shared_hashes[image_hash] = ordered
        for item in ordered[1:]: union.union(ordered[0], item)
    components: dict[str, set[str]] = collections.defaultdict(set)
    for item in items: components[union.find(item)].add(item)
    manifests = []
    item_to_family = {}
    for component in sorted(components.values(), key=lambda values: min((int(v) if v.isdigit() else math.inf, v) for v in values)):
        component_items = sorted(component, key=lambda x: (int(x) if x.isdigit() else math.inf, x))
        hashes = sorted({str(row["image_sha256"]) for row in rows if str(row["item_id"]) in component})
        payload = {"item_ids": component_items, "image_sha256s": hashes}
        family_id = "family_" + canonical_hash(payload)[:16]
        cases = sorted(str(row["case_id"]) for row in rows if str(row["item_id"]) in component)
        manifests.append({"family_id": family_id, **payload, "case_ids": cases, "case_count": len(cases)})
        for item in component_items: item_to_family[item] = family_id
    audit = {
        "family_count": len(manifests),
        "component_size_counts": dict(collections.Counter(len(row["item_ids"]) for row in manifests)),
        "cross_item_shared_image_hashes": shared_hashes,
        "fingerprint_without_cases": canonical_hash([{k: row[k] for k in ("family_id", "item_ids", "image_sha256s")} for row in manifests]),
    }
    return manifests, item_to_family, audit


def _options(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, set[tuple[str, str]]], dict[tuple[str, str, str], list[dict[str, Any]]]]:
    options: dict[str, set[tuple[str, str]]] = collections.defaultdict(set)
    records: dict[tuple[str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        family = str(row["family_id"]); answer = str(row["canonical_answer"]); side = str(row["sa_side"])
        options[family].add((answer, side)); records[family, answer, side].append(row)
    for values in records.values(): values.sort(key=record_key)
    return options, records


def allocate_test_families(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, tuple[str, str]], dict[str, dict[str, int]]]:
    options, _records = _options(rows)
    families = sorted(options, key=lambda f: min(record_key(row) for row in rows if row["family_id"] == f))
    graph = nx.DiGraph(); source = "source"; sink = "sink"
    for family in families:
        graph.add_edge(source, "family:" + family, capacity=1)
        for answer, side in sorted(options[family]): graph.add_edge("family:" + family, f"cell:{answer}:{side}", capacity=1)
    for answer in CANONICAL_ANSWERS:
        target = 9 if answer == "blue" else TEST_MAX_FAMILIES
        text_target = 4 if answer == "blue" else (8 if answer in {"cyan", "orange", "purple", "white"} else 7)
        image_target = target - text_target
        graph.add_edge(f"cell:{answer}:high_text", "answer:" + answer, capacity=text_target)
        graph.add_edge(f"cell:{answer}:high_image", "answer:" + answer, capacity=image_target)
        graph.add_edge("answer:" + answer, sink, capacity=target)
    value, flow = nx.maximum_flow(graph, source, sink, flow_func=nx.algorithms.flow.edmonds_karp)
    if value != 174: raise ValueError(f"Test family allocation is incomplete: {value}/174")
    assigned: dict[str, tuple[str, str]] = {}
    for family in families:
        for node, amount in flow["family:" + family].items():
            if amount:
                _prefix, answer, side = node.split(":")
                assigned[family] = (answer, side)
    distribution = {}
    for answer in CANONICAL_ANSWERS:
        sides = collections.Counter(side for _family, (candidate, side) in assigned.items() if candidate == answer)
        distribution[answer] = {"high_text": sides["high_text"], "high_image": sides["high_image"], "total": sum(sides.values())}
    return assigned, distribution


_EASY_TARGETS = {"black": 8, "blue": 7, "brown": 8, "cyan": 7, "gray": 7, "green": 9, "orange": 5, "pink": 7, "purple": 7, "red": 7, "white": 8, "yellow": 7}


def select_test_cases(rows: Sequence[dict[str, Any]], assigned: dict[str, tuple[str, str]]) -> list[dict[str, Any]]:
    _options_by_family, records = _options(rows)
    selected = []
    for answer in CANONICAL_ANSWERS:
        families = sorted([family for family, value in assigned.items() if value[0] == answer])
        forced_easy, forced_hard, flexible = [], [], []
        for family in families:
            side = assigned[family][1]; choices = records[family, answer, side]
            conditions = {str(row["condition"]) for row in choices}
            if conditions == {"conflict_easy"}: forced_easy.append(family)
            elif conditions == {"conflict_hard"}: forced_hard.append(family)
            else: flexible.append(family)
        need_easy = _EASY_TARGETS[answer] - len(forced_easy)
        if not 0 <= need_easy <= len(flexible): raise ValueError(f"Condition target is infeasible for {answer}")
        flexible.sort(key=lambda family: hashlib.sha256(f"{SEED}|condition|{answer}|{family}".encode()).hexdigest())
        easy_flexible = set(flexible[:need_easy])
        for family in families:
            side = assigned[family][1]
            desired = "conflict_easy" if family in forced_easy or family in easy_flexible else "conflict_hard"
            choices = [row for row in records[family, answer, side] if row["condition"] == desired]
            choices.sort(key=lambda row: (hashlib.sha256(f"{SEED}|case|{row['case_id']}".encode()).hexdigest(), row["case_id"]))
            chosen = dict(choices[0])
            chosen.update({"test_answer": answer, "test_side": side, "test_status": "exploratory_sparse" if answer == "blue" else "confirmatory"})
            selected.append(chosen)
    selected.sort(key=lambda row: (CANONICAL_ANSWERS.index(row["test_answer"]), row["family_id"]))
    return selected


def _cell_families(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, str], set[str]]:
    output = {(answer, side): set() for answer in CANONICAL_ANSWERS for side in ("high_text", "high_image")}
    for row in rows: output[str(row["canonical_answer"]), str(row["sa_side"])].add(str(row["family_id"]))
    return output


def _fold_metrics(folds: Sequence[set[str]], assigned: dict[str, tuple[str, str]], cells: dict[tuple[str, str], set[str]], threshold: int) -> dict[str, Any]:
    eligible_by_fold = []
    for heldout in folds:
        eligible_by_fold.append({answer for answer in CANONICAL_ANSWERS if len(cells[answer, "high_text"] - heldout) >= threshold and len(cells[answer, "high_image"] - heldout) >= threshold})
    test_sizes = [len(heldout & assigned.keys()) for heldout in folds]
    loao = [len(eligible_by_fold[index] - {assigned[family][0]}) for index, heldout in enumerate(folds) for family in heldout if family in assigned]
    return {
        "fold_sizes": [len(values) for values in folds], "test_sizes": test_sizes,
        "eligible_counts": [len(values) for values in eligible_by_fold],
        "eligible_by_fold": [sorted(values) for values in eligible_by_fold],
        "eligible_total": sum(map(len, eligible_by_fold)),
        "min_loao_eligible": min(loao) if loao else 0,
        "loao_failure_count": sum(value < 3 for value in loao),
    }


def choose_crossfit_folds(rows: Sequence[dict[str, Any]], assigned: dict[str, tuple[str, str]], *, repeats: int = FOLD_SEARCH_REPEATS) -> tuple[dict[int, list[set[str]]], dict[int, dict[str, Any]]]:
    families = sorted({str(row["family_id"]) for row in rows}, key=lambda family: min(record_key(row) for row in rows if row["family_id"] == family))
    cells = _cell_families(rows); rng = random.Random(SEED); chosen = {}; diagnostics = {}
    for fold_count in FOLD_COUNTS:
        best = None
        for _ in range(repeats):
            order = families.copy(); rng.shuffle(order)
            folds = [set(order[index::fold_count]) for index in range(fold_count)]
            metrics = _fold_metrics(folds, assigned, cells, CONSTRUCTION_MIN_FAMILIES)
            answer_imbalance = 0
            for answer in CANONICAL_ANSWERS:
                counts = [sum(1 for family in heldout if assigned.get(family, (None, None))[0] == answer) for heldout in folds]
                answer_imbalance += max(counts) - min(counts)
            score = (min(metrics["eligible_counts"]), metrics["eligible_total"], metrics["min_loao_eligible"], -(max(metrics["test_sizes"]) - min(metrics["test_sizes"])), -answer_imbalance)
            if best is None or score > best[0]: best = (score, folds, metrics)
        assert best is not None
        chosen[fold_count] = best[1]; diagnostics[fold_count] = {**best[2], "score": list(best[0])}
    return chosen, diagnostics


def _optimized_fixed_holdout(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    options, _records = _options(rows); families = sorted(options); cells = _cell_families(rows)
    edges = [(family, answer, side) for family in families for answer, side in sorted(options[family])]
    edge_count = len(edges); answer_index = {answer: index for index, answer in enumerate(CANONICAL_ANSWERS)}; total_vars = edge_count + len(CANONICAL_ANSWERS)
    objective = np.zeros(total_vars); objective[:edge_count] = -1
    for answer in CANONICAL_ANSWERS: objective[edge_count + answer_index[answer]] = -10000
    vectors = []; lower = []; upper = []
    for family in families:
        vector = np.zeros(total_vars); vector[[index for index, edge in enumerate(edges) if edge[0] == family]] = 1
        vectors.append(vector); lower.append(-np.inf); upper.append(1)
    for answer in CANONICAL_ANSWERS:
        vector = np.zeros(total_vars); vector[[index for index, edge in enumerate(edges) if edge[1] == answer]] = 1
        vectors.append(vector); lower.append(9 if answer == "blue" else TEST_MIN_FAMILIES); upper.append(9 if answer == "blue" else TEST_MAX_FAMILIES)
    for answer in CANONICAL_ANSWERS:
        for side in ("high_text", "high_image"):
            members = cells[answer, side]
            if len(members) < CONSTRUCTION_MIN_FAMILIES:
                vector = np.zeros(total_vars); vector[edge_count + answer_index[answer]] = 1
                vectors.append(vector); lower.append(0); upper.append(0); continue
            vector = np.zeros(total_vars)
            for index, edge in enumerate(edges):
                if edge[0] in members: vector[index] = 1
            vector[edge_count + answer_index[answer]] = len(members)
            vectors.append(vector); lower.append(-np.inf); upper.append(2 * len(members) - CONSTRUCTION_MIN_FAMILIES)
    result = milp(objective, integrality=np.ones(total_vars), bounds=Bounds(np.zeros(total_vars), np.ones(total_vars)), constraints=LinearConstraint(np.stack(vectors), np.asarray(lower), np.asarray(upper)))
    if not result.success: raise RuntimeError("Fixed holdout comparison optimizer failed")
    heldout = {edges[index][0] for index in range(edge_count) if result.x[index] > 0.5}
    eligible = [answer for answer in CANONICAL_ANSWERS if result.x[edge_count + answer_index[answer]] > 0.5]
    return {"scheme": "grouped_fixed_holdout", "test_family_count": len(heldout), "construction_family_count": len(families) - len(heldout), "eligible_answers": eligible, "eligible_count": len(eligible), "min_loao_eligible": max(0, len(eligible) - 1)}


def construction_distribution(rows: Sequence[dict[str, Any]], folds: Sequence[set[str]]) -> list[dict[str, Any]]:
    cells = _cell_families(rows); output = []
    for fold_id, heldout in enumerate(folds):
        eligible = {answer for answer in CANONICAL_ANSWERS if len(cells[answer, "high_text"] - heldout) >= CONSTRUCTION_MIN_FAMILIES and len(cells[answer, "high_image"] - heldout) >= CONSTRUCTION_MIN_FAMILIES}
        for answer in CANONICAL_ANSWERS:
            text_count = len(cells[answer, "high_text"] - heldout); image_count = len(cells[answer, "high_image"] - heldout)
            output.append({"fold": fold_id, "answer": answer, "construction_high_text_family_count": text_count, "construction_high_image_family_count": image_count, "eligible_for_direction": answer in eligible, "eligible_answer_count": len(eligible)})
    return output


def build_formal_design() -> dict[str, Any]:
    canonical, candidates = load_clean_source()
    families, item_to_family, family_audit = build_families(canonical)
    for row in candidates: row["family_id"] = item_to_family[str(row["item_id"])]
    case_fp = canonical_hash([row["case_id"] for row in candidates])
    identity_fp = canonical_hash([{"case_id": row["case_id"], "answer": row["canonical_answer"], "side": row["sa_side"], "image_sha256": row["image_sha256"]} for row in candidates])
    if case_fp != CANDIDATE_CASE_FINGERPRINT or identity_fp != CANDIDATE_IDENTITY_FINGERPRINT: raise ValueError("Full candidate source fingerprint changed")
    if family_audit["fingerprint_without_cases"] != FAMILY_FINGERPRINT: raise ValueError("Family graph fingerprint changed")
    assigned, test_distribution = allocate_test_families(candidates)
    test = select_test_cases(candidates, assigned)
    fold_candidates, candidate_metrics = choose_crossfit_folds(candidates, assigned)
    folds = fold_candidates[FINAL_FOLD_COUNT]
    family_to_fold = {family: fold for fold, heldout in enumerate(folds) for family in heldout}
    for row in test: row["fold"] = family_to_fold[row["family_id"]]
    fold_rows = [{"family_id": family, "fold": family_to_fold[family], "is_test_family": family in assigned} for family in sorted(family_to_fold)]
    construction = construction_distribution(candidates, folds)
    cells_manifest = []
    grouped: dict[tuple[str, str, str], list[str]] = collections.defaultdict(list)
    for row in candidates: grouped[row["family_id"], row["canonical_answer"], row["sa_side"]].append(row["case_id"])
    for fold, heldout in enumerate(folds):
        for (family, answer, side), cases in sorted(grouped.items()):
            if family not in heldout: cells_manifest.append({"fold": fold, "family_id": family, "answer": answer, "sa_side": side, "case_ids": sorted(cases), "record_count": len(cases)})
    comparison = {"grouped_fixed_holdout": _optimized_fixed_holdout(candidates)}
    for count in FOLD_COUNTS: comparison[f"grouped_crossfit_{count}"] = {"scheme": "grouped_crossfit", "fold_count": count, "test_family_count": len(test), **candidate_metrics[count]}
    return {"canonical": canonical, "candidates": candidates, "families": families, "family_audit": family_audit, "test": test, "test_distribution": test_distribution, "folds": folds, "fold_assignments": fold_rows, "construction_distribution": construction, "construction_cells": cells_manifest, "comparison": comparison}


def build_smoke_design() -> dict[str, Any]:
    canonical, candidates = load_clean_source(); families, item_to_family, family_audit = build_families(canonical)
    for row in candidates: row["family_id"] = item_to_family[str(row["item_id"])]
    answers = ("brown", "cyan", "orange", "yellow"); used = set(); construction_rows = []
    for answer in answers:
        for side in ("high_text", "high_image"):
            pool = [row for row in candidates if row["canonical_answer"] == answer and row["sa_side"] == side and row["family_id"] not in used]
            pool.sort(key=lambda row: (hashlib.sha256(f"{SEED}|smoke|{row['case_id']}".encode()).hexdigest(), record_key(row)))
            chosen_families = []
            for row in pool:
                if row["family_id"] in chosen_families: continue
                construction_rows.append(dict(row)); chosen_families.append(row["family_id"]); used.add(row["family_id"])
                if len(chosen_families) == SMOKE_CONSTRUCTION_MIN: break
            if len(chosen_families) != SMOKE_CONSTRUCTION_MIN: raise ValueError("Smoke construction is incomplete")
    test = []
    for index, answer in enumerate(answers):
        side = "high_text" if index % 2 == 0 else "high_image"
        pool = [row for row in candidates if row["canonical_answer"] == answer and row["sa_side"] == side and row["family_id"] not in used]
        pool.sort(key=lambda row: (hashlib.sha256(f"{SEED}|smoke_test|{row['case_id']}".encode()).hexdigest(), record_key(row)))
        chosen = dict(pool[0]); used.add(chosen["family_id"]); chosen.update({"fold": 0, "test_answer": answer, "test_side": side, "test_status": "smoke_only"}); test.append(chosen)
    selected = construction_rows + test
    heldout = {row["family_id"] for row in test}; construction_families = {row["family_id"] for row in construction_rows}
    cells = []
    for row in construction_rows: cells.append({"fold": 0, "family_id": row["family_id"], "answer": row["canonical_answer"], "sa_side": row["sa_side"], "case_ids": [row["case_id"]], "record_count": 1})
    distribution = []
    for answer in CANONICAL_ANSWERS:
        tc = len({row["family_id"] for row in construction_rows if row["canonical_answer"] == answer and row["sa_side"] == "high_text"})
        ic = len({row["family_id"] for row in construction_rows if row["canonical_answer"] == answer and row["sa_side"] == "high_image"})
        distribution.append({"fold": 0, "answer": answer, "construction_high_text_family_count": tc, "construction_high_image_family_count": ic, "eligible_for_direction": tc >= 2 and ic >= 2, "eligible_answer_count": 4})
    smoke_families = [row for row in families if row["family_id"] in construction_families | heldout]
    return {"canonical": canonical, "candidates": selected, "families": smoke_families, "family_audit": {**family_audit, "smoke_only": True}, "test": test, "test_distribution": {}, "folds": [heldout], "fold_assignments": [{"family_id": row["family_id"], "fold": 0 if row["family_id"] in heldout else -1, "is_test_family": row["family_id"] in heldout} for row in smoke_families], "construction_distribution": distribution, "construction_cells": cells, "comparison": {"smoke_only": True}}


def leakage_audit(design: dict[str, Any], *, smoke: bool) -> dict[str, Any]:
    candidates = design["candidates"]; by_family: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in candidates: by_family[row["family_id"]].append(row)
    folds = []
    for fold_id, heldout in enumerate(design["folds"]):
        construction = set(by_family) - set(heldout)
        def values(families: set[str], field: str) -> set[str]: return {str(row[field]) for family in families for row in by_family.get(family, [])}
        fold = {
            "fold": fold_id, "heldout_family_count": len(heldout), "construction_family_count": len(construction),
            "family_leakage_count": len(set(heldout) & construction),
            "item_leakage_count": len(values(set(heldout), "item_id") & values(construction, "item_id")),
            "image_hash_leakage_count": len(values(set(heldout), "image_sha256") & values(construction, "image_sha256")),
            "case_leakage_count": len(values(set(heldout), "case_id") & values(construction, "case_id")),
        }
        folds.append(fold)
    test_families = [row["family_id"] for row in design["test"]]
    threshold = SMOKE_CONSTRUCTION_MIN if smoke else CONSTRUCTION_MIN_FAMILIES
    loao_min = min((row["eligible_answer_count"] - int(bool(row["eligible_for_direction"])) for row in design["construction_distribution"]), default=0)
    passed = all(not any(row[key] for key in ("family_leakage_count", "item_leakage_count", "image_hash_leakage_count", "case_leakage_count")) for row in folds) and len(test_families) == len(set(test_families)) and loao_min >= 3
    if not smoke:
        passed = passed and len(design["candidates"]) == 1625 and len(design["families"]) == 178 and len(design["test"]) == 174 and min(row["eligible_answer_count"] for row in design["construction_distribution"]) >= 8
    return {"status": "passed" if passed else "failed", "smoke_only": smoke, "folds": folds, "test_family_count": len(test_families), "test_family_unique_count": len(set(test_families)), "loao_minimum": loao_min, "construction_threshold": threshold}
