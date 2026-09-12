from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Sequence

import networkx as nx

from .config import SEED, WINDOWS


def _tie(seed: int, recipient: str, donor: str) -> int:
    return int(hashlib.sha256(f"{seed}|{recipient}|{donor}".encode()).hexdigest()[:8], 16)


def eligible_edge(recipient: dict[str, Any], donor: dict[str, Any]) -> bool:
    if str(recipient["phase0_raw_answer"]) != str(donor["phase0_raw_answer"]): return False
    if str(recipient["template_sha256"]) != str(donor["template_sha256"]): return False
    if any(str(recipient[field]) == str(donor[field]) for field in ("family_id", "item_id", "image_sha256")): return False
    return all(recipient["windows"][name]["token_ids"] == donor["windows"][name]["token_ids"] for name in WINDOWS)


def distance_components(recipient: dict[str, Any], donor: dict[str, Any]) -> tuple[int, int, int]:
    position = sum(abs(int(recipient["windows"][name]["processed_start"]) - int(donor["windows"][name]["processed_start"])) for name in WINDOWS)
    return position, abs(int(recipient["image_token_count"]) - int(donor["image_token_count"])), abs(int(recipient["sequence_length"]) - int(donor["sequence_length"]))


def _assign(recipients: Sequence[dict[str, Any]], donors: Sequence[dict[str, Any]], *, donor_side: str,
            seed: int) -> tuple[list[dict[str, Any]], int]:
    pool = [row for row in donors if row["sa_side"] == donor_side]
    edges = [(recipient, donor, distance_components(recipient, donor)) for recipient in recipients for donor in pool if eligible_edge(recipient, donor)]
    for recipient in recipients:
        if not any(edge[0]["case_id"] == recipient["case_id"] for edge in edges):
            raise ValueError(f"No {donor_side} token-identical donor for {recipient['case_id']}")
    max_image = max((d[1] for _, _, d in edges), default=0); max_sequence = max((d[2] for _, _, d in edges), default=0)
    count = len(recipients); sequence_base = max_sequence * count + 1; image_base = max_image * count + 1; tie_base = 2**32 * count + 1
    def cost(parts: tuple[int, int, int], recipient: str, donor: str) -> int:
        position, image, sequence = parts
        return int((((position * image_base + image) * sequence_base + sequence) * tie_base) + _tie(seed, recipient, donor))
    for capacity in range(1, count + 1):
        graph = nx.DiGraph(); graph.add_node("source", demand=-count); graph.add_node("sink", demand=count)
        for recipient in recipients:
            node = "r:" + str(recipient["case_id"]); graph.add_node(node, demand=0); graph.add_edge("source", node, capacity=1, weight=0)
        for donor in pool:
            node = "d:" + str(donor["case_id"]); graph.add_node(node, demand=0); graph.add_edge(node, "sink", capacity=capacity, weight=0)
        for recipient, donor, parts in edges:
            graph.add_edge("r:" + str(recipient["case_id"]), "d:" + str(donor["case_id"]), capacity=1,
                           weight=cost(parts, str(recipient["case_id"]), str(donor["case_id"])))
        try: flow = nx.min_cost_flow(graph)
        except nx.NetworkXUnfeasible: continue
        output = []
        by_donor = {str(row["case_id"]): row for row in pool}
        for recipient in recipients:
            rnode = "r:" + str(recipient["case_id"])
            chosen = [node[2:] for node, amount in flow[rnode].items() if amount]
            if len(chosen) != 1: raise AssertionError("Matching did not select exactly one donor")
            donor = by_donor[chosen[0]]; parts = distance_components(recipient, donor)
            output.append({
                "recipient_case_id": str(recipient["case_id"]), "recipient_side": recipient["sa_side"],
                "recipient_answer": str(recipient["phase0_raw_answer"]), "donor_case_id": str(donor["case_id"]),
                "donor_side": donor_side, "donor_clean_sa": float(donor["soft_sa_image_score"]),
                "position_distance": parts[0], "image_token_distance": parts[1], "sequence_length_distance": parts[2],
                "window_start_deltas": {name: int(donor["windows"][name]["processed_start"]) - int(recipient["windows"][name]["processed_start"]) for name in WINDOWS},
                "window_token_ids_equal": True,
            })
        return output, capacity
    raise ValueError(f"No feasible global {donor_side} donor assignment")


def match_donors(recipients: Sequence[dict[str, Any]], donors: Sequence[dict[str, Any]], *, seed: int = SEED) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs: list[dict[str, Any]] = []; capacities = {}
    for side in ("high_image", "high_text"):
        assigned, capacity = _assign(recipients, donors, donor_side=side, seed=seed); pairs.extend(assigned); capacities[side] = capacity
    pairs.sort(key=lambda row: (row["recipient_case_id"], row["donor_side"]))
    usage = Counter(row["donor_case_id"] for row in pairs)
    if len(pairs) != 2 * len(recipients): raise AssertionError("Every recipient must have two donors")
    return pairs, {"minimal_capacity_by_side": capacities, "donor_reuse_counts": dict(sorted(usage.items())),
                   "max_donor_reuse": max(usage.values(), default=0), "unique_donor_count": len(usage)}

