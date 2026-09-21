from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from dp_SA.attention_block.masking import AttentionEdges

from .config import CONDITIONS


def class_margin(logits: Sequence[float], selected: int) -> float:
    values = np.asarray(logits, dtype=np.float64)
    if values.shape != (5,) or not 0 <= int(selected) < 5:
        raise ValueError("Five-class margin requires five logits and class 0..4")
    return float(values[int(selected)] - np.delete(values, int(selected)).mean())


def deterministic_argmax(logits: Sequence[float]) -> tuple[int, bool]:
    values = np.asarray(logits, dtype=np.float64)
    if values.shape != (5,) or not np.isfinite(values).all():
        raise ValueError("Expected five finite logits")
    maximum = values.max()
    winners = np.flatnonzero(values == maximum)
    return int(winners[0]), bool(len(winners) > 1)


def add_cle_plus_1(located: dict[str, Any], tokenizer: Any, input_ids: Any) -> dict[str, Any]:
    ids = input_ids.detach().cpu().reshape(-1).tolist() if hasattr(input_ids, "detach") else list(input_ids)
    index = int(located["indices"]["CLE"]) + 1
    sac = int(located["indices"]["SAC"])
    if not int(located["indices"]["CLE"]) < index < sac or index >= len(ids):
        raise ValueError("CLE+1 must be exactly one processed token after CLE and before SAC")
    token_id = int(ids[index])
    record = {
        "processed_index": index,
        "token_id": token_id,
        "token_text": tokenizer.decode([token_id], skip_special_tokens=False,
                                       clean_up_tokenization_spaces=False),
        "definition": "processed_token_immediately_after_CLE",
    }
    result = dict(located)
    result["indices"] = {**located["indices"], "CLE+1": index}
    result["positions"] = {**located["positions"], "CLE+1": record}
    return result


def edge_for_condition(condition: str, positions: dict[str, int]) -> tuple[AttentionEdges, str]:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    source = {
        "SAC_to_PANL": "PANL", "SAC_to_PANL_plus_1": "PANL+1",
        "SAC_to_CLE": "CLE", "SAC_to_CLE_plus_1": "CLE+1",
    }[condition]
    return AttentionEdges(((int(positions["SAC"]), int(positions[source])),)), source

