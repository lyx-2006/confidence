from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def add_cle_plus_1(located: dict[str, Any], tokenizer: Any, input_ids: torch.Tensor) -> dict[str, Any]:
    result = dict(located)
    index = int(located["P1_CLASS_LIST_END"]["processed_index"]) + 1
    ids = input_ids.detach().cpu().reshape(-1).tolist()
    if index >= len(ids):
        raise ValueError("CLE+1 is outside the processed prompt")
    token_id = int(ids[index])
    result["P1_CLASS_LIST_END_PLUS_1"] = {
        "processed_index": index,
        "token_id": token_id,
        "token_text": tokenizer.decode([token_id], skip_special_tokens=False,
                                       clean_up_tokenization_spaces=False),
        "definition": "P1_CLASS_LIST_END processed index + 1",
    }
    order = [int(result[name]["processed_index"]) for name in (
        "P1_PANL", "P1_PANL_PLUS_1", "P1_CLASS_LIST_END",
        "P1_CLASS_LIST_END_PLUS_1", "P1_SAC",
    )]
    if not all(left < right for left, right in zip(order, order[1:])):
        raise ValueError(f"Required causal order failed: {order}")
    return result


def trial_key(case_id: str, condition: str, window: tuple[int, int] | None = None) -> str:
    return f"{case_id}|{condition}" if window is None else f"{case_id}|{condition}|L{window[0]}-{window[1]}"


def trial_path(root: Path, case_id: str, condition: str,
               window: tuple[int, int] | None = None) -> Path:
    suffix = condition if window is None else f"{condition}__L{window[0]}-{window[1]}"
    return root / "artifacts" / "trials" / f"{case_id}__{suffix}.json"


def load_trials(root: Path) -> list[dict[str, Any]]:
    folder = root / "artifacts" / "trials"
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(folder.glob("*.json"))]

