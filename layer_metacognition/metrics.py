"""Numerically stable scalar metrics for layer trajectories."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as functional


def entropy_from_probabilities(probabilities: torch.Tensor) -> float:
    values = probabilities.float()
    positive = values[values > 0]
    return float(-(positive * positive.log()).sum().item())


def top1_top2_probability_margin(probabilities: torch.Tensor) -> float:
    if probabilities.numel() < 2:
        return 1.0
    top = torch.topk(probabilities.float(), k=2).values
    return float((top[0] - top[1]).item())


def panl_layer_statistics(
    hidden_by_layer: dict[int, torch.Tensor],
    selected_layers: list[int],
    eps: float = 1e-12,
) -> list[dict[str, Any]]:
    if not hidden_by_layer:
        return []
    final_layer = max(hidden_by_layer)
    final_hidden = hidden_by_layer[final_layer].detach().float().cpu()
    records: list[dict[str, Any]] = []
    for layer_index in selected_layers:
        hidden = hidden_by_layer[layer_index].detach().float().cpu()
        l2 = torch.linalg.vector_norm(hidden)
        previous = hidden_by_layer.get(layer_index - 1)
        if previous is None:
            cosine_previous = None
            delta_l2 = None
            relative_delta = None
        else:
            previous_float = previous.detach().float().cpu()
            cosine_previous = float(functional.cosine_similarity(hidden, previous_float, dim=0).item())
            delta = torch.linalg.vector_norm(hidden - previous_float)
            delta_l2 = float(delta.item())
            relative_delta = float((delta / (torch.linalg.vector_norm(previous_float) + eps)).item())
        records.append(
            {
                "layer_index": layer_index,
                "l2_norm": float(l2.item()),
                "rms": float(torch.sqrt(torch.mean(hidden.square())).item()),
                "cosine_with_previous_layer": cosine_previous,
                "cosine_with_final_layer": float(
                    functional.cosine_similarity(hidden, final_hidden, dim=0).item()
                ),
                "delta_l2_from_previous_layer": delta_l2,
                "relative_delta_l2": relative_delta,
            }
        )
    return records


def ensure_finite_json_number(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} is not finite: {value}")
    return float(value)
