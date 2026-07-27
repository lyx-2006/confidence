"""Per-layer, per-head attention sink extraction without tensor retention."""

from __future__ import annotations

from typing import Any

import torch


def compute_attention_sink(
    attention: torch.Tensor,
    *,
    query_span: list[int] | tuple[int, int],
    source_span: list[int] | tuple[int, int],
    expected_heads: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if attention.ndim != 4 or int(attention.shape[0]) != 1:
        raise ValueError(f"Attention must have shape [1, heads, query, key], got {attention.shape}")
    query_start, query_end = map(int, query_span)
    source_start, source_end = map(int, source_span)
    sequence_query = int(attention.shape[2])
    sequence_key = int(attention.shape[3])
    if not (0 <= query_start < query_end <= sequence_query):
        raise ValueError(f"Invalid query span [{query_start}, {query_end})")
    if not (0 <= source_start < source_end <= sequence_key):
        raise ValueError(f"Invalid source span [{source_start}, {source_end})")
    selected = attention[
        0,
        :,
        query_start:query_end,
        source_start:source_end,
    ].float()
    heads = int(attention.shape[1])
    if expected_heads is not None and heads != expected_heads:
        raise ValueError(f"Attention head mismatch: tensor={heads}, config={expected_heads}")
    expected_shape = (
        heads,
        query_end - query_start,
        source_end - source_start,
    )
    if selected.ndim != 3 or tuple(selected.shape) != expected_shape:
        raise RuntimeError(
            f"Attention slice shape mismatch: got={tuple(selected.shape)}, expected={expected_shape}"
        )
    if not torch.isfinite(selected).all():
        raise ValueError("Attention slice contains non-finite values")
    if bool((selected < 0).any()):
        raise ValueError("Attention slice contains negative values")
    sink = selected.mean(dim=(-1, -2)).detach().float().cpu()
    mass = selected.sum(dim=-1).mean(dim=-1).detach().float().cpu()
    if bool((sink < 0).any()) or bool((mass < 0).any()):
        raise RuntimeError("Computed attention metrics contain negative values")
    return sink, mass


def _configured_attention_heads(model: torch.nn.Module) -> int:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None) or config
    value = getattr(text_config, "num_attention_heads", None)
    if value is None:
        raise RuntimeError("Model config has no num_attention_heads")
    return int(value)


def collect_attention_sinks(
    model: torch.nn.Module,
    inputs: Any,
    target_sources: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Run one attention forward for a teacher-forced stage and aggregate it."""

    input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
    if int(input_ids.shape[0]) != 1:
        raise ValueError("Attention analysis requires batch_size=1")
    sequence_length = int(input_ids.shape[1])
    normalized: dict[str, dict[str, Any]] = {}
    for target, value in target_sources.items():
        target_position = int(value["target_position"])
        source_spans = {
            source: [int(span[0]), int(span[1])]
            for source, span in value["source_spans"].items()
        }
        if not source_spans:
            raise ValueError(f"Target {target!r} has no compared source spans")
        query_start = max(span[1] for span in source_spans.values())
        query_end = target_position
        if not (0 <= query_start < query_end <= sequence_length):
            raise ValueError(
                f"Empty/invalid query for target={target}: "
                f"[{query_start}, {query_end}), sequence={sequence_length}"
            )
        normalized[target] = {
            "query_span": [query_start, query_end],
            "source_spans": source_spans,
        }

    with torch.inference_mode():
        outputs = model(
            **dict(inputs),
            output_attentions=True,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
    attentions = getattr(outputs, "attentions", None)
    if attentions is None:
        raise RuntimeError("Model returned no attentions")
    expected_heads = _configured_attention_heads(model)
    result: dict[str, Any] = {
        target: {
            source: {
                "query_span": list(value["query_span"]),
                "source_span": list(source_span),
                "layers": {},
            }
            for source, source_span in value["source_spans"].items()
        }
        for target, value in normalized.items()
    }
    try:
        for layer_index, attention in enumerate(attentions):
            if attention is None:
                raise RuntimeError(f"Layer {layer_index} returned no attention tensor")
            for target, value in normalized.items():
                for source, source_span in value["source_spans"].items():
                    sink, mass = compute_attention_sink(
                        attention,
                        query_span=value["query_span"],
                        source_span=source_span,
                        expected_heads=expected_heads,
                    )
                    result[target][source]["layers"][str(layer_index)] = {
                        "sink_score_by_head": [
                            float(number) for number in sink.tolist()
                        ],
                        "attention_mass_by_head": [
                            float(number) for number in mass.tolist()
                        ],
                    }
            del attention
    finally:
        del attentions
        del outputs
    return result

