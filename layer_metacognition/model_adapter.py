"""Adapters around the existing QwenVLInference object and decoder blocks."""

from __future__ import annotations

import copy
import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class LanguageModules:
    language_layers: list[torch.nn.Module]
    final_norm: torch.nn.Module
    lm_head: torch.nn.Module
    hidden_size: int
    num_hidden_layers: int


@dataclass
class HookedForwardResult:
    hidden_by_name: dict[str, dict[int, torch.Tensor]]
    logits_by_position: dict[int, torch.Tensor]


def load_qwen_inference(
    model_path: str,
    inference_path: str | Path | None = None,
    min_pixels: int = 256 * 28 * 28,
    max_pixels: int = 1280 * 28 * 28,
) -> Any:
    root = Path(__file__).resolve().parents[1]
    source = Path(inference_path) if inference_path is not None else root / "qwen-2.5-vl" / "inference.py"
    source = source.resolve()
    specification = importlib.util.spec_from_file_location("layer_metacognition_qwen_inference", source)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load QwenVLInference from {source}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module.QwenVLInference(
        model_path=model_path,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )


def _resolve_path(root: Any, path: str) -> Any | None:
    current = root
    for component in path.split("."):
        if not hasattr(current, component):
            return None
        current = getattr(current, component)
    return current


def resolve_language_modules(model: torch.nn.Module) -> LanguageModules:
    layer_candidates = [
        "model.language_model.layers",
        "language_model.layers",
        "model.model.layers",
        "model.layers",
    ]
    norm_candidates = [
        "model.language_model.norm",
        "language_model.norm",
        "model.model.norm",
        "model.norm",
    ]
    head_candidates = ["lm_head", "model.lm_head", "language_model.lm_head"]
    layers = next((_resolve_path(model, path) for path in layer_candidates if _resolve_path(model, path) is not None), None)
    final_norm = next((_resolve_path(model, path) for path in norm_candidates if _resolve_path(model, path) is not None), None)
    lm_head = next((_resolve_path(model, path) for path in head_candidates if _resolve_path(model, path) is not None), None)
    if layers is None or final_norm is None or lm_head is None:
        raise RuntimeError("Could not resolve Qwen language layers, FinalNorm, and LM Head")
    language_layers = list(layers)
    if not language_layers or not all(isinstance(layer, torch.nn.Module) for layer in language_layers):
        raise RuntimeError("Resolved language layers are not decoder modules")
    for index, layer in enumerate(language_layers):
        if not hasattr(layer, "self_attn") or not hasattr(layer, "mlp"):
            raise RuntimeError(f"Resolved layer {index} does not look like a decoder block")

    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None) or config
    config_layers = int(getattr(text_config, "num_hidden_layers", len(language_layers)))
    hidden_size = int(getattr(text_config, "hidden_size", 0))
    if len(language_layers) != config_layers:
        raise RuntimeError(
            f"Decoder-block count mismatch: resolved={len(language_layers)}, config={config_layers}"
        )
    if hidden_size < 1:
        try:
            hidden_size = int(lm_head.in_features)
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError("Could not resolve language hidden size") from exc
    if hasattr(lm_head, "in_features") and int(lm_head.in_features) != hidden_size:
        raise RuntimeError(
            f"LM Head hidden size mismatch: head={lm_head.in_features}, config={hidden_size}"
        )
    return LanguageModules(language_layers, final_norm, lm_head, hidden_size, config_layers)


def model_input_device(inference: Any) -> torch.device:
    try:
        embeddings = inference.model.get_input_embeddings()
        return next(embeddings.parameters()).device
    except (AttributeError, StopIteration):
        return inference._get_inputs_device()


def parse_layer_selection(specification: str, num_hidden_layers: int) -> list[int]:
    if specification.strip().casefold() == "all":
        return list(range(num_hidden_layers))
    try:
        layers = [int(value.strip()) for value in specification.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid --layers value: {specification!r}") from exc
    if not layers or len(layers) != len(set(layers)):
        raise ValueError("--layers must contain distinct layer indices")
    if any(layer < 0 or layer >= num_hidden_layers for layer in layers):
        raise ValueError(f"--layers must be in [0, {num_hidden_layers - 1}]")
    return sorted(layers)


def generate_from_prefill(
    inference: Any,
    inputs: Any,
    max_new_tokens: int = 8,
) -> str:
    generation_config = copy.deepcopy(inference.model.generation_config)
    generation_config.do_sample = False
    generation_config.temperature = None
    generation_config.top_p = None
    generation_config.top_k = None
    with torch.inference_mode():
        generated = inference.model.generate(
            **inputs,
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    input_length = int(inputs.input_ids.shape[1])
    new_tokens = generated[0, input_length:]
    tokenizer = getattr(inference.processor, "tokenizer", inference.processor)
    return tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def _output_tensor(output: Any) -> torch.Tensor:
    tensor = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Decoder block returned unsupported output type: {type(output)!r}")
    return tensor


def run_hooked_forward(
    model: torch.nn.Module,
    inputs: Any,
    modules: LanguageModules,
    positions: dict[str, int],
    logits_positions: list[int] | None = None,
) -> HookedForwardResult:
    input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
    sequence_length = int(input_ids.shape[1])
    for name, position in positions.items():
        if position < 0 or position >= sequence_length:
            raise ValueError(f"{name} position {position} is outside sequence length {sequence_length}")

    captured: dict[str, dict[int, torch.Tensor]] = {name: {} for name in positions}
    handles: list[Any] = []
    try:
        for layer_index, layer in enumerate(modules.language_layers):
            def capture(_module: Any, _inputs: Any, output: Any, index: int = layer_index) -> None:
                tensor = _output_tensor(output)
                for name, position in positions.items():
                    captured[name][index] = tensor[0, position, :].detach().clone()

            handles.append(layer.register_forward_hook(capture))

        requested_positions = sorted(
            set(positions.values()) if logits_positions is None else set(logits_positions)
        )
        if not requested_positions:
            raise ValueError("At least one logits position must be requested")
        invalid_logits_positions = [
            position
            for position in requested_positions
            if position < 0 or position >= sequence_length
        ]
        if invalid_logits_positions:
            raise ValueError(
                f"Logits positions outside sequence length {sequence_length}: {invalid_logits_positions}"
            )
        kwargs = dict(inputs)
        signature = inspect.signature(model.forward)
        uses_selected_logits = "logits_to_keep" in signature.parameters
        if uses_selected_logits:
            try:
                norm_device = next(modules.final_norm.parameters()).device
            except StopIteration:
                norm_device = input_ids.device
            kwargs["logits_to_keep"] = torch.tensor(
                requested_positions,
                dtype=torch.long,
                device=norm_device,
            )
        with torch.inference_mode():
            outputs = model(
                **kwargs,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        if any(len(values) != modules.num_hidden_layers for values in captured.values()):
            counts = {name: len(values) for name, values in captured.items()}
            raise RuntimeError(f"Hooks did not capture every decoder block: {counts}")
        if uses_selected_logits:
            logits = {
                position: outputs.logits[0, local_index].detach().float().cpu()
                for local_index, position in enumerate(requested_positions)
            }
        else:
            logits = {
                position: outputs.logits[0, position].detach().float().cpu()
                for position in requested_positions
            }
        return HookedForwardResult(captured, logits)
    finally:
        for handle in handles:
            handle.remove()
