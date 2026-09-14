from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, Gemma3ForConditionalGeneration

from experiment_config import HIDDEN_SIZE, MODEL_PATH, NUM_HIDDEN_LAYERS


@dataclass(frozen=True)
class LanguageModules:
    layers: list[torch.nn.Module]
    final_norm: torch.nn.Module
    lm_head: torch.nn.Module
    hidden_size: int
    num_hidden_layers: int


@dataclass
class ForwardCapture:
    hidden_by_name: dict[str, dict[int, torch.Tensor]]
    logits_by_position: dict[int, torch.Tensor]


def _float_to_bf16(inputs: Any, device: torch.device) -> Any:
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            continue
        inputs[key] = value.to(device=device, dtype=torch.bfloat16 if value.is_floating_point() else None)
    return inputs


class GemmaRuntime:
    def __init__(self, model_path: str | Path = MODEL_PATH) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Gemma activation steering requires a CUDA device")
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Gemma model directory not found: {self.model_path}")
        # Pin the checkpoint's saved slow image processor while retaining a fast
        # tokenizer, which is required for exact character offsets.
        self.processor = AutoProcessor.from_pretrained(
            str(self.model_path), local_files_only=True, use_fast=False
        )
        self.processor.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), local_files_only=True, use_fast=True
        )
        if not self.processor.tokenizer.is_fast:
            raise RuntimeError("Gemma position alignment requires a fast tokenizer")
        self.device = torch.device("cuda:0")
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            str(self.model_path),
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).to(self.device).eval()
        self.model.generation_config.cache_implementation = None
        bad = [
            f"{name}:{parameter.dtype}"
            for name, parameter in self.model.named_parameters()
            if parameter.is_floating_point() and parameter.dtype != torch.bfloat16
        ]
        if bad:
            raise RuntimeError(f"Non-BF16 Gemma weights detected: {bad[:10]}")
        self.modules = resolve_language_modules(self.model)

    def build_messages(self, prompt: str, image_path: str | Path, prefill: str) -> list[dict[str, Any]]:
        with Image.open(image_path) as source:
            image = source.convert("RGB").copy()
        return [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": prefill}]},
        ]

    def prepare(self, messages: list[dict[str, Any]], prefill: str) -> tuple[str, Any]:
        rendered = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        if not rendered.endswith(prefill):
            raise RuntimeError(f"Gemma chat template did not preserve assistant prefill {prefill!r}")
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
            do_pan_and_scan=False,
        )
        if inputs.input_ids[0, :2].tolist() == [2, 2]:
            raise RuntimeError("Gemma input contains a duplicated BOS token")
        return rendered, _float_to_bf16(inputs, self.device)

    def generate(
        self,
        inputs: Any,
        max_new_tokens: int,
        allowed_first_tokens: Sequence[int] | None = None,
    ) -> tuple[list[int], str, bool]:
        generation_config = copy.deepcopy(self.model.generation_config)
        generation_config.top_p = None
        generation_config.top_k = None
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "use_cache": True,
            "generation_config": generation_config,
        }
        if allowed_first_tokens is not None:
            allowed = tuple(map(int, allowed_first_tokens))
            kwargs["prefix_allowed_tokens_fn"] = lambda _batch, _ids: list(allowed)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, **kwargs)
        input_length = int(inputs.input_ids.shape[1])
        tokens = [int(value) for value in generated[0, input_length:].tolist()]
        text = self.processor.tokenizer.decode(
            tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        eos_ids = self.model.generation_config.eos_token_id or []
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        return tokens, text.strip(), bool(tokens and tokens[-1] in set(eos_ids))


def resolve_language_modules(model: torch.nn.Module) -> LanguageModules:
    language_model = model.model.language_model
    layers = list(language_model.layers)
    text_config = model.config.text_config
    if len(layers) != NUM_HIDDEN_LAYERS or int(text_config.num_hidden_layers) != NUM_HIDDEN_LAYERS:
        raise RuntimeError(f"Expected {NUM_HIDDEN_LAYERS} Gemma layers, found {len(layers)}")
    if int(text_config.hidden_size) != HIDDEN_SIZE:
        raise RuntimeError(f"Expected Gemma hidden size {HIDDEN_SIZE}, found {text_config.hidden_size}")
    if not all(hasattr(layer, "self_attn") and hasattr(layer, "mlp") for layer in layers):
        raise RuntimeError("Resolved Gemma layers are not decoder blocks")
    return LanguageModules(layers, language_model.norm, model.lm_head, HIDDEN_SIZE, NUM_HIDDEN_LAYERS)


def _selected_logits_kwargs(model: torch.nn.Module, inputs: Any, positions: Sequence[int]) -> dict[str, Any]:
    kwargs = dict(inputs)
    kwargs["logits_to_keep"] = torch.tensor(
        sorted(set(map(int, positions))), dtype=torch.long, device=inputs.input_ids.device
    )
    return kwargs


def run_logits_forward(model: torch.nn.Module, inputs: Any, positions: Sequence[int]) -> dict[int, torch.Tensor]:
    requested = sorted(set(map(int, positions)))
    sequence_length = int(inputs.input_ids.shape[1])
    if not requested or any(position < 0 or position >= sequence_length for position in requested):
        raise ValueError(f"Invalid logits positions for length {sequence_length}: {requested}")
    with torch.inference_mode():
        outputs = model(
            **_selected_logits_kwargs(model, inputs, requested),
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    result = {
        position: outputs.logits[0, local].detach().float().cpu()
        for local, position in enumerate(requested)
    }
    del outputs
    return result


def run_capture_forward(
    model: torch.nn.Module,
    inputs: Any,
    modules: LanguageModules,
    positions: dict[str, int],
    layers: Sequence[int],
    logits_positions: Sequence[int],
) -> ForwardCapture:
    hidden: dict[str, dict[int, torch.Tensor]] = {name: {} for name in positions}
    handles = []

    def capture(layer_index: int, output: Any) -> None:
        tensor = output[0] if isinstance(output, tuple) else output
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            raise TypeError(f"Unsupported Gemma decoder output at L{layer_index}")
        for name, position in positions.items():
            hidden[name][layer_index] = tensor[0, position].detach().float().cpu().clone()

    for layer in layers:
        handles.append(modules.layers[layer].register_forward_hook(
            lambda _module, _args, output, index=layer: capture(index, output)
        ))
    try:
        logits = run_logits_forward(model, inputs, logits_positions)
    finally:
        for handle in handles:
            handle.remove()
    expected = len(positions) * len(tuple(layers))
    actual = sum(len(values) for values in hidden.values())
    if actual != expected:
        raise RuntimeError(f"Incomplete hidden capture: expected {expected}, found {actual}")
    return ForwardCapture(hidden, logits)


class AdditiveActivationHook:
    def __init__(
        self,
        modules: LanguageModules,
        layer_index: int,
        target_position: int,
        steering_vector: torch.Tensor,
        prefill_sequence_length: int,
    ) -> None:
        if layer_index < 0 or layer_index >= modules.num_hidden_layers:
            raise ValueError(f"Invalid steering layer: {layer_index}")
        if target_position < 0 or target_position >= prefill_sequence_length:
            raise ValueError(f"Invalid steering position: {target_position}")
        vector = steering_vector.detach().reshape(-1)
        if vector.numel() != modules.hidden_size or not bool(torch.isfinite(vector).all()):
            raise ValueError("Invalid steering vector")
        self.modules = modules
        self.layer_index = int(layer_index)
        self.target_position = int(target_position)
        self.vector = vector
        self.prefill_sequence_length = int(prefill_sequence_length)
        self.hook_call_count = 0
        self.applied_count = 0
        self.before: torch.Tensor | None = None
        self.after: torch.Tensor | None = None
        self.activation_dtype: str | None = None
        self.handle: Any | None = None

    def _hook(self, _module: Any, _args: Any, output: Any) -> Any:
        self.hook_call_count += 1
        tensor = output[0] if isinstance(output, tuple) else output
        trailing = output[1:] if isinstance(output, tuple) else None
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            raise TypeError("Unsupported Gemma decoder output for steering")
        if self.applied_count or int(tensor.shape[1]) != self.prefill_sequence_length:
            return output
        patched = tensor.clone()
        self.activation_dtype = str(tensor.dtype).removeprefix("torch.")
        self.before = tensor[0, self.target_position].detach().float().cpu()
        patched[0, self.target_position] += self.vector.to(patched.device, patched.dtype)
        self.after = patched[0, self.target_position].detach().float().cpu()
        self.applied_count += 1
        return patched if trailing is None else (patched, *trailing)

    def __enter__(self) -> "AdditiveActivationHook":
        self.handle = self.modules.layers[self.layer_index].register_forward_hook(self._hook)
        return self

    def __exit__(self, *_args: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def diagnostics(self) -> dict[str, Any]:
        if self.applied_count != 1 or self.before is None or self.after is None:
            raise RuntimeError(f"Steering hook applied {self.applied_count} times; expected once")
        return {
            "hook_call_count": self.hook_call_count,
            "steering_applied_count": self.applied_count,
            "activation_dtype": self.activation_dtype,
            "injection_site": "block_output",
            "injection_l2": float(torch.linalg.vector_norm(self.after - self.before)),
        }
