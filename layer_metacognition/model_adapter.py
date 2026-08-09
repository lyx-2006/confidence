"""Adapters around the existing QwenVLInference object and decoder blocks."""

from __future__ import annotations

import copy
import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

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


class AdditiveActivationHook:
    """Apply one additive decoder-output intervention during a model forward.

    ``prefill_sequence_length`` distinguishes the initial generation prefill
    from subsequent one-token cached forwards.  The hook remains registered
    for the complete generation call so its counters can prove that the
    intervention was applied exactly once.
    """

    intervention_mode = "single"

    def __init__(
        self,
        modules: LanguageModules,
        *,
        layer_index: int,
        target_position: int,
        steering_vector: torch.Tensor,
        prefill_sequence_length: int,
        capture_layer_indices: Sequence[int] | None = None,
        injection_site: str = "block_output",
    ) -> None:
        if layer_index < 0 or layer_index >= modules.num_hidden_layers:
            raise ValueError(
                f"Steering layer {layer_index} is outside "
                f"[0, {modules.num_hidden_layers - 1}]"
            )
        if prefill_sequence_length < 1:
            raise ValueError("prefill_sequence_length must be positive")
        if target_position < 0 or target_position >= prefill_sequence_length:
            raise ValueError(
                f"Steering position {target_position} is outside prefill length "
                f"{prefill_sequence_length}"
            )
        if injection_site not in {"block_output", "block_input"}:
            raise ValueError(
                "injection_site must be block_output or block_input, got "
                f"{injection_site!r}"
            )
        vector = steering_vector.detach().reshape(-1)
        if int(vector.numel()) != modules.hidden_size:
            raise ValueError(
                f"Steering vector size {vector.numel()} does not match model "
                f"hidden size {modules.hidden_size}"
            )
        if not bool(torch.isfinite(vector).all()):
            raise ValueError("Steering vector contains non-finite values")

        self.modules = modules
        self.layer_index = int(layer_index)
        self.target_position = int(target_position)
        self.prefill_sequence_length = int(prefill_sequence_length)
        self.steering_vector = vector
        self.injection_site = injection_site
        capture_layers = tuple(
            sorted(set(int(index) for index in (capture_layer_indices or ())))
        )
        invalid_capture_layers = [
            index
            for index in capture_layers
            if index < self.layer_index or index >= modules.num_hidden_layers
        ]
        if invalid_capture_layers:
            raise ValueError(
                "Capture layers must be at or after the Steering layer and inside "
                f"the model: {invalid_capture_layers}"
            )
        self.capture_layer_indices = capture_layers
        self.hook_call_count = 0
        self.applied_count = 0
        self.h_before: torch.Tensor | None = None
        self.h_after: torch.Tensor | None = None
        self.captured_after: dict[int, torch.Tensor] = {}
        self.activation_dtype: str | None = None
        self._handle: Any | None = None
        self._capture_handles: list[Any] = []

    def _patch_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        self.hook_call_count += 1
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            raise TypeError(
                "Decoder block hidden activation must be rank 3, got "
                f"{type(tensor)!r} shape={getattr(tensor, 'shape', None)}"
            )
        if int(tensor.shape[0]) < 1:
            raise ValueError("Decoder block returned an empty batch")
        if int(tensor.shape[2]) != self.modules.hidden_size:
            raise ValueError(
                f"Decoder hidden size {tensor.shape[2]} does not match "
                f"{self.modules.hidden_size}"
            )

        # Cached decode steps normally have sequence length one.  Only the
        # original full prompt is an eligible intervention target.
        if (
            self.applied_count > 0
            or int(tensor.shape[1]) != self.prefill_sequence_length
        ):
            return tensor
        if self.target_position >= int(tensor.shape[1]):
            raise ValueError(
                f"Steering position {self.target_position} is invalid for decoder "
                f"output shape {tuple(tensor.shape)}"
            )

        patched = tensor.clone()
        self.activation_dtype = str(tensor.dtype).removeprefix("torch.")
        before = tensor[0, self.target_position, :].detach().float().cpu()
        vector = self.steering_vector.to(device=patched.device, dtype=patched.dtype)
        patched[0, self.target_position, :] = (
            patched[0, self.target_position, :] + vector
        )
        after = patched[0, self.target_position, :].detach().float().cpu()
        self.h_before = before
        self.h_after = after
        self.applied_count += 1
        return patched

    def _hook(self, _module: Any, _args: Any, output: Any) -> Any:
        if isinstance(output, torch.Tensor):
            tensor = output
            trailing: tuple[Any, ...] | None = None
        elif isinstance(output, tuple) and output:
            tensor = output[0]
            trailing = output[1:]
        else:
            raise TypeError(
                "Decoder block returned unsupported steering output type: "
                f"{type(output)!r}"
            )
        patched = self._patch_tensor(tensor)
        if patched is tensor:
            return output
        if trailing is None:
            return patched
        return (patched, *trailing)

    def _pre_hook(
        self,
        _module: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if args and isinstance(args[0], torch.Tensor):
            patched = self._patch_tensor(args[0])
            if patched is args[0]:
                return args, kwargs
            return (patched, *args[1:]), kwargs
        hidden = kwargs.get("hidden_states")
        if isinstance(hidden, torch.Tensor):
            patched = self._patch_tensor(hidden)
            if patched is hidden:
                return args, kwargs
            updated = dict(kwargs)
            updated["hidden_states"] = patched
            return args, updated
        raise TypeError(
            "Decoder block input hook could not find a hidden-state tensor in "
            "args[0] or kwargs['hidden_states']"
        )

    def _capture_hook(self, layer_index: int, output: Any) -> None:
        if layer_index in self.captured_after:
            return
        tensor = output[0] if isinstance(output, tuple) and output else output
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            raise TypeError(
                "Decoder block hidden output must be rank 3 for trajectory "
                f"capture, got {type(tensor)!r} shape={getattr(tensor, 'shape', None)}"
            )
        if int(tensor.shape[1]) != self.prefill_sequence_length:
            return
        if int(tensor.shape[0]) < 1 or int(tensor.shape[2]) != self.modules.hidden_size:
            raise ValueError(
                f"Invalid trajectory hidden shape at layer {layer_index}: "
                f"{tuple(tensor.shape)}"
            )
        self.captured_after[layer_index] = (
            tensor[0, self.target_position, :].detach().float().cpu().clone()
        )

    def __enter__(self) -> "AdditiveActivationHook":
        if self._handle is not None:
            raise RuntimeError("AdditiveActivationHook cannot be entered twice")
        target_layer = self.modules.language_layers[self.layer_index]
        if self.injection_site == "block_output":
            self._handle = target_layer.register_forward_hook(self._hook)
        else:
            self._handle = target_layer.register_forward_pre_hook(
                self._pre_hook,
                with_kwargs=True,
            )
        for layer_index in self.capture_layer_indices:
            if (
                layer_index == self.layer_index
                and self.injection_site == "block_output"
            ):
                continue
            handle = self.modules.language_layers[layer_index].register_forward_hook(
                lambda _module, _args, output, index=layer_index: self._capture_hook(
                    index, output
                )
            )
            self._capture_handles.append(handle)
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        for handle in self._capture_handles:
            handle.remove()
        self._capture_handles.clear()

    def validate_applied_once(self) -> None:
        if self.applied_count != 1:
            raise RuntimeError(
                f"Steering hook for layer {self.layer_index} applied "
                f"{self.applied_count} times; expected 1"
            )
        if self.h_before is None or self.h_after is None:
            raise RuntimeError("Steering hook did not capture before/after activations")
        implicitly_captured = (
            {self.layer_index} if self.injection_site == "block_output" else set()
        )
        missing = set(self.capture_layer_indices).difference(
            implicitly_captured, self.captured_after
        )
        if missing:
            raise RuntimeError(
                f"Steering trajectory did not capture layer(s): {sorted(missing)}"
            )

    def trajectory_hidden(self) -> dict[int, torch.Tensor]:
        self.validate_applied_once()
        assert self.h_after is not None
        return {
            layer_index: (
                self.h_after.clone()
                if layer_index == self.layer_index
                and self.injection_site == "block_output"
                else self.captured_after[layer_index].clone()
            )
            for layer_index in self.capture_layer_indices
        }

    def diagnostics(self) -> dict[str, Any]:
        self.validate_applied_once()
        assert self.h_before is not None and self.h_after is not None
        return {
            "hook_call_count": int(self.hook_call_count),
            "steering_applied_count": int(self.applied_count),
            "activation_dtype": self.activation_dtype,
            "injection_site": self.injection_site,
            "trajectory_capture_layers": list(self.capture_layer_indices),
            "injection_l2": float(
                torch.linalg.vector_norm(self.h_after - self.h_before).item()
            ),
        }


class _ActivationHookView:
    """One layer of a multi-layer hook, shaped like AdditiveActivationHook."""

    def __init__(self, owner: "ReinjectingActivationHook", layer_index: int) -> None:
        self._owner = owner
        self._layer_index = layer_index
        self.h_before = owner.h_before_by_layer[layer_index]
        self.h_after = owner.h_after_by_layer[layer_index]
        self.steering_vector = owner.steering_vectors[layer_index]
        self.activation_dtype = owner.activation_dtype_by_layer[layer_index]

    def diagnostics(self) -> dict[str, Any]:
        self._owner.validate_applied_once()
        return {
            "hook_call_count": self._owner.hook_call_counts[self._layer_index],
            "steering_applied_count": self._owner.applied_counts[self._layer_index],
            "activation_dtype": self.activation_dtype,
            "injection_site": "block_output",
            "intervention_mode": "reinject",
            "injection_l2": float(
                torch.linalg.vector_norm(self.h_after - self.h_before).item()
            ),
        }


class ReinjectingActivationHook:
    """Inject a layer-specific vector at every requested decoder block output."""

    injection_site = "block_output"
    intervention_mode = "reinject"

    def __init__(
        self,
        modules: LanguageModules,
        *,
        primary_layer_index: int,
        target_position: int,
        steering_vectors: dict[int, torch.Tensor],
        prefill_sequence_length: int,
    ) -> None:
        layers = tuple(sorted(int(index) for index in steering_vectors))
        if not layers or primary_layer_index not in layers:
            raise ValueError("Reinjecting hook requires its primary layer vector")
        if layers[0] != int(primary_layer_index):
            raise ValueError("Reinjecting hook layers must start at the primary layer")
        if any(index < 0 or index >= modules.num_hidden_layers for index in layers):
            raise ValueError(f"Reinjecting hook layer is outside the model: {layers}")
        if prefill_sequence_length < 1:
            raise ValueError("prefill_sequence_length must be positive")
        if target_position < 0 or target_position >= prefill_sequence_length:
            raise ValueError("Reinjecting target position is outside the prefill")
        normalized: dict[int, torch.Tensor] = {}
        for index in layers:
            vector = steering_vectors[index].detach().reshape(-1)
            if vector.numel() != modules.hidden_size or not bool(
                torch.isfinite(vector).all()
            ):
                raise ValueError(
                    f"Invalid reinjection vector at layer {index}: {tuple(vector.shape)}"
                )
            normalized[index] = vector
        self.modules = modules
        self.layer_index = int(primary_layer_index)
        self.target_position = int(target_position)
        self.prefill_sequence_length = int(prefill_sequence_length)
        self.steering_vectors = normalized
        self.capture_layer_indices = layers
        self.hook_call_counts = {index: 0 for index in layers}
        self.applied_counts = {index: 0 for index in layers}
        self.h_before_by_layer: dict[int, torch.Tensor] = {}
        self.h_after_by_layer: dict[int, torch.Tensor] = {}
        self.activation_dtype_by_layer: dict[int, str] = {}
        self._handles: list[Any] = []

    @property
    def h_before(self) -> torch.Tensor | None:
        return self.h_before_by_layer.get(self.layer_index)

    @property
    def h_after(self) -> torch.Tensor | None:
        return self.h_after_by_layer.get(self.layer_index)

    @property
    def steering_vector(self) -> torch.Tensor:
        return self.steering_vectors[self.layer_index]

    @property
    def activation_dtype(self) -> str | None:
        return self.activation_dtype_by_layer.get(self.layer_index)

    @property
    def hook_call_count(self) -> int:
        return self.hook_call_counts[self.layer_index]

    @property
    def applied_count(self) -> int:
        return self.applied_counts[self.layer_index]

    def _hook(self, layer_index: int, output: Any) -> Any:
        self.hook_call_counts[layer_index] += 1
        if isinstance(output, torch.Tensor):
            tensor = output
            trailing: tuple[Any, ...] | None = None
        elif isinstance(output, tuple) and output:
            tensor = output[0]
            trailing = output[1:]
        else:
            raise TypeError(f"Unsupported decoder output at layer {layer_index}")
        if tensor.ndim != 3 or tensor.shape[0] < 1:
            raise TypeError(f"Invalid decoder output shape at layer {layer_index}")
        if tensor.shape[2] != self.modules.hidden_size:
            raise ValueError(f"Hidden-size mismatch at reinjection layer {layer_index}")
        if (
            self.applied_counts[layer_index] > 0
            or int(tensor.shape[1]) != self.prefill_sequence_length
        ):
            return output
        patched = tensor.clone()
        before = tensor[0, self.target_position, :].detach().float().cpu()
        vector = self.steering_vectors[layer_index].to(
            device=patched.device,
            dtype=patched.dtype,
        )
        patched[0, self.target_position, :] += vector
        after = patched[0, self.target_position, :].detach().float().cpu()
        self.h_before_by_layer[layer_index] = before
        self.h_after_by_layer[layer_index] = after
        self.activation_dtype_by_layer[layer_index] = str(tensor.dtype).removeprefix(
            "torch."
        )
        self.applied_counts[layer_index] += 1
        if trailing is None:
            return patched
        return (patched, *trailing)

    def __enter__(self) -> "ReinjectingActivationHook":
        if self._handles:
            raise RuntimeError("ReinjectingActivationHook cannot be entered twice")
        for layer_index in self.capture_layer_indices:
            self._handles.append(
                self.modules.language_layers[layer_index].register_forward_hook(
                    lambda _module, _args, output, index=layer_index: self._hook(
                        index, output
                    )
                )
            )
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def validate_applied_once(self) -> None:
        bad = {
            index: count
            for index, count in self.applied_counts.items()
            if count != 1
        }
        if bad:
            raise RuntimeError(f"Reinjection did not apply exactly once: {bad}")

    def trajectory_hidden(self) -> dict[int, torch.Tensor]:
        self.validate_applied_once()
        return {
            index: self.h_after_by_layer[index].clone()
            for index in self.capture_layer_indices
        }

    def layer_view(self, layer_index: int) -> _ActivationHookView:
        self.validate_applied_once()
        if layer_index not in self.steering_vectors:
            raise KeyError(layer_index)
        return _ActivationHookView(self, layer_index)

    def diagnostics(self) -> dict[str, Any]:
        return self.layer_view(self.layer_index).diagnostics() | {
            "reinjected_layers": list(self.capture_layer_indices),
        }


def _selected_logits_kwargs(
    model: torch.nn.Module,
    inputs: Any,
    positions: list[int],
    modules: LanguageModules | None = None,
) -> tuple[dict[str, Any], bool]:
    kwargs = dict(inputs)
    signature = inspect.signature(model.forward)
    uses_selected_logits = "logits_to_keep" in signature.parameters
    if uses_selected_logits:
        input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
        if modules is not None:
            try:
                target_device = next(modules.final_norm.parameters()).device
            except StopIteration:
                target_device = input_ids.device
        else:
            target_device = input_ids.device
        kwargs["logits_to_keep"] = torch.tensor(
            positions,
            dtype=torch.long,
            device=target_device,
        )
    return kwargs, uses_selected_logits


def run_logits_forward(
    model: torch.nn.Module,
    inputs: Any,
    positions: list[int],
    modules: LanguageModules | None = None,
) -> dict[int, torch.Tensor]:
    """Read reference vocab logits at selected teacher-forced positions."""

    input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
    sequence_length = int(input_ids.shape[1])
    requested = sorted(set(int(position) for position in positions))
    if not requested:
        raise ValueError("At least one logits position is required")
    if any(position < 0 or position >= sequence_length for position in requested):
        raise ValueError(
            f"Logits positions outside sequence length {sequence_length}: {requested}"
        )
    kwargs, selected = _selected_logits_kwargs(model, inputs, requested, modules)
    with torch.inference_mode():
        outputs = model(
            **kwargs,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    if selected:
        result = {
            position: outputs.logits[0, local_index].detach().float().cpu()
            for local_index, position in enumerate(requested)
        }
    else:
        result = {
            position: outputs.logits[0, position].detach().float().cpu()
            for position in requested
        }
    del outputs
    return result


def run_patched_logits_forward(
    model: torch.nn.Module,
    inputs: Any,
    modules: LanguageModules,
    *,
    layer_index: int,
    target_position: int,
    source_hidden: torch.Tensor,
) -> torch.Tensor:
    """Patch one decoder-block output and return one target-position logit row."""

    if layer_index < 0 or layer_index >= modules.num_hidden_layers:
        raise ValueError(
            f"Patch layer {layer_index} is outside [0, {modules.num_hidden_layers - 1}]"
        )
    input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
    if input_ids.ndim != 2 or int(input_ids.shape[0]) < 1:
        raise ValueError(
            f"Patched target input_ids must have shape [batch, sequence], got {input_ids.shape}"
        )
    sequence_length = int(input_ids.shape[1])
    if target_position < 0 or target_position >= sequence_length:
        raise ValueError(
            f"Patch target position {target_position} is outside sequence length "
            f"{sequence_length}"
        )
    vector = source_hidden.detach().reshape(-1)
    if int(vector.numel()) != modules.hidden_size:
        raise ValueError(
            f"Source hidden size {vector.numel()} does not match model hidden size "
            f"{modules.hidden_size}"
        )

    hook_calls = 0

    def patch_output(_module: Any, _args: Any, output: Any) -> Any:
        nonlocal hook_calls
        hook_calls += 1
        if isinstance(output, torch.Tensor):
            tensor = output
            trailing: tuple[Any, ...] | None = None
        elif isinstance(output, tuple) and output:
            tensor = output[0]
            trailing = output[1:]
        else:
            raise TypeError(
                "Decoder block returned unsupported patch output type: "
                f"{type(output)!r}"
            )
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            raise TypeError(
                "Decoder block hidden output must be a rank-3 tensor, got "
                f"{type(tensor)!r} shape={getattr(tensor, 'shape', None)}"
            )
        if int(tensor.shape[0]) < 1 or target_position >= int(tensor.shape[1]):
            raise ValueError(
                f"Patch position {target_position} is invalid for decoder output "
                f"shape {tuple(tensor.shape)}"
            )
        if int(tensor.shape[2]) != modules.hidden_size:
            raise ValueError(
                f"Decoder hidden size {tensor.shape[2]} does not match "
                f"{modules.hidden_size}"
            )
        patched = tensor.clone()
        patched[0, target_position, :] = vector.to(
            device=patched.device,
            dtype=patched.dtype,
        )
        if trailing is None:
            return patched
        return (patched, *trailing)

    handle = modules.language_layers[layer_index].register_forward_hook(patch_output)
    outputs = None
    try:
        kwargs, selected = _selected_logits_kwargs(
            model,
            inputs,
            [target_position],
            modules,
        )
        with torch.inference_mode():
            outputs = model(
                **kwargs,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        if hook_calls != 1:
            raise RuntimeError(
                f"Patch hook for layer {layer_index} fired {hook_calls} times; expected 1"
            )
        logits = (
            outputs.logits[0, 0]
            if selected
            else outputs.logits[0, target_position]
        )
        return logits.detach().float().cpu()
    finally:
        handle.remove()
        if outputs is not None:
            del outputs


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
        kwargs, uses_selected_logits = _selected_logits_kwargs(
            model,
            inputs,
            requested_positions,
            modules,
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
