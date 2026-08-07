"""Optional full-batch PyTorch logistic Probe used for fast fixed-C scans."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler


GRAD_TOLERANCE = 1e-5
LOSS_TOLERANCE = 1e-7


class TorchProbeNumericalError(RuntimeError):
    """Raised when a Torch Probe produces non-finite numerical values."""


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            "--backend torch requires PyTorch; install it in the active environment"
        ) from exc
    return torch


def resolve_torch_device(requested: str) -> str:
    torch = _import_torch()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported Torch device: {requested!r}")
    return requested


def balanced_sample_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    encoded = np.asarray(y, dtype=np.int64)
    counts = np.bincount(encoded, minlength=n_classes)
    if np.any(counts <= 0):
        raise ValueError("Every encoded training class must have at least one sample")
    class_weights = len(encoded) / (float(n_classes) * counts.astype(np.float64))
    return class_weights[encoded].astype(np.float32)


def sklearn_l2_strength(C: float, sample_weight_sum: float) -> float:
    if not math.isfinite(C) or C <= 0:
        raise ValueError("C must be a finite positive value")
    if not math.isfinite(sample_weight_sum) or sample_weight_sum <= 0:
        raise ValueError("sample_weight_sum must be finite and positive")
    return 1.0 / (float(C) * float(sample_weight_sum))


@dataclass
class TorchLogisticProbe:
    scaler: StandardScaler
    weight: np.ndarray
    intercept: np.ndarray
    n_classes: int
    binary: bool
    diagnostics: dict[str, Any]

    @property
    def classes_(self) -> np.ndarray:
        return np.arange(self.n_classes, dtype=np.int64)

    def _logits(self, X: np.ndarray) -> np.ndarray:
        transformed = self.scaler.transform(
            np.asarray(X, dtype=np.float32)
        ).astype(np.float32, copy=False)
        logits = transformed @ self.weight.T + self.intercept
        if not np.all(np.isfinite(logits)):
            raise TorchProbeNumericalError("Torch Probe produced non-finite logits")
        return logits

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self._logits(X).astype(np.float64, copy=False)
        if self.binary:
            values = logits[:, 0]
            positive = np.empty_like(values)
            mask = values >= 0
            positive[mask] = 1.0 / (1.0 + np.exp(-values[mask]))
            exp_values = np.exp(values[~mask])
            positive[~mask] = exp_values / (1.0 + exp_values)
            # For Conflict Probe, sklearn's encoded class 1 is ``consistent``.
            # Recover the semantic [consistent, conflict] pair first, then align
            # it back to the existing LabelEncoder order [conflict, consistent].
            p_consistent = positive
            p_conflict = 1.0 - positive
            semantic_probabilities = np.column_stack((p_consistent, p_conflict))
            probabilities = semantic_probabilities[:, [1, 0]]
        else:
            shifted = logits - np.max(logits, axis=1, keepdims=True)
            exponentials = np.exp(shifted)
            probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
        if not np.all(np.isfinite(probabilities)):
            raise TorchProbeNumericalError(
                "Torch Probe produced non-finite probabilities"
            )
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-7):
            raise TorchProbeNumericalError(
                "Torch Probe probabilities do not sum to one"
            )
        return probabilities

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1).astype(np.int64)


def fit_torch_logistic_probe(
    X: np.ndarray,
    y: np.ndarray,
    *,
    C: float,
    device: str = "auto",
    seed: int = 42,
    binary_single_logit: bool = False,
    max_iter_per_phase: int = 100,
    max_phases: int = 2,
    grad_tolerance: float = GRAD_TOLERANCE,
    loss_tolerance: float = LOSS_TOLERANCE,
) -> TorchLogisticProbe:
    """Fit a sklearn-compatible weighted L2 logistic model with Torch LBFGS."""

    torch = _import_torch()
    resolved_device = resolve_torch_device(device)
    features = np.asarray(X, dtype=np.float32)
    encoded = np.asarray(y, dtype=np.int64)
    if features.ndim != 2 or len(features) != len(encoded) or not len(encoded):
        raise ValueError("X/y must be non-empty aligned rank-2/rank-1 arrays")
    if not np.all(np.isfinite(features)):
        raise TorchProbeNumericalError("Torch Probe input contains NaN or Inf")
    classes = np.unique(encoded)
    if not np.array_equal(classes, np.arange(len(classes), dtype=np.int64)):
        raise ValueError("Torch Probe labels must be contiguous integers starting at 0")
    n_classes = len(classes)
    if n_classes < 2:
        raise ValueError("Torch Probe requires at least two training classes")
    if binary_single_logit and n_classes != 2:
        raise ValueError("binary_single_logit requires exactly two classes")
    binary = bool(binary_single_logit)
    preprocessing_start = time.perf_counter()
    scaler = StandardScaler()
    transformed = scaler.fit_transform(features).astype(np.float32, copy=False)
    sample_weights = balanced_sample_weights(encoded, n_classes)
    sample_weight_sum = float(np.sum(sample_weights, dtype=np.float64))
    l2_strength = sklearn_l2_strength(C, sample_weight_sum)
    preprocessing_seconds = time.perf_counter() - preprocessing_start

    torch.manual_seed(int(seed))
    if resolved_device == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.cuda.synchronize()
    transfer_start = time.perf_counter()
    train_X = torch.as_tensor(transformed, dtype=torch.float32, device=resolved_device)
    train_y = torch.as_tensor(encoded, dtype=torch.long, device=resolved_device)
    train_weights = torch.as_tensor(
        sample_weights, dtype=torch.float32, device=resolved_device
    )
    if resolved_device == "cuda":
        torch.cuda.synchronize()
    gpu_transfer_seconds = time.perf_counter() - transfer_start

    output_size = 1 if binary else n_classes
    linear = torch.nn.Linear(features.shape[1], output_size, bias=True).to(
        resolved_device
    )
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.zero_()
    optimizer = torch.optim.LBFGS(
        linear.parameters(),
        lr=1.0,
        max_iter=int(max_iter_per_phase),
        tolerance_grad=float(grad_tolerance),
        tolerance_change=float(loss_tolerance),
        history_size=100,
        line_search_fn="strong_wolfe",
    )
    closure_evaluations = 0

    def objective():
        logits = linear(train_X)
        if binary:
            pointwise = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[:, 0], train_y.to(torch.float32), reduction="none"
            )
        else:
            pointwise = torch.nn.functional.cross_entropy(
                logits, train_y, reduction="none"
            )
        weighted_loss = torch.sum(pointwise * train_weights) / torch.sum(train_weights)
        penalty = 0.5 * float(l2_strength) * torch.sum(linear.weight.square())
        return weighted_loss + penalty

    def closure():
        nonlocal closure_evaluations
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        if not bool(torch.isfinite(loss).item()):
            raise TorchProbeNumericalError("Torch Probe loss became NaN or Inf")
        loss.backward()
        closure_evaluations += 1
        return loss

    def final_diagnostics() -> tuple[float, float, float | None, bool]:
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        if not bool(torch.isfinite(loss).item()):
            raise TorchProbeNumericalError("Torch Probe final loss is NaN or Inf")
        loss.backward()
        gradients = [
            parameter.grad.detach().abs().max()
            for parameter in linear.parameters()
            if parameter.grad is not None
        ]
        gradient_norm = float(torch.stack(gradients).max().item())
        final_loss = float(loss.detach().item())
        first_parameter = next(iter(linear.parameters()))
        previous_loss = optimizer.state.get(first_parameter, {}).get("prev_loss")
        relative_change: float | None = None
        if previous_loss is not None and math.isfinite(float(previous_loss)):
            relative_change = abs(final_loss - float(previous_loss)) / max(
                1.0, abs(float(previous_loss))
            )
        converged = math.isfinite(final_loss) and (
            gradient_norm <= grad_tolerance
            or (
                relative_change is not None
                and relative_change <= loss_tolerance
            )
        )
        if not math.isfinite(gradient_norm):
            raise TorchProbeNumericalError("Torch Probe gradient became NaN or Inf")
        return final_loss, gradient_norm, relative_change, converged

    fit_start = time.perf_counter()
    phases_run = 0
    diagnostics: tuple[float, float, float | None, bool] | None = None
    for _phase in range(int(max_phases)):
        optimizer.step(closure)
        phases_run += 1
        diagnostics = final_diagnostics()
        if diagnostics[3]:
            break
    if resolved_device == "cuda":
        torch.cuda.synchronize()
    fit_seconds = time.perf_counter() - fit_start
    assert diagnostics is not None
    final_loss, gradient_norm, relative_change, converged = diagnostics
    first_parameter = next(iter(linear.parameters()))
    state = optimizer.state.get(first_parameter, {})
    iterations = int(state.get("n_iter", 0))
    with torch.no_grad():
        weight = linear.weight.detach().cpu().numpy().astype(np.float32, copy=True)
        intercept = linear.bias.detach().cpu().numpy().astype(np.float32, copy=True)

    result_diagnostics = {
        "backend": "torch",
        "device": resolved_device,
        "C": float(C),
        "iterations": iterations,
        "closure_evaluations": int(closure_evaluations),
        "final_loss": final_loss,
        "gradient_norm": gradient_norm,
        "relative_loss_change": relative_change,
        "retry_count": max(0, phases_run - 1),
        "converged": bool(converged),
        "grad_tolerance": float(grad_tolerance),
        "loss_tolerance": float(loss_tolerance),
        "gpu_transfer_seconds": float(gpu_transfer_seconds),
        "preprocessing_seconds": float(preprocessing_seconds),
        "fit_seconds": float(fit_seconds),
        "binary_single_logit": binary,
    }
    return TorchLogisticProbe(
        scaler=scaler,
        weight=weight,
        intercept=intercept,
        n_classes=n_classes,
        binary=binary,
        diagnostics=result_diagnostics,
    )


__all__ = [
    "GRAD_TOLERANCE",
    "LOSS_TOLERANCE",
    "TorchLogisticProbe",
    "TorchProbeNumericalError",
    "balanced_sample_weights",
    "fit_torch_logistic_probe",
    "resolve_torch_device",
    "sklearn_l2_strength",
]
