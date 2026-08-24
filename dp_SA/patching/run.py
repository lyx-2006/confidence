from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from dp_SA.config import DATASET_PATH, INFERENCE_PATH, MIDPOINTS, MODEL_PATH
from dp_SA.soft_score import class_token_ids
from layer_metacognition.model_adapter import load_qwen_inference, resolve_language_modules, run_logits_forward

from .artifacts import build_or_load_artifacts, evaluation_image_key
from .config import (
    BOOTSTRAP_REPEATS, DEFAULT_CAPTURE_DIR, DEFAULT_EVAL_CASES, DEFAULT_LAYERS,
    DEFAULT_OUTPUT_PARENT, DEFAULT_POSITIONS, FORMAT_VERSION, HISTORICAL_LAYERS,
    LOGIT_PARITY_TOLERANCE, MODEL_CONFIG_FILES, SEED, SOFT_PARITY_TOLERANCE, CORRUPTIONS,
    parse_layers, parse_positions,
)
from .hooks import (
    ActivationReplacementHook, EmbeddingReplacement, EmbeddingReplacementHook,
    EmptyActivationHook, PatchingInvariantError, ResidualActivationCacheHook,
    resolve_language_model,
)
from .io import (
    atomic_append_jsonl, atomic_json, atomic_jsonl, atomic_torch_save, canonical_hash,
    directory_code_hash, load_jsonl_strict, sha256_file,
)
from .lock import FormalRunLock
from .metrics import recovery, score_logits
from .protocol import prepare_delayed_case, span_positions
from .selection import load_and_select


def _default_output() -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return DEFAULT_OUTPUT_PARENT / f"delayed_patching_seed42_{stamp}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--positions", nargs="+", default=list(DEFAULT_POSITIONS))
    parser.add_argument("--layers", nargs="+", type=int, default=list(DEFAULT_LAYERS))
    parser.add_argument("--eval-cases", type=int, default=DEFAULT_EVAL_CASES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPEATS)
    parser.add_argument("--corruption", choices=CORRUPTIONS, default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def _model_files(model_path: Path) -> dict[str, str]:
    return {name: sha256_file(model_path / name) for name in MODEL_CONFIG_FILES if (model_path / name).is_file()}


def _git_version(root: Path) -> dict[str, Any]:
    def command(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, check=False).stdout.strip()
    return {"commit": command("rev-parse", "HEAD"), "status": command("status", "--short", "--", "dp_SA/patching")}


def _implementation_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = list(Path(__file__).resolve().parent.glob("*.py")) + [
        root / "dp_SA" / "prompts.py", root / "dp_SA" / "positions.py",
        root / "dp_SA" / "soft_score.py", root / "dp_SA" / "attention_block" / "spans.py",
        root / "layer_metacognition" / "conversation_builder.py",
        root / "layer_metacognition" / "sa_patching" / "sa_patching_hook.py",
        root / "layer_metacognition" / "sa_patching" / "artifacts.py",
        root / "qwen-2.5-vl" / "inference.py",
    ]
    return directory_code_hash(paths)


def _prepare_manifests(capture_dir: Path, output_dir: Path, *, eval_cases: int, seed: int, resume: bool):
    evaluation, calibration = load_and_select(capture_dir, eval_cases=eval_cases, seed=seed)
    for name, value in (("evaluation_manifest.json", evaluation), ("calibration_manifest.json", calibration)):
        path = output_dir / name
        if path.exists():
            current = json.loads(path.read_text())
            comparable = dict(current)
            comparable.pop("runtime_artifacts", None)
            if canonical_hash(comparable) != canonical_hash(value):
                raise ValueError(f"Existing {name} differs; refusing resume")
        else:
            atomic_json(path, value)
    return evaluation, calibration


def _link_calibration_artifacts(output_dir: Path, calibration: dict[str, Any], artifacts: Any) -> None:
    enriched = {
        **calibration,
        "runtime_artifacts": {
            "manifest": "corruption_embeddings/artifact_manifest.json",
            "fingerprint": artifacts.metadata["fingerprint"],
            "image_group_counts": artifacts.metadata["image_group_counts"],
            "text_position_counts": artifacts.metadata["text_position_counts"],
            "answer_counts": artifacts.metadata["answer_counts"],
            "source_metadata_counts": {
                name: len(artifacts.metadata["source_metadata"][name])
                for name in ("image", "text", "answer")
            },
        },
    }
    path = output_dir / "calibration_manifest.json"
    if path.exists() and canonical_hash(json.loads(path.read_text())) == canonical_hash(enriched):
        return
    atomic_json(path, enriched)


def _forward_class_logits(model: Any, inputs: Any, modules: Any, class_ids: Sequence[int], sac: int) -> list[float]:
    vocab = run_logits_forward(model, inputs, [sac], modules)[sac]
    values = [float(vocab[int(token)].item()) for token in class_ids]
    if len(values) != 9 or not all(math.isfinite(value) for value in values):
        raise PatchingInvariantError("Forward returned invalid class logits")
    return values


def _clean_cache_path(output_dir: Path, case_id: str) -> Path:
    return output_dir / "clean_cache" / f"{case_id}.pt"


def _historical_parity(capture_dir: Path, row: dict[str, Any], cache: dict[int, dict[str, torch.Tensor]], layers: Sequence[int]) -> dict[str, Any]:
    hidden_path = capture_dir.parent / str(row["hidden_file"])
    if not hidden_path.is_file():
        raise FileNotFoundError(f"Historical hidden capture is missing: {hidden_path}")
    checked: list[dict[str, Any]] = []
    with np.load(hidden_path) as historical:
        for layer in sorted(set(layers) & set(HISTORICAL_LAYERS)):
            for position in DEFAULT_POSITIONS:
                key = f"{position}__L{layer}"
                current = cache[layer][position].numpy().astype(np.float16)
                expected = historical[key]
                equal = np.array_equal(current, expected)
                checked.append({"position": position, "layer": layer, "equal_after_fp16": bool(equal),
                                "max_abs_difference": float(np.max(np.abs(current.astype(np.float32) - expected.astype(np.float32))))})
                if not equal:
                    raise PatchingInvariantError(f"Historical hidden parity failed: {row['case_id']} {key}")
    return {"hidden_file": str(hidden_path), "checks": checked, "all_equal": True}


def _parity(row: dict[str, Any], clean: dict[str, Any], generated_digit: str) -> dict[str, Any]:
    expected_logits = np.asarray(row["class_logits"], dtype=np.float64)
    actual_logits = np.asarray(clean["class_logits"], dtype=np.float64)
    logit_delta = float(np.max(np.abs(expected_logits - actual_logits)))
    soft_delta = abs(float(row["soft_sa_image_score"]) - float(clean["soft_sa"]))
    checks = {
        "fixed_answer_byte_exact": row["phase0_raw_answer"] == row["phase1_inserted_raw_answer"],
        "generated_digit_exact": generated_digit == str(clean["hard_class"]),
        "historical_digit_exact": generated_digit == str(row["raw_generated_class"]),
        "historical_hard_exact": int(clean["hard_class"]) == int(row["argmax_hard_class"]),
        "logits_within_tolerance": logit_delta <= LOGIT_PARITY_TOLERANCE,
        "soft_within_tolerance": soft_delta <= SOFT_PARITY_TOLERANCE,
    }
    if not all(checks.values()):
        raise PatchingInvariantError(f"Clean class/soft/logit parity failed for {row['case_id']}: {checks}")
    return {**checks, "max_abs_logit_difference": logit_delta,
            "abs_soft_difference": soft_delta,
            "logit_tolerance": LOGIT_PARITY_TOLERANCE, "soft_tolerance": SOFT_PARITY_TOLERANCE}


def _build_replacements(artifacts: Any, image_key: str, spans: dict[str, Any], *, corruption: str):
    image, text, answer = artifacts.replacements_for(image_key=image_key, spans=spans)
    replacements = {
        "image": EmbeddingReplacement("image", span_positions(spans, "IMAGE"), image),
        "text": EmbeddingReplacement("text", span_positions(spans, "TEXT_CLUE"), text),
        "answer": EmbeddingReplacement("answer", span_positions(spans, "ANSWER"), answer),
    }
    components = ("image", "text", "answer") if corruption == "all" else ("answer",)
    if corruption not in CORRUPTIONS:
        raise ValueError(f"Unsupported corruption: {corruption}")
    return [replacements[name] for name in components]


def _run_corrupt(model: Any, inputs: Any, modules: Any, language_model: Any, class_ids: Sequence[int],
                 sac: int, replacements: Any, sequence_length: int, *, empty_layer: int | None = None):
    embedding = EmbeddingReplacementHook(language_model, replacements=replacements,
                                         prefill_sequence_length=sequence_length,
                                         hidden_size=modules.hidden_size)
    empty = EmptyActivationHook(modules, layer=empty_layer, prefill_sequence_length=sequence_length) if empty_layer is not None else None
    with contextlib.ExitStack() as stack:
        stack.enter_context(embedding)
        if empty is not None:
            stack.enter_context(empty)
        logits = _forward_class_logits(model, inputs, modules, class_ids, sac)
    diagnostics = embedding.diagnostics()
    if diagnostics["hook_count"] != 1:
        raise PatchingInvariantError(f"Embedding hook fired {diagnostics['hook_count']} times; expected 1")
    if empty is not None:
        empty.validate()
    return logits, diagnostics


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    capture_dir = args.capture_dir.resolve()
    output_dir = (args.output_dir or _default_output()).resolve()
    model_path = args.model_path.resolve()
    dataset = args.dataset.resolve()
    image_root = args.image_root.resolve() if args.image_root else None
    positions = parse_positions(args.positions)
    layers = parse_layers(args.layers)
    if args.eval_cases < 2 or args.eval_cases % 2:
        raise ValueError("--eval-cases must be a positive even number")
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be positive")
    required = [capture_dir / "results.jsonl", capture_dir.parent / "steering" / "test_manifest.jsonl", dataset, INFERENCE_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing or not model_path.is_dir():
        raise FileNotFoundError(f"Missing patching inputs: {missing}; model={model_path}")
    if image_root is not None and not image_root.is_dir():
        raise FileNotFoundError(image_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume and ((output_dir / "results.jsonl").exists() or (output_dir / "run_config.json").exists()):
        raise FileExistsError("Patching output already contains a run; use --resume or a new directory")
    evaluation, calibration = _prepare_manifests(capture_dir, output_dir, eval_cases=args.eval_cases,
                                                  seed=args.seed, resume=args.resume)
    selected = list(evaluation["selected"])
    run_cases = selected
    run_layers = layers
    bootstrap = args.bootstrap
    if args.smoke:
        run_cases = []
        for side in ("image_side", "text_side"):
            run_cases.extend([row for row in selected if row["test_side"] == side][:2])
        run_cases = sorted(run_cases, key=lambda row: str(row["case_id"]))
        run_layers = (14,)
        bootstrap = 100
    grid = [(position, layer) for position in positions for layer in run_layers]
    expected_keys = {f"{row['case_id']}|{position}|L{layer}" for row in run_cases for position, layer in grid}
    baselines_path = output_dir / "baselines.jsonl"
    results_path = output_dir / "results.jsonl"
    failures_path = output_dir / "failures.jsonl"
    if not failures_path.exists():
        atomic_jsonl(failures_path, [])
    baselines = load_jsonl_strict(baselines_path, repair_trailing=args.resume)
    results = load_jsonl_strict(results_path, repair_trailing=args.resume)
    baseline_by_case: dict[str, dict[str, Any]] = {}
    for record in baselines:
        case_id = str(record.get("case_id", ""))
        if not case_id or case_id in baseline_by_case:
            raise ValueError(f"Duplicate/empty baseline case: {case_id!r}")
        baseline_by_case[case_id] = record
    completed: set[str] = set()
    for record in results:
        key = str(record.get("cell_key", ""))
        if not key or key in completed or key not in expected_keys:
            raise ValueError(f"Duplicate/invalid result cell: {key!r}")
        completed.add(key)
    with FormalRunLock(output_dir, enabled=not args.smoke):
        inference = load_qwen_inference(str(model_path), INFERENCE_PATH)
        modules = resolve_language_modules(inference.model)
        if any(layer >= modules.num_hidden_layers for layer in layers):
            raise ValueError(f"Requested layer outside model with {modules.num_hidden_layers} layers")
        tokenizer = getattr(inference.processor, "tokenizer", inference.processor)
        class_ids = class_token_ids(tokenizer)
        language_model = resolve_language_model(inference.model)
        artifacts = build_or_load_artifacts(artifact_dir=output_dir / "corruption_embeddings",
                                            calibration=calibration, inference=inference,
                                            hidden_size=modules.hidden_size, image_root=image_root)
        _link_calibration_artifacts(output_dir, calibration, artifacts)
        import transformers
        immutable = {
            "format_version": FORMAT_VERSION,
            "experiment": "independent_delayed_sa_activation_patching",
            "capture_dir": str(capture_dir), "dataset": str(dataset),
            "dataset_sha256": sha256_file(dataset), "model_path": str(model_path),
            "model_processor_sha256": _model_files(model_path),
            "transformers_version": transformers.__version__,
            "positions": list(positions), "layers": list(layers),
            "active_layers": list(run_layers), "eval_cases": args.eval_cases,
            "active_case_count": len(run_cases), "seed": args.seed,
            "bootstrap": bootstrap, "smoke": bool(args.smoke),
            "corruption": args.corruption, "class_token_ids": class_ids,
            "corruption_components": (["image", "text", "answer"] if args.corruption == "all" else ["answer"]),
            "midpoints": list(MIDPOINTS),
            "evaluation_manifest_sha256": sha256_file(output_dir / "evaluation_manifest.json"),
            "calibration_manifest_sha256": sha256_file(output_dir / "calibration_manifest.json"),
            "corruption_artifact_fingerprint": artifacts.metadata["fingerprint"],
            "implementation_sha256": _implementation_hashes(),
            "git": _git_version(Path(__file__).resolve().parents[2]),
            "expected_baselines": len(run_cases), "expected_patch_cells": len(expected_keys),
            "forward_budget": len(run_cases) * (2 + len(grid)),
            "activation_site": "decoder_block_output_post_mlp_residual",
            "embedding_site": "language_model_inputs_embeds_after_vision_replacement",
        }
        fingerprint = canonical_hash(immutable)
        config = {**immutable, "fingerprint": fingerprint}
        config_path = output_dir / "run_config.json"
        if config_path.exists():
            old = json.loads(config_path.read_text())
            if old.get("fingerprint") != fingerprint:
                raise ValueError("Run configuration/fingerprint changed; refusing resume")
        else:
            atomic_json(config_path, config)
        started = time.time()
        for ordinal, row in enumerate(run_cases, 1):
            case_id = str(row["case_id"])
            _rendered, inputs, details = prepare_delayed_case(inference, row, image_root=image_root)
            spans = details["spans"]
            sequence_length = int(inputs.input_ids.shape[1])
            sac = int(spans["SAC"])
            position_indices = {position: int(spans["PANL"] if position == "P1_PANL" else spans["PANL_PLUS_1"])
                                for position in positions}
            image_key = evaluation_image_key(inference, inputs, spans, hidden_size=modules.hidden_size)
            replacements = _build_replacements(artifacts, image_key, spans, corruption=args.corruption)
            baseline = baseline_by_case.get(case_id)
            cache_path = _clean_cache_path(output_dir, case_id)
            if baseline is None:
                targets = {layer: position_indices for layer in run_layers}
                cache_hook = ResidualActivationCacheHook(modules, targets=targets, prefill_sequence_length=sequence_length)
                with cache_hook:
                    clean_logits = _forward_class_logits(inference.model, inputs, modules, class_ids, sac)
                cache_hook.validate()
                clean = score_logits(clean_logits)
                digit = tokenizer.decode([class_ids[int(clean["hard_class"])]], skip_special_tokens=False,
                                         clean_up_tokenization_spaces=False)
                parity = _parity(row, clean, digit)
                historical = _historical_parity(capture_dir, row, cache_hook.cache, run_layers)
                cache_payload = {
                    "format_version": 1, "case_id": case_id, "run_fingerprint": fingerprint,
                    "positions": position_indices, "layers": list(run_layers),
                    "hidden": {layer: {name: value.contiguous() for name, value in values.items()}
                               for layer, values in cache_hook.cache.items()},
                }
                atomic_torch_save(cache_path, cache_payload)
                if args.smoke:
                    empty = EmptyActivationHook(modules, layer=run_layers[0], prefill_sequence_length=sequence_length)
                    with empty:
                        empty_clean_logits = _forward_class_logits(inference.model, inputs, modules, class_ids, sac)
                    empty.validate()
                    if empty_clean_logits != clean_logits:
                        raise PatchingInvariantError("Empty-hook clean parity failed")
                corrupt_logits, embedding_diagnostics = _run_corrupt(
                    inference.model, inputs, modules, language_model, class_ids, sac,
                    replacements, sequence_length,
                )
                if args.smoke:
                    no_patch_logits, _ = _run_corrupt(
                        inference.model, inputs, modules, language_model, class_ids, sac,
                        replacements, sequence_length, empty_layer=run_layers[0],
                    )
                    if no_patch_logits != corrupt_logits:
                        raise PatchingInvariantError("Corrupted no-patch parity failed")
                corrupt = score_logits(corrupt_logits, clean_class=int(clean["hard_class"]))
                clean = score_logits(clean_logits, clean_class=int(clean["hard_class"]))
                baseline = {
                    "case_id": case_id, "item_id": str(row["item_id"]), "test_side": row["test_side"],
                    "condition": row["condition"], "status": "completed",
                    "clean": clean, "corrupt": corrupt,
                    "clean_generated_digit": digit, "clean_parity": parity,
                    "historical_hidden_parity": historical,
                    "embedding_diagnostics": embedding_diagnostics,
                    "image_shape_key": image_key,
                    "span_lengths": {name: len(span_positions(spans, name)) for name in ("IMAGE", "TEXT_CLUE", "ANSWER")},
                    "spans": {name: spans[name] for name in ("IMAGE", "TEXT_CLUE", "ANSWER", "PANL", "PANL_PLUS_1", "SAC")},
                    "input_fingerprints": {key: details[key] for key in (
                        "messages_hash", "rendered_hash", "image_sha256", "question_hash", "text_clue_hash", "answer_hash")},
                }
                baselines = atomic_append_jsonl(baselines_path, baselines, baseline)
                baseline_by_case[case_id] = baseline
            else:
                if not cache_path.is_file():
                    raise ValueError(f"Baseline exists without clean cache: {case_id}")
                if baseline.get("image_shape_key") != image_key or baseline.get("spans", {}).get("PANL") != spans["PANL"]:
                    raise ValueError(f"Baseline token/image fingerprint changed: {case_id}")
            cache_payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            if cache_payload.get("run_fingerprint") != fingerprint:
                raise ValueError(f"Clean cache fingerprint changed: {case_id}")
            clean_class = int(baseline["clean"]["hard_class"])
            for position, layer in grid:
                cell_key = f"{case_id}|{position}|L{layer}"
                if cell_key in completed:
                    continue
                source = cache_payload["hidden"][layer][position]
                embedding = EmbeddingReplacementHook(language_model, replacements=replacements,
                                                     prefill_sequence_length=sequence_length,
                                                     hidden_size=modules.hidden_size)
                patch = ActivationReplacementHook(modules, layer=layer, position=position_indices[position],
                                                  source_hidden=source, prefill_sequence_length=sequence_length)
                with contextlib.ExitStack() as stack:
                    stack.enter_context(embedding)
                    stack.enter_context(patch)
                    patched_logits = _forward_class_logits(inference.model, inputs, modules, class_ids, sac)
                embedding_diag = embedding.diagnostics()
                patch_diag = patch.diagnostics()
                if embedding_diag["hook_count"] != 1 or patch_diag["hook_count"] != 1:
                    raise PatchingInvariantError("A formal hook did not fire exactly once")
                patched = score_logits(patched_logits, clean_class=clean_class)
                if not all(math.isfinite(float(value)) for value in patched_logits):
                    raise PatchingInvariantError("Patched logits contain NaN/Inf")
                cell = {
                    "cell_key": cell_key, "case_id": case_id, "item_id": str(row["item_id"]),
                    "test_side": row["test_side"], "condition": row["condition"],
                    "position": position, "token_position": position_indices[position],
                    "layer": layer, "corruption": args.corruption, "status": "completed",
                    "clean": baseline["clean"], "corrupt": baseline["corrupt"], "patched": patched,
                    "single_case_recovery": {
                        endpoint: recovery(float(baseline["clean"][endpoint]), float(baseline["corrupt"][endpoint]), float(patched[endpoint]))
                        for endpoint in ("fixed_clean_class_margin", "soft_sa", "hard_midpoint")
                    },
                    "first_token": {
                        "corrupt_changed": int(baseline["corrupt"]["hard_class"]) != clean_class,
                        "patched_changed_from_clean": int(patched["hard_class"]) != clean_class,
                        "clean_class_recovered": int(patched["hard_class"]) == clean_class,
                    },
                    "embedding_diagnostics": embedding_diag, "patch_diagnostics": patch_diag,
                }
                results = atomic_append_jsonl(results_path, results, cell)
                completed.add(cell_key)
                atomic_json(output_dir / "progress.json", {
                    "status": "running", "completed_patch_cells": len(completed),
                    "expected_patch_cells": len(expected_keys), "completed_baselines": len(baseline_by_case),
                    "expected_baselines": len(run_cases), "last_cell_key": cell_key,
                    "elapsed_seconds": time.time() - started,
                })
            del inputs, cache_payload
        if completed != expected_keys or set(baseline_by_case) != {str(row["case_id"]) for row in run_cases}:
            raise RuntimeError("Run ended with an incomplete baseline or patch grid")
        summary = {
            "status": "run_complete", "smoke": bool(args.smoke), "run_fingerprint": fingerprint,
            "baseline_count": len(baseline_by_case), "patch_cell_count": len(completed),
            "expected_forward_count": len(run_cases) * (2 + len(grid)),
            "smoke_extra_parity_forward_count": (2 * len(run_cases) if args.smoke else 0),
            "elapsed_seconds": time.time() - started,
        }
        atomic_json(output_dir / "run_completion.json", summary)
        atomic_json(output_dir / "progress.json", {**summary, "status": "run_complete"})
        del artifacts, language_model, modules, inference
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_experiment(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
