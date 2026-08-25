from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from confidence_test.runtime_imports import load_runtime
from dp_SA.positions import locate_phase1_positions
from dp_SA.prompts import SA_PREFILL
from dp_SA.soft_score import class_token_ids
from dp_SA.patching.protocol import prepare_delayed_case
from layer_metacognition.model_adapter import (
    load_qwen_inference, model_input_device, resolve_language_modules, run_hooked_forward, run_logits_forward,
)

from .config import (
    BOOTSTRAP_REPEATS, DEFAULT_LAYERS, DEFAULT_OUTPUT_PARENT, DEFAULT_POSITIONS,
    HIDDEN_DEFINITION, HISTORY_LAYERS, MODEL_CONFIG_FILES,
    RECIPIENTS_PER_SIDE, SMOKE_BOOTSTRAP_REPEATS, SMOKE_LAYERS, SMOKE_RECIPIENTS_PER_SIDE,
    SOURCE_ROOT, parse_layers, parse_csv_strings,
)
from .hooks import EmptyHook, SwapActivationHook, SwapInvariantError
from .manifests import enrich_length, load_frozen_manifests
from .matching import build_swap_pairs
from .metrics import score_logits
from .utils import (
    append_jsonl, atomic_json, atomic_jsonl, canonical_hash, directory_hash, load_jsonl,
    model_fingerprints, sha256_file, stable_key,
)


def _default_output() -> Path:
    return DEFAULT_OUTPUT_PARENT / f"delayed_sa_swap_seed42_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Delayed-SA complete activation swap")
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--positions", default=",".join(DEFAULT_POSITIONS))
    parser.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)))
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPEATS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-recipients-per-side", type=int, default=RECIPIENTS_PER_SIDE)
    return parser


def _model_path(args: argparse.Namespace) -> Path:
    from dp_SA.config import MODEL_PATH
    return (args.model_path or MODEL_PATH).resolve()


def _dataset_path(args: argparse.Namespace) -> Path:
    from dp_SA.config import DATASET_PATH
    return (args.dataset or DATASET_PATH).resolve()


def _source_hashes(source_root: Path) -> dict[str, str]:
    paths = [
        source_root / "capture" / "config.json", source_root / "capture" / "phase0_results.jsonl",
        source_root / "capture" / "results.jsonl", source_root / "steering" / "config.json",
        source_root / "steering" / "construction_manifest.jsonl", source_root / "steering" / "test_manifest.jsonl",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen delayed-SA inputs: {missing}")
    return {str(path.relative_to(source_root)): sha256_file(path) for path in paths}


def _git_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = list(Path(__file__).resolve().parent.glob("*.py")) + [
        root / "dp_SA" / "prompts.py", root / "dp_SA" / "positions.py", root / "dp_SA" / "soft_score.py",
        root / "dp_SA" / "patching" / "protocol.py", root / "layer_metacognition" / "conversation_builder.py",
        root / "layer_metacognition" / "model_adapter.py", root / "layer_metacognition" / "sa_patching" / "sa_patching_hook.py",
    ]
    return directory_hash(paths)


def _base_config(args: argparse.Namespace, *, output: Path, source_root: Path, model_path: Path, dataset: Path,
                 positions: Sequence[str], layers: Sequence[int], recipient_count: int, donor_count: int,
                 bootstrap: int, source_hashes: dict[str, str]) -> dict[str, Any]:
    import transformers
    immutable = {
        "format_version": 1, "experiment": "delayed_sa_complete_activation_swap",
        "source_root": str(source_root), "output_root": str(output), "dataset": str(dataset),
        "dataset_sha256": sha256_file(dataset), "model_path": str(model_path),
        "model_processor_sha256": model_fingerprints(model_path), "transformers_version": transformers.__version__,
        "source_hashes": source_hashes, "positions": list(positions), "layers": list(layers),
        "hidden_definition": HIDDEN_DEFINITION, "activation_site": HIDDEN_DEFINITION,
        "class_count": 9, "metric_midpoints": [index / 8 for index in range(9)],
        "seed": int(args.seed), "bootstrap": int(bootstrap), "smoke": bool(args.smoke),
        "recipient_count": int(recipient_count), "donor_count": int(donor_count),
        "expected_swap_forwards": int(recipient_count * 2 * len(positions) * len(layers)),
        "expected_clean_forwards": int(recipient_count + donor_count),
        "code_hashes": _git_hashes(),
    }
    immutable["fingerprint"] = canonical_hash(immutable)
    return immutable


def _check_or_write_config(path: Path, config: dict[str, Any], *, resume: bool) -> None:
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("fingerprint") != config.get("fingerprint"):
            raise ValueError("run fingerprint changed; refusing resume")
        if not resume:
            raise FileExistsError(f"run already exists: {path.parent}; pass --resume")
    else:
        atomic_json(path, config)


def _write_or_compare_jsonl(path: Path, rows: Sequence[dict[str, Any]], *, resume: bool) -> None:
    if path.exists():
        previous = load_jsonl(path, repair_trailing=resume)
        if canonical_hash(previous) != canonical_hash(list(rows)):
            raise ValueError(f"frozen manifest changed: {path}")
    else:
        atomic_jsonl(path, rows)


def _prepare(inference: Any, row: dict[str, Any]) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    rendered, inputs, details = prepare_delayed_case(inference, row)
    enriched = enrich_length(row, details)
    return rendered, inputs, details, enriched


def _historical_hidden_check(source_root: Path, row: dict[str, Any], hidden: dict[str, dict[int, torch.Tensor]],
                             positions: Sequence[str], layers: Sequence[int]) -> dict[str, Any]:
    hidden_file = source_root / str(row["hidden_file"])
    if not hidden_file.is_file():
        raise SwapInvariantError(f"missing frozen hidden file: {hidden_file}")
    checks = []
    with np.load(hidden_file) as payload:
        for layer in sorted(set(layers) & set(HISTORY_LAYERS)):
            for position in positions:
                key = f"{position}__L{layer}"
                if key not in payload:
                    raise SwapInvariantError(f"frozen hidden key missing: {row['case_id']} {key}")
                current = hidden[position][layer].numpy().astype(np.float16)
                expected = np.asarray(payload[key])
                equal = bool(np.array_equal(current, expected))
                checks.append({"position": position, "layer": layer, "equal_after_fp16": equal,
                               "max_abs_difference": float(np.max(np.abs(current.astype(np.float32) - expected.astype(np.float32))))})
                if not equal:
                    raise SwapInvariantError(f"historical hidden parity failed: {row['case_id']} {key}")
    return {"hidden_file": str(hidden_file), "checks": checks, "all_equal": True}


def _old_midpoint_score(logits: Sequence[float]) -> float:
    from dp_SA.config import MIDPOINTS
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max()
    probs = np.exp(shifted); probs /= probs.sum()
    return float(np.dot(probs, np.asarray(MIDPOINTS, dtype=np.float64)))


def _clean_record(row: dict[str, Any], logits: Sequence[float], *, role: str, hidden_file: str,
                  historical: dict[str, Any], input_fingerprint: str, cache_sha256: str) -> dict[str, Any]:
    score = score_logits(logits)
    expected = np.asarray(row["class_logits"], dtype=np.float64)
    actual = np.asarray(logits, dtype=np.float64)
    logit_delta = float(np.max(np.abs(expected - actual)))
    old_soft_delta = abs(_old_midpoint_score(logits) - float(row.get("soft_sa_image_score", _old_midpoint_score(expected))))
    if logit_delta > 0.125 or old_soft_delta > 1e-6 or int(score["hard_class"]) != int(row["argmax_hard_class"]):
        raise SwapInvariantError(f"clean parity failed for {row['case_id']}: logit={logit_delta}, old_soft={old_soft_delta}")
    return {
        "status": "completed", "role": role, "case_id": row["case_id"], "item_id": row["item_id"],
        "test_side": row.get("test_side"), "construction_side": row.get("construction_side"),
        "clean": score, "clean_old_midpoint": _old_midpoint_score(logits),
        "historical_logit_max_abs_difference": logit_delta,
        "historical_old_soft_abs_difference": old_soft_delta,
        "historical_hidden_parity": historical, "input_fingerprint": input_fingerprint,
        "cache_sha256": cache_sha256,
        "image_sha256": row["image_sha256"], "phase0_raw_answer": row["phase0_raw_answer"],
        "phase1_inserted_raw_answer": row["phase1_inserted_raw_answer"],
    }


def _load_cache(path: Path, fingerprint: str, case_id: str, expected_sha256: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"clean cache missing: {case_id}")
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError(f"clean cache bytes changed: {case_id}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("run_fingerprint") != fingerprint or payload.get("case_id") != case_id:
        raise ValueError(f"clean cache fingerprint changed: {case_id}")
    return payload


def _clean_one(inference: Any, modules: Any, row: dict[str, Any], *, positions: Sequence[str], layers: Sequence[int],
               source_root: Path, output: Path, fingerprint: str, tokenizer_ids: Sequence[int],
               clean_by_case: dict[str, dict[str, Any]], cache_by_case: dict[str, dict[str, Any]]) -> None:
    case_id = str(row["case_id"])
    cache_path = output / "clean_cache" / f"{case_id}.pt"
    if case_id in clean_by_case:
        record = clean_by_case[case_id]
        if record.get("input_fingerprint") != row.get("input_fingerprint"):
            raise ValueError(f"clean record input fingerprint changed: {case_id}")
        cache_by_case[case_id] = _load_cache(cache_path, fingerprint, case_id,
                                             record.get("cache_sha256"))
        return
    _rendered, inputs, details, enriched = _prepare(inference, row)
    position_indices = {position: int(details["located"][position]["processed_index"]) for position in positions}
    forward = run_hooked_forward(inference.model, inputs, modules, position_indices,
                                 logits_positions=[int(details["located"]["P1_SAC"]["processed_index"])])
    hidden = {position: {int(layer): forward.hidden_by_name[position][int(layer)].detach().float().cpu().clone() for layer in layers}
              for position in positions}
    historical = _historical_hidden_check(source_root, row, hidden, positions, layers)
    sac_position = int(details["located"]["P1_SAC"]["processed_index"])
    logits = [float(forward.logits_by_position[sac_position][int(token_id)].item()) for token_id in tokenizer_ids]
    cache_payload = {"format_version": 1, "run_fingerprint": fingerprint, "case_id": case_id,
                     "positions": position_indices, "layers": list(map(int, layers)), "hidden": hidden}
    from dp_SA.patching.io import atomic_torch_save
    atomic_torch_save(cache_path, cache_payload)
    cache_sha256 = sha256_file(cache_path)
    record = _clean_record(enriched, logits, role="recipient" if row.get("test_side") else "donor",
                           hidden_file=str(cache_path.relative_to(output)), historical=historical,
                           input_fingerprint=enriched["input_fingerprint"], cache_sha256=cache_sha256)
    clean_by_case[case_id] = record
    cache_by_case[case_id] = cache_payload


def _swap_row(recipient: dict[str, Any], donor: dict[str, Any], pair: dict[str, Any], *, position: str, layer: int,
              target_index: int, score: dict[str, Any], hook_diag: dict[str, Any], clean: dict[str, Any],
              donor_clean: dict[str, Any]) -> dict[str, Any]:
    clean_class = int(clean["clean"]["hard_class"])
    return {
        "status": "completed", "trial_key": f"{recipient['case_id']}|{pair['condition']}|{position}|L{layer}",
        "recipient_case_id": recipient["case_id"], "recipient_item_id": recipient["item_id"],
        "donor_case_id": donor["case_id"], "donor_item_id": donor["item_id"],
        "recipient_side": recipient["test_side"], "donor_side": pair["donor_side"],
        "condition": pair["condition"], "swap_kind": pair["swap_kind"], "position": position,
        "layer": int(layer), "token_position": int(target_index), "clean_class": clean_class,
        "clean_class_logits": clean["clean"]["class_logits"],
        "clean_class_probabilities": clean["clean"]["class_probabilities"],
        "clean_soft_sa": float(clean["clean"]["soft_sa"]), "clean_hard_midpoint": float(clean["clean"]["hard_midpoint"]),
        "clean_fixed_clean_class_margin": float(clean["clean"]["fixed_clean_class_margin"]),
        "donor_clean_soft_sa": float(donor_clean["clean"]["soft_sa"]),
        "donor_clean_hard_class": int(donor_clean["clean"]["hard_class"]),
        "donor_clean_hard_midpoint": float(donor_clean["clean"]["hard_midpoint"]),
        "swap_class_logits": score["class_logits"], "swap_class_probabilities": score["class_probabilities"],
        "swap_soft_sa": float(score["soft_sa"]), "swap_hard_class": int(score["hard_class"]),
        "swap_hard_midpoint": float(score["hard_midpoint"]),
        "swap_fixed_clean_class_margin": float(score["fixed_clean_class_margin"]),
        "first_token_changed": bool(int(score["hard_class"]) != clean_class),
        "matching": {key: pair[key] for key in ("answer_token_length_equal", "question_quantile_bin_equal",
                                                  "answer_quantile_bin_equal", "normalized_answer_match",
                                                  "question_token_length_delta", "answer_token_length_delta")},
        "activation_diagnostics": hook_diag,
    }


def _run_noop(inference: Any, modules: Any, inputs: Any, *, layer: int, sequence_length: int,
              sac: int, tokenizer_ids: Sequence[int], clean_logits: Sequence[float]) -> dict[str, Any]:
    with EmptyHook(modules, layer=layer, prefill_sequence_length=sequence_length) as hook:
        actual = run_logits_forward(inference.model, inputs, [sac], modules)[sac]
    hook.validate()
    values = [float(actual[int(token)].item()) for token in tokenizer_ids]
    return {"layer": layer, "hook_count": hook.hook_count, "bitwise_logits_equal": values == list(map(float, clean_logits))}


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.resolve()
    output = (args.output_root or _default_output()).resolve()
    model_path, dataset = _model_path(args), _dataset_path(args)
    positions = parse_csv_strings(args.positions, allowed=set(DEFAULT_POSITIONS))
    layers = parse_layers(args.layers)
    bootstrap = SMOKE_BOOTSTRAP_REPEATS if args.smoke else int(args.bootstrap)
    if args.smoke:
        positions, layers = DEFAULT_POSITIONS, SMOKE_LAYERS
        recipient_limit = SMOKE_RECIPIENTS_PER_SIDE
    else:
        recipient_limit = int(args.max_recipients_per_side)
    if bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    if not source_root.is_dir() or not model_path.is_dir() or not dataset.is_file():
        raise FileNotFoundError(f"invalid source/model/dataset: {source_root}, {model_path}, {dataset}")
    output.mkdir(parents=True, exist_ok=True)
    source_hashes = _source_hashes(source_root)
    frozen_donors, frozen_recipients = load_frozen_manifests(source_root, max_recipients_per_side=recipient_limit, smoke=args.smoke)
    # Matching cutpoints are always defined on the complete frozen cohort,
    # including when the smoke run uses only two recipients per side.
    if args.smoke:
        reference_donors, reference_recipients = load_frozen_manifests(
            source_root, max_recipients_per_side=RECIPIENTS_PER_SIDE, smoke=False
        )
    else:
        reference_donors, reference_recipients = frozen_donors, frozen_recipients
    if output.joinpath("run_config.json").exists() and args.resume:
        saved = json.loads(output.joinpath("run_config.json").read_text())
        if saved.get("positions") != list(positions) or saved.get("layers") != list(layers):
            raise ValueError("resume positions/layers differ")
    failures_path = output / "failures.jsonl"
    if not failures_path.exists():
        atomic_jsonl(failures_path, [])
    inference = load_qwen_inference(str(model_path))
    modules = resolve_language_modules(inference.model)
    if any(layer >= modules.num_hidden_layers for layer in layers):
        raise ValueError(f"requested layer outside model ({modules.num_hidden_layers} layers)")
    tokenizer = getattr(inference.processor, "tokenizer", inference.processor)
    token_ids = class_token_ids(tokenizer)
    enriched_reference: dict[str, dict[str, Any]] = {}
    for row in [*reference_donors, *reference_recipients]:
        _rendered, inputs, details, enriched = _prepare(inference, row)
        enriched_reference[str(enriched["case_id"])] = enriched
        del inputs
    enriched_donors = [enriched_reference[str(row["case_id"])] for row in frozen_donors]
    enriched_recipients = [enriched_reference[str(row["case_id"])] for row in frozen_recipients]
    pairs, matching = build_swap_pairs(
        enriched_recipients, enriched_donors, seed=int(args.seed),
        bin_reference_rows=list(enriched_reference.values()),
    )
    # Preserve a single deterministic manifest for all cells and all resumes.
    if (output / "swap_pair_manifest.jsonl").exists() and args.resume:
        old_pairs = load_jsonl(output / "swap_pair_manifest.jsonl", repair_trailing=True)
        if canonical_hash(old_pairs) != canonical_hash(pairs):
            raise ValueError("donor mapping changed; refusing resume")
    else:
        atomic_jsonl(output / "swap_pair_manifest.jsonl", pairs)
    # Update donor usage and matching diagnostics only after the mapping is frozen.
    usage = matching["donor_reuse_counts"]
    for row in enriched_donors:
        row["donor_reuse_count"] = int(usage.get(str(row["case_id"]), 0))
    _write_or_compare_jsonl(output / "donor_manifest.jsonl", enriched_donors, resume=args.resume)
    _write_or_compare_jsonl(output / "recipient_manifest.jsonl", enriched_recipients, resume=args.resume)
    atomic_json(output / "matching_diagnostics.json", matching)
    active_donor_ids = {pair["donor_case_id"] for pair in pairs}
    config_donor_count = len(active_donor_ids) if args.smoke else len(enriched_donors)
    config = _base_config(args, output=output, source_root=source_root, model_path=model_path, dataset=dataset,
                          positions=positions, layers=layers, recipient_count=len(enriched_recipients),
                          donor_count=config_donor_count, bootstrap=bootstrap, source_hashes=source_hashes)
    _check_or_write_config(output / "run_config.json", config, resume=args.resume)
    fingerprint = str(config["fingerprint"])
    input_fingerprints = {
        "run_fingerprint": fingerprint, "source_hashes": source_hashes,
        "dataset_sha256": sha256_file(dataset), "model_processor_sha256": model_fingerprints(model_path),
        "recipient_manifest_hash": canonical_hash(enriched_recipients),
        "donor_manifest_hash": canonical_hash(enriched_donors), "swap_pair_manifest_hash": canonical_hash(pairs),
        "recipient_input_fingerprints": {str(row["case_id"]): row["input_fingerprint"] for row in enriched_recipients},
        "donor_input_fingerprints": {str(row["case_id"]): row["input_fingerprint"] for row in enriched_donors},
    }
    input_fingerprint_path = output / "input_fingerprints.json"
    if input_fingerprint_path.exists():
        previous_fingerprints = json.loads(input_fingerprint_path.read_text(encoding="utf-8"))
        if canonical_hash(previous_fingerprints) != canonical_hash(input_fingerprints):
            raise ValueError("input fingerprints changed; refusing resume")
    else:
        atomic_json(input_fingerprint_path, input_fingerprints)
    clean_path = output / "clean_predictions.jsonl"
    clean_records = load_jsonl(clean_path, repair_trailing=args.resume)
    clean_by_case = {str(row["case_id"]): row for row in clean_records}
    cache_by_case: dict[str, dict[str, Any]] = {}
    active_rows = [*enriched_recipients, *[row for row in enriched_donors if str(row["case_id"]) in active_donor_ids]]
    # Formal runs intentionally materialize all 50 donor clean states; smoke only materializes referenced donors.
    if not args.smoke:
        active_rows = [*enriched_recipients, *enriched_donors]
    for row in active_rows:
        _clean_one(inference, modules, row, positions=positions, layers=layers, source_root=source_root,
                   output=output, fingerprint=fingerprint, tokenizer_ids=token_ids,
                   clean_by_case=clean_by_case, cache_by_case=cache_by_case)
        atomic_jsonl(clean_path, list(clean_by_case.values()))
    if len(clean_by_case) != len(active_rows):
        raise RuntimeError("clean prediction grid incomplete")
    swap_path = output / "swap_predictions.jsonl"
    swap_records = load_jsonl(swap_path, repair_trailing=args.resume)
    completed = {str(row["trial_key"]): row for row in swap_records}
    row_by_case = {str(row["case_id"]): row for row in enriched_recipients}
    donor_by_case = {str(row["case_id"]): row for row in enriched_donors}
    pair_by_recipient: dict[str, list[dict[str, Any]]] = {}
    for pair in pairs:
        pair_by_recipient.setdefault(str(pair["recipient_case_id"]), []).append(pair)
    expected_trials = {
        f"{recipient['case_id']}|{pair['condition']}|{position}|L{layer}": {
            "recipient_case_id": str(recipient["case_id"]), "donor_case_id": str(pair["donor_case_id"]),
            "condition": str(pair["condition"]), "position": str(position), "layer": int(layer),
        }
        for recipient in enriched_recipients
        for pair in pair_by_recipient[str(recipient["case_id"])]
        for position in positions for layer in layers
    }
    if not set(completed).issubset(expected_trials):
        raise ValueError("completed swap records contain trials outside the frozen grid")
    for trial_key, record in completed.items():
        expected = expected_trials[trial_key]
        if any(record.get(field) != value for field, value in expected.items()):
            raise ValueError(f"completed swap record mapping changed: {trial_key}")
        score_logits(record.get("swap_class_logits", []), clean_class=int(record.get("clean_class", -1)))
    noop_checks = []
    started = time.time()
    for recipient in enriched_recipients:
        _rendered, inputs, details, _enriched = _prepare(inference, recipient)
        sac = int(details["located"]["P1_SAC"]["processed_index"])
        sequence_length = int(inputs.input_ids.shape[1])
        clean = clean_by_case[str(recipient["case_id"])]
        clean_logits = clean["clean"]["class_logits"]
        if args.smoke:
            noop_checks.append(_run_noop(inference, modules, inputs, layer=14, sequence_length=sequence_length,
                                         sac=sac, tokenizer_ids=token_ids, clean_logits=clean_logits))
        positions_indices = {position: int(details["located"][position]["processed_index"]) for position in positions}
        for pair in pair_by_recipient[str(recipient["case_id"])]:
            donor = donor_by_case[str(pair["donor_case_id"])]
            donor_clean = clean_by_case[str(donor["case_id"])]
            source_cache = cache_by_case[str(donor["case_id"])]
            for position in positions:
                target = positions_indices[position]
                for layer in layers:
                    trial_key = f"{recipient['case_id']}|{pair['condition']}|{position}|L{layer}"
                    if trial_key in completed:
                        continue
                    source = source_cache["hidden"][position][int(layer)]
                    hook = SwapActivationHook(modules, layer=int(layer), position=target,
                                               source_hidden=source, prefill_sequence_length=sequence_length)
                    with hook:
                        vocab = run_logits_forward(inference.model, inputs, [sac], modules)[sac]
                    diag = hook.diagnostics()
                    scored = score_logits([float(vocab[int(token_id)].item()) for token_id in token_ids],
                                          clean_class=int(clean["clean"]["hard_class"]))
                    row = _swap_row(recipient, donor, pair, position=position, layer=int(layer),
                                    target_index=target, score=scored, hook_diag=diag,
                                    clean=clean, donor_clean=donor_clean)
                    completed[trial_key] = row
                    atomic_jsonl(swap_path, list(completed.values()))
                    atomic_json(output / "progress.json", {
                        "status": "running", "completed_swap_forwards": len(completed),
                        "expected_swap_forwards": config["expected_swap_forwards"],
                        "completed_clean_forwards": len(clean_by_case), "expected_clean_forwards": config["expected_clean_forwards"],
                        "last_trial_key": trial_key, "elapsed_seconds": time.time() - started,
                    })
        del inputs
    expected_keys = set(expected_trials)
    if set(completed) != expected_keys:
        raise RuntimeError(f"swap grid incomplete: {len(completed)}/{len(expected_keys)}")
    if args.smoke:
        atomic_json(output / "smoke_gate.json", {"status": "passed", "noop_checks": noop_checks,
                                                  "all_noop_bitwise": all(item["bitwise_logits_equal"] for item in noop_checks),
                                                  "expected_swap_forwards": len(expected_keys),
                                                  "recipient_count": len(enriched_recipients),
                                                  "donor_count_used": len(active_donor_ids)})
    completion = {"status": "run_complete", "run_fingerprint": fingerprint, "smoke": bool(args.smoke),
                  "recipient_count": len(enriched_recipients), "donor_count": len(enriched_donors),
                  "donor_count_used": len(active_donor_ids), "clean_forward_count": len(clean_by_case),
                  "swap_forward_count": len(completed), "expected_clean_forward_count": config["expected_clean_forwards"],
                  "expected_swap_forward_count": config["expected_swap_forwards"], "noop_checks": noop_checks}
    atomic_json(output / "run_completion.json", completion)
    atomic_json(output / "progress.json", {**completion, "status": "run_complete"})
    del inference, modules
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return completion


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = (args.output_root or _default_output()).resolve()
    try:
        run_experiment(args)
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=True)
        path = output / "failures.jsonl"
        rows = load_jsonl(path, repair_trailing=True)
        rows.append({"status": "failed", "error_type": type(exc).__name__, "error": str(exc), "timestamp": time.time()})
        atomic_jsonl(path, rows)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
