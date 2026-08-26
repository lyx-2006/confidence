from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from confidence_test.dataset_utils import load_evaluation_cases
from confidence_test.runtime_imports import load_runtime
from dp_SA.io_utils import atomic_json, atomic_jsonl, canonical_hash, load_jsonl, sha256_file
from dp_SA.activation_swap.utils import model_fingerprints
from dp_SA.patching.protocol import prepare_delayed_case
from dp_SA.soft_score import class_token_ids
from layer_metacognition.model_adapter import load_qwen_inference, resolve_language_modules

from .config import (
    BOOTSTRAP_REPEATS, CONDITIONS, DATASET_PATH, FLOAT_TOLERANCE, INFERENCE_PATH, LOGIT_TOLERANCE,
    MIDPOINTS, MODEL_PATH, OUTPUT_PARENT, PRIMARY_LAYER, PRIMARY_POSITION, RECIPIENTS_PER_CELL,
    SEED, SMOKE_PER_ORIGIN, SOURCE_ROOT, SPLIT_PATH, default_run_name,
)
from .probe_runtime import prepare_phase1_condition, prompt_only_answer_audit, reconstruct_probe, run_primary_forward
from .selection import canonical_answer_pool, eligible_cases, build_unrelated_manifest, select_recipients, stable_key, unrelated_candidate_count


def _sha256_paths(paths: Sequence[Path]) -> dict[str, str]:
    return {str(path.resolve()): sha256_file(path) for path in sorted({Path(path).resolve() for path in paths}) if path.is_file()}


def _source_hashes(source_root: Path, split_path: Path, oof_path: Path) -> dict[str, Any]:
    capture = source_root / "capture"
    result_rows = [row for row in load_jsonl(capture / "results.jsonl") if row.get("status") == "completed"]
    hidden_paths = [source_root / str(row["hidden_file"]) for row in result_rows]
    files = [capture / "config.json", capture / "phase0_results.jsonl", capture / "results.jsonl", split_path, oof_path, *hidden_paths]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen Answer-force inputs: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    return {"files": _sha256_paths(files), "capture_record_count": len(result_rows), "hidden_file_count": len(hidden_paths)}


def _code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = [root / "dp_SA" / "answer_force" / name for name in ("config.py", "selection.py", "probe_runtime.py", "run.py", "analyze.py", "run_pipeline.py")] + [
        root / "dp_SA" / "prompts.py", root / "dp_SA" / "positions.py", root / "dp_SA" / "soft_score.py",
        root / "dp_SA" / "patching" / "protocol.py", root / "layer_metacognition" / "model_adapter.py",
        root / "layer_metacognition" / "conversation_builder.py",
    ]
    return _sha256_paths(paths)


def _atomic_npz(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        with open(temporary, "wb") as handle:
            np.savez(handle, **{f"{PRIMARY_POSITION}__L{PRIMARY_LAYER}": np.asarray(array, dtype=np.float16)})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _input_ids(inputs: Any) -> list[int]:
    value = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
    return value[0].detach().cpu().tolist()


def _finite_record(record: Mapping[str, Any]) -> None:
    for key in ("panl_sa", "panl_sa_clipped", "final_soft_sa"):
        if not np.isfinite(float(record[key])):
            raise ValueError(f"non-finite {key} for {record.get('trial_key')}")
    logits = np.asarray(record["final_class_logits"], dtype=float)
    hidden = np.asarray(record["panl_hidden_fp16"], dtype=float)
    if logits.shape != (9,) or not np.isfinite(logits).all() or hidden.ndim != 1 or not np.isfinite(hidden).all():
        raise ValueError(f"non-finite or malformed values for {record.get('trial_key')}")


def _load_cases(dataset: Path) -> tuple[dict[tuple[str, int], Any], list[Any]]:
    cases, _metadata = load_evaluation_cases(dataset)
    mapping = {(str(case.item_id), int(case.prior_index)): case for case in cases}
    return mapping, list(cases)


def _manifest_rows(
    source_root: Path, dataset: Path, split_path: Path, tokenizer: Any, *, seed: int, smoke: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows = load_jsonl(source_root / "capture" / "results.jsonl")
    cases_by_key, cases = _load_cases(dataset)
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    split = {str(key): int(value) for key, value in split_payload["item_to_fold"].items()}
    for row in rows:
        if str(row.get("item_id")) in split:
            row["split_fold"] = split[str(row["item_id"])]
    eligible, excluded = eligible_cases(rows, cases_by_key, split, tokenizer=tokenizer)
    pool = canonical_answer_pool(cases)
    token_lengths = {str(row["answer"]): len(tokenizer.encode(str(row["answer"]), add_special_tokens=False)) for row in pool}
    donor_ineligible_ids: list[str] = []
    donor_feasible: list[Any] = []
    for item in eligible:
        candidate = dict(item.row)
        candidate.update({"origin": item.origin, "difficulty": item.difficulty, "text_answer": item.text_answer, "image_answer": item.image_answer, "normalized_answer": item.normalized_answer, "forced_opposite_answer": item.image_answer if item.origin == "text" else item.text_answer})
        if unrelated_candidate_count(candidate, pool, tokenizer, token_lengths=token_lengths) > 0:
            donor_feasible.append(item)
        else:
            donor_ineligible_ids.append(str(item.row["case_id"]))
    eligible = donor_feasible
    selected, selection_summary = select_recipients(eligible, seed=seed, per_cell=RECIPIENTS_PER_CELL, smoke=smoke)
    selected_ids = {str(row["case_id"]) for row in selected}
    for row in excluded:
        if str(row.get("case_id")) not in selected_ids:
            row.setdefault("reasons", []).append("not_selected_by_seeded_item_matching")
    for row in eligible:
        if str(row.row["case_id"]) not in selected_ids:
            excluded.append({"case_id": row.row.get("case_id"), "item_id": row.row.get("item_id"), "prior_index": row.row.get("prior_index"), "condition": row.row.get("condition"), "reasons": ["eligible_not_selected_by_seeded_item_matching"]})
    recipient_items = {str(row["item_id"]) for row in selected}
    # Canonical pool intentionally uses every dataset item, while recipient
    # item IDs are excluded by the donor matcher.
    unrelated, unrelated_diag = build_unrelated_manifest(selected, pool, tokenizer, seed=seed)
    selection_summary.update({"eligible_record_count": len(eligible), "excluded_record_count": len(excluded), "source_record_count": len(rows), "source_item_count": len({str(row.get('item_id')) for row in rows}), "no_unrelated_donor_case_count": len(donor_ineligible_ids)})
    if len(recipient_items) != len(selected):
        raise ValueError("recipient manifest contains duplicate item IDs")
    return selected, unrelated, selection_summary, unrelated_diag, {"cases_by_key": cases_by_key, "split": split, "no_unrelated_donor_case_ids": donor_ineligible_ids}


def _base_config(
    *, output: Path, source_root: Path, dataset: Path, split_path: Path, oof_path: Path, seed: int,
    smoke: bool, recipients: Sequence[Mapping[str, Any]], unrelated: Sequence[Mapping[str, Any]], source_hashes: Mapping[str, Any], model_path: Path,
) -> dict[str, Any]:
    model_processor_sha256 = model_fingerprints(model_path)
    payload: dict[str, Any] = {
        "format_version": 1, "experiment": "delayed_sa_answer_force", "output_root": str(output.resolve()),
        "source_root": str(source_root.resolve()), "dataset": str(dataset.resolve()), "split_path": str(split_path.resolve()),
        "oof_path": str(oof_path.resolve()), "seed": int(seed), "smoke": bool(smoke), "recipients_per_cell": RECIPIENTS_PER_CELL,
        "recipient_count": len(recipients), "condition_count": len(CONDITIONS), "primary_position": PRIMARY_POSITION,
        "primary_layer": PRIMARY_LAYER, "midpoints": list(MIDPOINTS), "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "source_hashes": source_hashes, "code_hashes": _code_hashes(),
        "recipient_manifest_hash": canonical_hash(list(recipients)), "unrelated_manifest_hash": canonical_hash(list(unrelated)),
        "model_path": str(model_path.resolve()), "model_processor_sha256": model_processor_sha256,
        "inference_path": str(INFERENCE_PATH.resolve()), "inference_sha256": sha256_file(INFERENCE_PATH),
        "dataset_sha256": sha256_file(dataset),
    }
    payload["fingerprint"] = canonical_hash(payload)
    return payload


def _check_output(output: Path, config: Mapping[str, Any], *, resume: bool) -> None:
    config_path = output / "run_config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old.get("fingerprint") != config.get("fingerprint"):
            raise ValueError("run fingerprint changed; refusing resume")
        if not resume:
            raise FileExistsError(f"Answer-force output exists: {output}; pass --resume")
    elif any(path.name != "probe_reconstruction_audit.json" for path in output.iterdir()) and not resume:
        raise FileExistsError(f"Answer-force output directory is non-empty: {output}")
    else:
        atomic_json(config_path, config)


def _write_frozen(path: Path, rows: Sequence[Mapping[str, Any]], *, resume: bool) -> None:
    expected = [dict(row) for row in rows]
    if path.exists():
        previous = load_jsonl(path)
        if canonical_hash(previous) != canonical_hash(expected):
            raise ValueError(f"frozen manifest changed: {path}")
    else:
        atomic_jsonl(path, expected)


def _clean_parity(row: Mapping[str, Any], details: Mapping[str, Any], score: Mapping[str, Any], hidden: np.ndarray, source_root: Path) -> dict[str, Any]:
    expected_positions = row.get("positions", {})
    checks: list[dict[str, Any]] = []
    for name in ("P1_PANL", "P1_PANL_PLUS_1", "P1_SAC"):
        expected = int(expected_positions[name]["processed_index"])
        actual = int(details["located"][name]["processed_index"])
        checks.append({"name": name, "expected": expected, "actual": actual, "equal": expected == actual})
        if expected != actual:
            raise ValueError(f"clean position parity failed {row['case_id']} {name}: {actual} != {expected}")
    if list(map(int, details["located"]["phase1_answer_span"])) != list(map(int, row["phase1_answer_span"])):
        raise ValueError(f"clean answer span parity failed: {row['case_id']}")
    if list(map(int, details["located"]["phase1_answer_token_ids"])) != list(map(int, row["phase1_answer_token_ids"])):
        raise ValueError(f"clean answer token parity failed: {row['case_id']}")
    expected_logits = np.asarray(row["class_logits"], dtype=float)
    actual_logits = np.asarray(score["class_logits"], dtype=float)
    logit_diff = float(np.max(np.abs(expected_logits - actual_logits)))
    soft_diff = abs(float(score["soft_sa_image_score"]) - float(row["soft_sa_image_score"]))
    if int(score["argmax_hard_class"]) != int(row["argmax_hard_class"]):
        raise ValueError(f"clean SAC hard class parity failed: {row['case_id']}")
    if soft_diff > FLOAT_TOLERANCE or logit_diff > LOGIT_TOLERANCE:
        raise ValueError(f"clean SAC parity failed: {row['case_id']} soft={soft_diff} logits={logit_diff}")
    historical_path = source_root / str(row["hidden_file"])
    with np.load(historical_path) as payload:
        expected_hidden = np.asarray(payload[f"{PRIMARY_POSITION}__L{PRIMARY_LAYER}"], dtype=np.float16)
    actual_hidden = np.asarray(hidden, dtype=np.float16)
    hidden_equal = bool(np.array_equal(expected_hidden, actual_hidden))
    if not hidden_equal:
        raise ValueError(f"clean PANL hidden parity failed: {row['case_id']}")
    return {
        "positions_equal": True, "answer_span_equal": True, "answer_tokens_equal": True,
        "hard_class_equal": True, "soft_abs_difference": soft_diff, "logit_max_abs_difference": logit_diff,
        "hidden_fp16_equal": hidden_equal,
    }


def _result_row(
    base: Mapping[str, Any], condition: str, answer: str, details: Mapping[str, Any], hidden: np.ndarray, score: Mapping[str, Any],
    panl_sa: float, hidden_file: Path, prompt_audit: Mapping[str, Any] | None, probe_fold: int,
) -> dict[str, Any]:
    midpoint = np.asarray(MIDPOINTS, dtype=float)
    panl_value = float(panl_sa)
    row: dict[str, Any] = {
        "status": "completed", "trial_key": f"{base['case_id']}|{condition}", "case_id": base["case_id"],
        "item_id": base["item_id"], "prior_index": base["prior_index"], "condition": condition,
        "origin": base["origin"], "difficulty": base["difficulty"], "forced_direction": base["forced_direction"],
        "phase0_raw_answer": base["phase0_raw_answer"], "fixed_answer": answer,
        "text_answer": base["text_answer"], "image_answer": base["image_answer"],
        "question": base["question"], "text_clue": base["text_clue"], "image_path": base["image_path"],
        "phase1_prompt": details.get("prompt", details["rendered"]), "phase1_prompt_hash": canonical_hash(details.get("prompt", details["rendered"])),
        "phase1_rendered_hash": details["rendered_hash"],
        "phase1_answer_span": details["located"]["phase1_answer_span"], "phase1_answer_token_ids": details["located"]["phase1_answer_token_ids"],
        "positions": details["located"], "answer_token_length": len(details["located"]["phase1_answer_token_ids"]),
        "probe_fold": int(probe_fold), "panl_sa": panl_value, "panl_sa_clipped": float(np.clip(panl_value, 0.0, 1.0)),
        "panl_pseudo_hard_class": int(np.argmin(np.abs(midpoint - panl_value))),
        "final_class_logits": list(map(float, score["class_logits"])), "final_class_probabilities": list(map(float, score["class_probabilities"])),
        "final_soft_sa": float(score["soft_sa_image_score"]), "final_hard_class": int(score["argmax_hard_class"]),
        "final_hard_midpoint": float(midpoint[int(score["argmax_hard_class"])]), "hidden_file": str(hidden_file),
        "panl_hidden_fp16": np.asarray(hidden, dtype=np.float16).astype(np.float32).tolist(),
        "prompt_only_answer_audit": dict(prompt_audit or {}), "image_sha256": details["image_sha256"],
    }
    _finite_record(row)
    return row


def run_experiment(*, output_root: Path, source_root: Path = SOURCE_ROOT, dataset: Path = DATASET_PATH, split_path: Path = SPLIT_PATH,
                   model_path: Path = MODEL_PATH, seed: int = SEED, resume: bool = False, smoke: bool = False) -> dict[str, Any]:
    # Ridge/LSQR reconstruction is part of the clean parity gate.  Pin the
    # numerical backends to one thread so parity is reproducible even when a
    # parent shell exports an invalid or nondeterministic thread setting.
    os.environ.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    output = Path(output_root).resolve()
    source_root, dataset, split_path, model_path = map(Path, (source_root, dataset, split_path, model_path))
    oof_path = source_root / "probe" / "oof_predictions.jsonl"
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()) and not resume:
        raise FileExistsError(f"Answer-force output directory exists; pass --resume: {output}")
    source_hashes = _source_hashes(source_root, split_path, oof_path)
    probe, probe_audit = reconstruct_probe(source_root, split_path, oof_path)
    atomic_json(output / "probe_reconstruction_audit.json", probe_audit)
    inference = load_qwen_inference(str(model_path), inference_path=INFERENCE_PATH)
    modules = resolve_language_modules(inference.model)
    tokenizer = getattr(inference.processor, "tokenizer", inference.processor)
    token_ids = class_token_ids(tokenizer)
    selected, unrelated, selection_summary, unrelated_diag, aux = _manifest_rows(source_root, dataset, split_path, tokenizer, seed=seed, smoke=smoke)
    config = _base_config(output=output, source_root=source_root, dataset=dataset, split_path=split_path, oof_path=oof_path, seed=seed, smoke=smoke, recipients=selected, unrelated=unrelated, source_hashes=source_hashes, model_path=model_path)
    _check_output(output, config, resume=resume)
    atomic_json(output / "selection_summary.json", selection_summary)
    atomic_json(output / "unrelated_matching_diagnostics.json", unrelated_diag)
    _write_frozen(output / "recipient_manifest.jsonl", selected, resume=resume)
    _write_frozen(output / "unrelated_answer_manifest.jsonl", unrelated, resume=resume)
    input_fingerprint = {
        "run_fingerprint": config["fingerprint"],
        "source_hashes": config["source_hashes"],
        "dataset_sha256": config["dataset_sha256"],
        "model_processor_sha256": config["model_processor_sha256"],
        "inference_sha256": config["inference_sha256"],
        "recipient_manifest_hash": config["recipient_manifest_hash"],
        "unrelated_manifest_hash": config["unrelated_manifest_hash"],
        "split_sha256": config["source_hashes"]["files"].get(str(split_path.resolve())),
        "oof_sha256": config["source_hashes"]["files"].get(str(oof_path.resolve())),
    }
    atomic_json(output / "input_fingerprint.json", input_fingerprint)
    # Keep the plural spelling used by neighboring experiment packages as a
    # compatibility alias; both files carry the same immutable payload.
    atomic_json(output / "input_fingerprints.json", input_fingerprint)
    failures_path = output / "failures.jsonl"
    if not failures_path.exists():
        atomic_jsonl(failures_path, [])
    leakage_rows: list[dict[str, Any]] = []
    for recipient in selected:
        item_id = str(recipient["item_id"])
        fold = probe.fold_for_item(item_id)
        train_items = {
            str(record["item_id"])
            for record in probe.records
            if probe.item_to_fold[str(record["item_id"])] != fold
        }
        in_training = item_id in train_items
        if in_training:
            raise ValueError(f"recipient item leaked into its held-out probe training fold: {item_id}")
        leakage_rows.append({
            "case_id": str(recipient["case_id"]), "item_id": item_id,
            "held_out_fold": int(fold), "recipient_item_in_training": False,
            "answer_force_hidden_or_label_used_for_fit": False,
            "training_item_count": len(train_items),
        })
    atomic_json(output / "probe_leakage_audit.json", {"status": "passed", "records": leakage_rows})
    # Recreate exclusions independently so the frozen selection function stays
    # usable by synthetic tests without carrying mutable side channels.
    all_rows = load_jsonl(source_root / "capture" / "results.jsonl")
    split = aux["split"]
    cases_by_key = aux["cases_by_key"]
    _eligible, excluded = eligible_cases(all_rows, cases_by_key, split, tokenizer=tokenizer)
    selected_ids = {str(row["case_id"]) for row in selected}
    no_donor_ids = {str(value) for value in aux.get("no_unrelated_donor_case_ids", [])}
    excluded.extend({"case_id": row["case_id"], "item_id": row["item_id"], "prior_index": row["prior_index"], "condition": row["condition"], "reasons": ["no_unrelated_donor_available"]} for row in all_rows if str(row.get("case_id")) in no_donor_ids)
    excluded_ids = {str(item.get("case_id")) for item in excluded}
    excluded.extend({"case_id": row["case_id"], "item_id": row["item_id"], "prior_index": row["prior_index"], "condition": row["condition"], "reasons": ["eligible_not_selected_by_seeded_item_matching"]} for row in all_rows if str(row.get("status")) == "completed" and str(row.get("case_id")) not in selected_ids and str(row.get("item_id")) in split and str(row.get("case_id")) not in excluded_ids)
    excluded_sorted = sorted(
        excluded,
        key=lambda row: (
            str(row.get("item_id")),
            int(row.get("prior_index", -1)),
            str(row.get("condition")),
            str(row.get("case_id")),
        ),
    )
    atomic_jsonl(output / "excluded_records.jsonl", excluded_sorted)
    # Keep the summary consistent with the complete, auditable exclusion
    # manifest.  The manifest is record-level (including duplicate capture
    # rows), so source records = selected records + excluded records.
    selection_summary["excluded_record_count"] = len(excluded_sorted)
    if selection_summary["source_record_count"] != selection_summary["selected_count"] + selection_summary["excluded_record_count"]:
        raise ValueError("selection accounting mismatch between source, selected, and excluded records")
    atomic_json(output / "selection_summary.json", selection_summary)
    manifest_by_id = {str(row["case_id"]): row for row in selected}
    unrelated_by_id = {str(row["recipient_case_id"]): row for row in unrelated}
    completed_rows = {str(row["trial_key"]): row for row in load_jsonl(output / "results.jsonl", repair_trailing=True)}
    expected_keys = {f"{row['case_id']}|{condition}" for row in selected for condition in CONDITIONS}
    if not set(completed_rows).issubset(expected_keys):
        raise ValueError("existing Answer-force results contain trials outside frozen grid")
    started = time.time()
    for base in sorted(selected, key=stable_key):
        case_id = str(base["case_id"])
        answers = {
            "clean": str(base["phase0_raw_answer"]),
            "force_opposite": str(base["forced_opposite_answer"]),
            "force_unrelated": str(unrelated_by_id[case_id]["forced_answer"]),
        }
        clean_rendered, clean_inputs, clean_details = prepare_delayed_case(inference, dict(base))
        clean_details = {**clean_details, "prompt": base["phase1_prompt"], "answer": answers["clean"], "inputs": clean_inputs}
        for condition in CONDITIONS:
            key = f"{case_id}|{condition}"
            if key in completed_rows:
                continue
            answer = answers[condition]
            if condition == "clean":
                rendered, inputs, details = clean_rendered, clean_inputs, clean_details
            else:
                rendered, inputs, details = prepare_phase1_condition(inference, base, answer)
                details = {**details, "answer": answer, "inputs": inputs}
                prompt_only_answer_audit(
                    {"answer": answers["clean"], "rendered": clean_rendered, "inputs": clean_inputs, "details": clean_details},
                    {"answer": answer, "rendered": rendered, "inputs": inputs, "details": details},
                )
            positions = details["located"]
            hidden, score = run_primary_forward(
                inference.model, inputs, modules,
                panl_position=int(positions[PRIMARY_POSITION]["processed_index"]),
                sac_position=int(positions["P1_SAC"]["processed_index"]), class_token_ids=token_ids,
            )
            parity = _clean_parity(base, details, score, hidden, source_root) if condition == "clean" else None
            fold = probe.fold_for_item(base["item_id"])
            panl_sa = probe.predict(base["item_id"], hidden.numpy())
            hidden_path = output / "hidden" / f"{case_id}__{condition}.npz"
            _atomic_npz(hidden_path, hidden.numpy().astype(np.float16))
            result = _result_row(base, condition, answer, details, hidden.numpy(), score, panl_sa, hidden_path.relative_to(output),
                                 {"passed": True, **(parity or {})} if condition == "clean" else {"passed": True}, fold)
            completed_rows[key] = result
            atomic_jsonl(output / "results.jsonl", list(completed_rows.values()))
            atomic_json(output / "progress.json", {"status": "running", "completed_trials": len(completed_rows), "expected_trials": len(expected_keys), "last_trial_key": key, "elapsed_seconds": time.time() - started})
            del inputs
        del clean_inputs
    if set(completed_rows) != expected_keys:
        raise RuntimeError(f"Answer-force trial grid incomplete: {len(completed_rows)}/{len(expected_keys)}")
    clean_parity = [
        {
            "case_id": row["case_id"], "item_id": row["item_id"],
            **dict(row.get("prompt_only_answer_audit", {})),
        }
        for row in completed_rows.values()
        if row.get("condition") == "clean"
    ]
    if len(clean_parity) != len(selected) or not all(row.get("hidden_fp16_equal", False) for row in clean_parity):
        raise ValueError("clean parity audit is incomplete")
    atomic_json(output / "clean_parity_audit.json", {"status": "passed", "records": sorted(clean_parity, key=lambda row: str(row["case_id"]))})
    completion = {"status": "run_complete", "run_fingerprint": config["fingerprint"], "smoke": bool(smoke), "recipient_count": len(selected), "trial_count": len(completed_rows), "expected_trial_count": len(expected_keys), "elapsed_seconds": time.time() - started}
    atomic_json(output / "completion.json", completion)
    atomic_json(output / "progress.json", {**completion, "status": "run_complete"})
    del inference, modules
    gc.collect()
    return completion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Delayed-SA Answer-force experiment")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--split", type=Path, default=SPLIT_PATH)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output_root or OUTPUT_PARENT / (args.run_name or default_run_name())
    run_experiment(output_root=output, source_root=args.source_root, dataset=args.dataset, split_path=args.split, model_path=args.model_path, seed=args.seed, resume=args.resume, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
