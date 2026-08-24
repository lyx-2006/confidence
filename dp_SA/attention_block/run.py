from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Sequence

import torch

from confidence_test.runtime_imports import load_runtime
from dp_SA.io_utils import append_jsonl, atomic_json, canonical_hash, load_jsonl, sha256_file
from dp_SA.soft_score import class_token_ids, soft_sa_from_logits
from layer_metacognition.model_adapter import resolve_language_modules

from .config import (
    ALL_LAYERS, BOOTSTRAP_REPEATS, COARSE_WINDOWS, DEFAULT_DELAYED_SOURCE,
    DEFAULT_JOINT_SOURCE, DEFAULT_OUTPUT_PARENT, FAILURE_RATE_LIMIT,
    GLOBAL_CONDITIONS, INFERENCE_PATH, LOGIT_PARITY_TOLERANCE,
    MAX_CASES_PER_SIDE, MIDPOINTS, MODEL_PATH, REFINE_Q_THRESHOLD,
    REFINE_WINDOWS, ROW_SUM_TOLERANCE, SEED, SMOKE_CONDITIONS,
    SOFT_PARITY_TOLERANCE, WINDOW_CONDITIONS,
    parse_windows,
)
from .masking import AttentionBlockContext
from .sources import delayed_manifest, input_fingerprints, joint_candidates, prepare_case, select_joint_manifest
from .spans import edges_for_condition, locate_spans


def _default_output() -> Path:
    return DEFAULT_OUTPUT_PARENT / time.strftime("formal_both_seed42_w12_%Y%m%dT%H%M%SZ", time.gmtime())


def _atomic_config(path: Path, config: dict[str, Any], *, resume: bool) -> None:
    fingerprint = canonical_hash(config)
    payload = {**config, "fingerprint": fingerprint}
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("fingerprint") != fingerprint:
            raise ValueError("Run configuration fingerprint changed; refusing resume")
        if not resume:
            raise FileExistsError(f"Output already exists: {path.parent}; use --resume")
    else:
        atomic_json(path, payload)


def _manifest_rows(output: Path, joint_source: Path, delayed_source: Path, *, per_side: int, seed: int, resume: bool):
    paths = {"joint": output / "joint_case_manifest.json", "delayed": output / "delayed_case_manifest.json"}
    if all(path.exists() for path in paths.values()):
        if not resume:
            raise FileExistsError("Case manifests already exist; use --resume")
        return {arm: json.loads(path.read_text()) for arm, path in paths.items()}
    joint = select_joint_manifest(joint_candidates(joint_source), per_side=per_side, seed=seed)
    delayed = delayed_manifest(delayed_source, per_side=per_side)
    manifests = {"joint": joint, "delayed": delayed}
    for arm, rows in manifests.items():
        atomic_json(paths[arm], rows)
    joint_by_item = {str(r["item_id"]): r for r in joint}
    delayed_by_item = {str(r["item_id"]): r for r in delayed}
    shared_items = sorted(set(joint_by_item) & set(delayed_by_item), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))
    joint_keys = {(str(r["item_id"]), int(r["prior_index"]), r["condition"], r["version"]): r for r in joint}
    delayed_keys = {(str(r["item_id"]), int(r["prior_index"]), r["condition"], r["version"]): r for r in delayed}
    shared_keys = sorted(set(joint_keys) & set(delayed_keys))
    overlap = {
        "shared_item_count": len(shared_items), "shared_item_ids": shared_items,
        "shared_composite_key_count": len(shared_keys),
        "normalized_answer_match_count": sum(
            str(joint_by_item[item]["raw_answer"]).strip().casefold() == str(delayed_by_item[item]["raw_answer"]).strip().casefold()
            for item in shared_items
        ),
        "raw_answer_match_count": sum(joint_by_item[item]["raw_answer"] == delayed_by_item[item]["raw_answer"] for item in shared_items),
    }
    atomic_json(output / "cross_arm_overlap.json", overlap)
    return manifests


def _select_smoke(rows: list[dict[str, Any]], per_side: int) -> list[dict[str, Any]]:
    if per_side < 1:
        return []
    return [
        row
        for side in ("image_side", "text_side")
        for row in [candidate for candidate in rows if candidate["test_side"] == side][
            :per_side
        ]
    ]


def _margin(logits: Sequence[float], target: int) -> float:
    return float(logits[target]) - sum(float(x) for index, x in enumerate(logits) if index != target) / 8.0


def _forward(model: Any, inputs: Any, sac: int, ids: Sequence[int]) -> tuple[list[float], dict[str, Any]]:
    with torch.inference_mode():
        output = model(**inputs, use_cache=False, output_attentions=False, return_dict=True)
    vocab = output.logits[0, int(sac)]
    logits = [float(vocab[int(token)].float().item()) for token in ids]
    score = soft_sa_from_logits(vocab, ids)
    del output
    return logits, score


def _joint_replay_forward(model: Any, base_inputs: Any, forced_ids: Sequence[int], ids: Sequence[int], tokenizer: Any) -> tuple[list[float], dict[str, Any]]:
    forced = tuple(int(x) for x in forced_ids)
    base_length = int(base_inputs.input_ids.shape[1])
    def allowed(_batch_id: int, input_ids: torch.Tensor):
        step = int(input_ids.shape[-1]) - base_length
        return [forced[step]] if step < len(forced) else list(map(int, ids))
    with torch.inference_mode():
        generated = model.generate(
            **base_inputs, max_new_tokens=len(forced) + 1, do_sample=False, use_cache=True,
            return_dict_in_generate=True, output_scores=True, output_logits=True,
            prefix_allowed_tokens_fn=allowed,
        )
    replayed = [int(x) for x in generated.sequences[0, base_length:base_length + len(forced)].tolist()]
    if replayed != list(forced):
        raise RuntimeError(f"Joint teacher-force replay failed: {replayed} != {list(forced)}")
    vocab = generated.logits[-1][0]
    logits = [float(vocab[int(token)].float().item()) for token in ids]
    score = soft_sa_from_logits(vocab, ids)
    del generated
    return logits, score


def _phase_grid(phase: str, selected_pairs: dict[str, list[str]], arm: str, smoke: bool,
                coarse_windows=COARSE_WINDOWS, refine_windows=REFINE_WINDOWS):
    if smoke:
        smoke_windows = (coarse_windows[0], coarse_windows[len(coarse_windows)//2], coarse_windows[-1])
        return [(condition, start, end, None) for condition in SMOKE_CONDITIONS for start, end in smoke_windows]
    if phase == "coarse":
        return [(condition, start, end, None) for condition in WINDOW_CONDITIONS for start, end in coarse_windows]
    names = selected_pairs.get(arm, [])
    from .config import MATCHED_PAIRS
    grid = []
    for name in names:
        for condition in MATCHED_PAIRS[name]:
            grid.extend((condition, start, end, name) for start, end in refine_windows)
    return grid


def run_experiment(
    *, output_dir: Path, arm: str = "both", phase: str = "coarse",
    joint_source: Path = DEFAULT_JOINT_SOURCE, delayed_source: Path = DEFAULT_DELAYED_SOURCE,
    bootstrap_repeats: int = BOOTSTRAP_REPEATS, seed: int = SEED,
    max_cases_per_side: int = MAX_CASES_PER_SIDE, resume: bool = False,
    smoke: bool = False, selected_pairs_path: Path | None = None,
    coarse_windows=COARSE_WINDOWS, refine_windows=REFINE_WINDOWS,
    refine_q_threshold: float = REFINE_Q_THRESHOLD, auto_refine: bool = True,
) -> dict[str, Any]:
    output_dir = output_dir.resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    if len(coarse_windows) < 3:
        raise ValueError("At least three coarse windows are required for W1/middle/final smoke coverage")
    if not (0 < refine_q_threshold < 1):
        raise ValueError("refine_q_threshold must be between 0 and 1")
    arms = ["joint", "delayed"] if arm == "both" else [arm]
    per_side = 5 if smoke else max_cases_per_side
    config = {
        "format_version": 1, "model_path": str(MODEL_PATH.resolve()),
        "model_config_sha256": sha256_file(MODEL_PATH / "config.json"),
        "processor_config_sha256": sha256_file(MODEL_PATH / "preprocessor_config.json"),
        "inference_path": str(INFERENCE_PATH.resolve()), "attention_backend": "eager",
        "joint_source_dir": str(joint_source.resolve()), "delayed_source_dir": str(delayed_source.resolve()),
        "arms": arms, "seed": seed, "max_cases_per_side": max_cases_per_side,
        "bootstrap_repeats": bootstrap_repeats, "coarse_windows": coarse_windows,
        "refine_windows": refine_windows, "window_conditions": WINDOW_CONDITIONS,
        "global_conditions": GLOBAL_CONDITIONS, "matched_pairs": __import__("dp_SA.attention_block.config", fromlist=["MATCHED_PAIRS"]).MATCHED_PAIRS,
        "refine_q_threshold": refine_q_threshold, "auto_refine": auto_refine, "midpoints": MIDPOINTS, "smoke": smoke,
    }
    _atomic_config(output_dir / "run_config.json", config, resume=resume)
    manifests = _manifest_rows(output_dir, joint_source, delayed_source, per_side=max_cases_per_side, seed=seed, resume=resume)
    fingerprints = input_fingerprints(joint_source, delayed_source, manifests)
    fingerprint_path = output_dir / "input_fingerprints.json"
    if fingerprint_path.exists() and json.loads(fingerprint_path.read_text()).get("fingerprint") != fingerprints["fingerprint"]:
        raise ValueError("Input fingerprints changed; refusing resume")
    if not fingerprint_path.exists(): atomic_json(fingerprint_path, fingerprints)
    selected_pairs = {"joint": [], "delayed": []}
    if selected_pairs_path and selected_pairs_path.exists():
        selected_pairs = json.loads(selected_pairs_path.read_text())["selected_pairs"]
    if phase == "refine":
        refine_plan = {
            "selected_pairs": {name: list(selected_pairs.get(name, [])) for name in arms},
            "refine_windows": refine_windows,
        }
        refine_plan["fingerprint"] = canonical_hash(refine_plan)
        refine_plan_path = output_dir / "refine_plan.json"
        if refine_plan_path.exists():
            previous = json.loads(refine_plan_path.read_text())
            if previous.get("fingerprint") != refine_plan["fingerprint"]:
                raise ValueError("Selected refine pairs changed; refusing resume")
        else:
            atomic_json(refine_plan_path, refine_plan)
    if phase == "refine" and not any(selected_pairs.get(a) for a in arms):
        return {"status": "skipped", "phase": phase, "reason": "no_selected_pairs"}

    runtime = load_runtime(INFERENCE_PATH)
    inference = runtime.QwenVLInference(str(MODEL_PATH))
    if getattr(inference.model.config, "_attn_implementation", None) != "eager":
        raise RuntimeError("Formal blocking requires attn_implementation='eager'")
    modules = resolve_language_modules(inference.model)
    if modules.num_hidden_layers != 28:
        raise RuntimeError(f"Expected 28 language layers, found {modules.num_hidden_layers}")
    tokenizer = getattr(inference.processor, "tokenizer", inference.processor)
    ids = class_token_ids(tokenizer)
    if len(set(ids)) != 9:
        raise RuntimeError(f"SA classes are not nine unique tokens: {ids}")

    clean_path = output_dir / "clean_baselines.jsonl"
    blocked_path = output_dir / "blocked_results.jsonl"
    failures_path = output_dir / "failures.jsonl"
    spans_paths = {a: output_dir / f"{a}_token_spans.jsonl" for a in arms}
    for path in (clean_path, blocked_path, failures_path, *spans_paths.values()):
        path.touch(exist_ok=True)
    clean = {(r["arm"], r["case_id"]): r for r in load_jsonl(clean_path)}
    completed = {
        (r["arm"], r["case_id"], r["phase"], r["condition"], int(r["window_start"]), r.get("refine_pair"))
        for r in load_jsonl(blocked_path)
    }
    stored_spans = {
        arm_name: {row["case_id"]: row for row in load_jsonl(spans_paths[arm_name])}
        for arm_name in arms
    }
    grids = {a: _phase_grid(phase, selected_pairs, a, smoke, coarse_windows, refine_windows) for a in arms}
    cases_by_arm = {a: _select_smoke(manifests[a], per_side) if smoke else manifests[a] for a in arms}
    total = sum(len(cases_by_arm[a]) * len(grids[a]) for a in arms)
    if phase == "coarse": total += sum(len(cases_by_arm[a]) for a in arms)
    if phase == "coarse" and not smoke: total += sum(len(cases_by_arm[a]) * len(GLOBAL_CONDITIONS) for a in arms)
    recent: deque[float] = deque(maxlen=200); started = time.time(); done_this_run = 0; failed = 0
    try:
        for current_arm in arms:
            for row in cases_by_arm[current_arm]:
                rendered, inputs = prepare_case(inference, row)
                spans = locate_spans(tokenizer, rendered, inputs, row)
                previous_span = stored_spans[current_arm].get(row["case_id"])
                span_record = {"arm": current_arm, "case_id": row["case_id"], **spans}
                if previous_span is not None and canonical_hash(previous_span) != canonical_hash(span_record):
                    raise RuntimeError(f"Token spans changed on resume for {row['case_id']}")
                if previous_span is None:
                    append_jsonl(spans_paths[current_arm], span_record)
                    stored_spans[current_arm][row["case_id"]] = span_record
                clean_key = (current_arm, row["case_id"])
                baseline = clean.get(clean_key)
                if baseline is None:
                    before = time.perf_counter()
                    logits, score = _forward(inference.model, inputs, spans["SAC"], ids)
                    recent.append(time.perf_counter() - before)
                    expected_logits = [float(x) for x in row["class_logits"]]
                    expected_soft = float(row["soft_sa_image_score"])
                    if int(score["argmax_hard_class"]) != int(row["argmax_hard_class"]):
                        raise RuntimeError(
                            f"Clean argmax parity failed for {row['case_id']}: "
                            f"teacher_forced={score['argmax_hard_class']} stored={row['argmax_hard_class']} "
                            f"teacher_logits={logits} stored_logits={expected_logits}"
                        )
                    if abs(float(score["soft_sa_image_score"]) - expected_soft) > SOFT_PARITY_TOLERANCE:
                        raise RuntimeError(f"Clean soft-SA parity failed for {row['case_id']}: {score['soft_sa_image_score']} != {expected_soft}")
                    target = int(score["argmax_hard_class"])
                    baseline = {
                        "arm": current_arm, "case_id": row["case_id"], "item_id": str(row["item_id"]),
                        "test_side": row["test_side"], "class_logits": logits, **score,
                        "clean_class": target, "clean_margin": _margin(logits, target),
                        "raw_answer": row["raw_answer"], "raw_output": row["raw_output"],
                        "parity_diagnostics": {
                            "expected_argmax": int(row["argmax_hard_class"]),
                            "argmax_exact": True,
                            "max_abs_logit_difference": max(abs(a-b) for a,b in zip(logits,expected_logits)),
                            "abs_soft_sa_difference": abs(float(score["soft_sa_image_score"]) - expected_soft),
                            "logit_tolerance": LOGIT_PARITY_TOLERANCE,
                            "soft_sa_tolerance": SOFT_PARITY_TOLERANCE,
                        },
                    }
                    append_jsonl(clean_path, baseline); clean[clean_key] = baseline; done_this_run += 1
                grid = list(grids[current_arm])
                if phase == "coarse" and not smoke:
                    grid += [(condition, 0, 27, None) for condition in GLOBAL_CONDITIONS]
                for condition, start_layer, end_layer, refine_pair in grid:
                    key = (current_arm, row["case_id"], phase, condition, start_layer, refine_pair)
                    if key in completed: continue
                    edges = edges_for_condition(spans, condition)
                    before = time.perf_counter()
                    with AttentionBlockContext(
                        modules.language_layers, layer_indices=range(start_layer, end_layer + 1), edges=edges,
                        sequence_length=spans["sequence_length"], row_sum_tolerance=ROW_SUM_TOLERANCE,
                    ) as blocking:
                        logits, score = _forward(inference.model, inputs, spans["SAC"], ids)
                    elapsed = time.perf_counter() - before; recent.append(elapsed)
                    diagnostics = blocking.diagnostics()
                    target = int(baseline["clean_class"])
                    blocked_margin = _margin(logits, target)
                    by_layer = diagnostics["by_layer"].values()
                    result = {
                        "arm": current_arm, "case_id": row["case_id"], "item_id": str(row["item_id"]),
                        "test_side": row["test_side"], "phase": phase, "condition": condition,
                        "refine_pair": refine_pair,
                        "window_start": start_layer, "window_end": end_layer,
                        "window_center": (start_layer + end_layer) / 2, "blocked_layer_count": end_layer - start_layer + 1,
                        "class_logits": logits, "blocked_class": int(score["argmax_hard_class"]),
                        "blocked_soft_sa": float(score["soft_sa_image_score"]), "clean_class": target,
                        "clean_margin": float(baseline["clean_margin"]), "blocked_margin": blocked_margin,
                        "logit_margin_disruption": float(baseline["clean_margin"]) - blocked_margin,
                        "first_token_changed": int(score["argmax_hard_class"]) != target,
                        "delta_soft_sa": float(score["soft_sa_image_score"]) - float(baseline["soft_sa_image_score"]),
                        "abs_delta_soft_sa": abs(float(score["soft_sa_image_score"]) - float(baseline["soft_sa_image_score"])),
                        "elapsed_seconds": elapsed,
                        "attention_diagnostics": {
                            "verified_layers": diagnostics["layers"],
                            "head_counts": sorted({int(x["head_count"]) for x in by_layer}),
                            "max_blocked_weight": max(float(x["max_blocked_weight"]) for x in by_layer),
                            "max_row_sum_error": max(float(x["max_row_sum_error"]) for x in by_layer),
                            "finite": all(bool(x["finite"]) for x in by_layer),
                            "edge_count": len(edges.pairs),
                        },
                    }
                    append_jsonl(blocked_path, result); completed.add(key); done_this_run += 1
                    completed_total = len(clean) + len(completed)
                    mean_recent = sum(recent) / len(recent)
                    remaining = max(0, total - done_this_run)
                    atomic_json(output_dir / "progress.json", {
                        "status": "running", "phase": phase, "arm": current_arm, "last_condition": condition,
                        "last_window": [start_layer, end_layer], "completed_this_run": done_this_run,
                        "expected_this_run": total, "completed_persisted": completed_total,
                        "fraction_this_run": done_this_run / max(1, total), "failed": failed,
                        "elapsed_seconds": time.time() - started, "recent_mean_forward_seconds": mean_recent,
                        "estimated_remaining_seconds": remaining * mean_recent,
                    })
                del inputs
        summary = {"status": "complete", "phase": phase, "completed_this_run": done_this_run, "failed": failed, "elapsed_seconds": time.time() - started}
        progress_path = output_dir / "progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        atomic_json(progress_path, {
            **progress, "status": "complete", "phase": phase,
            "completed_this_run": done_this_run, "expected_this_run": total,
            "fraction_this_run": 1.0, "failed": failed,
            "elapsed_seconds": time.time() - started,
            "estimated_remaining_seconds": 0.0,
        })
        if phase == "coarse" and not smoke and set(arms) == {"joint", "delayed"}:
            overlap_path = output_dir / "cross_arm_overlap.json"
            overlap = json.loads(overlap_path.read_text())
            manifests_by_item = {
                arm_name: {str(row["item_id"]): row for row in manifests[arm_name]}
                for arm_name in ("joint", "delayed")
            }
            spans_by_case = {
                arm_name: {row["case_id"]: row for row in load_jsonl(output_dir / f"{arm_name}_token_spans.jsonl")}
                for arm_name in ("joint", "delayed")
            }
            exact = 0
            for item in overlap["shared_item_ids"]:
                joint_row, delayed_row = manifests_by_item["joint"][item], manifests_by_item["delayed"][item]
                joint_span = spans_by_case["joint"][joint_row["case_id"]]
                delayed_span = spans_by_case["delayed"][delayed_row["case_id"]]
                exact += int(
                    joint_row["raw_answer"] == delayed_row["raw_answer"]
                    and joint_span["ANSWER"] == delayed_span["ANSWER"]
                    and [joint_span["token_ids"][index] for index in range(*joint_span["ANSWER"])]
                    == [delayed_span["token_ids"][index] for index in range(*delayed_span["ANSWER"])]
                )
            overlap["raw_answer_and_answer_token_span_exact_match_count"] = exact
            atomic_json(overlap_path, overlap)
        atomic_json(output_dir / f"{phase}_completion.json", summary)
        return summary
    finally:
        del inference
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("joint", "delayed", "both"), default="both")
    parser.add_argument("--phase", choices=("coarse", "refine"), default="coarse")
    parser.add_argument("--joint-source-dir", type=Path, default=DEFAULT_JOINT_SOURCE)
    parser.add_argument("--delayed-source-dir", type=Path, default=DEFAULT_DELAYED_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-repeats", type=int, default=BOOTSTRAP_REPEATS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-cases-per-side", type=int, default=MAX_CASES_PER_SIDE)
    parser.add_argument("--selected-pairs", type=Path)
    parser.add_argument("--coarse-windows", type=parse_windows, default=COARSE_WINDOWS)
    parser.add_argument("--refine-windows", type=parse_windows, default=REFINE_WINDOWS)
    parser.add_argument("--refine-q-threshold", type=float, default=REFINE_Q_THRESHOLD)
    parser.add_argument("--no-auto-refine", dest="auto_refine", action="store_false")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    run_experiment(
        output_dir=args.output_dir or _default_output(), arm=args.arm, phase=args.phase,
        joint_source=args.joint_source_dir, delayed_source=args.delayed_source_dir,
        bootstrap_repeats=args.bootstrap_repeats, seed=args.seed,
        max_cases_per_side=args.max_cases_per_side, resume=args.resume, smoke=args.smoke,
        selected_pairs_path=args.selected_pairs,
        coarse_windows=args.coarse_windows, refine_windows=args.refine_windows,
        refine_q_threshold=args.refine_q_threshold, auto_refine=args.auto_refine,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
