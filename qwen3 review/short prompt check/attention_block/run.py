from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
SHORT_ROOT = HERE.parent
REVIEW_ROOT = SHORT_ROOT.parent
REPOSITORY_ROOT = REVIEW_ROOT.parent
for candidate in (REPOSITORY_ROOT, REVIEW_ROOT, SHORT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np
import torch

from AttentionBlock.config import CLEAN_LOGIT_TOLERANCE, CLEAN_SOFT_SA_TOLERANCE, EXPERIMENTS, MODEL_PATH, ROW_SUM_TOLERANCE
from AttentionBlock.contracts import add_cle_plus_1
from AttentionBlock.run import class_margin
from dp_SA.attention_block.masking import AttentionBlockContext, AttentionEdges
from dp_SA.attention_block.run import _forward
from dp_SA.io_utils import append_jsonl, atomic_json, atomic_jsonl, canonical_hash, load_jsonl, sha256_file
from dp_SA.soft_score import class_token_ids
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant
from layer_metacognition.model_adapter import model_input_device, resolve_language_modules
from Steering.capture import _acquire_pid, _messages
from Steering.runtime import load_qwen3_inference

from capture.positions import locate_short_phase1_positions
from capture.short_prompt import SA_PREFILL


CAPTURE_ROOT = SHORT_ROOT / "output" / "capture"
OUTPUT_ROOT = SHORT_ROOT / "output" / "attention_block"
CANDIDATE_MANIFEST = REVIEW_ROOT / "AttentionBlock" / "output" / "shared_eager_selection" / "test_manifest.jsonl"
LONG_OUTPUT = REVIEW_ROOT / "AttentionBlock" / "output"
CONDITIONS = ("C1_main_block", "C2_source_plus_1_control")
REPEATS = 2000
SEED = 42
METRICS = ("delta_soft_sa", "abs_delta_soft_sa", "token_change_rate", "logit_change_diff")
GROUPS = ("answer_equal_macro", "family_micro", "image_side", "text_side")


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def _case_context(runtime: Any, row: dict[str, Any]):
    messages = _messages(row["phase1_prompt"], row["image_path"], SA_PREFILL)
    rendered = render_continued_assistant(runtime.processor, messages, SA_PREFILL)
    inputs = prepare_multimodal_inputs(runtime.processor, messages, rendered, device=model_input_device(runtime))
    located = add_cle_plus_1(
        locate_short_phase1_positions(runtime.processor.tokenizer, rendered, inputs, row["phase0_raw_answer"]),
        runtime.processor.tokenizer, inputs.input_ids,
    )
    saved_keys = {
        "P1_LAT": "LAT", "P1_PANL": "PANL", "P1_PANL_PLUS_1": "PANL+1",
        "P1_CLASS_LIST_END": "CLE", "P1_SAC": "SAC",
    }
    for internal, external in saved_keys.items():
        saved, current = row["positions"][external], located[internal]
        if (int(saved["processed_index"]), int(saved["token_id"])) != (
            int(current["processed_index"]), int(current["token_id"])
        ):
            raise RuntimeError(f"Short position parity failed: {row['case_id']} {external}")
    positions = {name: int(value["processed_index"]) for name, value in located.items()
                 if isinstance(value, dict) and "processed_index" in value}
    return inputs, located, positions


def _candidate_rows() -> list[dict[str, Any]]:
    candidates = load_jsonl(CANDIDATE_MANIFEST)
    short = {row["case_id"]: row for row in load_jsonl(CAPTURE_ROOT / "results.jsonl") if row.get("status") == "completed"}
    if len(candidates) != 100:
        raise ValueError(f"Expected 100 frozen attention candidates, found {len(candidates)}")
    result = []
    for source in candidates:
        row = short.get(source["case_id"])
        if row is None:
            raise ValueError(f"Frozen attention candidate absent from short capture: {source['case_id']}")
        result.append({**row, "family_id": source["family_id"], "test_answer": source["test_answer"],
                       "test_side": source["test_side"], "attention_selection_rank": source["selection_rank"]})
    return result


def eager_gate(runtime: Any, output_root: Path, *, resume: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = output_root / "shared_eager_gate" / "assessments.jsonl"
    existing = {row["case_id"]: row for row in load_jsonl(path)}
    ids = class_token_ids(runtime.processor.tokenizer)
    candidates = _candidate_rows()
    for row in candidates:
        if row["case_id"] in existing:
            continue
        inputs, located, positions = _case_context(runtime, row)
        logits, score = _forward(runtime.model, inputs, positions["P1_SAC"], ids)
        logit_error = max(abs(float(a) - float(b)) for a, b in zip(logits, row["class_logits"]))
        soft_error = abs(float(score["soft_sa_image_score"]) - float(row["soft_sa_image_score"]))
        hard_equal = int(score["argmax_hard_class"]) == int(row["argmax_hard_class"])
        passed = hard_equal and logit_error <= CLEAN_LOGIT_TOLERANCE and soft_error <= CLEAN_SOFT_SA_TOLERANCE
        assessment = {
            "case_id": row["case_id"], "family_id": row["family_id"], "test_side": row["test_side"],
            "passed": passed, "logit_max_abs_error": logit_error, "soft_sa_abs_error": soft_error,
            "hard_class_equal": hard_equal, "class_logits": logits,
            "soft_sa_image_score": float(score["soft_sa_image_score"]),
            "argmax_hard_class": int(score["argmax_hard_class"]), "positions": located,
        }
        append_jsonl(path, assessment); existing[row["case_id"]] = assessment
    passed = []
    excluded = []
    for row in candidates:
        assessment = existing[row["case_id"]]
        if assessment["passed"]:
            passed.append({**row, "eager_clean": assessment})
        else:
            excluded.append({**assessment, "item_id": row["item_id"], "test_answer": row["test_answer"]})
    atomic_jsonl(output_root / "shared_eager_gate" / "passing_manifest.jsonl", passed)
    atomic_jsonl(output_root / "shared_eager_gate" / "excluded.jsonl", excluded)
    summary = {
        "status": "complete", "candidate_count": len(candidates), "passing_count": len(passed),
        "excluded_count": len(excluded), "passing_side_counts": dict(Counter(row["test_side"] for row in passed)),
        "gate": {"logit_max_abs_error": CLEAN_LOGIT_TOLERANCE,
                 "soft_sa_abs_error": CLEAN_SOFT_SA_TOLERANCE, "hard_class_equal": True},
        "policy": "run_passing_intersection_without_replacement",
    }
    atomic_json(output_root / "shared_eager_gate" / "summary.json", summary)
    return passed, summary


def _trial_path(root: Path, case_id: str, condition: str, window: tuple[int, int] | None = None) -> Path:
    suffix = condition if window is None else f"{condition}__L{window[0]}-{window[1]}"
    return root / "artifacts" / "trials" / f"{case_id}__{suffix}.json"


def _edge(experiment: str, condition: str, positions: dict[str, int]) -> tuple[AttentionEdges, str, str]:
    spec = EXPERIMENTS[experiment]
    source = spec.main_source if condition == CONDITIONS[0] else spec.control_source
    return AttentionEdges(((positions[spec.query], positions[source]),)), spec.query, source


def run_experiment(runtime: Any, experiment: str, cases: Sequence[dict[str, Any]], output_root: Path, *, resume: bool) -> dict[str, Any]:
    root = output_root / experiment; root.mkdir(parents=True, exist_ok=True)
    spec = EXPERIMENTS[experiment]
    config = {
        "format_version": 1, "experiment": experiment, "prompt": "short",
        "case_ids": [row["case_id"] for row in cases], "case_count": len(cases),
        "manifest_fingerprint": canonical_hash([row["case_id"] for row in cases]),
        "query": spec.query, "main_source": spec.main_source, "control_source": spec.control_source,
        "windows": [list(value) for value in spec.windows], "window_semantics": "inclusive",
        "attention_implementation": "eager", "row_sum_tolerance": ROW_SUM_TOLERANCE,
        "bootstrap_repeats": REPEATS, "seed": SEED,
    }
    fingerprint = canonical_hash(config); config["fingerprint"] = fingerprint
    config_path = root / "run_config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old.get("fingerprint") != fingerprint: raise ValueError(f"{experiment} config changed")
        if not resume: raise FileExistsError(f"{experiment} output exists; use --resume")
    else: atomic_json(config_path, config)
    modules = resolve_language_modules(runtime.model)
    if modules.num_hidden_layers != 36 or runtime.model.config._attn_implementation != "eager":
        raise RuntimeError("Expected 36-layer Qwen3 eager attention")
    ids = class_token_ids(runtime.processor.tokenizer); new_forwards = 0; started = time.time()
    for ordinal, row in enumerate(cases, 1):
        inputs, located, positions = _case_context(runtime, row)
        clean_path = _trial_path(root, row["case_id"], "C0_clean")
        if clean_path.exists():
            clean = json.loads(clean_path.read_text(encoding="utf-8"))
            if clean["fingerprint"] != fingerprint: raise ValueError("Clean trial fingerprint mismatch")
        else:
            eager = row["eager_clean"]; clean_class = int(eager["argmax_hard_class"])
            clean = {
                "status": "completed", "experiment": experiment, "case_id": row["case_id"],
                "family_id": row["family_id"], "item_id": row["item_id"], "test_answer": row["test_answer"],
                "test_side": row["test_side"], "condition": "C0_clean", "window_start": None, "window_end": None,
                "clean_class_logits": eager["class_logits"], "blocked_class_logits": eager["class_logits"],
                "clean_soft_sa": eager["soft_sa_image_score"], "blocked_soft_sa": eager["soft_sa_image_score"],
                "clean_hard_sa_class": clean_class, "blocked_hard_sa_class": clean_class,
                "clean_margin": class_margin(eager["class_logits"], clean_class),
                "blocked_margin": class_margin(eager["class_logits"], clean_class),
                "delta_soft_sa": 0.0, "abs_delta_soft_sa": 0.0, "token_changed": False,
                "token_change_rate": 0.0, "logit_change_diff": 0.0, "positions": located,
                "attention_diagnostics": None, "fingerprint": fingerprint,
            }
            atomic_json(clean_path, clean)
        for window in spec.windows:
            for condition in CONDITIONS:
                path = _trial_path(root, row["case_id"], condition, window)
                if path.exists():
                    if resume: continue
                    raise FileExistsError(path)
                edges, query_name, source_name = _edge(experiment, condition, positions)
                with AttentionBlockContext(
                    modules.language_layers, layer_indices=range(window[0], window[1] + 1),
                    edges=edges, sequence_length=int(inputs.input_ids.shape[1]),
                    row_sum_tolerance=ROW_SUM_TOLERANCE,
                ) as context:
                    logits, score = _forward(runtime.model, inputs, positions["P1_SAC"], ids)
                diagnostics = context.diagnostics(); clean_class = int(clean["clean_hard_sa_class"])
                blocked_class = int(score["argmax_hard_class"]); blocked_margin = class_margin(logits, clean_class)
                delta = float(score["soft_sa_image_score"]) - float(clean["clean_soft_sa"])
                atomic_json(path, {
                    **{key: clean[key] for key in ("case_id", "family_id", "item_id", "test_answer", "test_side")},
                    "status": "completed", "experiment": experiment, "condition": condition,
                    "window_start": window[0], "window_end": window[1], "query_name": query_name,
                    "source_name": source_name, "query_index": positions[query_name], "source_index": positions[source_name],
                    "clean_class_logits": clean["clean_class_logits"], "blocked_class_logits": logits,
                    "clean_soft_sa": clean["clean_soft_sa"], "blocked_soft_sa": float(score["soft_sa_image_score"]),
                    "clean_hard_sa_class": clean_class, "blocked_hard_sa_class": blocked_class,
                    "clean_margin": clean["clean_margin"], "blocked_margin": blocked_margin,
                    "delta_soft_sa": delta, "abs_delta_soft_sa": abs(delta),
                    "token_changed": blocked_class != clean_class, "token_change_rate": float(blocked_class != clean_class),
                    "logit_change_diff": float(clean["clean_margin"]) - blocked_margin,
                    "positions": located, "attention_diagnostics": diagnostics, "fingerprint": fingerprint,
                }); new_forwards += 1
        atomic_json(root / "progress.json", {"status": "running", "completed_cases": ordinal,
                    "total_cases": len(cases), "new_gpu_forwards": new_forwards, "elapsed_seconds": time.time() - started})
    trials = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((root / "artifacts" / "trials").glob("*.json"))]
    expected = len(cases) * (1 + 2 * len(spec.windows))
    if len(trials) != expected: raise RuntimeError(f"{experiment} incomplete: {len(trials)}/{expected}")
    atomic_jsonl(root / "artifacts" / "trials.jsonl", trials)
    result = {"status": "complete", "case_count": len(cases), "trial_count": len(trials),
              "expected_trials": expected, "new_gpu_forwards": new_forwards}
    atomic_json(root / "run_summary.json", result); return result


def _aggregate(values: dict[str, float], manifest: dict[str, dict[str, Any]], group: str, seed: int) -> dict[str, Any]:
    if group in {"image_side", "text_side"}:
        values = {family: value for family, value in values.items() if manifest[family]["test_side"] == group}
    rng = np.random.default_rng(seed)
    if group == "answer_equal_macro":
        by_answer: dict[str, list[str]] = defaultdict(list)
        for family in sorted(values): by_answer[str(manifest[family]["test_answer"])].append(family)
        observed_parts, boots = [], []
        for answer in sorted(by_answer):
            vector = np.asarray([values[family] for family in by_answer[answer]], dtype=float)
            observed_parts.append(float(vector.mean()))
            draws = rng.integers(0, len(vector), size=(REPEATS, len(vector)))
            boots.append(vector[draws].mean(axis=1))
        observed = float(np.mean(observed_parts)); boot = np.stack(boots).mean(axis=0)
        answer_count = len(by_answer)
    else:
        vector = np.asarray([values[family] for family in sorted(values)], dtype=float)
        observed = float(vector.mean()); draws = rng.integers(0, len(vector), size=(REPEATS, len(vector)))
        boot = vector[draws].mean(axis=1); answer_count = len({manifest[f]["test_answer"] for f in values})
    low, high = np.percentile(boot, [2.5, 97.5])
    return {"mean": observed, "ci95_low": float(low), "ci95_high": float(high),
            "family_count": len(values), "answer_count": answer_count, "bootstrap_repeats": REPEATS}


def _long_trials(experiment: str, case_ids: set[str]) -> list[dict[str, Any]]:
    folder = LONG_OUTPUT / experiment / "artifacts" / "trials"
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(folder.glob("*.json"))]
    return [row for row in rows if row["case_id"] in case_ids and row["condition"] in CONDITIONS]


def analyze_experiment(experiment: str, cases: Sequence[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    root = output_root / experiment
    short = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((root / "artifacts" / "trials").glob("*.json"))]
    short = [row for row in short if row["condition"] in CONDITIONS]
    long = _long_trials(experiment, {row["case_id"] for row in cases})
    manifest = {str(row["family_id"]): row for row in cases}
    windows = EXPERIMENTS[experiment].windows
    effects, paired, comparisons = [], [], []
    counter = 0
    prompt_trials = {"short": short, "long": long}
    for prompt, trials in prompt_trials.items():
        for window in windows:
            window_rows = [row for row in trials if (int(row["window_start"]), int(row["window_end"])) == window]
            lookup = {(row["case_id"], row["condition"]): row for row in window_rows}
            for condition in CONDITIONS:
                cell = [row for row in window_rows if row["condition"] == condition]
                for metric in METRICS:
                    values = {str(row["family_id"]): abs(float(row["delta_soft_sa"])) if metric == "abs_delta_soft_sa" else float(row[metric]) for row in cell}
                    for group in GROUPS:
                        effects.append({"prompt": prompt, "window_start": window[0], "window_end": window[1],
                                        "condition": condition, "metric": metric, "group": group,
                                        **_aggregate(values, manifest, group, SEED + counter)})
                        counter += 1
            for metric in METRICS:
                values = {}
                for row in window_rows:
                    if row["condition"] != CONDITIONS[0]: continue
                    control = lookup[row["case_id"], CONDITIONS[1]]
                    main_value = abs(float(row["delta_soft_sa"])) if metric == "abs_delta_soft_sa" else float(row[metric])
                    control_value = abs(float(control["delta_soft_sa"])) if metric == "abs_delta_soft_sa" else float(control[metric])
                    values[str(row["family_id"])] = main_value - control_value
                for group in GROUPS:
                    paired.append({"prompt": prompt, "window_start": window[0], "window_end": window[1],
                                   "comparison": "main_minus_source_plus_1", "metric": metric, "group": group,
                                   **_aggregate(values, manifest, group, SEED + 10000 + counter)})
                    counter += 1
    pair_lookup = {(row["prompt"], row["window_start"], row["window_end"], row["metric"], row["group"]): row for row in paired}
    # For a valid paired CI, compute family-level (short main-control) - (long main-control), not a difference of aggregate CIs.
    for window in windows:
        short_window = {(row["case_id"], row["condition"]): row for row in short if (row["window_start"], row["window_end"]) == window}
        long_window = {(row["case_id"], row["condition"]): row for row in long if (row["window_start"], row["window_end"]) == window}
        for metric in METRICS:
            values = {}
            for case in cases:
                key = case["case_id"]
                sm, sc = short_window[key, CONDITIONS[0]], short_window[key, CONDITIONS[1]]
                lm, lc = long_window[key, CONDITIONS[0]], long_window[key, CONDITIONS[1]]
                def val(row: dict[str, Any]) -> float:
                    return abs(float(row["delta_soft_sa"])) if metric == "abs_delta_soft_sa" else float(row[metric])
                values[str(case["family_id"])] = (val(sm) - val(sc)) - (val(lm) - val(lc))
            for group in GROUPS:
                comparisons.append({"window_start": window[0], "window_end": window[1],
                                    "comparison": "short_minus_long_of_main_control_contrast",
                                    "metric": metric, "group": group,
                                    "short_point": pair_lookup["short", window[0], window[1], metric, group]["mean"],
                                    "long_point": pair_lookup["long", window[0], window[1], metric, group]["mean"],
                                    **_aggregate(values, manifest, group, SEED + 20000 + counter)})
                counter += 1
    tables = root / "tables"; _atomic_csv(tables / "condition_effects.csv", effects)
    _atomic_csv(tables / "paired_main_vs_control.csv", paired)
    _atomic_csv(tables / "long_short_paired_contrasts.csv", comparisons)
    _plots(experiment, effects, paired, comparisons, root / "figures")
    result = {"status": "complete", "experiment": experiment, "case_count": len(cases),
              "effect_rows": len(effects), "paired_rows": len(paired), "comparison_rows": len(comparisons),
              "bootstrap_repeats": REPEATS}
    atomic_json(root / "analysis_summary.json", result); return result


def _plots(experiment: str, effects: list[dict[str, Any]], paired: list[dict[str, Any]], comparisons: list[dict[str, Any]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, metric in zip(axes, ("delta_soft_sa", "abs_delta_soft_sa")):
        for condition in CONDITIONS:
            rows = sorted([row for row in effects if row["prompt"] == "short" and row["metric"] == metric
                           and row["group"] == "answer_equal_macro" and row["condition"] == condition], key=lambda row: row["window_start"])
            axis.plot(range(len(rows)), [row["mean"] for row in rows], marker="o", label=condition)
        axis.axhline(0, color="black", lw=.8); axis.set_xticks(range(len(rows)), [f"L{r['window_start']}–{r['window_end']}" for r in rows])
        axis.set_title(metric); axis.grid(axis="y", alpha=.2); axis.legend(fontsize=8)
    fig.suptitle(f"{experiment}: short main vs control"); fig.tight_layout()
    fig.savefig(output / "short_main_control_effects.png", dpi=220); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, metric in zip(axes, ("delta_soft_sa", "abs_delta_soft_sa")):
        for prompt in ("long", "short"):
            rows = sorted([row for row in paired if row["prompt"] == prompt and row["metric"] == metric
                           and row["group"] == "answer_equal_macro"], key=lambda row: row["window_start"])
            axis.plot(range(len(rows)), [row["mean"] for row in rows], marker="o", label=prompt)
        axis.axhline(0, color="black", lw=.8); axis.set_xticks(range(len(rows)), [f"L{r['window_start']}–{r['window_end']}" for r in rows])
        axis.set_title(metric); axis.grid(axis="y", alpha=.2); axis.legend()
    fig.suptitle(f"{experiment}: paired main-control long/short"); fig.tight_layout()
    fig.savefig(output / "long_short_main_control_contrast.png", dpi=220); plt.close(fig)


def run_all(*, resume: bool = False, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    output_root = output_root.resolve(); output_root.mkdir(parents=True, exist_ok=True)
    capture_config_path = CAPTURE_ROOT / "config.json"
    root_config = {
        "format_version": 1, "experiment": "qwen3_short_prompt_attention_block",
        "capture_config_sha256": sha256_file(capture_config_path),
        "candidate_manifest_sha256": sha256_file(CANDIDATE_MANIFEST),
        "candidate_count": 100, "gate_policy": "passing_intersection_without_replacement",
        "experiments": ["PANL2CLE", "CLE2SAC"], "bootstrap_repeats": REPEATS, "seed": SEED,
    }
    root_config["fingerprint"] = canonical_hash(root_config)
    root_config_path = output_root / "config.json"
    if root_config_path.exists():
        old = json.loads(root_config_path.read_text(encoding="utf-8"))
        if old.get("fingerprint") != root_config["fingerprint"]:
            raise ValueError("Short attention root config changed")
        if not resume:
            raise FileExistsError("Short attention output exists; use --resume")
    else:
        atomic_json(root_config_path, root_config)
    pid = output_root / "active.pid"; _acquire_pid(pid, "Short attention block")
    runtime = None
    try:
        runtime = load_qwen3_inference(MODEL_PATH, attn_implementation="eager")
        cases, gate = eager_gate(runtime, output_root, resume=resume)
        if not cases: raise RuntimeError("No frozen attention case passed short eager parity")
        runs = {}; analyses = {}
        for experiment in ("PANL2CLE", "CLE2SAC"):
            runs[experiment] = run_experiment(runtime, experiment, cases, output_root, resume=resume)
            analyses[experiment] = analyze_experiment(experiment, cases, output_root)
        result = {"status": "complete", "gate": gate, "runs": runs, "analyses": analyses}
        atomic_json(output_root / "completion.json", result); return result
    finally:
        if runtime is not None: del runtime
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if pid.exists() and pid.read_text(encoding="utf-8").strip() == str(os.getpid()): pid.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT); args = parser.parse_args(argv)
    print(json.dumps(run_all(resume=args.resume, output_root=args.output_root), ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
