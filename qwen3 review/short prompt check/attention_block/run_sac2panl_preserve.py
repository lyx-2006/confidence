"""Short-prompt SAC->PANL blocking with all non-target weights preserved."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
SHORT_ROOT = HERE.parent
REVIEW_ROOT = SHORT_ROOT.parent
REPOSITORY_ROOT = REVIEW_ROOT.parent
for candidate in (REPOSITORY_ROOT, REVIEW_ROOT, SHORT_ROOT, HERE):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import torch

from AttentionBlock.config import MODEL_PATH
from AttentionBlock.run import class_margin
from dp_SA.attention_block.run import _forward
from dp_SA.io_utils import atomic_json, atomic_jsonl, canonical_hash, load_jsonl, sha256_file
from dp_SA.soft_score import class_token_ids
from layer_metacognition.model_adapter import resolve_language_modules
from Steering.capture import _acquire_pid
from Steering.runtime import load_qwen3_inference

from attention_block.preserve_other_weights import PreserveOtherAttentionContext, PreservedAttentionEdges
from attention_block.run import _aggregate, _atomic_csv, _case_context, _trial_path


ATTENTION_ROOT = SHORT_ROOT / "output" / "attention_block"
GATE_MANIFEST = ATTENTION_ROOT / "shared_eager_gate" / "passing_manifest.jsonl"
GATE_SUMMARY = ATTENTION_ROOT / "shared_eager_gate" / "summary.json"
OUTPUT_ROOT = ATTENTION_ROOT / "SAC2PANL_preserve"
WINDOWS = ((8, 12), (12, 16), (16, 20), (20, 24))
CONDITIONS = ("C1_main_block", "C2_source_plus_1_control")
SOURCES = {"C1_main_block": "P1_PANL", "C2_source_plus_1_control": "P1_PANL_PLUS_1"}
QUERY = "P1_SAC"
METRICS = ("delta_soft_sa", "logit_change_diff", "token_change_rate")
GROUPS = ("answer_equal_macro", "family_micro", "image_side", "text_side")
REPEATS = 2000
SEED = 42


def validate_contract() -> dict[str, Any]:
    if not GATE_MANIFEST.exists() or not GATE_SUMMARY.exists():
        raise FileNotFoundError("Completed short attention eager gate is required")
    gate = json.loads(GATE_SUMMARY.read_text(encoding="utf-8"))
    cases = load_jsonl(GATE_MANIFEST)
    ids = [row["case_id"] for row in cases]
    if gate.get("status") != "complete" or len(cases) != int(gate.get("passing_count", -1)):
        raise RuntimeError("Gate summary/manifest mismatch")
    if len(ids) != len(set(ids)) or len({str(row["item_id"]) for row in cases}) != len(cases):
        raise RuntimeError("SAC2PANL requires case- and item-disjoint gate rows")
    for row in cases:
        positions = row["positions"]
        if not (positions["PANL"]["processed_index"] < positions["SAC"]["processed_index"]):
            raise RuntimeError(f"Non-causal PANL->SAC edge: {row['case_id']}")
        if positions["PANL+1"]["processed_index"] != positions["PANL"]["processed_index"] + 1:
            raise RuntimeError(f"PANL+1 mismatch: {row['case_id']}")
    return {
        "status": "validated",
        "case_count": len(cases),
        "case_ids_unique": True,
        "item_ids_unique": True,
        "query": QUERY,
        "main_source": "P1_PANL",
        "control_source": "P1_PANL_PLUS_1",
        "windows": [list(x) for x in WINDOWS],
        "mechanism": "post_softmax_zero_without_renormalization",
        "other_weights_policy": "bitwise_unchanged",
    }


def _config(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    value = {
        "format_version": 1,
        "experiment": "SAC2PANL_preserve",
        "prompt": "short",
        "case_ids": [row["case_id"] for row in cases],
        "case_count": len(cases),
        "gate_manifest_sha256": sha256_file(GATE_MANIFEST),
        "query": QUERY,
        "main_source": "P1_PANL",
        "control_source": "P1_PANL_PLUS_1",
        "windows": [list(x) for x in WINDOWS],
        "window_semantics": "inclusive",
        "attention_implementation": "eager",
        "blocking_mechanism": "post_softmax_zero_without_renormalization",
        "non_target_weight_tolerance": 0.0,
        "metrics": list(METRICS),
        "bootstrap_repeats": REPEATS,
        "seed": SEED,
    }
    value["fingerprint"] = canonical_hash(value)
    return value


def run_experiment(
    runtime: Any, cases: Sequence[dict[str, Any]], *, resume: bool, output_root: Path
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    config = _config(cases)
    config_path = output_root / "run_config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old.get("fingerprint") != config["fingerprint"]:
            raise ValueError("SAC2PANL preserve config changed")
        if not resume:
            raise FileExistsError("Output exists; use --resume")
    else:
        atomic_json(config_path, config)

    modules = resolve_language_modules(runtime.model)
    if modules.num_hidden_layers != 36 or runtime.model.config._attn_implementation != "eager":
        raise RuntimeError("Expected 36-layer Qwen3 eager attention")
    token_ids = class_token_ids(runtime.processor.tokenizer)
    new_forwards = 0
    started = time.time()
    for ordinal, row in enumerate(cases, 1):
        inputs, located, positions = _case_context(runtime, row)
        clean_path = _trial_path(output_root, row["case_id"], "C0_clean")
        if clean_path.exists():
            clean = json.loads(clean_path.read_text(encoding="utf-8"))
            if clean["fingerprint"] != config["fingerprint"]:
                raise ValueError("Clean trial fingerprint mismatch")
        else:
            eager = row["eager_clean"]
            clean_class = int(eager["argmax_hard_class"])
            clean = {
                "status": "completed", "experiment": "SAC2PANL_preserve",
                "case_id": row["case_id"], "family_id": row["family_id"], "item_id": row["item_id"],
                "test_answer": row["test_answer"], "test_side": row["test_side"], "condition": "C0_clean",
                "window_start": None, "window_end": None,
                "clean_class_logits": eager["class_logits"], "blocked_class_logits": eager["class_logits"],
                "clean_soft_sa": eager["soft_sa_image_score"], "blocked_soft_sa": eager["soft_sa_image_score"],
                "clean_hard_sa_class": clean_class, "blocked_hard_sa_class": clean_class,
                "clean_margin": class_margin(eager["class_logits"], clean_class),
                "blocked_margin": class_margin(eager["class_logits"], clean_class),
                "delta_soft_sa": 0.0, "logit_change_diff": 0.0, "token_change_rate": 0.0,
                "token_changed": False, "positions": located, "attention_diagnostics": None,
                "fingerprint": config["fingerprint"],
            }
            atomic_json(clean_path, clean)
        for window in WINDOWS:
            for condition in CONDITIONS:
                path = _trial_path(output_root, row["case_id"], condition, window)
                if path.exists():
                    if resume:
                        continue
                    raise FileExistsError(path)
                source_name = SOURCES[condition]
                edge = PreservedAttentionEdges(((positions[QUERY], positions[source_name]),))
                with PreserveOtherAttentionContext(
                    layer_indices=range(window[0], window[1] + 1),
                    edges=edge,
                    sequence_length=int(inputs.input_ids.shape[1]),
                ) as context:
                    logits, score = _forward(runtime.model, inputs, positions[QUERY], token_ids)
                diagnostics = context.diagnostics()
                clean_class = int(clean["clean_hard_sa_class"])
                blocked_class = int(score["argmax_hard_class"])
                blocked_margin = class_margin(logits, clean_class)
                delta = float(score["soft_sa_image_score"]) - float(clean["clean_soft_sa"])
                atomic_json(path, {
                    **{key: clean[key] for key in ("case_id", "family_id", "item_id", "test_answer", "test_side")},
                    "status": "completed", "experiment": "SAC2PANL_preserve", "condition": condition,
                    "window_start": window[0], "window_end": window[1], "query_name": QUERY,
                    "source_name": source_name, "query_index": positions[QUERY],
                    "source_index": positions[source_name], "clean_class_logits": clean["clean_class_logits"],
                    "blocked_class_logits": logits, "clean_soft_sa": clean["clean_soft_sa"],
                    "blocked_soft_sa": float(score["soft_sa_image_score"]),
                    "clean_hard_sa_class": clean_class, "blocked_hard_sa_class": blocked_class,
                    "clean_margin": clean["clean_margin"], "blocked_margin": blocked_margin,
                    "delta_soft_sa": delta, "token_changed": blocked_class != clean_class,
                    "token_change_rate": float(blocked_class != clean_class),
                    "logit_change_diff": float(clean["clean_margin"]) - blocked_margin,
                    "positions": located, "attention_diagnostics": diagnostics,
                    "fingerprint": config["fingerprint"],
                })
                new_forwards += 1
        atomic_json(output_root / "progress.json", {
            "status": "running", "completed_cases": ordinal, "total_cases": len(cases),
            "new_gpu_forwards": new_forwards, "elapsed_seconds": time.time() - started,
        })
    trials = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((output_root / "artifacts" / "trials").glob("*.json"))]
    expected = len(cases) * (1 + 2 * len(WINDOWS))
    if len(trials) != expected:
        raise RuntimeError(f"SAC2PANL preserve incomplete: {len(trials)}/{expected}")
    atomic_jsonl(output_root / "artifacts" / "trials.jsonl", trials)
    result = {"status": "complete", "case_count": len(cases), "trial_count": len(trials),
              "expected_trials": expected, "new_gpu_forwards": new_forwards}
    atomic_json(output_root / "run_summary.json", result)
    atomic_json(output_root / "progress.json", {**result, "completed_cases": len(cases), "total_cases": len(cases)})
    return result


def analyze(cases: Sequence[dict[str, Any]], *, output_root: Path) -> dict[str, Any]:
    trials = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((output_root / "artifacts" / "trials").glob("*.json"))]
    trials = [row for row in trials if row["condition"] in CONDITIONS]
    manifest = {str(row["family_id"]): row for row in cases}
    effects: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    counter = 0
    for window in WINDOWS:
        window_rows = [r for r in trials if (int(r["window_start"]), int(r["window_end"])) == window]
        lookup = {(r["case_id"], r["condition"]): r for r in window_rows}
        for condition in CONDITIONS:
            cell = [r for r in window_rows if r["condition"] == condition]
            for metric in METRICS:
                values = {str(r["family_id"]): float(r[metric]) for r in cell}
                for group in GROUPS:
                    effects.append({"prompt": "short", "window_start": window[0], "window_end": window[1],
                                    "condition": condition, "metric": metric, "group": group,
                                    **_aggregate(values, manifest, group, SEED + counter)})
                    counter += 1
        for metric in METRICS:
            values = {}
            for case in cases:
                main = lookup[case["case_id"], CONDITIONS[0]]
                control = lookup[case["case_id"], CONDITIONS[1]]
                values[str(case["family_id"])] = float(main[metric]) - float(control[metric])
            for group in GROUPS:
                paired.append({"prompt": "short", "window_start": window[0], "window_end": window[1],
                               "comparison": "main_minus_PANL_plus_1", "metric": metric, "group": group,
                               **_aggregate(values, manifest, group, SEED + 10000 + counter)})
                counter += 1
    _atomic_csv(output_root / "tables" / "condition_effects.csv", effects)
    _atomic_csv(output_root / "tables" / "paired_main_vs_control.csv", paired)
    _plot(effects, output_root=output_root)
    result = {"status": "complete", "case_count": len(cases), "effect_rows": len(effects),
              "paired_rows": len(paired), "bootstrap_repeats": REPEATS}
    atomic_json(output_root / "analysis_summary.json", result)
    return result


def _plot(effects: list[dict[str, Any]], *, output_root: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = output_root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for axis, metric in zip(axes, METRICS):
        for condition, label in ((CONDITIONS[0], "SAC→PANL block"), (CONDITIONS[1], "SAC→PANL+1 control")):
            rows = sorted([r for r in effects if r["metric"] == metric and r["group"] == "answer_equal_macro"
                           and r["condition"] == condition], key=lambda r: r["window_start"])
            means = [r["mean"] for r in rows]
            yerr = [[m-r["ci95_low"] for m, r in zip(means, rows)], [r["ci95_high"]-m for m, r in zip(means, rows)]]
            axis.errorbar(range(len(rows)), means, yerr=yerr, marker="o", capsize=3, label=label)
        axis.axhline(0, color="black", lw=.8)
        axis.set_xticks(range(len(rows)), [f"L{r['window_start']}–{r['window_end']}" for r in rows])
        axis.set_title(metric)
        axis.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Short prompt SAC2PANL: main and PANL+1 control")
    fig.tight_layout()
    fig.savefig(output / "short_main_vs_PANL_plus_1.png", dpi=220)
    plt.close(fig)


def run_all(*, resume: bool, smoke: bool = False) -> dict[str, Any]:
    contract = validate_contract()
    cases = load_jsonl(GATE_MANIFEST)
    output_root = OUTPUT_ROOT / "smoke" if smoke else OUTPUT_ROOT
    if smoke:
        cases = [next(row for row in cases if row["test_side"] == side) for side in ("text_side", "image_side")]
    output_root.mkdir(parents=True, exist_ok=True)
    pid = output_root / "active.pid"
    _acquire_pid(pid, "Short SAC2PANL preserve block")
    runtime = None
    try:
        runtime = load_qwen3_inference(MODEL_PATH, attn_implementation="eager")
        run = run_experiment(runtime, cases, resume=resume, output_root=output_root)
        analysis = analyze(cases, output_root=output_root)
        result = {"status": "complete", "smoke": smoke, "contract": contract, "run": run, "analysis": analysis}
        atomic_json(output_root / "completion.json", result)
        return result
    finally:
        if runtime is not None:
            del runtime
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if pid.exists() and pid.read_text(encoding="utf-8").strip() == str(os.getpid()):
            pid.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.check_only:
        print(json.dumps(validate_contract(), ensure_ascii=False, indent=2))
        return 0
    print(json.dumps(run_all(resume=args.resume, smoke=args.smoke), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
