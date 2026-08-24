from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from confidence_test.runtime_imports import load_runtime
from dp_SA.attention_block.analyze import bh_fdr
from dp_SA.attention_block.config import INFERENCE_PATH, MODEL_PATH, ROW_SUM_TOLERANCE
from dp_SA.attention_block.masking import AttentionBlockContext, AttentionEdges
from dp_SA.attention_block.run import _forward, _margin
from dp_SA.io_utils import append_jsonl, atomic_json, atomic_jsonl, canonical_hash, load_jsonl, sha256_file
from dp_SA.soft_score import class_token_ids
from layer_metacognition.model_adapter import resolve_language_modules

from .core import MINIMAL_PROMPT_TEMPLATE, WINDOWS, locate_minimal_positions, prepare_minimal_case


CONDITIONS = ("panl_to_evidence_answer", "panl_plus_1_to_evidence_answer")
LABELS = {
    "sac_to_panl": "SAC→PANL",
    "panl_to_evidence_answer": "PANL→E+A",
    "panl_plus_1_to_evidence_answer": "PANL+1→E+A",
}
COLORS = {
    "sac_to_panl": "#d62728",
    "panl_to_evidence_answer": "#2ca02c",
    "panl_plus_1_to_evidence_answer": "#e377c2",
}
DEFAULT_OUTPUT_PARENT = Path(__file__).resolve().parent / "outputs"


def _sem(values: Sequence[float]) -> float:
    return float(np.std(np.asarray(values, dtype=float), ddof=1) / math.sqrt(len(values)))


def _bootstrap(values: Sequence[float], seed: int, repeats: int = 2000) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = array[rng.integers(0, len(array), size=(repeats, len(array)))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "sem": _sem(array),
        "ci_low": float(np.percentile(sampled, 2.5)),
        "ci_high": float(np.percentile(sampled, 97.5)),
        "p_raw": float((1 + np.count_nonzero(sampled <= 0)) / (repeats + 1)),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _edges(condition: str, positions: dict[str, Any]) -> AttentionEdges:
    if condition == "panl_to_evidence_answer":
        query = positions["PANL"]
    elif condition == "panl_plus_1_to_evidence_answer":
        query = positions["PANL_PLUS_1"]
    else:
        raise ValueError(condition)
    return AttentionEdges.from_sets([query], positions["EVIDENCE_ANSWER"])


def _selection(base_output: Path, smoke: bool) -> list[dict[str, Any]]:
    rows = load_jsonl(base_output / "selection_manifest.jsonl")
    if smoke:
        return [row for side in ("image_side", "text_side") for row in [r for r in rows if r["test_side"] == side][:2]]
    if len(rows) != 30:
        raise ValueError(f"Expected the frozen 30-item minimal selection, found {len(rows)}")
    return rows


def _config(base_output: Path, selection: Sequence[dict[str, Any]], smoke: bool) -> dict[str, Any]:
    implementation = [Path(__file__), Path(__file__).with_name("core.py"), Path(__file__).parents[1] / "masking.py"]
    values = {
        "format_version": 1,
        "diagnostic": "minimal_prompt_panl_gather_addon",
        "base_minimal_output": str(base_output.resolve()),
        "base_run_config_sha256": sha256_file(base_output / "run_config.json"),
        "base_clean_sha256": sha256_file(base_output / "clean_results.jsonl"),
        "selection_hash": canonical_hash(selection),
        "model_path": str(MODEL_PATH.resolve()),
        "model_config_sha256": sha256_file(MODEL_PATH / "config.json"),
        "inference_path": str(INFERENCE_PATH.resolve()),
        "attention_backend": "eager",
        "seed": 42,
        "bootstrap_repeats": 2000,
        "smoke": smoke,
        "prompt": MINIMAL_PROMPT_TEMPLATE,
        "windows": WINDOWS,
        "conditions": CONDITIONS,
        "implementation_sha256": {str(path.resolve()): sha256_file(path) for path in implementation},
    }
    values["fingerprint"] = canonical_hash(values)
    return values


def run(output: Path, base_output: Path, *, smoke: bool, resume: bool) -> dict[str, Any]:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selection = _selection(base_output, smoke)
    config = _config(base_output, selection, smoke)
    config_path = output / "run_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous.get("fingerprint") != config["fingerprint"]:
            raise ValueError("Minimal gather-addon fingerprint changed; refusing resume")
        if not resume:
            raise FileExistsError(f"Output exists: {output}; use --resume")
    else:
        atomic_json(config_path, config)
        atomic_jsonl(output / "selection_manifest.jsonl", selection)

    base_clean = {row["case_id"]: row for row in load_jsonl(base_output / "clean_results.jsonl")}
    if not all(row["case_id"] in base_clean for row in selection):
        raise ValueError("Base minimal clean logits do not cover the addon selection")
    blocked_path = output / "blocked_results.jsonl"
    spans_path = output / "token_spans.jsonl"
    failures_path = output / "failures.jsonl"
    for path in (blocked_path, spans_path, failures_path):
        path.touch(exist_ok=True)
    completed = {
        (row["case_id"], row["condition"], int(row["window_start"]))
        for row in load_jsonl(blocked_path)
    }
    saved_spans = {row["case_id"]: row for row in load_jsonl(spans_path)}

    runtime = load_runtime(INFERENCE_PATH)
    inference = runtime.QwenVLInference(str(MODEL_PATH))
    if getattr(inference.model.config, "_attn_implementation", None) != "eager":
        raise RuntimeError("Minimal gather addon requires eager attention")
    modules = resolve_language_modules(inference.model)
    if modules.num_hidden_layers != 28:
        raise RuntimeError(f"Expected 28 language layers, found {modules.num_hidden_layers}")
    tokenizer = getattr(inference.processor, "tokenizer", inference.processor)
    digit_ids = class_token_ids(tokenizer)
    if [tokenizer.decode([token], skip_special_tokens=False, clean_up_tokenization_spaces=False) for token in digit_ids] != list("012345678"):
        raise RuntimeError("SA classes are not nine single-token digits")

    started = time.time()
    total = len(selection) * len(CONDITIONS) * len(WINDOWS)
    try:
        for row in selection:
            rendered, inputs, prompt = prepare_minimal_case(inference, row)
            positions = locate_minimal_positions(
                tokenizer, rendered, inputs, row["phase0_raw_answer"], row["question"], row["text_clue"]
            )
            if canonical_hash(prompt) != base_clean[row["case_id"]]["prompt_hash"]:
                raise RuntimeError(f"Minimal prompt differs from frozen clean for {row['case_id']}")
            span_row = {
                "case_id": row["case_id"], "item_id": str(row["item_id"]), "test_side": row["test_side"], **positions
            }
            if row["case_id"] in saved_spans:
                if canonical_hash(saved_spans[row["case_id"]]) != canonical_hash(span_row):
                    raise RuntimeError(f"Addon token spans changed for {row['case_id']}")
            else:
                append_jsonl(spans_path, span_row)
                saved_spans[row["case_id"]] = span_row

            baseline = base_clean[row["case_id"]]
            target = int(baseline["clean_class"])
            for condition in CONDITIONS:
                edges = _edges(condition, positions)
                for start, end in WINDOWS:
                    key = (row["case_id"], condition, start)
                    if key in completed:
                        continue
                    before = time.perf_counter()
                    with AttentionBlockContext(
                        modules.language_layers,
                        layer_indices=range(start, end + 1),
                        edges=edges,
                        sequence_length=positions["sequence_length"],
                        row_sum_tolerance=ROW_SUM_TOLERANCE,
                    ) as context:
                        logits, score = _forward(inference.model, inputs, positions["SAC"], digit_ids)
                    elapsed = time.perf_counter() - before
                    blocked_margin = _margin(logits, target)
                    result = {
                        "case_id": row["case_id"], "item_id": str(row["item_id"]), "test_side": row["test_side"],
                        "condition": condition, "window_start": start, "window_end": end,
                        "window_center": (start + end) / 2, "blocked_layer_count": end - start + 1,
                        "class_logits": logits, "blocked_class": int(score["argmax_hard_class"]),
                        "blocked_soft_sa": float(score["soft_sa_image_score"]), "clean_class": target,
                        "clean_margin": float(baseline["clean_margin"]), "blocked_margin": blocked_margin,
                        "logit_margin_disruption": float(baseline["clean_margin"]) - blocked_margin,
                        "first_token_changed": int(score["argmax_hard_class"]) != target,
                        "delta_soft_sa": float(score["soft_sa_image_score"]) - float(baseline["soft_sa_image_score"]),
                        "elapsed_seconds": elapsed, "edge_count": len(edges.pairs),
                        "attention_diagnostics": context.diagnostics(),
                    }
                    append_jsonl(blocked_path, result)
                    completed.add(key)
                    elapsed_total = time.time() - started
                    remaining = total - len(completed)
                    atomic_json(output / "progress.json", {
                        "status": "running", "completed": len(completed), "expected": total,
                        "failed": len(load_jsonl(failures_path)), "elapsed_seconds": elapsed_total,
                        "estimated_remaining_seconds": elapsed_total / max(1, len(completed)) * remaining,
                    })
            del inputs
        completion = {
            "status": "complete", "blocked": len(completed), "failures": len(load_jsonl(failures_path)),
            "elapsed_seconds": time.time() - started, "estimated_remaining_seconds": 0.0,
        }
        atomic_json(output / "completion.json", completion)
        atomic_json(output / "progress.json", completion)
        return completion
    except Exception as exc:
        append_jsonl(failures_path, {"error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        del inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def analyze(output: Path, base_output: Path, repeats: int = 2000, seed: int = 42) -> dict[str, Any]:
    selection = load_jsonl(output / "selection_manifest.jsonl")
    case_ids = {row["case_id"] for row in selection}
    addon = load_jsonl(output / "blocked_results.jsonl")
    existing_sac = [
        row for row in load_jsonl(base_output / "block_results.jsonl")
        if row["case_id"] in case_ids and row["condition"] == "sac_to_panl"
    ]
    combined = existing_sac + addon
    expected = len(selection) * len(WINDOWS)
    if len(existing_sac) != expected or len(addon) != 2 * expected:
        raise ValueError(f"Combined minimal counts invalid: SAC={len(existing_sac)}, addon={len(addon)}")
    if load_jsonl(output / "failures.jsonl"):
        raise ValueError("Minimal gather addon contains failures")

    item_rows: list[dict[str, Any]] = []
    for row in combined:
        item_rows.append({
            "case_id": row["case_id"], "item_id": row["item_id"], "test_side": row["test_side"],
            "condition": row["condition"], "condition_label": LABELS[row["condition"]],
            "window_start": row["window_start"], "window_end": row["window_end"],
            "window_center": (int(row["window_start"]) + int(row["window_end"])) / 2,
            "token_changed": int(row["first_token_changed"]),
            "logit_margin_disruption": float(row["logit_margin_disruption"]),
            "logit_difference_change": -float(row["logit_margin_disruption"]),
            "delta_soft_sa": float(row["delta_soft_sa"]),
        })
    summary_rows: list[dict[str, Any]] = []
    by: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for condition in LABELS:
        for start, end in WINDOWS:
            rows = [r for r in item_rows if r["condition"] == condition and int(r["window_start"]) == start]
            if len(rows) != len(selection):
                raise ValueError(f"Incomplete combined curve {condition} {start}-{end}: {len(rows)}")
            by[(condition, start)] = rows
            token = [float(r["token_changed"]) for r in rows]
            disruption = [float(r["logit_margin_disruption"]) for r in rows]
            summary_rows.append({
                "condition": condition, "condition_label": LABELS[condition], "window_start": start,
                "window_end": end, "window_center": (start + end) / 2, "n": len(rows),
                "token_change_rate": float(np.mean(token)), "token_change_rate_pct": 100 * float(np.mean(token)),
                "logit_margin_disruption_mean": float(np.mean(disruption)),
                "logit_difference_change_mean": -float(np.mean(disruption)),
                "delta_soft_sa_mean": float(np.mean([float(r["delta_soft_sa"]) for r in rows])),
            })

    main_tests, contrast_tests = [], []
    for wi, (start, end) in enumerate(WINDOWS):
        main = [float(r["logit_margin_disruption"]) for r in by[("panl_to_evidence_answer", start)]]
        control = {r["case_id"]: r for r in by[("panl_plus_1_to_evidence_answer", start)]}
        differences = [value - float(control[row["case_id"]]["logit_margin_disruption"]) for row, value in zip(by[("panl_to_evidence_answer", start)], main)]
        main_tests.append({"window_start": start, "window_end": end, "test": "PANL→E+A > 0", **_bootstrap(main, seed + wi, repeats)})
        contrast_tests.append({"window_start": start, "window_end": end, "test": "PANL→E+A − PANL+1→E+A > 0", **_bootstrap(differences, seed + 100 + wi, repeats)})
    for tests in (main_tests, contrast_tests):
        for row, q in zip(tests, bh_fdr([row["p_raw"] for row in tests])):
            row["q_bh"] = q
    decisions = [
        {
            "window_start": main["window_start"], "window_end": main["window_end"],
            "panl_gather_supported": main["ci_low"] > 0 and main["q_bh"] < 0.05
            and contrast["ci_low"] > 0 and contrast["q_bh"] < 0.05,
        }
        for main, contrast in zip(main_tests, contrast_tests)
    ]
    result = {
        "criterion": "PANL→E+A itself and PANL→E+A minus PANL+1→E+A both CI_low>0 and BH q<0.05",
        "supports_panl_gather_any_window": any(row["panl_gather_supported"] for row in decisions),
        "decisions": decisions, "main_tests": main_tests, "contrast_tests": contrast_tests,
        "technical": {"n": len(selection), "addon_rows": len(addon), "failures": 0},
    }
    _write_csv(output / "combined_item_level.csv", item_rows)
    _write_csv(output / "combined_summary.csv", summary_rows)
    atomic_json(output / "analysis.json", result)

    fig, axes = plt.subplots(2, 1, figsize=(8.3, 7.0), sharex=True, constrained_layout=True)
    for ax, metric, ylabel in (
        (axes[0], "token_change_rate_pct", "Token Change Rate (%)"),
        (axes[1], "logit_difference_change_mean", "Logit Difference Change"),
    ):
        for condition in LABELS:
            rows = [r for r in summary_rows if r["condition"] == condition]
            ax.plot([r["window_center"] for r in rows], [r[metric] for r in rows], marker="o", linewidth=1.6,
                    color=COLORS[condition], label=LABELS[condition])
        ax.axhline(0, color="#999999", linewidth=.8, linestyle=(0, (2, 2)))
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#dddddd", linewidth=.5, alpha=.55)
        ax.legend(frameon=False, ncol=3, fontsize=8.5)
    centers = [(start + end) / 2 for start, end in WINDOWS]
    axes[1].set_xticks(centers)
    axes[1].set_xticklabels([f"{center:g}" for center in centers])
    axes[1].set_xlabel("Center layer of selectively blocked 12-layer window")
    fig.suptitle("Minimal Prompt Attention Blocking (mean, n=30)", fontweight="bold")
    fig.savefig(output / "minimal_sac_panl_gather_combined.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "minimal_sac_panl_gather_combined.pdf", bbox_inches="tight")
    plt.close(fig)
    lines = ["# Minimal prompt SAC/PANL gather addon", "", f"- PANL gather supported: **{result['supports_panl_gather_any_window']}**", "",
             "|Window|PANL→E+A mean [CI]|q|Matched difference [CI]|q|Supported|", "|---|---:|---:|---:|---:|---|"]
    for main, contrast, decision in zip(main_tests, contrast_tests, decisions):
        lines.append(f"|{main['window_start']}–{main['window_end']}|{main['mean']:+.4f} [{main['ci_low']:+.4f},{main['ci_high']:+.4f}]|{main['q_bh']:.4f}|{contrast['mean']:+.4f} [{contrast['ci_low']:+.4f},{contrast['ci_high']:+.4f}]|{contrast['q_bh']:.4f}|{decision['panl_gather_supported']}|")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--base-output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_dir or DEFAULT_OUTPUT_PARENT / time.strftime("minimal_gather_addon_seed42_%Y%m%dT%H%M%SZ", time.gmtime())
    if not args.analyze_only:
        run(output, args.base_output.resolve(), smoke=args.smoke, resume=args.resume)
    analyze(output.resolve(), args.base_output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
