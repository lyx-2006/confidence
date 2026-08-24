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
from dp_SA.attention_block.config import INFERENCE_PATH, LOGIT_PARITY_TOLERANCE, MODEL_PATH, ROW_SUM_TOLERANCE
from dp_SA.attention_block.masking import AttentionBlockContext, AttentionEdges
from dp_SA.attention_block.run import _forward, _margin
from dp_SA.attention_block.sources import prepare_case
from dp_SA.attention_block.spans import edges_for_condition, locate_spans
from dp_SA.io_utils import append_jsonl, atomic_json, atomic_jsonl, canonical_hash, load_jsonl, sha256_file
from dp_SA.soft_score import class_token_ids
from layer_metacognition.model_adapter import resolve_language_modules


WINDOWS = ((10, 15), (12, 17), (14, 19))
CONDITIONS = (
    "all_later_to_answer",
    "all_later_to_answer_keep_panl",
    "all_later_to_answer_keep_panl_plus_1",
)
DEFAULT_OUTPUT_PARENT = Path(__file__).resolve().parent / "outputs"


def _answer_positions(spans: dict[str, Any]) -> list[int]:
    return list(range(int(spans["ANSWER"][0]), int(spans["ANSWER"][1])))


def relay_edges(spans: dict[str, Any], condition: str) -> AttentionEdges:
    full = edges_for_condition(spans, "all_later_to_answer")
    answer = _answer_positions(spans)
    if condition == "all_later_to_answer":
        return full
    if condition == "all_later_to_answer_keep_panl":
        return full.without([spans["PANL"]], answer)
    if condition == "all_later_to_answer_keep_panl_plus_1":
        return full.without([spans["PANL_PLUS_1"]], answer)
    raise ValueError(condition)


def rescue_values(full: float, keep_panl: float, keep_plus_1: float) -> tuple[float, float, float]:
    relay = float(full) - float(keep_panl)
    control = float(full) - float(keep_plus_1)
    return relay, control, relay - control


def _sem(values: Sequence[float]) -> float:
    return float(np.std(np.asarray(values, dtype=float), ddof=1) / math.sqrt(len(values)))


def _bootstrap(values: Sequence[float], seed: int, repeats: int = 2000) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = array[rng.integers(0, len(array), size=(repeats, len(array)))].mean(axis=1)
    return {
        "mean": float(array.mean()), "sem": _sem(array),
        "ci_low": float(np.percentile(sampled, 2.5)), "ci_high": float(np.percentile(sampled, 97.5)),
        "p_raw": float((1 + np.count_nonzero(sampled <= 0)) / (repeats + 1)),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _selection(base_output: Path, smoke: bool) -> list[dict[str, Any]]:
    rows = json.loads((base_output / "delayed_case_manifest.json").read_text())
    if smoke:
        return [row for side in ("image_side", "text_side") for row in [r for r in rows if r["test_side"] == side][:2]]
    if len(rows) != 100:
        raise ValueError(f"Expected frozen delayed n=100, found {len(rows)}")
    return rows


def _config(base_output: Path, selection: Sequence[dict[str, Any]], smoke: bool) -> dict[str, Any]:
    implementation = [Path(__file__), Path(__file__).with_name("masking.py"), Path(__file__).with_name("spans.py")]
    values = {
        "format_version": 1, "experiment": "answer_to_panl_relay_rescue", "arm": "delayed",
        "base_output": str(base_output.resolve()),
        "base_config_sha256": sha256_file(base_output / "run_config.json"),
        "base_clean_sha256": sha256_file(base_output / "clean_baselines.jsonl"),
        "selection_hash": canonical_hash(selection), "windows": WINDOWS, "conditions": CONDITIONS,
        "model_path": str(MODEL_PATH.resolve()), "model_config_sha256": sha256_file(MODEL_PATH / "config.json"),
        "inference_path": str(INFERENCE_PATH.resolve()), "attention_backend": "eager",
        "seed": 42, "bootstrap_repeats": 2000, "smoke": smoke,
        "implementation_sha256": {str(path.resolve()): sha256_file(path) for path in implementation},
    }
    values["fingerprint"] = canonical_hash(values)
    return values


def run(output: Path, base_output: Path, *, smoke: bool, resume: bool) -> dict[str, Any]:
    output = output.resolve(); output.mkdir(parents=True, exist_ok=True)
    selection = _selection(base_output, smoke)
    config = _config(base_output, selection, smoke)
    config_path = output / "run_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous.get("fingerprint") != config["fingerprint"]:
            raise ValueError("Relay-rescue fingerprint changed; refusing resume")
        if not resume:
            raise FileExistsError(f"Output exists: {output}; use --resume")
    else:
        atomic_json(config_path, config); atomic_jsonl(output / "selection_manifest.jsonl", selection)

    clean = {
        row["case_id"]: row for row in load_jsonl(base_output / "clean_baselines.jsonl") if row["arm"] == "delayed"
    }
    frozen_spans = {row["case_id"]: row for row in load_jsonl(base_output / "delayed_token_spans.jsonl")}
    if not all(row["case_id"] in clean and row["case_id"] in frozen_spans for row in selection):
        raise ValueError("Base delayed clean/spans do not cover relay-rescue selection")
    block_path = output / "blocked_results.jsonl"; span_path = output / "token_spans.jsonl"; failure_path = output / "failures.jsonl"
    for path in (block_path, span_path, failure_path): path.touch(exist_ok=True)
    completed = {(r["case_id"], r["condition"], int(r["window_start"])) for r in load_jsonl(block_path)}
    saved_spans = {r["case_id"]: r for r in load_jsonl(span_path)}

    runtime = load_runtime(INFERENCE_PATH); inference = runtime.QwenVLInference(str(MODEL_PATH))
    if getattr(inference.model.config, "_attn_implementation", None) != "eager":
        raise RuntimeError("Relay rescue requires eager attention")
    modules = resolve_language_modules(inference.model)
    if modules.num_hidden_layers != 28: raise RuntimeError(f"Expected 28 layers, found {modules.num_hidden_layers}")
    tokenizer = getattr(inference.processor, "tokenizer", inference.processor); digit_ids = class_token_ids(tokenizer)
    started = time.time(); total = len(selection) * len(CONDITIONS) * len(WINDOWS)
    try:
        for row in selection:
            rendered, inputs = prepare_case(inference, row); spans = locate_spans(tokenizer, rendered, inputs, row)
            frozen = {
                key: value for key, value in frozen_spans[row["case_id"]].items()
                if key not in {"arm", "case_id"}
            }
            if canonical_hash(spans) != canonical_hash(frozen):
                raise RuntimeError(f"Delayed spans changed for {row['case_id']}")
            if row["case_id"] not in saved_spans:
                append_jsonl(span_path, {"case_id": row["case_id"], "item_id": row["item_id"], "test_side": row["test_side"], **spans})
                saved_spans[row["case_id"]] = spans
            baseline = clean[row["case_id"]]; target = int(baseline["clean_class"])
            if smoke:
                logits, score = _forward(inference.model, inputs, spans["SAC"], digit_ids)
                max_diff = max(abs(float(a) - float(b)) for a, b in zip(logits, baseline["class_logits"]))
                if max_diff > LOGIT_PARITY_TOLERANCE or int(score["argmax_hard_class"]) != target:
                    raise RuntimeError(f"Smoke clean parity failed for {row['case_id']}: {max_diff}")
            for condition in CONDITIONS:
                edges = relay_edges(spans, condition)
                for start, end in WINDOWS:
                    key = (row["case_id"], condition, start)
                    if key in completed: continue
                    before = time.perf_counter()
                    with AttentionBlockContext(
                        modules.language_layers, layer_indices=range(start, end + 1), edges=edges,
                        sequence_length=spans["sequence_length"], row_sum_tolerance=ROW_SUM_TOLERANCE,
                    ) as context:
                        logits, score = _forward(inference.model, inputs, spans["SAC"], digit_ids)
                    blocked_margin = _margin(logits, target)
                    append_jsonl(block_path, {
                        "case_id": row["case_id"], "item_id": row["item_id"], "test_side": row["test_side"],
                        "condition": condition, "window_start": start, "window_end": end,
                        "window_center": (start + end) / 2, "class_logits": logits,
                        "clean_class": target, "blocked_class": int(score["argmax_hard_class"]),
                        "clean_margin": float(baseline["clean_margin"]), "blocked_margin": blocked_margin,
                        "logit_margin_disruption": float(baseline["clean_margin"]) - blocked_margin,
                        "first_token_changed": int(score["argmax_hard_class"]) != target,
                        "blocked_soft_sa": float(score["soft_sa_image_score"]),
                        "delta_soft_sa": float(score["soft_sa_image_score"]) - float(baseline["soft_sa_image_score"]),
                        "edge_count": len(edges.pairs), "elapsed_seconds": time.perf_counter() - before,
                        "attention_diagnostics": context.diagnostics(),
                    }); completed.add(key)
                    elapsed = time.time() - started; remaining = total - len(completed)
                    atomic_json(output / "progress.json", {
                        "status": "running", "completed": len(completed), "expected": total,
                        "failed": len(load_jsonl(failure_path)), "elapsed_seconds": elapsed,
                        "estimated_remaining_seconds": elapsed / max(1, len(completed)) * remaining,
                    })
            del inputs
        completion = {"status": "complete", "blocked": len(completed), "failures": len(load_jsonl(failure_path)),
                      "elapsed_seconds": time.time() - started, "estimated_remaining_seconds": 0.0}
        atomic_json(output / "completion.json", completion); atomic_json(output / "progress.json", completion)
        return completion
    except Exception as exc:
        append_jsonl(failure_path, {"error_type": type(exc).__name__, "error": str(exc)}); raise
    finally:
        del inference
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def analyze(output: Path, repeats: int = 2000, seed: int = 42) -> dict[str, Any]:
    selected = load_jsonl(output / "selection_manifest.jsonl"); blocked = load_jsonl(output / "blocked_results.jsonl")
    expected = len(selected) * len(CONDITIONS) * len(WINDOWS)
    if len(blocked) != expected or load_jsonl(output / "failures.jsonl"):
        raise ValueError(f"Relay-rescue formal counts invalid: {len(blocked)}/{expected}")
    lookup = {(r["case_id"], r["condition"], int(r["window_start"])): r for r in blocked}
    item_rows: list[dict[str, Any]] = []
    for start, end in WINDOWS:
        for case in selected:
            cid = case["case_id"]
            full = lookup[(cid, CONDITIONS[0], start)]; keep = lookup[(cid, CONDITIONS[1], start)]; control = lookup[(cid, CONDITIONS[2], start)]
            relay, relay_control, contrast = rescue_values(
                full["logit_margin_disruption"], keep["logit_margin_disruption"], control["logit_margin_disruption"]
            )
            item_rows.append({
                "case_id": cid, "item_id": case["item_id"], "test_side": case["test_side"],
                "window_start": start, "window_end": end, "window_center": (start + end) / 2,
                "D_full": full["logit_margin_disruption"], "D_keepPANL": keep["logit_margin_disruption"],
                "D_keepPANL_plus_1": control["logit_margin_disruption"], "R_relay_PANL": relay,
                "R_relay_PANL_plus_1": relay_control, "R_matched_difference": contrast,
                "change_rate_full": int(full["first_token_changed"]), "change_rate_keepPANL": int(keep["first_token_changed"]),
                "change_rate_keepPANL_plus_1": int(control["first_token_changed"]),
                "delta_soft_sa_full": full["delta_soft_sa"], "delta_soft_sa_keepPANL": keep["delta_soft_sa"],
                "delta_soft_sa_keepPANL_plus_1": control["delta_soft_sa"],
            })
    summary_rows: list[dict[str, Any]] = []; main_tests=[]; contrast_tests=[]
    for wi, (start, end) in enumerate(WINDOWS):
        window = [r for r in item_rows if r["window_start"] == start]
        main = _bootstrap([r["R_relay_PANL"] for r in window], seed + wi, repeats)
        contrast = _bootstrap([r["R_matched_difference"] for r in window], seed + 100 + wi, repeats)
        main_tests.append({"window_start": start, "window_end": end, "test": "R_relay(PANL)>0", **main})
        contrast_tests.append({"window_start": start, "window_end": end, "test": "R_relay(PANL)-R_relay(PANL+1)>0", **contrast})
        for group in ("all_100", "image_side", "text_side"):
            rows = window if group == "all_100" else [r for r in window if r["test_side"] == group]
            relay_stats = _bootstrap([r["R_relay_PANL"] for r in rows], seed + 300 + wi + len(group), repeats)
            control_stats = _bootstrap([r["R_relay_PANL_plus_1"] for r in rows], seed + 400 + wi + len(group), repeats)
            diff_stats = _bootstrap([r["R_matched_difference"] for r in rows], seed + 500 + wi + len(group), repeats)
            summary_rows.append({
                "window_start": start, "window_end": end, "window_center": (start + end) / 2,
                "group": group, "n": len(rows),
                "R_relay_PANL_mean": relay_stats["mean"], "R_relay_PANL_ci_low": relay_stats["ci_low"], "R_relay_PANL_ci_high": relay_stats["ci_high"],
                "R_relay_PANL_plus_1_mean": control_stats["mean"], "R_relay_PANL_plus_1_ci_low": control_stats["ci_low"], "R_relay_PANL_plus_1_ci_high": control_stats["ci_high"],
                "R_matched_difference_mean": diff_stats["mean"], "R_matched_difference_ci_low": diff_stats["ci_low"], "R_matched_difference_ci_high": diff_stats["ci_high"],
                "change_rate_full": float(np.mean([r["change_rate_full"] for r in rows])),
                "change_rate_keepPANL": float(np.mean([r["change_rate_keepPANL"] for r in rows])),
                "change_rate_keepPANL_plus_1": float(np.mean([r["change_rate_keepPANL_plus_1"] for r in rows])),
                "delta_soft_sa_full": float(np.mean([r["delta_soft_sa_full"] for r in rows])),
                "delta_soft_sa_keepPANL": float(np.mean([r["delta_soft_sa_keepPANL"] for r in rows])),
                "delta_soft_sa_keepPANL_plus_1": float(np.mean([r["delta_soft_sa_keepPANL_plus_1"] for r in rows])),
            })
    for tests in (main_tests, contrast_tests):
        for row, q in zip(tests, bh_fdr([r["p_raw"] for r in tests])): row["q_bh"] = q
    decisions = [{
        "window_start": m["window_start"], "window_end": m["window_end"],
        "relay_supported": m["ci_low"] > 0 and m["q_bh"] < .05 and c["ci_low"] > 0 and c["q_bh"] < .05,
    } for m, c in zip(main_tests, contrast_tests)]
    for row in summary_rows:
        if row["group"] == "all_100": continue
        all_row = next(r for r in summary_rows if r["group"] == "all_100" and r["window_start"] == row["window_start"])
        other = next(r for r in summary_rows if r["group"] not in {"all_100", row["group"]} and r["window_start"] == row["window_start"])
        row["direction_consistent_with_all"] = np.sign(row["R_relay_PANL_mean"]) == np.sign(all_row["R_relay_PANL_mean"])
        row["direction_consistent_between_sides"] = np.sign(row["R_relay_PANL_mean"]) == np.sign(other["R_relay_PANL_mean"])
    result = {"criterion": "R_relay(PANL)>0 and R_relay(PANL)-R_relay(PANL+1)>0, each CI_low>0 and BH q<0.05",
              "supports_relay_any_window": any(r["relay_supported"] for r in decisions), "decisions": decisions,
              "main_tests": main_tests, "contrast_tests": contrast_tests,
              "technical": {"n": len(selected), "blocked_rows": len(blocked), "failures": 0}}
    _write_csv(output / "relay_item_level.csv", item_rows); _write_csv(output / "relay_summary.csv", summary_rows)
    atomic_json(output / "analysis.json", result)
    lines=["# Answer→PANL relay rescue", "", f"- Relay supported: **{result['supports_relay_any_window']}**", "",
           "|Window|R(PANL) [CI]|q|R(PANL)-R(PANL+1) [CI]|q|Supported|", "|---|---:|---:|---:|---:|---|"]
    for m,c,d in zip(main_tests,contrast_tests,decisions):
        lines.append(f"|{m['window_start']}–{m['window_end']}|{m['mean']:+.4f} [{m['ci_low']:+.4f},{m['ci_high']:+.4f}]|{m['q_bh']:.4f}|{c['mean']:+.4f} [{c['ci_low']:+.4f},{c['ci_high']:+.4f}]|{c['q_bh']:.4f}|{d['relay_supported']}|")
    (output / "summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    all_rows=[r for r in summary_rows if r["group"]=="all_100"]
    fig,ax=plt.subplots(figsize=(7.5,4.5),constrained_layout=True)
    ax.plot([r["window_center"] for r in all_rows],[r["R_relay_PANL_mean"] for r in all_rows],marker="o",color="#2ca02c",label="Keep PANL→A rescue")
    ax.plot([r["window_center"] for r in all_rows],[r["R_relay_PANL_plus_1_mean"] for r in all_rows],marker="o",color="#e377c2",label="Keep PANL+1→A rescue")
    ax.axhline(0,color="#999999",linewidth=.8,linestyle=(0,(2,2)));ax.set_xticks([r["window_center"] for r in all_rows])
    ax.set_xlabel("Center layer of selectively blocked 6-layer window");ax.set_ylabel("Relay rescue");ax.legend(frameon=False)
    ax.set_title("Delayed Answer→PANL Relay Rescue (mean, n=100)",fontweight="bold")
    fig.savefig(output/"relay_rescue.png",dpi=300,bbox_inches="tight");fig.savefig(output/"relay_rescue.pdf",bbox_inches="tight");plt.close(fig)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser();parser.add_argument("--output-dir",type=Path);parser.add_argument("--base-output",type=Path,required=True)
    parser.add_argument("--smoke",action="store_true");parser.add_argument("--resume",action="store_true");parser.add_argument("--analyze-only",action="store_true")
    args=parser.parse_args(argv);output=args.output_dir or DEFAULT_OUTPUT_PARENT/time.strftime("relay_rescue_seed42_%Y%m%dT%H%M%SZ",time.gmtime())
    if not args.analyze_only: run(output,args.base_output.resolve(),smoke=args.smoke,resume=args.resume)
    analyze(output.resolve());return 0


if __name__=="__main__": raise SystemExit(main())
