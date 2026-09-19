from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[2]
for candidate in (REPOSITORY_ROOT, HERE):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from dp_SA.io_utils import atomic_json, atomic_jsonl, canonical_hash, load_jsonl, sha256_file

from confidence_core import (
    LOG_ODDS_EPSILON,
    atomic_csv,
    calibrated_probability,
    clipped_log_odds,
    fit_nll_temperature,
)
from config import COLORS, MODEL_PATH, SEED
from core import load_text_pool, normalize, stable_key
from prompts import image_only_prompt, text_only_prompt
from runtime import FaithfulQwenRunner


DEFAULT_EXPERIMENT_ROOT = (
    REPOSITORY_ROOT / "qwen3 review" / "short prompt check" / "output"
    / "faithful_check_extended" / "balanced_subset" / "exp"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_EXPERIMENT_ROOT / "logit_confidence"
CALIBRATION_DATASET = REPOSITORY_ROOT / "generate dataset" / "datasets" / "generated_shape_color_dataset.json"
CALIBRATION_IMAGE_ROOT = CALIBRATION_DATASET.parent
SUPPLEMENTAL_DATASET = REPOSITORY_ROOT / "datasets" / "datasets.json"
TEXT_POOL = REPOSITORY_ROOT / "merged_color_prior_pool.json"
CALIBRATION_COUNT = 100


def _question(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("text")
    result = str(value or "").strip()
    if not result:
        raise ValueError("empty calibration question")
    return result


def _shape(question: str) -> str:
    marker = "color of the "
    lower = question.casefold()
    start = lower.find(marker)
    end = question.find("?", start)
    if start < 0 or end < 0:
        raise ValueError(f"cannot parse shape: {question}")
    return normalize(question[start + len(marker):end])


def _dataset_items() -> list[dict[str, Any]]:
    result = []
    for source_name, dataset_path in (("generated", CALIBRATION_DATASET), ("supplemental", SUPPLEMENTAL_DATASET)):
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
        if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
            items = payload[0].get("items")
        elif isinstance(payload, dict):
            items = payload.get("items")
        else:
            items = None
        if not isinstance(items, list):
            raise ValueError(f"{dataset_path} has no items list")
        result.extend({"_source_name": source_name, "_source_path": str(dataset_path), **item} for item in items)
    return result


def _test_inputs(experiment_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = load_jsonl(experiment_root / "manifest.jsonl")
    trials = load_jsonl(experiment_root / "trials.jsonl")
    if len(manifest) != 110 or len(trials) != 110:
        raise ValueError(f"expected 110 CMA cases, found manifest={len(manifest)}, trials={len(trials)}")
    return manifest, trials


def build_calibration_manifest(experiment_root: Path, output_root: Path) -> dict[str, Any]:
    test_manifest, _trials = _test_inputs(experiment_root)
    test_items = {str(int(str(row["item_id"]))) for row in test_manifest}
    used_clues = {
        normalize(container["text_clue"])
        for row in test_manifest
        for container in (row["original_text"], row["counterfactual_text"])
    }
    test_hashes = {sha256_file(row["source_image"]) for row in test_manifest}
    candidates = []
    for item in _dataset_items():
        raw_item_id = str(item.get("id", "")).strip()
        item_id = str(int(raw_item_id)) if raw_item_id.isdigit() else raw_item_id
        source_name = str(item["_source_name"])
        if not item_id or item_id in test_items:
            continue
        question = _question(item.get("question"))
        text_color = normalize(item.get("answer", item.get("text_ans")))
        image_color = normalize(item.get("conflict_ans", item.get("conflict_answer")))
        if text_color not in COLORS or image_color not in COLORS or text_color == image_color:
            continue
        conflict = item.get("image_clue", {}).get("conflict")
        if not isinstance(conflict, dict):
            continue
        paths = {}
        hashes = {}
        for difficulty in ("easy", "hard"):
            raw = conflict.get(difficulty)
            if not isinstance(raw, str):
                break
            path = (Path(item["_source_path"]).resolve().parent / raw).resolve()
            if not path.is_file():
                break
            image_hash = sha256_file(path)
            if image_hash not in test_hashes:
                paths[difficulty] = str(path); hashes[difficulty] = image_hash
        if not paths:
            continue
        answer_classes = [normalize(value) for value in item.get("candidate_colors", COLORS)]
        if answer_classes != list(COLORS):
            raise ValueError(f"calibration item {item_id} candidate order differs from COLORS")
        candidates.append({
            "item_id": item_id, "source_name": source_name, "source_item_key": f"{source_name}:{item_id}",
            "question": question, "shape": _shape(question),
            "text_color": text_color, "image_color": image_color,
            "image_paths": paths, "image_hashes": hashes,
        })
    if len(candidates) < CALIBRATION_COUNT:
        raise ValueError(f"need {CALIBRATION_COUNT} calibration items, found {len(candidates)}")

    # Select the exact difficulty quotas first, then balance color and shape greedily.
    chosen_hashes: set[str] = set()
    def pick(pool: list[dict[str, Any]], difficulty: str, count: int, already: set[str]) -> list[dict[str, Any]]:
        remaining = [row for row in pool if row["source_item_key"] not in already and row["image_hashes"].get(difficulty) not in chosen_hashes]
        picked: list[dict[str, Any]] = []
        text_counts: Counter[str] = Counter(); image_counts: Counter[str] = Counter(); shape_counts: Counter[str] = Counter()
        while len(picked) < count:
            remaining = [row for row in remaining if row["image_hashes"].get(difficulty) not in chosen_hashes]
            if not remaining:
                raise ValueError("not enough independent calibration items for requested difficulty quota")
            if not remaining:
                raise ValueError("not enough unique calibration images for requested difficulty quota")
            chosen = min(remaining, key=lambda row: (
                text_counts[row["text_color"]], image_counts[row["image_color"]], shape_counts[row["shape"]],
                stable_key("confidence-calibration", row["source_item_key"], seed=SEED),
            ))
            remaining.remove(chosen); picked.append(chosen)
            text_counts[chosen["text_color"]] += 1; image_counts[chosen["image_color"]] += 1; shape_counts[chosen["shape"]] += 1
            chosen_hashes.add(chosen["image_hashes"][difficulty])
        return picked

    easy_selected = pick([row for row in candidates if "easy" in row["image_paths"]], "easy", CALIBRATION_COUNT // 2, set())
    selected = easy_selected + pick([row for row in candidates if "hard" in row["image_paths"]], "hard", CALIBRATION_COUNT // 2, {row["source_item_key"] for row in easy_selected})
    difficulty_counts: Counter[str] = Counter(); difficulty_color: Counter[tuple[str, str]] = Counter()
    for row in selected:
        difficulty = "easy" if row in easy_selected else "hard"
        row["difficulty"] = difficulty
        row["image_path"] = row["image_paths"][difficulty]
        row["image_sha256"] = row["image_hashes"][difficulty]
        difficulty_counts[difficulty] += 1; difficulty_color[difficulty, row["image_color"]] += 1

    pool = load_text_pool(TEXT_POOL)
    available = {
        color: [row for row in values if normalize(row["text_clue"]) not in used_clues]
        for color, values in pool.items()
    }
    clue_keys: set[str] = set(); bin_counts: Counter[int] = Counter(); color_bin_counts: Counter[tuple[str, int]] = Counter()
    output = []
    for row in sorted(selected, key=lambda value: stable_key("clue", value["item_id"])):
        color = row["text_color"]
        choices = [value for value in available[color] if normalize(value["text_clue"]) not in clue_keys]
        if not choices:
            raise ValueError(f"no unused calibration clue for {color}")
        clue = min(choices, key=lambda value: (
            bin_counts[int(value["source_bin_id"])], color_bin_counts[color, int(value["source_bin_id"])],
            stable_key("calibration-clue", row["item_id"], value["text_clue"]),
        ))
        key = normalize(clue["text_clue"]); clue_keys.add(key)
        bin_id = int(clue["source_bin_id"]); bin_counts[bin_id] += 1; color_bin_counts[color, bin_id] += 1
        output.append({
            "calibration_id": f"cal_{row['source_name']}_{row['item_id']}", "item_id": row["item_id"],
            "source_name": row["source_name"], "source_item_key": row["source_item_key"],
            "question": row["question"], "shape": row["shape"],
            "text_target": row["text_color"], "text_clue": clue["text_clue"],
            "text_clue_bin": bin_id, "text_clue_source": clue,
            "image_target": row["image_color"], "difficulty": row["difficulty"],
            "image_path": row["image_path"], "image_sha256": row["image_sha256"],
            "answer_classes": list(COLORS),
        })
    output.sort(key=lambda row: row["calibration_id"])
    if len({row["source_item_key"] for row in output}) != CALIBRATION_COUNT:
        raise AssertionError("calibration source items are not unique")
    if len({row["image_sha256"] for row in output}) != CALIBRATION_COUNT:
        raise AssertionError("calibration images are not unique")
    if Counter(row["difficulty"] for row in output) != {"easy": 50, "hard": 50}:
        raise AssertionError("calibration difficulty is not 50/50")
    if clue_keys & used_clues:
        raise AssertionError("calibration clue leakage")
    if {row["image_sha256"] for row in output} & test_hashes:
        raise AssertionError("calibration image leakage")

    atomic_jsonl(output_root / "calibration_manifest.jsonl", output)
    audit = {
        "status": "passed", "seed": SEED, "calibration_count": len(output),
        "source_candidate_count": len(candidates), "test_case_count": len(test_manifest),
        "overlap": {"item": 0, "image_sha256": 0, "text_clue": 0},
        "difficulty_counts": dict(Counter(row["difficulty"] for row in output)),
        "text_color_counts": dict(Counter(row["text_target"] for row in output)),
        "image_color_counts": dict(Counter(row["image_target"] for row in output)),
        "shape_counts": dict(Counter(row["shape"] for row in output)),
        "text_clue_bin_counts": dict(Counter(str(row["text_clue_bin"]) for row in output)),
        "inputs": {
            "calibration_dataset": {"path": str(CALIBRATION_DATASET), "sha256": sha256_file(CALIBRATION_DATASET)},
            "supplemental_dataset": {"path": str(SUPPLEMENTAL_DATASET), "sha256": sha256_file(SUPPLEMENTAL_DATASET)},
            "text_pool": {"path": str(TEXT_POOL), "sha256": sha256_file(TEXT_POOL)},
            "test_manifest": {"path": str(experiment_root / "manifest.jsonl"), "sha256": sha256_file(experiment_root / "manifest.jsonl")},
        },
    }
    atomic_json(output_root / "calibration_selection_audit.json", audit)
    return audit


def score_calibration(output_root: Path, *, resume: bool) -> dict[str, Any]:
    manifest = load_jsonl(output_root / "calibration_manifest.jsonl")
    destination = output_root / "calibration_raw_scores.jsonl"
    existing = load_jsonl(destination, repair_trailing=resume) if resume else []
    by_key = {(row["calibration_id"], row["modality"]): row for row in existing}
    expected = {(row["calibration_id"], modality) for row in manifest for modality in ("text", "image")}
    if set(by_key) - expected:
        raise ValueError("raw calibration scores contain unexpected keys")
    if set(by_key) == expected:
        return {"status": "complete", "score_count": len(by_key), "resumed_noop": True}
    runner = FaithfulQwenRunner(MODEL_PATH)
    rows = list(existing)
    for case in manifest:
        for modality in ("text", "image"):
            key = (case["calibration_id"], modality)
            if key in by_key:
                continue
            target = case[f"{modality}_target"]
            prompt = text_only_prompt(case["question"], case["text_clue"]) if modality == "text" else image_only_prompt(case["question"])
            image_path = None if modality == "text" else case["image_path"]
            result = runner.answer_run(prompt=prompt, image_path=image_path, expected_color=target, run_generation=False)
            score = {
                "calibration_id": case["calibration_id"], "item_id": case["item_id"],
                "modality": modality, "difficulty": case["difficulty"],
                "target_answer": target, "answer_classes": list(COLORS),
                "raw_candidate_scores": result["answer_class_logits"],
                "restricted_top1": result["restricted_top1"],
                "correct": result["restricted_top1"] == target,
                "prompt": result["prompt"], "prompt_hash": result["prompt_hash"],
                "rendered_hash": result["rendered_hash"], "input_modalities": ["text"] if modality == "text" else ["image", "text"],
                "image_path": result["image_path"], "image_sha256": result["image_sha256"],
                "model_fingerprint": result["model_fingerprint"],
            }
            score["score_fingerprint"] = canonical_hash(score)
            rows.append(score); by_key[key] = score
            atomic_jsonl(destination, sorted(rows, key=lambda row: (row["calibration_id"], row["modality"])))
    return {"status": "complete", "score_count": len(rows), "resumed_noop": False}


def fit_temperatures(output_root: Path) -> dict[str, Any]:
    rows = load_jsonl(output_root / "calibration_raw_scores.jsonl")
    if len(rows) != 2 * CALIBRATION_COUNT:
        raise ValueError(f"expected 200 calibration scores, found {len(rows)}")
    temperatures = {}; table = []; trace_rows = []
    for modality in ("text", "image"):
        subset = [row for row in rows if row["modality"] == modality]
        best, trace = fit_nll_temperature(subset)
        baseline = next(row for row in trace if row["temperature"] == 1.0)
        temperatures[modality] = float(best["temperature"])
        table.append({
            "modality": modality, "count": len(subset), "temperature": best["temperature"],
            **{f"uncalibrated_{key}": baseline[key] for key in ("nll", "ece", "brier", "accuracy")},
            **{f"calibrated_{key}": best[key] for key in ("nll", "ece", "brier", "accuracy")},
        })
        trace_rows.extend({"modality": modality, **value} for value in trace)
    atomic_csv(output_root / "temperature_calibration.csv", table)
    atomic_csv(output_root / "temperature_search_trace.csv", trace_rows)
    payload = {
        "objective": "multiclass_nll", "temperatures": temperatures,
        "calibration_count_per_modality": CALIBRATION_COUNT,
        "temperature_fingerprint": canonical_hash({"temperatures": temperatures, "scores": [row["score_fingerprint"] for row in rows]}),
    }
    atomic_json(output_root / "temperatures.json", payload)
    return payload


def build_test_confidence(experiment_root: Path, output_root: Path) -> dict[str, Any]:
    manifest_rows, trial_rows = _test_inputs(experiment_root)
    manifests = {row["case_id"]: row for row in manifest_rows}
    trials = {row["case_id"]: row for row in trial_rows}
    if set(manifests) != set(trials):
        raise ValueError("manifest/trial case mismatch")
    calibration_scores = load_jsonl(output_root / "calibration_raw_scores.jsonl")
    if not calibration_scores:
        raise FileNotFoundError("calibration scores missing")
    model_fingerprint = calibration_scores[0]["model_fingerprint"]
    if any(row["model_fingerprint"] != model_fingerprint for row in calibration_scores):
        raise ValueError("calibration model fingerprints differ")
    temperature_payload = json.loads((output_root / "temperatures.json").read_text(encoding="utf-8"))
    temperatures = temperature_payload["temperatures"]
    clip_counts: Counter[str] = Counter(); output = []
    for case_id in sorted(manifests):
        manifest = manifests[case_id]; trial = trials[case_id]
        if trial.get("status") != "completed":
            raise ValueError(f"incomplete trial: {case_id}")
        fixed = normalize(trial["fixed_answer"])
        if fixed != normalize(trial["original_answer"]["normalized_answer"]) or fixed not in COLORS:
            raise ValueError(f"invalid fixed answer: {case_id}")
        text_gate = manifest["original_text"]; image_gate = manifest["original_image_gate"]
        if not text_gate.get("gate_passed") or not image_gate.get("gate_passed"):
            raise ValueError(f"source unimodal gate failed: {case_id}")
        if text_gate["model_fingerprint"] != model_fingerprint or image_gate["model_fingerprint"] != model_fingerprint:
            raise ValueError(f"model fingerprint mismatch: {case_id}")
        if text_gate["prompt"] != text_only_prompt(manifest["question"], text_gate["text_clue"]):
            raise ValueError(f"text prompt mismatch: {case_id}")
        if image_gate["prompt"] != image_only_prompt(manifest["question"]):
            raise ValueError(f"image prompt mismatch: {case_id}")
        c_t = calibrated_probability(text_gate["answer_class_logits"], COLORS, fixed, float(temperatures["text"]))
        c_i = calibrated_probability(image_gate["answer_class_logits"], COLORS, fixed, float(temperatures["image"]))
        l_t, clip_t = clipped_log_odds(c_t); l_i, clip_i = clipped_log_odds(c_i)
        if clip_t: clip_counts[f"text_{clip_t}"] += 1
        if clip_i: clip_counts[f"image_{clip_i}"] += 1
        output.append({
            "case_id": case_id, "item_id": str(trial["item_id"]), "difficulty": trial["difficulty"],
            "fixed_answer": fixed, "text_target": manifest["text_color"], "image_target": manifest["image_color"],
            "C_t": c_t, "C_i": c_i, "L_t": l_t, "L_i": l_i,
            "G_L": l_i - l_t, "G_C": c_i - c_t,
            "text_temperature": float(temperatures["text"]), "image_temperature": float(temperatures["image"]),
            "text_raw_logits": text_gate["answer_class_logits"], "image_raw_logits": image_gate["answer_class_logits"],
            "cma_signed": float(trial["cma_logit"]["cma_signed"]),
            "temperature_fingerprint": temperature_payload["temperature_fingerprint"],
        })
    atomic_jsonl(output_root / "test_fixed_answer_confidence.jsonl", output)
    audit = {
        "status": "passed", "case_count": len(output), "item_count": len({row["item_id"] for row in output}),
        "fixed_answer_source": "trials.original_(I,T)_generation", "clip_epsilon": LOG_ODDS_EPSILON,
        "clip_counts": dict(clip_counts), "model_fingerprint": model_fingerprint,
        "temperature_fingerprint": temperature_payload["temperature_fingerprint"],
    }
    atomic_json(output_root / "test_confidence_audit.json", audit)
    return audit


def run(experiment_root: Path, output_root: Path, *, preflight_only: bool, resume: bool) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    selection = build_calibration_manifest(experiment_root, output_root)
    config = {
        "experiment": "qwen3_cma_unimodal_logit_confidence", "seed": SEED,
        "model_path": str(MODEL_PATH.resolve()), "model_config_sha256": sha256_file(MODEL_PATH / "config.json"),
        "experiment_root": str(experiment_root.resolve()), "calibration_count": CALIBRATION_COUNT,
        "temperature_objective": "multiclass_nll", "confidence_gap": "logit(C_i)-logit(C_t)",
    }
    config["fingerprint"] = canonical_hash(config)
    config_path = output_root / "run_config.json"
    if resume and config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old != config:
            raise ValueError("resume config fingerprint mismatch")
    atomic_json(config_path, config)
    if preflight_only:
        result = {"status": "preflight_complete", "selection": selection, "output_root": str(output_root.resolve())}
        atomic_json(output_root / "preflight.json", result); return result
    scores = score_calibration(output_root, resume=resume)
    temperatures = fit_temperatures(output_root)
    confidence = build_test_confidence(experiment_root, output_root)
    result = {"status": "complete", "selection": selection, "scores": scores, "temperatures": temperatures, "confidence": confidence}
    atomic_json(output_root / "completion.json", result); return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3 unimodal logit-confidence calibration for CMA cases")
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(); print(json.dumps(run(args.experiment_root, args.output_root, preflight_only=args.preflight_only, resume=args.resume), ensure_ascii=False, indent=2))
