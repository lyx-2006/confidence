from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from layer_metacognition.model_adapter import AdditiveActivationHook, run_logits_forward

from .config import SEED, T3_LABELS
from .io_utils import append_jsonl, array_hash, atomic_csv, atomic_json, atomic_jsonl, canonical_hash, load_jsonl
from .runtime import _append_candidate, class_token_ids, load_inference, prepare_case, score_prepared
from .sampling import validation_test
from .scoring import conditional_sequence_log_likelihood, numeric_score, t3_score
from .shared_axis_loto import BOOTSTRAPS, PILOT_NODE, PROBE_RUN_ROOT, TEMPLATES, equal_dose_vector, node_name, retention_eligible
from .templates import TEMPLATES as TEMPLATE_SPECS


def _hard(score: dict[str, Any], template: str) -> Any:
    return score["canonical_hard_label"] if template == "T3" else score["canonical_hard_class"]


@torch.inference_mode()
def steered_score(model: Any, modules: Any, tokenizer: Any, inputs: Any, located: dict[str, Any], template: str, position: str, layer: int, displacement: np.ndarray) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    spec = TEMPLATE_SPECS[template]; target = int(located[position]["processed_index"]); sac = int(located["P1_SAC"]["processed_index"]); vector = torch.from_numpy(np.asarray(displacement, np.float32)); prefix = int(inputs.input_ids.shape[1])
    if spec.kind != "labels":
        hook = AdditiveActivationHook(modules, layer_index=layer, target_position=target, steering_vector=vector, prefill_sequence_length=prefix, injection_site="block_output")
        with hook: logits = run_logits_forward(model, inputs, [sac], modules)[sac]
        ids = class_token_ids(tokenizer); score = numeric_score([float(logits[index]) for index in ids], reversed_scale=spec.kind == "numeric_reversed", token_ids=ids)
        return score, hook.diagnostics(), hook.h_before.numpy()
    token_ids = [list(map(int, tokenizer.encode(label, add_special_tokens=False))) for label in T3_LABELS]; likelihoods = []; diagnostics = []; before = None
    for ids in token_ids:
        candidate = _append_candidate(inputs, ids); hook = AdditiveActivationHook(modules, layer_index=layer, target_position=target, steering_vector=vector, prefill_sequence_length=int(candidate["input_ids"].shape[1]), injection_site="block_output")
        with hook: output = model(**candidate, use_cache=False, return_dict=True)
        likelihoods.append(conditional_sequence_log_likelihood(output.logits, prefix, ids)); diagnostics.append(hook.diagnostics())
        if before is None: before = hook.h_before.numpy()
    assert before is not None
    return t3_score(likelihoods, token_ids), {"candidate_hooks": diagnostics, "hook_count": len(diagnostics)}, before


def _trial_key(template: str, case: str, position: str, layer: int, direction: str, replicate: int, alpha: float) -> str:
    return f"{template}|{case}|{position}|L{layer}|{direction}|r{replicate}|a{alpha:g}"


def _load_node(root: Path, position: str, layer: int) -> dict[str, np.ndarray]:
    with np.load(root / f"artifacts/axes/{node_name(position, layer)}.npz") as payload: return {key: np.asarray(payload[key], np.float32) for key in payload.files}


def _scales(root: Path) -> dict[tuple[str, str, int], tuple[float, float]]:
    rows = load_jsonl(root / "tables/random_direction_audit.jsonl")
    if not rows:
        import csv
        with (root / "tables/random_direction_audit.csv").open(newline="", encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
    output = {}
    for row in rows: output[str(row["template"]), str(row["position"]), int(row["layer"])] = (float(row["sigma_loto"]), float(row["sigma_self"]))
    return output


def run_steering(root: Path, nodes: Sequence[tuple[str, int]], *, validation_cases: int, random_count: int, resume: bool) -> dict[str, Any]:
    test = validation_test(validation_cases); path = root / "artifacts/steering/trials.jsonl"; clean_path = root / "artifacts/steering/clean_baselines.jsonl"
    existing = {_trial_key(row["template"], str(row["case_id"]), row["position"], int(row["layer"]), row["direction"], int(row["replicate"]), float(row["alpha"])) for row in load_jsonl(path) if row.get("status") == "completed"}
    clean = {(row["template"], str(row["case_id"])): row for row in load_jsonl(clean_path) if row.get("status") == "completed"}; scales = _scales(root); inference, modules, tokenizer, device, processor = load_inference(); forwards = 0; started = time.time()
    for template in TEMPLATES:
        for record in test:
            case = str(record["case_id"]); inputs, rendered, located = prepare_case(inference.processor, tokenizer, device, record, TEMPLATE_SPECS[template]); clean_key = (template, case)
            if clean_key not in clean:
                score = score_prepared(inference.model, modules, tokenizer, inputs, located, TEMPLATE_SPECS[template]); forwards += 5 if template == "T3" else 1
                clean[clean_key] = {"status": "completed", "template": template, "case_id": case, "family_id": str(record["family_id"]), "test_answer": str(record["test_answer"]), "test_status": str(record["test_status"]), "rendered_prompt_sha256": __import__("hashlib").sha256(rendered.encode()).hexdigest(), "score": score, "processor": processor}; append_jsonl(clean_path, clean[clean_key])
            clean_score = clean[clean_key]["score"]
            for position, layer in nodes:
                arrays = _load_node(root, position, layer); sigma_loto, sigma_self = scales[template, position, layer]
                zero_key = _trial_key(template, case, position, layer, "zero_baseline", 0, 0.0)
                if zero_key not in existing:
                    zero_score, diagnostics, before = steered_score(inference.model, modules, tokenizer, inputs, located, template, position, layer, np.zeros(3584, np.float32)); forwards += 5 if template == "T3" else 1
                    error = abs(float(zero_score["canonical_soft_sa"]) - float(clean_score["canonical_soft_sa"])); passed = error <= 1e-6 and _hard(zero_score, template) == _hard(clean_score, template)
                    if not passed: raise ValueError(f"Alpha=0 parity failed: {zero_key} error={error}")
                    row = {"status": "completed", "template": template, "case_id": case, "family_id": str(record["family_id"]), "test_answer": str(record["test_answer"]), "test_status": str(record["test_status"]), "position": position, "layer": layer, "direction": "zero_baseline", "replicate": 0, "alpha": 0.0, "scale_name": "zero", "scale": 0.0, "clean_soft_sa": float(clean_score["canonical_soft_sa"]), "steered_soft_sa": float(zero_score["canonical_soft_sa"]), "delta_soft_sa": 0.0, "clean_hard": _hard(clean_score, template), "steered_hard": _hard(zero_score, template), "alpha_zero_abs_error": error, "alpha_zero_parity": passed, "natural_hidden_projection_loto": float(before @ arrays[f"loto_{template}"]), "planned_injection_norm": 0.0, "actual_injection_norm": 0.0, "hook_diagnostics": diagnostics, "scoring": zero_score}; append_jsonl(path, row); existing.add(zero_key)
                directions = [("shared_loto", 0, arrays[f"loto_{template}"], sigma_loto), ("self_matched_dose", 0, arrays[f"unit_{template}"], sigma_loto), ("self_natural_scale", 0, arrays[f"unit_{template}"], sigma_self)]
                randoms = arrays[f"random_{template}"]
                if randoms.shape[0] != random_count: raise ValueError(f"Random vector count mismatch: {template} {position} L{layer}")
                directions.extend(("random_matched", index + 1, vector, sigma_loto) for index, vector in enumerate(randoms))
                reference_norm = 2.0 * sigma_loto
                for direction, replicate, unit, scale in directions:
                    for alpha in (-2.0, 2.0):
                        key = _trial_key(template, case, position, layer, direction, replicate, alpha)
                        if key in existing:
                            if not resume: raise FileExistsError(key)
                            continue
                        displacement = equal_dose_vector(unit, scale, alpha); planned_norm = float(np.linalg.norm(displacement.astype(np.float64))); equal_error = abs(planned_norm - reference_norm) / reference_norm if direction != "self_natural_scale" else math.nan
                        if direction != "self_natural_scale" and equal_error > 1e-6: raise ValueError(f"Equal dose norm failed: {key} {equal_error}")
                        score, diagnostics, before = steered_score(inference.model, modules, tokenizer, inputs, located, template, position, layer, displacement); forwards += 5 if template == "T3" else 1
                        actual = diagnostics["candidate_hooks"][0]["injection_l2"] if template == "T3" else diagnostics["injection_l2"]
                        row = {"status": "completed", "template": template, "case_id": case, "family_id": str(record["family_id"]), "test_answer": str(record["test_answer"]), "test_status": str(record["test_status"]), "position": position, "layer": layer, "direction": direction, "replicate": replicate, "alpha": alpha, "scale_name": "sigma_self" if direction == "self_natural_scale" else "sigma_loto", "scale": float(scale), "sigma_loto": sigma_loto, "sigma_self": sigma_self, "clean_soft_sa": float(clean_score["canonical_soft_sa"]), "steered_soft_sa": float(score["canonical_soft_sa"]), "delta_soft_sa": float(score["canonical_soft_sa"]) - float(clean_score["canonical_soft_sa"]), "clean_hard": _hard(clean_score, template), "steered_hard": _hard(score, template), "hard_label_changed": _hard(clean_score, template) != _hard(score, template), "length_normalized_clean_soft_sa": clean_score.get("length_normalized_soft_sa"), "length_normalized_steered_soft_sa": score.get("length_normalized_soft_sa"), "planned_injection_norm": planned_norm, "equal_dose_relative_error": equal_error, "actual_injection_norm": float(actual), "unit_vector_sha256": array_hash(np.asarray(unit, np.float32)), "hook_diagnostics": diagnostics, "scoring": score}; append_jsonl(path, row); existing.add(key)
                        if forwards % 25 == 0: atomic_json(root / "progress/steering.json", {"status": "running", "nodes": [node_name(*node) for node in nodes], "new_gpu_forwards": forwards, "last": key, "elapsed_seconds": time.time() - started})
    result = {"status": "complete", "nodes": [node_name(*node) for node in nodes], "new_gpu_forwards": forwards, "resumed_noop": forwards == 0, "elapsed_seconds": time.time() - started}; atomic_json(root / "progress/steering.json", result); return result


class Bootstrap:
    def __init__(self, test: Sequence[dict[str, Any]], repeats: int = BOOTSTRAPS, random_count: int = 10):
        self.test = {str(row["case_id"]): row for row in test}; self.rng = np.random.default_rng(SEED + 911); self.repeats = repeats
        confirm = [row for row in test if row["test_status"] == "confirmatory"]; self.answers = sorted({str(row["test_answer"]) for row in confirm}); self.by_answer = {answer: sorted(str(row["case_id"]) for row in confirm if str(row["test_answer"]) == answer) for answer in self.answers}; self.answer_draws = {answer: self.rng.integers(0, len(ids), size=(repeats, len(ids))) for answer, ids in self.by_answer.items()}; self.all_ids = sorted(self.test); self.all_draws = self.rng.integers(0, len(self.all_ids), size=(repeats, len(self.all_ids)))
        self.direction_draws = self.rng.integers(0, random_count, size=(repeats, random_count))

    def aggregate(self, values: dict[str, float], mode: str) -> tuple[float, np.ndarray]:
        if mode == "answer_equal":
            points = []; boots = []
            for answer in self.answers:
                ids = self.by_answer[answer]; vector = np.asarray([values[case] for case in ids]); points.append(vector.mean()); boots.append(vector[self.answer_draws[answer]].mean(axis=1))
            return float(np.mean(points)), np.stack(boots).mean(axis=0)
        vector = np.asarray([values[case] for case in self.all_ids]); return float(vector.mean()), vector[self.all_draws].mean(axis=1)


def _effect_map(rows: Sequence[dict[str, Any]], template: str, position: str, layer: int, direction: str, replicate: int, *, length_normalized: bool = False) -> dict[str, float]:
    selected = [row for row in rows if row["template"] == template and row["position"] == position and int(row["layer"]) == layer and row["direction"] == direction and int(row["replicate"]) == replicate]
    field = "length_normalized_steered_soft_sa" if length_normalized else "steered_soft_sa"
    keyed = {(str(row["case_id"]), float(row["alpha"])): float(row[field]) for row in selected}; cases = sorted({case for case, _ in keyed})
    return {case: (keyed[case, 2.0] - keyed[case, -2.0]) / 2.0 for case in cases}


def _alpha_map(rows: Sequence[dict[str, Any]], template: str, position: str, layer: int, direction: str, replicate: int, alpha: float) -> dict[str, float]:
    """Case-level steered-minus-clean SA for one explicit alpha."""
    selected = [row for row in rows if row["template"] == template and row["position"] == position and int(row["layer"]) == layer and row["direction"] == direction and int(row["replicate"]) == replicate and float(row["alpha"]) == float(alpha)]
    grouped = defaultdict(list)
    for row in selected: grouped[str(row["case_id"])].append(float(row["delta_soft_sa"]))
    return {case: float(np.mean(values)) for case, values in grouped.items()}


def _random_alpha_map(rows: Sequence[dict[str, Any]], template: str, position: str, layer: int, random_count: int, alpha: float) -> dict[str, float]:
    grouped = defaultdict(list)
    for replicate in range(1, random_count + 1):
        for case, value in _alpha_map(rows, template, position, layer, "random_matched", replicate, alpha).items(): grouped[case].append(value)
    return {case: float(np.mean(values)) for case, values in grouped.items()}


def analyze(root: Path, nodes: Sequence[tuple[str, int]], *, validation_cases: int, random_count: int) -> dict[str, Any]:
    rows = load_jsonl(root / "artifacts/steering/trials.jsonl"); test = validation_test(validation_cases); bootstrap = Bootstrap(test, random_count=random_count); effects = []; comparisons = []; overall_rows = []; t3_sensitivity = []
    primary_gate = None
    for position, layer in nodes:
        template_shared = {}; template_shared_boot = {}; template_random = {}; template_random_boot = {}
        for template in TEMPLATES:
            maps = {direction: _effect_map(rows, template, position, layer, direction, 0) for direction in ("shared_loto", "self_matched_dose", "self_natural_scale")}; random_maps = [_effect_map(rows, template, position, layer, "random_matched", replicate) for replicate in range(1, random_count + 1)]
            for mode in ("answer_equal", "family_micro"):
                aggregates = {}
                for direction, values in maps.items():
                    point, boot = bootstrap.aggregate(values, mode); low, high = np.quantile(boot, [.025, .975]); aggregates[direction] = (point, boot); effects.append({"template": template, "position": position, "layer": layer, "direction": direction, "aggregation": mode, "s2": point, "ci_low": float(low), "ci_high": float(high), "case_count": 95 if mode == "answer_equal" else validation_cases})
                # Keep the explicit alpha responses in the result table; S² is
                # retained as a derived contrast, never the sole output.
                for direction in ("shared_loto", "self_matched_dose", "self_natural_scale"):
                    alpha_fields = {}
                    for alpha, label in ((-2.0, "m2"), (0.0, "0"), (2.0, "p2")):
                        amap = _alpha_map(rows, template, position, layer, direction, 0, alpha)
                        if amap:
                            alpha_fields[f"delta_sa_alpha_{label}"] = float(bootstrap.aggregate(amap, mode)[0])
                            alpha_fields[f"delta_sa_alpha_{label}_ci_low"], alpha_fields[f"delta_sa_alpha_{label}_ci_high"] = [float(x) for x in np.quantile(bootstrap.aggregate(amap, mode)[1], [.025, .975])]
                        else:
                            # alpha=0 is deliberately stored in the separate
                            # zero-baseline records; by parity its delta is
                            # exactly zero for every direction.
                            value = 0.0 if alpha == 0.0 else math.nan
                            alpha_fields[f"delta_sa_alpha_{label}"] = value; alpha_fields[f"delta_sa_alpha_{label}_ci_low"] = value; alpha_fields[f"delta_sa_alpha_{label}_ci_high"] = value
                    for effect in reversed(effects):
                        if effect["template"] == template and effect["position"] == position and int(effect["layer"]) == layer and effect["direction"] == direction and effect["aggregation"] == mode:
                            effect.update(alpha_fields); break
                random_points = []; random_boots = []
                for values in random_maps:
                    point, boot = bootstrap.aggregate(values, mode); random_points.append(point); random_boots.append(boot)
                random_point = float(np.mean(random_points)); random_boot_matrix = np.stack(random_boots); random_boot = np.asarray([random_boot_matrix[bootstrap.direction_draws[index], index].mean() for index in range(BOOTSTRAPS)])
                random_alpha_fields = {}
                for alpha, label in ((-2.0, "m2"), (0.0, "0"), (2.0, "p2")):
                    amap = _random_alpha_map(rows, template, position, layer, random_count, alpha)
                    if amap:
                        rb_point, rb_boot = bootstrap.aggregate(amap, mode); random_alpha_fields[f"delta_sa_alpha_{label}"] = float(rb_point); random_alpha_fields[f"delta_sa_alpha_{label}_ci_low"], random_alpha_fields[f"delta_sa_alpha_{label}_ci_high"] = [float(x) for x in np.quantile(rb_boot, [.025, .975])]
                    else:
                        value = 0.0 if alpha == 0.0 else math.nan
                        random_alpha_fields[f"delta_sa_alpha_{label}"] = value; random_alpha_fields[f"delta_sa_alpha_{label}_ci_low"] = value; random_alpha_fields[f"delta_sa_alpha_{label}_ci_high"] = value
                effects.append({"template": template, "position": position, "layer": layer, "direction": "random_matched", "aggregation": mode, "s2": random_point, "ci_low": float(np.quantile(random_boot, .025)), "ci_high": float(np.quantile(random_boot, .975)), "case_count": 95 if mode == "answer_equal" else validation_cases, **random_alpha_fields})
                shared_point, shared_boot = aggregates["shared_loto"]; self_point, self_boot = aggregates["self_matched_dose"]; natural_point, natural_boot = aggregates["self_natural_scale"]; elig = retention_eligible(self_point, self_boot); valid_ratio = shared_boot / self_boot; valid_ratio = valid_ratio[np.isfinite(valid_ratio) & (np.sign(self_boot) == np.sign(self_point))]
                comparisons.append({"template": template, "position": position, "layer": layer, "aggregation": mode, "s2_shared": shared_point, "s2_self_matched": self_point, "shared_minus_self_matched": shared_point - self_point, "shared_minus_self_ci_low": float(np.quantile(shared_boot - self_boot, .025)), "shared_minus_self_ci_high": float(np.quantile(shared_boot - self_boot, .975)), "retention_ratio": shared_point / self_point if elig["eligible"] else math.nan, "retention_ratio_ci_low": float(np.quantile(valid_ratio, .025)) if elig["eligible"] else math.nan, "retention_ratio_ci_high": float(np.quantile(valid_ratio, .975)) if elig["eligible"] else math.nan, "retention_eligible": elig["eligible"], "self_denominator_ci_low": elig["ci_low"], "self_denominator_ci_high": elig["ci_high"], "self_denominator_same_sign_fraction": elig["same_sign_fraction"], "s2_self_natural": natural_point, "self_natural_minus_matched": natural_point - self_point, "self_natural_minus_matched_ci_low": float(np.quantile(natural_boot - self_boot, .025)), "self_natural_minus_matched_ci_high": float(np.quantile(natural_boot - self_boot, .975)), "random_mean_s2": random_point, "shared_minus_random": shared_point - random_point, "shared_minus_random_ci_low": float(np.quantile(shared_boot - random_boot, .025)), "shared_minus_random_ci_high": float(np.quantile(shared_boot - random_boot, .975))})
                if mode == "answer_equal": template_shared[template] = shared_point; template_shared_boot[template] = shared_boot; template_random[template] = random_point; template_random_boot[template] = random_boot
                if template == "T3":
                    normalized = _effect_map(rows, template, position, layer, "shared_loto", 0, length_normalized=True); normalized_point, normalized_boot = bootstrap.aggregate(normalized, mode); normalized_low, normalized_high = np.quantile(normalized_boot, [.025, .975]); t3_sensitivity.append({"position": position, "layer": layer, "aggregation": mode, "total_log_likelihood_s2": shared_point, "length_normalized_s2": normalized_point, "length_normalized_ci_low": float(normalized_low), "length_normalized_ci_high": float(normalized_high), "direction_agrees": bool(np.sign(shared_point) == np.sign(normalized_point)), "stability_evidence_sufficient": bool(np.sign(shared_point) == np.sign(normalized_point))})
        overall = float(np.mean(list(template_shared.values()))); overall_boot = np.stack(list(template_shared_boot.values())).mean(axis=0); random_overall = float(np.mean(list(template_random.values()))); random_overall_boot = np.stack(list(template_random_boot.values())).mean(axis=0); low, high = np.quantile(overall_boot, [.025, .975]); contrast_boot = overall_boot - random_overall_boot
        overall_row = {"position": position, "layer": layer, "aggregation": "template_equal_of_answer_equal", "s2_shared": overall, "ci_low": float(low), "ci_high": float(high), "random_mean_s2": random_overall, "shared_minus_random": overall - random_overall, "shared_minus_random_ci_low": float(np.quantile(contrast_boot, .025)), "shared_minus_random_ci_high": float(np.quantile(contrast_boot, .975))}; overall_rows.append(overall_row)
        if (position, layer) == PILOT_NODE:
            per_template = {row["template"]: row for row in effects if row["position"] == position and int(row["layer"]) == layer and row["direction"] == "shared_loto" and row["aggregation"] == "answer_equal"}
            primary_gate = {"all_four_template_points_positive": all(row["s2"] > 0 for row in per_template.values()), "at_least_three_template_ci_lower_positive": sum(row["ci_low"] > 0 for row in per_template.values()) >= 3, "template_equal_overall_ci_lower_positive": overall_row["ci_low"] > 0, "shared_minus_random_overall_ci_lower_positive": overall_row["shared_minus_random_ci_low"] > 0, "t1_t2_points_positive": per_template["T1"]["s2"] > 0 and per_template["T2"]["s2"] > 0}; primary_gate["passed"] = all(primary_gate.values())
    atomic_csv(root / "tables/steering_effects.csv", effects); atomic_csv(root / "tables/shared_self_random_comparisons.csv", comparisons); atomic_csv(root / "tables/template_equal_overall.csv", overall_rows); atomic_csv(root / "tables/t3_length_sensitivity.csv", t3_sensitivity); atomic_json(root / "progress/pilot_causal_gate.json", primary_gate or {"passed": False, "reason": "pilot_not_analyzed"}); result = {"status": "complete", "nodes": [node_name(*node) for node in nodes], "effect_rows": len(effects), "comparison_rows": len(comparisons), "t3_sensitivity_rows": len(t3_sensitivity), "pilot_gate": primary_gate}; atomic_json(root / "progress/analyze.json", result); return result
