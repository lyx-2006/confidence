#!/usr/bin/env python3
"""Run and validate one full-layer V3/V4 source/sink GPU case."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layer_metacognition.hidden_state_store import (  # noqa: E402
    TargetLayerHiddenStateStore,
    load_jsonl,
)
from layer_metacognition.analyze_source_sink_results import (  # noqa: E402
    COMPACT_LAYER_COLUMNS,
    COMPACT_LAYER_COLUMNS_WITH_ANSWER_VAL,
)
from layer_metacognition.run_v3_v4_source_experiment import (  # noqa: E402
    ANALYSIS_MODE_ORDER,
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_PATH,
    expand_attribution_modes,
    main as experiment_main,
    normalize_analysis_modes,
    normalize_save_hidden_states,
    parse_save_hidden_state,
    serialize_save_hidden_states,
)
from layer_metacognition.source_patchscope import (  # noqa: E402
    ANSWER_PATCHSCOPE_VARIANTS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument(
        "--output-dir",
        default="layer_metacognition/output/smoke_v3_v4_source",
    )
    parser.add_argument("--item-id")
    parser.add_argument("--prior-index", type=int, default=0)
    parser.add_argument("--condition", default="consistent_easy")
    parser.add_argument("--version", choices=["v3", "v4"], default="v4")
    parser.add_argument(
        "--attribution-mode",
        choices=["none", "parallel", "joint", "all"],
        default="joint",
    )
    parser.add_argument(
        "--analysis_mode",
        nargs="+",
        choices=list(ANALYSIS_MODE_ORDER),
        default=["LMhead"],
    )
    parser.add_argument("--skip-attention", action="store_true")
    parser.add_argument("--answer_val", action="store_true")
    parser.add_argument(
        "--save_hidden_state",
        type=parse_save_hidden_state,
        nargs="+",
        default=[None],
        metavar="none|LAYER",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    save_hidden_states = normalize_save_hidden_states(args.save_hidden_state)
    if not torch.cuda.is_available():
        print("[SKIP] CUDA is not visible; full attention smoke test requires a GPU.")
        return 2
    command = [
        "--model-path",
        args.model_path,
        "--dataset",
        args.dataset,
        "--output-dir",
        args.output_dir,
        "--versions",
        args.version,
        "--attribution-mode",
        args.attribution_mode,
        "--analysis_mode",
        *args.analysis_mode,
        "--conditions",
        args.condition,
        "--max-items",
        "1",
        "--prior-indices",
        str(args.prior_index),
    ]
    if args.item_id:
        command.extend(["--item-ids", args.item_id])
    if args.skip_attention:
        command.append("--skip-attention")
    if args.answer_val:
        command.append("--answer_val")
    if save_hidden_states:
        command.extend(
            ["--save_hidden_state", *(str(value) for value in save_hidden_states)]
        )
    if args.resume:
        command.append("--resume")
    exit_code = experiment_main(command)
    if exit_code != 0:
        return exit_code

    output_dir = Path(args.output_dir)
    records = load_jsonl(output_dir / "results.jsonl", repair_trailing=False)
    expected_modes = expand_attribution_modes(args.attribution_mode)
    analysis_modes = normalize_analysis_modes(args.analysis_mode)
    if len(records) != len(expected_modes):
        raise RuntimeError(
            f"Smoke output must contain {len(expected_modes)} record(s), "
            f"got {len(records)}"
        )
    records_by_mode = {
        str(record.get("attribution_mode")): record for record in records
    }
    if set(records_by_mode) != set(expected_modes):
        raise RuntimeError(
            f"Smoke modes mismatch: {sorted(records_by_mode)} != "
            f"{list(expected_modes)}"
        )
    for mode in expected_modes:
        record = records_by_mode[mode]
        if record.get("status") != "completed":
            raise RuntimeError(
                f"Smoke case failed for mode={mode}: {record.get('error')}"
            )
        expected_layers = int(record["model_structure"]["num_hidden_layers"])
        expected_heads = int(record["model_structure"]["num_attention_heads"])
        for target in ("ac", "cc"):
            layers = record["direct_readout"][f"{target}_layers"]
            if len(layers) != expected_layers:
                raise RuntimeError(
                    f"{mode}/{target} layer count mismatch: "
                    f"{len(layers)} != {expected_layers}"
                )
        answer_patchscope_layers = record["direct_readout"][
            "answer_patchscope_layers"
        ]
        if len(answer_patchscope_layers) != expected_layers:
            raise RuntimeError(
                f"{mode}/answer_patchscope layer count mismatch: "
                f"{len(answer_patchscope_layers)} != {expected_layers}"
            )
        for layer in answer_patchscope_layers:
            probabilities = layer.get("answer_class_probabilities")
            if (
                not isinstance(probabilities, dict)
                or not probabilities
                or abs(sum(float(value) for value in probabilities.values()) - 1.0)
                >= 1e-5
            ):
                raise RuntimeError(
                    f"{mode}/answer_patchscope has invalid answer probabilities"
                )
            predicted = layer.get("predicted_answer")
            if predicted not in probabilities:
                raise RuntimeError(
                    f"{mode}/answer_patchscope predicted an unknown answer"
                )
            if abs(
                float(layer["predicted_answer_probability"])
                - float(probabilities[predicted])
            ) >= 1e-6:
                raise RuntimeError(
                    f"{mode}/answer_patchscope top-1 probability mismatch"
                )
        answer_patchscope_check = record["validation"].get(
            "answer_patchscope_last_layer"
        )
        if (
            not isinstance(answer_patchscope_check, dict)
            or not answer_patchscope_check.get("passed")
        ):
            raise RuntimeError(
                f"{mode}/answer_patchscope final-layer validation failed: "
                f"{answer_patchscope_check}"
            )
        answer_variants = record["direct_readout"].get(
            "answer_patchscope_layers_by_variant"
        )
        expected_answer_variants = (
            set(ANSWER_PATCHSCOPE_VARIANTS)
            if args.answer_val
            else {"original"}
        )
        if set(answer_variants or {}) != expected_answer_variants:
            raise RuntimeError(
                f"{mode}/answer Patchscope variants do not match "
                f"{sorted(expected_answer_variants)}"
            )
        for variant, layers in answer_variants.items():
            if len(layers) != expected_layers:
                raise RuntimeError(
                    f"{mode}/{variant} answer layer count mismatch"
                )
        answer_validation = record["validation"].get(
            "answer_patchscope_by_variant"
        )
        for variant in expected_answer_variants:
            check = (answer_validation or {}).get(variant)
            if not isinstance(check, dict) or not check.get("passed"):
                raise RuntimeError(
                    f"{mode}/{variant} answer final-layer validation failed"
                )
        if mode == "none" and record["direct_readout"]["sac_layers"]:
            raise RuntimeError("none mode unexpectedly produced SAC readout")
        sac_by_mode = record["direct_readout"]["sac_layers_by_mode"]
        if set(sac_by_mode) != set(ANALYSIS_MODE_ORDER):
            raise RuntimeError(f"{mode} has invalid sac_layers_by_mode keys")
        for analysis_mode in ANALYSIS_MODE_ORDER:
            layers = sac_by_mode[analysis_mode]
            expected_count = (
                expected_layers
                if mode != "none" and analysis_mode in analysis_modes
                else 0
            )
            if len(layers) != expected_count:
                raise RuntimeError(
                    f"{mode}/{analysis_mode} layer count mismatch: "
                    f"{len(layers)} != {expected_count}"
                )
            for layer in layers:
                probabilities = layer.get("class_probabilities")
                if (
                    not isinstance(probabilities, list)
                    or len(probabilities) != 9
                    or abs(sum(float(value) for value in probabilities) - 1.0)
                    >= 1e-5
                ):
                    raise RuntimeError(
                        f"{mode}/{analysis_mode} has invalid class probabilities"
                    )
                score = float(layer["soft_image_score"])
                if not 0.0 <= score <= 1.0:
                    raise RuntimeError(
                        f"{mode}/{analysis_mode} score is outside [0, 1]: {score}"
                    )
        legacy_sac = record["direct_readout"]["sac_layers"]
        if "LMhead" in analysis_modes and mode != "none":
            if legacy_sac != sac_by_mode["LMhead"]:
                raise RuntimeError("Legacy sac_layers differs from LMhead results")
        elif legacy_sac:
            raise RuntimeError("Legacy sac_layers must be empty without LMhead")
        validation = record["validation"]["sac_by_mode"]
        for analysis_mode in ANALYSIS_MODE_ORDER:
            check = validation[analysis_mode]
            should_run = mode != "none" and analysis_mode in analysis_modes
            if should_run and (not isinstance(check, dict) or not check.get("passed")):
                raise RuntimeError(
                    f"{mode}/{analysis_mode} final-layer validation failed: {check}"
                )
            if not should_run and check is not None:
                raise RuntimeError(
                    f"{mode}/{analysis_mode} validation unexpectedly ran"
                )
        expected_baselines = {
            analysis_mode
            for analysis_mode in analysis_modes
            if analysis_mode in ("Identity", "Semantic")
        }
        if set(record["patchscope_baselines"]) != expected_baselines:
            raise RuntimeError(
                f"{mode} patchscope baseline keys do not match selected targets"
            )
        for target, sources in record["attention_sinks"].items():
            for source, value in sources.items():
                if len(value["layers"]) != expected_layers:
                    raise RuntimeError(
                        f"{mode}/{target}/{source} attention layer count mismatch"
                    )
                for layer, metrics in value["layers"].items():
                    if len(metrics["sink_score_by_head"]) != expected_heads:
                        raise RuntimeError(
                            f"{mode}/{target}/{source}/layer={layer} "
                            "head count mismatch"
                        )
    required = {
        "config.json",
        "progress.json",
        "results.jsonl",
        "analysis_minimal.json",
        "analysis_layer_readout_minimal_v3.json",
        "analysis_layer_readout_minimal_v4.json",
        "analysis_source_sink_minimal.json",
        "summary.json",
        "run.log",
    }
    missing = sorted(name for name in required if not (output_dir / name).is_file())
    if missing:
        raise RuntimeError(f"Smoke output is missing files: {missing}")
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("analysis_modes") != list(analysis_modes):
        raise RuntimeError("config.json analysis_modes are not canonical")
    if bool(config.get("answer_val")) != args.answer_val:
        raise RuntimeError("config.json answer_val does not match smoke request")
    expected_saved_layer = serialize_save_hidden_states(save_hidden_states)
    if config.get("save_hidden_state") != expected_saved_layer:
        raise RuntimeError(
            "config.json save_hidden_state does not match smoke request"
        )
    if save_hidden_states:
        store = TargetLayerHiddenStateStore(
            output_dir,
            layer_index=list(save_hidden_states),
        )
        for record in records:
            if record.get("status") != "completed":
                continue
            restored, reference = store.read_case(str(record["case_id"]))
            hidden_size = int(reference["hidden_size"])
            expected_shape = (
                (2, hidden_size)
                if len(save_hidden_states) == 1
                else (len(save_hidden_states), 2, hidden_size)
            )
            if tuple(restored.shape) != expected_shape:
                raise RuntimeError(
                    f"Hidden-state shape mismatch: {tuple(restored.shape)}"
                )
            if restored.device.type != "cpu" or restored.dtype != torch.float16:
                raise RuntimeError("Stored hidden state is not CPU FP16")
            if reference["position_names"] != ["ac", "panl"]:
                raise RuntimeError("Stored hidden-state position order is invalid")
    expected_compact_columns = list(
        COMPACT_LAYER_COLUMNS_WITH_ANSWER_VAL
        if args.answer_val
        else COMPACT_LAYER_COLUMNS
    )
    if config.get("compact_layer_columns") != expected_compact_columns:
        raise RuntimeError("config.json has unexpected compact layer columns")
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    if bool((summary.get("answer_validation") or {}).get("enabled")) != args.answer_val:
        raise RuntimeError("summary.json answer validation status is incorrect")
    if set(summary.get("sac_readout_coverage_by_mode", {})) != set(
        ANALYSIS_MODE_ORDER
    ):
        raise RuntimeError("summary.json lacks per-mode SAC coverage")
    for filename in (
        "analysis_source_sink_minimal.json",
        f"analysis_layer_readout_minimal_{args.version}.json",
    ):
        minimal = json.loads((output_dir / filename).read_text(encoding="utf-8"))
        for minimal_record in minimal:
            for values in minimal_record.get("layers", {}).values():
                if len(values) != len(expected_compact_columns):
                    raise RuntimeError(
                        f"{filename} contains an unexpected layer row length"
                    )

    for mode in expected_modes:
        if mode == "none":
            continue
        record = records_by_mode[mode]
        print(f"[SAC] case_id={record['case_id']}")
        print("layer\tLMhead\tIdentity\tSemantic")
        maps = {
            analysis_mode: {
                int(layer["layer_index"]): float(layer["soft_image_score"])
                for layer in record["direct_readout"]["sac_layers_by_mode"][
                    analysis_mode
                ]
            }
            for analysis_mode in ANALYSIS_MODE_ORDER
        }
        for layer_index in range(int(record["model_structure"]["num_hidden_layers"])):
            values = [
                (
                    f"{maps[analysis_mode][layer_index]:.6f}"
                    if layer_index in maps[analysis_mode]
                    else "null"
                )
                for analysis_mode in ANALYSIS_MODE_ORDER
            ]
            print(f"{layer_index}\t" + "\t".join(values))
    print(
        json.dumps(
            {
                "status": "passed",
                "case_ids": [
                    records_by_mode[mode]["case_id"] for mode in expected_modes
                ],
                "modes": list(expected_modes),
                "analysis_modes": list(analysis_modes),
                "layers": expected_layers,
                "heads": expected_heads,
                "output_dir": str(output_dir.resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
