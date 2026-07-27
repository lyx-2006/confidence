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

from layer_metacognition.hidden_state_store import load_jsonl  # noqa: E402
from layer_metacognition.run_v3_v4_source_experiment import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_PATH,
    expand_attribution_modes,
    main as experiment_main,
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
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
        "--conditions",
        args.condition,
        "--max-items",
        "1",
        "--prior-indices",
        str(args.prior_index),
    ]
    if args.item_id:
        command.extend(["--item-ids", args.item_id])
    if args.resume:
        command.append("--resume")
    exit_code = experiment_main(command)
    if exit_code != 0:
        return exit_code

    output_dir = Path(args.output_dir)
    records = load_jsonl(output_dir / "results.jsonl", repair_trailing=False)
    expected_modes = expand_attribution_modes(args.attribution_mode)
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
        targets = ("ac", "cc") if mode == "none" else ("ac", "cc", "sac")
        for target in targets:
            layers = record["direct_readout"][f"{target}_layers"]
            if len(layers) != expected_layers:
                raise RuntimeError(
                    f"{mode}/{target} layer count mismatch: "
                    f"{len(layers)} != {expected_layers}"
                )
        if mode == "none" and record["direct_readout"]["sac_layers"]:
            raise RuntimeError("none mode unexpectedly produced SAC readout")
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
    print(
        json.dumps(
            {
                "status": "passed",
                "case_ids": [
                    records_by_mode[mode]["case_id"] for mode in expected_modes
                ],
                "modes": list(expected_modes),
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
