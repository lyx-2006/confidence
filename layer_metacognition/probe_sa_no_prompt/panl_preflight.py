"""Validate a small real-model no-SA capture before the full experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from layer_metacognition.probe.hidden_state_loader import HiddenStateLoader
from layer_metacognition.probe.common import iter_jsonl
from . import HIDDEN_STATE_DEFINITION


REQUIRED_POSITIONS = ("ptnl", "pit", "ac", "lat", "panl")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def validate_preflight(experiment_dir: str | Path, *, layers: list[int] | None = None, expected_items: int = 2) -> dict[str, Any]:
    root = Path(experiment_dir).resolve()
    config = _json(root / "config.json")
    index = _json(root / "hidden_states" / "index.json")
    requested_layers = [int(value) for value in (layers or [27])]
    if config.get("version") == "v3" or config.get("versions") != ["v4"]:
        raise ValueError("Preflight requires V4")
    if config.get("attribution_mode") != "none" or config.get("source_prompt_variant") != "baseline":
        raise ValueError("Preflight requires V4 none/baseline")
    if set(config.get("conditions", [])) != {"conflict_easy", "conflict_hard"}:
        raise ValueError("Preflight capture must contain only conflict conditions")
    available_layers = [int(value) for value in index.get("layer_indices", [])]
    available_positions = [str(value) for value in index.get("position_names", [])]
    missing_layers = set(requested_layers) - set(available_layers)
    missing_positions = set(REQUIRED_POSITIONS) - set(available_positions)
    if missing_layers or missing_positions:
        raise ValueError(f"Preflight hidden schema missing layers={sorted(missing_layers)}, positions={sorted(missing_positions)}")
    cases = index.get("cases", {})
    if not isinstance(cases, dict) or not cases:
        raise ValueError("Preflight hidden index has no cases")
    results_path = root / "results.jsonl"
    completed = []
    for result in iter_jsonl(results_path):
        if result.get("status") != "completed":
            continue
        completed.append(result)
        generated = result.get("generated", {})
        raw_output = generated.get("current_answer_result", {}).get("raw_output", "") if isinstance(generated, dict) and isinstance(generated.get("current_answer_result"), dict) else ""
        if "source attribution" in str(raw_output).lower() or "sa class" in str(raw_output).lower():
            raise ValueError(f"Source Attribution text found in no-SA output: {result.get('case_id')}")
        reference = result.get("hidden_state_reference")
        if not isinstance(reference, dict) or str(result.get("case_id")) not in cases:
            raise ValueError(f"Completed case lacks hidden reference: {result.get('case_id')}")
        if reference.get("hidden_state_definition") != HIDDEN_STATE_DEFINITION:
            raise ValueError(f"Unexpected hidden definition: {result.get('case_id')}")
        positions = result.get("token_positions", {})
        records = result.get("token_position_records", {})
        stages = result.get("token_position_stages", {})
        if "sac" in positions and positions.get("sac") is not None:
            raise ValueError(f"SAC position was created in none mode: {result.get('case_id')}")
        for name in REQUIRED_POSITIONS:
            if positions.get(name) is None or not isinstance(records.get(name), dict):
                raise ValueError(f"Missing {name} token position: {result.get('case_id')}")
            if name == "panl":
                if "\n" not in str(records[name].get("token_text", "")):
                    raise ValueError(f"PANL token does not contain newline: {result.get('case_id')}")
                if stages.get(name) != "answer" and records[name].get("stage") != "answer":
                    raise ValueError(f"PANL is not captured on answer stage: {result.get('case_id')}")
                if int(positions.get("lat")) >= int(positions.get("panl")):
                    raise ValueError(f"LAT is not before PANL: {result.get('case_id')}")
    if len(completed) < 1 or len({str(row.get("item_id")) for row in completed}) < min(expected_items, 1):
        raise ValueError(f"Preflight completed too few items: {len(completed)} records")
    loader = HiddenStateLoader(root)
    sample = completed[0]
    vector = loader.load_vector(sample, layer=requested_layers[0], position_name="panl")
    if vector.ndim != 1 or not vector.size:
        raise ValueError("PANL vector could not be loaded")
    return {"status": "passed", "experiment_dir": str(root), "completed_case_count": len(completed), "completed_item_count": len({str(row.get('item_id')) for row in completed}), "layers": requested_layers, "positions": list(REQUIRED_POSITIONS), "panl_hidden_size": int(vector.shape[0]), "sac_feature": False}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=[27])
    parser.add_argument("--expected-items", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(validate_preflight(args.experiment_dir, layers=args.layers, expected_items=args.expected_items), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
