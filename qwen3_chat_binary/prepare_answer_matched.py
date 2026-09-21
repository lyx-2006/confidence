from __future__ import annotations

import argparse
import hashlib
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from dp_SA.io_utils import atomic_json, atomic_jsonl, canonical_hash, load_jsonl, sha256_file

from .config import (
    ANSWER_MATCHED_OUTPUT_ROOT,
    CAPTURE_ROOT,
    EXPECTED_HIDDEN_SIZE,
    SEED,
    VECTOR_NORM_FRACTION,
)
from .contracts import ensure_fingerprinted_config, hidden_key, parse_layers, parse_positions
from .layout import capture_config_path, capture_results_path, ensure_output_layout


VARIANT = "native_boundary"
ANSWER_MATCHED_POSITIONS = ("LAT", "PANL", "CLE")
ANSWER_MATCHED_LAYERS = (8, 12, 18, 24, 30, 35)
TEST_COUNT = 79
MIN_PER_SIDE = 3
CANONICAL_ANSWERS = (
    "black", "blue", "brown", "cyan", "gray", "green",
    "orange", "pink", "purple", "red", "white", "yellow",
)


def sa_group(row: dict[str, Any]) -> str:
    label = str(row.get("predicted_label"))
    if label in {"0", "1"}:
        return "text"
    if label in {"3", "4"}:
        return "image"
    if label in {"2", "tie"}:
        return "neutral"
    raise ValueError(f"Invalid five-class predicted label: {label!r}")


def _stable_key(case_id: str) -> str:
    return hashlib.sha256(f"{SEED}|answer_matched_fixed|{case_id}".encode()).hexdigest()


def fixed_split(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if len(rows) != 396 or len({str(row["case_id"]) for row in rows}) != 396:
        raise ValueError("Answer-matched fixed split requires exactly 396 unique completed cases")
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for source in rows:
        row = dict(source)
        answer = str(row["phase0_normalized_answer"])
        if answer not in CANONICAL_ANSWERS:
            raise ValueError(f"Non-canonical fixed answer: {answer!r}")
        row["answer"] = answer
        row["sa_group"] = sa_group(row)
        strata[answer, row["sa_group"]].append(row)
    exact = {key: len(values) * TEST_COUNT / len(rows) for key, values in strata.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remaining = TEST_COUNT - sum(quotas.values())
    order = sorted(strata, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in order[:remaining]:
        quotas[key] += 1
    test_ids: set[str] = set()
    for key, values in sorted(strata.items()):
        ordered = sorted(values, key=lambda row: (_stable_key(str(row["case_id"])), str(row["case_id"])))
        test_ids.update(str(row["case_id"]) for row in ordered[:quotas[key]])
    enriched = [
        {
            **row,
            "answer": str(row["phase0_normalized_answer"]),
            "sa_group": sa_group(row),
        }
        for row in rows
    ]
    construction = sorted((row for row in enriched if str(row["case_id"]) not in test_ids), key=lambda row: str(row["case_id"]))
    test = sorted((row for row in enriched if str(row["case_id"]) in test_ids), key=lambda row: str(row["case_id"]))
    if len(construction) != 317 or len(test) != 79:
        raise AssertionError("Fixed split size mismatch")
    if {row["case_id"] for row in construction} & {row["case_id"] for row in test}:
        raise AssertionError("Construction/test case leakage")
    counts = Counter((row["answer"], row["sa_group"]) for row in construction)
    eligible = [
        answer for answer in CANONICAL_ANSWERS
        if counts[answer, "text"] >= MIN_PER_SIDE and counts[answer, "image"] >= MIN_PER_SIDE
    ]
    if len(eligible) < 4:
        raise ValueError(f"Only {len(eligible)} answers meet the construction gate")
    if any(len(set(eligible) - {row["answer"]}) < 3 for row in test):
        raise ValueError("A test answer has fewer than three LOAO construction answers")
    summary = {
        "split_unit": "case_id", "seed": SEED,
        "construction_count": len(construction), "test_count": len(test),
        "test_group_counts": dict(Counter(row["sa_group"] for row in test)),
        "test_answer_counts": dict(Counter(row["answer"] for row in test)),
        "eligible_answers": eligible,
        "ineligible_answers": [answer for answer in CANONICAL_ANSWERS if answer not in eligible],
        "construction_cell_counts": {
            answer: {side: counts[answer, side] for side in ("text", "image", "neutral")}
            for answer in CANONICAL_ANSWERS
        },
        "test_all_answers_covered": set(row["answer"] for row in test) == set(CANONICAL_ANSWERS),
    }
    return construction, test, summary


def smoke_subset(test: Sequence[dict[str, Any]], count: int = 20) -> list[dict[str, Any]]:
    remaining = {str(row["case_id"]): row for row in test}
    uncovered = (
        {("answer", value) for value in CANONICAL_ANSWERS}
        | {("pair", value) for value in sorted({str(row["pair_type"]) for row in test})}
        | {("group", value) for value in ("text", "image", "neutral")}
    )
    selected: list[dict[str, Any]] = []
    while uncovered:
        choices = []
        for case_id, row in remaining.items():
            features = {
                ("answer", str(row["answer"])),
                ("pair", str(row["pair_type"])),
                ("group", str(row["sa_group"])),
            }
            choices.append((len(features & uncovered), _stable_key(case_id), case_id, features))
        gain, _hash, case_id, features = min(choices, key=lambda value: (-value[0], value[1], value[2]))
        if gain == 0:
            raise ValueError(f"Smoke coverage is infeasible; uncovered={sorted(uncovered)}")
        selected.append(remaining.pop(case_id))
        uncovered -= features
    for case_id in sorted(remaining, key=lambda value: (_stable_key(value), value)):
        if len(selected) == count:
            break
        selected.append(remaining[case_id])
    if len(selected) != count:
        raise ValueError(f"Could not select {count} smoke cases")
    return sorted(selected, key=lambda row: str(row["case_id"]))


def _load_hidden(capture_root: Path, row: dict[str, Any], position: str, layer: int) -> np.ndarray:
    path = capture_root / VARIANT / str(row["hidden_file"])
    key = hidden_key(position, layer)
    with np.load(path) as payload:
        if key not in payload:
            raise KeyError(f"{path} does not contain {key}")
        value = np.asarray(payload[key], dtype=np.float32)
    if value.shape != (EXPECTED_HIDDEN_SIZE,) or not np.isfinite(value).all():
        raise ValueError(f"Invalid hidden state: {path} {key} {value.shape}")
    return value


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def loao_scaled_direction(
    direction_by_answer: dict[str, np.ndarray],
    hidden_by_answer: dict[str, Sequence[np.ndarray]],
    recipient: str,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, float]]:
    included = sorted(answer for answer in direction_by_answer if answer != recipient)
    if len(included) < 3:
        raise ValueError(f"LOAO leaves only {len(included)} answers for {recipient}")
    raw = np.stack([direction_by_answer[answer] for answer in included]).mean(0).astype(np.float32)
    raw_norm = float(np.linalg.norm(raw))
    residuals = [value for answer in included for value in hidden_by_answer[answer]]
    mean_residual_norm = float(np.mean([np.linalg.norm(value) for value in residuals]))
    target_norm = VECTOR_NORM_FRACTION * mean_residual_norm
    if not all(math.isfinite(value) and value > 0 for value in (raw_norm, mean_residual_norm, target_norm)):
        raise ValueError(f"Invalid vector norm for recipient {recipient}")
    scaled = np.asarray(raw / raw_norm * target_norm, dtype=np.float32)
    return raw, scaled, included, {
        "raw_norm": raw_norm, "mean_residual_norm": mean_residual_norm,
        "target_norm": target_norm, "scaled_norm": float(np.linalg.norm(scaled)),
    }


def build_vectors(
    capture_root: Path,
    construction: Sequence[dict[str, Any]],
    test: Sequence[dict[str, Any]],
    positions: Sequence[str],
    layers: Sequence[int],
    output_root: Path,
) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in construction:
        if row["sa_group"] in {"text", "image"}:
            groups[str(row["answer"]), str(row["sa_group"])].append(row)
    eligible = [
        answer for answer in CANONICAL_ANSWERS
        if len(groups[answer, "text"]) >= MIN_PER_SIDE and len(groups[answer, "image"]) >= MIN_PER_SIDE
    ]
    recipients = sorted({str(row["answer"]) for row in test}, key=CANONICAL_ANSWERS.index)
    vector_rows: list[dict[str, Any]] = []
    for position in positions:
        for layer in layers:
            direction_by_answer: dict[str, np.ndarray] = {}
            hidden_by_answer: dict[str, list[np.ndarray]] = {}
            cell_counts: dict[str, dict[str, int]] = {}
            for answer in eligible:
                text = [_load_hidden(capture_root, row, position, layer) for row in groups[answer, "text"]]
                image = [_load_hidden(capture_root, row, position, layer) for row in groups[answer, "image"]]
                direction_by_answer[answer] = np.stack(image).mean(0) - np.stack(text).mean(0)
                hidden_by_answer[answer] = image + text
                cell_counts[answer] = {"text": len(text), "image": len(image)}
            arrays: dict[str, np.ndarray] = {}
            pending: list[dict[str, Any]] = []
            for recipient in recipients:
                raw, scaled, included, norms = loao_scaled_direction(
                    direction_by_answer, hidden_by_answer, recipient
                )
                arrays[f"raw__{recipient}"] = raw
                arrays[f"scaled__{recipient}"] = scaled
                pending.append({
                    "position": position, "layer": int(layer), "recipient_answer": recipient,
                    "included_answers": included, "included_answer_count": len(included),
                    "cell_counts": {answer: cell_counts[answer] for answer in included},
                    "included_case_count": sum(sum(cell_counts[answer].values()) for answer in included),
                    **norms,
                    "scaled_key": f"scaled__{recipient}", "scaled_array_sha256": _array_hash(scaled),
                })
            relative = Path("native_boundary") / "tables" / "vectors" / f"{position}__L{layer}.npz"
            destination = output_root / relative
            _atomic_npz(destination, arrays)
            file_hash = sha256_file(destination)
            vector_rows.extend({**row, "vector_file": str(relative), "vector_file_sha256": file_hash} for row in pending)
    metadata = {
        "status": "complete", "variant": VARIANT, "direction": "matched_loao",
        "normalization_fraction": VECTOR_NORM_FRACTION, "vectors": vector_rows,
        "vector_count": len(vector_rows),
    }
    metadata["fingerprint"] = canonical_hash(vector_rows)
    atomic_json(output_root / VARIANT / "tables" / "vector_metadata.json", metadata)
    return metadata


def prepare(
    *, capture_root: Path = CAPTURE_ROOT, output_root: Path = ANSWER_MATCHED_OUTPUT_ROOT,
    positions: Sequence[str] = ANSWER_MATCHED_POSITIONS,
    layers: Sequence[int] = ANSWER_MATCHED_LAYERS,
    smoke: bool = False, resume: bool = False,
) -> dict[str, Any]:
    capture_root, output_root = capture_root.resolve(), output_root.resolve()
    positions, layers = parse_positions(positions), parse_layers(layers)
    capture_config = capture_config_path(capture_root)
    capture_results = capture_results_path(capture_root, VARIANT)
    if not capture_config.is_file() or not capture_results.is_file():
        raise FileNotFoundError("Native-boundary capture artifacts are missing")
    rows = [row for row in load_jsonl(capture_results) if row.get("status") == "completed"]
    construction, test, split_summary = fixed_split(rows)
    smoke_rows = smoke_subset(test)
    ensure_output_layout(output_root, (VARIANT,))
    payload = {
        "format_version": 1, "experiment": "native_answer_matched_steering_prepare",
        "capture_root": str(capture_root), "capture_config_sha256": sha256_file(capture_config),
        "capture_results_sha256": sha256_file(capture_results), "variant": VARIANT,
        "positions": list(positions), "layers": list(layers), "seed": SEED,
        "test_count": TEST_COUNT, "minimum_per_answer_side": MIN_PER_SIDE,
        "normalization_fraction": VECTOR_NORM_FRACTION, "smoke_output": bool(smoke),
        "split_fingerprint": canonical_hash({
            "construction": [row["case_id"] for row in construction],
            "test": [row["case_id"] for row in test],
        }),
    }
    config = ensure_fingerprinted_config(
        output_root / "progress" / "prepare_config.json", payload,
        resume=resume, label="Answer-matched prepare",
    )
    metadata_path = output_root / VARIANT / "tables" / "vector_metadata.json"
    if resume and metadata_path.is_file():
        metadata = __import__("json").loads(metadata_path.read_text(encoding="utf-8"))
        for row in metadata["vectors"]:
            path = output_root / row["vector_file"]
            if not path.is_file() or sha256_file(path) != row["vector_file_sha256"]:
                raise ValueError("Answer-matched vector artifact fingerprint mismatch")
        return {
            "status": "complete", "resumed_noop": True,
            "construction_count": len(construction), "test_count": len(test),
            "smoke_test_count": len(smoke_rows), "vector_count": metadata["vector_count"],
            "config_fingerprint": config["fingerprint"],
        }
    atomic_jsonl(output_root / "tables" / "construction_manifest.jsonl", construction)
    atomic_jsonl(output_root / "tables" / "test_manifest.jsonl", test)
    atomic_jsonl(output_root / "tables" / "smoke_manifest.jsonl", smoke_rows)
    atomic_json(output_root / "tables" / "split_summary.json", split_summary)
    metadata = build_vectors(capture_root, construction, test, positions, layers, output_root)
    summary = {
        "status": "complete", "resumed_noop": False,
        "construction_count": len(construction), "test_count": len(test),
        "smoke_test_count": len(smoke_rows), "vector_count": metadata["vector_count"],
        "eligible_answers": split_summary["eligible_answers"],
        "config_fingerprint": config["fingerprint"],
    }
    atomic_json(output_root / "progress" / "prepare_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare native-boundary answer-matched LOAO steering")
    parser.add_argument("--capture-root", type=Path, default=CAPTURE_ROOT)
    parser.add_argument("--output-root", type=Path, default=ANSWER_MATCHED_OUTPUT_ROOT)
    parser.add_argument("--positions", nargs="+", default=list(ANSWER_MATCHED_POSITIONS))
    parser.add_argument("--layers", nargs="+", type=int, default=list(ANSWER_MATCHED_LAYERS))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    print(prepare(**vars(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
