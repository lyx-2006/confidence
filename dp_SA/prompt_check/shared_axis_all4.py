from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .io_utils import array_hash, atomic_csv, atomic_json, atomic_jsonl, atomic_npz, canonical_hash, inventory, verify_inventory
from .shared_axis_loto import DEFAULT_NODES, PROBE_RUN_ROOT, TEMPLATES, load_unit_probes, node_name, parse_nodes, uncentered_shared_axis


DEFAULT_ROOT = Path(__file__).resolve().parent / "output/shared_axis_all4"


def run(root: Path, nodes: Sequence[tuple[str, int]], *, resume: bool = False) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    units, probe_audit, probe_paths = load_unit_probes(nodes)
    source_paths = [PROBE_RUN_ROOT / "artifacts/probes/probe_index.jsonl", PROBE_RUN_ROOT / "artifacts/manifests/construction_manifest.jsonl", *probe_paths]
    sources = inventory(source_paths)
    code = inventory([Path(__file__), Path(__file__).with_name("shared_axis_loto.py")])
    config = {
        "format_version": 1,
        "experiment": "t0_t3_all4_shared_sa_axis_geometry_only",
        "templates": list(TEMPLATES),
        "nodes": [node_name(*node) for node in nodes],
        "construction": "uncentered_svd_of_four_unit_raw_probe_weights",
        "sign_rule": "mean_probe_projection_positive",
        "steering_executed": False,
        "source_hashes": sources,
        "implementation_hashes": code,
    }
    fingerprint = canonical_hash(config)
    config_path = root / "artifacts/config_and_fingerprint.json"
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old.get("fingerprint") != fingerprint:
            raise ValueError("Resume configuration fingerprint mismatch")
        if not resume:
            raise FileExistsError(config_path)
        completion = root / "completion.json"
        if completion.exists() and json.loads(completion.read_text()).get("status") == "complete":
            return {**json.loads(completion.read_text()), "resumed_noop": True}

    atomic_json(config_path, {**config, "fingerprint": fingerprint})
    atomic_json(root / "artifacts/source_hashes_before.json", sources)
    atomic_jsonl(root / "tables/probe_direction_audit.jsonl", probe_audit)

    geometry: list[dict[str, Any]] = []
    pairwise: list[dict[str, Any]] = []
    probe_to_shared: list[dict[str, Any]] = []
    axis_index: list[dict[str, Any]] = []
    for position, layer in nodes:
        ordered = [units[template, position, layer] for template in TEMPLATES]
        shared, singular = uncentered_shared_axis(ordered)
        projections = np.stack(ordered) @ shared
        if float(np.mean(projections)) <= 0:
            raise ValueError("Shared-axis sign rule failed")
        energy = singular**2 / np.sum(singular**2)
        node = node_name(position, layer)
        axis_path = root / f"artifacts/axes/{node}.npz"
        arrays = {f"unit_{template}": units[template, position, layer].astype(np.float32) for template in TEMPLATES}
        arrays["shared_all4"] = shared.astype(np.float32)
        arrays["singular_values"] = singular.astype(np.float32)
        atomic_npz(axis_path, arrays)
        axis_index.append({
            "position": position,
            "layer": layer,
            "axis_file": str(axis_path.relative_to(root)),
            "shared_axis_sha256": array_hash(arrays["shared_all4"]),
            "shared_axis_unit_norm": float(np.linalg.norm(shared)),
            "mean_probe_projection": float(np.mean(projections)),
            "steering_executed": False,
        })
        geometry.append({
            "position": position,
            "layer": layer,
            "minimum_probe_to_shared_cosine": float(np.min(projections)),
            "mean_probe_to_shared_cosine": float(np.mean(projections)),
            "sigma1": float(singular[0]),
            "sigma2": float(singular[1]),
            "sigma3": float(singular[2]),
            "sigma4": float(singular[3]),
            "first_axis_energy": float(energy[0]),
            "first_two_axes_energy": float(energy[:2].sum()),
            "sigma1_over_sigma2": float(singular[0] / singular[1]),
            "single_axis_flag": bool(energy[0] >= 0.60),
        })
        for index, template in enumerate(TEMPLATES):
            probe_to_shared.append({"position": position, "layer": layer, "template": template, "cosine_to_all4_shared": float(projections[index])})
        for left_index, left in enumerate(TEMPLATES):
            for right in TEMPLATES[left_index + 1 :]:
                cosine = float(units[left, position, layer] @ units[right, position, layer])
                pairwise.append({"position": position, "layer": layer, "template_a": left, "template_b": right, "signed_cosine": cosine, "absolute_cosine": abs(cosine)})

    atomic_csv(root / "tables/geometry_summary.csv", geometry)
    atomic_csv(root / "tables/probe_to_shared_cosines.csv", probe_to_shared)
    atomic_csv(root / "tables/pairwise_probe_cosines.csv", pairwise)
    atomic_jsonl(root / "artifacts/axes/axis_index.jsonl", axis_index)
    verify_inventory(sources)
    atomic_json(root / "artifacts/source_hashes_after.json", inventory(source_paths))
    gates = {
        "node_count_5": len(geometry) == 5,
        "probe_to_shared_rows_20": len(probe_to_shared) == 20,
        "pairwise_rows_30": len(pairwise) == 30,
        "all_axes_unit_norm": all(abs(row["shared_axis_unit_norm"] - 1.0) <= 1e-10 for row in axis_index),
        "all_sign_rules_positive": all(row["mean_probe_projection"] > 0 for row in axis_index),
        "steering_not_executed": all(not row["steering_executed"] for row in axis_index),
        "historical_sources_unchanged": True,
    }
    if not all(gates.values()):
        raise ValueError(f"Acceptance failed: {gates}")
    result = {"status": "complete", "fingerprint": fingerprint, "node_count": len(geometry), "probe_count": len(probe_to_shared), "pairwise_count": len(pairwise), "steering_executed": False, "gates": gates, "completed_at_unix": time.time()}
    atomic_json(root / "completion.json", result)
    readme = """# T0–T3 All-4 Shared SA Axis

本实验在每个预注册位置/层独立使用 T0–T3 四个 canonical SA probe 的原始 hidden-space 单位方向，进行不中心化 SVD，并以四个 probe 的平均投影为正规定符号。

本目录只包含方向构造与几何相似度审计。没有执行 steering；all-4 轴使用了目标模板自身的 probe，因此只能作为描述性共同轴/上界，不能替代 LOTO 的跨模板泛化证据。
"""
    (root / "README_zh.md").write_text(readme, encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", nargs="+", default=[f"{position}:{layer}" for position, layer in DEFAULT_NODES])
    parser.add_argument("--output-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    nodes = parse_nodes(args.nodes)
    print(json.dumps(run(Path(args.output_root), nodes, resume=args.resume), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
