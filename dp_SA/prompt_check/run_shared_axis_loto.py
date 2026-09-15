from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .io_utils import atomic_json
from .shared_axis_loto import DEFAULT_NODES, PILOT_NODE, PROBE_RUN_ROOT, TEMPLATES, node_name, parse_nodes, prepare_cpu, verify_sources
from .shared_axis_steering import analyze, run_steering


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = PACKAGE_ROOT / "output/shared_axis_loto"
SMOKE_ROOT = PACKAGE_ROOT / "output/shared_axis_loto_smoke"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle: return list(csv.DictReader(handle))


def make_plots(root: Path) -> list[str]:
    output = []; geometry = _rows(root / "tables/geometry_summary.csv"); effects = _rows(root / "tables/steering_effects.csv")
    labels = [node_name(row["position"], int(row["layer"])) for row in geometry]
    fig, ax = plt.subplots(figsize=(9, 4)); ax.plot(labels, [float(row["first_axis_energy"]) for row in geometry], marker="o", label="E1"); ax.plot(labels, [float(row["first_two_axes_energy"]) for row in geometry], marker="o", label="E1+E2"); ax.axhline(.6, color="gray", linestyle="--", linewidth=1); ax.set_ylabel("SVD energy fraction"); ax.tick_params(axis="x", rotation=25); ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); path = root / "figures/geometry_energy.png"; path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=220); plt.close(fig); output.append(str(path))
    selected = [row for row in effects if row["direction"] == "shared_loto" and row["aggregation"] == "answer_equal"]
    if selected:
        fig, ax = plt.subplots(figsize=(10, 4)); colors = {"T0": "black", "T1": "#2166ac", "T2": "#4daf4a", "T3": "#b2182b"}
        for template in TEMPLATES:
            rows = [row for row in selected if row["template"] == template]; x = range(len(rows)); y = [float(row["s2"]) for row in rows]; lo = [float(row["ci_low"]) for row in rows]; hi = [float(row["ci_high"]) for row in rows]; ax.errorbar(x, y, yerr=[[a-b for a,b in zip(y,lo)], [b-a for a,b in zip(y,hi)]], marker="o", label=template, color=colors[template])
        ax.axhline(0, color="gray", linewidth=1); ax.set_xticks(range(len(rows)), [node_name(row["position"], int(row["layer"])) for row in rows], rotation=25); ax.set_ylabel("Shared LOTO S²"); ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); path = root / "figures/shared_loto_s2.png"; fig.savefig(path, dpi=220); plt.close(fig); output.append(str(path))
    return output


def _readme(root: Path, cpu: dict[str, Any], analysis: dict[str, Any], expanded: bool) -> None:
    gate = analysis.get("pilot_gate") or {}; status = "通过，已扩展五节点" if expanded else "未通过，按预注册规则停在主节点"
    text = f"""# T0–T3 LOTO Shared-SA Axis

本实验使用不中心化SVD构造共享方向。每条LOTO轴只使用另外三个模板，方向符号也不读取目标模板probe。

Shared、self matched-dose和random均使用同一注入尺度 `sigma_loto`；`self_natural_scale`仅是敏感性分析，不作为保持率分母。

主分析使用95条confirmatory，全部100条family-micro为补充。Probe线性可解码性和几何相似不等同于因果作用。

主节点门槛：{status}。

CPU geometry gate: `{json.dumps(cpu.get('gpu_gate',{}),ensure_ascii=False)}`

Pilot causal gate: `{json.dumps(gate,ensure_ascii=False)}`

若T3完整序列与length-normalized方向不一致，T3稳定迁移证据不足。若第一轴能量低而二维累计能量明显更高，结果解释为共享子空间而不是稳定单轴。
"""
    (root / "README_zh.md").write_text(text, encoding="utf-8")


def verify(root: Path, executed_nodes: Sequence[tuple[str, int]], validation_cases: int, random_count: int) -> dict[str, bool]:
    import math
    from .io_utils import load_jsonl
    trials = load_jsonl(root / "artifacts/steering/trials.jsonl"); non_natural = [row for row in trials if row["direction"] in {"shared_loto", "self_matched_dose", "random_matched"}]
    expected_zero = len(TEMPLATES) * validation_cases * len(executed_nodes); zeros = [row for row in trials if row["direction"] == "zero_baseline"]
    gates = {"axis_grid_complete": all((root / f"artifacts/axes/{node_name(*node)}.npz").is_file() for node in DEFAULT_NODES), "zero_baseline_complete": len(zeros) == expected_zero and all(row["alpha_zero_parity"] for row in zeros), "equal_dose_norm": bool(non_natural) and all(float(row["equal_dose_relative_error"]) <= 1e-6 for row in non_natural), "random_count": all(sum(1 for row in non_natural if row["template"] == template and row["position"] == position and int(row["layer"]) == layer and row["direction"] == "random_matched" and float(row["alpha"]) == 2.0) == validation_cases * random_count for template in TEMPLATES for position, layer in executed_nodes), "outputs": all((root / path).is_file() and (root / path).stat().st_size > 0 for path in ("tables/geometry_summary.csv", "tables/loto_prediction_metrics.csv", "tables/steering_effects.csv", "tables/shared_self_random_comparisons.csv", "tables/t3_length_sensitivity.csv"))}
    if not all(gates.values()): raise ValueError(f"Acceptance failed: {gates}")
    return gates


def run(root: Path, *, nodes: Sequence[tuple[str, int]], validation_cases: int, random_count: int, auto_expand: bool, smoke: bool, resume: bool) -> dict[str, Any]:
    cpu = prepare_cpu(root, nodes, random_count, validation_cases=validation_cases, resume=resume)
    if not cpu["gpu_gate"]["passed"]:
        result = {"status": "stopped_at_cpu_gate", "cpu": cpu, "completed_at_unix": time.time()}; atomic_json(root / "completion.json", result); return result
    pilot_nodes = [PILOT_NODE] if not smoke else [PILOT_NODE, ("P1_PANL", 15)]
    steering_pilot = run_steering(root, pilot_nodes, validation_cases=validation_cases, random_count=random_count, resume=resume); pilot_analysis = analyze(root, pilot_nodes, validation_cases=validation_cases, random_count=random_count)
    expanded = bool(auto_expand and not smoke and pilot_analysis["pilot_gate"]["passed"])
    executed_nodes = list(pilot_nodes)
    if expanded:
        support = [node for node in nodes if node != PILOT_NODE]; run_steering(root, support, validation_cases=validation_cases, random_count=random_count, resume=resume); executed_nodes = list(nodes)
    final_analysis = analyze(root, executed_nodes, validation_cases=validation_cases, random_count=random_count); figures = make_plots(root); verify_sources(root); gates = verify(root, executed_nodes, validation_cases, random_count); _readme(root, cpu, final_analysis, expanded)
    result = {"status": "complete", "smoke_only": smoke, "cpu": cpu, "pilot_steering": steering_pilot, "analysis": final_analysis, "expanded_to_supporting_nodes": expanded, "executed_nodes": [node_name(*node) for node in executed_nodes], "figures": figures, "gates": gates, "historical_sources_unchanged": True, "completed_at_unix": time.time()}; atomic_json(root / "completion.json", result); return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--nodes", nargs="+", default=[f"{position}:{layer}" for position,layer in DEFAULT_NODES]); parser.add_argument("--validation-cases", type=int, default=100); parser.add_argument("--alphas", nargs="+", type=float, default=[-2,0,2]); parser.add_argument("--random-directions", type=int, default=10); parser.add_argument("--pilot-node", default="P1_LAT:14"); parser.add_argument("--auto-expand", action="store_true"); parser.add_argument("--num-gpus", type=int, choices=(1,), default=1); parser.add_argument("--resume", action="store_true"); parser.add_argument("--smoke", action="store_true"); parser.add_argument("--output-root"); args = parser.parse_args(argv)
    nodes = parse_nodes(args.nodes)
    if args.pilot_node != "P1_LAT:14" or set(map(float,args.alphas)) != {-2.,0.,2.}: raise ValueError("Pilot node and alpha grid are preregistered")
    validation_cases = 4 if args.smoke else args.validation_cases; random_count = 1 if args.smoke else args.random_directions
    if not args.smoke and (validation_cases != 100 or random_count != 10): raise ValueError("Formal run requires 100 cases and 10 random directions")
    root = Path(args.output_root).resolve() if args.output_root else (SMOKE_ROOT if args.smoke else DEFAULT_ROOT)
    if (PACKAGE_ROOT / "output").resolve() not in root.parents: raise ValueError("Output root must be under dp_SA/prompt_check/output")
    print(json.dumps(run(root, nodes=nodes, validation_cases=validation_cases, random_count=random_count, auto_expand=args.auto_expand, smoke=args.smoke, resume=args.resume), ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
