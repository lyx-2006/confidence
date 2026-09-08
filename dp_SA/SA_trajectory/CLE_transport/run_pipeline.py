from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from dp_SA.io_utils import atomic_json, sha256_file

from .analyze import analyze
from .config import EXPERIMENTS, WINDOWS, default_output, parse_windows
from .prepare import prepare
from .run import run


def history_hashes() -> dict[str, str]:
    repository = Path(__file__).resolve().parents[3]
    roots = [
        repository / "dp_SA" / "attention_block",
        repository / "dp_SA" / "SA_trajectory" / "LAT2PANL",
        repository / "dp_SA" / "positions.py",
        repository / "dp_SA" / "soft_score.py",
        repository / "dp_SA" / "answer_matched_lat_steering" / "output"
        / "lat_panl_comparison" / "artifacts" / "manifests" / "test_manifest.jsonl",
        repository / "dp_SA" / "answer_matched_lat_steering" / "output"
        / "lat_panl_comparison" / "artifacts" / "diagnostics" / "clean_capture.jsonl",
    ]
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        else:
            files.extend(path for path in root.rglob("*") if path.is_file())
    selected = sorted({path.resolve() for path in files
                       if "__pycache__" not in path.parts and path.suffix != ".pyc"
                       and "CLE_transport" not in path.parts})
    return {str(path.relative_to(repository)): sha256_file(path) for path in selected}


def run_cpu_tests(output: Path) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q",
               "dp_SA/SA_trajectory/CLE_transport/tests",
               "dp_SA/attention_block/tests/test_masking.py"]
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[3],
                            capture_output=True, text=True)
    log = result.stdout + result.stderr
    (output / "progress").mkdir(parents=True, exist_ok=True)
    (output / "progress" / "cpu_tests.log").write_text(log, encoding="utf-8")
    record = {"status": "passed" if result.returncode == 0 else "failed",
              "returncode": result.returncode, "command": command,
              "summary": log.strip().splitlines()[-1] if log.strip() else ""}
    atomic_json(output / "progress" / "cpu_tests.json", record)
    if result.returncode:
        raise RuntimeError(f"CPU tests failed; see {output / 'progress' / 'cpu_tests.log'}")
    return record


def _next_smoke_root(experiment: str) -> tuple[int, Path]:
    parent = default_output(experiment) / "smoke"
    numbers = [int(path.name.removeprefix("round_")) for path in parent.glob("round_*")
               if path.name.removeprefix("round_").isdigit()]
    number = max(numbers, default=0) + 1
    if number > 5:
        raise RuntimeError(f"Maximum five GPU smoke rounds reached for {experiment}")
    return number, parent / f"round_{number}"


def _summary_markdown(root: Path, result: dict[str, Any]) -> None:
    run_result = result["run"]
    analysis = result["analysis"]
    lines = [
        f"# {result['experiment']} 运行摘要", "",
        f"- 状态：{result['status']}",
        f"- 模式：{'smoke' if result['smoke'] else 'formal'}",
        f"- 新增 GPU forwards：{run_result['new_gpu_forwards']}",
        f"- clean parity：{run_result['clean_gate']['status']}",
        f"- 完成窗口：{analysis['complete_windows']}",
        f"- attention audit rows：{analysis['attention_audit_rows']}",
        f"- 耗时：{run_result['elapsed_seconds']:.2f} 秒", "",
    ]
    (root / "progress" / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def pipeline(*, experiment: str, windows: tuple[tuple[int, int], ...] = WINDOWS,
             num_gpus: int = 1, resume: bool = False, smoke: bool = False,
             output_root: Path | None = None) -> dict[str, Any]:
    if smoke and output_root is None:
        round_number, root = _next_smoke_root(experiment)
    else:
        round_number = None
        root = Path(output_root or default_output(experiment)).resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    before = history_hashes()
    tests = run_cpu_tests(root)
    prepared = prepare(experiment=experiment, output_root=root, smoke=smoke,
                       resume=resume)
    if not smoke:
        latest = default_output(experiment) / "smoke" / "latest_status.json"
        if not latest.is_file() or json.loads(latest.read_text()).get("status") != "passed":
            raise RuntimeError(
                f"Formal run requires a passed smoke for {experiment}; run this command with --smoke first"
            )
        smoke_record = json.loads(latest.read_text())
        smoke_config = json.loads((Path(smoke_record["output_root"]) / "run_config.json").read_text())
        formal_config = json.loads((root / "run_config.json").read_text())
        if smoke_config.get("implementation_sha256") != formal_config.get("implementation_sha256"):
            raise RuntimeError("The passed smoke was produced by a different implementation")
    executed = run(experiment=experiment, output_root=root, windows=windows,
                   num_gpus=num_gpus, resume=resume)
    analyzed = analyze(experiment=experiment, output_root=root, smoke=smoke)
    resume_zero = None
    if smoke:
        repeated = run(experiment=experiment, output_root=root, windows=windows,
                       num_gpus=num_gpus, resume=True)
        resume_zero = repeated["new_gpu_forwards"] == 0
        if not resume_zero:
            raise RuntimeError("Smoke resume repeated completed GPU forwards")
    after = history_hashes()
    if before != after:
        changed = sorted(set(before) ^ set(after) | {key for key in set(before) & set(after)
                                                     if before[key] != after[key]})
        raise RuntimeError(f"Historical files changed during pipeline: {changed[:10]}")
    status = "passed" if smoke else analyzed["status"]
    result = {"status": status, "experiment": experiment, "smoke": smoke,
              "smoke_round": round_number, "output_root": str(root), "cpu_tests": tests,
              "prepare": prepared, "run": executed, "analysis": analyzed,
              "resume_zero_forward": resume_zero, "historical_unchanged": True,
              "elapsed_seconds": time.time() - started}
    _summary_markdown(root, result)
    if smoke:
        latest = default_output(experiment) / "smoke" / "latest_status.json"
        atomic_json(latest, result)
        atomic_json(root / "completion.json", result)
    elif analyzed["all_windows_complete"]:
        atomic_json(root / "completion.json", result)
    else:
        atomic_json(root / "progress" / "partial_completion.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS), required=True)
    parser.add_argument("--windows", type=parse_windows, default=WINDOWS,
                        help="Comma-separated frozen ranges, e.g. 8-12,18-22")
    parser.add_argument("--num-gpus", type=int, choices=(1, 2), default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    result = pipeline(experiment=args.experiment, windows=args.windows,
                      num_gpus=args.num_gpus, resume=args.resume, smoke=args.smoke,
                      output_root=args.output_root)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
