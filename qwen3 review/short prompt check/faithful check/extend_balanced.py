from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[2]
for candidate in (REPOSITORY_ROOT, HERE):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from dp_SA.io_utils import append_jsonl, atomic_json, atomic_jsonl, load_jsonl
from config import HARD_QUOTA, MODEL_PATH, OUTPUT_ROOT, SOURCE_DATASET, TEXT_POOL
from run_pipeline import Pipeline, preflight


EXTENDED_ROOT = OUTPUT_ROOT.parent / "faithful_check_extended"


def _copy_if_present(source: Path, destination: Path) -> None:
    if source.is_file() and not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def prepare_extended_root(source_root: Path, destination_root: Path) -> None:
    destination_root.mkdir(parents=True, exist_ok=True)
    for filename in (
        "text_single_modal.jsonl", "image_single_modal.jsonl", "excluded.jsonl",
        "manifest.jsonl", "trials.jsonl",
    ):
        _copy_if_present(source_root / filename, destination_root / filename)
    source_images = source_root / "counterfactual_images"
    destination_images = destination_root / "counterfactual_images"
    if source_images.is_dir() and not destination_images.exists():
        shutil.copytree(source_images, destination_images)
    atomic_json(destination_root / "extension_parent.json", {
        "source_root": str(source_root.resolve()),
        "source_run_summary": str((source_root / "run_summary.json").resolve()),
        "policy": "retain all old cases; add remaining hard candidates; balanced subset prefers new CMA-text cases",
    })


def build_balanced_subset(
    extended_root: Path, balanced_root: Path, original_root: Path,
) -> dict[str, Any]:
    manifests = {row["case_id"]: row for row in load_jsonl(extended_root / "manifest.jsonl")}
    trials = {row["case_id"]: row for row in load_jsonl(extended_root / "trials.jsonl") if row.get("status") == "completed"}
    old_case_ids = {row["case_id"] for row in load_jsonl(original_root / "trials.jsonl")}
    old_trials = [trials[case_id] for case_id in old_case_ids if case_id in trials]
    new_trials = [row for case_id, row in trials.items() if case_id not in old_case_ids and row["difficulty"] == "hard"]
    new_text = [row for row in new_trials if float(row["cma_logit"]["cma_signed"]) < 0]
    new_image = [row for row in new_trials if float(row["cma_logit"]["cma_signed"]) > 0]
    new_other = [row for row in new_trials if float(row["cma_logit"]["cma_signed"]) == 0]

    def side(row: dict[str, Any]) -> str:
        return "image" if float(row["cma_logit"]["cma_signed"]) > 0 else "text"

    old_image = sum(side(row) == "image" for row in old_trials)
    old_text = sum(side(row) == "text" for row in old_trials)
    desired_new_image = max(0, min(len(new_image), old_text + len(new_text) - old_image))
    new_image.sort(key=lambda row: (abs(float(row["cma_logit"]["cma_signed"])), row["case_id"]))
    selected = old_trials + new_text + new_image[:desired_new_image]
    selected.sort(key=lambda row: (row["difficulty"], row["case_id"]))

    balanced_root.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(balanced_root / "manifest.jsonl", [manifests[row["case_id"]] for row in selected])
    atomic_jsonl(balanced_root / "trials.jsonl", selected)
    atomic_json(balanced_root / "selection.json", {
        "policy": "All original cases retained; all new hard CMA-text cases retained; add closest new CMA-image cases only if needed for balance",
        "original_count": len(old_trials), "new_hard_completed": len(new_trials),
        "new_text_selected": len(new_text), "new_image_available": len(new_image),
        "new_image_selected": desired_new_image, "new_other_excluded": len(new_other),
        "selected_total": len(selected),
        "selected_sides": Counter(side(row) for row in selected),
        "source_extended_root": str(extended_root.resolve()),
    })
    return json.loads((balanced_root / "selection.json").read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extend faithful check with hard cases and build CMA-balanced analysis")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--source-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--extended-root", type=Path, default=EXTENDED_ROOT)
    parser.add_argument("--balanced-root", type=Path, default=EXTENDED_ROOT / "balanced_subset")
    parser.add_argument("--skip-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_extended_root(args.source_root.resolve(), args.extended_root.resolve())
    if not args.skip_run:
        quotas = {"easy": 70, "hard": HARD_QUOTA + 51}
        preflight(
            model_path=MODEL_PATH, dataset_path=SOURCE_DATASET, pool_path=TEXT_POOL,
            output_root=args.extended_root, quotas=quotas,
        )
        pipeline = Pipeline(
            model_path=MODEL_PATH, dataset_path=SOURCE_DATASET, pool_path=TEXT_POOL,
            output_root=args.extended_root, quotas=quotas, resume=args.resume,
        )
        print(json.dumps(pipeline.run(), ensure_ascii=False, indent=2))
    selection = build_balanced_subset(args.extended_root.resolve(), args.balanced_root.resolve(), args.source_root.resolve())
    atomic_json(args.extended_root / "balanced_selection.json", selection)
    print(json.dumps(selection, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
