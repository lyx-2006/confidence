#!/usr/bin/env python3
"""Process generated_shape_color_dataset.summary.json:

1. For conflict samples (conflict_easy, conflict_hard): update ground_truth_answer
   to use the model's actual answer (not the ideal/expected conflict_answer).
   Also check that conflict answers differ from consistent answer.

2. Flag conflict samples where:
   - |easy_entropy - hard_entropy| < 0.25, OR
   - conflict answer == consistent answer
   Write flagged items to a new JSON.
"""

import json
import os
import shutil
from datetime import datetime, timezone

INPUT_PATH = "generate dataset/datasets/generated_shape_color_dataset.summary.json"
BACKUP_PATH = "generate dataset/datasets/generated_shape_color_dataset.summary.json.bak"
FLAGGED_PATH = "generate dataset/datasets/generated_shape_color_dataset.flagged.json"


def main():
    # Load
    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    # Backup original
    shutil.copy(INPUT_PATH, BACKUP_PATH)
    print(f"Backup saved to: {BACKUP_PATH}")

    flagged_items = []
    gt_update_count = 0

    for item in data["items"]:
        item_id = item["id"]
        consistent_answer = item["answer"]
        conflict_answer = item.get("conflict_answer", "")

        # Each conflict group to process
        flagged = {
            "id": item_id,
            "question": item["question"],
            "consistent_answer": consistent_answer,
            "conflict_answer": conflict_answer,
            "flags": [],
            "details": {},
        }

        ce = item["groups"].get("conflict_easy", {})
        ch = item["groups"].get("conflict_hard", {})

        # ---- Step 1: Update ground_truth_answer = answer for conflict groups ----
        for gname, gdata in [("conflict_easy", ce), ("conflict_hard", ch)]:
            if not gdata:
                continue
            model_answer = gdata["answer"]
            old_gt = gdata["ground_truth_answer"]

            if old_gt != model_answer:
                gdata["ground_truth_answer"] = model_answer
                gt_update_count += 1
                print(f"  Updated {gname} ground_truth for id={item_id}: '{old_gt}' -> '{model_answer}'")

            # Also update individual runs
            for run in gdata.get("runs", []):
                if run["ground_truth_answer"] != model_answer:
                    run["ground_truth_answer"] = model_answer

        # ---- Step 2: Check conditions for flagging ----
        flags = []

        # Condition A: answer == consistent answer
        if ce and ce["answer"] == consistent_answer:
            flags.append("conflict_easy_answer_equals_consistent")
        if ch and ch["answer"] == consistent_answer:
            flags.append("conflict_hard_answer_equals_consistent")

        # Condition B: |easy entropy - hard entropy| < 0.25
        if ce and ch:
            ent_diff = abs(ce["normalized_entropy"] - ch["normalized_entropy"])
            if ent_diff < 0.25:
                flags.append("entropy_diff_below_0.25")

            flagged["details"]["conflict_easy"] = {
                "answer": ce["answer"],
                "ground_truth_answer": ce["ground_truth_answer"],
                "normalized_entropy": ce["normalized_entropy"],
                "entropy": ce["entropy"],
            }
            flagged["details"]["conflict_hard"] = {
                "answer": ch["answer"],
                "ground_truth_answer": ch["ground_truth_answer"],
                "normalized_entropy": ch["normalized_entropy"],
                "entropy": ch["entropy"],
            }
            flagged["details"]["entropy_diff"] = round(ent_diff, 6)
            flagged["details"]["entropy_diff_below_threshold"] = ent_diff < 0.25

        if flags:
            flagged["flags"] = flags
            flagged_items.append(flagged)

    # ---- Write modified summary JSON ----
    data["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    with open(INPUT_PATH, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nModified summary written: {INPUT_PATH}")
    print(f"  ground_truth_answer updates: {gt_update_count}")

    # ---- Write flagged JSON ----
    flagged_output = {
        "source_dataset": INPUT_PATH,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Items from the summary dataset whose conflict samples are flagged. "
            "Flags: 'conflict_easy_answer_equals_consistent' — conflict_easy answer matches consistent; "
            "'conflict_hard_answer_equals_consistent' — conflict_hard answer matches consistent; "
            "'entropy_diff_below_0.25' — |normalized_entropy(easy) - normalized_entropy(hard)| < 0.25."
        ),
        "total_flagged": len(flagged_items),
        "flag_summary": {},
        "items": flagged_items,
    }

    # Summarize flag counts
    flag_counts = {}
    for fi in flagged_items:
        for f in fi["flags"]:
            flag_counts[f] = flag_counts.get(f, 0) + 1
    flagged_output["flag_summary"] = flag_counts

    # Also list items with entropy_diff_below_0.25 separately for convenience
    low_entropy_diff_ids = [fi["id"] for fi in flagged_items if "entropy_diff_below_0.25" in fi["flags"]]
    answer_same_ids = [fi["id"] for fi in flagged_items if any("answer_equals_consistent" in f for f in fi["flags"])]
    flagged_output["entropy_diff_below_0.25_ids"] = low_entropy_diff_ids
    flagged_output["answer_equals_consistent_ids"] = answer_same_ids

    with open(FLAGGED_PATH, "w") as f:
        json.dump(flagged_output, f, indent=2, ensure_ascii=False)
    print(f"\nFlagged JSON written: {FLAGGED_PATH}")
    print(f"  Total flagged: {len(flagged_items)}")
    print(f"  Flag summary: {json.dumps(flag_counts)}")
    print(f"  entropy_diff_below_0.25: {len(low_entropy_diff_ids)} items")
    print(f"  answer_equals_consistent: {len(answer_same_ids)} items")

    # Print items where answer == consistent (should be investigated)
    print("\n--- Items with conflict answer == consistent answer ---")
    for fi in flagged_items:
        if any("answer_equals_consistent" in f for f in fi["flags"]):
            print(f"  id={fi['id']}: consistent='{fi['consistent_answer']}', "
                  f"ce_ans='{fi['details'].get('conflict_easy', {}).get('answer', 'N/A')}', "
                  f"ch_ans='{fi['details'].get('conflict_hard', {}).get('answer', 'N/A')}'")


if __name__ == "__main__":
    main()
