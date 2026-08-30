#!/usr/bin/env python3
"""
Refine valid/invalid split:
- VALID: only keep items where conflict_easy.answer == conflict_hard.answer
- Move CE != CH items from valid to invalid (flag: conflict_easy_hard_answer_mismatch)
- Re-number all items and re-copy images, then replace old dirs.
"""

import json
import os
import shutil
from datetime import datetime, timezone

BASE_DIR = "generate dataset/datasets"
VALID_JSON = f"{BASE_DIR}/valid_datasets/generated_shape_color_dataset.json"
INVALID_JSON = f"{BASE_DIR}/invalid_datasets/generated_shape_color_dataset.json"
VALID_IMG = f"{BASE_DIR}/valid_datasets/images"
INVALID_IMG = f"{BASE_DIR}/invalid_datasets/images"

TMP_VALID_IMG = f"{BASE_DIR}/valid_datasets/images_tmp"
TMP_INVALID_IMG = f"{BASE_DIR}/invalid_datasets/images_tmp"

GROUP_MAP = {
    "consistent_easy": "consist_easy",
    "consistent_hard": "consist_hard",
    "conflict_easy": "conflict_easy",
    "conflict_hard": "conflict_hard",
}

SUFFIXES = [".layout.json", ".occluder_mask.png", ".target_mask.png"]


def copy_images(old_id, new_idx, src_dir, dst_dir):
    for pattern in GROUP_MAP.values():
        old_base = f"{old_id}_{pattern}"
        new_base = f"{new_idx:03d}_{pattern}"
        for ext in [".png"] + SUFFIXES:
            src = os.path.join(src_dir, f"{old_base}{ext}")
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dst_dir, f"{new_base}{ext}"))


def rebuild_json(items, src_dirs, output_path, img_dir, img_dir_rel, description, extra_meta=None):
    """Rebuild items: renumber, copy images, write JSON."""
    os.makedirs(img_dir, exist_ok=True)

    new_items = []
    for new_idx, item in enumerate(items, start=1):
        old_id = item["id"]
        src_dir = src_dirs[old_id] if isinstance(src_dirs, dict) else src_dirs
        copy_images(old_id, new_idx, src_dir, img_dir)

        new_item = json.loads(json.dumps(item))
        new_item["id"] = f"{new_idx:03d}"
        for gname, pattern in GROUP_MAP.items():
            if gname in new_item.get("groups", {}):
                new_item["groups"][gname]["image"] = f"{img_dir_rel}/{new_idx:03d}_{pattern}.png"
        new_items.append(new_item)

    output = {
        "dataset_type": extra_meta.get("type", "unknown") if extra_meta else "unknown",
        "description": description,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "item_count": len(new_items),
        "group_count": len(new_items) * 4,
        "items": new_items,
    }
    if extra_meta:
        for k, v in extra_meta.items():
            if k != "type":
                output[k] = v

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return new_items


def main():
    with open(VALID_JSON) as f:
        valid_old = json.load(f)
    with open(INVALID_JSON) as f:
        invalid_old = json.load(f)

    # Split valid
    valid_keep, valid_move = [], []
    for item in valid_old["items"]:
        ce = item["groups"]["conflict_easy"]["answer"]
        ch = item["groups"]["conflict_hard"]["answer"]
        if ce == ch:
            valid_keep.append(item)
        else:
            item["_flags"] = item.get("_flags", []) + ["conflict_easy_hard_answer_mismatch"]
            item.setdefault("_flag_details", {})["ce_ch_mismatch"] = {
                "conflict_easy_answer": ce,
                "conflict_hard_answer": ch,
            }
            valid_move.append(item)

    print(f"Valid: {len(valid_old['items'])} -> {len(valid_keep)} (CE==CH)")
    print(f"  Moved to invalid: {len(valid_move)}")

    # Moved item IDs (old valid numbering, like "001", "002", ...)
    moved_ids = {it["id"] for it in valid_move}

    # Combine invalid
    new_invalid_items = invalid_old["items"] + valid_move
    print(f"Invalid: {len(invalid_old['items'])} -> {len(new_invalid_items)}")

    # Src dir map for invalid items: moved ones come from VALID_IMG, rest from INVALID_IMG
    invalid_src_dirs = {}
    for it in new_invalid_items:
        invalid_src_dirs[it["id"]] = VALID_IMG if it["id"] in moved_ids else INVALID_IMG

    # ---- Rebuild valid ----
    print("\nRebuilding valid...")
    rebuild_json(
        items=valid_keep,
        src_dirs=VALID_IMG,
        output_path=f"{BASE_DIR}/valid_datasets/_tmp.json",
        img_dir=TMP_VALID_IMG,
        img_dir_rel="images",
        description="Conflict samples where conflict_easy.answer == conflict_hard.answer (model consistent across difficulty) AND entropy_diff>=0.25 AND answer differs from consistent answer.",
        extra_meta={"type": "valid", "filter": "CE==CH"},
    )

    # ---- Rebuild invalid ----
    print("Rebuilding invalid...")
    rebuild_json(
        items=new_invalid_items,
        src_dirs=invalid_src_dirs,
        output_path=f"{BASE_DIR}/invalid_datasets/_tmp.json",
        img_dir=TMP_INVALID_IMG,
        img_dir_rel="images",
        description="Flagged conflict samples: entropy_diff<0.25, OR answer==consistent, OR conflict_easy.answer!=conflict_hard.answer.",
        extra_meta={"type": "invalid", "flag_criteria": [
            "entropy_diff_below_0.25",
            "conflict_answer_equals_consistent",
            "conflict_easy_hard_answer_mismatch",
        ]},
    )

    # ---- Replace old with new ----
    shutil.rmtree(VALID_IMG)
    os.rename(TMP_VALID_IMG, VALID_IMG)
    os.rename(f"{BASE_DIR}/valid_datasets/_tmp.json", VALID_JSON)

    shutil.rmtree(INVALID_IMG)
    os.rename(TMP_INVALID_IMG, INVALID_IMG)
    os.rename(f"{BASE_DIR}/invalid_datasets/_tmp.json", INVALID_JSON)

    print(f"\nDone! Valid: {len(valid_keep)}, Invalid: {len(new_invalid_items)}")


if __name__ == "__main__":
    main()
