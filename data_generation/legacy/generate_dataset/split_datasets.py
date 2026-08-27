#!/usr/bin/env python3
"""
Split dataset into valid and invalid based on flagged.json.

- valid_datasets/: items NOT flagged → new summary JSON + renumbered images
- invalid_datasets/: flagged items → new summary JSON + renumbered images

Images are renumbered starting from 1, keeping the group suffixes.
All associated files (.png, .layout.json, .occluder_mask.png, .target_mask.png)
are copied with the new IDs.
"""

import json
import os
import shutil
from datetime import datetime, timezone

BASE_DIR = "generate dataset/datasets"
SUMMARY_PATH = f"{BASE_DIR}/generated_shape_color_dataset.summary.json"
FLAGGED_PATH = f"{BASE_DIR}/generated_shape_color_dataset.flagged.json"
IMAGE_DIR = f"{BASE_DIR}/generated_shape_color_images"

VALID_DIR = f"{BASE_DIR}/valid_datasets"
INVALID_DIR = f"{BASE_DIR}/invalid_datasets"
VALID_IMAGE_DIR = f"{VALID_DIR}/images"
INVALID_IMAGE_DIR = f"{INVALID_DIR}/images"

# Mapping from JSON group key → image filename group pattern
GROUP_TO_FILE_PATTERN = {
    "consistent_easy": "consist_easy",
    "consistent_hard": "consist_hard",
    "conflict_easy": "conflict_easy",
    "conflict_hard": "conflict_hard",
}

# Associated file suffixes (beyond the main .png)
ASSOCIATED_SUFFIXES = [
    ".layout.json",
    ".occluder_mask.png",
    ".target_mask.png",
]


def get_old_base(group_file_pattern, old_id):
    """e.g. 121_consist_easy"""
    return f"{old_id}_{group_file_pattern}"


def get_new_base(group_file_pattern, new_id):
    """e.g. 1_consist_easy"""
    if isinstance(new_id, int):
        return f"{new_id:03d}_{group_file_pattern}"
    else:
        return f"{new_id}_{group_file_pattern}"


def copy_images(old_id, new_id, target_dir, dry_run=False):
    """Copy all image-related files for a given old_id, renaming to new_id."""
    copied = []
    for pattern in GROUP_TO_FILE_PATTERN.values():
        old_base = get_old_base(pattern, old_id)
        new_base = get_new_base(pattern, new_id)

        # Main png
        src = os.path.join(IMAGE_DIR, f"{old_base}.png")
        dst = os.path.join(target_dir, f"{new_base}.png")
        if os.path.exists(src):
            if not dry_run:
                shutil.copy2(src, dst)
            copied.append(dst)

        # Associated files
        for suffix in ASSOCIATED_SUFFIXES:
            src = os.path.join(IMAGE_DIR, f"{old_base}{suffix}")
            dst = os.path.join(target_dir, f"{new_base}{suffix}")
            if os.path.exists(src):
                if not dry_run:
                    shutil.copy2(src, dst)
                copied.append(dst)

    return copied


def update_item(item, new_id, image_dir_rel):
    """Update an item's id and image paths for the new dataset."""
    item["id"] = f"{new_id:03d}"
    for gname, pattern in GROUP_TO_FILE_PATTERN.items():
        if gname in item["groups"]:
            old_image = item["groups"][gname]["image"]
            # Extract just the filename and replace
            old_filename = os.path.basename(old_image)
            new_filename = f"{new_id:03d}_{pattern}.png"
            item["groups"][gname]["image"] = f"{image_dir_rel}/{new_filename}"
    return item


def main():
    # Load data
    with open(SUMMARY_PATH) as f:
        summary = json.load(f)
    with open(FLAGGED_PATH) as f:
        flagged = json.load(f)
    if str(summary.get("schema_version", "")).startswith("shape_color_dataset.") or str(flagged.get("schema_version", "")).startswith("shape_color_dataset."):
        raise ValueError("Legacy split tool cannot process shape_color_dataset.v2; use V2 consumers")

    # Get flagged (invalid) IDs
    flagged_ids = set(item["id"] for item in flagged["items"])
    print(f"Flagged (invalid) IDs: {len(flagged_ids)}")
    print(f"  {sorted(flagged_ids, key=int)}")

    # Split items
    valid_items = []
    invalid_items = []
    for item in summary["items"]:
        if item["id"] in flagged_ids:
            invalid_items.append(item)
        else:
            valid_items.append(item)

    print(f"\nValid items:   {len(valid_items)}")
    print(f"Invalid items: {len(invalid_items)}")

    # Create output directories
    for d in [VALID_IMAGE_DIR, INVALID_IMAGE_DIR]:
        os.makedirs(d, exist_ok=True)

    # ---- Process valid dataset ----
    print("\n--- Processing valid dataset ---")
    new_valid_items = []
    valid_image_dir_rel = "images"  # relative path in JSON

    for new_idx, item in enumerate(valid_items, start=1):
        old_id = item["id"]
        new_id = f"{new_idx:03d}"
        copied = copy_images(old_id, new_idx, VALID_IMAGE_DIR)
        new_item = update_item(json.loads(json.dumps(item)), new_idx, valid_image_dir_rel)
        new_valid_items.append(new_item)
        if new_idx <= 3:
            print(f"  {old_id} -> {new_id}: {len(copied)} files")

    # Build valid summary JSON
    valid_summary = {
        "source_dataset": SUMMARY_PATH,
        "dataset_type": "valid",
        "description": "Conflict samples without entropy_diff<0.25 or answer-equals-consistent issues. Rationally consistent data.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "item_count": len(new_valid_items),
        "group_count": len(new_valid_items) * 4,
        "items": new_valid_items,
    }

    valid_json_path = f"{VALID_DIR}/generated_shape_color_dataset.json"
    with open(valid_json_path, "w") as f:
        json.dump(valid_summary, f, indent=2, ensure_ascii=False)
    print(f"  Valid JSON written: {valid_json_path}")

    # ---- Process invalid dataset ----
    print("\n--- Processing invalid dataset ---")
    new_invalid_items = []
    invalid_image_dir_rel = "images"

    for new_idx, item in enumerate(invalid_items, start=1):
        old_id = item["id"]
        new_id = f"{new_idx:03d}"
        copied = copy_images(old_id, new_idx, INVALID_IMAGE_DIR)
        new_item = update_item(json.loads(json.dumps(item)), new_idx, invalid_image_dir_rel)

        # Attach flag info from flagged.json
        flag_info = next((fi for fi in flagged["items"] if fi["id"] == old_id), None)
        if flag_info:
            new_item["_flags"] = flag_info["flags"]
            new_item["_flag_details"] = flag_info.get("details", {})

        new_invalid_items.append(new_item)
        if new_idx <= 3:
            print(f"  {old_id} -> {new_id}: {len(copied)} files, flags={flag_info['flags'] if flag_info else 'N/A'}")

    # Build invalid summary JSON
    invalid_summary = {
        "source_dataset": SUMMARY_PATH,
        "flagged_source": FLAGGED_PATH,
        "dataset_type": "invalid",
        "description": (
            "Conflict samples flagged as problematic: "
            "entropy_diff < 0.25 between conflict_easy and conflict_hard, "
            "OR conflict answer equals consistent answer."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "item_count": len(new_invalid_items),
        "group_count": len(new_invalid_items) * 4,
        "flag_criteria": flagged["description"],
        "items": new_invalid_items,
    }

    invalid_json_path = f"{INVALID_DIR}/generated_shape_color_dataset.json"
    with open(invalid_json_path, "w") as f:
        json.dump(invalid_summary, f, indent=2, ensure_ascii=False)
    print(f"  Invalid JSON written: {invalid_json_path}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"Done! Summary:")
    print(f"  Valid dataset:   {len(new_valid_items):3d} items → {VALID_DIR}/")
    print(f"  Invalid dataset: {len(new_invalid_items):3d} items → {INVALID_DIR}/")

    # Count total image files copied
    valid_files = len(os.listdir(VALID_IMAGE_DIR))
    invalid_files = len(os.listdir(INVALID_IMAGE_DIR))
    print(f"  Valid images:   {valid_files} files")
    print(f"  Invalid images: {invalid_files} files")


if __name__ == "__main__":
    main()
