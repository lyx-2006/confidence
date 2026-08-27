#!/usr/bin/env python3
"""
Merge valid datasets into datasets/datasets.json

1. Load old dataset (dataset_with_images.json), filter to 12 basic colors
2. Load new valid dataset (generated_shape_color_dataset.json)
3. Copy images from valid_datasets/images/ to datasets/images/ with renamed IDs
4. Allocate text priors from new_color_prior_pool.json (round-robin, max 3 per bin per item)
5. Write merged datasets.json
"""

import json
import os
import shutil
from collections import defaultdict

# ─── Configuration ───────────────────────────────────────────────────────────

BASIC_COLORS = [
    "red", "orange", "yellow", "green", "blue", "cyan",
    "purple", "pink", "brown", "white", "black", "gray"
]

BASIC_COLORS_SET = set(BASIC_COLORS)

BIN_ID_TO_LABEL = {
    0: "0.0-0.2",
    1: "0.2-0.4",
    2: "0.4-0.6",
    3: "0.6-0.8",
    4: "0.8-1.0",
}

MAX_PRIORS_PER_BIN = 3

BASE_DIR = "/root/autodl-tmp"
OLD_DATASET_PATH = os.path.join(BASE_DIR, "datasets/dataset_with_images.json")
NEW_DATASET_PATH = os.path.join(BASE_DIR, "generation_v2_outputs/formal/image/shape_color_dataset.json")
NEW_POOL_PATH = os.path.join(BASE_DIR, "generation_v2_outputs/formal/text/text_entropy_pool.json")
SRC_IMAGES_DIR = os.path.join(BASE_DIR, "generation_v2_outputs/formal/image/images")
DST_IMAGES_DIR = os.path.join(BASE_DIR, "datasets/images")
OUTPUT_PATH = os.path.join(BASE_DIR, "datasets/datasets.json")

NEW_ITEM_START_ID = 121  # Continue from old max (120)


# ─── Step 1: Load Source Data ────────────────────────────────────────────────

def load_json(path):
    print(f"Loading: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_old_items():
    """Load old dataset and filter to 12 basic colors only."""
    data = load_json(OLD_DATASET_PATH)
    old_items = data[0]["items"]
    null_image = data[0].get("null_image", "images/null.png")

    filtered = []
    excluded = []
    for item in old_items:
        if item["answer"] in BASIC_COLORS_SET:
            filtered.append(item)
        else:
            excluded.append((item["id"], item["answer"]))

    print(f"Old items: {len(old_items)} total → {len(filtered)} after color filter")
    if excluded:
        print(f"  Excluded: {excluded}")
    return filtered, null_image


def load_new_items():
    """Load valid dataset items."""
    data = load_json(NEW_DATASET_PATH)
    if isinstance(data, dict) and data.get("schema_version") == "shape_color_dataset.v2":
        items = data.get("items", [])
    elif isinstance(data, dict):
        items = data.get("items", [])
    else:
        raise ValueError("New dataset must be an object with items; V2 arrays are unsupported")
    normalized = []
    for item in items:
        if not isinstance(item, dict):
            continue
        value = dict(item)
        question = value.get("question")
        if isinstance(question, dict):
            value["question"] = question.get("text", "")
        if "conflict_answer" not in value and "conflict_ans" in value:
            value["conflict_answer"] = value["conflict_ans"]
        if isinstance(value.get("image_clue"), dict):
            v2_images = {}
            for branch in ("consistent", "conflict"):
                branch_value = value["image_clue"].get(branch, {})
                if not isinstance(branch_value, dict):
                    continue
                for difficulty in ("easy", "hard"):
                    variants = branch_value.get(difficulty)
                    if isinstance(variants, list) and variants:
                        first = variants[0]
                        raw = first.get("image") if isinstance(first, dict) else first
                        if isinstance(raw, str):
                            v2_images[f"{branch}_{difficulty}"] = raw
            if v2_images:
                value["_v2_images"] = v2_images
        normalized.append(value)
    items = normalized
    print(f"New items: {len(items)}")
    return items


def build_prior_lookup():
    """
    Build lookup: prior_clues[color][bin_id] = [text_clue, ...]
    Only for the 12 basic colors, only accepted priors.
    """
    data = load_json(NEW_POOL_PATH)
    prior_clues = defaultdict(lambda: defaultdict(list))

    if isinstance(data, dict) and data.get("schema_version") == "text_entropy_pool.v2":
        color_objects = data.get("colors", [])
    elif isinstance(data, list):
        color_objects = data
    else:
        raise ValueError("Expected text_entropy_pool.v2 or legacy confidence pool")

    for color_obj in color_objects:
        color = color_obj["color"]
        if color not in BASIC_COLORS_SET:
            continue
        levels = color_obj.get("entropy_bins", color_obj.get("prior_levels", []))
        for level in levels:
            bin_id = level.get("entropy_bin_id", level.get("bin_id"))
            for prior in level.get("priors", []):
                if prior.get("accepted", False):
                    clue = prior.get("text_clue", prior.get("clue"))
                    if isinstance(clue, str) and clue.strip():
                        prior_clues[color][int(bin_id)].append(clue)

    # Print summary
    print("\nPrior pool summary (accepted priors only):")
    header = f"{'Color':<8} | {'Bin0':>5} {'Bin1':>5} {'Bin2':>5} {'Bin3':>5} {'Bin4':>5} | {'Total':>5}"
    print(header)
    print("-" * len(header))
    total_all = 0
    for color in BASIC_COLORS:
        counts = [len(prior_clues[color][b]) for b in range(5)]
        total = sum(counts)
        total_all += total
        print(f"{color:<8} | {counts[0]:>5} {counts[1]:>5} {counts[2]:>5} {counts[3]:>5} {counts[4]:>5} | {total:>5}")
    print(f"{'TOTAL':<8} |                               | {total_all:>5}")
    return prior_clues


# ─── Step 2: Copy Images ─────────────────────────────────────────────────────

def copy_images(new_items):
    """Copy PNG files from valid_datasets/images to datasets/images with new IDs."""
    os.makedirs(DST_IMAGES_DIR, exist_ok=True)
    copied = 0
    errors = 0

    for i, item in enumerate(new_items):
        old_id = item["id"]  # e.g., "001"
        new_id = str(NEW_ITEM_START_ID + i)  # e.g., "121"

        v2_images = item.get("_v2_images", {})
        for suffix in ["consist_easy", "consist_hard", "conflict_easy", "conflict_hard"]:
            if v2_images:
                branch = "consistent" if suffix.startswith("consist") else "conflict"
                difficulty = "easy" if suffix.endswith("easy") else "hard"
                raw = v2_images.get(f"{branch}_{difficulty}")
                src_path = raw if isinstance(raw, str) else None
                if src_path and not os.path.isabs(src_path):
                    src_path = os.path.join(os.path.dirname(NEW_DATASET_PATH), src_path)
                src_path = src_path or os.path.join(SRC_IMAGES_DIR, "__missing__.png")
            else:
                src_path = os.path.join(SRC_IMAGES_DIR, f"{old_id}_{suffix}.png")
            src_filename = f"{old_id}_{suffix}.png"
            dst_filename = f"{new_id}_{suffix}.png"
            dst_path = os.path.join(DST_IMAGES_DIR, dst_filename)

            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                copied += 1
            else:
                print(f"  WARNING: Source not found: {src_path}")
                errors += 1

    print(f"\nCopied {copied} images to {DST_IMAGES_DIR}")
    if errors:
        print(f"  WARNING: {errors} source files missing")


# ─── Step 3: Build New Items (without priors yet) ────────────────────────────

def build_new_item_base(item, i):
    """Build a new item dict (without selected_text_priors, which comes later)."""
    new_id = str(NEW_ITEM_START_ID + i)

    return {
        "id": new_id,
        "_new_idx": i,  # temporary, used for prior allocation
        "_answer": item["answer"],  # temporary, used for prior allocation
        "order": "text_image",
        "question": {"text": item["question"]},
        "answer": item["answer"],
        "text_ans": item["answer"],
        "conflict_ans": item["conflict_answer"],
        "image_clue": {
            "consistent": {
                "easy": f"images/{new_id}_consist_easy.png",
                "easy_calibration": None,
                "hard": f"images/{new_id}_consist_hard.png",
                "hard_calibration": None,
                "entropy_check": None,
            },
            "conflict": {
                "easy": f"images/{new_id}_conflict_easy.png",
                "easy_calibration": None,
                "hard": f"images/{new_id}_conflict_hard.png",
                "hard_calibration": None,
                "entropy_check": None,
            },
            "irrelevant": {
                "image": "images/null.png",
                "calibration": {},
                "stability_check": {},
            },
        },
    }


# ─── Step 4: Allocate Text Priors ────────────────────────────────────────────

def allocate_priors(new_items_base, prior_clues):
    """
    Allocate text priors with round-robin distribution.
    - Group items by answer color
    - For each (color, bin_id), distribute clues round-robin, max 3 per item per bin
    """
    # Group items by answer color
    color_groups = defaultdict(list)
    for item in new_items_base:
        color_groups[item["_answer"]].append(item["_new_idx"])

    # Initialize priors per item
    item_priors = defaultdict(list)

    total_allocated = 0
    total_available = 0

    for color in BASIC_COLORS:
        items_of_color = color_groups.get(color, [])
        if not items_of_color:
            continue

        n_items = len(items_of_color)

        for bin_id in range(5):  # bins 0-4
            clues = prior_clues[color].get(bin_id, [])
            if not clues:
                continue

            n_clues = len(clues)
            total_available += n_clues

            # Track how many each item already has from this bin
            item_bin_counts = {idx: 0 for idx in items_of_color}
            item_round_idx = 0
            allocated_from_bin = 0

            for clue in clues:
                # Find next item that hasn't hit the cap yet
                attempts = 0
                assigned = False
                while attempts < n_items:
                    candidate_idx = items_of_color[item_round_idx % n_items]
                    item_round_idx += 1
                    if item_bin_counts[candidate_idx] < MAX_PRIORS_PER_BIN:
                        item_bin_counts[candidate_idx] += 1
                        item_priors[candidate_idx].append({
                            "confidence_bin": BIN_ID_TO_LABEL[bin_id],
                            "clue": clue,
                            "source_bin_id": bin_id,
                        })
                        allocated_from_bin += 1
                        assigned = True
                        break
                    attempts += 1

                if not assigned:
                    # All items at cap for this bin
                    break

            total_allocated += allocated_from_bin
            if allocated_from_bin < n_clues:
                skipped = n_clues - allocated_from_bin
                print(f"  {color} bin{bin_id}: {allocated_from_bin}/{n_clues} allocated "
                      f"(cap reached, {skipped} unused)")

    print(f"\nPrior allocation: {total_allocated} clues allocated "
          f"(out of {total_available} available in pool)")

    # Attach priors to items and clean up temp keys
    for item in new_items_base:
        item["selected_text_priors"] = item_priors[item["_new_idx"]]
        del item["_new_idx"]
        del item["_answer"]

    return new_items_base


# ─── Step 5: Merge and Write ─────────────────────────────────────────────────

def write_output(old_items, new_items, null_image):
    """Combine old + new items and write datasets.json."""
    output = [{
        "category": "colour",
        "items": old_items + new_items,
        "null_image": null_image,
    }]

    print(f"\nWriting output: {len(old_items)} old + {len(new_items)} new = "
          f"{len(old_items) + len(new_items)} total items")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    file_size = os.path.getsize(OUTPUT_PATH)
    print(f"Written: {OUTPUT_PATH} ({file_size:,} bytes)")


# ─── Step 6: Verify ──────────────────────────────────────────────────────────

def verify(new_items):
    """Run validation checks."""
    print("\n─── Verification ───")

    # Reload output to verify it's valid JSON
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 1, "Top-level should be array of length 1"
    assert data[0]["category"] == "colour", "Category should be 'colour'"

    items = data[0]["items"]
    print(f"Total items: {len(items)}")

    # Check all answers in 12 basic colors
    bad_colors = [it for it in items if it["answer"] not in BASIC_COLORS_SET]
    assert not bad_colors, f"Items with non-basic colors: {[(it['id'], it['answer']) for it in bad_colors]}"
    print("✓ All item answers are in the 12 basic colors")

    # Check new items (IDs 121+)
    new_items_in_output = [it for it in items if int(it["id"]) >= NEW_ITEM_START_ID]
    print(f"New items (ID >= {NEW_ITEM_START_ID}): {len(new_items_in_output)}")
    assert len(new_items_in_output) == 81, f"Expected 81 new items, got {len(new_items_in_output)}"

    # Check new item IDs are sequential
    new_ids = [int(it["id"]) for it in new_items_in_output]
    expected_ids = list(range(NEW_ITEM_START_ID, NEW_ITEM_START_ID + 81))
    assert new_ids == expected_ids, f"New IDs not sequential: {new_ids[:5]}...{new_ids[-5:]}"
    print(f"✓ New item IDs are sequential: {NEW_ITEM_START_ID}-{NEW_ITEM_START_ID + 80}")

    # Check all new items have order="text_image"
    bad_order = [it["id"] for it in new_items_in_output if it["order"] != "text_image"]
    assert not bad_order, f"Items with bad order: {bad_order}"
    print("✓ All new items have order='text_image'")

    # Check image paths
    for it in new_items_in_output:
        ic = it["image_clue"]
        paths = [
            ic["consistent"]["easy"],
            ic["consistent"]["hard"],
            ic["conflict"]["easy"],
            ic["conflict"]["hard"],
        ]
        for p in paths:
            full_path = os.path.join(BASE_DIR, "datasets", p)
            assert os.path.exists(full_path), f"Image not found: {full_path}"

    print("✓ All new item image paths point to existing files")

    # Check all new items have selected_text_priors
    empty_priors = [it["id"] for it in new_items_in_output if not it["selected_text_priors"]]
    assert not empty_priors, f"Items with no priors: {empty_priors}"
    print("✓ All new items have at least 1 text prior")

    # Check per-bin cap
    for it in new_items_in_output:
        bin_counts = defaultdict(int)
        for p in it["selected_text_priors"]:
            bin_counts[p["source_bin_id"]] += 1
        for bin_id, count in bin_counts.items():
            assert count <= MAX_PRIORS_PER_BIN, (
                f"Item {it['id']} has {count} priors from bin {bin_id} (cap={MAX_PRIORS_PER_BIN})"
            )
    print(f"✓ All items respect the {MAX_PRIORS_PER_BIN}-per-bin cap")

    # Check answer distribution in new items
    from collections import Counter
    expected_dist = {
        "black": 5, "blue": 6, "brown": 10, "cyan": 5, "gray": 7,
        "green": 4, "orange": 5, "pink": 8, "purple": 5, "red": 8,
        "white": 8, "yellow": 10,
    }
    actual_dist = Counter(it["answer"] for it in new_items_in_output)
    assert dict(actual_dist) == expected_dist, (
        f"Answer distribution mismatch:\n  Expected: {expected_dist}\n  Actual: {dict(actual_dist)}"
    )
    print("✓ Answer color distribution in new items matches expected")

    # Prior stats per item
    prior_counts = [len(it["selected_text_priors"]) for it in new_items_in_output]
    print(f"✓ Prior counts per new item: min={min(prior_counts)}, max={max(prior_counts)}, "
          f"avg={sum(prior_counts)/len(prior_counts):.1f}")

    print("\n─── All checks passed! ───")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Merging valid datasets into datasets/datasets.json")
    print("=" * 60)

    # Step 1: Load
    print("\n[1] Loading source data...")
    old_items, null_image = load_old_items()
    new_items_raw = load_new_items()
    prior_clues = build_prior_lookup()

    # Step 2: Copy images
    print("\n[2] Copying images...")
    copy_images(new_items_raw)

    # Step 3: Build new item bases
    print("\n[3] Building new item structures...")
    new_items_base = [build_new_item_base(item, i) for i, item in enumerate(new_items_raw)]

    # Step 4: Allocate priors
    print("\n[4] Allocating text priors...")
    new_items = allocate_priors(new_items_base, prior_clues)

    # Step 5: Write output
    print("\n[5] Writing merged output...")
    write_output(old_items, new_items, null_image)

    # Step 6: Verify
    print("\n[6] Verifying...")
    verify(new_items)

    print("\nDone!")


if __name__ == "__main__":
    main()
