#!/usr/bin/env python3
"""统计 v1~v4 结果中 consistent_easy > consistent_hard > conflict_hard > conflict_easy 的条目数"""

import json
import os

FILES = [
    "confidence_test/output/v1_simplified.json",
    "confidence_test/output/v2_simplified.json",
    "confidence_test/output/v3_simplified.json",
    "confidence_test/output/v4_simplified.json",
]

total = 0
matched = 0
per_file = {}

for fpath in FILES:
    with open(fpath) as f:
        data = json.load(f)

    f_total = 0
    f_matched = 0

    for item in data:
        for prior in item.get("priors", []):
            cond = prior.get("conditions", {})
            # 取出四个条件下的第四项（index 3）
            ce = cond.get("consistent_easy", [None, None, None, None])[3]
            ch = cond.get("consistent_hard", [None, None, None, None])[3]
            cfh = cond.get("conflict_hard", [None, None, None, None])[3]
            cfe = cond.get("conflict_easy", [None, None, None, None])[3]

            if None in (ce, ch, cfh, cfe):
                continue

            f_total += 1
            if ce > ch > cfh > cfe:
                f_matched += 1

    per_file[fpath] = (f_matched, f_total)
    total += f_total
    matched += f_matched

print("=" * 60)
print("条件: consistent_easy > consistent_hard > conflict_hard > conflict_easy")
print("      （比较各 condition 数组的第 4 个值，即 index 3）")
print("=" * 60)

for fpath, (m, t) in per_file.items():
    pct = m / t * 100 if t else 0
    print(f"  {fpath}")
    print(f"    满足 / 总数 = {m} / {t} = {pct:.2f}%")
    print()

print("-" * 60)
print(f"  全部合计: {matched} / {total} = {matched / total * 100:.2f}%")
print("=" * 60)
