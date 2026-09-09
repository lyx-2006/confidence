from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

from .config import AUDIT_MANIFEST, CELL_MANIFEST, CONSTRUCTION_DISTRIBUTION, TEST_MANIFEST
from .io_utils import load_jsonl


def _round_robin(rows:Sequence[dict[str,Any]],group_key:str,count:int)->list[dict[str,Any]]:
    groups:dict[str,list[dict[str,Any]]]=defaultdict(list)
    for row in sorted(rows,key=lambda r:str(r["case_id"])):groups[str(row[group_key])].append(row)
    output=[];offset=0
    while len(output)<count:
        added=False
        for key in sorted(groups):
            if offset<len(groups[key]):output.append(groups[key][offset]);added=True
            if len(output)==count:return output
        if not added:break
        offset += 1
    if len(output)!=count:raise ValueError(f"Cannot sample {count} rows from {len(rows)}")
    return output


def validation_audit(count:int)->list[dict[str,Any]]:
    return _round_robin(load_jsonl(AUDIT_MANIFEST),"family_id",count)


def validation_test(count:int)->list[dict[str,Any]]:
    rows=load_jsonl(TEST_MANIFEST);exploratory_target=round(count*sum(r["test_status"]!="confirmatory" for r in rows)/len(rows));confirmatory_target=count-exploratory_target
    confirmatory=_round_robin([r for r in rows if r["test_status"]=="confirmatory"],"test_answer",confirmatory_target)
    exploratory=sorted([r for r in rows if r["test_status"]!="confirmatory"],key=lambda r:str(r["case_id"]))[:exploratory_target]
    result=sorted(confirmatory+exploratory,key=lambda r:str(r["case_id"]))
    if len(result)!=count:raise ValueError(f"Validation test sample is {len(result)}, expected {count}")
    return result


def validation_geometry_cells(case_budget:int)->list[dict[str,Any]]:
    cells=load_jsonl(CELL_MANIFEST);distribution=load_jsonl(CONSTRUCTION_DISTRIBUTION);fold=0
    eligible=sorted(r["answer"] for r in distribution if int(r["fold"])==fold and r["eligible_for_direction"])
    available={(answer,side):sorted([r for r in cells if int(r["fold"])==fold and r["answer"]==answer and r["sa_side"]==side],key=lambda r:(str(r["family_id"]),str(r.get("cell_id","")))) for answer in eligible for side in ("high_text","high_image")}
    selected=[];level=0
    while True:
        proposal=selected+[available[key][level] for key in sorted(available) if level<len(available[key])]
        unique={str(case) for cell in proposal for case in cell["case_ids"]}
        if len(unique)>case_budget:break
        selected=proposal;level += 1
    if not selected or any(not any(r["answer"]==answer and r["sa_side"]==side for r in selected) for answer in eligible for side in ("high_text","high_image")):raise ValueError("Validation geometry cannot cover every answer/side")
    return selected
