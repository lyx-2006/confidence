from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .config import CANONICAL_COLORS, CONDITIONS, FAMILY_MANIFEST, RESULTS_ROOT, SEED, SOURCE_MANIFEST
from .io_utils import atomic_json, atomic_jsonl, canonical_hash, ensure_layout, load_jsonl, sha256_file


def record_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    item = str(row["item_id"])
    return (int(item) if item.isdigit() else math.inf, item, int(row["prior_index"]), str(row["condition"]), str(row["case_id"]))


class UnionFind:
    def __init__(self, values: Iterable[str]): self.parent = {value: value for value in values}
    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]; value = self.parent[value]
        return value
    def union(self, left: str, right: str) -> None:
        a,b=self.find(left),self.find(right)
        if a != b: self.parent[max(a,b)] = min(a,b)


def build_families(rows: Sequence[dict[str, Any]], historical: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    items=sorted({str(r["item_id"]) for r in rows}, key=lambda x:(int(x) if x.isdigit() else math.inf,x)); union=UnionFind(items)
    by_hash: dict[str,list[str]]=collections.defaultdict(list)
    for row in rows: by_hash[str(row["image_sha256"])].append(str(row["item_id"]))
    for linked in by_hash.values():
        linked=sorted(set(linked))
        for item in linked[1:]: union.union(linked[0],item)
    known=set(items)
    for family in historical:
        linked=sorted(set(map(str,family.get("item_ids",[]))) & known)
        for item in linked[1:]: union.union(linked[0],item)
    components: dict[str,set[str]]=collections.defaultdict(set)
    for item in items: components[union.find(item)].add(item)
    manifests=[]; mapping={}
    for component in sorted(components.values(),key=lambda x:min((int(v) if v.isdigit() else math.inf,v) for v in x)):
        item_ids=sorted(component,key=lambda x:(int(x) if x.isdigit() else math.inf,x)); hashes=sorted({str(r["image_sha256"]) for r in rows if str(r["item_id"]) in component})
        family_id="family_"+canonical_hash({"item_ids":item_ids,"image_sha256s":hashes})[:16]
        cases=sorted(str(r["case_id"]) for r in rows if str(r["item_id"]) in component)
        manifests.append({"family_id":family_id,"item_ids":item_ids,"image_sha256s":hashes,"case_ids":cases,"case_count":len(cases)})
        for item in item_ids: mapping[item]=family_id
    return manifests,mapping


def decision_side(row: dict[str, Any]) -> str | None:
    answer=str(row["phase0_normalized_answer"]); text=str(row["text_answer"]); image=str(row["image_answer"])
    if answer == text and answer != image: return "follow_text"
    if answer == image and answer != text: return "follow_image"
    return None


def paired_options(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str,int],dict[str,dict[str,Any]]]=collections.defaultdict(dict)
    for row in rows: by_key[str(row["item_id"]),int(row["prior_index"])][str(row["condition"])]=row
    output=[]
    for (item,prior),values in by_key.items():
        if set(values) != set(CONDITIONS): continue
        sides=tuple(decision_side(values[c]) for c in CONDITIONS)
        if None in sides: continue
        if len({str(values[c]["family_id"]) for c in CONDITIONS}) != 1: raise AssertionError("Pair crosses families")
        output.append({"item_id":item,"prior_index":prior,"family_id":values[CONDITIONS[0]]["family_id"],"sides":sides,"rows":values})
    return sorted(output,key=lambda x:(record_sort_key(x["rows"][CONDITIONS[0]])))


def _base_test_constraints(options: Sequence[dict[str, Any]]) -> tuple[list[np.ndarray],list[float],list[float]]:
    n=len(options); A=[np.ones(n)]; low=[50.0]; high=[50.0]
    for index in range(2):
        A.append(np.asarray([o["sides"][index]=="follow_text" for o in options],float)); low.append(25); high.append(25)
    for family in sorted({o["family_id"] for o in options}):
        A.append(np.asarray([o["family_id"]==family for o in options],float)); low.append(0); high.append(1)
    return A,low,high


def select_test(options: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]],dict[str,Any]]:
    n=len(options); A,low,high=_base_test_constraints(options)
    if not n: raise ValueError("No paired test options")
    color_vectors={color:np.asarray([sum(str(o["rows"][c]["phase0_normalized_answer"])==color for c in CONDITIONS) for o in options],float) for color in CANONICAL_COLORS}
    feasible=None; color_bound=None
    for delta in range(101):
        lo=max(0,8-delta); hi=9+delta; a=list(A); l=list(low); h=list(high)
        for color in CANONICAL_COLORS: a.append(color_vectors[color]); l.append(lo); h.append(hi)
        candidate=milp(np.zeros(n),integrality=np.ones(n),bounds=Bounds(np.zeros(n),np.ones(n)),constraints=LinearConstraint(np.stack(a),l,h))
        if candidate.success: feasible=(a,l,h); color_bound=(lo,hi); break
    if feasible is None:
        # Maximum feasible diagnostic without silently accepting it.
        result=milp(-np.ones(n),integrality=np.ones(n),bounds=Bounds(np.zeros(n),np.ones(n)),constraints=LinearConstraint(np.stack(A[1:]),np.zeros(len(A)-1),np.asarray(high[1:])))
        raise ValueError(f"Strict 100-record test infeasible; maximum paired families={int(round(-result.fun)) if result.success else 'unknown'}")
    a,l,h=feasible; variables=n+len(CANONICAL_COLORS); AA=[]; ll=[]; hh=[]
    for row,lo,hi in zip(a,l,h,strict=True): AA.append(np.r_[row,np.zeros(len(CANONICAL_COLORS))]); ll.append(lo); hh.append(hi)
    for ci,color in enumerate(CANONICAL_COLORS):
        row=np.zeros(variables); row[:n]=color_vectors[color]; row[n+ci]=-1; AA.append(row); ll.append(-np.inf); hh.append(100/12)
        row=np.zeros(variables); row[:n]=-color_vectors[color]; row[n+ci]=-1; AA.append(row); ll.append(-np.inf); hh.append(-100/12)
    bounds=Bounds(np.r_[np.zeros(n),np.zeros(12)],np.r_[np.ones(n),np.full(12,100)]); integrality=np.r_[np.ones(n),np.zeros(12)]
    objective=np.r_[np.zeros(n),np.ones(12)]; balanced=milp(objective,integrality=integrality,bounds=bounds,constraints=LinearConstraint(np.stack(AA),ll,hh))
    if not balanced.success: raise RuntimeError("Color balance optimizer failed")
    constraint=np.r_[np.zeros(n),np.ones(12)]; AA.append(constraint); ll.append(balanced.fun-1e-7); hh.append(balanced.fun+1e-7)
    tie=np.asarray([int(hashlib.sha256(f"{SEED}|test|{o['item_id']}|{o['prior_index']}".encode()).hexdigest()[:15],16)/16**15 for o in options])
    chosen=milp(np.r_[tie,np.zeros(12)],integrality=integrality,bounds=bounds,constraints=LinearConstraint(np.stack(AA),ll,hh),options={"mip_rel_gap":0})
    if not chosen.success: raise RuntimeError("Deterministic test tie-break failed")
    selected=[options[i] for i,value in enumerate(chosen.x[:n]) if value>.5]
    records=[]
    for option in selected:
        for condition in CONDITIONS: records.append({**option["rows"][condition],"split":"test"})
    records.sort(key=record_sort_key)
    audit={"paired_option_count":n,"selected_family_count":len(selected),"selected_record_count":len(records),"color_bounds":list(color_bound),"color_counts":dict(collections.Counter(str(r["phase0_normalized_answer"]) for r in records)),"cell_counts":{f"{c}__{s}":sum(r["condition"]==c and decision_side(r)==s for r in records) for c in CONDITIONS for s in ("follow_text","follow_image")}}
    return records,audit


def _balanced_sample(rows: Sequence[dict[str, Any]], count: int, strata: Sequence[tuple[str,Callable[[dict[str,Any]],str]]], hard: Sequence[tuple[Callable[[dict[str,Any]],bool],int]] = ()) -> list[dict[str, Any]]:
    n=len(rows)
    if n < count: raise ValueError(f"Calibration requires {count}, only {n} available")
    # Binary rows plus one continuous absolute-deviation variable per stratum value.
    groups=[]
    for name,getter in strata:
        values=sorted({getter(row) for row in rows})
        target=count/len(values)
        for value in values: groups.append((name,value,getter,target))
    variables=n+len(groups); A=[]; low=[]; high=[]
    row=np.zeros(variables); row[:n]=1; A.append(row); low.append(count); high.append(count)
    for predicate,total in hard:
        row=np.zeros(variables); row[:n]=[predicate(value) for value in rows]; A.append(row); low.append(total); high.append(total)
    for gi,(_name,value,getter,target) in enumerate(groups):
        vector=np.asarray([getter(r)==value for r in rows],float)
        row=np.zeros(variables); row[:n]=vector; row[n+gi]=-1; A.append(row); low.append(-np.inf); high.append(target)
        row=np.zeros(variables); row[:n]=-vector; row[n+gi]=-1; A.append(row); low.append(-np.inf); high.append(-target)
    objective=np.r_[np.asarray([int(canonical_hash([SEED,"cal",r.get("case_id"),r.get("unique_key")])[:12],16)/16**12 for r in rows])*1e-7,np.ones(len(groups))]
    result=milp(objective,integrality=np.r_[np.ones(n),np.zeros(len(groups))],bounds=Bounds(np.zeros(variables),np.r_[np.ones(n),np.full(len(groups),count)]),constraints=LinearConstraint(np.stack(A),low,high))
    if not result.success: raise ValueError(f"Exact calibration selection infeasible: {result.message}")
    return [rows[i] for i,value in enumerate(result.x[:n]) if value>.5]


def _specs(train: Sequence[dict[str, Any]]) -> tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    text={}; image={}
    for row in train:
        tk=(str(row["item_id"]),int(row["prior_index"])); text.setdefault(tk,{"modality":"text","unique_key":list(tk),"item_id":tk[0],"prior_index":tk[1],"prior_bin":row["prior_bin"],"target_answer":row["text_answer"],"question":row["question"],"text_clue":row["text_clue"],"answer_classes":row["answer_classes"]})
        ik=(str(row["item_id"]),str(row["condition"]),str(row["image_sha256"])); image.setdefault(ik,{"modality":"image","unique_key":list(ik),"item_id":ik[0],"condition":ik[1],"image_hash":ik[2],"target_answer":row["image_answer"],"question":row["question"],"image_path":row["image_path"],"answer_classes":row["answer_classes"]})
    return list(text.values()),list(image.values())


def select_text_calibration(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) < 200: raise ValueError("Text calibration requires 200 unique keys")
    bins=sorted({str(r["prior_bin"]) for r in rows},key=lambda x:float(x.split("-",1)[0])); counts=collections.Counter(str(r["prior_bin"]) for r in rows)
    scarce=min(bins,key=lambda value:(counts[value],value)); scarce_count=min(counts[scarce],40)
    remaining=200-scarce_count; others=[value for value in bins if value != scarce]; lo=remaining//len(others); hi=math.ceil(remaining/len(others))
    n=len(rows); A=[np.ones(n),np.asarray([str(r["prior_bin"])==scarce for r in rows],float)]; low=[200,scarce_count]; high=[200,scarce_count]
    for value in others: A.append(np.asarray([str(r["prior_bin"])==value for r in rows],float)); low.append(lo); high.append(hi)
    for color in CANONICAL_COLORS: A.append(np.asarray([str(r["target_answer"])==color for r in rows],float)); low.append(16); high.append(17)
    tie=np.asarray([int(canonical_hash([SEED,"text_calibration",r["unique_key"]])[:15],16)/16**15 for r in rows])
    result=milp(tie,integrality=np.ones(n),bounds=Bounds(np.zeros(n),np.ones(n)),constraints=LinearConstraint(np.stack(A),low,high))
    if not result.success: raise ValueError(f"Exact text calibration infeasible: {result.message}")
    return [rows[i] for i,value in enumerate(result.x) if value>.5]


def select_image_calibration(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) < 200: raise ValueError("Image calibration requires 200 unique keys")
    n=len(rows); pool=collections.Counter(str(r["target_answer"]) for r in rows); lower=min(pool.values()); base=[np.ones(n)]; low=[200.0]; high=[200.0]
    for condition in CONDITIONS: base.append(np.asarray([r["condition"]==condition for r in rows],float)); low.append(100); high.append(100)
    solution=None
    for upper in range(math.ceil(200/12),201):
        A=list(base); lo=list(low); hi=list(high)
        for color in CANONICAL_COLORS: A.append(np.asarray([str(r["target_answer"])==color for r in rows],float)); lo.append(min(lower,pool[color])); hi.append(upper)
        tie=np.asarray([int(canonical_hash([SEED,"image_calibration",r["unique_key"]])[:15],16)/16**15 for r in rows])
        result=milp(tie,integrality=np.ones(n),bounds=Bounds(np.zeros(n),np.ones(n)),constraints=LinearConstraint(np.stack(A),lo,hi))
        if result.success: solution=result; break
    if solution is None: raise ValueError("Exact image calibration infeasible")
    return [rows[i] for i,value in enumerate(solution.x) if value>.5]


def build_split(root: Path) -> dict[str, Any]:
    source=sorted(load_jsonl(SOURCE_MANIFEST),key=record_sort_key)
    if not source: raise FileNotFoundError(SOURCE_MANIFEST)
    families,item_to_family=build_families(source,load_jsonl(FAMILY_MANIFEST))
    rows=[{**row,"family_id":item_to_family[str(row["item_id"])]} for row in source]
    # The optimizer receives an explicit allow-list, so prohibited confidence,
    # difficulty, SA, hidden, probe, and intervention fields cannot influence it.
    allowed=("case_id","item_id","prior_index","condition","image_sha256","phase0_normalized_answer","text_answer","image_answer","family_id")
    selection_rows=[{key:row[key] for key in allowed} for row in rows]
    try:
        selected_minimal,test_audit=select_test(paired_options(selection_rows))
    except Exception as exc:
        atomic_json(root/"shared/split_audit.json",{"status":"failed","stage":"strict_test","error":{"type":type(exc).__name__,"message":str(exc)},"no_constraints_relaxed":True})
        raise
    selected_ids={r["case_id"] for r in selected_minimal}; test=[{**row,"split":"test"} for row in rows if row["case_id"] in selected_ids]
    test_families={r["family_id"] for r in test}; test_ids={r["case_id"] for r in test}
    train=[{**row,"split":"train"} for row in rows if row["family_id"] not in test_families]
    excluded=[{**row,"split":"excluded_test_family_variant"} for row in rows if row["family_id"] in test_families and row["case_id"] not in test_ids]
    text_pool,image_pool=_specs(train)
    try:
        text_cal=select_text_calibration(text_pool); image_cal=select_image_calibration(image_pool)
    except Exception as exc:
        atomic_json(root/"shared/split_audit.json",{"status":"failed","stage":"calibration","error":{"type":type(exc).__name__,"message":str(exc)},"no_constraints_relaxed":True})
        raise
    # Include only chosen test pairs plus all train records in the downstream manifest.
    probe=sorted(train+test,key=record_sort_key)
    atomic_jsonl(root/"shared/manifests/family_manifest.jsonl",families)
    atomic_jsonl(root/"shared/manifests/test_manifest.jsonl",test)
    atomic_jsonl(root/"shared/manifests/probe_train_manifest.jsonl",train)
    atomic_jsonl(root/"shared/manifests/probe_manifest.jsonl",probe)
    atomic_jsonl(root/"shared/manifests/excluded_test_family_variants.jsonl",excluded)
    atomic_jsonl(root/"shared/manifests/text_calibration_manifest.jsonl",sorted(text_cal,key=lambda r:tuple(r["unique_key"])))
    atomic_jsonl(root/"shared/manifests/image_calibration_manifest.jsonl",sorted(image_cal,key=lambda r:tuple(r["unique_key"])))
    train_items={r["item_id"] for r in train}; test_items={r["item_id"] for r in test}; train_hash={r["image_sha256"] for r in train}; test_hash={r["image_sha256"] for r in test}
    text_test={(r["item_id"],r["prior_index"]) for r in test}; text_train={(r["item_id"],r["prior_index"]) for r in train}; image_test={(r["item_id"],r["condition"],r["image_sha256"]) for r in test}; image_train={(r["item_id"],r["condition"],r["image_sha256"]) for r in train}
    overlaps={"sample":len({r["case_id"] for r in train}&test_ids),"item":len(train_items&test_items),"family":len({r["family_id"] for r in train}&test_families),"image_hash":len(train_hash&test_hash),"text_unique_key":len(text_train&text_test),"image_unique_key":len(image_train&image_test)}
    if any(overlaps.values()): raise AssertionError(f"Leakage: {overlaps}")
    audit={"status":"passed","seed":SEED,"source":{"path":str(SOURCE_MANIFEST.resolve()),"sha256":sha256_file(SOURCE_MANIFEST),"records":len(rows)},"families":{"count":len(families),"item_count":len({r['item_id'] for r in rows}),"image_hash_count":len({r['image_sha256'] for r in rows})},"test":test_audit|{"item_count":len(test_items),"image_hash_count":len(test_hash),"excluded_variant_count":len(excluded)},"probe_train":{"record_count":len(train),"family_count":len({r['family_id'] for r in train}),"item_count":len(train_items),"image_hash_count":len(train_hash)},"calibration":{"text_count":len(text_cal),"image_count":len(image_cal),"text_pool_count":len(text_pool),"image_pool_count":len(image_pool),"text_prior_bins":dict(collections.Counter(r['prior_bin'] for r in text_cal)),"text_colors":dict(collections.Counter(r['target_answer'] for r in text_cal)),"image_conditions":dict(collections.Counter(r['condition'] for r in image_cal)),"image_colors":dict(collections.Counter(r['target_answer'] for r in image_cal))},"overlaps":overlaps}
    atomic_json(root/"shared/split_audit.json",audit)
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-root",default=str(RESULTS_ROOT)); parser.add_argument("--resume",action="store_true")
    args=parser.parse_args(argv); root=ensure_layout(args.output_root,resume=args.resume); print(json.dumps(build_split(root),ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
