from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

from .analyze import analyze
from .capture import _validate_frozen_sources, build_reuse_manifest, capture_clean, prepare_training_records
from .config import BASE_CAPTURE_ROOT, DIRECTIONS, EXPECTED_HASHES, PARENT_AUDIT_PREDICTIONS, PARENT_FAST_CLEAN, PARENT_FAST_CONFIG, PARENT_FAST_ROOT, PARENT_PANL_G_PROBE, RESULTS_ROOT, SEALED_TEST_MANIFEST, SMOKE_ROOT
from .io_utils import atomic_bytes, atomic_json, atomic_jsonl, ensure_layout, inventory, load_jsonl, require_output_root, sha256_file, verify_inventory
from .run import run_trajectory
from .train_probes import train_probes


def cpu_preflight() -> dict[str,Any]:
    from .analyze import component_additivity, persistent_onset
    from .io_utils import canonical_forward_key, stable_shard
    from .run import symmetric_derivative
    assert float(symmetric_derivative(1.0,-1.0))==2.0
    assert component_additivity(6,2,3)==1
    assert persistent_onset([{"layer":14,"mean":1,"ci_low":.1,"ci_high":2,"readout_reliable":True},{"layer":15,"mean":2,"ci_low":.2,"ci_high":3,"readout_reliable":True}])==14
    assert canonical_forward_key("x",None,0)==canonical_forward_key("x",None,0)
    assert stable_shard("x",1)==0
    return {"status":"passed","checks":["derivative","onset","additivity","canonical-key"]}


def _formal_unseal(root:Path)->list[dict[str,Any]]:
    if sha256_file(SEALED_TEST_MANIFEST)!=EXPECTED_HASHES[SEALED_TEST_MANIFEST]:raise ValueError("Sealed formal manifest hash mismatch")
    data=SEALED_TEST_MANIFEST.read_bytes();destination=root/"artifacts/manifests/runtime_manifest.jsonl";atomic_bytes(destination,data)
    if sha256_file(destination)!=EXPECTED_HASHES[SEALED_TEST_MANIFEST]:raise ValueError("Formal manifest copy changed")
    rows=load_jsonl(destination)
    if len(rows)!=100 or len({str(r["family_id"]) for r in rows})!=50:raise ValueError("Formal cardinality changed")
    train=load_jsonl(root/"artifacts/manifests/construction_manifest.jsonl")+load_jsonl(root/"artifacts/manifests/audit_manifest.jsonl")
    overlaps={field:len({str(r[field]) for r in train}&{str(r[field]) for r in rows}) for field in ("case_id","family_id","item_id","image_sha256")}
    if any(overlaps.values()):raise ValueError(f"Formal split leakage: {overlaps}")
    atomic_json(root/"artifacts/manifests/split_audit.json",{"status":"passed","construction_cases":882,"audit_cases":230,"formal_cases":100,"formal_families":50,"three_way_overlaps":overlaps,"runtime_manifest_sha256":sha256_file(destination),"formal_test_opened":True})
    return rows


def _verify_completion(root:Path)->dict[str,Any]:
    probes=load_jsonl(root/"artifacts/probes/probe_index.jsonl");trials=load_jsonl(root/"artifacts/trials/forward_trials.jsonl");cells=load_jsonl(root/"artifacts/trials/trajectory_cells.jsonl")
    baselines=[r for r in trials if r["direction"]=="baseline"]
    parity_failures=load_jsonl(root/"artifacts/diagnostics/alpha0_parity_failures.jsonl")
    required=[root/f"figures/{name}" for name in ("panl_trajectory.png","confidence_trajectory_heatmap.png","sa_trajectory_heatmap.png")]+[root/f"tables/{name}" for name in ("probe_metrics.csv","trajectory_readouts.csv","hidden_transport.csv","onset_summary.csv","component_additivity.csv","direct_final_endpoints.csv")]+[root/"README_RESULTS_zh.md"]
    failed_baselines=[r for r in baselines if not r.get("alpha0_parity",{}).get("passed")]
    gates={"probe_count":len(probes)==208,"forward_count":len(trials)==700,"hidden_per_forward":all((root/r["hidden_file"]).is_file() for r in trials),"trajectory_cells":len(cells)==100*3*4*13*4,"directions":{r["direction"] for r in trials if r["direction"]!="baseline"}==set(DIRECTIONS),"outputs":all(p.is_file() and p.stat().st_size>0 for p in required),"alpha0_parity_recorded":len(baselines)==100 and all("passed" in r.get("alpha0_parity",{}) for r in baselines),"alpha0_failures_recorded":{r["case_id"] for r in failed_baselines}=={r["case_id"] for r in parity_failures},"merge":json.loads((root/"artifacts/diagnostics/canonical_merge_audit.json").read_text())["canonical_equal"]}
    if not all(gates.values()):raise ValueError(f"Completion gates failed: {gates}")
    return gates


def run_smoke(root:Path,*,num_gpus:int,resume:bool)->dict[str,Any]:
    ensure_layout(root,resume=resume);cpu=cpu_preflight();prepared=prepare_training_records(root,resume=resume)
    audit_rows=load_jsonl(root/"artifacts/manifests/audit_manifest.jsonl")
    smoke_cases=[r for r in audit_rows if str(r["family_id"]) in __import__('dp_SA.confidence_steering.trajectory.config',fromlist=['SMOKE_FAMILIES']).SMOKE_FAMILIES]
    extra=[PARENT_AUDIT_PREDICTIONS,PARENT_PANL_G_PROBE,*[BASE_CAPTURE_ROOT/r["historical_hidden_file"] for r in smoke_cases]]
    before={**prepared["frozen_sources"],**inventory(extra)};atomic_json(root/"artifacts/diagnostics/source_hashes_before.json",before)
    first=run_trajectory(root,num_gpus=num_gpus,smoke=True);second=run_trajectory(root,num_gpus=num_gpus,smoke=True)
    if first["forward_count"]!=168 or not first["raw_direction_complete"] or second["new_gpu_forwards"]!=0:raise ValueError("Smoke count/raw/resume gate failed")
    verify_inventory(before);atomic_json(root/"artifacts/diagnostics/source_hashes_after.json",inventory(before))
    result={"status":"passed" if first["alpha0_historical_parity"] else "failed_alpha0_historical_parity","cpu_preflight":cpu,"actual_gpu_count":num_gpus,"actual_forward_count":168,"resume_second_new_forwards":0,"directions":list(DIRECTIONS),"raw_direction_included":True,"alpha0_historical_parity":first["alpha0_historical_parity"],"two_worker_check":"simulated canonical merge only; no real dual-GPU inference","formal_manifest_opened":False,"root":str(root)}
    atomic_json(root/"smoke_report.json",result);return result


def run_formal(root:Path,*,num_gpus:int,resume:bool)->dict[str,Any]:
    ensure_layout(root,resume=resume);cpu_preflight();prepared=prepare_training_records(root,resume=resume);reuse=build_reuse_manifest(root,prepared["construction"]+prepared["audit"])
    capture_clean(root,num_gpus=num_gpus,resume=resume);train_probes(root,resume=resume)
    # A successful, code-current smoke is mandatory before the sealed manifest is opened.
    smoke_root=SMOKE_ROOT/"formal_gate";smoke=run_smoke(smoke_root,num_gpus=num_gpus,resume=resume if smoke_root.exists() else False)
    if smoke["status"]!="passed":raise ValueError(f"Formal manifest remains sealed: smoke gate failed ({smoke['status']})")
    _formal_unseal(root);_validate_frozen_sources(include_formal=True)
    fast_rows=load_jsonl(PARENT_FAST_CLEAN)
    if len(fast_rows)!=100 or len({r["family_id"] for r in fast_rows})!=50:raise ValueError("Parent Fast clean baseline cardinality failed")
    runtime_rows=load_jsonl(root/"artifacts/manifests/runtime_manifest.jsonl")
    if {r["case_id"] for r in fast_rows}!={r["case_id"] for r in runtime_rows}:raise ValueError("Parent Fast clean/formal case identity mismatch")
    fast_protected=inventory([PARENT_FAST_CLEAN,PARENT_FAST_CONFIG,*[PARENT_FAST_ROOT/r["hidden_file"] for r in fast_rows]])
    run_trajectory(root,num_gpus=num_gpus,smoke=False);analyze(root,formal=True)
    protected={**reuse["protected"],**fast_protected};verify_inventory(protected);atomic_json(root/"artifacts/diagnostics/source_hashes_after.json",inventory(protected))
    gates=_verify_completion(root);completion={"status":"complete","completed_at_unix":time.time(),"gates":gates,"smoke_report":str(smoke_root/"smoke_report.json"),"num_gpus_execution_metadata":num_gpus};atomic_json(root/"completion.json",completion);return completion


def main(argv:Sequence[str]|None=None)->int:
    parser=argparse.ArgumentParser(description="Independent LAT→PANL/SAC layerwise trajectory experiment");parser.add_argument("--num-gpus",type=int,choices=(1,2),default=1);parser.add_argument("--resume",action="store_true");parser.add_argument("--output-root");parser.add_argument("--smoke",action="store_true");args=parser.parse_args(argv)
    if args.output_root:root=require_output_root(args.output_root)
    elif args.smoke:root=SMOKE_ROOT/f"round_{time.strftime('%Y%m%d_%H%M%S')}"
    else:root=RESULTS_ROOT
    result=run_smoke(root,num_gpus=args.num_gpus,resume=args.resume) if args.smoke else run_formal(root,num_gpus=args.num_gpus,resume=args.resume)
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
