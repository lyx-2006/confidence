from __future__ import annotations

import argparse,json,time
from pathlib import Path
from typing import Any,Sequence
import torch

from confidence_test.runtime_imports import load_runtime
from dp_SA.attention_block.config import INFERENCE_PATH,LOGIT_PARITY_TOLERANCE,MODEL_PATH,ROW_SUM_TOLERANCE,SOFT_PARITY_TOLERANCE
from dp_SA.attention_block.masking import AttentionBlockContext
from dp_SA.attention_block.run import _forward,_margin
from dp_SA.attention_block.sources import prepare_case
from dp_SA.attention_block.spans import locate_spans
from dp_SA.io_utils import append_jsonl,atomic_json,atomic_jsonl,canonical_hash,load_jsonl,sha256_file
from dp_SA.soft_score import class_token_ids
from layer_metacognition.model_adapter import resolve_language_modules
from .core import CONDITIONS,WINDOWS,edges_for_condition,validate_symmetry

ROOT=Path(__file__).resolve().parents[3]
DEFAULT_BASE=ROOT/'dp_SA/attention_block/outputs/formal_both_seed42_w12_20260823T093446Z'
DEFAULT_OUTPUT_PARENT=Path(__file__).resolve().parent/'outputs'

def _selection(base:Path,smoke:bool):
    rows=json.loads((base/'delayed_case_manifest.json').read_text())
    if len(rows)!=100 or {s:sum(r['test_side']==s for r in rows) for s in ('image_side','text_side')}!={'image_side':50,'text_side':50}: raise ValueError('Frozen delayed manifest is not 50+50')
    return [r for s in ('image_side','text_side') for r in [x for x in rows if x['test_side']==s][:2]] if smoke else rows

def _config(base:Path,selection,smoke:bool):
    files=[Path(__file__),Path(__file__).with_name('core.py'),Path(__file__).with_name('analyze.py'),Path(__file__).with_name('run_pipeline.py'),Path(__file__).parents[1]/'masking.py',Path(__file__).parents[1]/'spans.py']
    value={'format_version':1,'experiment':'direct_panl_bridge','base_output':str(base.resolve()),'base_config_sha256':sha256_file(base/'run_config.json'),'manifest_sha256':sha256_file(base/'delayed_case_manifest.json'),'clean_sha256':sha256_file(base/'clean_baselines.jsonl'),'spans_sha256':sha256_file(base/'delayed_token_spans.jsonl'),'selection_hash':canonical_hash(selection),'prompt_hash':canonical_hash([r['phase1_prompt_hash'] for r in selection]),'model_path':str(MODEL_PATH.resolve()),'model_config_sha256':sha256_file(MODEL_PATH/'config.json'),'processor_config_sha256':sha256_file(MODEL_PATH/'preprocessor_config.json'),'inference_path':str(INFERENCE_PATH.resolve()),'windows':WINDOWS,'conditions':CONDITIONS,'seed':42,'bootstrap_repeats':2000,'sign_flip_repeats':20000,'smoke':smoke,'implementation_sha256':{str(p.resolve()):sha256_file(p) for p in files}}
    value['fingerprint']=canonical_hash(value);return value

def ensure_config(path:Path,config:dict[str,Any],*,resume:bool)->None:
    if path.exists():
        if json.loads(path.read_text()).get('fingerprint')!=config['fingerprint']:raise ValueError('Fingerprint changed; refusing resume')
        if not resume:raise FileExistsError(f'{path.parent} exists; use --resume')
    else:atomic_json(path,config)

def run(output:Path,base:Path=DEFAULT_BASE,*,smoke=False,resume=False):
    output=output.resolve();output.mkdir(parents=True,exist_ok=True);selection=_selection(base,smoke);config=_config(base,selection,smoke);cp=output/'run_config.json'
    existed=cp.exists();ensure_config(cp,config,resume=resume)
    if not existed:atomic_jsonl(output/'selection_manifest.jsonl',selection)
    clean={r['case_id']:r for r in load_jsonl(base/'clean_baselines.jsonl') if r['arm']=='delayed'};frozen={r['case_id']:r for r in load_jsonl(base/'delayed_token_spans.jsonl')}
    bp=output/'blocked_results.jsonl';pp=output/'clean_parity.jsonl';sp=output/'token_spans.jsonl';fp=output/'failures.jsonl'
    for p in (bp,pp,sp,fp):p.touch(exist_ok=True)
    completed={(r['case_id'],r['condition'],int(r['window_start'])) for r in load_jsonl(bp)};parity={r['case_id'] for r in load_jsonl(pp)};saved={r['case_id'] for r in load_jsonl(sp)}
    runtime=load_runtime(INFERENCE_PATH);inference=runtime.QwenVLInference(str(MODEL_PATH));modules=resolve_language_modules(inference.model);tokenizer=getattr(inference.processor,'tokenizer',inference.processor);ids=class_token_ids(tokenizer)
    if getattr(inference.model.config,'_attn_implementation',None)!='eager' or modules.num_hidden_layers!=28:raise RuntimeError('Expected eager 28-layer model')
    started=time.time();total=len(selection)*(1+len(CONDITIONS)*len(WINDOWS))
    try:
      for row in selection:
       try:
        rendered,inputs=prepare_case(inference,row);spans=locate_spans(tokenizer,rendered,inputs,row);f={k:v for k,v in frozen[row['case_id']].items() if k not in {'arm','case_id'}}
        if canonical_hash(spans)!=canonical_hash(f):raise RuntimeError(f"Frozen spans changed: {row['case_id']}")
        validate_symmetry(spans)
        if row['case_id'] not in saved:append_jsonl(sp,{'case_id':row['case_id'],'item_id':row['item_id'],'test_side':row['test_side'],**spans});saved.add(row['case_id'])
        baseline=clean[row['case_id']];target=int(baseline['clean_class'])
        if row['case_id'] not in parity:
            logits,score=_forward(inference.model,inputs,spans['SAC'],ids);md=max(abs(float(a)-float(b)) for a,b in zip(logits,baseline['class_logits']));sd=abs(float(score['soft_sa_image_score'])-float(baseline['soft_sa_image_score']))
            if md>LOGIT_PARITY_TOLERANCE or sd>SOFT_PARITY_TOLERANCE or int(score['argmax_hard_class'])!=target:raise RuntimeError(f"Clean parity failed {row['case_id']}: {md},{sd}")
            append_jsonl(pp,{'case_id':row['case_id'],'max_abs_logit_difference':md,'abs_soft_sa_difference':sd,'hard_equal':True});parity.add(row['case_id'])
        for condition in CONDITIONS:
         edges=edges_for_condition(spans,condition)
         for start,end in WINDOWS:
          key=(row['case_id'],condition,start)
          if key in completed:continue
          before=time.perf_counter()
          with AttentionBlockContext(modules.language_layers,layer_indices=range(start,end+1),edges=edges,sequence_length=spans['sequence_length'],row_sum_tolerance=ROW_SUM_TOLERANCE) as context:logits,score=_forward(inference.model,inputs,spans['SAC'],ids)
          margin=_margin(logits,target);diag=context.diagnostics()
          append_jsonl(bp,{'case_id':row['case_id'],'item_id':row['item_id'],'test_side':row['test_side'],'condition':condition,'window_start':start,'window_end':end,'window_center':(start+end)/2,'class_logits':logits,'clean_class':target,'blocked_class':int(score['argmax_hard_class']),'margin':margin,'clean_margin':float(baseline['clean_margin']),'logit_margin_disruption':float(baseline['clean_margin'])-margin,'soft_sa':float(score['soft_sa_image_score']),'delta_soft_sa':float(score['soft_sa_image_score'])-float(baseline['soft_sa_image_score']),'token_changed':int(score['argmax_hard_class'])!=target,'edge_count':len(edges.pairs),'attention_diagnostics':diag,'elapsed_seconds':time.perf_counter()-before});completed.add(key)
          elapsed=time.time()-started;done=len(parity)+len(completed);atomic_json(output/'progress.json',{'status':'running','completed':done,'expected':total,'failed':len(load_jsonl(fp)),'elapsed_seconds':elapsed,'estimated_remaining_seconds':elapsed/max(1,done)*(total-done)})
        del inputs
       except Exception as exc:append_jsonl(fp,{'case_id':row.get('case_id'),'error_type':type(exc).__name__,'error':str(exc)});raise
      result={'status':'complete','clean_parity':len(parity),'blocked':len(completed),'failures':len(load_jsonl(fp)),'elapsed_seconds':time.time()-started,'estimated_remaining_seconds':0.0};atomic_json(output/'completion.json',result);atomic_json(output/'progress.json',result);return result
    finally:
      del inference
      if torch.cuda.is_available():torch.cuda.empty_cache()

def main(argv:Sequence[str]|None=None):
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path);p.add_argument('--base-output',type=Path,default=DEFAULT_BASE);p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args(argv);out=a.output_dir or DEFAULT_OUTPUT_PARENT/time.strftime(('smoke' if a.smoke else 'formal')+'_seed42_%Y%m%dT%H%M%SZ',time.gmtime());run(out,a.base_output.resolve(),smoke=a.smoke,resume=a.resume);return 0
if __name__=='__main__':raise SystemExit(main())
