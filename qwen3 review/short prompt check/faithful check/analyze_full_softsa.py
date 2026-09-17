from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[2]
for candidate in (REPOSITORY_ROOT, HERE):
    if str(candidate) not in sys.path: sys.path.insert(0, str(candidate))
from dp_SA.io_utils import atomic_json, load_jsonl
from config import BOOTSTRAP_REPEATS, OUTPUT_ROOT, SEED


def _csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True); fields=sorted({k for r in rows for k in r})
    fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8",newline="") as h:
            w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(rows); h.flush(); os.fsync(h.fileno())
        os.replace(tmp,path)
    except BaseException:
        try: os.unlink(tmp)
        except FileNotFoundError: pass
        raise


def _pairs(rows,x,y):
    p=[(float(r[x]),float(r[y])) for r in rows if r.get(x) is not None and r.get(y) is not None]
    return (np.array([a for a,b in p]),np.array([b for a,b in p])) if p else (np.array([]),np.array([]))


def _corr(x,y,kind):
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:return None,None
    z=stats.pearsonr(x,y) if kind=='pearson' else stats.spearmanr(x,y);return float(z.statistic),float(z.pvalue)


def _fit(x,y):
    if len(x)<2 or np.ptp(x)==0:return {'intercept':None,'slope':None,'r2':None,'mae':None,'rmse':None}
    X=np.column_stack([np.ones(len(x)),x]);b=np.linalg.lstsq(X,y,rcond=None)[0];e=y-X@b;tot=np.sum((y-y.mean())**2)
    return {'intercept':float(b[0]),'slope':float(b[1]),'r2':float(1-np.sum(e**2)/tot) if tot>0 else None,'mae':float(np.mean(abs(e))),'rmse':float(np.sqrt(np.mean(e**2)))}


def _boot(rows,fn,repeats):
    groups=defaultdict(list)
    for r in rows:groups[str(r['item_id'])].append(r)
    ids=sorted(groups)
    if len(ids)<2:return [None,None]
    rng=np.random.default_rng(SEED); vals=[]
    for _ in range(repeats):
        sample=[r for i in rng.choice(ids,len(ids),replace=True) for r in groups[str(i)]];v=fn(sample)
        if math.isfinite(v):vals.append(v)
    return [float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5))] if vals else [None,None]


def _metric(rows,group,xkey,ykey,label,repeats):
    x,y=_pairs(rows,xkey,ykey);p,pp=_corr(x,y,'pearson');s,sp=_corr(x,y,'spearman');fit=_fit(x,y)
    def c(sample,kind):
        a,b=_pairs(sample,xkey,ykey);v,_=_corr(a,b,kind);return float('nan') if v is None else v
    def slope(sample):
        a,b=_pairs(sample,xkey,ykey);v=_fit(a,b)['slope'];return float('nan') if v is None else v
    pci=_boot(rows,lambda z:c(z,'pearson'),repeats);sci=_boot(rows,lambda z:c(z,'spearman'),repeats);lci=_boot(rows,slope,repeats)
    signs=[np.sign(float(r[xkey]))==np.sign(float(r[ykey])) for r in rows if float(r[xkey])!=0 and float(r[ykey])!=0]
    return {'group':group,'comparison':label,'n':len(x),'pearson':p,'pearson_p_value':pp,'pearson_ci_low':pci[0],'pearson_ci_high':pci[1],'spearman':s,'spearman_p_value':sp,'spearman_ci_low':sci[0],'spearman_ci_high':sci[1],**fit,'slope_ci_low':lci[0],'slope_ci_high':lci[1],'sign_agreement_rate':float(np.mean(signs)) if signs else None}


def analyze(root:Path,full_root:Path,repeats:int):
    trials={r['case_id']:r for r in load_jsonl(root/'trials.jsonl') if r.get('status')=='completed'}; full={r['case_id']:r for r in load_jsonl(full_root/'full_softsa.jsonl') if r.get('status')=='completed'}
    if len(trials)!=110 or len(full)!=110:raise ValueError(f'Expected 110 records, found {len(trials)} and {len(full)}')
    rows=[]
    for cid,t in trials.items():
        f=full[cid];rows.append({'case_id':cid,'item_id':t['item_id'],'difficulty':t['difficulty'],'cma_signed':t['cma_logit']['cma_signed'],'cma_log_probability_signed':t['cma_log_probability']['cma_signed'],'full_sa_signed':f['soft_sa_signed'],'full_sa_raw':f['soft_sa_image_score'],'short_sa_signed':t['short_sa']['soft_sa_signed'],'full_minus_short':f['soft_sa_signed']-t['short_sa']['soft_sa_signed']})
    table=full_root/'tables';_csv(table/'case_level.csv',rows);metrics=[]
    for g,sub in [('overall',rows),('easy',[r for r in rows if r['difficulty']=='easy']),('hard',[r for r in rows if r['difficulty']=='hard'])]:
        metrics += [_metric(sub,g,'cma_signed','full_sa_signed','cma_vs_full_soft_sa',repeats),_metric(sub,g,'short_sa_signed','full_sa_signed','short_vs_full_soft_sa',repeats),_metric(sub,g,'cma_log_probability_signed','full_sa_signed','log_probability_cma_vs_full_soft_sa',repeats)]
    _csv(table/'correlation_and_fit.csv',metrics)
    out={'status':'complete','n':110,'bootstrap_repeats':repeats,'metrics':metrics,'mean_full_sa_signed':float(np.mean([r['full_sa_signed'] for r in rows])),'mean_short_sa_signed':float(np.mean([r['short_sa_signed'] for r in rows])),'mean_full_minus_short':float(np.mean([r['full_minus_short'] for r in rows])),'full_vs_short_mae':float(np.mean(abs(np.array([r['full_minus_short'] for r in rows])))),'full_vs_short_sign_agreement':float(np.mean(np.sign([r['full_sa_signed'] for r in rows])==np.sign([r['short_sa_signed'] for r in rows]))) }
    atomic_json(full_root/'summary.json',out);return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=OUTPUT_ROOT.parent/'faithful_check_extended'/'balanced_subset');p.add_argument('--full-root',type=Path,default=OUTPUT_ROOT.parent/'faithful_check_extended'/'balanced_subset'/'full_softsa');p.add_argument('--bootstrap-repeats',type=int,default=BOOTSTRAP_REPEATS);a=p.parse_args();print(json.dumps(analyze(a.root,a.full_root,a.bootstrap_repeats),ensure_ascii=False,indent=2))
