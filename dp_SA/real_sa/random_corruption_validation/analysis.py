from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

from dp_SA.real_sa.metrics import case_metrics, family_cluster_bootstrap

from .config import BOOTSTRAP_REPEATS, MEAN_CASE_METRICS, REPLICATES, SEED
from .io_utils import atomic_csv, load_csv

METRICS=("G_R","D_I","D_T","phi_I","phi_T","J")


def _finite(value: Any) -> float:
    result=float(value)
    if not math.isfinite(result): raise ValueError(f"Non-finite value: {value}")
    return result


def _corr(rows: Sequence[dict[str,Any]], x: str, y: str, kind: str) -> float|None:
    if len(rows)<3: return None
    a=np.asarray([_finite(row[x]) for row in rows]); b=np.asarray([_finite(row[y]) for row in rows])
    if np.ptp(a)==0 or np.ptp(b)==0: return None
    return float(pearsonr(a,b).statistic if kind=="pearson" else spearmanr(a,b).statistic)


def _bootstrap(rows: Sequence[dict[str,Any]], fn: Callable[[Sequence[dict[str,Any]]],float|None])->dict[str,Any]:
    return family_cluster_bootstrap(rows,fn,repeats=BOOTSTRAP_REPEATS,seed=SEED)


def _groups(rows: Sequence[dict[str,Any]]):
    yield "all",list(rows)
    for field,values in (("dataset_condition",("conflict_easy","conflict_hard")),
                         ("answer_side",("follow_image","follow_text"))):
        for value in values: yield value,[row for row in rows if row[field]==value]


def build_case_metrics(score_rows: Sequence[dict[str,Any]], cohort_rows: Sequence[dict[str,Any]]) -> tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    by_key={(str(row["case_id"]),row["replicate"],str(row["corruption_condition"])):row for row in score_rows}
    replicates=[]; aggregated=[]
    for source in cohort_rows:
        case=str(source["case_id"]); clean=by_key[(case,None,"clean")]
        v11=_finite(clean["fixed_answer_probability"]); condition_values={name:[] for name in ("10_random_text","01_gaussian_image","00_both_random")}
        for rep in range(REPLICATES):
            values={name:_finite(by_key[(case,rep,name)]["fixed_answer_probability"]) for name in condition_values}
            result=case_metrics(v11,values["10_random_text"],values["01_gaussian_image"],values["00_both_random"])
            replicates.append({"case_id":case,"family_id":source["family_id"],"item_id":source["item_id"],
                "dataset_condition":source["condition"],"answer_side":source["answer_side"],"replicate":rep,
                "v11":v11,"v10":values["10_random_text"],"v01":values["01_gaussian_image"],"v00":values["00_both_random"],**result})
            for name,value in values.items(): condition_values[name].append(value)
        means={name:float(np.mean(values)) for name,values in condition_values.items()}
        result=case_metrics(v11,means["10_random_text"],means["01_gaussian_image"],means["00_both_random"])
        per_case_sd={f"{metric}_replicate_sd":float(np.std([record[metric] for record in replicates if record["case_id"]==case],ddof=1)) for metric in METRICS}
        verbal=_finite(source["soft_sa_image_score"])
        aggregated.append({"case_id":case,"family_id":source["family_id"],"item_id":source["item_id"],
            "dataset_condition":source["condition"],"answer_side":source["answer_side"],"fixed_answer":source["phase0_normalized_answer"],
            "verbal_sa":verbal,"signed_verbal_sa":2*verbal-1,"v11":v11,"v10":means["10_random_text"],
            "v01":means["01_gaussian_image"],"v00":means["00_both_random"],**result,**per_case_sd})
    return replicates,aggregated


def comparisons(new_rows: Sequence[dict[str,Any]]) -> list[dict[str,Any]]:
    old={row["case_id"]:row for row in load_csv(MEAN_CASE_METRICS)}
    joined=[]
    for row in new_rows:
        prior=old.get(str(row["case_id"]));
        if prior is None: raise ValueError(f"Mean Real SA row missing: {row['case_id']}")
        joined.append({**row,**{f"mean_{metric}":_finite(prior[metric]) for metric in METRICS}})
    output=[]
    for group,members in _groups(joined):
        for metric in METRICS:
            x=f"mean_{metric}"; y=metric
            functions={"pearson":lambda s,x=x,y=y:_corr(s,x,y,"pearson"),
                       "spearman":lambda s,x=x,y=y:_corr(s,x,y,"spearman"),
                       "mae":lambda s,x=x,y=y:float(np.mean([abs(_finite(r[x])-_finite(r[y])) for r in s])),
                       "sign_agreement":lambda s,x=x,y=y:float(np.mean([np.sign(_finite(r[x]))==np.sign(_finite(r[y])) for r in s])) if metric=="G_R" else None}
            for statistic,fn in functions.items():
                if statistic=="sign_agreement" and metric!="G_R": continue
                ci=_bootstrap(members,fn); output.append({"group":group,"metric":metric,"statistic":statistic,
                    "estimate":fn(members),"ci_low":ci["low"],"ci_high":ci["high"],"case_count":len(members),
                    "family_count":len({r['family_id'] for r in members}),"bootstrap_repeats":BOOTSTRAP_REPEATS,"valid_bootstrap_repeats":ci["valid"]})
    return output


def verbal_comparisons(rows: Sequence[dict[str,Any]]) -> list[dict[str,Any]]:
    def ols(sample):
        x=np.asarray([r["signed_verbal_sa"] for r in sample]); y=np.asarray([r["G_R"] for r in sample]); X=np.column_stack((np.ones(len(x)),x))
        intercept,slope=np.linalg.lstsq(X,y,rcond=None)[0]; pred=X@np.array([intercept,slope]); denom=np.sum((y-y.mean())**2)
        return float(intercept),float(slope),float(1-np.sum((y-pred)**2)/denom) if denom else None
    output=[]
    for group,members in _groups(rows):
        base=ols(members)
        fns={"pearson":lambda s:_corr(s,"signed_verbal_sa","G_R","pearson"),"spearman":lambda s:_corr(s,"signed_verbal_sa","G_R","spearman"),
             "r2":lambda s:ols(s)[2],"intercept":lambda s:ols(s)[0],"slope":lambda s:ols(s)[1]}
        values={"intercept":base[0],"slope":base[1],"r2":base[2],"pearson":fns["pearson"](members),"spearman":fns["spearman"](members)}
        for statistic,fn in fns.items():
            ci=_bootstrap(members,fn); output.append({"group":group,"statistic":statistic,"estimate":values[statistic],
                "ci_low":ci["low"],"ci_high":ci["high"],"case_count":len(members),"family_count":len({r['family_id'] for r in members}),
                "bootstrap_repeats":BOOTSTRAP_REPEATS,"valid_bootstrap_repeats":ci["valid"]})
    return output


def stability(replicate_rows: Sequence[dict[str,Any]]) -> list[dict[str,Any]]:
    output=[]; pairs=list(itertools.combinations(range(REPLICATES),2))
    for group,members in _groups(replicate_rows):
        case_rows={case:[r for r in members if r["case_id"]==case] for case in sorted({r["case_id"] for r in members})}
        families={case:rs[0]["family_id"] for case,rs in case_rows.items()}
        for metric in METRICS:
            sdrows=[{"family_id":families[c],"value":float(np.std([r[metric] for r in rs],ddof=1))} for c,rs in case_rows.items()]
            for stat,fn in (("mean_sd",lambda s:float(np.mean([r["value"] for r in s]))),("median_sd",lambda s:float(np.median([r["value"] for r in s]))),("p95_sd",lambda s:float(np.percentile([r["value"] for r in s],95)))):
                ci=_bootstrap(sdrows,fn); output.append({"group":group,"metric":metric,"summary":stat,"replicate_a":"","replicate_b":"","estimate":fn(sdrows),"ci_low":ci["low"],"ci_high":ci["high"],"case_count":len(sdrows),"valid_bootstrap_repeats":ci["valid"]})
            pair_values={}
            case_records=[{"family_id":families[c],"values":[next(r[metric] for r in rs if r["replicate"]==rep) for rep in range(REPLICATES)]} for c,rs in case_rows.items()]
            for a,b in pairs:
                records=[{"family_id":families[c],"a":next(r[metric] for r in rs if r["replicate"]==a),"b":next(r[metric] for r in rs if r["replicate"]==b)} for c,rs in case_rows.items()]
                for kind in ("pearson","spearman"):
                    fn=lambda s,k=kind:_corr(s,"a","b",k); ci=_bootstrap(records,fn); estimate=fn(records); pair_values.setdefault(kind,[]).append(estimate)
                    output.append({"group":group,"metric":metric,"summary":f"pairwise_{kind}","replicate_a":a,"replicate_b":b,"estimate":estimate,"ci_low":ci["low"],"ci_high":ci["high"],"case_count":len(records),"valid_bootstrap_repeats":ci["valid"]})
            for kind,values in pair_values.items():
                def mean_pairs(sample,k=kind):
                    estimates=[]
                    for a,b in pairs:
                        records=[{"family_id":r["family_id"],"a":r["values"][a],"b":r["values"][b]} for r in sample]
                        value=_corr(records,"a","b",k)
                        if value is not None: estimates.append(value)
                    return float(np.mean(estimates)) if estimates else None
                valid=[v for v in values if v is not None]; ci=_bootstrap(case_records,mean_pairs)
                output.append({"group":group,"metric":metric,"summary":f"mean_pairwise_{kind}","replicate_a":"","replicate_b":"","estimate":float(np.mean(valid)) if valid else None,"ci_low":ci["low"],"ci_high":ci["high"],"case_count":len(case_rows),"valid_bootstrap_repeats":ci["valid"]})
    return output


def _figures(root:Path,new_rows:Sequence[dict[str,Any]])->None:
    old={r["case_id"]:r for r in load_csv(MEAN_CASE_METRICS)}; x=np.array([float(old[r["case_id"]]["G_R"]) for r in new_rows]); y=np.array([r["G_R"] for r in new_rows])
    fig,ax=plt.subplots(figsize=(6,5)); ax.scatter(x,y,s=22,alpha=.75); lo=min(x.min(),y.min()); hi=max(x.max(),y.max()); ax.plot([lo,hi],[lo,hi],"--",color="grey"); ax.axhline(0,color="black",lw=.7); ax.axvline(0,color="black",lw=.7); ax.set(xlabel="Mean-embedding G_R",ylabel="Random-corruption G_R"); fig.tight_layout(); fig.savefig(root/"figures/new_vs_mean_gr.png",dpi=180); plt.close(fig)
    x=np.array([r["signed_verbal_sa"] for r in new_rows]); y=np.array([r["G_R"] for r in new_rows]); slope,intercept=np.polyfit(x,y,1)
    fig,ax=plt.subplots(figsize=(6,5)); ax.scatter(x,y,s=22,alpha=.75); grid=np.linspace(x.min(),x.max(),100); ax.plot(grid,intercept+slope*grid,color="tab:red"); ax.axhline(0,color="black",lw=.7); ax.axvline(0,color="black",lw=.7); ax.set(xlabel="Signed verbal SA",ylabel="Random-corruption G_R"); fig.tight_layout(); fig.savefig(root/"figures/new_gr_vs_signed_verbal_sa.png",dpi=180); plt.close(fig)


def analyze(root:Path,score_rows:Sequence[dict[str,Any]],cohort_rows:Sequence[dict[str,Any]])->dict[str,Any]:
    replicates,aggregated=build_case_metrics(score_rows,cohort_rows); comparison=comparisons(aggregated); verbal=verbal_comparisons(aggregated); stable=stability(replicates)
    atomic_csv(root/"tables/replicate_case_metrics.csv",replicates); atomic_csv(root/"tables/aggregated_case_metrics.csv",aggregated); atomic_csv(root/"tables/new_vs_mean_real_sa.csv",comparison); atomic_csv(root/"tables/new_real_sa_vs_verbal_sa.csv",verbal); atomic_csv(root/"tables/randomization_stability.csv",stable); _figures(root,aggregated)
    return {"case_count":len(aggregated),"replicate_case_count":len(replicates),"tables":5,"figures":2}

__all__=["analyze","build_case_metrics","comparisons","stability","verbal_comparisons"]
