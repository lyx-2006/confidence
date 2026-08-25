from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from dp_SA.attention_block.analyze import bh_fdr
from dp_SA.attention_block.masking import AttentionEdges

G_WINDOWS=((10,15),(12,17))
R_WINDOWS=((16,21),(18,23),(20,25),(22,27))
WINDOW_PAIRS=tuple((g,r) for g in G_WINDOWS for r in R_WINDOWS if g[1]<r[0])
FAMILIES=("C00","C10","C01","C11","CTRL")


@dataclass(frozen=True)
class Cell:
    key:str
    family:str
    gathering:tuple[int,int]|None=None
    readout:tuple[int,int]|None=None


def cells()->tuple[Cell,...]:
    values=[Cell("C00","C00")]
    values += [Cell(f"C10_G{g[0]}_{g[1]}","C10",g,None) for g in G_WINDOWS]
    values += [Cell(f"C01_R{r[0]}_{r[1]}","C01",None,r) for r in R_WINDOWS]
    values += [Cell(f"C11_G{g[0]}_{g[1]}_R{r[0]}_{r[1]}","C11",g,r) for g,r in WINDOW_PAIRS]
    values += [Cell(f"CTRL_G{g[0]}_{g[1]}_R{r[0]}_{r[1]}","CTRL",g,r) for g,r in WINDOW_PAIRS]
    if len(values)!=21 or len({v.key for v in values})!=21:raise AssertionError("Expected 21 unique cached cells")
    return tuple(values)


def _answer(spans:dict[str,Any])->tuple[int,...]:
    return tuple(range(int(spans["ANSWER"][0]),int(spans["ANSWER"][1])))


def base_pairs(spans:dict[str,Any])->set[tuple[int,int]]:
    answer=_answer(spans);panl=int(spans["PANL"]);plus1=int(spans["PANL_PLUS_1"]);sac=int(spans["SAC"])
    pairs={(query,source) for source in answer for query in range(source+1,sac+1)}
    pairs|={(query,panl) for query in range(panl+1,sac+1)}
    pairs|={(query,plus1) for query in range(plus1+1,sac+1)}
    return pairs


def restorations_by_layer(spans:dict[str,Any],cell:Cell)->dict[int,set[tuple[int,int]]]:
    answer=_answer(spans);panl=int(spans["PANL"]);plus1=int(spans["PANL_PLUS_1"]);sac=int(spans["SAC"])
    result={layer:set() for layer in range(28)}
    if cell.family in {"C10","C11"}:
        assert cell.gathering is not None
        for layer in range(cell.gathering[0],cell.gathering[1]+1):result[layer]|={(panl,s) for s in answer}
    if cell.family in {"C01","C11"}:
        assert cell.readout is not None
        for layer in range(cell.readout[0],cell.readout[1]+1):result[layer].add((sac,panl))
    if cell.family=="CTRL":
        assert cell.gathering is not None and cell.readout is not None
        for layer in range(cell.gathering[0],cell.gathering[1]+1):result[layer]|={(plus1,s) for s in answer}
        for layer in range(cell.readout[0],cell.readout[1]+1):result[layer].add((sac,plus1))
    return result


def layer_edges(spans:dict[str,Any],cell:Cell)->dict[int,AttentionEdges]:
    base=base_pairs(spans);restored=restorations_by_layer(spans,cell)
    if any(not values<=base for values in restored.values()):raise ValueError("Restoration not contained in common base")
    output={layer:AttentionEdges(tuple(sorted(base-values))) for layer,values in restored.items()}
    validate_layer_edges(spans,cell,output)
    return output


def validate_layer_edges(spans:dict[str,Any],cell:Cell,edges:dict[int,AttentionEdges])->None:
    if set(edges)!=set(range(28)):raise ValueError("All 28 layers must be masked")
    base=base_pairs(spans);expected=restorations_by_layer(spans,cell);answer=_answer(spans)
    panl=int(spans["PANL"]);plus1=int(spans["PANL_PLUS_1"]);sac=int(spans["SAC"])
    for layer in range(28):
        actual=set(edges[layer].pairs)
        if actual!=base-expected[layer]:raise ValueError(f"Layer {layer} edge set differs from preregistration")
        if any((sac,a) not in actual for a in answer):raise ValueError(f"SAC→Answer leakage at layer {layer}")
        if cell.family=="C10" and (sac,panl) not in actual:raise ValueError(f"C10 SAC→PANL leakage at layer {layer}")
        if cell.family=="C01" and any((panl,a) not in actual for a in answer):raise ValueError(f"C01 PANL→Answer leakage at layer {layer}")
        allowed_panl={(sac,panl)} if cell.family in {"C01","C11"} and cell.readout and cell.readout[0]<=layer<=cell.readout[1] else set()
        panl_leaks={(q,panl) for q in range(panl+1,sac+1) if (q,panl) not in actual}
        if panl_leaks!=allowed_panl:raise ValueError(f"PANL downstream leakage at layer {layer}: {panl_leaks-allowed_panl}")
    if cell.family=="CTRL":
        peer=Cell("peer","C11",cell.gathering,cell.readout);peer_restored=restorations_by_layer(spans,peer)
        for layer in range(28):
            mapped={(plus1 if q==panl else q,plus1 if s==panl else s) for q,s in peer_restored[layer]}
            if mapped!=expected[layer]:raise ValueError(f"CTRL asymmetry at layer {layer}")


def effects(m00:float,m10:float,m01:float,m11:float,mctrl:float)->tuple[float,float,float]:
    return m11-m10-m01+m00,m11-m00,m11-mctrl


def one_sided_sign_flip(values:Sequence[float],*,seed:int,repeats:int)->float:
    array=np.asarray(values,dtype=float);observed=float(array.mean());rng=np.random.default_rng(seed);exceed=0;done=0
    while done<repeats:
        size=min(2000,repeats-done);signs=rng.choice((-1.0,1.0),size=(size,len(array)))
        exceed+=int(np.count_nonzero((signs*array).mean(axis=1)>=observed));done+=size
    return float((exceed+1)/(repeats+1))


def iut_q(rows:list[dict[str,float]])->list[float]:
    p_iut=[max(float(r["p_interaction"]),float(r["p_bridge_gain"]),float(r["p_matched_gain"])) for r in rows]
    return bh_fdr(p_iut)
