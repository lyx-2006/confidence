from __future__ import annotations

import json

import pytest

from dp_SA.attention_block.staged_panl_bridge.core import Cell,WINDOW_PAIRS,base_pairs,cells,effects,iut_q,layer_edges,one_sided_sign_flip,restorations_by_layer
from dp_SA.attention_block.staged_panl_bridge.run import c10_parity,ensure_config,pending_cells


def spans():return {"ANSWER":[2,4],"PANL":7,"PANL_PLUS_1":8,"SAC":10,"sequence_length":11}


def test_preregistered_cells_are_cached_to_exactly_21():
    values=cells();assert len(values)==21;assert len({v.key for v in values})==21;assert len(WINDOW_PAIRS)==7


def test_c00_blocks_common_base_in_every_layer():
    value=spans();base=base_pairs(value);edges=layer_edges(value,Cell("C00","C00"))
    assert all(set(edges[layer].pairs)==base for layer in range(28))


def test_c10_restores_gathering_only_and_never_leaks_sac_to_panl():
    value=spans();cell=Cell("x","C10",(10,15),None);edges=layer_edges(value,cell)
    for layer in range(28):
        blocked=set(edges[layer].pairs);assert (10,7) in blocked
        assert ((7,2) not in blocked and (7,3) not in blocked)==(10<=layer<=15)


def test_c01_blocks_panl_answer_all_layers_and_only_restores_readout():
    value=spans();cell=Cell("x","C01",None,(18,23));edges=layer_edges(value,cell)
    for layer in range(28):
        blocked=set(edges[layer].pairs);assert {(7,2),(7,3)}<=blocked
        assert ((10,7) not in blocked)==(18<=layer<=23)


def test_c11_and_ctrl_exact_layer_specific_sets_and_symmetry():
    value=spans();c11=Cell("x","C11",(10,15),(18,23));ctrl=Cell("y","CTRL",(10,15),(18,23))
    a=restorations_by_layer(value,c11);b=restorations_by_layer(value,ctrl);layer_edges(value,c11);layer_edges(value,ctrl)
    for layer in range(28):
        mapped={(8 if q==7 else q,8 if source==7 else source) for q,source in a[layer]};assert mapped==b[layer]
        assert (10,2) in set(layer_edges(value,c11)[layer].pairs)


def test_interaction_and_iut_bh():
    assert effects(1,2,3,8,4)==(4,7,4)
    rows=[{"p_interaction":.01,"p_bridge_gain":.02,"p_matched_gain":.03},{"p_interaction":.2,"p_bridge_gain":.01,"p_matched_gain":.01}]
    assert iut_q(rows)==pytest.approx([.06,.2])


def test_sign_flip_is_centered_null_and_directional():
    assert one_sided_sign_flip([1.0]*20,seed=3,repeats=2000)<.01
    assert one_sided_sign_flip([-1.0]*20,seed=3,repeats=2000)>.99


def test_fingerprint_and_missing_cell_resume(tmp_path):
    path=tmp_path/"run_config.json";config={"fingerprint":"same","x":1};ensure_config(path,config,resume=False);ensure_config(path,config,resume=True)
    with pytest.raises(ValueError):ensure_config(path,{"fingerprint":"different"},resume=True)
    complete={("case",c.key) for c in cells()[:5]};pending=pending_cells("case",complete)
    assert len(pending)==16 and {c.key for c in pending}.isdisjoint({c.key for c in cells()[:5]})
    assert json.loads(path.read_text())==config


def test_c10_numeric_isolation_parity_gate():
    row={"class_logits":[1.0,2.0],"margin":.5,"soft_sa":4.0,"blocked_class":3}
    assert c10_parity(row,dict(row))["passed"]
    changed={**row,"class_logits":[1.0,2.01]}
    with pytest.raises(RuntimeError,match="C10 isolation parity failed"):c10_parity(row,changed)
