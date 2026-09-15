from __future__ import annotations
import sys
from pathlib import Path
import pytest

SHORT_ROOT=Path(__file__).resolve().parents[2];REPO=SHORT_ROOT.parent.parent
for p in (REPO,SHORT_ROOT):
    if str(p) not in sys.path:sys.path.insert(0,str(p))

from cle_swap.config import DIRECTIONS,LAYERS
from cle_swap.prepare import select_manifest

def test_grid_contract():
    assert LAYERS==(12,16,18,20,22,24,26,28,30)
    assert DIRECTIONS==("reverse_to_short","short_to_reverse")
    assert 50*(2+len(LAYERS)*len(DIRECTIONS))==1000
    assert 50*len(LAYERS)*len(DIRECTIONS)==900

def test_selection_text_first_extremes_and_item_disjoint():
    short=[];reverse=[]
    for side,base in (("text",.01),("image",.99)):
        for i in range(250):
            case=f"{side}{i}";score=base+(i*.001 if side=="text" else -i*.001)
            common={"status":"completed","case_id":case,"item_id":case,"phase0_answer_fingerprint":case,
                    "image_sha256":case,"phase0_normalized_answer":"a"}
            short.append({**common,"soft_sa_image_score":score});reverse.append({**common,"soft_sa_image_score":.2})
    out=select_manifest(short,reverse)
    assert len(out)==50 and len({r["item_id"] for r in out})==50
    assert sum(r["test_side"]=="text_side" for r in out)==25
    assert sum(r["test_side"]=="image_side" for r in out)==25
    assert max(r["short_clean_sa"] for r in out if r["test_side"]=="text_side")==pytest.approx(.034)
    assert min(r["short_clean_sa"] for r in out if r["test_side"]=="image_side")==pytest.approx(.966)
