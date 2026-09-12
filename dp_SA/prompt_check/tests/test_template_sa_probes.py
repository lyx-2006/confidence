from __future__ import annotations

import numpy as np

from dp_SA.prompt_check.io_utils import load_jsonl
from dp_SA.prompt_check.train_template_sa_probes import (
    AUDIT_SOURCE, CONSTRUCTION_SOURCE, DEFAULT_LAYERS, DEFAULT_POSITIONS,
    DEFAULT_TEMPLATES, T0_FROZEN_LAYERS, T0_NEW_LAYERS, _frozen_t0,
    choose_alpha,
)


def test_frozen_t0_split_contract_and_no_leakage():
    construction=load_jsonl(CONSTRUCTION_SOURCE);audit=load_jsonl(AUDIT_SOURCE)
    assert (len(construction),len(audit))==(882,230)
    assert {int(r["outer_fold"]) for r in construction}=={1,2,3,4}
    assert {int(r["outer_fold"]) for r in audit}=={0}
    for field in ("case_id","family_id","item_id","image_sha256"):
        assert not ({str(r[field]) for r in construction}&{str(r[field]) for r in audit})


def test_registered_probe_grid_is_48_cells():
    newly_fitted=len(DEFAULT_TEMPLATES)*len(DEFAULT_POSITIONS)*len(DEFAULT_LAYERS)+len(DEFAULT_POSITIONS)*len(T0_NEW_LAYERS)
    frozen=len(DEFAULT_POSITIONS)*len(T0_FROZEN_LAYERS)
    assert (newly_fitted,frozen,newly_fitted+frozen)==(40,8,48)


def test_four_fold_alpha_selection_returns_complete_oof():
    rng=np.random.default_rng(4);x=rng.normal(size=(24,5));y=x[:,0]*.2+rng.normal(scale=.01,size=24);folds=np.repeat([1,2,3,4],6)
    alpha,oof,trace=choose_alpha(x,y,folds)
    assert np.isfinite(oof).all() and len(trace)==9 and alpha in {r["alpha"] for r in trace}


def test_eight_frozen_t0_probe_references_are_valid():
    assert len(_frozen_t0())==8
