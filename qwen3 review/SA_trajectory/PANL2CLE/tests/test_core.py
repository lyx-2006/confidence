from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.metrics import r2_score

from SA_trajectory.PANL2CLE.analyze import effect_values
from SA_trajectory.PANL2CLE.config import CAUSAL_PAIRS
from SA_trajectory.PANL2CLE.contracts import atomic_bf16_npz, expected_logical_count, expected_physical_count, load_bf16, validate_layer_design
from SA_trajectory.PANL2CLE.hooks import PANLCLEMediationHook, tensor_and_tail
from SA_trajectory.PANL2CLE.probes import choose_alpha, pipeline, raw_parameters


def test_causal_layer_design_and_counts():
    assert CAUSAL_PAIRS == ((14,15),(14,17),(14,19),(14,21),(16,17),(16,19),(16,21),(18,19),(18,21))
    validate_layer_design(); assert expected_physical_count(81) == 3483; assert expected_logical_count(81) == 5832
    with pytest.raises(ValueError): validate_layer_design(((14,15),(16,15)))


def test_bf16_roundtrip_is_bitwise(tmp_path):
    source=torch.randn(32,dtype=torch.bfloat16);path=tmp_path/"hidden.npz";atomic_bf16_npz(path,{"x":source});loaded=load_bf16(path,"x")
    assert torch.equal(source.view(torch.uint16),loaded.view(torch.uint16))


def test_tensor_and_tuple_outputs():
    value=torch.zeros(1,3,4);assert tensor_and_tail(value)==(value,None);tensor,tail=tensor_and_tail((value,"cache"));assert tensor is value and tail==("cache",)
    with pytest.raises(TypeError):tensor_and_tail("bad")


def test_hook_injects_and_patches_once():
    layers=torch.nn.ModuleList([torch.nn.Identity() for _ in range(22)]);modules=SimpleNamespace(language_layers=layers,num_hidden_layers=22,hidden_size=4)
    source=torch.tensor([1,2,3,4],dtype=torch.bfloat16);hook=PANLCLEMediationHook(modules,prefill_sequence_length=3,panl_position=0,cle_position=2,panl_layer=14,steering_vector=torch.ones(4),patch_layer=15,patch_source=source,capture_cle_layers=(15,))
    hidden=torch.zeros(1,3,4,dtype=torch.bfloat16)
    with hook:
        after14=layers[14](hidden);after15=layers[15](after14)
    hook.validate();assert hook.injection_count==1 and hook.patch_count==1;assert torch.equal(after15[0,2],source);assert torch.equal(after15[0,1],hidden[0,1]);assert hook.replacement_bitwise_equal


def test_probe_alpha_selection_and_raw_expression():
    rng=np.random.default_rng(42);x=rng.normal(size=(40,8));y=x[:,0]*.7-x[:,1]*.2;folds=np.tile(np.arange(1,5),10)
    alpha,oof,trace=choose_alpha(x,y,folds);assert alpha in {row["alpha"] for row in trace};assert r2_score(y,oof)>.9
    model=pipeline(alpha);model.fit(x,y);weight,intercept=raw_parameters(model);assert np.max(np.abs(model.predict(x)-(x@weight+intercept)))<1e-10


def test_four_cell_effect_formulas():
    rows=[{"condition":"C0","v":1.0},{"condition":"C1","v":1.8},{"condition":"C2","v":1.3},{"condition":"C3","v":1.4}]
    assert effect_values(rows,"v") == pytest.approx({"total":.8,"residual":.3,"attenuation":.5,"transfer":.4,"interaction":.1})
