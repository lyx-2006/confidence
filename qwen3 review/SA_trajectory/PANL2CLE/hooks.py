from __future__ import annotations

from typing import Any, Sequence

import torch

from layer_metacognition.model_adapter import LanguageModules


def tensor_and_tail(output: Any) -> tuple[torch.Tensor, tuple[Any, ...] | None]:
    if isinstance(output, torch.Tensor): return output, None
    if isinstance(output, (tuple,list)) and output and isinstance(output[0],torch.Tensor): return output[0],tuple(output[1:])
    raise TypeError(f"Unsupported decoder output: {type(output)!r}")


class PANLCLEMediationHook:
    """One-prefill PANL addition, CLE replacement, and CLE trajectory capture."""

    def __init__(self, modules: LanguageModules, *, prefill_sequence_length: int, panl_position: int,
                 cle_position: int, panl_layer: int | None = None, steering_vector: torch.Tensor | None = None,
                 patch_layer: int | None = None, patch_source: torch.Tensor | None = None,
                 capture_cle_layers: Sequence[int] = ()) -> None:
        self.modules=modules;self.prefill_sequence_length=int(prefill_sequence_length);self.panl_position=int(panl_position);self.cle_position=int(cle_position)
        if not 0<=self.panl_position<self.cle_position<self.prefill_sequence_length: raise ValueError("Invalid PANL/CLE positions")
        self.panl_layer=None if panl_layer is None else int(panl_layer);self.steering_vector=None if steering_vector is None else steering_vector.detach().reshape(-1)
        if (self.panl_layer is None)!=(self.steering_vector is None): raise ValueError("panl_layer and steering_vector must be paired")
        self.patch_layer=None if patch_layer is None else int(patch_layer);self.patch_source=None if patch_source is None else patch_source.detach().reshape(-1)
        if (self.patch_layer is None)!=(self.patch_source is None): raise ValueError("patch_layer and patch_source must be paired")
        if self.panl_layer is not None and self.patch_layer is not None and self.patch_layer<=self.panl_layer: raise ValueError("CLE patch must be downstream of PANL steering")
        for value in (self.steering_vector,self.patch_source):
            if value is not None and value.numel()!=modules.hidden_size: raise ValueError("Hidden-size mismatch")
        self.capture_cle_layers=tuple(sorted(set(map(int,capture_cle_layers))));layers=set(self.capture_cle_layers)
        if self.panl_layer is not None: layers.add(self.panl_layer)
        if self.patch_layer is not None: layers.add(self.patch_layer)
        if not layers or any(x<0 or x>=modules.num_hidden_layers for x in layers): raise ValueError("Invalid hook layers")
        self.layers=tuple(sorted(layers));self.hook_calls={x:0 for x in self.layers};self.prefill_hits={x:0 for x in self.layers};self.cle_hidden={}
        self.injection_count=0;self.patch_count=0;self.non_target_unchanged=True;self.replacement_bitwise_equal=None;self.activation_dtype=None;self._handles=[]

    @staticmethod
    def _outside_equal(before: torch.Tensor,after: torch.Tensor,positions:set[int])->bool:
        cursor=0
        for position in sorted(positions):
            if not torch.equal(before[:,cursor:position],after[:,cursor:position]): return False
            cursor=position+1
        return torch.equal(before[:,cursor:],after[:,cursor:])

    def _hook(self,layer:int,output:Any)->Any:
        self.hook_calls[layer]+=1;tensor,tail=tensor_and_tail(output)
        if int(tensor.shape[1])!=self.prefill_sequence_length: return output
        if self.prefill_hits[layer]: raise RuntimeError(f"Repeated full prefill at L{layer}")
        if tensor.ndim!=3 or tensor.shape[0]!=1 or tensor.shape[2]!=self.modules.hidden_size: raise ValueError(f"Bad hidden shape: {tuple(tensor.shape)}")
        if tensor.dtype!=torch.bfloat16: raise TypeError(f"Expected bf16 activation, got {tensor.dtype}")
        self.prefill_hits[layer]+=1;self.activation_dtype="bfloat16";patched=tensor;modified=set()
        if layer==self.panl_layer:
            patched=tensor.clone();vector=self.steering_vector.to(device=tensor.device,dtype=tensor.dtype);patched[0,self.panl_position]+=vector;modified.add(self.panl_position);self.injection_count+=1
        if layer==self.patch_layer:
            if patched is tensor: patched=tensor.clone()
            source=self.patch_source.to(device=tensor.device)
            if source.dtype!=torch.bfloat16: raise TypeError("CLE patch source must be bf16")
            patched[0,self.cle_position]=source;modified.add(self.cle_position);self.patch_count+=1
            self.replacement_bitwise_equal=torch.equal(patched[0,self.cle_position].detach().cpu().view(torch.uint16),self.patch_source.cpu().view(torch.uint16))
            if not self.replacement_bitwise_equal: raise RuntimeError("CLE replacement is not bitwise equal")
        if modified and not self._outside_equal(tensor,patched,modified): self.non_target_unchanged=False;raise RuntimeError("Non-target hidden changed")
        if layer in self.capture_cle_layers: self.cle_hidden[layer]=patched[0,self.cle_position].detach().cpu().clone()
        return output if patched is tensor else (patched if tail is None else (patched,*tail))

    def __enter__(self):
        for layer in self.layers: self._handles.append(self.modules.language_layers[layer].register_forward_hook(lambda _m,_a,o,index=layer:self._hook(index,o)))
        return self

    def __exit__(self,*_args):
        for handle in self._handles: handle.remove()
        self._handles.clear()

    def validate(self)->None:
        if any(v!=1 for v in self.hook_calls.values()) or any(v!=1 for v in self.prefill_hits.values()): raise RuntimeError(f"Hook count gate failed: {self.hook_calls}/{self.prefill_hits}")
        if set(self.cle_hidden)!=set(self.capture_cle_layers): raise RuntimeError("CLE capture incomplete")
        if self.injection_count!=int(self.steering_vector is not None): raise RuntimeError("PANL injection count failed")
        if self.patch_count!=int(self.patch_source is not None): raise RuntimeError("CLE patch count failed")
        if self.patch_source is not None and not self.replacement_bitwise_equal: raise RuntimeError("CLE replacement gate failed")

    def diagnostics(self)->dict[str,Any]:
        self.validate();return {"layers":list(self.layers),"hook_calls":self.hook_calls,"prefill_hits":self.prefill_hits,"activation_dtype":self.activation_dtype,"panl_injection_count":self.injection_count,"cle_patch_count":self.patch_count,"replacement_bitwise_equal":self.replacement_bitwise_equal,"non_target_unchanged":self.non_target_unchanged}

