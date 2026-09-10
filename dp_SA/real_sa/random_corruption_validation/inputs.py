from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Sequence

import torch

from dp_SA.prompts import ANSWER_PREFILL
from dp_SA.real_sa.protocol import locate_evidence, phase0_messages
from layer_metacognition.conversation_builder import prepare_multimodal_inputs, render_continued_assistant

from .io_utils import canonical_hash, sha256_file


def messages_with_image(row: dict[str, Any], image_path: Path | None) -> tuple[str, list[dict[str, Any]]]:
    prompt, messages = phase0_messages(row)
    result = copy.deepcopy(messages)
    if image_path is not None:
        if not image_path.is_file(): raise FileNotFoundError(image_path)
        result[0]["content"][0]["image"] = str(image_path.resolve())
    return prompt, result


def apply_text_tokens(inputs: Any, details: dict[str, Any], replacement_ids: Sequence[int],
                      banned_ids: frozenset[int]) -> tuple[Any, dict[str, Any]]:
    positions=list(map(int,details["text_positions"])); replacement=list(map(int,replacement_ids))
    original_ids=inputs.input_ids.clone(); original_mask=inputs.attention_mask.clone()
    if len(replacement)!=len(positions): raise ValueError("Replacement/text span length mismatch")
    if set(replacement)&set(map(int,banned_ids)): raise ValueError("Replacement contains a banned token")
    corrupted=inputs.copy(); corrupted["input_ids"]=inputs.input_ids.clone()
    corrupted["input_ids"][0,positions]=torch.as_tensor(replacement,device=inputs.input_ids.device,dtype=inputs.input_ids.dtype)
    if tuple(corrupted["input_ids"].shape)!=tuple(original_ids.shape): raise ValueError("Text replacement changed sequence length")
    if not torch.equal(corrupted["attention_mask"],original_mask): raise ValueError("Text replacement changed attention mask")
    outside=torch.ones(original_ids.shape[1],dtype=torch.bool,device=original_ids.device); outside[positions]=False
    outside_equal=bool(torch.equal(corrupted["input_ids"][:,outside],original_ids[:,outside]))
    if not outside_equal: raise ValueError("Text replacement changed tokens outside clue span")
    replaced=[int(value) for value in corrupted["input_ids"][0,positions].tolist()]
    if replaced!=replacement: raise ValueError("Text replacement IDs were not installed exactly")
    if replaced==[int(value) for value in original_ids[0,positions].tolist()]: raise ValueError("Text replacement is identical")
    return corrupted,{"sequence_length_before":int(original_ids.shape[1]),"sequence_length_after":int(corrupted["input_ids"].shape[1]),
                      "text_positions":positions,"text_token_count":len(positions),"original_text_token_ids":[int(value) for value in original_ids[0,positions].tolist()],
                      "replacement_text_token_ids":replacement,"outside_input_ids_equal":outside_equal,
                      "attention_mask_equal":True,"replacement_contains_banned":False}


def prepare_condition_inputs(processor: Any, row: dict[str, Any], *, device: Any,
                             image_path: Path | None = None,
                             replacement_ids: Sequence[int] | None = None,
                             banned_ids: frozenset[int] = frozenset(),
                             rendered_suffix: str = "") -> tuple[str, Any, dict[str, Any], dict[str, Any]]:
    prompt,messages=messages_with_image(row,image_path)
    rendered=render_continued_assistant(processor,messages,ANSWER_PREFILL)+rendered_suffix
    inputs=prepare_multimodal_inputs(processor,messages,rendered,device=device)
    details=locate_evidence(processor,row,rendered,inputs)
    text_audit:dict[str,Any]={}
    if replacement_ids is not None:
        inputs,text_audit=apply_text_tokens(inputs,details,replacement_ids,banned_ids)
    else:
        text_audit={"text_positions":list(map(int,details["text_positions"])),
                    "original_text_token_ids":[int(value) for value in inputs.input_ids[0,details["text_positions"]].tolist()]}
    ids_tensor=inputs["input_ids"].detach().cpu().contiguous()
    details={"prompt":prompt,"prompt_hash":canonical_hash(prompt),"rendered_hash":canonical_hash(rendered),
             "image_path":str((image_path or Path(row["image_path"])).resolve()),
             "image_sha256":sha256_file(image_path or Path(row["image_path"])),**details}
    details["input_ids_sha256"]=hashlib.sha256(ids_tensor.numpy().tobytes()).hexdigest()
    return rendered,inputs,details,text_audit


__all__=["apply_text_tokens","messages_with_image","prepare_condition_inputs"]
