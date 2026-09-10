from __future__ import annotations

import io
import string
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from dp_SA.prompts import ANSWER_PREFILL, PHASE0_TEMPLATE

from .config import MIN_TOKEN_FREQUENCY, NOISE_MEAN, NOISE_STD, SEED
from .io_utils import atomic_bytes, canonical_json, sha256_bytes, sha256_file


def stable_seed(case_id: str, replicate: int, modality: str, base_seed: int = SEED) -> dict[str, Any]:
    import hashlib
    payload=[int(base_seed),str(case_id),int(replicate),str(modality)]
    digest=hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    return {"payload":payload,"sha256":digest,"uint64":int.from_bytes(bytes.fromhex(digest[:16]),"big",signed=False)}


def gaussian_png_bytes(size: tuple[int,int], seed: int) -> tuple[bytes,np.ndarray]:
    width,height=map(int,size)
    if width<=0 or height<=0: raise ValueError("Image dimensions must be positive")
    rng=np.random.default_rng(int(seed))
    pixels=np.clip(rng.normal(NOISE_MEAN,NOISE_STD,size=(height,width,3)),0,255).astype(np.uint8)
    stream=io.BytesIO()
    Image.fromarray(pixels,mode="RGB").save(stream,format="PNG",compress_level=6,optimize=False)
    return stream.getvalue(),pixels


def ensure_gaussian_image(original_path: Path, destination: Path, case_id: str, replicate: int) -> dict[str,Any]:
    with Image.open(original_path) as image:
        size=tuple(map(int,image.size)); source_mode=str(image.mode)
    seed=stable_seed(case_id,replicate,"image")
    payload,pixels=gaussian_png_bytes(size,int(seed["uint64"]))
    expected=sha256_bytes(payload)
    if destination.exists():
        if sha256_file(destination)!=expected: raise ValueError(f"Existing Gaussian PNG mismatch: {destination}")
    else: atomic_bytes(destination,payload)
    return {"seed":seed,"path":str(destination.resolve()),"sha256":expected,"width":size[0],"height":size[1],
            "source_mode":source_mode,"target_mode":"RGB","pixel_mean":float(pixels.mean()),
            "pixel_std":float(pixels.std()),"distribution_mean":NOISE_MEAN,"distribution_std":NOISE_STD}


@dataclass(frozen=True)
class TokenPool:
    token_ids: np.ndarray
    counts: np.ndarray
    probabilities: np.ndarray
    banned_ids: frozenset[int]
    audit: dict[str,Any]


def _encoded_ids(tokenizer: Any, texts: Sequence[str]) -> set[int]:
    output:set[int]=set()
    for text in texts: output.update(map(int,tokenizer.encode(text,add_special_tokens=False)))
    return output


def build_token_pool(tokenizer: Any, train_rows: Sequence[dict[str,Any]]) -> TokenPool:
    frequencies:Counter[int]=Counter()
    for row in train_rows: frequencies.update(map(int,tokenizer.encode(str(row["text_clue"]),add_special_tokens=False)))
    reasons:dict[str,set[int]]=defaultdict(set)
    reasons["special"].update(map(int,tokenizer.all_special_ids))
    for token,token_id in tokenizer.get_vocab().items():
        if "<|" in token or "|>" in token: reasons["control_fragment"].add(int(token_id))
    colors=sorted({str(answer) for row in train_rows for answer in row["answer_classes"]})
    reasons["color_answer"].update(_encoded_ids(tokenizer,[value for color in colors for value in (color," "+color)]))
    formatter=string.Formatter()
    literals=[literal for literal,_field,_format,_conversion in formatter.parse(PHASE0_TEMPLATE) if literal]
    structural=[*literals,ANSWER_PREFILL,"Question:","Text clue:","Answer:","system","user","assistant","<your answer>"]
    reasons["prompt_structure"].update(_encoded_ids(tokenizer,structural))
    banned=set().union(*reasons.values())
    exclusions=Counter()
    candidates=[]
    for token_id,count in frequencies.items():
        decoded=tokenizer.decode([token_id],skip_special_tokens=False,clean_up_tokenization_spaces=False)
        why=[]
        if count<MIN_TOKEN_FREQUENCY: why.append("frequency_below_minimum")
        if token_id in banned: why.append("banned_id")
        if not decoded or decoded.isspace(): why.append("empty_or_whitespace")
        if decoded and not decoded.isprintable(): why.append("non_printable")
        if "<|" in decoded or "|>" in decoded: why.append("decoded_control_fragment")
        if why:
            exclusions.update(why); continue
        candidates.append((int(token_id),int(count),decoded))
    candidates.sort(key=lambda value:value[0])
    if not candidates: raise ValueError("Random text token pool is empty")
    ids=np.asarray([value[0] for value in candidates],dtype=np.int64)
    counts=np.asarray([value[1] for value in candidates],dtype=np.float64)
    probabilities=counts/counts.sum()
    audit={"status":"passed","minimum_frequency":MIN_TOKEN_FREQUENCY,"train_record_count":len(train_rows),
           "raw_unique_token_count":len(frequencies),"candidate_count":len(candidates),"candidate_frequency_mass":int(counts.sum()),
           "exclusion_counts":dict(exclusions),"banned_reason_counts":{name:len(values) for name,values in reasons.items()},
           "colors":colors,"candidates":[{"token_id":i,"decoded":d,"frequency":c,"sampling_probability":float(c/counts.sum())} for i,c,d in candidates]}
    return TokenPool(ids,counts,probabilities,frozenset(map(int,banned)),audit)


def sample_text_tokens(pool:TokenPool,tokenizer:Any,original_ids:Sequence[int],case_id:str,replicate:int)->tuple[list[int],dict[str,Any]]:
    excluded=set(map(int,original_ids)); mask=np.asarray([int(value) not in excluded for value in pool.token_ids],dtype=bool)
    ids=pool.token_ids[mask]; weights=pool.counts[mask]
    if not len(ids): raise ValueError(f"No case-specific random tokens: {case_id}")
    probabilities=weights/weights.sum(); seed=stable_seed(case_id,replicate,"text")
    rng=np.random.default_rng(int(seed["uint64"]))
    sampled=list(map(int,rng.choice(ids,size=len(original_ids),replace=True,p=probabilities).tolist()))
    if sampled==list(map(int,original_ids)): raise ValueError("Random replacement equals original clue")
    if set(sampled)&excluded: raise ValueError("Replacement contains a case-original clue token")
    return sampled,{"seed":seed,"token_ids":sampled,"decoded_tokens":[tokenizer.decode([value],skip_special_tokens=False,clean_up_tokenization_spaces=False) for value in sampled],
                    "length":len(sampled),"available_token_count":int(len(ids)),"available_frequency_mass":int(weights.sum())}


__all__=["TokenPool","build_token_pool","ensure_gaussian_image","gaussian_png_bytes","sample_text_tokens","stable_seed"]
