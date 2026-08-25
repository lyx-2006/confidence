from __future__ import annotations

from typing import Any, Sequence

from dp_SA.attention_block.masking import AttentionEdges

WINDOWS = ((10, 15), (12, 17), (14, 19))
CONDITIONS = ("C00", "C10", "C01", "C11", "CTRL")


def _answer(spans: dict[str, Any]) -> list[int]:
    return list(range(int(spans["ANSWER"][0]), int(spans["ANSWER"][1])))


def base_pairs(spans: dict[str, Any]) -> set[tuple[int, int]]:
    answer = _answer(spans); panl = int(spans["PANL"]); plus1 = int(spans["PANL_PLUS_1"]); sac = int(spans["SAC"])
    pairs = {(query, source) for source in answer for query in range(source + 1, sac + 1)}
    pairs |= {(query, panl) for query in range(panl + 1, sac + 1)}
    pairs |= {(query, plus1) for query in range(plus1 + 1, sac + 1)}
    return pairs


def restored_pairs(spans: dict[str, Any], condition: str) -> set[tuple[int, int]]:
    answer = _answer(spans); panl = int(spans["PANL"]); plus1 = int(spans["PANL_PLUS_1"]); sac = int(spans["SAC"])
    groups = {
        "C00": set(),
        "C10": {(panl, source) for source in answer},
        "C01": {(sac, panl)},
        "C11": {(panl, source) for source in answer} | {(sac, panl)},
        "CTRL": {(plus1, source) for source in answer} | {(sac, plus1)},
    }
    if condition not in groups: raise ValueError(condition)
    return groups[condition]


def edges_for_condition(spans: dict[str, Any], condition: str) -> AttentionEdges:
    base = base_pairs(spans); restored = restored_pairs(spans, condition)
    if not restored <= base: raise ValueError(f"Restored edges were not blocked in base: {restored-base}")
    result = base - restored
    answer = _answer(spans); sac = int(spans["SAC"])
    if condition in {"C11", "CTRL"} and not all((sac, source) in result for source in answer):
        raise ValueError(f"SAC→Answer must remain blocked in {condition}")
    return AttentionEdges(tuple(sorted(result)))


def validate_symmetry(spans: dict[str, Any]) -> None:
    answer = _answer(spans); panl = int(spans["PANL"]); plus1 = int(spans["PANL_PLUS_1"]); sac = int(spans["SAC"])
    c11 = restored_pairs(spans, "C11"); ctrl = restored_pairs(spans, "CTRL")
    mapped = {(plus1 if q == panl else sac, source if source not in {panl, plus1} else plus1) for q, source in c11}
    expected = {(plus1, source) for source in answer} | {(sac, plus1)}
    if mapped != expected or ctrl != expected or len(c11) != len(ctrl): raise ValueError("PANL/PANL+1 restorations are not symmetric")


def effects(m00: float, m10: float, m01: float, m11: float, mctrl: float) -> tuple[float, float, float]:
    return m11 - m10 - m01 + m00, m11 - m00, m11 - mctrl


def one_sided_sign_flip(values: Sequence[float], *, seed: int, repeats: int) -> float:
    import numpy as np
    array=np.asarray(values,dtype=float); observed=float(array.mean()); rng=np.random.default_rng(seed); exceed=0; done=0
    while done < repeats:
        size=min(2000,repeats-done); signs=rng.choice((-1.0,1.0),size=(size,len(array)))
        exceed += int(np.count_nonzero((signs*array).mean(axis=1) >= observed)); done += size
    return float((1+exceed)/(repeats+1))
