from __future__ import annotations

from typing import Any, Iterable

from layer_metacognition.token_positions import locate_image_pad_span
from layer_metacognition.token_spans import build_rendered_alignment


def _span(alignment: Any, rendered: str, text: str, *, start_at: int = 0, end_at: int | None = None, final: bool = False) -> list[int]:
    end_at = len(rendered) if end_at is None else end_at
    char = rendered.rfind(text, start_at, end_at) if final else rendered.find(text, start_at, end_at)
    if char < 0:
        raise ValueError(f"Could not locate exact rendered text: {text!r}")
    tokens = alignment.processed_tokens_for_char_span(char, char + len(text))
    return [tokens[0], tokens[-1] + 1]


def _positions(span: list[int]) -> list[int]:
    return list(range(int(span[0]), int(span[1])))


def locate_spans(tokenizer: Any, rendered: str, inputs: Any, row: dict[str, Any]) -> dict[str, Any]:
    ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
    mask = inputs.get("attention_mask") if isinstance(inputs, dict) else inputs.attention_mask
    alignment = build_rendered_alignment(tokenizer, rendered, ids, mask)
    processed = alignment.processed_ids
    image = locate_image_pad_span(tokenizer, processed)["span"]
    question = _span(alignment, rendered, row["question"])
    text = _span(alignment, rendered, row["text_clue"])
    if row["arm"] == "joint":
        prefix_start = len(rendered) - len(row["assistant_prefix"])
        answer_marker = "**Answer**: "
        answer_chars = prefix_start + len(answer_marker)
        answer_tokens = alignment.processed_tokens_for_char_span(answer_chars, answer_chars + len(row["raw_answer"]))
        answer = [answer_tokens[0], answer_tokens[-1] + 1]
        newline_char = answer_chars + len(row["raw_answer"])
        panl_tokens = alignment.processed_tokens_for_char_span(newline_char, newline_char + 1)
        panl = panl_tokens[0]
        source_marker = _span(
            alignment,
            rendered,
            "**Source Attribution**:",
            start_at=prefix_start,
            final=True,
        )
        sac = source_marker[1] - 1
        if sac + 1 >= len(processed):
            raise ValueError("Joint teacher stage is missing the class digit after SAC")
        expected_digit = str(row["parsed_class"])
        actual_digit = tokenizer.decode(
            [processed[sac + 1]],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if actual_digit != expected_digit:
            raise ValueError(
                f"Joint class token after SAC differs: {actual_digit!r} != {expected_digit!r}"
            )
    else:
        answer = list(row["phase1_answer_span"])
        panl = int(row["positions"]["P1_PANL"]["processed_index"])
        sac = int(row["positions"]["P1_SAC"]["processed_index"])
    if not (answer[0] <= answer[-1] <= panl < panl + 1 < sac):
        raise ValueError(f"Invalid answer/PANL/SAC order: {answer}, {panl}, {sac}")
    evidence = sorted(set(_positions(text)) | set(_positions(image)))
    all_content = sorted(set(_positions(question)) | set(evidence) | set(_positions(answer)))
    if not all(map(bool, (_positions(question), _positions(text), _positions(image), _positions(answer)))):
        raise ValueError("Required source span is empty")
    if set(_positions(text)) & set(_positions(image)):
        raise ValueError("TEXT and IMAGE spans overlap")
    return {
        "sequence_length": len(processed), "QUESTION": question, "TEXT_CLUE": text, "IMAGE": image,
        "EVIDENCE": evidence, "ANSWER": answer, "LAST_ANSWER": answer[1] - 1,
        "PANL": panl, "PANL_PLUS_1": panl + 1, "SAC": sac, "ALL_CONTENT": all_content,
        "ALL_DOWNSTREAM_OF_PANL": list(range(panl + 1, sac + 1)),
        "token_ids": processed,
    }


def edges_for_condition(spans: dict[str, Any], condition: str):
    from .masking import AttentionEdges
    evidence = spans["EVIDENCE"]
    answer = _positions(spans["ANSWER"])
    combined = sorted(set(evidence) | set(answer))
    sac, panl, plus1 = spans["SAC"], spans["PANL"], spans["PANL_PLUS_1"]
    simple = {
        "panl_to_evidence": ([panl], evidence), "panl_to_answer": ([panl], answer),
        "panl_to_evidence_answer": ([panl], combined), "panl_plus_1_to_evidence_answer": ([plus1], combined),
        "sac_to_panl": ([sac], [panl]), "sac_to_panl_plus_1": ([sac], [plus1]),
        "sac_to_evidence": ([sac], evidence), "sac_to_answer": ([sac], answer),
        "sac_to_all_content": ([sac], spans["ALL_CONTENT"]),
        "all_downstream_to_panl": (spans["ALL_DOWNSTREAM_OF_PANL"], [panl]),
        "all_downstream_to_panl_plus_1": (list(range(plus1 + 1, sac + 1)), [plus1]),
    }
    if condition in simple:
        return AttentionEdges.from_sets(*simple[condition])
    source = evidence if "evidence_answer" not in condition and "evidence" in condition else answer if "answer" in condition and "evidence_answer" not in condition else combined
    pairs = [(q, s) for s in source for q in range(s + 1, sac + 1)]
    result = AttentionEdges(tuple(sorted(set(pairs))))
    if condition.endswith("keep_panl"):
        result = result.without([panl], source)
    return result
