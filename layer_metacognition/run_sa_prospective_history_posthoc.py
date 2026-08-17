"""Generate read-only post-hoc sensitivities for the completed Stage-09 panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "layer_metacognition"

from .sa_formation.prospective_history_posthoc import (  # noqa: E402
    BASE_SEED,
    BOOTSTRAP_ITERATIONS,
    BRANCHES,
    contrast_specs,
    run_posthoc_analysis,
)


PANEL_DIR = "09_prospective_history_response_panel"
DEFAULT_PANEL = (
    Path(__file__).resolve().parent
    / "output"
    / "Final_v4_run"
    / "answer_basis_9"
    / "stage3_sa_computational_bridge"
    / PANEL_DIR
)
REPORT_INPUT = "report_formation_results.jsonl"
ENDPOINT_INPUT = "endpoint_manifest.json"
GATE_INPUT = "qualification_gate.json"
JSON_OUTPUT = "posthoc_sensitivity.json"
MARKDOWN_OUTPUT = "POSTHOC_SENSITIVITY.md"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--panel-dir", default=str(DEFAULT_PANEL))
    value.add_argument(
        "--overwrite",
        action="store_true",
        help="replace only this script's two post-hoc outputs if they differ",
    )
    return value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _combined_fingerprint(values: Mapping[str, str]) -> str:
    payload = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _load_jsonl_read_only(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
        records.append(value)
    return records


def _atomic_write(path: Path, text: str, *, overwrite: bool) -> str:
    if path.exists():
        current = path.read_text(encoding="utf-8")
        if current == text:
            return "unchanged"
        if not overwrite:
            raise FileExistsError(
                f"post-hoc output differs and is protected: {path}; pass --overwrite"
            )
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return "written"


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not (number == number and abs(number) != float("inf")):
        return "NA"
    return f"{number:.{digits}f}"


def _ci(summary: Mapping[str, Any], key: str = "ci95") -> str:
    interval = summary.get(key, [None, None])
    if not isinstance(interval, Sequence) or len(interval) != 2:
        return "[NA, NA]"
    return f"[{_fmt(interval[0])}, {_fmt(interval[1])}]"


def _association_cell(summary: Mapping[str, Any]) -> str:
    return f"{_fmt(summary.get('spearman'))} {_ci(summary, 'spearman_ci95')}"


def _contrast_label(identifier: str) -> str:
    family, effect = identifier.split(".", 1)
    family_names = {
        "relevant": "相关 History",
        "irrelevant": "无关 History",
        "bundle_difference": "相关−无关 bundle",
    }
    effect_names = {
        "modality": "模态（图−文）",
        "replay": "回放侧（AI−AT）",
        "interaction": "模态×回放交互",
        "history_vs_none": "History−无 History",
    }
    return f"{family_names[family]} / {effect_names[effect]}"


def _ci_excludes_zero(summary: Mapping[str, Any], key: str = "ci95") -> bool:
    interval = summary.get(key)
    return (
        isinstance(interval, Sequence)
        and len(interval) == 2
        and interval[0] is not None
        and interval[1] is not None
        and (float(interval[0]) > 0 or float(interval[1]) < 0)
    )


def render_markdown(artifact: Mapping[str, Any]) -> str:
    results = artifact["results"]
    counts = results["analysis_counts"]
    lines = [
        "# Stage 09：History 形成实验的 Post-hoc 敏感性分析",
        "",
        "> 这是一份**事后（post-hoc）、探索性、非 gate**分析。它不会改写正式 qualification gate，",
        "> 也不会把相关性解释成内部因果中介。所有正式输入均以只读方式读取。",
        "",
        "## 1. 先解释本文术语",
        "",
        "- **A\\***：Phase 0 在无 History 条件下冻结的最终答案；后续分层不会使用 History 之后的新答案。",
        "- **endpoint side（终点答案侧）**：A\\* 等于图像单模态答案时记为 `image`，等于文本单模态答案时记为 `text`。本样本为 image=29、text=11。",
        "- **A（attribution readout）**：PANL L18 隐状态经冻结 readout 得到的连续归因预测；数值越大越偏图像来源。",
        "- **V（verbal SA）**：Common-9 verbal Source Attribution 的语义图像侧分数；越大越偏图像来源。",
        "- **AT / AI**：History 中回放文本侧答案 / 图像侧答案。它们不是“是否等于 A\\*”的同义词；A\\* 在不同 item 上可能位于不同侧。",
        "- **modality contrast（模态对比）**：图像 History 减文本 History。",
        "- **replay contrast（回放侧对比）**：AI 减 AT。",
        "- **interaction（交互）**：`image_AI - image_AT - text_AI + text_AT`。",
        "- **history-vs-none**：四个 History cell 的平均值减无 History。",
        "- **bundle difference**：完整的相关 History bundle 对比减无关 History bundle 对比；它不是纯粹的 relevance effect。",
        "- **within-fold bootstrap**：在每个冻结 fold 内重抽 recipient item，共 1000 次。endpoint-side 差异还固定 fold×side 的样本数。",
        "- **token count**：Common-9 测量分支的完整 tokenized input 长度，不是只数 History 文本中的词。",
        "",
        "## 2. 分析边界与数据审计",
        "",
        f"- 输入：`{artifact['input_artifacts']['report_formation_results.jsonl']['path']}`。",
        f"- 完整分支：{counts['report_branch_n']}（{counts['recipient_item_n']} item × 9 branch）。",
        f"- endpoint side：image={counts['endpoint_side_counts']['image']}，text={counts['endpoint_side_counts']['text']}。",
        f"- exact ordered-pair：{counts['exact_ordered_pair_item_n']}/40；fallback：{counts['fallback_item_n']}/40。",
        f"- Bootstrap：{artifact['bootstrap']['iterations']} 次；base seed={artifact['bootstrap']['base_seed']}；每项实际 seed 已写入 JSON。",
        "- 12 组对比均为探索性敏感性检查，没有多重比较校正，也不参与任何 gate。",
        "- 推断单位始终是 recipient item；同一 item 的多个 History cell 先做均值或差分。",
        "",
        "## 3. 各 branch 的 token 数",
        "",
        "| Branch | n | 完整输入 token：均值±SD | 范围 | 相对无 History 的配对增量：均值±SD | 增量范围 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    token_stats = results["branch_token_count_statistics"]
    for branch in BRANCHES:
        total = token_stats[branch]["total_input_token_count"]
        increment = token_stats[branch]["paired_increment_vs_no_history"]
        lines.append(
            "| "
            + " | ".join(
                (
                    branch,
                    str(total["n"]),
                    f"{_fmt(total['mean'], 1)} ± {_fmt(total['sd'], 1)}",
                    f"{_fmt(total['min'], 0)}–{_fmt(total['max'], 0)}",
                    f"{_fmt(increment['mean'], 1)} ± {_fmt(increment['sd'], 1)}",
                    f"{_fmt(increment['min'], 0)}–{_fmt(increment['max'], 0)}",
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "解释：图像 History 分支明显更长，因此必须检查“图像模态效应”是否只是 token 长度的影子。相关检查只能发现共变，不能证明或排除长度因果。",
            "",
            "## 4. token-count contrast 与 A/V contrast 的相关",
            "",
            "表中为 Spearman \\(r_s\\) 及 95% bootstrap CI；若 token contrast 没有跨 item 变异，相关记为 NA。",
            "",
            "| 12 组 contrast | token contrast 均值±SD | token↔A：r_s [CI] | token↔V：r_s [CI] |",
            "|---|---:|---:|---:|",
        ]
    )
    token_alignment = results["token_count_contrast_alignment"]
    for spec in contrast_specs():
        entry = token_alignment[spec["id"]]
        token = entry["token_contrast"]
        lines.append(
            f"| {_contrast_label(spec['id'])} | {_fmt(token['mean'], 2)} ± {_fmt(token['sd'], 2)} "
            f"| {_association_cell(entry['token_vs_A_prediction'])} "
            f"| {_association_cell(entry['token_vs_V'])} |"
        )

    lines.extend(
        [
            "",
            "## 5. 按冻结 endpoint side 分层的 12 组 contrast",
            "",
            "每格为该 stratum 的均值和 95% CI。image stratum n=29，text stratum n=11。",
            "",
            "| Contrast | A / endpoint=image | A / endpoint=text | V / endpoint=image | V / endpoint=text |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    stratified = results["endpoint_side_stratified_contrasts"]
    for spec in contrast_specs():
        entry = stratified[spec["id"]]["strata"]
        ai = entry["image"]["outcomes"]["A_prediction"]
        at = entry["text"]["outcomes"]["A_prediction"]
        vi = entry["image"]["outcomes"]["V"]
        vt = entry["text"]["outcomes"]["V"]
        lines.append(
            f"| {_contrast_label(spec['id'])} | {_fmt(ai['mean'])} {_ci(ai)} "
            f"| {_fmt(at['mean'])} {_ci(at)} | {_fmt(vi['mean'])} {_ci(vi)} "
            f"| {_fmt(vt['mean'])} {_ci(vt)} |"
        )

    lines.extend(
        [
            "",
            "## 6. endpoint=image 减 endpoint=text 的差异",
            "",
            "这里问的是“同一个 History contrast 在两类冻结终点中是否不同”，不是 History 模态的 image-minus-text。bootstrap 在 fold×endpoint-side 内重抽，因此固定 29/11 组成。",
            "",
            "| Contrast | A 的 side difference [CI] | V 的 side difference [CI] |",
            "|---|---:|---:|",
        ]
    )
    side_differences = results["endpoint_image_minus_text_differences"]
    for spec in contrast_specs():
        entry = side_differences[spec["id"]]["outcomes"]
        lines.append(
            f"| {_contrast_label(spec['id'])} | {_fmt(entry['A_prediction']['estimate'])} {_ci(entry['A_prediction'])} "
            f"| {_fmt(entry['V']['estimate'])} {_ci(entry['V'])} |"
        )

    lines.extend(
        [
            "",
            "## 7. A change 与 V change 是否逐 item 对齐",
            "",
            "这不是比较两个均值是否同号，而是问：某个 item 的 A 改变量越大时，它的 V 改变量是否也越大。",
            "",
            "| Contrast | Pearson r | Spearman r_s [95% CI] | 同号率 |",
            "|---|---:|---:|---:|",
        ]
    )
    av = results["A_V_change_alignment"]
    for spec in contrast_specs():
        entry = av[spec["id"]]["A_vs_V"]
        lines.append(
            f"| {_contrast_label(spec['id'])} | {_fmt(entry['pearson'])} "
            f"| {_association_cell(entry)} | {_fmt(entry['sign_agreement'])} |"
        )

    lines.extend(
        [
            "",
            "## 8. 把 replay 重新解释为“回放答案是否等于 A*”",
            "",
            "这里重新计算 `normalized(replayed_answer) == normalized(answer_star)`；**没有使用**原字段 `answer_identity_matches_target`，因为后者表示“等于目标同侧答案”，不是“等于 A*”。",
            "",
            "每个 eligible item 内先分别平均 match cells 与 mismatch cells，再计算 match−mismatch。因此 cell 数只用于透明审计，不被当作独立 n。",
            "",
            "| Scope | 全部 History cells：match / mismatch | 有 match 的 item | paired item n | A：match−mismatch [CI] | V：match−mismatch [CI] |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    match = results["answer_match_reinterpretation"]
    scope_names = {
        "relevant_history": "仅相关 History",
        "irrelevant_history": "仅无关 History",
    }
    for scope in scope_names:
        entry = match[scope]
        a = entry["outcomes"]["A_prediction"]["matched_minus_mismatched"]
        v = entry["outcomes"]["V"]["matched_minus_mismatched"]
        lines.append(
            f"| {scope_names[scope]} | {entry['matched_history_cell_n']} / {entry['mismatched_history_cell_n']} "
            f"| {entry['items_with_at_least_one_match_n']}/40 | {entry['paired_eligible_item_n']} | {_fmt(a['mean'])} {_ci(a)} "
            f"| {_fmt(v['mean'])} {_ci(v)} |"
        )

    lines.extend(
        [
            "",
            "为区分“相对 no-History 的总位移”和“match 对 mismatch 的差”，再列出两个组成部分：",
            "",
            "| Scope | A：match−none | A：mismatch−none | V：match−none | V：mismatch−none |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for scope in scope_names:
        entry = match[scope]["outcomes"]
        lines.append(
            f"| {scope_names[scope]} | "
            f"{_fmt(entry['A_prediction']['matched_minus_no_history']['mean'])} {_ci(entry['A_prediction']['matched_minus_no_history'])} | "
            f"{_fmt(entry['A_prediction']['mismatched_minus_no_history']['mean'])} {_ci(entry['A_prediction']['mismatched_minus_no_history'])} | "
            f"{_fmt(entry['V']['matched_minus_no_history']['mean'])} {_ci(entry['V']['matched_minus_no_history'])} | "
            f"{_fmt(entry['V']['mismatched_minus_no_history']['mean'])} {_ci(entry['V']['mismatched_minus_no_history'])} |"
        )

    lines.extend(
        [
            "",
            "覆盖解释：Relevant 中每个 item 都有一个回放侧与 A* 相符、另一个不符，故 40/40 可配对；Irrelevant 中只有 12/40 至少有一个 A* match，另外 28/40 的 AT 与 AI 都不等于 A*。Irrelevant 的 n=12 结果只代表这 12 个可配对 recipient。",
            "",
            "没有报告把 Relevant 与 Irrelevant 混在一起的 `all_history` match contrast：在缺少 Irrelevant match 的 item 中，match mean 会只来自 Relevant，而 mismatch mean 同时含 Relevant 与 Irrelevant，导致两个组的成分不对称。",
        ]
    )

    side_nonzero_a = [
        _contrast_label(spec["id"])
        for spec in contrast_specs()
        if _ci_excludes_zero(
            side_differences[spec["id"]]["outcomes"]["A_prediction"]
        )
    ]
    side_nonzero_v = [
        _contrast_label(spec["id"])
        for spec in contrast_specs()
        if _ci_excludes_zero(side_differences[spec["id"]]["outcomes"]["V"])
    ]
    aligned = [
        _contrast_label(spec["id"])
        for spec in contrast_specs()
        if _ci_excludes_zero(
            av[spec["id"]]["A_vs_V"], key="spearman_ci95"
        )
    ]
    lines.extend(
        [
            "",
            "## 9. 全面但克制的结论",
            "",
            "1. **长度是一个真实的设计差异。** 图像 History 需要编码图像，因此完整输入远长于文本 History；这使 token-count sensitivity 必不可少。表 4 给出逐 item 的共变证据，但相关为零或不显著都不能单独证明“长度没有因果作用”。",
            "2. **endpoint side 存在明显不平衡。** image=29、text=11，因此总体 replay 平均会更接近 image endpoint 的模式。表 5–6 把这个混合显式拆开，而不是把总体均值当作所有 endpoint 都成立。",
            "3. **side heterogeneity 是探索性证据。** A 的 image−text side difference CI 不跨 0 的项目："
            + ("；".join(side_nonzero_a) if side_nonzero_a else "无")
            + "。V 对应项目："
            + ("；".join(side_nonzero_v) if side_nonzero_v else "无")
            + "。这些是 12 组未校正 post-hoc 比较，不能作为新 gate。",
            "4. **A 与 V 的均值一起移动，不等于逐 item 同步。** A–V Spearman CI 不跨 0 的项目："
            + ("；".join(aligned) if aligned else "无")
            + "。其余项目不能据此断言 A change 会传递成 verbal SA change。",
            "5. **answer-match 重编码更贴近问题本身。** AT/AI 是来源侧编码，而 match/mismatch 是相对每个 item 的 A* 编码；两者在 endpoint=text 时符号相反。表 8 直接显示回放与 A* 相同答案相对于不同答案的配对差异。",
            "6. **不能把 A 当 causal mediator。** 本分析没有对 A 做 intervention、clamp 或 transplant；A↔V 的 change alignment 仍是关联。允许的因果语言只限于完整外部 History prompt bundle 对下游 A/V 的配对影响。",
            "",
            "## 10. 必须保留的 caveat",
            "",
            "- 只有 6/40 item 的 irrelevant donor 与 target 具有 exact ordered text/image answer pair；34/40 使用 fallback。",
            "- 因此 relevant−irrelevant 是两个完整 History bundle 的差，不是纯 relevance 操作：历史 item、答案 identity、target repetition 和 tokenization 可能同时变化。",
            "- answer-match 分组同样是事后重编码，并与 relevance、endpoint side、donor identity 有结构性联系。",
            "- endpoint=text 只有 11 个 item；其 CI 会更不稳定。",
            "- 12 组 contrast × 多个 outcome 未做多重比较校正；CI 只用于敏感性描述。",
            "- 本文不更改 `qualification_gate.json`，不回写正式 `summary.json/results.jsonl`，也不产生新的 formal claim。",
            "",
            "## 11. 可复现性",
            "",
            f"- 输入组合 fingerprint：`{artifact['input_fingerprint']}`",
            f"- 代码组合 fingerprint：`{artifact['code_fingerprint']}`",
            f"- JSON 输出：`{JSON_OUTPUT}`",
            f"- Bootstrap iterations：{artifact['bootstrap']['iterations']}；base seed：{artifact['bootstrap']['base_seed']}。",
            "- 每一个 bootstrap 的实际 seed、有效 replicate 数和 resampling scheme 均保存在 JSON。",
            "",
        ]
    )
    return "\n".join(lines)


def build_artifact(panel: Path) -> dict[str, Any]:
    if panel.name != PANEL_DIR:
        raise ValueError(f"expected a {PANEL_DIR} directory, got {panel}")
    required = {
        REPORT_INPUT: panel / REPORT_INPUT,
        ENDPOINT_INPUT: panel / ENDPOINT_INPUT,
        GATE_INPUT: panel / GATE_INPUT,
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing Stage-09 post-hoc inputs: {missing}")
    input_shas_before = {name: sha256_file(path) for name, path in required.items()}
    report_rows = _load_jsonl_read_only(required[REPORT_INPUT])
    endpoint = _load_json(required[ENDPOINT_INPUT])
    gate = _load_json(required[GATE_INPUT])
    endpoint_rows = endpoint.get("rows")
    if not isinstance(endpoint_rows, list):
        raise ValueError("endpoint_manifest.json lacks rows")
    authorizations = gate.get("authorizations")
    if not isinstance(authorizations, Mapping):
        raise ValueError("qualification_gate.json lacks authorizations")
    if authorizations.get("report_formation_history") is not True:
        raise ValueError("formal report-formation History track was not authorized")
    results = run_posthoc_analysis(
        report_rows,
        endpoint_rows,
        iterations=BOOTSTRAP_ITERATIONS,
        require_frozen_shape=True,
    )
    input_shas_after = {name: sha256_file(path) for name, path in required.items()}
    if input_shas_after != input_shas_before:
        raise RuntimeError("a formal Stage-09 input changed during read-only analysis")

    code_paths = {
        "prospective_history_posthoc.py": Path(
            sys.modules[run_posthoc_analysis.__module__].__file__  # type: ignore[union-attr]
        ).resolve(),
        "run_sa_prospective_history_posthoc.py": Path(__file__).resolve(),
    }
    code_shas = {name: sha256_file(path) for name, path in code_paths.items()}
    input_artifacts = {
        name: {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": input_shas_before[name],
            "read_only_sha_verified_after_analysis": True,
        }
        for name, path in required.items()
    }
    return {
        "format_version": 1,
        "title": "Stage-09 prospective History response post-hoc sensitivity",
        "analysis_status": "complete",
        "analysis_class": "posthoc_sensitivity_non_gating",
        "formal_gate_modified": False,
        "formal_results_modified": False,
        "qualification_gate_snapshot": {
            "behavior_readout_history": authorizations.get("behavior_readout_history"),
            "report_formation_history": authorizations.get("report_formation_history"),
            "full_four_layer": authorizations.get("full_four_layer"),
            "source_sha256": input_shas_before[GATE_INPUT],
        },
        "input_artifacts": input_artifacts,
        "input_fingerprint": _combined_fingerprint(input_shas_before),
        "analysis_code": {
            name: {"path": str(code_paths[name]), "sha256": digest}
            for name, digest in code_shas.items()
        },
        "code_fingerprint": _combined_fingerprint(code_shas),
        "bootstrap": {
            "iterations": BOOTSTRAP_ITERATIONS,
            "base_seed": BASE_SEED,
            "seed_scheme": "fixed deterministic offsets by section/contrast/outcome",
            "primary_resampling": "recipient-item-within-fixed-fold",
            "endpoint_side_difference_resampling": (
                "recipient-item-within-fixed-fold-and-endpoint_side"
            ),
        },
        "claim_boundary": {
            "posthoc": True,
            "gate_bearing": False,
            "multiplicity_adjustment": "none; exploratory sensitivity only",
            "external_history_prompt_bundle_causal_contrast": True,
            "pure_relevance_claim_allowed": False,
            "A_as_causal_mediator_claim_allowed": False,
            "activation_intervention_run": False,
            "exact_pair_caveat": (
                "Only 6/40 items have exact ordered-pair History donors; the 34/40 "
                "fallback rows confound relevance bundle, donor item, answer identity, "
                "target repetition, and tokenization."
            ),
        },
        "results": results,
    }


def main() -> None:
    args = parser().parse_args()
    panel = Path(args.panel_dir).resolve()
    artifact = build_artifact(panel)
    json_text = json.dumps(artifact, ensure_ascii=False, indent=2) + "\n"
    markdown_text = render_markdown(artifact)
    json_status = _atomic_write(panel / JSON_OUTPUT, json_text, overwrite=args.overwrite)
    markdown_status = _atomic_write(
        panel / MARKDOWN_OUTPUT,
        markdown_text,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "analysis_class": artifact["analysis_class"],
                "json": {"path": str(panel / JSON_OUTPUT), "status": json_status},
                "markdown": {
                    "path": str(panel / MARKDOWN_OUTPUT),
                    "status": markdown_status,
                },
                "input_fingerprint": artifact["input_fingerprint"],
                "code_fingerprint": artifact["code_fingerprint"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
