"""Read-only planning for the Stage-09 v2 prospective History panel.

The module freezes *which exact contexts* may be used before any new History
outcome is observed.  It deliberately does not load a model, run a forward, or
write an artifact.  The runner can serialise the returned manifests verbatim.

The candidate definition is intentionally stricter than merely choosing a new
prior/condition for an old item:

* targets come from the completed Truth-01 conflict pool;
* their item belongs to the 76 completed Bridge-01 confirmatory items;
* the exact case was never a Bridge-01/02 target or donor; and
* the item has never been a target of a prior History experiment.

On the frozen repository this produces 395 exact contexts from 67 items.  The
primary cohort contains 40 item-unique contexts (eight per fold).  Every other
candidate is retained as an ordered reserve, and all primary/reserve contexts
receive donor assignments before a new outcome is available.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from confidence_test.dataset_utils import EvaluationCase, load_evaluation_cases
from confidence_test.inference_extension import ASSISTANT_ANSWER_PREFILL
from confidence_test.prompt_utils import STAGE1_TEXT_ANSWER_PROMPT
from layer_metacognition.hidden_state_store import load_jsonl
from layer_metacognition.probe.prompts import IMAGE_ONLY_ANSWER_PROMPT

from .core import SAFormationArtifacts, canonical_message_hash, stable_hash
from .confirmatory_attribution_panel import (
    build_joint_messages as build_frozen_joint_messages,
    panel_protocols,
)
from .reliance_measurement import (
    NO_TEXT_PLACEHOLDER,
    build_answer_only_messages,
    contains_verbal_sa_request,
)
from .runtime import assistant_message, image_content, text_content


PANEL_VERSION = 2
SEED = 42
PRIMARY_N = 40
PRIMARY_PER_FOLD = 8
PRIMARY_PER_FOLD_DIFFICULTY = 4
EXPECTED_CANDIDATE_N = 395
EXPECTED_CANDIDATE_ITEM_N = 67
EXPECTED_FOLD_CONTEXT_COUNTS = {0: 87, 1: 98, 2: 64, 3: 81, 4: 65}

SELECTION_SALT = "stage09-v2|42|"
SOURCE_RESULTS = Path("stage3_sa_truth_audit/01_counterfactual_source_use/results.jsonl")
BRIDGE01_CONFIRMATORY_RESULTS = Path(
    "stage3_sa_computational_bridge/01_actual_source_reliance/confirmatory_results.jsonl"
)
USED_EXACT_MANIFESTS = (
    Path("stage3_sa_computational_bridge/01_actual_source_reliance/development_cohort_manifest.json"),
    Path("stage3_sa_computational_bridge/01_actual_source_reliance/confirmatory_cohort_manifest.json"),
    Path("stage3_sa_computational_bridge/02_donor_replication_extension/development_cohort_manifest.json"),
    Path("stage3_sa_computational_bridge/02_donor_replication_extension/confirmatory_cohort_manifest.json"),
)
HISTORY_JSON_MANIFESTS = (
    Path("stage3_sa_formation/02_history/cohort_manifest.json"),
    Path("stage3_sa_formation_followup/02_history_exact_factorial/cohort_manifest.json"),
)
HISTORY_JSONL_RESULTS = (
    Path("stage3_sa_mechanism/01_old_direction_natural_audit/history_results.jsonl"),
    Path("stage3_sa_mechanism/03_relevant_irrelevant_history/results.jsonl"),
    Path("stage3_sa_second_order/01_history_behavior_dissociation/results.jsonl"),
    Path("stage3_sa_truth_audit/03_history_factorial_reanalysis/results.jsonl"),
    Path("stage3_sa_truth_audit/09_history_conditioned_fixed_answer_deletion/results.jsonl"),
)

USED_EXACT_ROLES = (
    "case_id",
    "donor1_case_id",
    "donor2_case_id",
    "donor3_case_id",
    "donor4_case_id",
)
HISTORY_TARGET_ROLES = ("case_id", "base_case_id", "target_case_id")

EVIDENCE_CONDITIONS = (
    "full",
    "no_text",
    "no_image",
    "replace_text_d5",
    "replace_image_d5",
    "replace_text_d6",
    "replace_image_d6",
)


def _history_branch_name(relevance: str, modality: str, replay_side: str) -> str:
    # Keep the established factorial vocabulary used by the analysis layer:
    # AT means replay A_T (the text-only answer), AI means replay A_I.
    replay_label = {"text": "at", "image": "ai"}[replay_side]
    return f"{relevance}_{modality}_{replay_label}"


HISTORY_BRANCH_FACTORS: dict[str, dict[str, str]] = {
    _history_branch_name(relevance, modality, replay_side): {
        "relevance": relevance,
        "modality": modality,
        "replay_side": replay_side,
    }
    for relevance in ("relevant", "irrelevant")
    for modality in ("text", "image")
    for replay_side in ("text", "image")
}
BRANCHES = ("no_history", *HISTORY_BRANCH_FACTORS)

# Hard items are the only stratum in this unused-context pool containing both
# legacy Text-side and Image-side endpoints.  Fold 2 has only one eligible
# Text-side hard context, so its frozen allocation is 1/3 rather than 2/2.
HARD_SIDE_QUOTAS: dict[int, dict[str, int]] = {
    0: {"text": 2, "image": 2},
    1: {"text": 2, "image": 2},
    2: {"text": 1, "image": 3},
    3: {"text": 2, "image": 2},
    4: {"text": 2, "image": 2},
}


@dataclass(frozen=True)
class CandidateInventory:
    """Frozen exact-context universe and its auditable exclusions."""

    rows: tuple[dict[str, Any], ...]
    source_rows: tuple[dict[str, Any], ...]
    item_universe: frozenset[str]
    used_exact_case_ids: frozenset[str]
    prior_history_item_ids: frozenset[str]
    audit: dict[str, Any]


@dataclass(frozen=True)
class ProspectiveHistoryResponsePlan:
    """Complete CPU plan for primary, reserve, donor, and message cells."""

    artifacts: SAFormationArtifacts
    cases: dict[tuple[str, int], EvaluationCase]
    inventory: CandidateInventory
    primary_rows: tuple[dict[str, Any], ...]
    reserve_rows: tuple[dict[str, Any], ...]
    all_rows: tuple[dict[str, Any], ...]
    donor_diagnostics: dict[str, Any]

    @property
    def rows(self) -> tuple[dict[str, Any], ...]:
        """Compatibility alias: formal rows are the frozen primary cohort."""

        return self.primary_rows


def _require_file(root: Path, relative: Path) -> Path:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Required Stage-09 planning input is missing: {path}")
    return path


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_rows(path: Path) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    # Every input to cohort construction is an older, read-only artifact.  A
    # malformed tail is therefore a hard provenance failure, never something
    # Stage 09 is allowed to repair in place.
    for index, row in enumerate(load_jsonl(path, repair_trailing=False)):
        key = str(row.get("intervention_key") or row.get("case_id") or index)
        latest[key] = dict(row)
    return list(latest.values())


def _case_item_id(case_id: str) -> str:
    item_id, separator, _ = str(case_id).partition("__")
    if not separator or not item_id:
        raise ValueError(f"Cannot recover item id from case id: {case_id!r}")
    return item_id


def _selection_digest(case_id: str) -> str:
    return hashlib.sha256((SELECTION_SALT + str(case_id)).encode("utf-8")).hexdigest()


def _side_name(row: Mapping[str, Any]) -> str:
    return "image" if int(row["final_image"]) == 1 else "text"


def replacement_stratum(row: Mapping[str, Any]) -> str:
    """Return the frozen structural replacement cell for one target.

    Easy rows were selected without a legacy-side quota, so conditioning their
    reserve order on that historical outcome would add an unregistered rule.
    Hard rows retain the fold-specific legacy-side quotas used at selection.
    """

    base = f"fold={int(row['fold'])}|difficulty={str(row['difficulty'])}"
    if str(row["difficulty"]) == "hard":
        return f"{base}|legacy_side={_side_name(row)}"
    return base


def _eligible_source_row(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("status") == "completed"
        and row.get("text_answer")
        and row.get("image_answer")
        and str(row["text_answer"]) != str(row["image_answer"])
        and str(row.get("condition", "")).startswith("conflict_")
    )


def _used_exact_cases(root: Path) -> tuple[set[str], dict[str, Any]]:
    used: set[str] = set()
    file_audit: list[dict[str, Any]] = []
    for relative in USED_EXACT_MANIFESTS:
        path = _require_file(root, relative)
        payload = _load_json(path)
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise ValueError(f"Cohort manifest has no rows list: {path}")
        before = len(used)
        role_counts = Counter()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"Non-object cohort row in {path}")
            for role in USED_EXACT_ROLES:
                value = row.get(role)
                if value:
                    used.add(str(value))
                    role_counts[role] += 1
        file_audit.append(
            {
                "path": str(relative),
                "row_n": len(rows),
                "new_unique_exact_case_n": len(used) - before,
                "role_counts": dict(sorted(role_counts.items())),
            }
        )
    return used, {"unique_exact_case_n": len(used), "files": file_audit}


def _manifest_history_case_ids(payload: Mapping[str, Any]) -> set[str]:
    case_ids: set[str] = set()
    listed = payload.get("case_ids", [])
    if isinstance(listed, list):
        case_ids.update(str(value) for value in listed if value)
    for group_name in ("rows", "targets"):
        group = payload.get(group_name, [])
        if not isinstance(group, list):
            continue
        for entry in group:
            if isinstance(entry, str):
                case_ids.add(entry)
            elif isinstance(entry, Mapping):
                for role in HISTORY_TARGET_ROLES:
                    if entry.get(role):
                        case_ids.add(str(entry[role]))
    return case_ids


def _prior_history_items(root: Path) -> tuple[set[str], dict[str, Any]]:
    items: set[str] = set()
    file_audit: list[dict[str, Any]] = []
    for relative in HISTORY_JSON_MANIFESTS:
        path = _require_file(root, relative)
        case_ids = _manifest_history_case_ids(_load_json(path))
        path_items = {_case_item_id(case_id) for case_id in case_ids}
        items.update(path_items)
        file_audit.append(
            {
                "path": str(relative),
                "format": "json_manifest",
                "target_case_n": len(case_ids),
                "target_item_n": len(path_items),
            }
        )
    for relative in HISTORY_JSONL_RESULTS:
        path = _require_file(root, relative)
        case_ids: set[str] = set()
        for row in _latest_rows(path):
            for role in HISTORY_TARGET_ROLES:
                if row.get(role):
                    case_ids.add(str(row[role]))
        path_items = {_case_item_id(case_id) for case_id in case_ids}
        items.update(path_items)
        file_audit.append(
            {
                "path": str(relative),
                "format": "jsonl_results",
                "target_case_n": len(case_ids),
                "target_item_n": len(path_items),
            }
        )
    return items, {"unique_target_item_n": len(items), "files": file_audit}


def build_candidate_inventory(
    artifacts: SAFormationArtifacts | str | Path,
) -> CandidateInventory:
    """Reproduce the frozen 395-context / 67-item target universe."""

    if not isinstance(artifacts, SAFormationArtifacts):
        artifacts = SAFormationArtifacts.discover(artifacts)
    root = artifacts.experiment_dir
    source_path = _require_file(root, SOURCE_RESULTS)
    source_rows = [row for row in _latest_rows(source_path) if _eligible_source_row(row)]
    if len(source_rows) != 1300:
        raise ValueError(f"Eligible Truth-01 source pool drifted: {len(source_rows)} != 1300")

    confirmatory_path = _require_file(root, BRIDGE01_CONFIRMATORY_RESULTS)
    confirmatory = [
        row
        for row in _latest_rows(confirmatory_path)
        if row.get("status") == "completed"
    ]
    item_universe = {str(row["item_id"]) for row in confirmatory}
    if len(item_universe) != 76:
        raise ValueError(f"Bridge-01 completed item universe drifted: {len(item_universe)} != 76")

    used_exact, exact_audit = _used_exact_cases(root)
    history_items, history_audit = _prior_history_items(root)
    candidates: list[dict[str, Any]] = []
    for source in source_rows:
        case_id = str(source["case_id"])
        item_id = str(source["item_id"])
        if item_id not in item_universe:
            continue
        if case_id in used_exact:
            continue
        if item_id in history_items:
            continue
        candidate = {
            "case_id": case_id,
            "item_id": item_id,
            "prior_index": int(source["prior_index"]),
            "condition": str(source["condition"]),
            "difficulty": str(source["difficulty"]),
            "fold": int(source["fold"]),
            "text_answer": str(source["text_answer"]),
            "image_answer": str(source["image_answer"]),
            "legacy_final_image": int(source["final_image"]),
            # Keep the generic field for compatibility with established
            # measurement helpers, while naming its historical status above.
            "final_image": int(source["final_image"]),
            "prior_strength": float(source.get("prior_strength", 0.0)),
            "selection_sha256": _selection_digest(case_id),
            "source_row_fingerprint": stable_hash(source),
        }
        candidates.append(candidate)
    candidates.sort(key=lambda row: (row["selection_sha256"], row["case_id"]))

    fold_counts = Counter(int(row["fold"]) for row in candidates)
    candidate_items = {str(row["item_id"]) for row in candidates}
    if len(candidates) != EXPECTED_CANDIDATE_N:
        raise ValueError(
            f"Stage-09 v2 candidate contexts drifted: {len(candidates)} != {EXPECTED_CANDIDATE_N}"
        )
    if len(candidate_items) != EXPECTED_CANDIDATE_ITEM_N:
        raise ValueError(
            "Stage-09 v2 candidate items drifted: "
            f"{len(candidate_items)} != {EXPECTED_CANDIDATE_ITEM_N}"
        )
    if dict(sorted(fold_counts.items())) != EXPECTED_FOLD_CONTEXT_COUNTS:
        raise ValueError(
            "Stage-09 v2 fold-context counts drifted: "
            f"{dict(sorted(fold_counts.items()))} != {EXPECTED_FOLD_CONTEXT_COUNTS}"
        )

    audit = {
        "source": {
            "path": str(SOURCE_RESULTS),
            "eligible_completed_conflict_n": len(source_rows),
            "eligible_item_n": len({str(row["item_id"]) for row in source_rows}),
        },
        "item_universe": {
            "path": str(BRIDGE01_CONFIRMATORY_RESULTS),
            "completed_item_n": len(item_universe),
        },
        "used_exact_cases": exact_audit,
        "prior_history_targets": history_audit,
        "candidate_context_n": len(candidates),
        "candidate_item_n": len(candidate_items),
        "fold_context_counts": {str(key): value for key, value in sorted(fold_counts.items())},
        "rules": [
            "Truth-01 status=completed",
            "A_T and A_I are non-empty and unequal",
            "condition starts with conflict_",
            "item belongs to the 76 completed Bridge-01 confirmatory items",
            "exact case_id is absent from every Bridge-01/02 target/d1-d4 role",
            "item is absent from every enumerated prior-History target role",
        ],
    }
    return CandidateInventory(
        rows=tuple(candidates),
        source_rows=tuple(source_rows),
        item_universe=frozenset(item_universe),
        used_exact_case_ids=frozenset(used_exact),
        prior_history_item_ids=frozenset(history_items),
        audit=audit,
    )


def _take_item_unique(
    rows: Sequence[dict[str, Any]],
    n: int,
    used_items: set[str],
    *,
    stratum: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda value: (value["selection_sha256"], value["case_id"])):
        item_id = str(row["item_id"])
        if item_id in used_items:
            continue
        selected.append(dict(row))
        used_items.add(item_id)
        if len(selected) == n:
            break
    if len(selected) != n:
        raise ValueError(f"Only {len(selected)} item-unique rows available for {stratum}; expected {n}")
    return selected


def select_primary_and_reserve(
    candidates: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply frozen quotas, then retain every non-primary exact context.

    Hard-side quotas are filled before easy rows because the fold-2/Text-hard
    cell has only one exact context.  Hash order is the sole ordering within a
    quota cell.  Reserve rows include alternate contexts of primary items; the
    ``eligible_as_item_replacement`` flag tells a runner which rows can replace
    a failed item without violating primary item uniqueness.
    """

    primary: list[dict[str, Any]] = []
    used_items: set[str] = set()
    for fold in range(5):
        quotas = HARD_SIDE_QUOTAS[fold]
        for side in ("text", "image"):
            cell = [
                row
                for row in candidates
                if int(row["fold"]) == fold
                and str(row["difficulty"]) == "hard"
                and _side_name(row) == side
            ]
            primary.extend(
                _take_item_unique(
                    cell,
                    quotas[side],
                    used_items,
                    stratum=f"fold={fold}/hard/{side}",
                )
            )
        easy = [
            row
            for row in candidates
            if int(row["fold"]) == fold and str(row["difficulty"]) == "easy"
        ]
        primary.extend(
            _take_item_unique(
                easy,
                PRIMARY_PER_FOLD_DIFFICULTY,
                used_items,
                stratum=f"fold={fold}/easy",
            )
        )

    primary.sort(
        key=lambda row: (
            int(row["fold"]),
            str(row["difficulty"]),
            _side_name(row),
            row["selection_sha256"],
            row["case_id"],
        )
    )
    for rank, row in enumerate(primary, start=1):
        row["selection_role"] = "primary"
        row["primary_rank"] = rank
        row["selection_stratum"] = replacement_stratum(row)

    primary_case_ids = {str(row["case_id"]) for row in primary}
    primary_item_ids = {str(row["item_id"]) for row in primary}
    reserve = [dict(row) for row in candidates if str(row["case_id"]) not in primary_case_ids]
    reserve.sort(
        key=lambda row: (
            int(row["fold"]),
            str(row["difficulty"]),
            _side_name(row) if str(row["difficulty"]) == "hard" else "",
            row["selection_sha256"],
            row["case_id"],
        )
    )
    within_stratum: defaultdict[str, int] = defaultdict(int)
    for rank, row in enumerate(reserve, start=1):
        stratum = replacement_stratum(row)
        within_stratum[stratum] += 1
        row.update(
            {
                "selection_role": "reserve",
                "reserve_rank": rank,
                "reserve_rank_within_stratum": within_stratum[stratum],
                "selection_stratum": stratum,
                "item_already_primary": str(row["item_id"]) in primary_item_ids,
                "eligible_as_item_replacement": str(row["item_id"]) not in primary_item_ids,
            }
        )

    if len(primary) != PRIMARY_N or len(primary_item_ids) != PRIMARY_N:
        raise RuntimeError("Frozen primary selection is not 40 item-unique contexts")
    fold_counts = Counter(int(row["fold"]) for row in primary)
    difficulty_counts = Counter((int(row["fold"]), str(row["difficulty"])) for row in primary)
    if any(fold_counts[fold] != PRIMARY_PER_FOLD for fold in range(5)):
        raise RuntimeError(f"Primary fold allocation drifted: {dict(fold_counts)}")
    if any(
        difficulty_counts[(fold, difficulty)] != PRIMARY_PER_FOLD_DIFFICULTY
        for fold in range(5)
        for difficulty in ("easy", "hard")
    ):
        raise RuntimeError(f"Primary difficulty allocation drifted: {dict(difficulty_counts)}")
    for fold, quotas in HARD_SIDE_QUOTAS.items():
        observed = Counter(
            _side_name(row)
            for row in primary
            if int(row["fold"]) == fold and str(row["difficulty"]) == "hard"
        )
        if dict(observed) != quotas:
            raise RuntimeError(
                f"Primary hard-side allocation drifted in fold {fold}: {dict(observed)} != {quotas}"
            )
    if len(primary) + len(reserve) != len(candidates):
        raise RuntimeError("Primary/reserve partition lost exact contexts")
    return primary, reserve


def choose_structural_replacement(
    plan: "ProspectiveHistoryResponsePlan",
    failed_row: Mapping[str, Any],
    active_rows: Sequence[Mapping[str, Any]],
    attempted_case_ids: set[str] | frozenset[str],
) -> dict[str, Any]:
    """Choose the next outcome-blind reserve after a Phase-0 structural fail.

    The failed row is treated as removed before checking item uniqueness.  No
    History or B/U/A/V value is accepted by this function, which makes the
    replacement rule auditable independently of model outcomes.
    """

    stratum = replacement_stratum(failed_row)
    failed_case_id = str(failed_row["case_id"])
    active_items = {
        str(row["item_id"])
        for row in active_rows
        if str(row["case_id"]) != failed_case_id
    }
    candidates = sorted(
        (
            row
            for row in plan.reserve_rows
            if str(row["selection_stratum"]) == stratum
            and str(row["case_id"]) not in attempted_case_ids
            and str(row["item_id"]) not in active_items
        ),
        key=lambda row: (
            int(row["reserve_rank_within_stratum"]),
            str(row["selection_sha256"]),
            str(row["case_id"]),
        ),
    )
    if not candidates:
        raise ValueError(
            f"No item-unique structural reserve remains for {failed_case_id} in {stratum}"
        )
    replacement = dict(candidates[0])
    replacement["replacement_for_case_id"] = failed_case_id
    replacement["replacement_reason_scope"] = "phase0_structural_only"
    return replacement


def _case_map(artifacts: SAFormationArtifacts) -> dict[tuple[str, int], EvaluationCase]:
    cases, _ = load_evaluation_cases(artifacts.dataset)
    return {(str(case.item_id), int(case.prior_index)): case for case in cases}


def _case_for(
    cases: Mapping[tuple[str, int], EvaluationCase], row: Mapping[str, Any]
) -> EvaluationCase:
    key = (str(row["item_id"]), int(row["prior_index"]))
    if key not in cases:
        raise KeyError(f"Dataset has no case for item/prior {key}")
    case = cases[key]
    if str(row["condition"]) not in case.conditions:
        raise KeyError(f"Dataset case {key} has no condition {row['condition']!r}")
    return case


def _prior_bin(case: EvaluationCase) -> str:
    return str(case.prior_bin or "")


def _history_prompt_length(case: EvaluationCase) -> int:
    return len(
        STAGE1_TEXT_ANSWER_PROMPT.format(question=case.question, text_clue=case.text_clue)
    ) + len(IMAGE_ONLY_ANSWER_PROMPT.format(question=case.question))


def _donor_hash(case_id: str, role: str) -> str:
    return hashlib.sha256(
        f"stage09-v2-donor|42|{role}|{case_id}".encode("utf-8")
    ).hexdigest()


def _history_rank(
    target: Mapping[str, Any],
    donor: Mapping[str, Any],
    cases: Mapping[tuple[str, int], EvaluationCase],
) -> tuple[Any, ...]:
    target_case = _case_for(cases, target)
    donor_case = _case_for(cases, donor)
    ordered_pair_exact = (
        str(target["text_answer"]), str(target["image_answer"])
    ) == (str(donor["text_answer"]), str(donor["image_answer"]))
    return (
        int(not ordered_pair_exact),
        int(_prior_bin(target_case) != _prior_bin(donor_case)),
        int(str(target["condition"]) != str(donor["condition"])),
        abs(_history_prompt_length(target_case) - _history_prompt_length(donor_case)),
        _donor_hash(str(donor["case_id"]), "history"),
        str(donor["case_id"]),
    )


def _replacement_rank(
    target: Mapping[str, Any],
    donor: Mapping[str, Any],
    cases: Mapping[tuple[str, int], EvaluationCase],
    role: str,
) -> tuple[Any, ...]:
    target_case = _case_for(cases, target)
    donor_case = _case_for(cases, donor)
    return (
        int(_prior_bin(target_case) != _prior_bin(donor_case)),
        int(str(target["condition"]) != str(donor["condition"])),
        abs(len(target_case.text_clue) - len(donor_case.text_clue)),
        abs(len(target_case.question) - len(donor_case.question)),
        _donor_hash(str(donor["case_id"]), role),
        str(donor["case_id"]),
    )


def _donor_payload(
    row: Mapping[str, Any],
    *,
    rank: Sequence[Any],
    role: str,
) -> dict[str, Any]:
    return {
        "role": role,
        "case_id": str(row["case_id"]),
        "item_id": str(row["item_id"]),
        "prior_index": int(row["prior_index"]),
        "condition": str(row["condition"]),
        "difficulty": str(row["difficulty"]),
        "fold": int(row["fold"]),
        "text_answer": str(row["text_answer"]),
        "image_answer": str(row["image_answer"]),
        "rank_components": list(rank),
    }


def _choose_distinct_item(
    ranked: Sequence[tuple[tuple[Any, ...], dict[str, Any]]],
    excluded_items: set[str],
    *,
    target_case_id: str,
    role: str,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    for rank, row in ranked:
        if str(row["item_id"]) not in excluded_items:
            return rank, row
    raise ValueError(f"No distinct {role} donor remains for {target_case_id}")


def _plan_donors(
    target: Mapping[str, Any],
    donor_pool: Sequence[dict[str, Any]],
    cases: Mapping[tuple[str, int], EvaluationCase],
) -> dict[str, Any]:
    eligible = [
        row
        for row in donor_pool
        if int(row["fold"]) == int(target["fold"])
        and str(row["difficulty"]) == str(target["difficulty"])
        and str(row["item_id"]) != str(target["item_id"])
    ]
    distinct_eligible_items = {str(row["item_id"]) for row in eligible}
    if len(distinct_eligible_items) < 3:
        raise ValueError(
            f"Fewer than three panel-external donor items for {target['case_id']}: "
            f"{len(distinct_eligible_items)}"
        )

    history_ranked = sorted(
        ((_history_rank(target, row, cases), row) for row in eligible),
        key=lambda pair: pair[0],
    )
    history_rank, history = _choose_distinct_item(
        history_ranked,
        {str(target["item_id"])},
        target_case_id=str(target["case_id"]),
        role="history",
    )
    exact_ordered_pair_available_items = {
        str(row["item_id"])
        for row in eligible
        if (str(row["text_answer"]), str(row["image_answer"]))
        == (str(target["text_answer"]), str(target["image_answer"]))
    }

    used_items = {str(target["item_id"]), str(history["item_id"])}
    replacement_ranked = sorted(
        (
            (_replacement_rank(target, row, cases, "replacement"), row)
            for row in eligible
        ),
        key=lambda pair: pair[0],
    )
    donor5_rank, donor5 = _choose_distinct_item(
        replacement_ranked,
        used_items,
        target_case_id=str(target["case_id"]),
        role="d5",
    )
    used_items.add(str(donor5["item_id"]))
    donor6_rank, donor6 = _choose_distinct_item(
        replacement_ranked,
        used_items,
        target_case_id=str(target["case_id"]),
        role="d6",
    )

    ordered_pair_exact = (
        str(history["text_answer"]) == str(target["text_answer"])
        and str(history["image_answer"]) == str(target["image_answer"])
    )
    role_items = {
        str(history["item_id"]), str(donor5["item_id"]), str(donor6["item_id"])
    }
    if len(role_items) != 3:
        raise RuntimeError(f"Donor roles collide within {target['case_id']}")
    return {
        "eligible_panel_external_context_n": len(eligible),
        "eligible_panel_external_item_n": len(distinct_eligible_items),
        "exact_ordered_pair_available_item_n": len(exact_ordered_pair_available_items),
        "history_match_tier": (
            "exact_ordered_text_image_answer_pair"
            if ordered_pair_exact
            else "fallback_same_fold_difficulty"
        ),
        "history_answer_identity": {
            "ordered_pair_exact": ordered_pair_exact,
            "text_answer_equal": str(history["text_answer"]) == str(target["text_answer"]),
            "image_answer_equal": str(history["image_answer"])
            == str(target["image_answer"]),
        },
        "history_donor": _donor_payload(history, rank=history_rank, role="history"),
        "donor5": _donor_payload(donor5, rank=donor5_rank, role="d5"),
        "donor6": _donor_payload(donor6, rank=donor6_rank, role="d6"),
    }


def build_plan(
    artifacts: SAFormationArtifacts | str | Path,
) -> ProspectiveHistoryResponsePlan:
    """Build the entire deterministic plan without writing or model access."""

    if not isinstance(artifacts, SAFormationArtifacts):
        artifacts = SAFormationArtifacts.discover(artifacts)
    inventory = build_candidate_inventory(artifacts)
    primary, reserve = select_primary_and_reserve(inventory.rows)
    cases = _case_map(artifacts)

    # Donors are external to the complete 67-item target universe, not merely
    # external to the 40 selected primary rows.  This prevents a reserve target
    # from having appeared as another row's donor before replacement decisions.
    target_item_universe = {str(row["item_id"]) for row in inventory.rows}
    donor_pool = [
        dict(row)
        for row in inventory.source_rows
        if str(row["item_id"]) not in target_item_universe
    ]
    if len({str(row["item_id"]) for row in donor_pool}) < 3:
        raise ValueError("Panel-external donor pool has fewer than three distinct items")

    selection_by_case = {
        str(row["case_id"]): row for row in (*primary, *reserve)
    }
    all_rows: list[dict[str, Any]] = []
    fallback_cases: list[str] = []
    for candidate in inventory.rows:
        selected = selection_by_case[str(candidate["case_id"])]
        donors = _plan_donors(selected, donor_pool, cases)
        row = {**selected, **donors}
        if row["history_match_tier"] != "exact_ordered_text_image_answer_pair":
            fallback_cases.append(str(row["case_id"]))
        all_rows.append(row)
    by_case = {str(row["case_id"]): row for row in all_rows}
    primary_planned = [by_case[str(row["case_id"])] for row in primary]
    reserve_planned = [by_case[str(row["case_id"])] for row in reserve]

    exact_primary = sum(
        row["history_match_tier"] == "exact_ordered_text_image_answer_pair"
        for row in primary_planned
    )
    diagnostics = {
        "donor_pool_context_n": len(donor_pool),
        "donor_pool_item_n": len({str(row["item_id"]) for row in donor_pool}),
        "panel_external_item_n": len(target_item_universe),
        "planned_context_n": len(all_rows),
        "history_exact_ordered_pair_n": len(all_rows) - len(fallback_cases),
        "history_fallback_n": len(fallback_cases),
        "primary_history_exact_ordered_pair_n": exact_primary,
        "primary_history_fallback_n": len(primary_planned) - exact_primary,
        "fallback_case_ids": fallback_cases,
        "fallback_implication": (
            "Relevant/irrelevant replay-answer identity is not held constant. "
            "Use the saved per-branch answer-identity factor in analysis."
        ),
    }
    return ProspectiveHistoryResponsePlan(
        artifacts=artifacts,
        cases=cases,
        inventory=inventory,
        primary_rows=tuple(primary_planned),
        reserve_rows=tuple(reserve_planned),
        all_rows=tuple(all_rows),
        donor_diagnostics=diagnostics,
    )


def protocol_manifest() -> dict[str, Any]:
    """Return the frozen design specification, including cell arithmetic."""

    payload: dict[str, Any] = {
        "format_version": 1,
        "panel_version": PANEL_VERSION,
        "experiment": "prospective_history_response_panel_v2",
        "seed": SEED,
        "selection_salt": SELECTION_SALT,
        "primary_n": PRIMARY_N,
        "branches": list(BRANCHES),
        "history_factorial": {
            "relevance": ["relevant", "irrelevant"],
            "modality": ["text", "image"],
            "replay_side": ["text", "image"],
            "history_branch_n": len(HISTORY_BRANCH_FACTORS),
            "plus_no_history": True,
            "branch_factors": HISTORY_BRANCH_FACTORS,
        },
        "evidence_conditions": list(EVIDENCE_CONDITIONS),
        "primary_allocation": {
            "per_fold": PRIMARY_PER_FOLD,
            "per_fold_easy": PRIMARY_PER_FOLD_DIFFICULTY,
            "per_fold_hard": PRIMARY_PER_FOLD_DIFFICULTY,
            "hard_legacy_side_quotas": {
                str(fold): quotas for fold, quotas in HARD_SIDE_QUOTAS.items()
            },
        },
        "target_novelty": {
            "unit": "exact case_id plus prior-History target item",
            "candidate_expected_context_n": EXPECTED_CANDIDATE_N,
            "candidate_expected_item_n": EXPECTED_CANDIDATE_ITEM_N,
            "candidate_expected_fold_context_counts": {
                str(key): value for key, value in EXPECTED_FOLD_CONTEXT_COUNTS.items()
            },
        },
        "donors": {
            "planned_for": "all primary and reserve exact contexts before new outcomes",
            "pool": "Truth-01 eligible conflicts outside the complete 67-item target universe",
            "shared_constraints": ["same fold", "same difficulty", "item differs from target"],
            "history_preference": "same ordered (A_T,A_I) pair, then metadata/length/hash",
            "history_fallback": (
                "same fold/difficulty; answer identity is retained as an explicit analysis factor"
            ),
            "replacement_constraints": "history, d5, and d6 are distinct items within row",
        },
        "answer_only": {
            "verbal_sa_request_allowed": False,
            "history_prefix_turns": 2,
            "final_target_turn_identical_across_branches_within_condition": True,
        },
        "planned_cells": {
            "behavior_per_item": len(BRANCHES) * len(EVIDENCE_CONDITIONS),
            "behavior_primary_total": PRIMARY_N
            * len(BRANCHES)
            * len(EVIDENCE_CONDITIONS),
            "postanswer_hidden_per_item_if_all_branches_measured": len(BRANCHES),
            "postanswer_hidden_primary_total_if_all_branches_measured": PRIMARY_N
            * len(BRANCHES),
            "joint_common9_per_item_if_all_branches_measured": len(BRANCHES),
            "joint_common9_primary_total_if_all_branches_measured": PRIMARY_N
            * len(BRANCHES),
            "formal_forward_per_item_if_both_tracks_authorized": (
                len(BRANCHES) * (len(EVIDENCE_CONDITIONS) + 2)
            ),
            "formal_forward_primary_total_if_both_tracks_authorized": (
                PRIMARY_N * len(BRANCHES) * (len(EVIDENCE_CONDITIONS) + 2)
            ),
        },
        "claim_scope": (
            "History is a controlled paired prompt intervention; hidden/readout coordinates remain "
            "measurements unless separately intervention-authorized."
        ),
    }
    payload["protocol_fingerprint"] = stable_hash(payload)
    return payload


def _cohort_row(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "case_id",
        "item_id",
        "prior_index",
        "condition",
        "difficulty",
        "fold",
        "text_answer",
        "image_answer",
        "legacy_final_image",
        "prior_strength",
        "selection_sha256",
        "source_row_fingerprint",
        "selection_role",
        "selection_stratum",
        "primary_rank",
        "reserve_rank",
        "reserve_rank_within_stratum",
        "item_already_primary",
        "eligible_as_item_replacement",
    )
    return {key: row[key] for key in keys if key in row}


def cohort_candidate_manifest(plan: ProspectiveHistoryResponsePlan) -> dict[str, Any]:
    """Return the full candidate partition, not only the selected 40 rows."""

    primary = [_cohort_row(row) for row in plan.primary_rows]
    reserve = [_cohort_row(row) for row in plan.reserve_rows]
    payload: dict[str, Any] = {
        "format_version": 1,
        "panel_version": PANEL_VERSION,
        "candidate_n": len(plan.all_rows),
        "candidate_item_n": len({str(row["item_id"]) for row in plan.all_rows}),
        "primary_n": len(primary),
        "primary_item_n": len({row["item_id"] for row in primary}),
        "reserve_exact_context_n": len(reserve),
        "reserve_nonprimary_item_n": len(
            {
                row["item_id"]
                for row in reserve
                if row.get("eligible_as_item_replacement") is True
            }
        ),
        "selection": protocol_manifest()["primary_allocation"],
        "inventory_audit": plan.inventory.audit,
        "primary_rows": primary,
        "reserve_rows": reserve,
    }
    payload["cohort_fingerprint"] = stable_hash(payload)
    return payload


def donor_manifest(plan: ProspectiveHistoryResponsePlan) -> dict[str, Any]:
    """Return preplanned History and replacement donors for all 395 rows."""

    rows = [
        {
            "case_id": row["case_id"],
            "item_id": row["item_id"],
            "selection_role": row["selection_role"],
            "fold": row["fold"],
            "difficulty": row["difficulty"],
            "target_ordered_answer_pair": [row["text_answer"], row["image_answer"]],
            "eligible_panel_external_context_n": row[
                "eligible_panel_external_context_n"
            ],
            "eligible_panel_external_item_n": row["eligible_panel_external_item_n"],
            "exact_ordered_pair_available_item_n": row[
                "exact_ordered_pair_available_item_n"
            ],
            "history_match_tier": row["history_match_tier"],
            "history_answer_identity": row["history_answer_identity"],
            "history_donor": row["history_donor"],
            "donor5": row["donor5"],
            "donor6": row["donor6"],
        }
        for row in plan.all_rows
    ]
    payload: dict[str, Any] = {
        "format_version": 1,
        "panel_version": PANEL_VERSION,
        "selection_time": "before every new History response outcome",
        "rank_order": {
            "history": [
                "ordered (A_T,A_I) mismatch",
                "prior-bin mismatch",
                "condition mismatch",
                "combined Text/Image prompt-length difference",
                "seeded SHA256",
            ],
            "d5_d6": [
                "prior-bin mismatch",
                "condition mismatch",
                "text-clue length difference",
                "question length difference",
                "seeded SHA256",
            ],
        },
        "diagnostics": plan.donor_diagnostics,
        "rows": rows,
    }
    payload["donor_fingerprint"] = stable_hash(payload)
    return payload


def evidence_condition_sources(
    plan: ProspectiveHistoryResponsePlan,
    row: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Resolve the seven symmetric evidence cells for one target row."""

    target_case = _case_for(plan.cases, row)
    target_item = str(row["item_id"])
    target_image = str(
        target_case.conditions[str(row["condition"])].resolved_image_path
    )
    null_image = str(target_case.conditions["null"].resolved_image_path)
    result: dict[str, dict[str, Any]] = {
        "full": {
            "text_clue": target_case.text_clue,
            "image_path": target_image,
            "text_source_item": target_item,
            "image_source_item": target_item,
        },
        "no_text": {
            "text_clue": NO_TEXT_PLACEHOLDER,
            "image_path": target_image,
            "text_source_item": None,
            "image_source_item": target_item,
        },
        "no_image": {
            "text_clue": target_case.text_clue,
            "image_path": null_image,
            "text_source_item": target_item,
            "image_source_item": None,
        },
    }
    for index in (5, 6):
        donor = row[f"donor{index}"]
        donor_case = _case_for(plan.cases, donor)
        donor_item = str(donor["item_id"])
        donor_image = str(
            donor_case.conditions[str(donor["condition"])].resolved_image_path
        )
        result[f"replace_text_d{index}"] = {
            "text_clue": donor_case.text_clue,
            "image_path": target_image,
            "text_source_item": donor_item,
            "image_source_item": target_item,
        }
        result[f"replace_image_d{index}"] = {
            "text_clue": target_case.text_clue,
            "image_path": donor_image,
            "text_source_item": target_item,
            "image_source_item": donor_item,
        }
    ordered = {condition: result[condition] for condition in EVIDENCE_CONDITIONS}
    for index in (5, 6):
        replace_text = ordered[f"replace_text_d{index}"]
        replace_image = ordered[f"replace_image_d{index}"]
        if replace_text["text_source_item"] != replace_image["image_source_item"]:
            raise RuntimeError(f"Donor d{index} is not symmetric for {row['case_id']}")
        if replace_text["image_source_item"] != replace_image["text_source_item"]:
            raise RuntimeError(f"Target context is not symmetric for {row['case_id']}")
    return ordered


def _history_first_turn(
    case: EvaluationCase, condition: str, modality: str
) -> dict[str, Any]:
    if modality == "text":
        prompt = STAGE1_TEXT_ANSWER_PROMPT.format(
            question=case.question, text_clue=case.text_clue
        )
        return {"role": "user", "content": text_content(prompt)}
    if modality == "image":
        prompt = IMAGE_ONLY_ANSWER_PROMPT.format(question=case.question)
        return {
            "role": "user",
            "content": image_content(
                str(case.conditions[str(condition)].resolved_image_path), prompt
            ),
        }
    raise ValueError(f"Unknown History modality: {modality}")


def history_branch_factors(
    row: Mapping[str, Any], branch: str
) -> dict[str, Any]:
    """Expose replay congruence and answer-identity factors for analysis."""

    if branch == "no_history":
        return {"branch": branch, "has_history": False}
    if branch not in HISTORY_BRANCH_FACTORS:
        raise ValueError(f"Unknown Stage-09 v2 branch: {branch}")
    factors = HISTORY_BRANCH_FACTORS[branch]
    relevance = factors["relevance"]
    modality = factors["modality"]
    replay_side = factors["replay_side"]
    history_row = row if relevance == "relevant" else row["history_donor"]
    replayed_answer = str(history_row[f"{replay_side}_answer"])
    modality_answer = str(history_row[f"{modality}_answer"])
    target_replay_answer = str(row[f"{replay_side}_answer"])
    return {
        "branch": branch,
        "has_history": True,
        **factors,
        "history_case_id": str(history_row["case_id"]),
        "history_item_id": str(history_row["item_id"]),
        "replayed_answer": replayed_answer,
        "target_same_side_answer": target_replay_answer,
        "answer_identity_matches_target": replayed_answer == target_replay_answer,
        "history_source_congruent_with_replay": replayed_answer == modality_answer,
        "factorial_congruence_expected": modality == replay_side,
        "history_ordered_pair_matches_target": bool(
            relevance == "relevant"
            or row["history_answer_identity"]["ordered_pair_exact"]
        ),
    }


def build_messages(
    plan: ProspectiveHistoryResponsePlan,
    row: Mapping[str, Any],
    branch: str,
    condition: str,
    *,
    assistant_text: str = ASSISTANT_ANSWER_PREFILL,
) -> list[dict[str, Any]]:
    """Construct one answer-only History/evidence cell.

    The final two messages are built once from the target/evidence condition
    and then prefixed with History.  Consequently the causal final target turn
    is byte-identical across all nine branches for a fixed condition.
    """

    if branch not in BRANCHES:
        raise ValueError(f"Unknown Stage-09 v2 branch: {branch}")
    if condition not in EVIDENCE_CONDITIONS:
        raise ValueError(f"Unknown Stage-09 v2 evidence condition: {condition}")
    target_case = _case_for(plan.cases, row)
    source = evidence_condition_sources(plan, row)[condition]
    final_messages = build_answer_only_messages(
        target_case,
        text_clue=str(source["text_clue"]),
        image_path=str(source["image_path"]),
        assistant_text=assistant_text,
    )
    if branch == "no_history":
        messages = final_messages
    else:
        factors = history_branch_factors(row, branch)
        history_row = row if factors["relevance"] == "relevant" else row["history_donor"]
        history_case = _case_for(plan.cases, history_row)
        messages = [
            _history_first_turn(
                history_case,
                str(history_row["condition"]),
                str(factors["modality"]),
            ),
            assistant_message(f"**Answer**: {factors['replayed_answer']}"),
            *final_messages,
        ]
    if contains_verbal_sa_request(messages):
        raise ValueError(f"Answer-only Stage-09 branch leaks verbal SA: {row['case_id']} {branch}")
    return messages


def build_joint_history_messages(
    plan: ProspectiveHistoryResponsePlan,
    row: Mapping[str, Any],
    branch: str,
    *,
    answer_star: str,
) -> tuple[list[dict[str, Any]], str]:
    """Construct the frozen common-nine A/V context for one History branch.

    The answer-only prompt cannot measure verbal attribution because it never
    asks for Source Attribution.  This builder therefore uses the exact
    Bridge-06 common-nine target turn and prepends only the already frozen
    History bundle.  Across branches, the joint target user/assistant turn is
    byte-identical for a fixed recipient and ``A*``.
    """

    if branch not in BRANCHES:
        raise ValueError(f"Unknown Stage-09 v2 branch: {branch}")
    answer = str(answer_star)
    if not answer:
        raise ValueError("Stage-09 joint measurement requires a fixed A*")
    protocol = panel_protocols()[0]
    if protocol.name != "common_9_ordered":
        raise RuntimeError("Frozen common-nine protocol order drifted")
    target_case = _case_for(plan.cases, row)
    final_messages, assistant_text = build_frozen_joint_messages(
        target_case,
        str(row["condition"]),
        protocol,
        answer_star=answer,
    )
    if branch == "no_history":
        return final_messages, assistant_text
    # Reuse the audited answer-only builder only to obtain the two-turn History
    # prefix.  Its final target turn is discarded, not repurposed as an SA
    # prompt.
    prefix = build_messages(plan, row, branch, "full")[:2]
    return [*prefix, *final_messages], assistant_text


def joint_message_audit(
    plan: ProspectiveHistoryResponsePlan,
    endpoints: Mapping[str, str],
) -> dict[str, Any]:
    """Audit frozen common-nine reconstruction after Phase-0 A* selection."""

    failures: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for row in plan.primary_rows:
        case_id = str(row["case_id"])
        answer = endpoints.get(case_id)
        if not answer:
            failures.append(
                {"case_id": case_id, "error": "missing frozen Phase-0 A*"}
            )
            continue
        final_hashes: set[str] = set()
        prefix_hashes: dict[str, str] = {}
        message_hashes: dict[str, str] = {}
        for branch in BRANCHES:
            try:
                messages, assistant_text = build_joint_history_messages(
                    plan, row, branch, answer_star=answer
                )
                expected_turns = 2 if branch == "no_history" else 4
                if len(messages) != expected_turns:
                    raise RuntimeError(
                        f"joint turn_n={len(messages)}; expected {expected_turns}"
                    )
                if messages[-1]["content"][0]["text"] != assistant_text:
                    raise RuntimeError("joint assistant continuation drifted")
                final_hashes.add(canonical_message_hash(messages[-2:]))
                message_hashes[branch] = canonical_message_hash(messages)
                if branch != "no_history":
                    prefix_hashes[branch] = canonical_message_hash(messages[:2])
            except Exception as exc:
                failures.append(
                    {
                        "case_id": case_id,
                        "branch": branch,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        if len(final_hashes) != 1:
            failures.append(
                {
                    "case_id": case_id,
                    "error_type": "JointFinalTurnMismatch",
                    "error": f"joint target turn has {len(final_hashes)} hashes",
                }
            )
        rows.append(
            {
                "case_id": case_id,
                "answer_star": answer,
                "joint_final_turn_hash": (
                    next(iter(final_hashes)) if len(final_hashes) == 1 else None
                ),
                "history_prefix_hashes": prefix_hashes,
                "message_hashes": message_hashes,
            }
        )
    payload: dict[str, Any] = {
        "format_version": 1,
        "panel_version": PANEL_VERSION,
        "row_n": len(rows),
        "branch_n": len(BRANCHES),
        "failure_n": len(failures),
        "passed": not failures and len(rows) == len(plan.primary_rows),
        "failures": failures,
        "rows": rows,
    }
    payload["joint_message_audit_fingerprint"] = stable_hash(payload)
    return payload


def message_audit(
    plan: ProspectiveHistoryResponsePlan,
    *,
    include_reserve: bool = True,
) -> dict[str, Any]:
    """Audit reconstruction invariants and return all reproducibility hashes."""

    rows = plan.all_rows if include_reserve else plan.primary_rows
    failures: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for row in rows:
        final_hashes: dict[str, set[str]] = {
            condition: set() for condition in EVIDENCE_CONDITIONS
        }
        full_hashes: dict[str, dict[str, str]] = {}
        prefix_hashes: dict[str, str] = {}
        branch_factors: dict[str, dict[str, Any]] = {}
        for branch in BRANCHES:
            full_hashes[branch] = {}
            factors = history_branch_factors(row, branch)
            branch_factors[branch] = factors
            for condition in EVIDENCE_CONDITIONS:
                try:
                    messages = build_messages(plan, row, branch, condition)
                    expected_turn_n = 2 if branch == "no_history" else 4
                    if len(messages) != expected_turn_n:
                        raise RuntimeError(
                            f"turn_n={len(messages)}; expected {expected_turn_n}"
                        )
                    if contains_verbal_sa_request(messages):
                        raise RuntimeError("verbal-SA request leakage")
                    final_hash = canonical_message_hash(messages[-2:])
                    final_hashes[condition].add(final_hash)
                    full_hashes[branch][condition] = canonical_message_hash(messages)
                    if branch != "no_history":
                        prefix_hash = canonical_message_hash(messages[:2])
                        prior = prefix_hashes.setdefault(branch, prefix_hash)
                        if prior != prefix_hash:
                            raise RuntimeError("History prefix changed across evidence conditions")
                except Exception as exc:
                    failures.append(
                        {
                            "case_id": row["case_id"],
                            "branch": branch,
                            "condition": condition,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
        final_turn_hashes: dict[str, str | None] = {}
        for condition, hashes in final_hashes.items():
            if len(hashes) != 1:
                failures.append(
                    {
                        "case_id": row["case_id"],
                        "condition": condition,
                        "error_type": "FinalTurnMismatch",
                        "error": f"Final target turn has {len(hashes)} hashes across branches",
                    }
                )
                final_turn_hashes[condition] = None
            else:
                final_turn_hashes[condition] = next(iter(hashes))
        audits.append(
            {
                "case_id": row["case_id"],
                "selection_role": row["selection_role"],
                "history_match_tier": row["history_match_tier"],
                "final_turn_hashes": final_turn_hashes,
                "history_prefix_hashes": prefix_hashes,
                "full_message_hashes": full_hashes,
                "branch_factors": branch_factors,
            }
        )
    payload: dict[str, Any] = {
        "format_version": 1,
        "panel_version": PANEL_VERSION,
        "include_reserve": include_reserve,
        "row_n": len(rows),
        "branch_n": len(BRANCHES),
        "condition_n": len(EVIDENCE_CONDITIONS),
        "message_cell_n": len(rows) * len(BRANCHES) * len(EVIDENCE_CONDITIONS),
        "failure_n": len(failures),
        "passed": not failures,
        "failures": failures,
        "rows": audits,
    }
    payload["message_audit_fingerprint"] = stable_hash(payload)
    return payload


def audit_plan_messages(
    plan: ProspectiveHistoryResponsePlan,
    *,
    include_reserve: bool = True,
) -> dict[str, Any]:
    """Compatibility alias for callers following the Stage-09 v1 API."""

    return message_audit(plan, include_reserve=include_reserve)
