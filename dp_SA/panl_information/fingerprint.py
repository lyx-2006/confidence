from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from confidence_test.inference_extension import ASSISTANT_ANSWER_PREFILL
from confidence_test.prompt_utils import STAGE1_TEXT_ANSWER_PROMPT
from layer_metacognition.probe.prompts import IMAGE_ONLY_ANSWER_PROMPT

from dp_SA.prompts import PHASE1_TEMPLATE, SA_PREFILL

from .config import (
    DATASET_PATH, HISTORICAL_LAYERS, HISTORICAL_POSITIONS, LAYERS, MODEL_PATH,
    PACKAGE_ROOT, POSITIONS, SOURCE_CONFIG, SOURCE_OOF, SOURCE_PHASE0,
    SOURCE_RESULTS, SPLIT_PATH,
)
from .io_utils import canonical_hash, load_jsonl, sha256_file


def _tree_hashes(root: Path, patterns: tuple[str, ...]) -> dict[str, str]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(root.glob(pattern))
    return {str(path.relative_to(root)): sha256_file(path) for path in sorted(set(paths)) if path.is_file()}


def compute_input_fingerprints(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    candidates = sorted({candidate for row in manifest for candidate in row["answer_classes"]})
    tokenization = {candidate: tokenizer.encode(ASSISTANT_ANSWER_PREFILL + candidate, add_special_tokens=False)[-len(tokenizer.encode(candidate, add_special_tokens=False)):] for candidate in candidates}
    image_paths = sorted({str(Path(row["image_path"]).resolve()) for row in manifest})
    relevant_existing = [
        Path("dp_SA/capture.py"), Path("dp_SA/positions.py"), Path("dp_SA/prompts.py"), Path("dp_SA/soft_score.py"),
        Path("layer_metacognition/model_adapter.py"), Path("layer_metacognition/conversation_builder.py"),
        Path("layer_metacognition/probe/probe_models.py"), Path("layer_metacognition/probe/split_utils.py"),
        Path("confidence_test/dataset_utils.py"), Path("confidence_test/prompt_utils.py"),
    ]
    repo = PACKAGE_ROOT.parents[1]
    source_files = sorted(PACKAGE_ROOT.glob("*.py")) + [repo / path for path in relevant_existing]
    payload = {
        "format_version": 1,
        "dataset": {"path": str(DATASET_PATH.resolve()), "sha256": sha256_file(DATASET_PATH)},
        "model": {"path": str(MODEL_PATH.resolve()), "files": _tree_hashes(MODEL_PATH, ("*.json", "*.txt", "*.safetensors"))},
        "processor_tokenizer": _tree_hashes(MODEL_PATH, ("tokenizer*", "vocab.json", "merges.txt", "preprocessor_config.json", "chat_template.json")),
        "chat_template_hash": canonical_hash(getattr(tokenizer, "chat_template", None)),
        "prompts": {"text_only": canonical_hash(STAGE1_TEXT_ANSWER_PROMPT), "image_only": canonical_hash(IMAGE_ONLY_ANSWER_PROMPT), "phase1": canonical_hash(PHASE1_TEMPLATE), "answer_prefill": ASSISTANT_ANSWER_PREFILL, "sa_prefill": SA_PREFILL},
        "frozen_answers": {"phase0_sha256": sha256_file(SOURCE_PHASE0), "phase1_results_sha256": sha256_file(SOURCE_RESULTS), "capture_config_sha256": sha256_file(SOURCE_CONFIG)},
        "images": {path: sha256_file(path) for path in image_paths},
        "candidate_color_tokenization": tokenization,
        "split_assignments": {"path": str(SPLIT_PATH.resolve()), "sha256": sha256_file(SPLIT_PATH)},
        "historical_oof": {"path": str(SOURCE_OOF.resolve()), "sha256": sha256_file(SOURCE_OOF)},
        "positions_layers": {"positions": list(POSITIONS), "layers": list(LAYERS), "historical_positions": list(HISTORICAL_POSITIONS), "historical_layers": list(HISTORICAL_LAYERS)},
        "source_code": {str(path.relative_to(repo)): sha256_file(path) for path in source_files},
        "target_definitions": {"difficulty": "100*restricted_candidate_entropy/log(12)", "decision": ["follow_text", "follow_image"], "gap": "D_text/100-D_image/100", "overall": "(D_text/100+D_image/100)/2"},
        "output_schema": {"root": "dp_SA/panl_information/output/results", "tables": ["difficulty_probe.csv", "decision_probe.csv", "sa_factor_correlations.csv", "regression_parameters.md"], "figures": ["difficulty_probe_R2.png", "difficulty_probe_spearman.png", "decision_probe_accuracy.png"]},
    }
    payload["fingerprint"] = canonical_hash(payload)
    return payload
