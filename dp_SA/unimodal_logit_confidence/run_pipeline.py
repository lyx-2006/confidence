from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .analyze import analyze
from .build_split import build_split
from .capture import capture
from .config import (BOOTSTRAP_REPEATS, DATASET_PATH, FAMILY_MANIFEST, MODEL_PATH, PACKAGE_ROOT, PROBE_LAYERS, PROBE_POSITIONS, RESULTS_ROOT,
                     SEED, SOURCE_CAPTURE, SOURCE_CONFIG, SOURCE_MANIFEST, SUPPLEMENT_CAPTURE, TARGETS, TEXT_PHASE0_TEMPLATE, IMAGE_PHASE0_TEMPLATE)
from .fit_temperature import fit_temperature
from .io_utils import atomic_json, canonical_hash, ensure_layout, sha256_file, validate_fingerprint
from .score_unimodal import score_unimodal
from .train_probe import train_probe


def initialize(root: Path, *, resume: bool, num_gpus: int) -> dict[str,Any]:
    payload={"format_version":1,"experiment":"unimodal_logit_confidence","seed":SEED,"dataset":{"path":str(DATASET_PATH.resolve()),"sha256":sha256_file(DATASET_PATH)},"source_manifest":{"path":str(SOURCE_MANIFEST.resolve()),"sha256":sha256_file(SOURCE_MANIFEST)},"source_capture_config_sha256":sha256_file(SOURCE_CONFIG),"source_capture_sha256":sha256_file(SOURCE_CAPTURE),"supplement_capture_sha256":sha256_file(SUPPLEMENT_CAPTURE),"family_manifest_sha256":sha256_file(FAMILY_MANIFEST),"model":str(MODEL_PATH.resolve()),"model_config_sha256":sha256_file(MODEL_PATH/"config.json"),"tokenizer_sha256":sha256_file(MODEL_PATH/"tokenizer.json"),"templates":{"text":canonical_hash(TEXT_PHASE0_TEMPLATE),"image":canonical_hash(IMAGE_PHASE0_TEMPLATE)},"positions":list(PROBE_POSITIONS),"layers":list(PROBE_LAYERS),"targets":list(TARGETS),"shard_policy":"sha256 canonical JSON modulo num_gpus","num_gpus":num_gpus,"source_code":{path.name:sha256_file(path) for path in sorted(PACKAGE_ROOT.glob("*.py"))}}
    fingerprint=validate_fingerprint(root/"shared/run_config.json",payload,resume=resume); inputs={**payload,"fingerprint":fingerprint}; atomic_json(root/"shared/input_fingerprints.json",inputs); return inputs


def run_pipeline(root: Path, *, num_gpus: int, resume: bool, bootstrap: int = BOOTSTRAP_REPEATS) -> dict[str,Any]:
    initialize(root,resume=resume,num_gpus=num_gpus); split=build_split(root); score=score_unimodal(root,num_gpus=num_gpus,resume=resume); temperature=fit_temperature(root); hidden=capture(root,num_gpus=num_gpus,resume=resume); probe=train_probe(root,bootstrap=bootstrap); analysis=analyze(root)
    result={"status":"complete","split":split,"score":score,"temperature":temperature,"hidden":hidden,"probe":probe,"analysis":analysis}; atomic_json(root/"shared/completion.json",result); return result


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--num-gpus",type=int,choices=(1,2),default=1); parser.add_argument("--output-root",default=str(RESULTS_ROOT)); parser.add_argument("--resume",action="store_true"); parser.add_argument("--bootstrap",type=int,default=BOOTSTRAP_REPEATS)
    args=parser.parse_args(argv); root=ensure_layout(args.output_root,resume=args.resume); print(json.dumps(run_pipeline(root,num_gpus=args.num_gpus,resume=args.resume,bootstrap=args.bootstrap),ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
