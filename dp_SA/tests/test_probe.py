from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dp_SA.io_utils import atomic_jsonl, load_jsonl
from dp_SA.probe import bootstrap_seed, main as probe_main, parse_cells, run_probe


SIX_CELLS=(
    "P1_LAT:12","P1_LAT:14","P1_LAT:16",
    "P1_CLASS_LIST_END:16","P1_CLASS_LIST_END:18","P1_CLASS_LIST_END:20",
)


def test_exact_cells_are_deduplicated_without_cartesian_expansion():
    cells=parse_cells([*SIX_CELLS,"P1_LAT:12"])
    assert len(cells)==6
    assert cells[0]==("P1_LAT",12) and cells[-1]==("P1_CLASS_LIST_END",20)


@pytest.mark.parametrize("value",["bad","P1_LAT:x","P1_LAT:-1","UNKNOWN:12","P1_LAT:1:2"])
def test_invalid_cell_formats_fail(value):
    with pytest.raises(ValueError): parse_cells([value])


def test_cell_and_candidates_cli_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        probe_main(["--cell","P1_LAT:12","--candidates","candidates.jsonl"])


def test_bootstrap_seed_is_stable_and_position_aware():
    assert bootstrap_seed(42,"P1_LAT",16)==bootstrap_seed(42,"P1_LAT",16)
    assert bootstrap_seed(42,"P1_LAT",16)!=bootstrap_seed(42,"P1_CLASS_LIST_END",16)


def _probe_fixture(root: Path, cells: list[tuple[str,int]]) -> Path:
    records=[]; split={}
    for item in range(10):
        split[str(item)]=item%5
        for replicate in range(2):
            case_id=f"c{item}_{replicate}"; relative=Path("capture")/"hidden"/f"{case_id}.npz"
            path=root/relative; path.parent.mkdir(parents=True,exist_ok=True)
            arrays={f"{position}__L{layer}":np.asarray([item,replicate,item+replicate],dtype=np.float16)
                    for position,layer in cells}
            np.savez(path,**arrays)
            records.append({"status":"completed","case_id":case_id,"item_id":str(item),
                            "soft_sa_image_score":item/10+replicate/100,"hidden_file":str(relative)})
    atomic_jsonl(root/"capture"/"results.jsonl",records)
    split_path=root/"split.json"; split_path.write_text(json.dumps({"item_to_fold":split}))
    return split_path


def test_explicit_probe_cells_run_item_oof_and_candidate_path_remains_supported(tmp_path: Path):
    cells=[("P1_LAT",12),("P1_CLASS_LIST_END",16)]
    split=_probe_fixture(tmp_path,cells)
    summary=run_probe(tmp_path,None,split,bootstrap=20,cells=cells)
    assert [(row["position"],row["layer"]) for row in summary["metrics"]]==cells
    predictions=load_jsonl(tmp_path/"probe"/"oof_predictions.jsonl")
    assert {(row["position"],row["layer"]) for row in predictions}==set(cells)
    assert all(row["fold"]==int(row["item_id"])%5 for row in predictions)
    candidates=tmp_path/"candidates.jsonl"; atomic_jsonl(candidates,[{"position":"P1_LAT","layer":12}])
    assert run_probe(tmp_path,candidates,split,bootstrap=10)["candidate_count"]==1


def test_probe_reports_missing_hidden_key(tmp_path: Path):
    split=_probe_fixture(tmp_path,[("P1_LAT",12)])
    with pytest.raises(KeyError,match="P1_CLASS_LIST_END__L16"):
        run_probe(tmp_path,None,split,bootstrap=2,cells=[("P1_CLASS_LIST_END",16)])
