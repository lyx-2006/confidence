from __future__ import annotations

from pathlib import Path

import pytest

from dp_SA.answer_matched_lat_steering.config import HISTORICAL_CAPTURE, HISTORICAL_CONSTRUCTION, HISTORICAL_TEST
from dp_SA.answer_matched_lat_steering.fingerprint import check_or_write
from dp_SA.answer_matched_lat_steering.io_utils import sha256_file


def test_resume_fingerprint_gate(tmp_path: Path):
    path = tmp_path / "config.json"
    check_or_write(path, {"fingerprint": "a"}, resume=False)
    check_or_write(path, {"fingerprint": "a"}, resume=True)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        check_or_write(path, {"fingerprint": "b"}, resume=True)
    with pytest.raises(FileExistsError):
        check_or_write(path, {"fingerprint": "a"}, resume=False)


def test_historical_inputs_are_present_and_read_only_source_fingerprints_stable():
    expected = {
        HISTORICAL_CAPTURE: "356e73b66fd483676b952f4b78ce41ee9404a1d0e52f4174a1a34ec4fdef7332",
        HISTORICAL_CONSTRUCTION: "6e7843a8fa51de219d6f6b883c79a5a31d6ffdf8692ad04779188c476f0a888e",
        HISTORICAL_TEST: "ad44ec8499ecd8ba0c3f75c9cad2c91b90df6ede69ad96733364ca9e71d52a06",
    }
    assert {path: sha256_file(path) for path in expected} == expected
