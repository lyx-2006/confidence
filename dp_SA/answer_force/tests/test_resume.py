from __future__ import annotations

import json

import pytest

from dp_SA.answer_force.run import _check_output


def test_resume_fingerprint_refuses_incompatible_input(tmp_path):
    config = {"fingerprint": "abc"}
    _check_output(tmp_path, config, resume=False)
    with pytest.raises(FileExistsError):
        _check_output(tmp_path, config, resume=False)
    with pytest.raises(ValueError, match="fingerprint"):
        _check_output(tmp_path, {"fingerprint": "different"}, resume=True)

