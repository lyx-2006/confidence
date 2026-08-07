from __future__ import annotations

from pathlib import Path

from layer_metacognition.probe.common import probe_output_dir
from layer_metacognition.probe.generate_unimodal_labels import (
    DEFAULT_DATASET_PATH,
    _parser as label_parser,
)


def test_probe_output_dir_default_and_override(tmp_path: Path, monkeypatch) -> None:
    experiment = tmp_path / "experiment"
    assert probe_output_dir(experiment) == experiment.resolve() / "probe"

    monkeypatch.chdir(tmp_path)
    assert probe_output_dir(experiment, "custom-probe") == (
        tmp_path / "custom-probe"
    ).resolve()


def test_label_generation_defaults_to_merged_dataset() -> None:
    parsed = label_parser().parse_args(
        ["--experiment-dir", "experiment", "--model-path", "model"]
    )
    assert Path(parsed.dataset).resolve() == DEFAULT_DATASET_PATH.resolve()
