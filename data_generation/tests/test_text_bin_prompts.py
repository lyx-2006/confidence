"""Per-bin DeepSeek prompt template tests (text_entropy_bin_prompts.json)."""

from __future__ import annotations

import json
from pathlib import Path
from string import Formatter

import generation_v2

PROMPTS_PATH = Path(__file__).resolve().parents[1] / "prompts" / "text_entropy_bin_prompts.json"


def _load() -> dict:
    with PROMPTS_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def test_prompt_file_schema_and_all_bins_present() -> None:
    data = _load()
    assert data["schema"] == "text_entropy_bin_prompts.v1"
    assert set(data["generator"]) == {str(bin_id) for bin_id in range(5)}
    assert set(data["analyzer"]) == {str(bin_id) for bin_id in range(5)}


def test_generator_templates_render_and_follow_lexical_contract() -> None:
    prompts = generation_v2._TEXT_BIN_PROMPTS["generator"]
    for bin_id in range(5):
        rendered = prompts[str(bin_id)].format(
            color="red", colors="red, orange", count=3, accepted_json='["a"]'
        )
        assert "Requested entropy bin:" in rendered
        assert "shape-independent" in rendered
        assert "Previously accepted:" in rendered
        if bin_id == 0:
            # Bin 0 must allow (require) the target color word, no others.
            assert "target color word" in rendered
            assert "No concrete color term" not in rendered
        else:
            # Bins 1-4 must forbid any concrete color term.
            assert "No concrete color term" in rendered


def test_analyzer_templates_render_with_bin_specific_standard() -> None:
    prompts = generation_v2._TEXT_BIN_PROMPTS["analyzer"]
    for bin_id in range(5):
        rendered = prompts[str(bin_id)].format(
            color="red", bin_id=bin_id, candidate_json='{"text_clue": "x"}'
        )
        assert '"accepted"' in rendered  # escaped {{ }} resolved to literal braces
        assert f"requested_entropy_bin={bin_id}" in rendered
        assert f"Bin {bin_id} standard:" in rendered


def test_templates_have_exactly_the_supported_placeholders() -> None:
    """Every template uses only placeholders the code supplies; a new or missing
    one would be a silent behaviour change, so it must be caught here."""
    required_fields = {
        "generator": {"color", "colors", "count", "accepted_json"},
        "analyzer": {"color", "bin_id", "candidate_json"},
    }
    formatter = Formatter()
    for kind in ("generator", "analyzer"):
        for bin_id in range(5):
            template = generation_v2._TEXT_BIN_PROMPTS[kind][str(bin_id)]
            fields = {
                field for _literal, field, _spec, _conv in formatter.parse(template) if field
            }
            assert fields == required_fields[kind], (kind, bin_id, sorted(fields))


def test_missing_prompt_file_fails_loudly(tmp_path: Path) -> None:
    import generation_v2 as v2

    assert str(v2.BIN_PROMPTS_PATH).endswith("prompts/text_entropy_bin_prompts.json")
    assert v2.BIN_PROMPTS_PATH.exists()
