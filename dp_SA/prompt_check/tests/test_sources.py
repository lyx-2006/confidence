from dp_SA.prompt_check.sources import audit_frozen_sources


def test_frozen_source_counts_and_probe_non_leakage():
    result=audit_frozen_sources()
    assert result["status"]=="passed"
    assert result["counts"]["audit"]==230
    assert result["counts"]["candidates"]==1625
    assert result["counts"]["cells"]==8792
    assert result["counts"]["test"]==174
    assert result["audit_used_for_fit"] is False
    assert len(result["probe_rows"])==52
