"""nceval.baseline_state: a missing, failed or stale central rebaseline is never silent."""

from nceval import baseline_state


def test_only_a_completed_rebaseline_under_todays_conditions_is_quiet(tmp_path, monkeypatch):
    monkeypatch.setenv("NC_DATA_ROOT", str(tmp_path))
    assert "no rebaseline" in baseline_state.notice()
    baseline_state.record("08c752f1f2ee", baseline_state.FAILED, "aws_measure exit 1")
    assert "failed: aws_measure exit 1" in baseline_state.notice()
    baseline_state.record("08c752f1f2ee", baseline_state.OK)
    assert baseline_state.notice() is None
    monkeypatch.setattr(baseline_state, "conditions_tag", lambda: "conditions=other")
    assert "other benchmark conditions" in baseline_state.notice()
