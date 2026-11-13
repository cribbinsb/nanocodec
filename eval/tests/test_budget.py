"""Wall-time budget: estimates, refusal with cuts, logged override."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nceval import budget


def test_cost_scales_with_pixels_and_frames_and_rescoring_is_cheaper():
    full = budget.point_cpu_seconds(1280, 720, 150)
    assert budget.point_cpu_seconds(1280, 720, 300) == pytest.approx(2 * full)
    assert budget.point_cpu_seconds(1280, 720, 150, rescore=True) < full


def test_over_budget_is_refused_with_cuts_unless_forced(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("NC_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(budget, "available_slots", lambda: 4)
    assert budget.check("small", 60 * 4 * 5, "cut", False) == pytest.approx(5)
    with pytest.raises(budget.OverBudget, match="estimated at 40 min.*Cut it: fewer clips"):
        budget.check("big", 60 * 4 * 40, "fewer clips", False)
    assert budget.check("big", 60 * 4 * 40, "fewer clips", True) == pytest.approx(40)
    assert "big: 40 min forced" in (tmp_path / "run" / "over-budget.log").read_text()


def test_over_budget_sweep_runs_in_balanced_parts_not_refused(monkeypatch):
    from types import SimpleNamespace
    from nceval import sweep as sweep_module
    from nceval.sweep import Sweep, SweepConfig
    clips = [SimpleNamespace(name=f"c{i}", content_class=("static", "pan")[i % 2]) for i in range(11)]
    jobs = [(SimpleNamespace(name="enc"), clip) for clip in clips]
    runner = Sweep.__new__(Sweep)
    monkeypatch.setattr(runner, "_curve_cpu", lambda encoder, clip, cfg: 60.0, raising=False)
    monkeypatch.setattr(sweep_module, "estimate_minutes", lambda cpu: cpu / 60.0 * 49 / 11)
    monkeypatch.setattr(budget, "available_slots", lambda: 1.0)
    parts = runner._budget_parts(jobs, SweepConfig(10))
    assert len(parts) == 3 and sorted(len(p) for p in parts) == [3, 4, 4]
    assert sorted(id(j) for p in parts for j in p) == sorted(id(j) for j in jobs)
