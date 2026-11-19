"""tools/release/release.py: the plan's stage overlap and the chart chains (no stage runs)."""
import argparse
import importlib.util
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

TOOL = Path(__file__).resolve().parents[2] / "tools" / "release" / "release.py"
spec = importlib.util.spec_from_file_location("release", TOOL)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def make_release(tmp_path, **flags) -> "release.Release":
    args = argparse.Namespace(version="9.9.9", commit=None, start=None, only=None,
                              pipe_clean=False, real=True, over_budget=False,
                              yes_publish=False, **flags)
    with patch.object(release, "RELEASE_ROOT", tmp_path):
        return release.Release(args)


def test_every_chart_chain_is_priced_and_every_clip_set_chart_has_a_chain():
    """Each site chart view's clip set runs in a priced chain (2026-11-18: VC-1K's view
    was in no release chain)."""
    priced = release.chain_minutes()
    for chain in release.clip_chains():
        for clipset in chain:
            for command in ("", "-zero-line", "-zero-line-score", "-psnr"):
                assert release.chart_chain(f"heldout-{clipset.stem}{command}") in priced
    stems = {clipset.stem for chain in release.clip_chains() for clipset in chain}
    assert "vc1k_2026-11-18" in stems


def test_an_unpriced_chain_stops_the_plan():
    with patch.dict(release.CHART_CHAIN_MINUTES, clear=True), pytest.raises(SystemExit):
        release.chain_minutes()


def test_quality_chains_overlap_the_pgo_builds_and_uploads_overlap_the_charts(tmp_path):
    """The plan's end is the longest path through STAGE_NEEDS, not the stages' sum."""
    minutes = {chain: 1 for chain in release.chain_minutes()}
    quality = next(chain for chain in minutes if chain.startswith(release.QUALITY_CHAIN_PREFIX))
    minutes.update({quality: 50, "lens-jetson": 40})
    run = make_release(tmp_path)
    with patch.dict(release.CHART_CHAIN_MINUTES, minutes):
        run.plan(list(release.STAGES))
        commit, pgo = release.STAGE_MINUTES["commit"], release.STAGE_MINUTES["pgo"]
        charts = max(commit + 50, commit + pgo + 40) + release.CHARTS_LAND_MINUTES
        assert run.chart_finish(commit, commit + pgo) == charts
        minutes[quality] = 200  # a 200-minute chain puts the plan over the limit
        with patch.dict(release.CHART_CHAIN_MINUTES, minutes), pytest.raises(SystemExit):
            run.plan(list(release.STAGES))


def test_a_stage_waits_for_what_it_needs_and_a_failure_stops_its_dependents(tmp_path):
    run = make_release(tmp_path)
    run.running = {"pgo": threading.Event()}
    run.failed.add("pgo")
    run.running["pgo"].set()
    with pytest.raises(SystemExit):
        run.wait_for("pgo")
    with pytest.raises(SystemExit):  # neither in this run nor recorded
        run.wait_for("crates")
    run.state["stages"]["crates"] = {"minutes": 1}
    run.wait_for("crates")


def test_the_clip_set_scoring_sessions_fit_each_regions_spot_quota():
    """Both clip-set chains score at once: at most two c7a.metal-48xl (512 spot vCPU)
    per region, or a session waits for the quota and fails."""
    from collections import Counter
    regions = Counter(region for chain in release.clip_chains() for region in
                      release.CLIPSET_CLOUD_REGIONS.get(
                          chain[0].stem, release.HELDOUT_CLOUD_REGIONS).split(","))
    assert max(regions.values()) <= 2


def test_every_fixed_qp_sample_view_is_re_measured_in_a_priced_chain():
    """The Fixed QP views on clip-set samples come from the site's own list too
    (2026-11-18: VC-1K's would have stayed on 0.3.0's build)."""
    priced = release.chain_minutes()
    ids = {view.id for view in release.qp_views()}
    assert "qp-vc1k" in ids
    for view in release.qp_views():
        for command in ("gap", "site", "timing-run", "timing-merge"):
            name = f"{view.id}-{command}"
            assert release.chart_chain(name) == view.id and view.id in priced
            assert not view.id.startswith(release.QUALITY_CHAIN_PREFIX)  # waits for pgo


def test_the_lens_quality_chain_is_priced_and_waits_for_pgo():
    assert release.chart_chain("lens-quality-box") == release.LENS_QUALITY_CHAIN
    assert release.chart_chain("lens-x86-run") == "lens-x86"
    assert release.LENS_QUALITY_CHAIN in release.chain_minutes()


def test_side_by_side_commands_fail_if_any_fails():
    import subprocess
    line = release.all_of(["true", "false", "true"])
    assert subprocess.run(["bash", "-c", line]).returncode == 1
    assert subprocess.run(["bash", "-c", release.all_of(["true", "true"])]).returncode == 0
