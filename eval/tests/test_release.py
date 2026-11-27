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
    args = argparse.Namespace(**{
        "version": "9.9.9", "commit": None, "start": None, "only": None, "pipe_clean": False,
        "real": True, "over_budget": False, "yes_publish": False, "no_chain_retry": False,
        "local_charts": False,
        **flags})
    with patch.object(release, "RELEASE_ROOT", tmp_path):
        return release.Release(args)


def test_every_chart_chain_is_priced_and_every_clip_set_chart_has_a_chain():
    """Each site chart view's clip set runs in a priced chain (2026-11-18: VC-1K's view
    was in no release chain)."""
    priced = release.chain_minutes()
    for chain in release.clip_chains():
        for clipset in chain:
            for command in ("", "-psnr"):
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
    with patch.dict(release.CHART_CHAIN_MINUTES, minutes), \
         patch.object(release, "charts_preflight", return_value=[]):
        run.plan(list(release.STAGES))
        commit, pgo = release.STAGE_MINUTES["commit"], release.STAGE_MINUTES["pgo"]
        charts = max(commit + 50, commit + pgo + 40) + release.CHARTS_LAND_MINUTES
        assert run.chart_finish(commit, commit + pgo) == charts
        minutes[quality] = 200  # a 200-minute chain puts the plan over the limit
        with patch.dict(release.CHART_CHAIN_MINUTES, minutes), pytest.raises(SystemExit):
            run.plan(list(release.STAGES))


def test_a_failed_charts_preflight_stops_the_plan(tmp_path):
    run = make_release(tmp_path)
    with patch.object(release, "charts_preflight", return_value=["missing: x"]), \
         pytest.raises(SystemExit, match="missing: x"):
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


def test_the_charts_rented_sessions_fit_the_home_regions_spot_quota():
    """The main chart and every clip-set chain measure at once: their fleets' boxes fit
    eu-north-1's quota beside the build box's share, so none waits for room."""
    chains = len(release.clip_chains())
    boxes = release.MAIN_CLOUD_BOXES + chains * release.CLIPSET_CLOUD_BOXES
    assert boxes * release.FLEET_BOX_VCPUS <= release.HOME_SPOT_VCPUS - release.BUILD_BOX_VCPUS


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


def test_side_by_side_commands_all_run_in_the_directory_given(tmp_path):
    import subprocess
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "marker").write_text("")
    line = release.all_of(["test -f marker"] * 3, cwd="eval")
    assert subprocess.run(["bash", "-c", line], cwd=tmp_path).returncode == 0


def test_the_jetson_lens_times_two_points_of_every_encoders_scale():
    from nceval.ctc.conditions import POINTS
    chosen = {int(p) for p in release.jetson_lens_points().split(",")}
    for scale, points in POINTS.items():
        assert len(chosen & set(points)) == 2, scale


def test_the_charts_preflight_names_each_missing_tool_and_a_full_work_budget(tmp_path):
    from nceval import diskguard
    harness, sources = tmp_path / "bin" / "openh264-harness", tmp_path / "src"
    sources.mkdir()  # present but empty: the lens finds no sequence in it
    work = tmp_path / "work"
    problems = release.charts_preflight(work, [harness, sources])
    assert problems == [f"missing (or empty): {harness}", f"missing (or empty): {sources}"]
    harness.parent.mkdir()
    harness.write_text("")
    (sources / "a.yuv").write_text("")
    assert release.charts_preflight(work, [harness, sources]) == []
    full = diskguard.WORK_BUDGET_BYTES - (release.CHARTS_WORK_HEADROOM_GIB - 1) * diskguard.GIB
    with patch.object(diskguard, "shared_work_bytes", return_value=full), \
         patch.object(diskguard, "measure_work_bytes", return_value=full):
        [problem] = release.charts_preflight(work, [harness, sources])
    assert "24 GiB left" in problem


def test_the_charts_preflight_measures_afresh_before_it_refuses(tmp_path):
    """2026-11-20: the shared size cache still said 75 GiB after a prune; the pre-flight
    refused twice on it."""
    import json
    import time
    from nceval import diskguard
    work = tmp_path / "work"
    work.mkdir()
    (work / "kept.bin").write_bytes(b"x" * 1000)
    stale = diskguard.WORK_BUDGET_BYTES - 10 * diskguard.GIB  # 10 GiB left, says the cache
    (work / diskguard.USAGE_CACHE).write_text(json.dumps({"bytes": stale, "time": time.time()}))
    assert release.charts_preflight(work, []) == []
    assert json.loads((work / diskguard.USAGE_CACHE).read_text())["bytes"] < 2000  # shared


def test_the_charts_headroom_is_that_of_the_chains_still_to_write(tmp_path):
    """Only renders, merges and the site left: no headroom needed (2026-11-20: a resume
    with only those left was refused twice for 25 GiB)."""
    run = make_release(tmp_path)
    assert run.charts_work_gib() == release.CHARTS_WORK_HEADROOM_GIB  # a first run
    tree = tmp_path / "charts-tree"
    (tree / ".git").mkdir(parents=True)
    commands = [("lens-x86-run", []), ("lens-x86-merge", []), ("cbr-zen3", []),
                ("heldout-vc1k_2026-11-18-psnr", [])]
    minutes = release.chain_minutes()
    with patch.object(release, "CHART_TREE", tree), \
         patch.object(release.Release, "chart_commands", lambda _self, _tree: commands):
        run.state["charts_progress"] = {"lens-x86-run": 1, "cbr-zen3": 1}
        assert run.charts_work_gib() == 0
        run.state["charts_progress"] = {"lens-x86-run": 1}
        assert run.charts_work_gib() == pytest.approx(
            release.CHARTS_WORK_HEADROOM_GIB * minutes["cbr-zen3"] / sum(minutes.values()))


def test_a_failed_chart_command_is_announced_at_once_and_retried_once(tmp_path, capsys):
    run = make_release(tmp_path)
    run.state["release_commit"] = "0" * 40
    calls = []

    def flaky(command, cwd, log, env):
        calls.append(log.name)
        if len(calls) == 1:
            raise SystemExit(f"failed (1): see {log}")

    with patch.object(release, "run", flaky), patch.object(release, "CHAIN_RETRY_WAIT_S", 0):
        run.run_chart_command("heldout-x", "heldout-x-zero-line-score", ["false"], tmp_path)
    out = capsys.readouterr()
    log = run.log("charts-heldout-x-zero-line-score")
    assert f"== charts chain heldout-x FAILED: {log}" in out.out
    assert f"== charts chain heldout-x FAILED: {log}" in out.err  # stdout captured apart
    assert calls == ["charts-heldout-x-zero-line-score.log",
                     "charts-heldout-x-zero-line-score-retry.log"]


def test_a_chart_command_that_fails_its_retry_fails_its_chain_alone(tmp_path):
    run = make_release(tmp_path)
    run.state["release_commit"] = "0" * 40

    def fails(command, cwd, log, env):
        raise SystemExit(f"failed (1): see {log}")

    def passes(command, cwd, log, env):
        pass

    timings = {}
    with patch.object(release, "CHAIN_RETRY_WAIT_S", 0), \
         patch.object(release.Release, "save", lambda _self: None):
        with patch.object(release, "run", fails), pytest.raises(SystemExit):
            run.run_chart_chain("heldout-x", [("heldout-x-a", []), ("heldout-x-b", [])],
                                tmp_path, timings)
        assert timings == {}
        with patch.object(release, "run", passes):
            run.run_chart_chain("heldout-y", [("heldout-y-a", [])], tmp_path, timings)
        assert list(timings) == ["heldout-y-a"]
    no_retry = make_release(tmp_path, no_chain_retry=True)
    no_retry.state["release_commit"] = "0" * 40
    tries = []
    with patch.object(release, "run", lambda *a, **k: tries.append(1) or fails(*a, **k)), \
         pytest.raises(SystemExit):
        no_retry.run_chart_command("c", "c-a", [], tmp_path)
    assert len(tries) == 1


def test_crates_run_the_release_trees_own_script(tmp_path):
    """2026-11-20: the launch tree's script published the launch tree (0.4.0)."""
    run = make_release(tmp_path, yes_publish=True)
    run.tree = tmp_path / "release-tree"
    run.tree.mkdir()
    (run.tree / "Cargo.toml").write_text('[workspace.package]\nversion = "9.9.9"\n')
    commands = []
    with patch.object(release, "run", lambda command, **_: commands.append(command)):
        run.stage_crates()
    assert commands == [["bash", str(run.tree / "tools/release/publish_crates.sh"),
                         "--yes-publish"]]
    (run.tree / "Cargo.toml").write_text('[workspace.package]\nversion = "9.9.8"\n')
    with pytest.raises(SystemExit, match="not 9.9.9"):
        run.stage_crates()


def test_the_launch_tree_must_contain_the_release_base():
    head = release.git("rev-parse", "HEAD")
    release.check_launch_tree(release.git("rev-parse", "HEAD~1"))
    child = release.git("commit-tree", f"{head}^{{tree}}", "-p", head, "-m", "later")
    with pytest.raises(SystemExit, match="contains the release base"):
        release.check_launch_tree(child)


def test_the_github_stage_keeps_a_tag_and_release_already_published():
    """2026-11-20: a resume tried to push a second v0.5.0 tag object (refused)."""
    assert release.github_actions(None, "t1", False) == ["push", "release"]
    assert release.github_actions("t1", "t1", False) == ["release"]
    assert release.github_actions("t1", "t1", True) == []
    with pytest.raises(SystemExit, match="another tree"):
        release.github_actions("t0", "t1", True)


def test_the_charts_preflight_checks_the_paths_the_chains_use():
    names = {path.name for path in release.chart_tool_paths()}
    assert {"openh264-harness", "src", "bin"} <= names
