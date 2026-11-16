"""The lever ledger: rows from tool results, dedup, CPU join, staleness and replay."""

from nceval import levers


def adopt_result(dcpu=None, k=0.6):
    stage = {"stage": "final", "frames": 150,
             "summary": {"bd_psnr_yuv": {"balanced": -1.0}, "bd_vmaf_v1": {"balanced": -2.0},
                         "bd_ms_ssim": {"balanced": -0.5}},
             "verdict": {"dcpu_pct": dcpu, "k_cpu": k}}
    shift = k * (dcpu or 0.0)
    return {"status": "done", "mode": "real", "codec": "hevc", "binary": "/nonexistent/nc",
            "anchor_binary": "/nonexistent/nc", "params": ["hevc.inter_rounding=100"],
            "anchor_params": [], "levels": {"final": {"decision": "better", "n": 27,
                                                      "mean": -0.8 + shift, "lo": -1.2 + shift,
                                                      "hi": -0.4 + shift, "guards": []}},
            "draw": {"rotation": "2026-11-15", "frames": 150}, "stages": [stage],
            "build_info": {"candidate": {"commit": "abc"}}}


def test_screen_rows_keep_quality_j_without_the_cpu_term():
    [row] = levers.screen_rows(adopt_result(dcpu=2.0), "x.json")
    assert row["level"] == "L3" and row["overrides"] == {"hevc.inter_rounding": 100.0}
    assert abs(row["j"] + 0.8) < 1e-9 and abs(row["hi"] + 0.4) < 1e-9
    assert row["x86_pct"] == 2.0 and row["psnr"] == -1.0 and row["commit"] == "abc"


def test_append_dedups_and_cpu_joins_from_a_cost_row(tmp_path):
    [row] = levers.screen_rows(adopt_result(), "x.json")
    cost = levers.cost_rows("hevc", "medium", {"r100": ({"hevc.inter_rounding": 100.0}, 3.0)},
                            levers.CALLGRIND_P, "abc", "c.json")
    path = tmp_path / "l.jsonl"
    assert levers.append([row, row] + cost, path) == 2
    assert levers.append([row], path) == 0
    joined = [r for r in levers.resolved(levers.read_rows(path)) if r["j"] is not None]
    assert joined[0]["x86_pct"] == 3.0 and "joined" in joined[0]["x86_how"]


def test_staleness_and_replay_verdicts():
    [row] = levers.screen_rows(adopt_result(), "x.json")
    row = {**row, "base_values": {"hevc.inter_rounding": 128.0}, "x86_pct": 2.0}
    assert levers.base_status(row, {"hevc.inter_rounding": 128.0}) == "current"
    assert levers.base_status(row, {"hevc.inter_rounding": 120.0}) == "stale"
    assert levers.base_status(row, {"hevc.inter_rounding": 100.0}) == "in-base"
    assert levers.decide(row, 0.03) == "better"  # -0.8 + 0.06
    assert levers.decide(row, 0.6) == "ambiguous"  # -0.8 + 1.2 = +0.4, lo -0.0


def test_lens_rows_join_the_cpu_of_a_timing_session(tmp_path):
    """A lens comparison (Weighted and J in BD points against the base) takes its CPU from
    the cost row a timing summary gave the same change."""
    change = {"av1.cdef_local_search_period": 8.0}
    lens = levers.lens_row("av1", "faster", {}, change, "cdef8", "aom-ld", 12, 1.5, 0.9,
                           "abc", "results")
    summary = {"nanocodec-av1@faster": {"repeat-1": {"fps": 300.0}, "repeat-2": {"fps": 300.0}},
               "nanocodec-av1@cdef8": {"repeat-1": {"fps": 330.0}, "repeat-2": {"fps": 270.0}}}
    costs = levers.timing_cost_rows(summary, "av1", "faster", "nanocodec-av1@faster",
                                    {"nanocodec-av1@cdef8": change}, "c8a", "abc", "s.json",
                                    "lens")
    path = tmp_path / "l.jsonl"
    levers.append([lens] + costs, path)
    [row] = [r for r in levers.resolved(levers.read_rows(path)) if r["j"] is not None]
    assert row["level"] == "lens" and row["rate"] == "lens" and row["weighted"] == 1.5
    assert abs(row["x86_pct"]) < 1e-9 and "joined" in row["x86_how"]  # mean fps 300


def preset_final_check(old_dir, new_dir):
    stage = {"stage": "final", "frames": 600, "summary": {}, "verdict": {}}
    result = {"levels": {"final": {"decision": "better", "n": 28, "mean": -1.0, "lo": -1.5,
                                   "hi": -0.5, "guards": []}},
              "draw": {"rotation": "2026-11-16", "frames": 600}, "stages": [stage]}
    return {"old": "/x/old/nanocodec", "new": "/x/new/nanocodec", "old_commit": "abc",
            "new_commit": "abc", "preset": "fast", "codecs": {"av1": result},
            "presets": {"old": str(old_dir), "new": str(new_dir)}}


def test_preset_only_final_checks_on_one_commit_keep_distinct_ids(tmp_path):
    def presets(name, text):
        directory = tmp_path / name
        (directory / "av1").mkdir(parents=True)
        (directory / "av1" / "fast.toml").write_text(text)
        return directory
    old = presets("old", "quant.trellis = 0\n")
    a = presets("a", "quant.trellis = 1\n")
    b = presets("b", "quant.trellis = 1\neffort.regular_fraction = 0\n")
    [row_a] = levers.final_check_rows(preset_final_check(old, a), "a.json")
    [row_b] = levers.final_check_rows(preset_final_check(old, b), "b.json")
    assert row_a["overrides"] == {"quant.trellis": 1.0}
    assert row_b["overrides"] == {"quant.trellis": 1.0, "effort.regular_fraction": 0.0}
    assert row_a["id"] != row_b["id"]
    assert levers.append([row_a, row_b], tmp_path / "l.jsonl") == 2
    # A setting only removed still changes the id.
    [row_c] = levers.final_check_rows(preset_final_check(old, presets("c", "")), "c.json")
    assert row_c["overrides"] == {} and row_c["id"] != row_a["id"]


def test_binary_final_checks_are_named_by_build_and_path():
    report = preset_final_check("/x", "/x")
    del report["presets"], report["preset"]
    report.update(old_commit="", new_commit="", old="/w/run/head/nanocodec",
                  build_info={"old": {"commit": "cb6a3b5fa070", "dirty": False},
                              "new": {"commit": "cb6a3b5fa070", "dirty": True}})
    [row_a] = levers.final_check_rows(dict(report, new="/w/run/a/nanocodec"), "a.json")
    [row_b] = levers.final_check_rows(dict(report, new="/w/run/b/nanocodec"), "b.json")
    assert row_a["change"].startswith("build cb6a3b5fa@") and "cb6a3b5fa-dirty@" in row_a["change"]
    assert row_a["id"] != row_b["id"]
