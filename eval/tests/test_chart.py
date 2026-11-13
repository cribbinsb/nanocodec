"""The bitrate-vs-speed chart: preset anchors, aggregation, Pareto front, outputs."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from nceval import cbr, chart


def point(name, bd, fps=None, scored=11, ours=False, missing=()):
    return chart.ChartPoint(family=name, label=name, ours=ours, bd_j=bd, bd_psnr=bd, bd_vmaf=bd,
                            iqr=(bd - 2, bd + 2) if bd is not None else None, scored=scored,
                            clips=11, missing_classes=missing, rate_invalid=0, refused=0,
                            fps=fps, speed=fps and fps / 10)


def test_preset_anchor_names_pick_the_program_and_preset(tmp_path):
    with patch.object(cbr, "find_tool", side_effect=lambda name, paths: Path("/" + name)):
        x264 = cbr.preset_anchor("x264-medium", None)
        svt = cbr.preset_anchor("svtav1-m8", None)
        with pytest.raises(ValueError):
            cbr.preset_anchor("x266-fast", None)
    assert x264.kind == "x264" and x264.anchor_preset == "medium"
    assert svt.kind == "svt" and svt.anchor_preset == "8" and svt.codec is cbr.Codec.AV1


def test_default_presets_keep_their_cache_identity(tmp_path):
    clip_file = tmp_path / "c.yuv"
    clip_file.write_bytes(b"\0" * 16)
    tool = tmp_path / "x264"
    tool.write_bytes(b"x264")
    bench = cbr.Bench(paths=None, frames=150, intra=300, work=tmp_path)
    clip = type("Clip", (), {"name": "c", "yuv": clip_file, "start_frame": 0})()
    legacy = cbr.Family("x264-veryfast", cbr.Codec.H264, tool, kind="x264")
    named = cbr.Family("x264-veryfast", cbr.Codec.H264, tool, kind="x264", preset="veryfast")
    medium = cbr.Family("x264-medium", cbr.Codec.H264, tool, kind="x264", preset="medium")
    key = lambda f: bench.identity(f, clip, "cbr600")  # noqa: E731
    assert key(legacy) == key(named) != key(medium)


def test_preset_reaches_the_command_line(tmp_path):
    clip = type("Clip", (), {"name": "c", "yuv": Path("/c.yuv"), "width": 1280,
                             "height": 720, "fps": 30.0})()
    medium = cbr.Family("x265-medium", cbr.Codec.HEVC, Path("/x265"), kind="x265",
                        preset="medium")
    cmd = medium.command(clip, 150, 300, Path("o.hevc"), None, kbps=600)
    assert cmd[cmd.index("--preset") + 1] == "medium" and "zerolatency" in cmd


def test_pareto_front_skips_flagged_and_dominated_points():
    fast, slow, dominated = point("a", 10, fps=100), point("b", -20, fps=10), \
        point("c", 15, fps=50)
    flagged = point("d", -40, fps=200, scored=5)
    assert chart.pareto([fast, slow, dominated, flagged]) == [fast, slow]
    partial = point("e", 0, scored=8, missing=("static",))
    assert not point("f", 0, scored=10).partial
    assert flagged.flagged and partial.partial and not partial.flagged


def test_per_clip_bd_j_needs_both_terms():
    assert chart.per_clip_j({"quality": -10.0, "quality_low": -20.0}) == -15.0
    assert chart.per_clip_j({"quality": -10.0, "quality_low": None}) is None


def test_speed_uses_the_paired_anchor_runs():
    timing = {"points": {"x264-medium": {"ms_per_frame": {"a": 40.0}},
                         "b": {"ms_per_frame": {"a": 20.0}, "anchor_ms_per_frame": {"a": 50.0}}}}
    timed = chart.with_speed([point("b", -5)], timing)[0]
    assert timed.speed == pytest.approx(2.5) and timed.fps == pytest.approx(50.0)


def test_outputs_list_every_point():
    points = [point("x264-medium", 0.0, fps=30), point("nanocodec-av1", -30, fps=20, ours=True),
              point("svtav1-m8", None, fps=5, scored=0)]
    front = chart.pareto(points)
    assert len(chart.csv_text(points, front).splitlines()) == 4
    table = chart.markdown_table(points, front)
    assert "FLAGGED" in table and "**nanocodec-av1**" in table
    svg = chart.svg_chart(points, front, "test")
    assert svg.startswith("<svg") and "nanocodec-av1" in svg and "Pareto" in svg


def test_x86_outputs_name_their_platform():
    points = [point("x264-medium", 0.0, fps=30), point("nanocodec-av1", -30, fps=20, ours=True)]
    x86 = [point("x264-medium", 0.0, fps=90), point("nanocodec-av1", -30, fps=120, ours=True)]
    assert chart.csv_text(x86, chart.pareto(x86), chart.X86).startswith(
        "family,label,ours,bd_j_pct,bd_psnr_yuv_pct,bd_vmaf_v1_pct,iqr_low_pct,iqr_high_pct,clips_scored,"
        "clips,fps_x86")
    table = chart.markdown_table(points, chart.pareto(points), x86)
    assert "fps (x86, 1 core)" in table and "| 120.0 | 12.00× |" in table
    svg = chart.svg_chart(x86, chart.pareto(x86), "test", chart.X86)
    assert "single-threaded encode speed: one x86-64 core (AMD Ryzen 9 5950X)<" in svg
    assert "core (NVIDIA Jetson Orin Nano)<" in chart.svg_chart(points, chart.pareto(points), "t")


def test_pairs_table_marks_partial_pairs():
    table = chart.pairs_table([("openh264", point("nanocodec-av1", -52.0, scored=10,
                                                     ours=True, missing=("static",)))])
    assert "| nanocodec-av1 | openh264 | -52.0%† |" in table


def test_labels_are_unambiguous_or_get_a_leader():
    # three markers in a row: a label right of A would sit beside B
    items = [(point("A", -40, ours=True), 100.0, 200.0), (point("B", -40, ours=True), 118.0, 200.0),
             (point("C", -40), 136.0, 200.0)]
    labels = chart.place_labels(items, bars=[])
    for p, cx, cy in items:
        tx, ty, _, leader = labels[p.family]
        nearest = min(items, key=lambda i: (i[1] - tx) ** 2 + (i[2] - ty) ** 2)
        assert leader or nearest[0].family == p.family or abs(tx - cx) <= 16


def test_labels_avoid_bars():
    p = point("A", -40, ours=True)
    bar = (108.0, 150.0, 300.0, 250.0)  # blocks everything right of the marker
    x, _, anchor, _ = chart.place_labels([(p, 100.0, 200.0)], bars=[bar])["A"]
    assert x <= 100.0 or anchor != "start"


def terms(value):
    """A clip's BD terms, all `value` (None: no reliable overlap)."""
    row = {term: value for term in chart.CHAINED_TERMS}
    row.update(quality=value, quality_low=value)
    return row


def test_compose_multiplies_bitrate_ratios():
    assert chart.compose(-20.0, -50.0) == pytest.approx(-60.0)
    assert chart.compose(10.0, -10.0) == pytest.approx(-1.0)
    assert chart.compose(0.0, -30.0) == pytest.approx(-30.0)
    assert chart.compose(None, -30.0) is None and chart.compose(-30.0, None) is None


def test_chained_row_composes_every_metric_then_bd_j():
    leg = {"class": "static", **terms(-20.0)}
    leg["bd_psnr_yuv"] = -40.0  # one metric differs: quality is the mean of the composed ones
    row = chart.chained_row(leg, {"class": "static", **terms(-50.0)})
    assert row["class"] == "static" and row["bd_psnr_yuv"] == pytest.approx(-70.0)
    assert row["low_psnr_yuv"] == pytest.approx(-60.0)
    assert row["quality"] == pytest.approx((-70.0 - 60.0 - 60.0) / 3)
    assert row["j"] == pytest.approx((row["quality"] + row["quality_low"]) / 2)
    assert chart.chained_row(leg, {"class": "static", **terms(None)})["j"] is None


def candidate(name, codec, overlap, own=True, legs=(-10.0, -20.0)):
    return chart.Candidate(name, codec, own, {"j": legs[0]}, {"j": legs[1]}, overlap)


def test_intermediate_prefers_lineage_then_side_then_overlap():
    av2 = cbr.Family("nanocodec-av2", cbr.Codec.AV2, Path("/nc"), own=True)
    hevc = candidate("nanocodec-hevc", cbr.Codec.HEVC, 0.9)
    svt = candidate("svtav1-m12", cbr.Codec.AV1, 0.8, own=False)
    av1 = candidate("nanocodec-av1", cbr.Codec.AV1, 0.6)
    vp9 = candidate("nanocodec-vp9", cbr.Codec.VP9, 0.7)
    assert chart.choose_intermediate(av2, [hevc, svt, av1, vp9]) is vp9  # best overlap of ours
    assert chart.choose_intermediate(av2, [hevc, svt, av1]) is av1  # our side over SVT
    assert chart.choose_intermediate(av2, [hevc, svt]) is svt  # lineage over overlap
    assert chart.choose_intermediate(av2, [hevc]) is hevc
    tie = candidate("nanocodec-av1@av1-fast", cbr.Codec.AV1, 0.6)
    assert chart.choose_intermediate(av2, [tie, av1]) is av1  # by name: deterministic
    unscored = candidate("nanocodec-vp9", cbr.Codec.VP9, 0.9, legs=(None, -20.0))
    assert chart.choose_intermediate(av2, [unscored]) is None


def test_chains_never_take_more_than_one_hop():
    """P reaches the anchor only via C -> B -> anchor (C has no direct BD): no chain.
    With a direct leg P -> B it chains through B."""
    clip = type("Clip", (), {"name": "c", "content_class": "static"})()
    fam = lambda name: cbr.Family(name, cbr.Codec.AV1, Path("/" + name))  # noqa: E731
    anchor, p, b, c = fam("x264-medium"), fam("P"), fam("B"), fam("C")
    curves = {(f.name, "c"): f.name for f in (anchor, p, b, c)}
    direct = {"x264-medium": {"c": {"j": 0.0, **terms(0.0)}},
              "P": {"c": {"j": None}}, "B": {"c": {"class": "static", "j": -30.0, **terms(-30.0)}},
              "C": {"c": {"j": None}}}

    def run(legs):
        compare_clip = lambda a, t: terms(legs.get((t, a)))  # noqa: E731
        with patch.object(chart.jreal, "compare_clip", side_effect=compare_clip), \
                patch.object(chart.jreal, "curve_overlap", return_value=0.8):
            return chart.chain_clip(curves, anchor.name, p, clip, [anchor, p, b, c], direct)

    assert run({("P", "C"): -10.0, ("C", "B"): -10.0}) is None
    link, row = run({("P", "C"): -10.0, ("P", "B"): -20.0})
    assert link.intermediate == "B" and link.j == pytest.approx(-44.0)
    assert row["j"] == pytest.approx(-44.0)


def test_chained_points_report_both_means_and_their_clips():
    chained = chart.ChainedClip("static_cam29", "nanocodec-av1", -10.0, -40.0, -46.0)
    p = chart.ChartPoint(**{**point("nanocodec-av2", -47.0, fps=20, ours=True).__dict__,
                            "chained": (chained,), "direct_bd_j": -44.0})
    front = chart.pareto([p])
    table = chart.markdown_table([p], front)
    assert "| -47.0%‡ | -44.0% |" in table and "‡ 1 clip chained" in table
    assert "| static_cam29 | nanocodec-av1 | -10.0% | -40.0% | -46.0% |" in chart.chained_table([p])
    assert chart.csv_text([p], front).splitlines()[1].endswith(
        ",1,-44.00,static_cam29:nanocodec-av1:-46.00")
    assert chart.chart_label(p) == "nanocodec-av2"  # marked in the table only


def test_polyline_boxes_cover_every_segment():
    boxes = chart.polyline_boxes([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)])
    assert all(-1 <= b[1] and b[3] <= 11 for b in boxes)
    for x, y in ((0, 0), (5, 0), (10, 5), (10, 10)):
        assert any(chart.box_distance(b, x, y) <= chart.POLYLINE_STEP / 2 for b in boxes)


def test_low_rungs_go_only_to_clips_some_curve_misses():
    clips = [type("Clip", (), {"name": n})() for n in ("a", "b")]
    fam = lambda name: cbr.Family(name, cbr.Codec.AV1, Path("/" + name))  # noqa: E731
    families = [fam("x264-medium"), fam("P"), fam("Q")]
    curves = {(f.name, c.name): (f.name, c.name) for f in families for c in clips}
    missing = {("Q", "b")}  # Q's curve on b has no reliable overlap

    def compare_clip(anchor, test):
        return terms(None if test in missing else -10.0)
    with patch.object(chart.jreal, "compare_clip", side_effect=compare_clip):
        assert [c.name for c in chart.low_rung_clips(curves, families, clips)] == ["b"]


def test_every_point_shares_one_clip_set():
    clips = [type("Clip", (), {"name": n, "content_class": k})()
             for n, k in (("a", "pan"), ("b", "pan"), ("c", "static"))]
    direct = {"P": {"a": {"j": -1.0}, "b": {"j": None}, "c": {"j": -2.0}},
              "Q": {"a": {"j": -3.0}, "b": {"j": -4.0}, "c": {"j": -5.0}}}
    assert [c.name for c in chart.common_clips(direct, ["P", "Q"], clips)] == ["a", "c"]
    # b's class keeps clip a, so b is never rescued by chaining
    assert chart.rescue_classes({}, "x264-medium", [], clips, [clips[0], clips[2]], direct) == {}


def test_a_vanishing_class_is_chained_only_for_every_point_at_once():
    clips = [type("Clip", (), {"name": n, "content_class": k})()
             for n, k in (("a", "pan"), ("s1", "static"), ("s2", "static"))]
    fam = lambda name: cbr.Family(name, cbr.Codec.AV1, Path("/" + name))  # noqa: E731
    families = [fam("x264-medium"), fam("P"), fam("Q")]
    direct = {"x264-medium": {n: {"j": 0.0} for n in ("a", "s1", "s2")},
              "P": {"a": {"j": -1.0}, "s1": {"j": None}, "s2": {"j": None}},
              "Q": {"a": {"j": -1.0}, "s1": {"j": -2.0}, "s2": {"j": None}}}
    # P chains on s1 and s2, Q on nothing: s1 (only P unscored) is kept, s2 is not
    links = {("P", "s1"): "P@s1", ("P", "s2"): "P@s2"}
    with patch.object(chart, "chain_clip",
                      side_effect=lambda c, a, f, clip, fs, d: links.get((f.name, clip.name))):
        rescued = chart.rescue_classes({}, "x264-medium", families, clips, [clips[0]], direct)
    assert rescued == {"s1": {"P": "P@s1"}}


def test_outputs_carry_the_updated_line():
    assert chart.updated_line({"commit": "abc", "build": "abc; H.264 def"}, "2026-11-11") == \
        "Updated 2026-11-11, build abc; H.264 def"
    assert chart.updated_line({"commit": "abc"}, "2026-11-11") == "Updated 2026-11-11, build abc"
    assert chart.updated_line({}, "2026-11-11") == "Updated 2026-11-11"
    points = [point("x264-medium", 0.0, fps=30), point("nanocodec-av1", -30, fps=20, ours=True)]
    svg = chart.svg_chart(points, chart.pareto(points), "test", stamp="Updated 2026-11-11, build abc")
    assert 'font-size="10" text-anchor="end"' in svg and ">Updated 2026-11-11, build abc<" in svg


def test_nvenc_families_are_low_delay_cbr_and_keyed_per_encoder(tmp_path):
    with patch.object(cbr, "find_tool", side_effect=lambda name, paths: tmp_path / name):
        (tmp_path / "ffmpeg").write_bytes(b"ffmpeg")
        h264, hevc = cbr.preset_anchor("nvenc-h264", None), cbr.preset_anchor("nvenc-hevc", None)
        best = cbr.preset_anchor("nvenc-av1-p7hq", None)
    clip = type("Clip", (), {"name": "c", "yuv": tmp_path / "ffmpeg", "width": 1280,
                             "height": 720, "fps": 30.0, "start_frame": 0})()
    cmd = h264.command(clip, 150, 300, Path("o.264"), None, kbps=600)
    assert h264.hardware and not h264.cqp
    assert cmd[cmd.index("-c:v") + 1] == "h264_nvenc" and cmd[cmd.index("-tune") + 1] == "ll"
    assert cmd[cmd.index("-bf") + 1] == "0" and cmd[cmd.index("-rc") + 1] == "cbr"
    assert cmd[cmd.index("-bufsize") + 1] == "600k" and cmd[cmd.index("-g") + 1] == "300"
    av1 = best.command(clip, 150, 300, Path("o.ivf"), None, kbps=600)
    assert av1[av1.index("-preset") + 1] == "p7" and av1[-3:-1] == ["ivf", "-y"]
    with pytest.raises(ValueError):
        h264.command(clip, 150, 300, Path("o.264"), None, qp=30)
    bench = cbr.Bench(paths=None, frames=150, intra=300, work=tmp_path)
    assert bench.identity(h264, clip, "cbr600") != bench.identity(hevc, clip, "cbr600")


def test_hardware_points_sit_at_their_fps_or_past_the_axis_when_faster():
    timed = [point("a", 10, fps=100), point("b", -20, fps=10)]
    speeds = chart.hardware_speeds({"points": {
        "nvenc-hevc": {"aggregate_fps": {"1": 500.0, "4": 600.0, "8": 550.0},
                       "encode_fps": {"x": 400.0, "y": 900.0}},
        "nvenc-av1": {"aggregate_fps": {"1": 50.0}, "encode_fps": {"x": 50.0},
                      "pipeline_fps": {"x": 20.0}}}})
    assert speeds["nvenc-hevc"] == chart.HardwareSpeed(600.0, 4, pytest.approx(600.0), None)
    gpu = chart.with_hardware_fps([point("nvenc-hevc", -30), point("nvenc-av1", -25)], speeds)
    assert gpu[0].fps == 600.0
    table = chart.hardware_table(gpu, speeds, "GPU")
    assert "| 600 (4) | 600 | n/a |" in table and "| 50 (1) | 50 | 20 |" in table
    front = chart.pareto(timed)
    note = chart.hardware_note(speeds)
    assert "1/4 parallel streams" in note and "frames in GPU memory" in note
    svg = chart.svg_chart(timed, front, "", hardware=gpu, hardware_footnote=note)
    plain = chart.svg_chart(timed, front, "")
    frame = chart.Frame(0, 1, 0, 1)
    # nvenc-hevc (600 fps) is past the axis (140 fps): a chevron in the gutter
    assert "nvenc-hevc · 600 fps" in svg and "off scale" in svg and "off scale" not in plain
    assert f'width="{frame.left + frame.width + chart.HARDWARE_GUTTER}"' in svg
    # nvenc-av1 (50 fps) is on the axis: a labelled hexagon, not in the Pareto front
    assert ">nvenc-av1</text>" in svg and "NVENC (RTX 5090 GPU)" in svg
    assert note in svg
    on_axis_only = chart.svg_chart(timed, front, "", hardware=gpu[1:])
    assert "off scale" not in on_axis_only
    assert f'width="{frame.left + frame.width + 30}"' in on_axis_only
