"""The SVC chart: libaom's temporal-unit framing and the reference encoders' commands."""

from __future__ import annotations

from pathlib import Path

from nceval import svc_chart, svcbase
from nceval.clips import Clip


def test_merge_temporal_units_joins_layers_of_one_pts(tmp_path: Path):
    path = tmp_path / "s.ivf"
    header = b"DKIF" + bytes(28)
    frames = [(0, b"a"), (0, b"bb"), (1, b"c"), (1, b"dd"), (2, b"e")]
    svcbase.write_ivf(path, header, frames)
    svc_chart.merge_temporal_units(path)
    assert svcbase.ivf_frames(path.read_bytes()) == [(0, b"abb"), (1, b"cdd"), (2, b"e")]
    assert int.from_bytes(path.read_bytes()[24:28], "little") == 3


def test_aom_l2t3_layer_rates_end_at_each_layer_target(tmp_path: Path):
    clip = Clip(name="c", yuv=tmp_path / "c.yuv", width=1280, height=720, fps=30.0, frames=150,
                source_kbps=10000, content_class="pan", split="val")
    bench = svc_chart.SvcBench.__new__(svc_chart.SvcBench)
    bench.paths, bench.frames, bench.intra = None, 150, 300
    bench.aom_tool = lambda: Path("svc_encoder_rtc")
    l2t3 = next(p for p in svc_chart.POINTS if p.kind == "aom" and p.temporal_layers == 3)
    cmd = bench.aom_command(l2t3, clip, 600, tmp_path / "o.ivf")
    rates = [int(r) for r in cmd[cmd.index("-bl") + 1].split(",")]
    base, full = svcbase.split_rate(600)
    assert rates[2] == base and rates[5] == full
    assert cmd[cmd.index("-b") + 1] == "600" and cmd[-1] == "-"
    assert f"--max-q={svc_chart.AOM_MAX_Q}" in cmd and "--test-decode=0" in cmd


def test_vpx_l2t1_command_splits_the_rate_by_spatial_layer(tmp_path: Path):
    clip = Clip(name="c", yuv=tmp_path / "c.yuv", width=1280, height=720, fps=29.97, frames=150,
                source_kbps=10000, content_class="pan", split="val")
    bench = svc_chart.SvcBench.__new__(svc_chart.SvcBench)
    bench.paths, bench.frames, bench.intra = None, 150, 300
    bench.vpx_tool = lambda: Path("vp9_spatial_svc_encoder")
    point = next(p for p in svc_chart.POINTS if p.kind == "vpx")
    cmd = bench.vpx_command(point, clip, 600, tmp_path / "o.ivf")
    base, full = svcbase.split_rate(600)
    assert cmd[cmd.index("-bl") + 1] == f"{base},{full}" and cmd[cmd.index("-b") + 1] == "600"
    assert cmd[cmd.index("-t") + 1] == "1000/29970" and cmd[cmd.index("-f") + 1] == "150"
    assert cmd[-1] == str(clip.yuv) and f"--max-q={svc_chart.VPX_MAX_Q},{svc_chart.VPX_MAX_Q}" in cmd


def test_aom_l3t3_command_codes_three_layers_at_the_three_layer_shares(tmp_path: Path):
    clip = Clip(name="c", yuv=tmp_path / "c.yuv", width=1280, height=720, fps=30.0, frames=150,
                source_kbps=10000, content_class="pan", split="val")
    bench = svc_chart.SvcBench.__new__(svc_chart.SvcBench)
    bench.paths, bench.frames, bench.intra = None, 150, 300
    bench.aom_tool = lambda: Path("svc_encoder_rtc")
    l3t3 = next(p for p in svc_chart.POINTS
                if p.kind == "aom" and p.layers == 3 and p.temporal_layers == 3)
    cmd = bench.aom_command(l3t3, clip, 600, tmp_path / "o.ivf")
    rates = [int(r) for r in cmd[cmd.index("-bl") + 1].split(",")]
    assert len(rates) == 9 and rates[2::3] == list(svcbase.split_rate3(600))
    assert cmd[cmd.index("-sl") + 1] == "3" and cmd[cmd.index("-lm") + 1] == "9"


def test_layer_rates_follow_the_point_layer_count(tmp_path: Path):
    two = next(p for p in svc_chart.POINTS if p.kind == "simulcast" and p.layers == 2)
    three = next(p for p in svc_chart.POINTS if p.kind == "simulcast" and p.layers == 3)
    assert two.layer_rates(600) == svcbase.split_rate(600)
    assert three.layer_rates(600) == svcbase.split_rate3(600)
