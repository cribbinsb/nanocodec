"""The CBR charts' reference settings (owner, 2026-11-23): libvpx / libaom real-time with
cyclic-refresh AQ, and openh264 as two points (CAVLC medium complexity, and CABAC high
complexity). A settings change must give new point identities, so no point coded the
old way is reused."""
from pathlib import Path

import pytest

from nceval import cbr
from nceval.cbr import Bench, Codec, Family, Point
from nceval.clips import Clip
from nceval.config import load_paths
from nceval.gatepolicy import EVALUATION_GATE
from nceval.realuse import DEFAULT_FRAMES, Runner

VPX = Family("libvpx-vp9-cpu7", Codec.VP9, Path("/t/vpxenc"), kind="vpx", preset="7")
AOM = Family("libaom-av1-cpu8", Codec.AV1, Path("/t/aomenc"), kind="aom", preset="8")
OPENH264 = Family("openh264-baseline", Codec.H264, Path("/t/openh264-harness"), kind="openh264")


@pytest.fixture
def clip(tmp_path):
    yuv = tmp_path / "c.yuv"
    yuv.write_bytes(b"")
    return Clip(name="c", yuv=yuv, width=1280, height=720, fps=30.0, frames=300,
                source_kbps=20000.0, content_class="crowd", split="val")


@pytest.fixture
def bench(tmp_path, monkeypatch, clip):
    monkeypatch.setenv("NC_WORK", str(tmp_path / "work"))
    monkeypatch.setattr(Bench, "tool_hash", lambda self, family: "tool")
    runner = Runner(load_paths(), EVALUATION_GATE, workers=1, over_budget=True)
    return runner.points([OPENH264], [clip], DEFAULT_FRAMES, None)[0][0]


def command(bench, family, clip, kbps=600, qp=None):
    return bench.command(Point(family, clip, kbps=kbps, qp=qp), Path("/o"), None)


@pytest.mark.parametrize("family", [VPX, AOM])
def test_libvpx_libaom_cbr_codes_with_cyclic_refresh_aq(bench, clip, family):
    assert "--aq-mode=3" in command(bench, family, clip)
    assert "--aq-mode=3" not in command(bench, family, clip, kbps=None, qp=40)


@pytest.mark.parametrize("family", [VPX, AOM])
def test_their_cbr_identity_is_not_the_aq_off_one(bench, clip, family, monkeypatch):
    with_aq = bench.identity(family, clip, "cbr600")
    fixed_qp = bench.identity(family, clip, "qp40")
    monkeypatch.setattr(cbr, "rt_cyclic_aq", lambda kind, profile: False)
    assert bench.identity(family, clip, "cbr600") != with_aq
    assert bench.identity(family, clip, "qp40") == fixed_qp


@pytest.mark.parametrize("family", [VPX, AOM])
def test_their_webrtc_profile_keeps_its_own_aq(bench, clip, family):
    from dataclasses import replace
    argv = command(bench, replace(family, profile="webrtc"), clip)
    assert argv.count("--aq-mode=3") == 1


def test_openh264_two_points(bench, clip):
    from dataclasses import replace
    base = command(bench, OPENH264, clip)
    cabac = command(bench, replace(OPENH264, name="openh264-baseline~cabac", profile="cabac"),
                    clip)
    assert base[base.index("--cabac") + 1] == "0"
    assert base[base.index("--complexity") + 1] == "medium"
    assert cabac[cabac.index("--cabac") + 1] == "1"
    assert cabac[cabac.index("--complexity") + 1] == "high"


def test_the_charts_build_their_openh264_cabac_reference(tmp_path):
    from nceval.chart import chart_families

    families = chart_families(Path("/bin/true"), tmp_path, ["openh264-baseline~cabac"])
    assert [(f.name, f.kind, f.profile) for f in families][1:] == [
        ("openh264-baseline~cabac", "openh264", "cabac")]


def test_quality_runs_the_timed_reference_build(tmp_path):
    from nceval.config import QUALITY_MACHINE, REFERENCE_BUILDS, openh264_harness, quality_reference

    bin_dir = tmp_path / REFERENCE_BUILDS[QUALITY_MACHINE] / "bin"
    bin_dir.mkdir(parents=True)
    for name in ("x264", "openh264-harness"):
        (bin_dir / name).write_text("")
        (bin_dir / name).chmod(0o755)
    assert quality_reference("x264", tmp_path) == bin_dir / "x264"
    assert quality_reference("x265", tmp_path) is None  # not in that build: the usual search
    assert openh264_harness(tmp_path) == bin_dir / "openh264-harness"
