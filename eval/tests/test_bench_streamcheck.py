"""tools/bench/streamcheck.py, report.py's V29-V31 and localscore.py's V27 helpers and follow loop."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "bench"))
import localscore  # noqa: E402
import report  # noqa: E402
import streamcheck  # noqa: E402


def bits(*fields: tuple[int, int]) -> bytes:
    """MSB-first fields (value, width), zero-padded to whole bytes."""
    value, width = 0, 0
    for v, w in fields:
        value, width = (value << w) | v, width + w
    pad = -width % 8
    return (value << pad).to_bytes((width + pad) // 8, "big")


def obu(obu_type: int, payload: bytes) -> bytes:
    """An AV1 OBU with has_size_field (sizes under 128: one leb128 byte)."""
    return bytes([(obu_type << 3) | 0b10, len(payload)]) + payload


def ivf(units: list[bytes]) -> bytes:
    out = b"DKIF" + bytes(28)
    for i, unit in enumerate(units):
        out += len(unit).to_bytes(4, "little") + i.to_bytes(8, "little") + unit
    return out


SEQ = obu(streamcheck.OBU_SEQUENCE_HEADER, bits((0, 3), (0, 1), (0, 1)))


def av1_frame(show_existing=0, frame_type=0, show=1) -> bytes:
    fields = [(show_existing, 1)] if show_existing else [(0, 1), (frame_type, 2), (show, 1)]
    return obu(streamcheck.OBU_FRAME, bits(*fields) + b"\0")


def test_av1_low_delay_stream_is_clean(tmp_path):
    stream = tmp_path / "s.ivf"
    stream.write_bytes(ivf([SEQ + av1_frame(frame_type=0)] + [av1_frame(frame_type=1)] * 4))
    s = streamcheck.structure(stream, "av1")
    assert s == {"probe": "av1 frame headers", "reordered": False, "hidden_frames": 0, "keys": 1,
                 "first_is_key": True, "frames": 5}


def test_av1_hidden_frame_and_show_existing_are_reordering(tmp_path):
    stream = tmp_path / "s.ivf"
    hidden_then_shown = av1_frame(frame_type=1, show=0) + av1_frame(frame_type=1)
    stream.write_bytes(ivf([SEQ + av1_frame(), hidden_then_shown, av1_frame(show_existing=1)]))
    s = streamcheck.structure(stream, "av1")
    assert s["reordered"] and s["hidden_frames"] == 2 and s["frames"] == 3


def vp9_frame(frame_type=1, show=1, show_existing=0) -> bytes:
    fields = [(2, 2), (0, 1), (0, 1), (show_existing, 1)]
    if not show_existing:
        fields += [(frame_type, 1), (show, 1)]
    return bits(*fields) + b"\0\0"


def superframe(frames: list[bytes]) -> bytes:
    marker = 0xC0 | (len(frames) - 1)  # one byte per size
    index = bytes([marker]) + bytes(len(f) for f in frames) + bytes([marker])
    return b"".join(frames) + index


def test_vp9_superframe_with_a_hidden_frame(tmp_path):
    stream = tmp_path / "s.ivf"
    stream.write_bytes(ivf([vp9_frame(frame_type=0),
                            superframe([vp9_frame(show=0), vp9_frame()]),
                            vp9_frame(show_existing=1)]))
    s = streamcheck.structure(stream, "vp9")
    assert s["keys"] == 1 and s["first_is_key"] and s["hidden_frames"] == 2 and s["reordered"]


def test_vp9_low_delay_stream_is_clean(tmp_path):
    stream = tmp_path / "s.ivf"
    stream.write_bytes(ivf([vp9_frame(frame_type=0)] + [vp9_frame()] * 3))
    s = streamcheck.structure(stream, "vp9")
    assert not s["reordered"] and s["frames"] == 4


@pytest.mark.parametrize("codec, first_vcl, key", [
    ("h264", b"\x65\x88", True), ("h264", b"\x41\x9a", False),      # IDR / non-IDR slice
    ("hevc", b"\x26\x01", True), ("hevc", b"\x02\x01", False),      # IDR_W_RADL / TRAIL_R
    ("vvc", b"\x00\x41", True), ("vvc", b"\x00\x11", False),        # IDR_N_LP / STSA
])
def test_h26x_first_picture_key_from_its_nal_type_without_a_decode(tmp_path, codec, first_vcl,
                                                                   key):
    parameter_set = {"h264": b"\x67\x42", "hevc": b"\x40\x01", "vvc": b"\x00\x79"}[codec]
    stream = tmp_path / "s.bin"
    stream.write_bytes(b"\x00\x00\x00\x01" + parameter_set + b"\xff" * 4 +
                       b"\x00\x00\x01" + first_vcl + b"\x80" * 8)
    s = streamcheck.structure(stream, codec)
    assert s == {"probe": "annex-b first picture", "reordered": None, "first_is_key": key}


def test_av2_is_counted_only(tmp_path):
    stream = tmp_path / "s.ivf"
    stream.write_bytes(ivf([b"x"] * 3))
    assert streamcheck.structure(stream, "av2") == {"probe": "count-only", "frames": 3}


def point(**extra):
    base = {"family": "f", "metrics_backend": "vmafx", "cpu_s": 1.0, "wall_s": 1.0,
            "structure": {"probe": "av1 frame headers", "reordered": False, "hidden_frames": 0,
                          "keys": 1, "first_is_key": True, "frames": 150}}
    return {**base, **extra}


def test_report_structure_checks_pass_and_summarise():
    out = report.check_structure({"s": {"a": point(), "b": {"family": "old"},
                                        "c": point(family="av2", structure={"probe": "count-only",
                                                                           "frames": 150})}})
    assert out == {"keys_per_stream": {"f": {1: 1}}, "count_only": {"av2": 1},
                   "order_unread": {}, "unchecked_points": 1}
    h26x = point(family="x264", structure={"probe": "annex-b first picture", "reordered": None,
                                           "first_is_key": True})
    assert report.check_structure({"s": {"a": h26x}})["order_unread"] == {"x264": 1}


@pytest.mark.parametrize("bad, check", [
    ({"structure": {"probe": "x", "reordered": True, "hidden_frames": 3, "keys": 1,
                    "first_is_key": True}}, "V29"),
    ({"structure": {"probe": "failed", "error": "boom"}}, "V29"),
    ({"structure": None}, "V29"),
    ({"structure": {"probe": "x", "reordered": False, "hidden_frames": 0, "keys": 1,
                    "first_is_key": False}}, "V30"),
    ({"structure": {"probe": "annex-b first picture", "reordered": None,
                    "first_is_key": False}}, "V30"),
    ({"cpu_s": 2.0, "wall_s": 1.0}, "V31"),
])
def test_report_structure_refusals(bad, check):
    with pytest.raises(report.ReportRefusal, match=check):
        report.check_structure({"s": {"a": point(**bad)}})


def test_v27_sample_covers_every_codec_and_the_minimum():
    jobs = {f"{c}{i}": {"codec": c} for c in ("h264", "av1", "vp9") for i in range(100)}
    records = [{"id": i, "per_frame": {}} for i in jobs]
    ids = localscore.v27_sample(records, jobs, "seed")
    assert len(ids) == localscore.V27_MIN
    assert {jobs[i]["codec"] for i in ids} == {"h264", "av1", "vp9"}
    assert localscore.v27_sample(records, jobs, "seed") == ids  # seeded


def frames(vmaf, ms, psnr):
    return {"vmaf_v1": vmaf, "ms_ssim": ms, "psnr_y": psnr, "psnr_u": psnr, "psnr_v": psnr}


def test_v27_tolerances():
    cloud = {"stream_sha256": "h", "per_frame": frames([90.123456], [0.95], [40.0])}
    same = {"stream_sha256": "h", "per_frame": frames([90.1234561], [0.950004], [40.00005])}
    assert localscore.parity_mismatches(same, cloud) == []
    worse = {"stream_sha256": "h", "per_frame": frames([90.123457], [0.96], [40.1])}
    assert len(localscore.parity_mismatches(worse, cloud)) == 5  # VMAF, MS-SSIM, three planes
    verdict = localscore.check_parity({"a": same}, {}, ["a"])
    assert verdict["mismatched"] == {"a": ["no cloud score"]}


def test_a_following_local_scorer_waits_for_a_busy_gpu_instead_of_ending(tmp_path, monkeypatch):
    import argparse
    (tmp_path / "jobs.json").write_text("[]")
    passes = []

    def run_pass(run, base, todo, work, workers):
        passes.append(work.name)
        if len(passes) == 1:
            raise localscore.GpuBusy("the GPU scoring service is busy: queue full")
        return work / "scores.jsonl"
    todo = [[{"id": "a|clip"}]]
    monkeypatch.setattr(localscore, "proven", lambda identity: True)
    monkeypatch.setattr(localscore, "scorer_identity", lambda: "build")
    monkeypatch.setattr(localscore, "unscored", lambda *args: todo.pop(0) if todo else [])
    monkeypatch.setattr(localscore, "run_pass", run_pass)
    monkeypatch.setattr(localscore, "merge_pass", lambda base, scores: 1)
    monkeypatch.setattr(localscore.time, "sleep", lambda s: None)
    clock = iter(range(0, 10**6, 3600))  # an hour a reading: the follow ends when idle
    monkeypatch.setattr(localscore.time, "monotonic", lambda: next(clock))
    a = argparse.Namespace(verify_scorer=False, run=tmp_path, series="c8g", ids=None,
                           score_workers=1, tag="-", rescore=False, no_merge=False, follow=0.0)
    with pytest.raises(SystemExit, match="refused: the GPU scoring service is busy"):
        localscore.score_local(a, lambda base: set(), None, None)
    todo[:] = [[{"id": "a|clip"}], [{"id": "a|clip"}]]
    passes.clear()
    a.follow = 30.0
    localscore.score_local(a, lambda base: set(), None, None)
    assert passes == ["score-local-0", "score-local-0"]  # the busy pass, then the same again
