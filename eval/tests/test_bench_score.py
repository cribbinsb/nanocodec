"""tools/bench/score.py: frame sizes, dropped-frame filling and decoder argv."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "bench"))
import score  # noqa: E402
from nceval.codecs import Codec  # noqa: E402


def ivf(frames, rate=30, scale=1):
    head = b"DKIF" + bytes(12) + rate.to_bytes(4, "little") + scale.to_bytes(4, "little") + bytes(8)
    body = b"".join(len(p).to_bytes(4, "little") + pts.to_bytes(8, "little") + p for pts, p in frames)
    return head + body


def test_ivf_frame_bytes_and_slots():
    data = ivf([(0, b"aaaa"), (1, b"bb"), (3, b"c")])
    assert score.frame_bytes(Path("x.ivf"), Codec.AV1, data) == [4, 2, 1]
    assert score.ivf_slots(data, 30.0) == [0, 1, 3]


def test_dropped_frames_show_the_previous_picture(tmp_path):
    raw = tmp_path / "dec.yuv"
    raw.write_bytes(b"A" * 6 + b"B" * 6 + b"C" * 6)
    dropped = score.fill_dropped(raw, [0, 1, 3], 6, 5)
    assert dropped == 2
    assert raw.read_bytes() == b"A" * 6 + b"B" * 6 + b"B" * 6 + b"C" * 6 + b"C" * 6


def test_every_stream_is_scored_by_the_standard_decoder_first():
    assert score.CROSS_DECODERS[Codec.AV2][0] == ["{BIN}/avmdec"]
    assert all(d[0] in (["ffmpeg"], ["{BIN}/avmdec"]) for d in score.CROSS_DECODERS.values())


def test_ffmpeg_gives_one_frame_per_coded_frame(tmp_path):
    argv = score.decode_argv(["ffmpeg"], tmp_path / "s.ivf", tmp_path / "d.yuv")
    assert argv[argv.index("-fps_mode") + 1] == "passthrough"


def test_scoring_workers_are_capped_by_memory():
    """2026-11-24: 90 workers' 1080p decodes filled a scoring box's disk; decodes now live in
    memory, so the workers are capped by what fits (two decodes plus overhead each)."""
    hd = {"w": 1920, "h": 1080, "frames": 130}  # 404 MB per decode
    meminfo = f"MemAvailable: {146 * 2**20} kB\n"  # the c6i.24xlarge diagnostic box
    assert score.worker_cap(90, [hd], meminfo, cpus=96) == \
        (138 << 30) // (2 * 404_352_000 + (1 << 30))
    assert score.worker_cap(4, [hd], meminfo, cpus=96) == 4
    assert score.worker_cap(0, [hd], meminfo, cpus=32) == 32  # 0: one per CPU
    assert score.worker_cap(90, [hd], f"MemAvailable: {4 * 2**20} kB\n", cpus=96) == 1


def test_an_annex_b_drop_row_is_placed_from_its_coded_frame_list():
    """T16: openh264's '+drop' stream has no timestamps; the encoder's list places it."""
    job = {"fps": "30/1", "frames": 5}
    assert score.drop_slots({"coded_frames": [0, 1, 3]}, job, Codec.H264, b"") == [0, 1, 3]


def test_an_annex_b_drop_row_without_its_list_is_refused():
    with pytest.raises(score.StreamRefusal):
        score.drop_slots({}, {"fps": "30/1", "frames": 5}, Codec.H264, b"")


def test_a_dropped_slot_counts_zero_bytes_over_the_whole_duration():
    assert score.bytes_per_slot([40, 10, 20], [0, 1, 3], 5) == [40, 10, 0, 20, 0]
    with pytest.raises(score.StreamRefusal):
        score.bytes_per_slot([40, 10], [0, 1, 3], 5)
