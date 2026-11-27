"""Scores by pictures (owner, 2026-11-25): identical pictures are scored once, whatever the
command that coded them; different pictures are always scored."""

import threading
from pathlib import Path
from types import SimpleNamespace

from nceval import cbr
from nceval.conformance import GateResult
from nceval.metrics import Measurement


class FakeSession:
    calls = 0

    def __init__(self, *_):
        self.sinks, self.cpu_threads = [], 0

    def __enter__(self):
        FakeSession.calls += 1
        return self

    def __exit__(self, *_):
        return False

    def result(self, frames):
        return Measurement({"psnr_yuv": 40.0, "vmaf_v1": 90.0}, [{"vmaf_v1": 90.0}] * frames)


def bench(tmp_path: Path):
    b = object.__new__(cbr.Bench)
    b.work, b.frames, b.measurement, b.metric_set = tmp_path / "work", 2, "m1", "real"
    b.xpsnr, b.lock, b.scores_reused = False, threading.Lock(), 0
    b.prebuilt, b.paths = tmp_path / "prebuilt", SimpleNamespace()
    b.components = lambda spec: []
    return b


def point(tmp_path: Path, label: str, stream_bytes: bytes):
    yuv = tmp_path / "clip.yuv"
    yuv.write_bytes(b"\0" * 64)
    clip = SimpleNamespace(name="c", yuv=yuv, start_frame=0, fps=30.0)
    family = SimpleNamespace(own=False, name=f"x264-{label}",
                             codec=SimpleNamespace(value="h264", bitstream_suffix=".264"))
    p = SimpleNamespace(family=family, clip=clip, label=label, check=cbr.PointCheck.NONE)
    return p, stream_bytes


def score(b, tmp_path, p, data, monkeypatch):
    base = tmp_path / "prebuilt" / p.label
    base.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{base}.done").touch()
    Path(f"{base}.264").write_bytes(data)
    monkeypatch.setattr(b, "prebuilt_base", lambda _p: base, raising=False)
    stream = tmp_path / f"stream-{p.label}.264"
    return b._score_prebuilt(p, stream, None, tmp_path / "times.txt",
                             SimpleNamespace(fmt=None))


def test_an_identical_stream_under_another_command_is_scored_once(tmp_path, monkeypatch):
    monkeypatch.setattr(cbr, "MetricSession", FakeSession)
    monkeypatch.setattr(cbr, "gate_stream",
                        lambda *a, **k: GateResult(passed=True, level="sample"))
    monkeypatch.setattr(cbr, "gpu_slots", lambda n: __import__("contextlib").nullcontext())
    monkeypatch.setattr(cbr, "metrics_backend", lambda: "vmafx-cuda")
    FakeSession.calls = 0
    b = bench(tmp_path)
    first = score(b, tmp_path, *point(tmp_path, "hold-off", b"same bytes"), monkeypatch)
    again = score(b, tmp_path, *point(tmp_path, "hold-on", b"same bytes"), monkeypatch)
    assert FakeSession.calls == 1 and b.scores_reused == 1
    assert again[1] == first[1] and again[0].passed
    score(b, tmp_path, *point(tmp_path, "changed", b"other bytes"), monkeypatch)
    assert FakeSession.calls == 2 and b.scores_reused == 1


def test_our_points_key_on_the_reconstruction_so_header_only_changes_reuse(tmp_path, monkeypatch):
    """2026-11-25: hold changes H.264's fixed_frame_rate_flag (byte 24) in every stream while
    the pictures stay identical: such points reuse their scores."""
    monkeypatch.setattr(cbr, "metrics_backend", lambda: "vmafx-cuda")
    b = bench(tmp_path)
    p, _ = point(tmp_path, "x", b"")
    recon = tmp_path / "r.txt"
    recon.write_text("aa\nbb\n")
    s1, s2 = tmp_path / "a.264", tmp_path / "b.264"
    s1.write_bytes(b"header-0 pictures")
    s2.write_bytes(b"header-1 pictures")
    assert b.score_key(p, s1, recon, None) == b.score_key(p, s2, recon, None)
    other = tmp_path / "r2.txt"
    other.write_text("aa\ncc\n")
    assert b.score_key(p, s1, recon, None) != b.score_key(p, s1, other, None)
    assert b.score_key(p, s1, None, None) != b.score_key(p, s2, None, None)  # anchors: bytes
