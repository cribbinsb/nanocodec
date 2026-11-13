"""Concurrent requests for one cached point: one encode, every requester gets it.

Two agents' screens needing the same anchor point at the same moment used to
share its stream file (named after the cache entry): one encode clobbered or
deleted the other's bitstream mid-gate. These tests run the real cache code
in separate processes (as two agents would) with the tools faked.
"""
from __future__ import annotations

import json
import multiprocessing
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from nceval import cachefile, cbr
from nceval.conformance import GateResult
from nceval.metrics import Measurement

FRAMES = 2
#: Long enough that the second requester arrives while the first encodes.
ENCODE_SECONDS = 0.5


class _FakeSession:
    """MetricSession stand-in: no metric tools, fixed per-frame scores."""

    def __init__(self, spec, components):
        self.sinks, self.cpu_threads = [], 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def result(self, frames):
        row = {"psnr_y": 40.0, "psnr_u": 42.0, "psnr_v": 42.0, "vmaf_v1": 90.0}
        return Measurement({"psnr_yuv": 40.5, "vmaf_v1": 90.0}, [dict(row)] * frames)


def _fake_tools(monkeypatch, encodes: Path, final_stream: Path) -> None:
    """Encodes log themselves to `encodes`; the gate fails on a clobbered stream."""
    def encode(bench, point, out, source=None):
        assert out != final_stream and not out.exists(), "stream path is not private"
        payload = f"{out.name}\n".encode()
        out.write_bytes(payload)
        with encodes.open("a") as log:
            log.write(out.name + "\n")
        time.sleep(ENCODE_SECONDS)
        if out.read_bytes() != payload:
            raise AssertionError("another requester overwrote this stream")

    def gate(stream, codec, fmt, frames, paths, **_):
        ok = stream.read_bytes() == f"{stream.name}\n".encode()
        return GateResult(ok, reason="" if ok else "clobbered stream", level="full")

    monkeypatch.setattr(cbr.Bench, "encode", encode)
    monkeypatch.setattr(cbr.Bench, "components", lambda bench, spec: [])
    monkeypatch.setattr(cbr, "check_disk", lambda work: None)
    monkeypatch.setattr(cbr, "MetricSession", _FakeSession)
    monkeypatch.setattr(cbr, "gate_stream", gate)
    monkeypatch.setattr(cbr, "probe_frames", lambda stream, paths: [
        SimpleNamespace(bytes=100, frame_type="I" if i == 0 else "P") for i in range(FRAMES)])


def _request(bench, point, start, results) -> None:
    start.wait()
    try:
        results.put(("ok", bench.measure(point)))
    except BaseException as err:  # noqa: BLE001 - reported to the parent
        results.put(("error", repr(err)))


def test_two_processes_measure_one_anchor_point_with_one_encode(tmp_path, monkeypatch):
    tool = tmp_path / "x264"
    tool.write_bytes(b"anchor build")
    clip_yuv = tmp_path / "clip.yuv"
    clip_yuv.write_bytes(b"\0" * 96)
    clip = SimpleNamespace(name="clip", yuv=clip_yuv, metric_reference=clip_yuv,
                           width=8, height=8, fps=30.0, start_frame=0, direct=True,
                           clean_reference=None)
    family = cbr.Family("x264-veryfast", cbr.Codec.H264, tool, kind="x264")
    bench = cbr.Bench(paths=SimpleNamespace(work=tmp_path / "cache"), frames=FRAMES, intra=300, work=tmp_path / "cache")
    point = cbr.Point(family, clip, kbps=600)
    encodes = tmp_path / "encodes.log"
    final_stream = bench.result_path(point).with_suffix(family.codec.bitstream_suffix)
    _fake_tools(monkeypatch, encodes, final_stream)

    # fork: the children inherit the fakes, as two agents share the real tools.
    context = multiprocessing.get_context("fork")
    start, results = context.Barrier(2), context.Queue()
    workers = [context.Process(target=_request, args=(bench, point, start, results))
               for _ in range(2)]
    for worker in workers:
        worker.start()
    outcomes = [results.get(timeout=30) for _ in workers]
    for worker in workers:
        worker.join(timeout=30)

    assert [status for status, _ in outcomes] == ["ok", "ok"], outcomes
    first, second = (value for _, value in outcomes)
    assert first == second and first["kbps"] == pytest.approx(100 * 8 * 30 / 1000)
    assert len(encodes.read_text().splitlines()) == 1
    assert cachefile.read_json(bench.result_path(point)) == first
    # Only the result (and its lock) remain: no stream, no temporaries.
    left = sorted(p.name for p in bench.result_path(point).parent.iterdir())
    assert left == [bench.result_path(point).name,
                    cachefile.lock_path(bench.result_path(point)).name]


def test_get_or_create_computes_once_across_threads(tmp_path):
    entry, calls = tmp_path / "entry.json", []

    def create():
        calls.append(1)
        time.sleep(0.2)
        return {"value": 7}

    def read(path):
        return cachefile.read_json(path)

    values = []
    threads = [threading.Thread(target=lambda: values.append(
        cachefile.get_or_create(entry, read, create, json.dumps))) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert values == [{"value": 7}] * 4 and len(calls) == 1


def test_identity_cached_recomputes_a_stale_entry_once(tmp_path):
    entry = tmp_path / "entry.json"
    assert cachefile.identity_cached(entry, {"v": (1, 2)}, lambda: "old") == "old"
    assert cachefile.identity_cached(entry, {"v": [1, 2]}, lambda: "unused") == "old"
    assert cachefile.identity_cached(entry, {"v": 3}, lambda: "new") == "new"


def test_real_use_result_of_the_old_ms_ssim_protocol_is_remeasured_when_used(tmp_path):
    from nceval.gatepolicy import PointCheck
    from nceval.metrics import REAL_MS_SSIM_REVISION
    bench = cbr.Bench(paths=None, frames=FRAMES, intra=300, work=tmp_path, measurement="m",
                      metric_set="real")
    point = SimpleNamespace(check=PointCheck.NONE)
    stored = tmp_path / "point.json"
    stored.write_text(json.dumps({"gate_level": "full", "ms_ssim": 0.9}))  # every 4th frame
    assert bench._reusable(point, stored) is None
    stored.write_text(json.dumps({"gate_level": "full", "ms_ssim": 0.9,
                                  "ms_ssim_revision": REAL_MS_SSIM_REVISION}))
    assert bench._reusable(point, stored) is not None
    bench.metric_set = "classic"  # no MS-SSIM: unaffected
    stored.write_text(json.dumps({"gate_level": "full"}))
    assert bench._reusable(point, stored) is not None
