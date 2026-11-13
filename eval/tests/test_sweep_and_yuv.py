"""Tests for quantiser picking, target fitting, coding structures, bitrate
accounting and the YUV MD5 helpers."""

import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nceval.clips import Clip  # noqa: E402
from nceval.codecs import Codec  # noqa: E402
from nceval.config import Paths  # noqa: E402
from nceval.conformance import GateResult  # noqa: E402
from nceval.encoders.base import EncoderAdapter  # noqa: E402
from nceval.sweep import (Sweep, SweepConfig, PROBE_SECONDS, _coarse_grid, fit_targets, payload_bytes,  # noqa: E402
                          pick_quantizers)
from nceval.yuv import YuvFormat, first_mismatch, frame_hashes  # noqa: E402


def _synthetic_rates() -> dict[int, float]:
    # Bitrate halves every 6 QP steps, as H.264/HEVC roughly do.
    return {q: 4000.0 * 2 ** (-(q - 10) / 6) for q in _coarse_grid((10, 51))}


def test_pick_quantizers_hits_targets():
    rates = _synthetic_rates()
    # 4000 * 2^(-(q-10)/6): q=28 -> 500 kbps, q=34 -> 250 kbps.
    assert pick_quantizers(rates, [500.0, 250.0], (10, 51)) == [28, 34]


def test_pick_quantizers_dedupes_and_stays_in_range():
    rates = _synthetic_rates()
    picked = pick_quantizers(rates, [1.0, 1.1, 100000.0], (10, 51))
    assert picked == [10, 51]


def test_fit_targets_respects_cap_and_floor():
    nominal = [150.0, 210.0, 300.0, 420.0, 590.0, 800.0]
    assert fit_targets(nominal, floor_kbps=50.0, cap_kbps=10000.0) == \
        list(np.geomspace(150.0, 800.0, 6))
    raised = fit_targets(nominal, floor_kbps=400.0, cap_kbps=10000.0)
    assert abs(raised[0] - 440.0) < 1e-9 and abs(raised[-1] / raised[0] - 800 / 150) < 1e-9
    capped = fit_targets(nominal, floor_kbps=50.0, cap_kbps=500.0)
    assert capped[-1] == 500.0
    assert fit_targets(nominal, floor_kbps=400.0, cap_kbps=600.0) == []


def test_frame_hashes_and_first_mismatch():
    fmt = YuvFormat(16, 8)
    frames = np.zeros((3, fmt.frame_size), dtype=np.uint8)
    frames[2, 5] = 1
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "a.yuv", Path(tmp) / "b.yuv"
        frames.tofile(a)
        frames[:2].tofile(b)
        md5_a, md5_b = frame_hashes(a, fmt), frame_hashes(b, fmt)
    assert len(md5_a) == 3 and md5_a[0] == md5_a[1] != md5_a[2]
    assert first_mismatch(md5_a, md5_a) is None
    assert first_mismatch(md5_a, md5_b) == 2  # lengths differ
    assert first_mismatch(md5_a, [md5_a[0], md5_a[2], md5_a[2]]) == 1


def _clip(fps: float) -> Clip:
    return Clip(name="c", yuv=Path("/x.yuv"), width=1280, height=720, fps=fps, frames=300,
                source_kbps=4000.0, content_class="static", split="train")


def test_intra_period_rounds_up_to_whole_temporal_gops():
    tl3 = SweepConfig(intra_period_seconds=10, temporal_layers=3)
    assert tl3.structure_for(_clip(25.0)).intra_period_frames == 252   # 250 -> 4-frame GOPs
    assert tl3.structure_for(_clip(30.0)).intra_period_frames == 300
    tl1 = SweepConfig(intra_period_seconds=2, temporal_layers=1)
    assert tl1.structure_for(_clip(25.0)).intra_period_frames == 50
    assert tl1.structure_tag == "ip2s-tl1-vmaf-v1n"


def test_payload_bytes_strips_ivf_framing_only():
    with tempfile.TemporaryDirectory() as tmp:
        stream = Path(tmp) / "s.bin"
        sizes = [80] * 9 + [16]
        stream.write_bytes(b"\0" * 32 + b"".join(
            size.to_bytes(4, "little") + b"\0" * (8 + size) for size in sizes))
        assert payload_bytes(stream, Codec.AV1, 10) == sum(sizes)
        # Input frames our encoder held have no IVF frame: the file counts.
        assert payload_bytes(stream, Codec.AV1, 12) == sum(sizes)
        assert payload_bytes(stream, Codec.HEVC, 10) == 32 + 12 * 10 + sum(sizes)


def _check_probe_encode_reuse(frames: int, verify_recon: bool) -> None:
    class FakeEncoder(EncoderAdapter):
        name = "fake-cache-probe"
        codec = Codec.HEVC
        quantizer_range = (10, 51)

        def verify_recon(self):
            return verify_recon

        def command(self, job, paths):
            return [str(job.bitstream), str(job.frames), str(job.quantizer)]

    calls = []

    def fake_run(command, label):
        path, count, quantizer = Path(command[0]), int(command[1]), int(command[2])
        calls.append((path, count, quantizer))
        path.write_bytes(b"x" * count * (64 - quantizer))

    def fake_streamed(command, fmt, sinks):
        args = command("/unused-reconstruction-pipe")
        fake_run(args, "streamed")
        for _ in range(int(args[1])):
            for sink in sinks:
                sink.consume(b"fake reconstruction")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        sweep = Sweep(Paths(root, root, root), None, "test")
        encoder, clip, cfg = FakeEncoder(), _clip(30.0), SweepConfig(intra_period_seconds=10)
        work = root / "curve"
        probe_frames = min(frames, round(PROBE_SECONDS * clip.fps))
        with patch("nceval.sweep._run", side_effect=fake_run), \
                patch("nceval.sweep.encode_streamed", side_effect=fake_streamed), \
                patch("nceval.sweep.cpu_slots", side_effect=lambda _: nullcontext()), \
                patch.object(sweep, "_gate", return_value=GateResult(True)), \
                patch.object(sweep, "_check_and_measure", return_value=True):
            rates = sweep._estimate_rates(encoder, clip, cfg, frames, work)
            grid = _coarse_grid(encoder.quantizer_range)
            assert set(rates) == set(grid)
            if frames == probe_frames:
                assert len(calls) == len(grid)
                assert all(path.parent == work for path, _, _ in calls)
                selected = grid[0]
            else:
                assert len(calls) == len(grid) + 1  # excerpts plus one calibration
                assert all(path.parent == work / f"probe-{probe_frames}f"
                           and count == probe_frames for path, count, _ in calls[:-1])
                assert calls[-1][0].parent == work and calls[-1][1] == frames
                selected = next(q for q in grid if q != calls[-1][2])
            before = len(calls)
            assert sweep._measure_point(encoder, clip, cfg, selected, frames, work, "test")
            # Exercise the real on-disk _encode cache, including reconstruction
            # hashes for native adapters, rather than mocking the cache itself.
            assert len(calls) == before + int(frames != probe_frames)


def test_full_length_rate_probes_reuse_final_encode_cache():
    for verify_recon in (False, True):
        _check_probe_encode_reuse(16, verify_recon)
        _check_probe_encode_reuse(60, verify_recon)  # exact excerpt boundary


def test_short_rate_probes_do_not_replace_full_length_encodes():
    for verify_recon in (False, True):
        _check_probe_encode_reuse(90, verify_recon)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
