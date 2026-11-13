"""Streamed points: frame pipes, the hashed gate, run_point and CPU slots.

Real child processes stand in for encoders and decoders (small Python
programs that write frames to the /dev/fd path they are given), so the
pipe plumbing itself is exercised, not mocked.
"""
import io
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nceval import conformance, framepipe, framestats, metrics, point, slots
from nceval.codecs import Codec
from nceval.yuv import YuvFormat

FMT = YuvFormat(4, 4)  # 24-byte frames
WRITE_FILE = "import sys; open(sys.argv[2], \"wb\").write(open(sys.argv[1], \"rb\").read())"


def frames(*values: int) -> bytes:
    return b"".join(bytes([v]) * FMT.frame_size for v in values)


class Collect:
    def __init__(self):
        self.frames, self.finished = [], False

    def consume(self, frame):
        self.frames.append(frame)

    def finish(self):
        self.finished = True


class Failing(Collect):
    def consume(self, frame):
        raise BrokenPipeError("consumer died")


def test_pump_hands_whole_frames_to_every_sink():
    a, b = Collect(), Collect()
    assert framepipe.pump_frames(io.BytesIO(frames(1, 2, 3)), FMT, [a, b]) == 3
    assert a.frames == b.frames == [frames(1), frames(2), frames(3)]
    assert a.finished and b.finished


def test_pump_rejects_partial_frame():
    with pytest.raises(framepipe.FrameStreamError, match="partial frame"):
        framepipe.pump_frames(io.BytesIO(frames(1) + b"xx"), FMT, [Collect()])


def test_failing_sink_is_detached_but_stream_is_drained():
    source, healthy = io.BytesIO(frames(1, 2, 3)), Collect()
    with pytest.raises(BrokenPipeError):
        framepipe.pump_frames(source, FMT, [Failing(), healthy])
    assert len(healthy.frames) == 3 and source.read() == b""


def test_psnr_sink_matches_direct_computation_and_reads_only_a_prefix(tmp_path):
    rng = np.random.default_rng(1)
    reference = rng.integers(0, 256, 5 * FMT.frame_size, dtype=np.uint8)
    distorted = reference[:3 * FMT.frame_size].copy()
    distorted[::7] ^= 3
    clip = tmp_path / "clip.yuv"
    clip.write_bytes(reference.tobytes())  # longer than the stream: no trimmed copy
    sink = framepipe.PsnrSink(clip, FMT)
    framepipe.pump_frames(io.BytesIO(distorted.tobytes()), FMT, [sink])
    diff = reference[:distorted.size].astype(np.int16) - distorted
    luma = diff.reshape(3, -1)[:, :FMT.luma_size].astype(np.int64)
    expected = 10 * np.log10(255.0 ** 2 / ((luma ** 2).sum(axis=1) / FMT.luma_size))
    np.testing.assert_array_equal(sink.per_plane_psnr()[:, 0], np.minimum(100, expected))


def test_pipe_output_reads_what_a_child_writes_to_its_path(tmp_path):
    source = tmp_path / "frames"
    source.write_bytes(frames(4, 5))
    sink = framepipe.FrameHashSink()
    output = framepipe.PipeOutput(FMT, [sink])
    child = subprocess.Popen([sys.executable, "-c", WRITE_FILE, str(source), output.path],
                             pass_fds=output.pass_fds)
    output.start()
    child.wait()
    output.join(10)
    assert output.frames == 2 and len(sink.digests) == 2
    assert list(tmp_path.iterdir()) == [source]


def test_pipe_output_ends_cleanly_if_the_child_never_opens_it():
    output = framepipe.PipeOutput(FMT, [Collect()])
    child = subprocess.Popen([sys.executable, "-c", "pass"], pass_fds=output.pass_fds)
    output.start()
    child.wait()
    output.join(10)
    assert output.frames == 0


class FakeDecoder(conformance._Decoder):
    """A 'decoder' that copies a prepared file of frames to its output path."""

    def __init__(self, name, data: Path):
        super().__init__(name)
        object.__setattr__(self, "data", data)

    def binary(self, paths):
        return Path(sys.executable)

    def command(self, bitstream, out_yuv, codec, paths):
        return [sys.executable, "-c", WRITE_FILE, str(self.data), str(out_yuv)]


def fake_decoders(tmp_path, first: bytes, second: bytes):
    paths = []
    for name, data in (("dec-a", first), ("dec-b", second)):
        path = tmp_path / name
        path.write_bytes(data)
        paths.append(FakeDecoder(name, path))
    return patch.dict(conformance.DECODERS, {Codec.H264: tuple(paths)})


def md5s(data: bytes):
    sink = framepipe.FrameHashSink()
    framepipe.pump_frames(io.BytesIO(data), FMT, [sink])
    return sink.digests


@pytest.fixture(autouse=True)
def private_slots(tmp_path, monkeypatch):
    monkeypatch.setenv("NC_SLOT_DIR", str(tmp_path / "slots"))
    monkeypatch.setenv("NC_VERIFIED_DIR", str(tmp_path / "verified"))
    monkeypatch.setenv("NC_CPU_SLOTS", "8")


def gate(tmp_path, first, second, stream=b"stream", **kwargs):
    (tmp_path / "s.264").write_bytes(stream)
    with fake_decoders(tmp_path, first, second):
        return conformance.gate_stream(tmp_path / "s.264", Codec.H264, FMT, 2, None, **kwargs)


def test_gate_passes_identical_decodes_matching_the_reconstruction(tmp_path):
    extra = Collect()
    result = gate(tmp_path, frames(1, 2), frames(1, 2), expected_hashes=md5s(frames(1, 2)),
                  extra_sinks=[extra])
    assert result.passed and result.frame_hashes == md5s(frames(1, 2))
    assert extra.frames == [frames(1), frames(2)]  # metric sinks see the first decode
    assert result.decoded_yuv is None


@pytest.mark.parametrize("second, expected, reason", [
    (frames(1, 3), None, "dec-a and dec-b differ at frame 1"),
    (frames(1), None, "dec-b: decoded 1 frames, expected 2"),
    (frames(1) + b"x", None, "dec-b: stream ends with a partial frame"),
    (frames(1, 2), frames(1, 9), "encoder reconstruction differs from decoders at frame 1"),
])
def test_gate_failures_name_the_cause(tmp_path, second, expected, reason):
    result = gate(tmp_path, frames(1, 2), second,
                  expected_hashes=md5s(expected) if expected else None)
    assert not result.passed and reason in result.reason


def gate_held(tmp_path, decoded, shown, **kwargs):
    (tmp_path / "s.264").write_bytes(b"stream")
    with fake_decoders(tmp_path, decoded, decoded):
        return conformance.gate_stream(tmp_path / "s.264", Codec.H264, FMT, len(md5s(shown)),
                                       None, expected_hashes=md5s(shown), **kwargs)


def test_held_frames_are_filled_in_for_metrics_and_located(tmp_path):
    # Input frames 1 and 3 were held: the decoders show frames 0 and 2 again.
    extra = Collect()
    result = gate_held(tmp_path, frames(1, 2), frames(1, 1, 2, 2), extra_sinks=[extra])
    assert result.passed and result.coded == [True, False, True, False]
    assert extra.frames == [frames(1), frames(1), frames(2), frames(2)]


@pytest.mark.parametrize("decoded, shown", [
    (frames(1, 3), frames(1, 2, 3)),  # a skipped picture that is not a repeat
    (frames(1, 2, 3), frames(1, 2)),  # more decoded frames than shown
])
def test_held_frames_must_repeat_the_last_picture(tmp_path, decoded, shown):
    assert not gate_held(tmp_path, decoded, shown).passed


def test_held_frames_expand_the_frame_statistics():
    info = [framestats.FrameInfo(10, "I"), framestats.FrameInfo(5, "P")]
    expanded = framestats.with_held_frames(info, [True, False, True])
    assert [f.bytes for f in expanded] == [10, 0, 5]
    assert framestats.with_held_frames(info, None) is info


def test_coded_frames_come_from_the_encoders_coded_frame_times(tmp_path):
    times = tmp_path / "times.txt"  # nanocodec --timestamps-out at 30000/1001 fps
    times.write_text("# timestamp format v2\n0.000\n33.367\n133.468\n")
    assert framestats.coded_at_times(times, 30000 / 1001, 5) == [True, True, False, False, True]


def test_gate_includes_our_own_decoder(tmp_path):
    own = tmp_path / "own-decoder"
    own.write_text(f"#!{sys.executable}\nimport sys\n"
                   f"open(sys.argv[sys.argv.index('--output') + 1], 'wb').write(bytes([7]) * 48)\n")
    own.chmod(0o755)
    result = gate(tmp_path, frames(1, 2), frames(1, 2), own_decoder=own)
    assert not result.passed and "dec-a and nanocodec differ at frame 0" in result.reason


def test_av1_decodes_are_grain_free():
    """Film grain streams are gated and scored without their grain."""
    ffmpeg, aomdec = conformance.DECODERS[Codec.AV1]
    stream, out = Path("s.ivf"), Path("/dev/fd/3")
    with (patch.object(conformance._Ffmpeg, "binary", return_value=Path("ffmpeg")),
          patch.object(conformance._Decoder, "binary", return_value=Path("aomdec"))):
        ffmpeg_cmd = ffmpeg.command(stream, out, Codec.AV1, None)
        aomdec_cmd = aomdec.command(stream, out, Codec.AV1, None)
    assert ffmpeg_cmd[ffmpeg_cmd.index("-export_side_data") + 1] == "film_grain"
    assert ffmpeg_cmd.index("-export_side_data") < ffmpeg_cmd.index("-i")
    assert "--skip-film-grain" in aomdec_cmd
    own = conformance.own_decode_command(Path("nanocodec"), stream, out, Codec.AV1)
    assert own[-2:] == ["--film-grain", "0"]
    vp9 = conformance.own_decode_command(Path("nanocodec"), stream, out, Codec.VP9)
    assert "--film-grain" not in vp9


def test_run_point_scores_the_reconstruction_and_writes_only_the_bitstream(tmp_path):
    reference = tmp_path / "clip.yuv"
    reference.write_bytes(frames(1, 2, 9))
    recon_data = tmp_path / "recon-data"
    recon_data.write_bytes(frames(1, 3))
    work = tmp_path / "work"
    work.mkdir()
    stream = work / "s.264"
    encoder = (f"import sys; open(sys.argv[1],'w').write(sys.argv[3]); "
               f"open(sys.argv[2],'wb').write(open({str(recon_data)!r},'rb').read())")
    command = lambda recon: [sys.executable, "-c", encoder, str(stream), recon, "bits"]
    spec = metrics.StreamSpec(reference, FMT, 2, 30, None)
    with fake_decoders(tmp_path, frames(1, 3), frames(1, 3)):
        result = point.run_point(command, stream, Codec.H264, spec, None, [metrics.PsnrMetric(spec)])
    assert result.gate.passed
    assert result.measurement.summary["psnr_y"] == pytest.approx((100 + 10 * np.log10(255 ** 2)) / 2)
    assert list(work.iterdir()) == [stream]
    other = lambda recon: command(recon)[:-1] + ["other bits"]  # not in the verified cache
    with fake_decoders(tmp_path, frames(1, 3), frames(1, 4)):
        failed = point.run_point(other, stream, Codec.H264, spec, None, [metrics.PsnrMetric(spec)])
    assert not failed.gate.passed and failed.measurement is None


def test_slots_bound_concurrency_and_are_all_or_nothing(monkeypatch):
    monkeypatch.setenv("NC_CPU_SLOTS", "2")
    monkeypatch.setattr(slots, "POLL_SECONDS", (0.001, 0.002))
    with slots.cpu_slots(1):
        assert slots.slots_in_use() == 1
        acquired = threading.Event()

        def want_two():
            with slots.cpu_slots(2):
                acquired.set()
        waiter = threading.Thread(target=want_two)
        waiter.start()
        time.sleep(0.1)
        assert not acquired.is_set() and slots.slots_in_use() == 1  # no partial hold
    waiter.join(5)
    assert acquired.is_set() and slots.slots_in_use() == 0


def test_slot_requests_larger_than_the_pool_are_capped(monkeypatch):
    monkeypatch.setenv("NC_CPU_SLOTS", "2")
    with slots.cpu_slots(5):
        assert slots.slots_in_use() == 2


def test_slots_wait_for_memory(monkeypatch):
    monkeypatch.setattr(slots, "POLL_SECONDS", (0.001, 0.002))
    readings = iter([0, 0, slots.MIN_AVAILABLE_BYTES])
    monkeypatch.setattr(slots, "available_memory", lambda: next(readings))
    with slots.cpu_slots(1):
        pass
    with pytest.raises(StopIteration):
        next(readings)


def test_process_set_kills_leftovers():
    processes = framepipe.ProcessSet()
    child = processes.spawn([sys.executable, "-c", "import time; time.sleep(60)"], "sleeper")
    processes.kill_all()
    assert child.returncode is not None  # killed and reaped


def test_process_set_reports_failures_with_their_stderr():
    with framepipe.ProcessSet() as processes:
        processes.spawn([sys.executable, "-c", "import sys; sys.exit('boom')"], "tool")
        with pytest.raises(framepipe.ProcessFailed, match="tool failed.*boom"):
            processes.wait_all(10)


def test_metric_session_rejects_a_short_stream(tmp_path):
    reference = tmp_path / "clip.yuv"
    reference.write_bytes(frames(1, 2))
    spec = metrics.StreamSpec(reference, FMT, 2, 30, None)
    with metrics.MetricSession(spec, [metrics.PsnrMetric(spec)]) as session:
        count = framepipe.pump_frames(io.BytesIO(frames(1)), FMT, session.sinks)
        with pytest.raises(ValueError, match="streamed 1 frames, expected 2"):
            session.result(count)


def test_file_api_streams_through_the_same_components(tmp_path):
    reference, distorted = tmp_path / "ref.yuv", tmp_path / "dist.yuv"
    reference.write_bytes(frames(10, 20))
    distorted.write_bytes(frames(10, 21))
    result = metrics.psnr(reference, distorted, FMT)
    assert result.per_frame[0]["psnr_y"] == 100
    assert result.per_frame[1]["psnr_y"] == pytest.approx(10 * np.log10(255 ** 2))
    short = tmp_path / "short.yuv"
    short.write_bytes(frames(1))
    with pytest.raises(ValueError, match="frame counts differ"):
        metrics.psnr(reference, short, FMT)


def test_harness_subprocesses_run_at_lowest_priority():
    with framepipe.ProcessSet() as processes:
        child = processes.spawn([sys.executable, "-c", "import os; print(os.nice(0))"],
                                "probe", log_stdout=True)
        processes.wait_all(10)
        assert processes.stderr_text(child).strip() == str(slots.NICENESS)


def test_one_process_holds_at_most_its_share(monkeypatch):
    monkeypatch.setenv("NC_CPU_SLOTS", "16")
    monkeypatch.setattr(slots, "POLL_SECONDS", (0.001, 0.002))
    assert slots.owner_share() == 8
    with slots.cpu_slots(10):
        assert slots.slots_in_use() == 8
        second = threading.Event()

        def another_worker():
            with slots.cpu_slots(1):
                second.set()
        worker = threading.Thread(target=another_worker)
        worker.start()
        time.sleep(0.1)
        assert not second.is_set()  # the process's share is used up
    worker.join(5)
    assert second.is_set()


def test_zero_slots_hold_nothing():
    with slots.cpu_slots(0):
        assert slots.slots_in_use() == 0


def test_run_point_takes_slots_before_starting_metric_processes(tmp_path, monkeypatch):
    order = []

    class Recorder:
        cpu_threads, uses_gpu = 2, False

        def start(self, processes, scratch):
            order.append("metrics started")
            return []

        def measurement(self, processes):
            return metrics.Measurement({})

    real = slots.cpu_slots

    def recording_slots(count):
        order.append(f"slots {count}")
        return real(count)
    monkeypatch.setattr(point, "cpu_slots", recording_slots)
    spec = metrics.StreamSpec(tmp_path / "clip", FMT, 0, 30, None)
    with patch.object(point, "gate_stream"), patch.object(point, "encode_streamed", return_value=0):
        point.run_point(lambda recon: [], tmp_path / "s", Codec.H264, spec, None, [Recorder()])
    assert order == ["slots 3", "metrics started"]


def test_verified_stream_skips_standard_decoders_but_still_checks_the_recon(tmp_path):
    good = md5s(frames(1, 2))
    assert gate(tmp_path, frames(1, 2), frames(1, 2), expected_hashes=good).level == "full"
    # Same stream bytes: broken "decoders" are not run any more.
    cached = gate(tmp_path, b"", b"", expected_hashes=good)
    assert cached.passed and cached.level == "cached-verified"
    wrong = gate(tmp_path, b"", b"", expected_hashes=md5s(frames(1, 3)))
    assert not wrong.passed and "reconstruction differs" in wrong.reason
    fresh = gate(tmp_path, b"", b"", stream=b"other stream", expected_hashes=good)
    assert not fresh.passed  # a different stream is decoded again


def test_failed_gate_is_not_cached(tmp_path):
    gate(tmp_path, frames(1, 2), frames(1, 3))
    assert gate(tmp_path, frames(1, 2), frames(1, 2)).level == "full"


def own_decoder_writing(tmp_path, data: bytes) -> Path:
    own = tmp_path / "own-decoder"
    own.write_text(f"#!{sys.executable}\nimport sys\n"
                   f"open(sys.argv[sys.argv.index('--output') + 1], 'wb').write({data!r})\n")
    own.chmod(0o755)
    return own


def test_sampled_gate_uses_only_our_decoder_and_the_recon(tmp_path):
    own = own_decoder_writing(tmp_path, frames(1, 2))
    result = gate(tmp_path, b"", b"", own_decoder=own, standard=False,
                  expected_hashes=md5s(frames(1, 2)))
    assert result.passed and result.level == "own-decoder"
    mismatch = gate(tmp_path, b"", b"", own_decoder=own, standard=False,
                    expected_hashes=md5s(frames(1, 4)))
    assert not mismatch.passed


def test_an_off_gate_decodes_nothing(tmp_path):
    result = gate(tmp_path, b"", b"", standard=False)  # the decoders would fail
    assert result.passed and result.level == "unchecked"


def test_an_off_gate_decodes_an_anchor_once_for_its_metrics(tmp_path):
    extra = Collect()
    result = gate(tmp_path, frames(1, 2), b"", standard=False, extra_sinks=[extra])
    assert result.passed and result.level == "unchecked"  # the second decoder never ran
    assert extra.frames == [frames(1), frames(2)]


def test_levels_satisfy_the_checks_they_had():
    from nceval.gatepolicy import PointCheck, level_satisfies
    assert level_satisfies("unchecked", PointCheck.NONE)
    assert not level_satisfies("unchecked", PointCheck.OWN)
    assert level_satisfies("own-decoder", PointCheck.OWN)
    assert not level_satisfies("own-decoder", PointCheck.STANDARD)
    assert level_satisfies("cached-verified", PointCheck.STANDARD)
    assert level_satisfies("", PointCheck.STANDARD)  # stored before levels: fully gated


def test_sample_policy_picks_first_and_one_rotating_point():
    from nceval.gatepolicy import GatePolicy, standard_decoders_needed
    for key in ("a", "b", "c"):
        chosen = [i for i in range(6) if standard_decoders_needed(GatePolicy.SAMPLE, key, i, 6)]
        assert chosen[0] == 0 and len(chosen) == 2
    rotations = {next(i for i in range(1, 6)
                      if standard_decoders_needed(GatePolicy.SAMPLE, str(k), i, 6))
                 for k in range(40)}
    assert len(rotations) > 1
    assert all(standard_decoders_needed(GatePolicy.FULL, "a", i, 6) for i in range(6))


def test_waiting_multi_slot_request_is_not_overtaken(tmp_path, monkeypatch):
    """FIFO across processes: a later single-slot request queues behind a
    waiting two-slot request instead of taking the slot that frees first."""
    monkeypatch.setenv("NC_CPU_SLOTS", "2")
    holder = ("import sys, time; sys.path.insert(0, sys.argv[1]); from nceval import slots; "
              "slots.POLL_SECONDS = (0.001, 0.005); "
              "cm = slots.cpu_slots(int(sys.argv[2])); cm.__enter__(); "
              "open(sys.argv[3], 'w').write(str(time.monotonic())); time.sleep(0.3); cm.__exit__(None, None, None)")
    root = str(Path(__file__).resolve().parents[1])
    with slots.cpu_slots(1):
        two = subprocess.Popen([sys.executable, "-c", holder, root, "2", str(tmp_path / "two")])
        time.sleep(0.5)  # the two-slot request is now queued
        one = subprocess.Popen([sys.executable, "-c", holder, root, "1", str(tmp_path / "one")])
        time.sleep(0.5)  # one slot is free, yet the single request must wait its turn
        assert not (tmp_path / "one").exists() and not (tmp_path / "two").exists()
    assert two.wait(10) == 0 and one.wait(10) == 0
    assert float((tmp_path / "two").read_text()) < float((tmp_path / "one").read_text())


def test_dead_waiters_tickets_are_skipped(tmp_path):
    queue = tmp_path / "queue"
    queue.mkdir()
    (queue / f"{0:020d}-1-1.ticket").touch()  # unlocked: its holder died
    with slots._ticket(queue) as ticket:
        assert slots._is_first(ticket)
    assert list(queue.iterdir()) == []


def test_available_slots_shrink_with_contenders_and_load(monkeypatch):
    monkeypatch.setenv("NC_CPU_SLOTS", "16")
    monkeypatch.setattr(slots.os, "getloadavg", lambda: (0.0, 0.0, 0.0))
    monkeypatch.setattr(slots, "contending_owners", lambda: set())
    assert slots.available_slots() == 8  # the nominal share
    monkeypatch.setattr(slots, "contending_owners", lambda: {1, 2, 3, 4, 5, 6, 7})
    assert slots.available_slots() == 2  # 16 slots over 8 processes
    monkeypatch.setattr(slots.os, "getloadavg", lambda: (40.0, 40.0, 40.0))
    assert slots.available_slots() == 1  # a saturated machine


def test_contending_owners_sees_holders_and_queued_waiters(monkeypatch):
    monkeypatch.setenv("NC_CPU_SLOTS", "2")
    holder = ("import sys, time; sys.path.insert(0, sys.argv[1]); from nceval import slots; "
              "cm = slots.cpu_slots(1); cm.__enter__(); print('held', flush=True); time.sleep(5)")
    root = str(Path(__file__).resolve().parents[1])
    child = subprocess.Popen([sys.executable, "-c", holder, root], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "held"
        assert child.pid in slots.contending_owners()
    finally:
        child.kill()
        child.wait()

@pytest.mark.parametrize("serial", [False, True])
def test_serial_gate_retains_every_decoder_check(tmp_path, serial):
    # Sequential mode changes scheduling only; a disagreeing second decoder
    # must still fail and no mismatching stream may enter the verified cache.
    with patch.object(conformance, "_run_decoders", wraps=conformance._run_decoders) as run:
        result = gate(tmp_path, frames(1, 2), frames(1, 3), serial=serial)
    assert not result.passed and "differ at frame 1" in result.reason
    assert run.call_count == (2 if serial else 1)
    assert [call.args[-1] for call in run.call_args_list] == ([1, 1] if serial else [2])
