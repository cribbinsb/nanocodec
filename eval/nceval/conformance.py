"""Conformance gate: every bitstream must decode bit-exactly with standard decoders.

For each codec two independent decoders are run (CLAUDE.md "Testing"):

    H.264  ffmpeg (libavcodec)  +  JM ldecod (reference)
    HEVC   ffmpeg (libavcodec)  +  HM TAppDecoder (reference)
    VVC    ffmpeg (native vvc)  +  VVdeC vvdecapp
    AV1    ffmpeg (libdav1d)    +  libaom aomdec (reference)
    VP9    ffmpeg (native vp9)  +  libvpx vpxdec (reference)

The reference decoders are strict and catch syntax errors that ffmpeg would
silently conceal; ffmpeg is run with error concealment disabled as well.

AV1 is decoded without film grain (aomdec --skip-film-grain, ffmpeg's
libdav1d exporting the grain as side data instead of applying it, our
decoder with --film-grain 0), as AOM's common test conditions and Netflix
score it: synthesized grain is noise by design, so it is judged by eye,
and the gate and metrics see the grain-free picture, which must match the
encoder's reconstruction like any other stream.

"ffmpeg" here is always the pinned decode-only build in the anchors prefix
(bin/ffmpeg-decode, tools/anchors/versions.env), never one on $PATH: a conda
FFmpeg 7.1.1 first on PATH once mis-decoded VVC ALF streams (ticket 11325),
so the VVC gate also refuses any FFmpeg older than 8.0.

Decoders write into pipes (/dev/fd/N) that are hashed frame by frame, so a
check writes no decoded video at all.

A stream passes only if:
  - both decoders exit cleanly,
  - both produce the expected number of frames,
  - their per-frame hashes (yuv.FRAME_HASH) are identical,
  - and, when the encoder gave us its reconstruction, it matches too.

Our encoder codes no frame for an input frame that would not change the
picture (frame hold) or that rate control drops, and a decoder shows the
previous picture again there. Its reconstruction is therefore the picture
shown at every input frame (`--display-recon`): the decoders' frames must be
those pictures with the repeats of uncoded frames left out
(`framepipe.shown_frames`), and metrics from a decoder see the repeats
filled back in (`framepipe.HeldFrameSink`).

Nothing downstream (metrics, BD-rate, speed tables) may use a stream that
did not pass.
"""

from __future__ import annotations

import csv
import hashlib
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from . import dav2d_crop, jm_split
from .codecs import Codec
from .config import Paths, find_anchor_tool, find_tool
from .slots import cpu_slots
from .gatepolicy import GateLevel, VerifiedStreams, file_sha256
from .display import DisplaySink, switched_format
from .framepipe import (FileSink, FrameSink, FrameStreamError, FrameHashSink, HeldFrameSink,
                        PipeOutput, ProcessSet, shown_frames)
from .yuv import FRAME_HASH, SwitchedFormat, YuvFormat, first_mismatch, frame_hashes

DECODE_TIMEOUT_SECONDS = 600
# One decode thread: writing into a pipe, ffmpeg decodes a 720p 150-frame
# point in ~0.35 CPU-s, and the shared slot limiter (nceval.slots) provides
# the parallelism. Thread count never changes the decoded pixels.
FFMPEG_DECODE_THREADS = 1
# FFmpeg before 8.0 reads uninitialised CC-ALF indexes whenever VVC ALF is on
# (ticket 11325, fixed in f80af3657f, first released in n8.0), so it can
# mis-decode our ALF streams at random (docs/handoff/vvc.md).
VVC_MIN_FFMPEG_VERSION = (8, 0)
# "ffmpeg version n8.0.1" (a tag build), "ffmpeg version 8.0.1-3ubuntu2" (a
# distribution). A git snapshot ("N-12345-g...") has no comparable version.
_FFMPEG_VERSION = re.compile(r"^ffmpeg version n?(\d+)\.(\d+)")


@dataclass
class GateResult:
    passed: bool
    reason: str = ""
    decoded_yuv: Path | None = None  # a verified decode, for metrics
    frame_hashes: list[str] = field(default_factory=list)
    level: str = ""  # GateLevel of a passed gate (nceval.gatepolicy)
    #: Per input frame, whether a frame was coded for it (None: every one,
    #: or no reconstruction to tell): nceval.framestats.with_held_frames.
    coded: list[bool] | None = None


@dataclass(frozen=True)
class _Decoder:
    name: str

    tool = ""  # the executable, located with find_tool

    def binary(self, paths: Paths) -> Path:
        return find_tool(self.tool, paths)

    def command(self, bitstream: Path, out_yuv: Path, codec: Codec, paths: Paths) -> list[str]:
        raise NotImplementedError


@dataclass(frozen=True)
class _Ffmpeg(_Decoder):
    min_version: tuple[int, int] = (0, 0)  # refuse an older FFmpeg for this gate

    tool = "ffmpeg-decode"

    def binary(self, paths: Paths) -> Path:
        # The pinned build (build_anchors.sh ffmpeg), never one on $PATH.
        binary = find_anchor_tool(self.tool, paths)
        if self.min_version > (0, 0):
            require_ffmpeg_version(binary, self.min_version)
        return binary

    def command(self, bitstream, out_yuv, codec, paths):
        demuxer = {Codec.H264: "h264", Codec.HEVC: "hevc", Codec.VVC: "vvc", Codec.AV1: "ivf", Codec.VP9: "ivf"}[codec]
        # libdav1d applies film grain unless it is exported as side data.
        decoder = ({Codec.AV1: ["-export_side_data", "film_grain", "-c:v", "libdav1d"],
                    Codec.VVC: ["-c:v", "vvc"],
                    Codec.VP9: ["-c:v", "vp9"]}.get(codec, []))
        return [str(self.binary(paths)), "-nostdin", "-v", "error", "-xerror",
                "-err_detect", "explode", "-filter_threads", "1",
                "-threads", str(FFMPEG_DECODE_THREADS), "-f", demuxer, *decoder, "-i", str(bitstream),
                "-fps_mode", "passthrough", "-threads", "1", "-noautoscale",
                "-f", "rawvideo", "-pix_fmt", "yuv420p",
                "-y", str(out_yuv)]


class _Jm(_Decoder):
    tool = "ldecod"

    def command(self, bitstream, out_yuv, codec, paths):
        # A stream that changes picture size is decoded per coded video
        # sequence (nceval.jm_split: the pinned JM cannot switch sizes).
        if jm_split.distinct_sps_count(Path(bitstream).read_bytes()) > 1:
            return [sys.executable, "-I", jm_split.__file__, str(self.binary(paths)),
                    str(bitstream), str(out_yuv)]
        return [str(self.binary(paths)), "-i", str(bitstream), "-o", str(out_yuv)]


class _Hm(_Decoder):
    tool = "TAppDecoder"

    def command(self, bitstream, out_yuv, codec, paths):
        return [str(self.binary(paths)), "-b", str(bitstream),
                "-o", str(out_yuv), "-d", "8"]


class _Vvdec(_Decoder):
    tool = "vvdecapp"

    def binary(self, paths: Paths) -> Path:
        # The pinned VVdeC build (build_anchors.sh vvdec), never one on $PATH.
        return find_anchor_tool(self.tool, paths)

    def command(self, bitstream, out_yuv, codec, paths):
        return [str(self.binary(paths)), "-v", "0", "-t", "1",
                "-b", str(bitstream), "-o", str(out_yuv)]


class _Aomdec(_Decoder):
    tool = "aomdec"

    def command(self, bitstream, out_yuv, codec, paths):
        return [str(self.binary(paths)), "--rawvideo", "--i420", "--skip-film-grain",
                "-o", str(out_yuv), str(bitstream)]


class _Vpxdec(_Decoder):
    tool = "vpxdec"

    def command(self, bitstream, out_yuv, codec, paths):
        return [str(self.binary(paths)), "--codec=vp9", "--rawvideo", "--i420",
                "--threads=1", "-o", str(out_yuv), str(bitstream)]


class _Avmdec(_Decoder):
    tool = "avmdec"

    def command(self, bitstream, out_yuv, codec, paths):
        return [str(self.binary(paths)), "--rawvideo", "--i420", "--threads=1",
                "-o", str(out_yuv), str(bitstream)]


class _Dav2d(_Decoder):
    tool = "dav2d"

    def command(self, bitstream, out_yuv, codec, paths):
        # dav2d outputs the coded frame, not the conformance window
        # (nceval.dav2d_crop crops it to the visible size).
        if dav2d_crop.needs_crop(Path(bitstream)):
            return [sys.executable, "-I", dav2d_crop.__file__, str(self.binary(paths)),
                    str(bitstream), str(out_yuv)]
        return dav2d_crop.dav2d_command(str(self.binary(paths)), str(bitstream), str(out_yuv))


DECODERS: dict[Codec, tuple[_Decoder, _Decoder]] = {
    Codec.H264: (_Ffmpeg("ffmpeg"), _Jm("jm")),
    Codec.HEVC: (_Ffmpeg("ffmpeg"), _Hm("hm")),
    Codec.VVC: (_Ffmpeg("ffmpeg-native-vvc", VVC_MIN_FFMPEG_VERSION), _Vvdec("vvdec")),
    Codec.AV1: (_Ffmpeg("ffmpeg-dav1d"), _Aomdec("aomdec")),
    Codec.VP9: (_Ffmpeg("ffmpeg-native-vp9"), _Vpxdec("vpxdec")),
    Codec.AV2: (_Avmdec("avmdec"), _Dav2d("dav2d")),
}


OWN_DECODER_NAME = "nanocodec"


def require_ffmpeg_version(binary: Path, minimum: tuple[int, int]) -> None:
    """Raise RuntimeError unless `binary` reports an FFmpeg >= `minimum`."""
    stat = binary.stat()
    version = _ffmpeg_version(binary.resolve(), stat.st_size, stat.st_mtime_ns)
    if version is None or version < minimum:
        found = "an unrecognised version" if version is None else ".".join(map(str, version))
        wanted = ".".join(map(str, minimum))
        raise RuntimeError(
            f"{binary} is FFmpeg {found}; the VVC gate needs FFmpeg >= {wanted} "
            f"(older ones mis-decode VVC ALF, ticket 11325). "
            f"Run tools/anchors/build_anchors.sh ffmpeg.")


@lru_cache(maxsize=16)
def _ffmpeg_version(binary: Path, size: int, mtime_ns: int) -> tuple[int, int] | None:
    """`binary`'s (major, minor), once per build (size and mtime key the cache)."""
    banner = subprocess.run([str(binary), "-hide_banner", "-version"], stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, check=True).stdout
    match = _FFMPEG_VERSION.match(banner)
    return (int(match[1]), int(match[2])) if match else None


def gate_stream(bitstream: Path, codec: Codec, fmt: YuvFormat, expected_frames: int,
                paths: Paths, *, expected_hashes: list[str] | None = None,
                own_decoder: Path | None = None,
                extra_sinks: list[FrameSink] = (), extra_threads: int = 0,
                standard: bool = True, hold_slots: bool = True,
                serial: bool = False,
                coded_sizes: SwitchedFormat | None = None) -> GateResult:
    """Decode `bitstream` and compare every decode's per-frame hashes.

    Every decoder writes into a pipe that is hashed frame by frame; no
    decoded YUV is written. `expected_hashes` are the encoder reconstruction's
    per-frame hashes; `own_decoder` adds our native decoder to the comparison.
    `standard=False` skips the two standard decoders (the sample policy,
    nceval.gatepolicy); with no `own_decoder` either (the off policy) only
    `extra_sinks`' decoder runs, for an anchor's metrics, or, without them,
    nothing is decoded (level `unchecked`). A stream that already
    passed both standard decoders is not decoded by them again: its cached
    hashes take their place (level `cached-verified`).
    `extra_sinks` (e.g. a MetricSession's) receive the frames of the first
    standard decoder, which then always runs; they must be discarded if the
    gate fails. The check holds one CPU slot per decoder plus `extra_threads`.
    `serial=True` checks decoders one at a time, useful for tiny conformance
    packets when only one shared slot is available. All checks still run.
    `hold_slots=False` when the caller already holds them (it must acquire
    them before starting the sinks' processes).
    With `expected_hashes` (one per input frame, the pictures shown), the
    decoders may output fewer frames: held ones (module docstring).

    `coded_sizes` is the size of each decoded frame of a stream that codes
    frames smaller than `fmt`, its stream (render) size (adaptive
    resolution): decoders output frames of these sizes; their coded frames
    must match each other, and each frame scaled to `fmt` by the player
    model (nceval.display) is what is compared with `expected_hashes`, the
    encoder's reconstruction as displayed, cached, and handed to
    `extra_sinks`.
    """
    if not standard and own_decoder is None and not extra_sinks:
        return GateResult(True, level=GateLevel.UNCHECKED.value)
    cache = VerifiedStreams(_decoder_identity(codec, paths))
    stream_sha = file_sha256(bitstream)
    cached = cache.lookup(stream_sha, codec.value, fmt.width, fmt.height, expected_frames,
                          held=expected_hashes is not None)
    run_standard = cached is None and standard
    level = (GateLevel.FULL if run_standard else
             GateLevel.CACHED if cached is not None else
             GateLevel.OWN_DECODER if own_decoder is not None else GateLevel.UNCHECKED)

    if expected_hashes is not None and extra_sinks:
        extra_sinks = [HeldFrameSink(list(extra_sinks), expected_hashes)]
    decoders: list[tuple[str, object, list[FrameSink]]] = []
    hashes: dict[str, FrameHashSink] = {}
    coded_hashes: dict[str, FrameHashSink] = {}

    def sinks_of(name: str, shown: list[FrameSink]) -> list[FrameSink]:
        if coded_sizes is None:
            return shown
        coded_hashes[name] = FrameHashSink()
        return [coded_hashes[name], DisplaySink(coded_sizes, fmt, shown)]

    for index, decoder in enumerate(DECODERS[codec]):
        if run_standard or (index == 0 and extra_sinks):
            hashes[decoder.name] = FrameHashSink()
            sinks = [hashes[decoder.name], *(extra_sinks if index == 0 else ())]
            decoders.append((decoder.name, decoder, sinks_of(decoder.name, sinks)))
    if own_decoder is not None:
        hashes[OWN_DECODER_NAME] = FrameHashSink()
        decoders.append((OWN_DECODER_NAME, own_decoder,
                         sinks_of(OWN_DECODER_NAME, [hashes[OWN_DECODER_NAME]])))
    output_fmt = coded_sizes if coded_sizes is not None else fmt

    errors = {}
    groups = [[decoder] for decoder in decoders] if serial else [decoders]
    for group in groups:
        errors.update(_run_decoders(bitstream, codec, output_fmt, paths, group,
                                   len(group) + extra_threads if hold_slots else 0))
    decoded = {name: sink.digests for name, sink in hashes.items()}
    for name, digests in decoded.items():
        if errors.get(name):
            return GateResult(False, f"{name}: {errors[name]}")
        held_allowed = expected_hashes is not None and len(digests) <= expected_frames
        if len(digests) != expected_frames and not held_allowed:
            return GateResult(False, f"{name}: decoded {len(digests)} frames, "
                                     f"expected {expected_frames}")
    coded_digests = [(name, sink.digests) for name, sink in coded_hashes.items()]
    for (first_name, first), (name, digests) in zip(coded_digests, coded_digests[1:]):
        mismatch = first_mismatch(first, digests)
        if mismatch is not None:
            return GateResult(False, f"{first_name} and {name} differ at frame {mismatch} "
                                     f"(coded size)")
    if cached is not None:
        decoded = {"verified-cache": cached, **decoded}
    (first_name, first), *others = decoded.items()
    for name, digests in others:
        mismatch = first_mismatch(first, digests)
        if mismatch is not None:
            return GateResult(False, f"{first_name} and {name} differ at frame {mismatch}")
    coded = None
    if expected_hashes is not None:
        coded = shown_frames(first, expected_hashes)
        if coded is None:
            mismatch = first_mismatch(first, expected_hashes)
            return GateResult(False, f"encoder reconstruction differs from decoders "
                                     f"at frame {mismatch} ({len(first)} decoded frames, "
                                     f"{len(expected_hashes)} shown)")
    if run_standard:
        cache.store(stream_sha, codec.value, fmt.width, fmt.height, first, expected_frames)
    return GateResult(True, frame_hashes=first, level=level.value, coded=coded)


# Hashes cached before AV1 was decoded grain-free may be of grain-applied
# output (a stream gated without its reconstruction): this mark keeps them
# apart from grain-free hashes.
AV1_GRAIN_FREE_MARK = "grain=off"


def _decoder_identity(codec: Codec, paths: Paths) -> str:
    """The standard decoder binaries' hashes (a new build invalidates the
    cache), the per-frame hash, and for AV1 that they decode grain-free."""
    identity = [f"{decoder.name}={file_sha256(decoder.binary(paths))}"
                for decoder in DECODERS[codec]]
    identity.append(f"frame-hash={FRAME_HASH}")  # records of another hash never match
    if codec is Codec.AV1:
        identity.append(AV1_GRAIN_FREE_MARK)
    return ",".join(identity)


def _run_decoders(bitstream: Path, codec: Codec, fmt: YuvFormat, paths: Paths,
                  decoders: list[tuple[str, object, list[FrameSink]]],
                  slots: int) -> dict[str, str]:
    """Run all decoders concurrently, each into its own hashed pipe.

    Holds `slots` CPU slots. Returns an error description per failing decoder. A decoder is either a
    standard `_Decoder` or the path of our own binary.
    """
    errors: dict[str, str] = {}
    # Reference decoders write log/trace files into the working directory.
    with (cpu_slots(slots), tempfile.TemporaryDirectory(prefix="nc-dec-") as cwd,
          ProcessSet() as processes):
        running = []
        for name, decoder, sinks in decoders:
            output = PipeOutput(fmt, sinks)
            out_path = Path(output.path)
            cmd = (decoder.command(bitstream, out_path, codec, paths)
                   if isinstance(decoder, _Decoder) else
                   own_decode_command(decoder, bitstream, out_path, codec))
            try:
                process = processes.spawn(cmd, name, pass_fds=output.pass_fds,
                                          log_stdout=True, cwd=Path(cwd))
            except BaseException:
                output.close_unstarted()
                raise
            running.append((name, process, output.start()))
        for name, process, output in running:
            try:
                process.wait(DECODE_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                errors[name] = "timed out"
            try:
                output.join(DECODE_TIMEOUT_SECONDS)
            except (ValueError, FrameStreamError) as err:  # not whole frames
                errors.setdefault(name, str(err))
            if process.returncode not in (0, None) and name not in errors:
                tail = processes.stderr_text(process).strip().splitlines()[-3:]
                errors[name] = f"exit code {process.returncode}: {' | '.join(tail)}"
    return errors


def own_decode_command(binary: Path, bitstream: Path, out_yuv: Path, codec: Codec) -> list[str]:
    """Our decoder's command line; AV1 shows frames grain-free, as the
    standard decoders here do."""
    grain_free = ["--film-grain", "0"] if codec is Codec.AV1 else []
    return [str(binary), "decode", "--codec", codec.value,
            "--input", str(bitstream), "--output", str(out_yuv), *grain_free]


def decode_to_file(decoder: _Decoder, bitstream: Path, out_yuv: Path, codec: Codec,
                   fmt: YuvFormat, paths: Paths) -> str:
    """Decode with one standard decoder into a YUV file; return an error or "".

    Only for tools that score retained streams from files (nceval.tune_rescore).
    """
    with out_yuv.open("wb") as output:
        errors = _run_decoders(bitstream, codec, fmt, paths,
                               [(decoder.name, decoder, [FileSink(output)])], 1)
    return errors.get(decoder.name, "")


def check_stream(bitstream: Path, codec: Codec, fmt: YuvFormat, expected_frames: int,
                 out_dir: Path, paths: Paths, recon_yuv: Path | None = None) -> GateResult:
    """File-based gate for callers that still need a decoded YUV (nceval.cbr).

    Same checks as `gate_stream`; on success the first decoder's output is
    also written to `out_dir` as `decoded_yuv`. New code should stream.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    decoded = out_dir / f"{bitstream.stem}.{DECODERS[codec][0].name}.yuv"
    expected = frame_hashes(recon_yuv, fmt) if recon_yuv is not None else None
    with decoded.open("wb") as output:
        gate = gate_stream(bitstream, codec, fmt, expected_frames, paths,
                           expected_hashes=expected, extra_sinks=[FileSink(output)])
    if not gate.passed:
        decoded.unlink(missing_ok=True)
        return gate
    gate.decoded_yuv = decoded
    return gate


class _BitReader:
    """Minimal MSB-first bit reader for AV1 `sequence_header_obu()` (`AV1 §5.5`)."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._bit = 0

    def read(self, n: int) -> int:
        val = 0
        for _ in range(n):
            byte_idx, bit_idx = divmod(self._bit, 8)
            if byte_idx >= len(self._data):
                raise ValueError("truncated AV1 sequence header")
            val = (val << 1) | ((self._data[byte_idx] >> (7 - bit_idx)) & 1)
            self._bit += 1
        return val

    def read_uvlc(self) -> int:
        leading = 0
        while self.read(1) == 0:
            leading += 1
            if leading >= 32:
                return 0xFFFFFFFF
        if leading == 0:
            return 0
        return self.read(leading) + (1 << leading) - 1


def _first_av1_seq_header_payload(bitstream: Path) -> bytes:
    """Extract the first `OBU_SEQUENCE_HEADER` (`obu_type == 1`) payload from an IVF file."""
    data = bitstream.read_bytes()
    if len(data) < 44 or data[:4] != b"DKIF":
        raise ValueError(f"not a valid IVF stream: {bitstream}")
    tu_size = int.from_bytes(data[32:36], "little")
    tu = data[44:44 + tu_size]
    pos = 0
    while pos < len(tu):
        hdr = tu[pos]
        pos += 1
        obu_type = (hdr >> 3) & 0x0F
        has_ext = (hdr >> 2) & 1
        has_size = (hdr >> 1) & 1
        if has_ext:
            pos += 1
        if has_size:
            size = 0
            shift = 0
            while True:
                b = tu[pos]
                pos += 1
                size |= (b & 0x7F) << shift
                if (b & 0x80) == 0:
                    break
                shift += 7
        else:
            size = len(tu) - pos
        payload = tu[pos:pos + size]
        pos += size
        if obu_type == 1:
            return payload
    raise ValueError(f"no OBU_SEQUENCE_HEADER in first temporal unit of {bitstream}")


def av1_stream_operating_point_idcs(bitstream: Path) -> list[int]:
    """Read `operating_point_idc[0..operating_points_cnt]` directly from the
    stream's `OBU_SEQUENCE_HEADER` (`AV1 §5.5.1`)."""
    r = _BitReader(_first_av1_seq_header_payload(bitstream))
    r.read(3)  # seq_profile
    r.read(1)  # still_picture
    reduced = r.read(1)  # reduced_still_picture_header
    if reduced:
        return [0]
    decoder_model_present = 0
    buffer_delay_len = 0
    if r.read(1):  # timing_info_present_flag
        r.read(32)  # num_units_in_display_tick
        r.read(32)  # time_scale
        if r.read(1):  # equal_picture_interval
            r.read_uvlc()  # num_ticks_per_picture_minus_1
        decoder_model_present = r.read(1)
        if decoder_model_present:
            buffer_delay_len = r.read(5) + 1
            r.read(32)  # num_units_in_decoding_tick
            r.read(5)  # buffer_removal_time_length_minus_1
            r.read(5)  # frame_presentation_time_length_minus_1
    initial_display_delay_present = r.read(1)
    op_cnt = r.read(5) + 1
    idcs: list[int] = []
    for _ in range(op_cnt):
        idc = r.read(12)
        level = r.read(5)
        if level > 7:
            r.read(1)  # seq_tier
        if decoder_model_present and r.read(1):
            r.read(2 * buffer_delay_len + 1)
        if initial_display_delay_present and r.read(1):
            r.read(4)
        idcs.append(idc)
    return idcs


def _idc_max_layers(idc: int) -> tuple[int, int]:
    """`(max_spatial_id, max_temporal_id)` enabled by a 12-bit `operating_point_idc`."""
    if idc == 0:
        return (0, 0)
    max_t = max((t for t in range(8) if (idc >> t) & 1), default=0)
    max_s = max((s for s in range(4) if (idc >> (8 + s)) & 1), default=0)
    return (max_s, max_t)


def av1_stream_operating_points(bitstream: Path) -> list[tuple[int, int]]:
    """`(max_spatial_id, max_temporal_id)` list parsed from `bitstream`'s sequence header."""
    return [_idc_max_layers(idc) for idc in av1_stream_operating_point_idcs(bitstream)]


def coded_format_from_layer_md5(layer_md5: Path | None, fmt: YuvFormat) -> SwitchedFormat | None:
    """Output picture sizes of operating point 0 from `--layer-md5`, or `None`
    when every shown picture already matches `fmt`."""
    if layer_md5 is None or not layer_md5.is_file():
        return None
    by_tu: dict[int, tuple[int, int]] = {}
    with layer_md5.open(newline="") as handle:
        for row in csv.DictReader(handle):
            by_tu[int(row["tu"])] = (int(row["width"]), int(row["height"]))
    if not by_tu:
        return None
    sizes = [by_tu[tu] for tu in sorted(by_tu)]
    if all(size == (fmt.width, fmt.height) for size in sizes):
        return None
    switched = switched_format(sizes)
    return switched if isinstance(switched, SwitchedFormat) else SwitchedFormat(switched, ())


def _pipe_i420_md5s(cmd: list[str], expected_dims: list[tuple[int, int]]) -> tuple[str, list[str]]:
    """Run `cmd` (writing raw I420 to stdout) and hash each expected `(w, h)` frame in memory."""
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    assert proc.stdout is not None and proc.stderr is not None
    digests: list[str] = []
    try:
        for idx, (width, height) in enumerate(expected_dims):
            chroma_w = (width + 1) // 2
            chroma_h = (height + 1) // 2
            frame_bytes = width * height + 2 * chroma_w * chroma_h
            chunk = proc.stdout.read(frame_bytes)
            if len(chunk) != frame_bytes:
                err = proc.stderr.read().decode(errors="replace").strip()
                proc.wait(DECODE_TIMEOUT_SECONDS)
                return (f"EOF at frame {idx}/{len(expected_dims)} "
                        f"({len(chunk)}/{frame_bytes} B): {err[-240:]}"), digests
            digests.append(hashlib.md5(chunk).hexdigest())
        extra = proc.stdout.read(1)
        err_text = proc.stderr.read().decode(errors="replace").strip()
        proc.wait(DECODE_TIMEOUT_SECONDS)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    if proc.returncode != 0:
        return f"exit code {proc.returncode}: {err_text[-240:]}", digests
    if extra:
        return f"decoded more than {len(expected_dims)} frames", digests
    return "", digests


def gate_av1_operating_points(bitstream: Path, layer_md5: Path,
                              operating_points: list[tuple[int, int]] | None, paths: Paths,
                              own_decoder: Path | None = None) -> GateResult:
    """Verify every AV1 spatial/temporal operating point signalled in the
    stream's `OBU_SEQUENCE_HEADER` (plus `--alllayers 1` and `--all-layers` on
    multi-spatial points) bit-exact against `--layer-md5` across `dav1d`,
    `aomdec`, and `nanocodec decode` with zero YUV files on disk."""
    try:
        idcs = av1_stream_operating_point_idcs(bitstream)
    except ValueError as err:
        return GateResult(False, f"sequence header operating points: {err}")
    stream_ops = [_idc_max_layers(idc) for idc in idcs]
    if operating_points is not None and stream_ops != operating_points:
        return GateResult(
            False,
            f"sequence header operating points {stream_ops} != expected {operating_points}",
        )

    tus: dict[int, list[tuple[int, int, int, int, str, int, str]]] = {}
    with layer_md5.open(newline="") as handle:
        for row in csv.DictReader(handle):
            sid = int(row["spatial_id"])
            ftype = row.get("frame_type", "")
            default_il = 1 if (sid > 0 and ftype == "Inter") else 0
            il_ref = int(row.get("inter_layer_ref") or default_il)
            tus.setdefault(int(row["tu"]), []).append((
                sid, int(row["temporal_layer"]),
                int(row["width"]), int(row["height"]),
                row["md5"].strip().lower(), il_ref, ftype))
    ordered_tus = [tus[tu] for tu in sorted(tus)]
    max_stream_t = max((t for _, t in stream_ops), default=0)
    dav1d = find_anchor_tool("dav1d", paths)
    aomdec = find_tool("aomdec", paths)

    with cpu_slots(1), tempfile.TemporaryDirectory(prefix="nc-av1-op-") as tmp_dir:
        tmp = Path(tmp_dir)
        for op_idx, (idc, (max_s, max_t)) in enumerate(zip(idcs, stream_ops)):
            shown = []
            all_layers = []
            extracted_shown = []
            extracted_layers = []
            has_stripped_lower = False
            stripped_lower_since_key = False
            for layers in ordered_tus:
                kept = [
                    entry for entry in layers
                    if idc == 0 or (((idc >> entry[1]) & 1) != 0 and ((idc >> (8 + entry[0])) & 1) != 0)
                ]
                if kept:
                    if kept[0][0] == 0 and kept[0][6] == "Key":
                        stripped_lower_since_key = False
                    shown.append((kept[-1][2], kept[-1][3], kept[-1][4]))
                    all_layers.extend((e[2], e[3], e[4]) for e in kept)
                    if stripped_lower_since_key and kept[-1][0] < max_s:
                        continue
                    chain = [kept[-1]]
                    cur_idx = len(kept) - 1
                    while cur_idx > 0 and kept[cur_idx][5] != 0:
                        prev_sid = kept[cur_idx][0] - 1
                        prev_idx = next(
                            (k for k in range(cur_idx - 1, -1, -1) if kept[k][0] == prev_sid),
                            None,
                        )
                        if prev_idx is None:
                            break
                        chain.append(kept[prev_idx])
                        cur_idx = prev_idx
                    chain.reverse()
                    if len(chain) < len(kept):
                        has_stripped_lower = True
                        stripped_lower_since_key = True
                    extracted_shown.append((kept[-1][2], kept[-1][3], kept[-1][4]))
                    extracted_layers.extend((e[2], e[3], e[4]) for e in chain)
            shown_dims = [(w, h) for w, h, _ in shown]
            shown_md5s = [m for _, _, m in shown]
            ext_shown_dims = [(w, h) for w, h, _ in extracted_shown]
            ext_shown_md5s = [m for _, _, m in extracted_shown]

            for label, cmd in (
                (f"dav1d op={op_idx} alllayers=0",
                 [str(dav1d), "-q", "--filmgrain", "0", "--oppoint", str(op_idx),
                  "--alllayers", "0", "--muxer", "yuv", "-i", str(bitstream), "-o", "-"]),
                (f"aomdec op={op_idx}",
                 [str(aomdec), "--rawvideo", "--i420", "--skip-film-grain",
                  f"--oppoint={op_idx}", "-o", "-", str(bitstream)]),
            ):
                err, got = _pipe_i420_md5s(cmd, shown_dims)
                if err:
                    return GateResult(False, f"{label}: {err}")
                bad = first_mismatch(got, shown_md5s)
                if bad is not None:
                    return GateResult(False, f"{label}: MD5 mismatch at frame {bad}")

            if own_decoder is not None:
                res = subprocess.run(
                    [str(own_decoder), "decode", "--codec", "av1", "--film-grain", "0",
                     "--operating-point", str(op_idx), "--input", str(bitstream),
                     "--frame-md5", "-"],
                    stdin=subprocess.DEVNULL, capture_output=True, text=True,
                    timeout=DECODE_TIMEOUT_SECONDS)
                if res.returncode != 0:
                    return GateResult(False, f"nanocodec op={op_idx}: {res.stderr.strip()[-240:]}")
                got_own = [line.strip().lower() for line in res.stdout.splitlines() if line.strip()]
                if len(got_own) != len(shown_md5s):
                    return GateResult(False, f"nanocodec op={op_idx}: {len(got_own)} frames, "
                                             f"expected {len(shown_md5s)}")
                bad = first_mismatch(got_own, shown_md5s)
                if bad is not None:
                    return GateResult(False, f"nanocodec op={op_idx}: MD5 mismatch at frame {bad}")

            if own_decoder is not None and (len(idcs) > 1 or has_stripped_lower):
                op_ivf = tmp / f"op{op_idx}.ivf"
                ext = subprocess.run(
                    [str(own_decoder), "extract", "--codec", "av1",
                     "--operating-point", str(op_idx),
                     "--input", str(bitstream), "--output", str(op_ivf)],
                    stdin=subprocess.DEVNULL, capture_output=True, text=True,
                    timeout=DECODE_TIMEOUT_SECONDS)
                if ext.returncode != 0:
                    return GateResult(
                        False,
                        f"nanocodec extract av1 op={op_idx}: {ext.stderr.strip()[-240:]}",
                    )
                if max_s == 0 and max_t == max_stream_t and len(idcs) > 1:
                    base_ivf = tmp / f"base_{op_idx}.ivf"
                    ext_base = subprocess.run(
                        [str(own_decoder), "extract", "--codec", "av1", "--base-only",
                         "--input", str(bitstream), "--output", str(base_ivf)],
                        stdin=subprocess.DEVNULL, capture_output=True, text=True,
                        timeout=DECODE_TIMEOUT_SECONDS)
                    if ext_base.returncode != 0:
                        return GateResult(
                            False,
                            f"nanocodec extract av1 --base-only: {ext_base.stderr.strip()[-240:]}",
                        )
                    if base_ivf.read_bytes() != op_ivf.read_bytes():
                        return GateResult(
                            False,
                            f"nanocodec extract av1 --base-only differs from op={op_idx}",
                        )
                for label, cmd in (
                    (f"dav1d extracted op={op_idx}",
                     [str(dav1d), "-q", "--filmgrain", "0",
                      "--alllayers", "0", "--muxer", "yuv", "-i", str(op_ivf), "-o", "-"]),
                    (f"aomdec extracted op={op_idx}",
                     [str(aomdec), "--rawvideo", "--i420", "--skip-film-grain",
                      "-o", "-", str(op_ivf)]),
                ):
                    err, got = _pipe_i420_md5s(cmd, ext_shown_dims)
                    if err:
                        return GateResult(False, f"{label}: {err}")
                    bad = first_mismatch(got, ext_shown_md5s)
                    if bad is not None:
                        return GateResult(False, f"{label}: MD5 mismatch at frame {bad}")
                for own_label, own_flags in (
                    (f"nanocodec extracted op={op_idx}", []),
                    (f"nanocodec extracted --operating-point {op_idx}",
                     ["--operating-point", str(op_idx)]),
                ):
                    res_ext = subprocess.run(
                        [str(own_decoder), "decode", "--codec", "av1", "--film-grain", "0",
                         *own_flags, "--input", str(op_ivf), "--frame-md5", "-"],
                        stdin=subprocess.DEVNULL, capture_output=True, text=True,
                        timeout=DECODE_TIMEOUT_SECONDS)
                    if res_ext.returncode != 0:
                        return GateResult(
                            False, f"{own_label}: {res_ext.stderr.strip()[-240:]}")
                    got_ext = [
                        line.strip().lower() for line in res_ext.stdout.splitlines() if line.strip()
                    ]
                    if len(got_ext) != len(ext_shown_md5s):
                        return GateResult(
                            False,
                            f"{own_label}: {len(got_ext)} frames, expected {len(ext_shown_md5s)}",
                        )
                    bad = first_mismatch(got_ext, ext_shown_md5s)
                    if bad is not None:
                        return GateResult(False, f"{own_label}: MD5 mismatch at frame {bad}")
                if has_stripped_lower:
                    ext_all_md5s = [m for _, _, m in extracted_layers]
                    res_ext_all = subprocess.run(
                        [str(own_decoder), "decode", "--codec", "av1", "--film-grain", "0",
                         "--all-layers", "1", "--input", str(op_ivf), "--frame-md5", "-"],
                        stdin=subprocess.DEVNULL, capture_output=True, text=True,
                        timeout=DECODE_TIMEOUT_SECONDS)
                    if res_ext_all.returncode != 0:
                        return GateResult(
                            False,
                            f"nanocodec extracted op={op_idx} all-layers: "
                            f"{res_ext_all.stderr.strip()[-240:]}",
                        )
                    got_ext_all = [
                        line.strip().lower()
                        for line in res_ext_all.stdout.splitlines()
                        if line.strip()
                    ]
                    if len(got_ext_all) != len(ext_all_md5s):
                        return GateResult(
                            False,
                            f"nanocodec extracted op={op_idx} all-layers: "
                            f"{len(got_ext_all)} pictures, expected {len(ext_all_md5s)}",
                        )
                    bad = first_mismatch(got_ext_all, ext_all_md5s)
                    if bad is not None:
                        return GateResult(
                            False,
                            f"nanocodec extracted op={op_idx} all-layers: "
                            f"MD5 mismatch at picture {bad}",
                        )

            if max_s > 0:
                all_dims = [(w, h) for w, h, _ in all_layers]
                all_md5s = [m for _, _, m in all_layers]
                for label, cmd in (
                    (f"dav1d op={op_idx} alllayers=1",
                     [str(dav1d), "-q", "--filmgrain", "0", "--oppoint", str(op_idx),
                      "--alllayers", "1", "--muxer", "yuv", "-i", str(bitstream), "-o", "-"]),
                    (f"aomdec op={op_idx} all-layers",
                     [str(aomdec), "--rawvideo", "--i420", "--skip-film-grain",
                      f"--oppoint={op_idx}", "--all-layers", "-o", "-", str(bitstream)]),
                ):
                    err, got = _pipe_i420_md5s(cmd, all_dims)
                    if err:
                        return GateResult(False, f"{label}: {err}")
                    bad = first_mismatch(got, all_md5s)
                    if bad is not None:
                        return GateResult(False, f"{label}: MD5 mismatch at picture {bad}")
                if own_decoder is not None:
                    res = subprocess.run(
                        [str(own_decoder), "decode", "--codec", "av1", "--film-grain", "0",
                         "--operating-point", str(op_idx), "--all-layers", "1",
                         "--input", str(bitstream), "--frame-md5", "-"],
                        stdin=subprocess.DEVNULL, capture_output=True, text=True,
                        timeout=DECODE_TIMEOUT_SECONDS)
                    if res.returncode != 0:
                        return GateResult(
                            False,
                            f"nanocodec op={op_idx} all-layers: {res.stderr.strip()[-240:]}",
                        )
                    got_own_all = [
                        line.strip().lower() for line in res.stdout.splitlines() if line.strip()
                    ]
                    if len(got_own_all) != len(all_md5s):
                        return GateResult(
                            False,
                            f"nanocodec op={op_idx} all-layers: {len(got_own_all)} pictures, "
                            f"expected {len(all_md5s)}",
                        )
                    bad = first_mismatch(got_own_all, all_md5s)
                    if bad is not None:
                        return GateResult(
                            False,
                            f"nanocodec op={op_idx} all-layers: MD5 mismatch at picture {bad}",
                        )

    return GateResult(True, level=GateLevel.FULL.value)


def gate_vp9_operating_points(bitstream: Path, layer_md5: Path, paths: Paths,
                              own_decoder: Path | None = None) -> GateResult:
    """Verify every VP9 spatial/temporal operating point (full superframe stream,
    K-SVC and lower-layer stripped streams, and any temporal sub-operating points
    in `op_mask`, plus `nanocodec decode --operating-point 0..N-1` and
    `--all-layers 1` on the unstripped stream) bit-exact against `--layer-md5`
    across `vpxdec`, `ffmpeg-decode`, and `nanocodec decode` with zero YUV files on disk."""
    from .svcbase import VP9_SHOW_FRAME_BIT, ivf_frames, vp9_coded_layers

    tus: dict[int, list[tuple[int, int, int, int, str, int, int, str]]] = {}
    with layer_md5.open(newline="") as handle:
        for row in csv.DictReader(handle):
            sid = int(row["spatial_id"])
            op_mask = int(row.get("op_mask") or (0b11 if sid == 0 else 0b01))
            tus.setdefault(int(row["tu"]), []).append((
                sid, int(row["temporal_layer"]),
                int(row["width"]), int(row["height"]),
                row["md5"].strip().lower(), op_mask,
                int(row.get("inter_layer_ref") or 0),
                row["frame_type"].strip()))
    ordered_tus = [tus[tu] for tu in sorted(tus)]
    num_ops = max((entry[5].bit_length() for layers in ordered_tus for entry in layers), default=1)
    num_spatial = max((entry[0] for layers in ordered_tus for entry in layers), default=0) + 1
    # Each spatial view's full-rate operating point: the first (highest quality)
    # one whose top layer is that view's. Not a fixed stride: a decimated
    # enhancement layer has fewer temporal operating points than the base.
    op_tops = [max((e[0] for layers in ordered_tus for e in layers if (e[5] >> op) & 1),
                   default=0) for op in range(num_ops)]
    view_ops: dict[int, int] = {}
    for op, top in enumerate(op_tops):
        view_ops.setdefault(num_spatial - 1 - top, op)
    first_pkts = ivf_frames(bitstream.read_bytes())
    first_coded = vp9_coded_layers(first_pkts[0][1]) if first_pkts else []
    top_is_hidden = len(first_coded) > 1 and not bool(first_coded[-1][0] & VP9_SHOW_FRAME_BIT)
    vpxdec = find_tool("vpxdec", paths)
    ffmpeg = find_anchor_tool("ffmpeg-decode", paths)

    def expected_for_op(op_idx: int) -> tuple[list[tuple[int, int, str]], list[tuple[int, int, str]]]:
        bit = 1 << op_idx
        max_s = op_tops[op_idx]
        shown: list[tuple[int, int, str]] = []
        ext_shown: list[tuple[int, int, str]] = []
        stripped_lower_since_key = False
        for layers in ordered_tus:
            kept = [entry for entry in layers if (entry[5] & bit) != 0]
            if not kept:
                continue
            if kept[0][0] == 0 and kept[0][7] == "Key":
                stripped_lower_since_key = False
            shown.append((kept[-1][2], kept[-1][3], kept[-1][4]))
            sel = next((e for e in kept if e[0] == max_s), None)
            if sel is None:
                if stripped_lower_since_key:
                    continue
                below = [e for e in kept if e[0] < max_s]
                if not below:
                    continue
                sel = below[-1]
            below_kept = [e for e in kept if e[0] <= sel[0]]
            if (sel[0] + 1 < num_spatial) or top_is_hidden:
                chain = [sel]
                cur = sel
                while cur[6] != 0 and cur[0] > 0:
                    prev = next((e for e in reversed(below_kept) if e[0] == cur[0] - 1), None)
                    if prev is None:
                        break
                    chain.append(prev)
                    cur = prev
                if len(chain) < len(below_kept):
                    stripped_lower_since_key = True
            ext_shown.append((sel[2], sel[3], sel[4]))
        return shown, ext_shown

    with cpu_slots(1), tempfile.TemporaryDirectory(prefix="nc-vp9-op-") as tmp_dir:
        tmp = Path(tmp_dir)
        if own_decoder is not None:
            all_md5s = [entry[4] for layers in ordered_tus for entry in layers]
            res = subprocess.run(
                [str(own_decoder), "decode", "--codec", "vp9", "--all-layers", "1",
                 "--input", str(bitstream), "--frame-md5", "-"],
                stdin=subprocess.DEVNULL, capture_output=True, text=True,
                timeout=DECODE_TIMEOUT_SECONDS)
            if res.returncode != 0:
                return GateResult(False, f"nanocodec vp9 all-layers: {res.stderr.strip()[-240:]}")
            got_all = [line.strip().lower() for line in res.stdout.splitlines() if line.strip()]
            if len(got_all) != len(all_md5s):
                return GateResult(
                    False,
                    f"nanocodec vp9 all-layers: {len(got_all)} pictures, expected {len(all_md5s)}",
                )
            bad = first_mismatch(got_all, all_md5s)
            if bad is not None:
                return GateResult(False, f"nanocodec vp9 all-layers: MD5 mismatch at picture {bad}")

            for sp_op in range(num_spatial):
                _, sp_ext_shown = expected_for_op(view_ops[sp_op])
                sp_md5s = [m for _, _, m in sp_ext_shown]
                res_sp = subprocess.run(
                    [str(own_decoder), "decode", "--codec", "vp9",
                     "--operating-point", str(sp_op),
                     "--input", str(bitstream), "--frame-md5", "-"],
                    stdin=subprocess.DEVNULL, capture_output=True, text=True,
                    timeout=DECODE_TIMEOUT_SECONDS)
                if res_sp.returncode != 0:
                    return GateResult(
                        False,
                        f"nanocodec vp9 --operating-point {sp_op}: "
                        f"{res_sp.stderr.strip()[-240:]}",
                    )
                got_sp = [
                    line.strip().lower() for line in res_sp.stdout.splitlines() if line.strip()
                ]
                if len(got_sp) != len(sp_md5s):
                    return GateResult(
                        False,
                        f"nanocodec vp9 --operating-point {sp_op}: {len(got_sp)} frames, "
                        f"expected {len(sp_md5s)}",
                    )
                bad = first_mismatch(got_sp, sp_md5s)
                if bad is not None:
                    return GateResult(
                        False,
                        f"nanocodec vp9 --operating-point {sp_op}: MD5 mismatch at frame {bad}",
                    )

        for op_idx in range(num_ops):
            shown, ext_shown = expected_for_op(op_idx)
            shown_dims = [(w, h) for w, h, _ in shown]
            shown_md5s = [m for _, _, m in shown]
            ext_dims = [(w, h) for w, h, _ in ext_shown]
            ext_md5s = [m for _, _, m in ext_shown]

            if own_decoder is None:
                if op_idx > 0:
                    continue
                op_ivf = bitstream
                ivf_targets = [(f"op={op_idx}", op_ivf, shown_dims, shown_md5s)]
            else:
                op_ivf = tmp / f"op{op_idx}.ivf"
                ext = subprocess.run(
                    [str(own_decoder), "extract", "--codec", "vp9",
                     "--layer-md5", str(layer_md5), "--operating-point", str(op_idx),
                     "--input", str(bitstream), "--output", str(op_ivf)],
                    stdin=subprocess.DEVNULL, capture_output=True, text=True,
                    timeout=DECODE_TIMEOUT_SECONDS)
                if ext.returncode != 0:
                    return GateResult(
                        False,
                        f"nanocodec extract vp9 op={op_idx}: {ext.stderr.strip()[-240:]}",
                    )
                if num_ops == num_spatial:  # no temporal subsets: the op is the view
                    noside_ivf = tmp / f"noside_op{op_idx}.ivf"
                    ext_noside = subprocess.run(
                        [str(own_decoder), "extract", "--codec", "vp9",
                         "--operating-point", str(op_idx),
                         "--input", str(bitstream), "--output", str(noside_ivf)],
                        stdin=subprocess.DEVNULL, capture_output=True, text=True,
                        timeout=DECODE_TIMEOUT_SECONDS)
                    if ext_noside.returncode != 0:
                        return GateResult(
                            False,
                            f"nanocodec extract vp9 nosidecar op={op_idx}: "
                            f"{ext_noside.stderr.strip()[-240:]}",
                        )
                    if noside_ivf.read_bytes() != op_ivf.read_bytes():
                        return GateResult(
                            False,
                            f"nanocodec extract vp9 --operating-point {op_idx} differs from "
                            f"sidecar op={op_idx}",
                        )
                if op_idx == view_ops[num_spatial - 1] and len(shown) == len(ordered_tus):
                    base_ivf = tmp / "base_only.ivf"
                    ext_base = subprocess.run(
                        [str(own_decoder), "extract", "--codec", "vp9", "--base-only",
                         "--input", str(bitstream), "--output", str(base_ivf)],
                        stdin=subprocess.DEVNULL, capture_output=True, text=True,
                        timeout=DECODE_TIMEOUT_SECONDS)
                    if ext_base.returncode != 0:
                        return GateResult(
                            False,
                            f"nanocodec extract vp9 --base-only: {ext.stderr.strip()[-240:] or ext_base.stderr.strip()[-240:]}",
                        )
                    if base_ivf.read_bytes() != op_ivf.read_bytes():
                        return GateResult(
                            False,
                            f"nanocodec extract vp9 --base-only differs from sidecar op={op_idx}",
                        )
                ivf_targets = [(f"op={op_idx}", op_ivf, ext_dims, ext_md5s)]
                if (op_idx == 0 and op_ivf != bitstream and len(shown) == len(ordered_tus)
                        and op_ivf.read_bytes() != bitstream.read_bytes()):
                    ivf_targets.insert(0, ("unstripped", bitstream, shown_dims, shown_md5s))

            for tag, target_ivf, target_dims, target_md5s in ivf_targets:
                for label, cmd in (
                    (f"vpxdec {tag}",
                     [str(vpxdec), "--codec=vp9", "--rawvideo", "--i420", "--threads=1",
                      "-o", "-", str(target_ivf)]),
                    (f"ffmpeg-vp9 {tag}",
                     [str(ffmpeg), "-nostdin", "-v", "error", "-xerror", "-err_detect", "explode",
                      "-filter_threads", "1", "-threads", "1", "-f", "ivf", "-c:v", "vp9",
                      "-i", str(target_ivf), "-fps_mode", "passthrough", "-threads", "1",
                      "-noautoscale", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"]),
                ):
                    err, got = _pipe_i420_md5s(cmd, target_dims)
                    if err:
                        return GateResult(False, f"{label}: {err}")
                    bad = first_mismatch(got, target_md5s)
                    if bad is not None:
                        return GateResult(False, f"{label}: MD5 mismatch at frame {bad}")

            if own_decoder is not None:
                res = subprocess.run(
                    [str(own_decoder), "decode", "--codec", "vp9",
                     "--input", str(op_ivf), "--frame-md5", "-"],
                    stdin=subprocess.DEVNULL, capture_output=True, text=True,
                    timeout=DECODE_TIMEOUT_SECONDS)
                if res.returncode != 0:
                    return GateResult(
                        False,
                        f"nanocodec vp9 stripped op={op_idx}: {res.stderr.strip()[-240:]}",
                    )
                got_own = [line.strip().lower() for line in res.stdout.splitlines() if line.strip()]
                if len(got_own) != len(ext_md5s):
                    return GateResult(
                        False,
                        f"nanocodec vp9 stripped op={op_idx}: {len(got_own)} frames, "
                        f"expected {len(ext_md5s)}",
                    )
                bad = first_mismatch(got_own, ext_md5s)
                if bad is not None:
                    return GateResult(
                        False,
                        f"nanocodec vp9 stripped op={op_idx}: MD5 mismatch at frame {bad}",
                    )

    return GateResult(True, level=GateLevel.FULL.value)
