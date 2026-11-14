"""Spatial SVC streams: their layers' bytes and their lower layers' quality.

A two-layer stream (our `spatial.layers = 2`, presets/<codec>-svc*.toml)
carries a half-size base layer (operating point 1) under the full-size one
(operating point 0); a three-layer AV1 stream (`spatial.layers = 3`,
presets/av1-svc3*.toml) a quarter-size S0 (operating point 2) and a
half-size S1 (operating point 1) under it. Measuring it the way a receiver sees it takes two
things the single-layer path has no use for, shared here by the SVC chart
(nceval.svc_chart) and by every CBR point of a spatial setting (nceval.cbr:
screens, tune, levels):

  - per temporal unit, the bytes of each layer: AV1 OBUs carry their
    spatial_id (AV1 section 5.3.3); VP9 has no layer ids, but a unit coding
    both layers is a superframe [S0, S1] whose index lists the frames
    (VP9 Annex B.3);
  - the base layer scored at its own size against the source downscaled to
    it (ffmpeg Lanczos, served through FIFOs: no YUV is written): decoded at
    operating point 1 (AV1: dav1d; VP9: vpxdec on `nanocodec extract
    --base-only`'s stream) and required to equal the encoder's own base
    reconstruction (`--layer-md5`).

The full-size output (operating point 0) is scored like any point, from the
encoder's reconstruction, at the total stream rate.
"""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Callable

from .clipframes import ClipFrames
from .clips import Clip
from .codecs import Codec
from .config import Paths, find_anchor_tool, find_tool
from .conformance import av1_stream_operating_points
from .encoders.base import fps_fraction
from .metrics import MetricSession, StreamSpec, gpu_slot_need, real_metrics
from .point import encode_streamed
from .slots import cpu_slots, gpu_slots
from .sweep import IVF_FILE_HEADER_BYTES, IVF_FRAME_HEADER_BYTES
from .yuv import YuvFormat

#: Base layer size: half the stream in each dimension (our `spatial.base_scale`
#: default and the reference SVC encoders' default for two layers).
BASE_SCALE_DEN = 2
#: Base layer share of a two-layer point's total target: our encoder's
#: `spatial.base_rate_share` default (docs/handoff/spatial-svc.md section 1.1),
#: so the references and simulcast split the rate as our encoder starts from,
#: and a CBR point's base layer is judged against that share of its target.
BASE_RATE_SHARE = 0.35
#: Three-layer points' S0 and S1 shares of the total target (S2 takes the
#: rest): our `spatial.s0_rate_share` / `spatial.s1_rate_share` defaults
#: (docs/handoff/spatial-svc.md section 12), for the references and simulcast
#: as BASE_RATE_SHARE is for two layers.
THREE_LAYER_RATE_SHARES = (0.08, 0.26)
#: The parameter that makes a setting two-layer, and its two-layer value.
SPATIAL_LAYERS_PARAM = "spatial.layers"
TWO_LAYERS = 2
#: The codecs with spatial SVC.
SVC_CODECS = (Codec.AV1, Codec.VP9)
#: AV1 OBU header fields and frame-header flags (AV1 sections 5.3, 5.9.2).
OBU_HAS_EXTENSION = 0x04
OBU_HAS_SIZE = 0x02
OBU_TEMPORAL_DELIMITER = 2
OBU_FRAME_HEADER = 3
OBU_FRAME = 6
#: Top three bits of an uncompressed AV1 frame header (`show_existing_frame` +
#: `frame_type`, AV1 section 5.9.2): all zero iff `KEY_FRAME` (`frame_type == 0`).
AV1_KEY_HEADER_MASK = 0xE0
#: VP9 superframe index (VP9 Annex B.3): its marker byte's top bits.
SUPERFRAME_MARKER_MASK = 0xE0
SUPERFRAME_MARKER = 0xC0
#: A VP9 Profile 0 frame's first byte: its show_existing_frame, frame_type
#: (`0 = KEY_FRAME`, `1 = INTER_FRAME`), and show_frame bits (VP9 section 6.2).
VP9_SHOW_EXISTING_BIT = 0x08
VP9_FRAME_TYPE_BIT = 0x04
VP9_SHOW_FRAME_BIT = 0x02
#: A show_existing_frame (Profile 0): one byte.
VP9_SHOW_EXISTING_BYTES = 1


def is_spatial(extra: list[str] | tuple[str, ...]) -> bool:
    """Whether a command's `--param` settings code two spatial layers."""
    for name, value in (arg.partition("=")[::2] for arg in extra if "=" in arg):
        if name == SPATIAL_LAYERS_PARAM and float(value) == TWO_LAYERS:
            return True
    return False


def layer_size(clip: Clip, steps: int) -> tuple[int, int]:
    """The size `steps` 2:1 steps below the clip's, each rounded up to even, as our
    encoder sizes its lower layers (nc-encoder resolution.rs `half_size`, cascaded;
    the chart's clips halve exactly)."""
    def half(n: int) -> int:
        return (-(-n // BASE_SCALE_DEN) + 1) // 2 * 2
    width, height = clip.width, clip.height
    for _ in range(steps):
        width, height = half(width), half(height)
    return width, height


def base_size(clip: Clip) -> tuple[int, int]:
    """The two-layer base layer's size: half the clip's (`layer_size`)."""
    return layer_size(clip, 1)


def split_rate(total_kbps: int) -> tuple[int, int]:
    """(base, full-resolution layer) kbps of a two-layer point."""
    base = round(total_kbps * BASE_RATE_SHARE)
    return base, total_kbps - base


def split_rate3(total_kbps: int) -> tuple[int, int, int]:
    """(S0, S1, S2) kbps of a three-layer point (THREE_LAYER_RATE_SHARES)."""
    s0, s1 = (round(total_kbps * share) for share in THREE_LAYER_RATE_SHARES)
    return s0, s1, total_kbps - s0 - s1


# --- the base-size source -----------------------------------------------------------------

class ScaledFrames:
    """The clip's measured frames downscaled to `size` (default: the base size),
    served once per
    `reader()` through a FIFO (one ffmpeg Lanczos downscale each; no YUV is
    written). Usable as a StreamSpec reference and as an encoder's input.
    `source` serves the clip's frames (pool clips, rotated starts); without
    one the clip must be read from its file, from frame 0."""

    def __init__(self, clip: Clip, frames: int, scratch: Path, paths: Paths,
                 source: ClipFrames | None = None, size: tuple[int, int] | None = None):
        if source is None and not clip.direct:
            raise ValueError(f"{clip.name}: a downscale without a frame source reads raw "
                             "clips from frame 0")
        self.clip, self.frames, self.paths, self.source = clip, frames, paths, source
        self.size = size or base_size(clip)
        scratch.mkdir(parents=True, exist_ok=True)
        self._directory = Path(tempfile.mkdtemp(prefix="nc-scaled-", dir=scratch))
        self._processes: list[subprocess.Popen] = []
        self._lock = threading.Lock()

    def __enter__(self) -> ScaledFrames:
        return self

    def __exit__(self, *exc) -> None:
        with self._lock:
            for process in self._processes:
                if process.poll() is None:
                    process.kill()
                process.wait()
        shutil.rmtree(self._directory, ignore_errors=True)

    def reader(self) -> Path:
        width, height = self.size
        with self._lock:
            fifo = self._directory / f"reader{len(self._processes)}.yuv"
            os.mkfifo(fifo)
            num, den = fps_fraction(self.clip.fps)
            frames_in = self.source.reader() if self.source else self.clip.yuv
            # ffmpeg opens the FIFO for writing, so it waits for its reader.
            self._processes.append(subprocess.Popen(
                [str(find_tool("ffmpeg", self.paths)), "-nostdin", "-v", "error",
                 "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s",
                 f"{self.clip.width}x{self.clip.height}", "-r", f"{num}/{den}",
                 "-i", str(frames_in), "-frames:v", str(self.frames),
                 "-vf", f"scale={width}:{height}:flags=lanczos", "-f", "rawvideo",
                 "-pix_fmt", "yuv420p", "-y", str(fifo)],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL))
        return fifo


# --- temporal units and layer bytes -----------------------------------------------------

def ivf_frames(data: bytes) -> list[tuple[int, bytes]]:
    """(pts, payload) of every IVF frame."""
    frames, offset = [], IVF_FILE_HEADER_BYTES
    while offset + IVF_FRAME_HEADER_BYTES <= len(data):
        size = int.from_bytes(data[offset:offset + 4], "little")
        pts = int.from_bytes(data[offset + 4:offset + 12], "little")
        start = offset + IVF_FRAME_HEADER_BYTES
        frames.append((pts, data[start:start + size]))
        offset = start + size
    return frames


def write_ivf(path: Path, header: bytes, frames: list[tuple[int, bytes]]) -> None:
    header = header[:24] + len(frames).to_bytes(4, "little") + header[28:32]
    with path.open("wb") as out:
        out.write(header)
        for pts, payload in frames:
            out.write(len(payload).to_bytes(4, "little") + pts.to_bytes(8, "little"))
            out.write(payload)


def leb128(data: bytes, pos: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7


def av1_base_bytes(unit: bytes, max_spatial_id: int = 0) -> int:
    """Bytes of one temporal unit that a receiver of spatial layers up to
    `max_spatial_id` gets (default: the base layer): OBUs with spatial_id up to
    it, or without an extension header (temporal delimiter, sequence header:
    every receiver needs them; AV1 sections 5.3.1-5.3.3)."""
    base = pos = 0
    while pos < len(unit):
        start, header = pos, unit[pos]
        pos += 1
        spatial_id = 0
        if header & OBU_HAS_EXTENSION:
            spatial_id = (unit[pos] >> 3) & 0x03
            pos += 1
        if header & OBU_HAS_SIZE:
            size, pos = leb128(unit, pos)
        else:
            size = len(unit) - pos
        pos += size
        if spatial_id <= max_spatial_id:
            base += pos - start
    return base


def av1_ksvc_receiver_bytes(unit: bytes, target_spatial_id: int) -> int:
    """Bytes of one AV1 K-SVC temporal unit downloaded by a receiver of
    `target_spatial_id`: `{S0..=target_spatial_id}` on key units (where `S0` is a
    `KEY_FRAME`, AV1 section 5.9.2) and `{target_spatial_id}` alone (plus one
    temporal delimiter and any sequence/metadata header) on non-key units."""
    if target_spatial_id <= 0:
        return av1_base_bytes(unit, 0)
    s0_is_key = has_target = seen_td = False
    shared_bytes = target_bytes = pos = 0
    while pos < len(unit):
        start, header = pos, unit[pos]
        obu_type = (header >> 3) & 0x0F
        has_ext = bool(header & OBU_HAS_EXTENSION)
        pos += 1
        spatial_id = 0
        if has_ext:
            spatial_id = (unit[pos] >> 3) & 0x03
            pos += 1
        if header & OBU_HAS_SIZE:
            size, pos = leb128(unit, pos)
        else:
            size = len(unit) - pos
        payload_start = pos
        pos += size
        obu_len = pos - start
        if obu_type in (OBU_FRAME_HEADER, OBU_FRAME) and spatial_id == 0 and size > 0:
            if (unit[payload_start] & AV1_KEY_HEADER_MASK) == 0:
                s0_is_key = True
        if obu_type == OBU_TEMPORAL_DELIMITER:
            if not seen_td:
                seen_td = True
                shared_bytes += obu_len
        elif not has_ext:
            shared_bytes += obu_len
        elif spatial_id == target_spatial_id:
            has_target = True
            target_bytes += obu_len
    if s0_is_key:
        return av1_base_bytes(unit, target_spatial_id)
    return (shared_bytes + target_bytes) if has_target else 0


def vp9_superframe_sizes(unit: bytes) -> list[int] | None:
    """The frame sizes a VP9 superframe index lists (Annex B.3), or None for a
    unit that is one frame."""
    marker = unit[-1] if unit else 0
    if marker & SUPERFRAME_MARKER_MASK != SUPERFRAME_MARKER:
        return None
    count, magnitude = (marker & 0x07) + 1, ((marker >> 3) & 0x03) + 1
    index_size = 2 + magnitude * count
    if len(unit) < index_size or unit[-index_size] != marker:
        return None
    pos = len(unit) - index_size + 1
    return [int.from_bytes(unit[pos + i * magnitude:pos + (i + 1) * magnitude], "little")
            for i in range(count)]


def vp9_pack_superframe(frames: list[bytes]) -> bytes:
    """Pack `frames` as a single VP9 IVF frame (`frames[0]` alone when there is
    one frame, or concatenated with a VP9 Annex B.3 superframe index)."""
    if len(frames) == 1:
        return frames[0]
    magnitude = max(1, max((len(f).bit_length() + 7) // 8 for f in frames))
    marker = SUPERFRAME_MARKER | ((magnitude - 1) << 3) | (len(frames) - 1)
    index = (bytes([marker])
             + b"".join(len(f).to_bytes(magnitude, "little") for f in frames)
             + bytes([marker]))
    return b"".join(frames) + index


def vp9_coded_layers(unit: bytes) -> list[bytes]:
    """The coded spatial-layer frames of one VP9 temporal unit, lowest first,
    excluding any trailing 1-byte `show_existing_frame` marker."""
    sizes = vp9_superframe_sizes(unit)
    if sizes is None:
        return [unit] if unit else []
    frames, pos = [], 0
    for size in sizes:
        frames.append(unit[pos:pos + size])
        pos += size
    if len(frames) > 1 and (frames[-1][0] & VP9_SHOW_EXISTING_BIT):
        frames.pop()
    return frames


def vp9_packed_bytes(kept: list[bytes]) -> int:
    """Wire bytes of `kept` VP9 coded layers as delivered to a receiver: a shown
    top frame goes alone (1 frame) or with an `N`-frame superframe index; a
    hidden top frame adds a 1-byte `show_existing_frame` and an `(N+1)`-frame
    superframe index (`nc-vp9` `pack_coded_layers`)."""
    if not kept:
        return 0
    magnitude = max(1, max((len(f).bit_length() + 7) // 8 for f in kept))
    if kept[-1][0] & (VP9_SHOW_EXISTING_BIT | VP9_SHOW_FRAME_BIT):
        if len(kept) == 1:
            return len(kept[0])
        return sum(len(f) for f in kept) + 2 + len(kept) * magnitude
    count = len(kept) + 1
    return sum(len(f) for f in kept) + VP9_SHOW_EXISTING_BYTES + 2 + count * magnitude


def vp9_operating_point_bytes(unit: bytes, max_spatial_id: int = 0) -> int:
    """Wire bytes of one VP9 temporal unit for a Full-SVC receiver of spatial
    layers `0..=max_spatial_id`."""
    coded = vp9_coded_layers(unit)
    return vp9_packed_bytes(coded[:max_spatial_id + 1])


def vp9_ksvc_receiver_bytes(unit: bytes, target_spatial_id: int) -> int:
    """Wire bytes of one VP9 K-SVC temporal unit downloaded by a receiver of
    `target_spatial_id`: `{S0..=target_spatial_id}` on key units (where `S0` is
    a `KEY_FRAME`, VP9 section 6.2) and `{target_spatial_id}` alone on non-key units."""
    if target_spatial_id <= 0:
        return vp9_operating_point_bytes(unit, 0)
    coded = vp9_coded_layers(unit)
    if not coded:
        return 0
    s0_is_key = (coded[0][0] & (VP9_SHOW_EXISTING_BIT | VP9_FRAME_TYPE_BIT)) == 0
    if s0_is_key:
        return vp9_packed_bytes(coded[:target_spatial_id + 1])
    if target_spatial_id >= len(coded):
        return 0
    return vp9_packed_bytes([coded[target_spatial_id]])


def vp9_base_bytes(unit: bytes) -> int:
    """Base-layer bytes of one VP9 temporal unit: what a base-only receiver gets.
    A superframe's first frame is S0 (ours and libvpx's pack the layers lowest
    first); a unit that is one frame is the base's. libvpx's S0 is shown and
    goes alone; ours is hidden, so the base-only unit adds a one-byte
    show_existing_frame and a two-frame superframe index (nc-vp9
    superframe.rs: `[S0, show_existing_frame]`, as a unit without S1 is coded)."""
    return vp9_operating_point_bytes(unit, 0)


def base_layer_bytes(unit: bytes, codec: Codec) -> int:
    return av1_base_bytes(unit) if codec is Codec.AV1 else vp9_base_bytes(unit)


def vp9_extract_view(source: Path, out: Path, spatial_id: int = 0,
                     key_mode: bool = False) -> None:
    """Extract spatial layer `spatial_id`'s receiver stream from a libvpx SVC
    stream (`show_frame = 1` on every spatial layer): `{S0..=spatial_id}` on
    Full SVC and key units of K-SVC, and `{S_{spatial_id}}` alone on non-key
    units of K-SVC."""
    data = source.read_bytes()
    frames = []
    for pts, unit in ivf_frames(data):
        coded = vp9_coded_layers(unit)
        s0_is_key = bool(coded) and (coded[0][0] & (VP9_SHOW_EXISTING_BIT | VP9_FRAME_TYPE_BIT)) == 0
        if key_mode and spatial_id > 0 and not s0_is_key:
            kept = [coded[spatial_id]] if spatial_id < len(coded) else []
        else:
            kept = coded[:spatial_id + 1]
        if kept:
            frames.append((pts, vp9_pack_superframe(kept)))
    write_ivf(out, data[:IVF_FILE_HEADER_BYTES], frames)


def vp9_first_frames(source: Path, out: Path) -> None:
    """The base-only stream of a libvpx SVC stream: each superframe's first
    frame. libvpx codes every spatial layer with show_frame = 1, so these
    frames decode as they are (ours code S0 hidden: `nanocodec extract
    --base-only` adds a show_existing_frame)."""
    vp9_extract_view(source, out, spatial_id=0, key_mode=False)


def operating_point(stream: Path, spatial_id: int) -> int:
    """The sequence header's operating point that decodes spatial layer `spatial_id`
    (and those below) at every temporal layer (AV1 section 5.5.1: L2T3 has six)."""
    points = av1_stream_operating_points(stream)
    wanted = max(t for s, t in points if s == spatial_id)
    return points.index((spatial_id, wanted))


def stream_layer_bits(path: Path, codec: Codec) -> tuple[list[int], list[int]]:
    """Per temporal unit: (total bits, base-layer bits)."""
    totals, bases = [], []
    for _, unit in ivf_frames(path.read_bytes()):
        totals.append(len(unit) * 8)
        bases.append(base_layer_bytes(unit, codec) * 8)
    return totals, bases


def av1_operating_point_bits(path: Path, max_spatial_id: int) -> list[int]:
    """Per temporal unit: the bits of spatial layers up to `max_spatial_id` (an AV1
    stream's lower operating point at every temporal layer)."""
    return [av1_base_bytes(unit, max_spatial_id) * 8
            for _, unit in ivf_frames(path.read_bytes())]


def av1_ksvc_receiver_bits(path: Path, target_spatial_id: int) -> list[int]:
    """Per temporal unit: the wire bits a K-SVC receiver of `target_spatial_id`
    downloads (`{S0..=target_spatial_id}` on key units, `{target_spatial_id}`
    alone on non-key units)."""
    return [av1_ksvc_receiver_bytes(unit, target_spatial_id) * 8
            for _, unit in ivf_frames(path.read_bytes())]


def operating_point_bits(path: Path, codec: Codec, max_spatial_id: int) -> list[int]:
    """Per temporal unit: wire bits of spatial layers `0..=max_spatial_id` for a
    Full-SVC stream (`AV1` or `VP9`)."""
    if codec is Codec.AV1:
        return av1_operating_point_bits(path, max_spatial_id)
    return [vp9_operating_point_bytes(unit, max_spatial_id) * 8
            for _, unit in ivf_frames(path.read_bytes())]


def ksvc_receiver_bits(path: Path, codec: Codec, target_spatial_id: int) -> list[int]:
    """Per temporal unit: wire bits a K-SVC receiver of `target_spatial_id`
    downloads (`{S0..=target_spatial_id}` on key units, `{target_spatial_id}`
    alone on non-key units) for `AV1` or `VP9`."""
    if codec is Codec.AV1:
        return av1_ksvc_receiver_bits(path, target_spatial_id)
    return [vp9_ksvc_receiver_bytes(unit, target_spatial_id) * 8
            for _, unit in ivf_frames(path.read_bytes())]


def per_input(sizes: list[int], coded: list[bool]) -> list[int]:
    """Per-coded-frame sizes as per-input-frame sizes: 0 where nothing was coded."""
    if len(sizes) != sum(coded):
        raise RuntimeError(f"{len(sizes)} temporal units for {sum(coded)} coded frames")
    stream = iter(sizes)
    return [next(stream) if is_coded else 0 for is_coded in coded]


# --- decoding and scoring a layer -----------------------------------------------------------

class HeldRepeatSink:
    """Feeds `sinks` one picture per input frame from a decoder's coded frames: an
    input frame our encoder held (not coded) repeats the last picture, as a
    player shows it."""

    def __init__(self, sinks: list, coded: list[bool]):
        self._sinks, self._coded = sinks, coded
        self._at = 0
        self._last: bytes | None = None

    def _forward(self, frame: bytes) -> None:
        for sink in self._sinks:
            sink.consume(frame)
        self._at += 1

    def _fill_held(self) -> None:
        while self._at < len(self._coded) and not self._coded[self._at]:
            if self._last is None:
                raise RuntimeError("the first frame was not coded")
            self._forward(self._last)

    def consume(self, frame: bytes) -> None:
        self._fill_held()
        self._last = frame
        self._forward(frame)

    def finish(self) -> None:
        self._fill_held()
        for sink in self._sinks:
            sink.finish()


class Md5Sink:
    """MD5 per frame (the encoder's `--layer-md5` digest)."""

    def __init__(self) -> None:
        self.digests: list[str] = []

    def consume(self, frame: bytes) -> None:
        self.digests.append(hashlib.md5(frame).hexdigest())

    def finish(self) -> None:
        pass


def decode_scored(decode: Callable[[str], list[str]], reference: Path | ScaledFrames,
                  fmt: YuvFormat, frames: int, fps: float, coded: list[bool],
                  paths: Paths) -> tuple[object, list[str]]:
    """Run `decode` (its command, given the path to write I420 to) into the J_real
    metrics against `reference`; (the measurement, each decoded picture's MD5).
    `coded`: the input frames coded (a held frame shows the last picture)."""
    md5s = Md5Sink()
    spec = StreamSpec(reference, fmt, frames, fps, paths)
    components = real_metrics(spec)
    with gpu_slots(gpu_slot_need(components)), \
            cpu_slots(1 + sum(m.cpu_threads for m in components)), \
            MetricSession(spec, components) as session:
        decoded = encode_streamed(decode, fmt, [md5s, HeldRepeatSink(session.sinks, coded)])
        if decoded != sum(coded):
            raise RuntimeError(f"decoded {decoded} frames of {fmt.width}x{fmt.height}, "
                               f"expected {sum(coded)}")
        return session.result(frames), md5s.digests


def layer_md5s(layer_md5: Path, spatial_id: int) -> list[str]:
    """One spatial layer's per-picture MD5s from the encoder's `--layer-md5` log."""
    with layer_md5.open(newline="") as log:
        return [row["md5"].strip().lower() for row in csv.DictReader(log)
                if row["spatial_id"] == str(spatial_id)]


def base_decode_command(stream: Path, codec: Codec, paths: Paths,
                        spatial_id: int = 0) -> Callable[[str], list[str]]:
    """The standard decoder's command for spatial layer `spatial_id` of `stream`
    (default the base; AV1: dav1d at that layer's operating point; VP9: vpxdec, on
    a base-only stream)."""
    if codec is Codec.AV1:
        dav1d, op = find_anchor_tool("dav1d", paths), operating_point(stream, spatial_id)
        return lambda out: [str(dav1d), "-q", "--threads", "1", "--filmgrain", "0",
                            "--oppoint", str(op), "--alllayers", "0", "--muxer", "yuv",
                            "-i", str(stream), "-o", out]
    vpxdec = find_anchor_tool("vpxdec", paths)
    return lambda out: [str(vpxdec), "--codec=vp9", "--rawvideo", "--i420", "--threads=1",
                        "-o", out, str(stream)]


def own_base_layer(stream: Path, codec: Codec, binary: Path, layer_md5: Path, clip: Clip,
                   frames: int, coded: list[bool], scratch: Path, paths: Paths,
                   source: ClipFrames | None = None, spatial_id: int = 0, steps: int = 1,
                   key_mode: bool = False):
    """Our stream's lower spatial layer `spatial_id` (default the two-layer base,
    operating point 1), `steps` 2:1 steps below the full size, scored at its own
    size against the downscaled source; it must decode to the encoder's own
    reconstruction of that layer (`layer_md5`). VP9's base-only stream (and AV1's
    K-SVC per-receiver `S1` stream) is `nanocodec extract`'s (written beside
    `stream`, removed after)."""
    if codec not in SVC_CODECS:
        raise ValueError(f"{codec.value} has no spatial layers")
    size = layer_size(clip, steps)
    base_stream = stream
    if codec is Codec.VP9:
        base_stream = stream.with_name(f"{stream.stem}.op{steps}{stream.suffix}")
        extract = subprocess.run(
            [str(binary), "extract", "--codec", "vp9", "--operating-point", str(steps),
             "--input", str(stream), "--output", str(base_stream)],
            stdin=subprocess.DEVNULL, capture_output=True, text=True)
        if extract.returncode:
            raise RuntimeError(
                f"{stream}: extract vp9 --operating-point {steps}: {extract.stderr[-300:]}")
    elif codec is Codec.AV1 and key_mode and spatial_id > 0:
        op = operating_point(stream, spatial_id)
        base_stream = stream.with_name(f"{stream.stem}.op{op}{stream.suffix}")
        extract = subprocess.run(
            [str(binary), "extract", "--codec", "av1", "--operating-point", str(op),
             "--input", str(stream), "--output", str(base_stream)],
            stdin=subprocess.DEVNULL, capture_output=True, text=True)
        if extract.returncode:
            raise RuntimeError(
                f"{stream}: extract --operating-point {op}: {extract.stderr[-300:]}")
    try:
        with ScaledFrames(clip, frames, scratch, paths, source, size) as scaled:
            measured, md5s = decode_scored(
                base_decode_command(base_stream, codec, paths, spatial_id), scaled,
                YuvFormat(*size), frames, clip.fps, coded, paths)
    finally:
        if base_stream != stream:
            base_stream.unlink(missing_ok=True)
    if md5s != layer_md5s(layer_md5, spatial_id):
        raise RuntimeError(f"{stream}: spatial layer {spatial_id} decodes differ from the "
                           "encoder's")
    return measured
