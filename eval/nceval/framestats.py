"""Per-frame bitstream statistics: size and picture type of every coded frame.

Our encoder codes no frame for an input frame that would not change the
picture (frame hold) or that rate control drops; `with_held_frames` puts
those back as empty entries, so frame i is input frame i again.

Read with ffprobe, or AV2 IVF/OBU framing where ffprobe lacks codec support.
Which input frames our encoder coded comes from its own output
(`coded_at_times`: `--timestamps-out`, any rate mode; `coded_frames`: the CBR
rate log), so no decode is needed to place the held ones. Our low-delay streams have no frame reordering, so decode order equals
display order and frame i here is frame i in the metrics.

QP and temporal id are not exposed by ffprobe for these codecs, so they are
recorded as NULL for anchors (our own encoder will report them directly).
"""

from __future__ import annotations

import csv
import json
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import Paths, find_tool


@dataclass(frozen=True)
class FrameInfo:
    bytes: int
    frame_type: str  # "I" or "P" (anything else is kept as reported)
    temporal_id: int | None = None
    qp: float | None = None


def probe_frames(bitstream: Path, paths: Paths, picture_types: bool = False) -> list[FrameInfo]:
    """Every coded frame's size and type: "I" (key) or "P" from the packets alone,
    or with `picture_types` the decoder's picture type (I/P/B), which needs a
    full decode (~0.4 CPU-s per 150-frame 720p stream, 2026-11-11)."""
    with bitstream.open('rb') as source:
        header = source.read(32)
        if header[:4] == b'DKIF' and header[8:12] == b'AV02':
            return _av2_frames(source, header)
    if not picture_types:
        return _packet_frames(bitstream, paths)
    cmd = [str(find_tool("ffprobe", paths)), "-v", "error",
           "-show_entries", "frame=pkt_size,pict_type,key_frame:packet=flags", "-of", "json",
           str(bitstream)]
    proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                          check=True)
    # Asking for both sections interleaves them in one list, tagged by type.
    entries = json.loads(proc.stdout).get("packets_and_frames", [])
    frames = [e for e in entries if e.get("type") == "frame"]
    packet_keys = [KEY_PACKET_FLAG in e.get("flags", "") for e in entries
                   if e.get("type") == "packet"]
    return [FrameInfo(bytes=int(f["pkt_size"]), frame_type=frame_type(f, packet_keys, i))
            for i, f in enumerate(frames)]


def _packet_frames(bitstream: Path, paths: Paths) -> list[FrameInfo]:
    """Sizes and key flags of the packets, parsed without decoding. Our low-delay
    streams and the anchors' carry one packet per shown frame, in display order."""
    cmd = [str(find_tool("ffprobe", paths)), "-v", "error",
           "-show_entries", "packet=size,flags", "-of", "json", str(bitstream)]
    proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                          check=True)
    return [FrameInfo(bytes=int(p["size"]),
                      frame_type="I" if KEY_PACKET_FLAG in p.get("flags", "") else "P")
            for p in json.loads(proc.stdout).get("packets", [])]


#: `FrameInfo.frame_type` of an input frame with no coded frame (held or dropped).
HELD_FRAME_TYPE = "held"


#: `--rate-log` frame types of input frames with no coded frame.
UNCODED_FRAME_TYPES = ("Dropped", "Held")


def coded_frames(rate_log: Path) -> list[bool]:
    """Per input frame, whether a frame was coded (nanocodec --rate-log; one row per input frame)."""
    with rate_log.open(newline="") as log:
        return [row["type"] not in UNCODED_FRAME_TYPES for row in csv.DictReader(log)]


#: `--timestamps-out` times are milliseconds (the mkvmerge v2 format).
MS_PER_SECOND = 1000.0


def coded_at_times(timestamps: Path, fps: float, frames: int) -> list[bool]:
    """Per input frame, whether a frame was coded, from nanocodec
    `--timestamps-out` (the time of every coded frame) at nominal timing."""
    coded = [False] * frames
    for line in timestamps.read_text().splitlines():
        if line and not line.startswith("#"):
            coded[round(float(line) * fps / MS_PER_SECOND)] = True
    return coded


def with_held_frames(frames: list[FrameInfo], coded: list[bool] | None) -> list[FrameInfo]:
    """`frames` (one per coded frame) with an empty entry at each input frame
    that `coded` marks as not coded (nceval.conformance.GateResult.coded);
    `frames` itself when `coded` is None."""
    if coded is None:
        return frames
    if sum(coded) != len(frames):
        raise ValueError(f"{len(frames)} coded frames, {sum(coded)} expected")
    stream = iter(frames)
    return [next(stream) if is_coded else FrameInfo(bytes=0, frame_type=HELD_FRAME_TYPE)
            for is_coded in coded]


#: ffprobe's packet flag for a random-access (key) packet.
KEY_PACKET_FLAG = "K"


def frame_type(frame: dict, packet_keys: list[bool], index: int) -> str:
    """"I" for a key frame, else the decoder's picture type.

    FFmpeg 7.1's native VVC decoder sets neither `key_frame` nor
    `pict_type` on its frames (pict_type "?"); its parser still flags the
    IDR packets. Our low-delay streams have one packet per frame, so frame i
    is packet i: an untyped frame takes its packet's key flag.
    """
    if frame.get("key_frame") == 1:
        return "I"
    picture = frame.get("pict_type", "?")
    if picture == "?" and index < len(packet_keys):
        return "I" if packet_keys[index] else "P"
    return picture


def _av2_frames(source, header: bytes) -> list[FrameInfo]:
    """Inspect already-gated low-delay AV02 packets without an FFmpeg AV2 decoder.

    Final AVM avm_codec.h defines closed/open-loop key OBUs as 4/5; frame
    groups and show-existing/TIP/switch forms are inter-picture packets here.
    Payload syntax is checked by the independent decoder gate, not this probe.
    """
    if len(header) != 32 or struct.unpack_from('<HH', header, 4) != (0, 32):
        raise ValueError('unsupported AV02 IVF header')
    result = []
    while frame_header := source.read(12):
        if len(frame_header) != 12:
            raise ValueError('truncated AV02 IVF frame header')
        size = struct.unpack_from('<I', frame_header)[0]
        if not 0 < size <= 64 * 1024 * 1024:
            raise ValueError('invalid AV02 IVF packet size')
        packet = source.read(size)
        if len(packet) != size:
            raise ValueError('truncated AV02 IVF packet')
        offset, kinds = 0, set()
        while offset < size:
            length = 0
            for shift in range(0, 35, 7):
                if offset == size:
                    raise ValueError('truncated AV2 OBU length')
                byte = packet[offset]
                offset += 1
                length |= (byte & 127) << shift
                if byte < 128:
                    break
            else:
                raise ValueError('oversized AV2 OBU length')
            if not 0 < length <= size - offset:
                raise ValueError('invalid AV2 OBU extent')
            kinds.add((packet[offset] >> 2) & 31)
            offset += length
        picture_type = ('I' if kinds & {4, 5} else
                        'P' if kinds & {6, 7, 10, 11, 12, 13, 14, 19, 21} else '?')
        result.append(FrameInfo(bytes=size, frame_type=picture_type))
    return result
