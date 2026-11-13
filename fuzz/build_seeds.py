#!/usr/bin/env python3
"""Build seed inputs for the decoder fuzz targets (fuzz/fuzz_targets).

Each seed is one input in the layout `fuzz_targets/common/mod.rs` reads: a
configuration byte, then either packets separated by `NCPK` (browser / C API
paths, configuration 0 or 1) or a whole container file (command-line path,
configuration 2, or 3 for --all-layers): Annex-B for H.264/HEVC/VVC, IVF for
AV1/VP9/AV2. Every stream named on the command line yields one seed of each
form, holding as many leading frames as fit in --max-len (at most
--max-frames); --headers adds header-only seeds (parameter sets, sequence
headers, a VP9 superframe index, IVF headers, truncated first packets).

    fuzz/build_seeds.py --codec vvc --out DIR STREAM...

Streams may be Annex-B files (.264/.h264/.jsv/.26l/.jvt/.avc/.hevc/.265/
.bit/.bin/.vvc/.266) or IVF files (any codec; for H.264/HEVC an IVF payload
is an Annex-B access unit, as in web/streams-ref). Seeds are named by content
hash, so rerunning is idempotent.
"""

import argparse
import hashlib
import struct
import sys
from pathlib import Path

SEPARATOR = b"NCPK"
CONFIG_BROWSER, CONFIG_CAPI, CONFIG_CLI, CONFIG_CLI_ALL_LAYERS = 0, 1, 2, 3
IVF_HEADER, IVF_FRAME_HEADER = 32, 12
IVF_CODECS = {"av1": b"AV01", "vp9": b"VP90", "av2": b"AV02"}
ANNEX_B_CODECS = ("h264", "hevc", "vvc")
START_CODE = b"\x00\x00\x00\x01"
# Default frames per seed: enough for references, reordering and a layer cycle,
# few enough that a sanitizer build decodes a seed well inside the 10 s
# timeout (a 49-frame 832x480 VVC vector takes ~45 s under ASan).
MAX_FRAMES = 8


def ivf_frames(data: bytes) -> list[bytes]:
    """The frame payloads of an IVF file (a truncated tail is dropped)."""
    frames, at = [], IVF_HEADER
    while at + IVF_FRAME_HEADER <= len(data):
        (size,) = struct.unpack_from("<I", data, at)
        at += IVF_FRAME_HEADER
        if size == 0 or at + size > len(data):
            break
        frames.append(data[at:at + size])
        at += size
    return frames


def nal_units(data: bytes) -> list[bytes]:
    """NAL units of an Annex-B stream, without start codes."""
    units, i, start = [], 0, None
    while i + 3 <= len(data):
        if data[i:i + 3] == b"\x00\x00\x01":
            if start is not None:
                units.append(data[start:i].rstrip(b"\x00"))
            start = i + 3
            i += 3
        else:
            i += 1
    if start is not None:
        units.append(data[start:].rstrip(b"\x00"))
    return [u for u in units if u]


def is_vcl(codec: str, nal: bytes) -> bool:
    """Whether `nal` carries a slice (H.264 Table 7-1, HEVC Table 7-1,
    VVC Table 5)."""
    if codec == "h264":
        return nal[0] & 31 in (1, 5)
    if len(nal) < 2:
        return False
    if codec == "hevc":
        return (nal[0] >> 1) & 63 < 32
    return nal[1] >> 3 < 12


def access_units(codec: str, data: bytes) -> list[bytes]:
    """Annex-B access units: the non-VCL units before each slice, plus it
    (every stream we seed from codes one slice per picture or tolerates the
    split)."""
    units, pending = [], []
    for nal in nal_units(data):
        pending.append(START_CODE + nal)
        if is_vcl(codec, nal):
            units.append(b"".join(pending))
            pending = []
    return units


def ivf_file(fourcc: bytes, frames: list[bytes]) -> bytes:
    """A minimal IVF file around `frames` (the CLI ignores the picture size)."""
    header = b"DKIF" + struct.pack("<HH4sHHIIII", 0, IVF_HEADER, fourcc, 16, 16, 30, 1,
                                   len(frames), 0)
    body = b"".join(struct.pack("<IQ", len(f), i) + f for i, f in enumerate(frames))
    return header + body


def leading(frames: list[bytes], cost, max_len: int, max_frames: int) -> list[bytes]:
    """The longest prefix of `frames` (at most `max_frames`) whose encoding
    (`cost`) fits."""
    taken = []
    for frame in frames[:max_frames]:
        if cost(taken + [frame]) > max_len:
            break
        taken.append(frame)
    return taken


def leb128(data: bytes, at: int) -> tuple[int, int]:
    """An AV1/AV2 `leb128()` value at `at` and the offset after it."""
    value = 0
    for i in range(8):
        byte = data[at + i]
        value |= (byte & 0x7F) << (7 * i)
        if not byte & 0x80:
            return value, at + i + 1
    raise ValueError("leb128 too long")


def obus(codec: str, packet: bytes) -> list[tuple[int, bytes]]:
    """(obu_type, whole OBU) of a low-overhead temporal unit. AV1 (§5.3.1):
    header, then a size; AV2 (§5): a size covering the header, whose type
    is bits 2..6 of its first byte."""
    units, at = [], 0
    while at < len(packet):
        if codec == "av2":
            size, payload = leb128(packet, at)
            units.append(((packet[payload] >> 2) & 31, packet[at:payload + size]))
            at = payload + size
            continue
        header = packet[at]
        start, at = at, at + 1 + ((header >> 2) & 1)
        if not header & 2:
            raise ValueError("OBU without size")
        size, at = leb128(packet, at)
        at += size
        units.append(((header >> 3) & 15, packet[start:at]))
    return units


# OBU types that carry only headers: AV1 temporal delimiter, sequence
# header, metadata, padding (§6.2.2); AV2 sequence header, temporal
# delimiter, multi-stream decoder operation.
HEADER_OBUS = {"av1": (1, 2, 5, 15), "av2": (1, 2, 20)}
# Bytes of a first packet kept by the truncated header seeds.
TRUNCATIONS = (16, 48, 128)
SUPERFRAME_MARKER_MASK, SUPERFRAME_MARKER = 0xE0, 0xC0


def header_packets(codec: str, frames: list[bytes]) -> list[bytes]:
    """Packets holding headers only: parameter sets, sequence-header OBUs,
    a VP9 superframe index, and truncations of the first packet."""
    first = frames[0]
    packets = [first[:n] for n in TRUNCATIONS if n < len(first)]
    if codec in ANNEX_B_CODECS:
        parameters = [n for n in nal_units(first) if not is_vcl(codec, n)]
        if parameters:
            packets.append(b"".join(START_CODE + n for n in parameters))
    elif codec in HEADER_OBUS:
        kept = [unit for kind, unit in obus(codec, first) if kind in HEADER_OBUS[codec]]
        if kept:
            packets.append(b"".join(kept))
    else:
        last = first[-1]
        if last & SUPERFRAME_MARKER_MASK == SUPERFRAME_MARKER:
            index_len = 2 + (((last >> 3) & 3) + 1) * ((last & 7) + 1)
            packets.append(first[-index_len:])
    return packets


def seeds_for(codec: str, path: Path, max_len: int, max_frames: int,
              headers: bool) -> list[bytes]:
    data = path.read_bytes()
    if data[:4] == b"DKIF":
        frames = ivf_frames(data)
    elif codec in ANNEX_B_CODECS:
        frames = access_units(codec, data)
    else:
        raise ValueError(f"{path}: {codec} streams must be IVF")
    if not frames:
        return []
    packets = leading(frames, lambda f: 1 + len(SEPARATOR.join(f)), max_len, max_frames)
    seeds = [bytes([CONFIG_BROWSER]) + SEPARATOR.join(packets)] if packets else []
    if codec in ANNEX_B_CODECS:
        container = leading(frames, lambda f: 1 + sum(map(len, f)), max_len, max_frames)
        if container:
            seeds.append(bytes([CONFIG_CLI]) + b"".join(container))
    else:
        fourcc = IVF_CODECS[codec]
        container = leading(frames, lambda f: 1 + len(ivf_file(fourcc, f)), max_len, max_frames)
        if container:
            seeds.append(bytes([CONFIG_CLI]) + ivf_file(fourcc, container))
            if codec in ("av1", "vp9"):
                seeds.append(bytes([CONFIG_CLI_ALL_LAYERS]) + ivf_file(fourcc, container))
    if headers:
        for packet in header_packets(codec, frames):
            seeds.append(bytes([CONFIG_BROWSER]) + packet)
            if codec in ANNEX_B_CODECS:
                seeds.append(bytes([CONFIG_CLI]) + packet)
            else:
                seeds.append(bytes([CONFIG_CLI]) + ivf_file(IVF_CODECS[codec], [packet]))
        if codec not in ANNEX_B_CODECS:
            seeds.append(bytes([CONFIG_CLI]) + ivf_file(IVF_CODECS[codec], []))
    return [seed for seed in seeds if len(seed) <= max_len]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--codec", required=True,
                        choices=sorted([*ANNEX_B_CODECS, *IVF_CODECS]))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-len", type=int, default=65536)
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--headers", action="store_true",
                        help="also write header-only and truncated seeds")
    parser.add_argument("streams", type=Path, nargs="+")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    written = 0
    for stream in args.streams:
        try:
            seeds = seeds_for(args.codec, stream, args.max_len, args.max_frames, args.headers)
        except (OSError, ValueError, IndexError) as error:
            print(f"skip {stream}: {error}", file=sys.stderr)
            continue
        for seed in seeds:
            name = hashlib.sha1(seed).hexdigest()
            target = args.out / name
            if not target.exists():
                target.write_bytes(seed)
                written += 1
    print(f"{written} new seeds in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
