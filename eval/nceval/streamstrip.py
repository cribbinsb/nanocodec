"""Remove non-picture metadata from a coded stream, the same way for every encoder.

docs/benchmark-procedure.md (BP-2, "Bits"): before a point's bits are counted, every
stream (ours included) loses its SEI NAL units and metadata OBUs. They carry no picture
data: encoder names, option strings, timing hints. A decoder shows the same frames
without them, which the scorer checks per stream. x264 and x265 write about 0.6 kB and
1.5-2.4 kB of information SEI, which at the low rungs is several percent of a stream
(docs/analysis/benchmark-procedure-response-2026-11-23.md, B4). This runs after the
timed encode, outside the timed span.

    strip(stream, codec) -> int   rewrites `stream` without those units; the bytes removed

H.264 SEI is NAL type 6 (H.264 Table 7-1), HEVC prefix/suffix SEI 39/40 (HEVC Table
7-1), VVC prefix/suffix SEI 23/24 (VVC Table 5). AV1 metadata is OBU type 5 (AV1
section 6.2.2) inside IVF frames. AV2 metadata is OBU type 8 (OBU_METADATA_SHORT) or 9
(OBU_METADATA_GROUP) (AVM v1.0.0 avm_codec.h), whose OBU starts with its size (header
included) and then a header byte holding the type in bits 6..2. VP9 has no metadata units. Every SEI
message goes, including buffering-period and picture-timing SEI: decoding does not need them.
"""

from __future__ import annotations

import re
from pathlib import Path

from .codecs import Codec

START_CODE = re.compile(b"\x00\x00\x01")
SEI_TYPES = {Codec.H264: (6,), Codec.HEVC: (39, 40), Codec.VVC: (23, 24)}
OBU_METADATA = 5
AV2_OBU_METADATA = (8, 9)
IVF_HEADER, IVF_FRAME_HEADER = 32, 12


def _nal_type(nal: bytes, codec: Codec) -> int:
    if codec is Codec.H264:
        return nal[0] & 0x1F
    if codec is Codec.HEVC:
        return (nal[0] >> 1) & 0x3F
    return (nal[1] >> 3) & 0x1F  # VVC: nal_unit_type is in the second header byte


def strip_annexb(data: bytes, codec: Codec) -> tuple[bytes, int]:
    starts = [m.start() for m in START_CODE.finditer(data)]
    kept, removed = bytearray(data[:starts[0]] if starts else data), 0
    for index, start in enumerate(starts):
        begin = start - 1 if start > 0 and data[start - 1] == 0 else start
        end = starts[index + 1] if index + 1 < len(starts) else len(data)
        if index + 1 < len(starts) and data[end - 1] == 0:
            end -= 1
        unit = data[begin:end]
        if index == 0:
            kept = bytearray(data[:begin])
        if _nal_type(data[start + 3:end], codec) in SEI_TYPES[codec]:
            removed += len(unit)
        else:
            kept += unit
    return bytes(kept), removed


def _leb128(buf: bytes, pos: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if byte < 0x80:
            return value, pos


def _leb128_bytes(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _strip_ivf(data: bytes, read_obu) -> tuple[bytes, int]:
    """IVF `data` without its metadata OBUs: `read_obu(frame, i)` gives whether the OBU
    at `i` is metadata and where it ends."""
    out, removed, pos = bytearray(data[:IVF_HEADER]), 0, IVF_HEADER
    while pos + IVF_FRAME_HEADER <= len(data):
        size = int.from_bytes(data[pos:pos + 4], "little")
        frame = data[pos + IVF_FRAME_HEADER:pos + IVF_FRAME_HEADER + size]
        kept, i = bytearray(), 0
        while i < len(frame):
            metadata, end = read_obu(frame, i)
            if metadata:
                removed += end - i
            else:
                kept += frame[i:end]
            i = end
        out += len(kept).to_bytes(4, "little") + data[pos + 4:pos + IVF_FRAME_HEADER] + kept
        pos += IVF_FRAME_HEADER + size
    return bytes(out), removed


def _av2_obu(frame: bytes, i: int) -> tuple[bool, int]:
    size, header = _leb128(frame, i)
    return (frame[header] >> 2) & 0x1F in AV2_OBU_METADATA, header + size


def _av1_obu(frame: bytes, i: int) -> tuple[bool, int]:
    header = frame[i]
    has_extension, has_size = (header >> 2) & 1, (header >> 1) & 1
    if not has_size:
        raise ValueError("AV1 OBU without obu_size: not supported")
    body, after = _leb128(frame, i + 1 + has_extension)
    return (header >> 3) & 0xF == OBU_METADATA, after + body


def strip_ivf_av1(data: bytes) -> tuple[bytes, int]:
    return _strip_ivf(data, _av1_obu)


def strip_ivf_av2(data: bytes) -> tuple[bytes, int]:
    return _strip_ivf(data, _av2_obu)


def strip(stream: Path, codec: Codec) -> int:
    """Rewrite `stream` without its SEI / metadata units; returns the bytes removed."""
    data = stream.read_bytes()
    if codec in SEI_TYPES:
        new, removed = strip_annexb(data, codec)
    elif codec is Codec.AV1:
        new, removed = strip_ivf_av1(data)
    elif codec is Codec.AV2:
        new, removed = strip_ivf_av2(data)
    elif codec is Codec.VP9:
        return 0
    else:
        raise ValueError(f"{codec.value}: no metadata rule")
    if removed:
        stream.write_bytes(new)
    return removed
