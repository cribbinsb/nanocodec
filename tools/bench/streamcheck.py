"""What each stream actually does, read from the stream itself (procedure section 10, V29-V30).

    structure(stream, codec) -> {"probe": ..., "reordered": bool, "hidden_frames": int,
                                 "keys": int, "first_is_key": bool, "frames": int}

The owner's rule (2026-11-25): check encoder behaviour from the streams, never from the
commands. Two things are read:

Only what costs nothing beyond reading the stream's bytes (owner, 2026-11-25, D19: no
extra decode in the benchmark):

  - **Reordering** (D2: no picture may reference a later one), AV1 and VP9 only: a frame
    header with show_frame = 0 (coded now, shown later or never), or a show_existing_frame (an
    earlier hidden frame shown). Read from the uncompressed frame headers: AV1 §5.9.2, VP9
    §6.2; VP9 superframes are split by their index (Annex B). H.264, HEVC and VVC are not
    checked for reordering: their output order needs a decode (an ffprobe pass doubled the
    decode CPU of every such stream), so their records say "reordered": None.
  - **Key frames:** whether the first frame is one (measurement condition: one key at the start
    of the window), and for AV1/VP9 their count. H.26x: the first VCL NAL unit's type (H.264
    IDR, H.264 §7.4.1.2; HEVC IRAP, §7.4.2.2; VVC IRAP, §7.4.2.2); AV1: frame_type
    KEY_FRAME; VP9: frame_type 0.

AV2 (AVM) frame headers are not parsed here: its record says "probe": "count-only", with
`frames`, the temporal units; the report states AV2 as unchecked for reordering. Standard
library only: this runs on scoring boxes beside score.py.
"""

from __future__ import annotations

from pathlib import Path

IVF_HEADER, IVF_FRAME_HEADER = 32, 12
#: AV1 OBU types (§6.2.2).
OBU_SEQUENCE_HEADER, OBU_FRAME_HEADER, OBU_FRAME = 1, 3, 6
AV1_KEY_FRAME = 0
VP9_FRAME_MARKER = 2
VP9_KEY_FRAME = 0
#: VP9 Annex B: a superframe index ends with a marker byte 110xxxxx.
VP9_SUPERFRAME_MASK, VP9_SUPERFRAME_MARKER = 0xE0, 0xC0
VP9_PROFILE_WITH_RESERVED_BIT = 3


class BitReader:
    """MSB-first bits of a byte string."""

    def __init__(self, data: bytes):
        self.data, self.pos = data, 0

    def bit(self) -> int:
        byte = self.data[self.pos >> 3]
        value = (byte >> (7 - (self.pos & 7))) & 1
        self.pos += 1
        return value

    def bits(self, n: int) -> int:
        value = 0
        for _ in range(n):
            value = (value << 1) | self.bit()
        return value


def ivf_frames(data: bytes) -> list[bytes]:
    """The IVF container's frames (temporal units)."""
    frames, pos = [], IVF_HEADER
    while pos + IVF_FRAME_HEADER <= len(data):
        size = int.from_bytes(data[pos:pos + 4], "little")
        start = pos + IVF_FRAME_HEADER
        frames.append(data[start:start + size])
        pos = start + size
    return frames


def leb128(data: bytes, pos: int) -> tuple[int, int]:
    """(value, next position) of an AV1 leb128 (§4.10.5)."""
    value = 0
    for i in range(8):
        byte = data[pos + i]
        value |= (byte & 0x7F) << (7 * i)
        if not byte & 0x80:
            return value, pos + i + 1
    raise ValueError("leb128 longer than 8 bytes")


def av1_frame_headers(unit: bytes, reduced: list[bool]) -> list[tuple[bool, int, bool]]:
    """(show_existing_frame, frame_type, show_frame) of each frame header in a temporal unit;
    `reduced` holds the sequence header's reduced_still_picture_header (updated in place)."""
    headers, pos = [], 0
    while pos < len(unit):
        header = unit[pos]
        obu_type, extension, has_size = (header >> 3) & 0xF, (header >> 2) & 1, (header >> 1) & 1
        pos += 1 + extension
        if has_size:
            size, pos = leb128(unit, pos)
        else:
            size = len(unit) - pos
        payload = unit[pos:pos + size]
        pos += size
        if obu_type == OBU_SEQUENCE_HEADER:
            r = BitReader(payload)
            r.bits(3)  # seq_profile
            r.bit()  # still_picture
            reduced[0] = bool(r.bit())
        elif obu_type in (OBU_FRAME_HEADER, OBU_FRAME):
            if reduced[0]:
                headers.append((False, AV1_KEY_FRAME, True))
                continue
            r = BitReader(payload)
            if r.bit():  # show_existing_frame
                headers.append((True, -1, True))
                continue
            frame_type = r.bits(2)
            headers.append((False, frame_type, bool(r.bit())))
    return headers


def vp9_superframe(unit: bytes) -> list[bytes]:
    """The frames of a VP9 temporal unit (Annex B superframe index), or the unit itself."""
    marker = unit[-1] if unit else 0
    if marker & VP9_SUPERFRAME_MASK == VP9_SUPERFRAME_MARKER:
        frames = (marker & 7) + 1
        size_bytes = ((marker >> 3) & 3) + 1
        index = 2 + size_bytes * frames
        if len(unit) >= index and unit[-index] == marker:
            out, pos = [], 0
            for i in range(frames):
                at = len(unit) - index + 1 + i * size_bytes
                size = int.from_bytes(unit[at:at + size_bytes], "little")
                out.append(unit[pos:pos + size])
                pos += size
            return out
    return [unit]


def vp9_frame_header(frame: bytes) -> tuple[bool, int, bool]:
    """(show_existing_frame, frame_type, show_frame) of a VP9 frame (§6.2)."""
    r = BitReader(frame)
    if r.bits(2) != VP9_FRAME_MARKER:
        raise ValueError("VP9 frame marker missing")
    profile = r.bit() | (r.bit() << 1)
    if profile == VP9_PROFILE_WITH_RESERVED_BIT:
        r.bit()
    if r.bit():  # show_existing_frame
        return True, -1, True
    frame_type = r.bit()
    return False, frame_type, bool(r.bit())


def from_headers(headers: list[tuple[bool, int, bool]], probe: str, key_type: int) -> dict:
    hidden = sum(1 for existing, _, shown in headers if existing or not shown)
    keys = sum(1 for existing, kind, shown in headers if not existing and kind == key_type)
    first = headers[0] if headers else (False, -1, False)
    return {"probe": probe, "reordered": hidden > 0, "hidden_frames": hidden, "keys": keys,
            "first_is_key": (not first[0]) and first[1] == key_type,
            "frames": sum(1 for existing, _, shown in headers if existing or shown)}


#: Per codec: (the NAL unit type of a byte-aligned NAL header, its VCL types, its key types).
H26X_NAL = {
    "h264": (lambda h: h[0] & 0x1F, range(1, 6), {5}),             # IDR (H.264 Table 7-1)
    "hevc": (lambda h: (h[0] >> 1) & 0x3F, range(0, 32), set(range(16, 24))),  # IRAP
    "vvc": (lambda h: h[1] >> 3, range(0, 12), {7, 8, 9}),          # IDR_W_RADL, IDR_N_LP, CRA
}


def annexb_nal_headers(data: bytes):
    """The first two bytes of each NAL unit of an Annex-B byte stream (start code 00 00 01)."""
    pos = data.find(b"\x00\x00\x01")
    while pos >= 0:
        yield data[pos + 3:pos + 5]
        pos = data.find(b"\x00\x00\x01", pos + 3)


def h26x_structure(data: bytes, codec: str) -> dict:
    """Whether the first picture is a key: the type of the stream's first VCL NAL unit."""
    nal_type, vcl, keys = H26X_NAL[codec]
    for header in annexb_nal_headers(data):
        if len(header) == 2 and nal_type(header) in vcl:
            return {"probe": "annex-b first picture", "reordered": None,
                    "first_is_key": nal_type(header) in keys}
    raise ValueError("no coded picture in the stream")


def structure(stream: Path, codec: str) -> dict:
    """The stream's reordering and key frames (module docstring); `codec` is the job's."""
    data = stream.read_bytes()
    if codec in H26X_NAL:
        return h26x_structure(data, codec)
    units = ivf_frames(data)
    if codec == "av1":
        reduced = [False]
        headers = [h for unit in units for h in av1_frame_headers(unit, reduced)]
        return from_headers(headers, "av1 frame headers", AV1_KEY_FRAME)
    if codec == "vp9":
        headers = [vp9_frame_header(f) for unit in units for f in vp9_superframe(unit) if f]
        return from_headers(headers, "vp9 frame headers", VP9_KEY_FRAME)
    return {"probe": "count-only", "frames": len(units)}
