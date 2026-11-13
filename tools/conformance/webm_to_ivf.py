#!/usr/bin/env python3
"""Rewrap the video track of a WebM/Matroska file as IVF, packet for packet.

The libvpx VP9 test vectors are mostly WebM, and two libaom AV1 vectors are
Matroska; nanocodec's tools read IVF. This
is a minimal EBML walker (Segment > Cluster > SimpleBlock / BlockGroup >
Block, unlaced blocks of the first video track), enough for those vectors.
Unlike `ffmpeg -c copy`, it keeps every packet: ffmpeg drops the leading
packets of streams that start with an intra-only frame instead of a key frame.

Usage: webm_to_ivf.py INPUT.webm OUTPUT.ivf [FOURCC]  (FOURCC default VP90; AV1: AV01)
"""

import struct
import sys

SEGMENT = 0x18538067
CLUSTER = 0x1F43B675
BLOCK_GROUP = 0xA0
BLOCK = 0xA1
SIMPLE_BLOCK = 0xA3
TRACKS = 0x1654AE6B
TRACK_ENTRY = 0xAE
TRACK_NUMBER = 0xD7
PIXEL_WIDTH = 0xB0
PIXEL_HEIGHT = 0xBA
VIDEO = 0xE0
# Master elements whose children are walked.
MASTERS = {SEGMENT, CLUSTER, BLOCK_GROUP, TRACKS, TRACK_ENTRY, VIDEO}
UNKNOWN_SIZE = -1


def read_vint(data, pos, keep_marker):
    """An EBML variable-length integer at `pos`: (value, next position)."""
    first = data[pos]
    length = 1
    while length <= 8 and not first & (0x80 >> (length - 1)):
        length += 1
    if length > 8:
        raise ValueError("invalid EBML vint")
    value = first if keep_marker else first & (0xFF >> length)
    for byte in data[pos + 1 : pos + length]:
        value = (value << 8) | byte
    if not keep_marker and value == (1 << (7 * length)) - 1:
        value = UNKNOWN_SIZE
    return value, pos + length


def walk(data, start, end, out):
    pos = start
    while pos < end:
        element, pos = read_vint(data, pos, True)
        size, pos = read_vint(data, pos, False)
        stop = end if size == UNKNOWN_SIZE else min(pos + size, end)
        if element in MASTERS:
            walk(data, pos, stop, out)
        elif element in (SIMPLE_BLOCK, BLOCK):
            track, body = read_vint(data, pos, False)
            flags = data[body + 2]
            if flags & 0x06:
                raise ValueError("laced blocks are not supported")
            out["blocks"].append((track, data[body + 3 : stop]))
        elif element in (TRACK_NUMBER, PIXEL_WIDTH, PIXEL_HEIGHT):
            out[element] = int.from_bytes(data[pos:stop], "big")
        pos = stop


def main():
    source, target = sys.argv[1], sys.argv[2]
    fourcc = sys.argv[3].encode() if len(sys.argv) > 3 else b"VP90"
    with open(source, "rb") as f:
        data = f.read()
    out = {"blocks": []}
    walk(data, 0, len(data), out)
    track = out.get(TRACK_NUMBER, 1)
    packets = [payload for number, payload in out["blocks"] if number == track]
    header = struct.pack(
        "<4sHH4sHHIIII", b"DKIF", 0, 32, fourcc,
        out.get(PIXEL_WIDTH, 0), out.get(PIXEL_HEIGHT, 0), 1000, 1, len(packets), 0,
    )
    with open(target, "wb") as f:
        f.write(header)
        for index, packet in enumerate(packets):
            f.write(struct.pack("<IQ", len(packet), index))
            f.write(packet)


if __name__ == "__main__":
    main()
