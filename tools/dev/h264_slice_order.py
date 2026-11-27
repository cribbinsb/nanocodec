#!/usr/bin/env python3
"""Put the slices of every picture of an H.264 Annex-B stream in reverse
order: arbitrary slice order (ASO, H.264 Annex A.2.1, A.2.3) for the
supplementary conformance streams (tools/dev/h264_conformance_jm.sh).

A picture's slices are the VCL NAL units from one with first_mb_in_slice 0
(H.264 §7.4.3) up to the next non-VCL NAL unit or the next such slice; a
data partition A keeps its B and C partitions behind it. The decoded
pictures are those of the input, so the input's reference decode is the
output's.

Usage: python3 -I tools/dev/h264_slice_order.py IN.264 OUT.264
"""

import sys
from pathlib import Path

START_CODE = b"\x00\x00\x00\x01"
#: nal_unit_type (H.264 Table 7-1): slices and partition A start a slice;
#: partitions B and C belong to the slice before them.
SLICE_STARTS = {1, 2, 5}
PARTITIONS_BC = {3, 4}


def nal_units(data):
    """The NAL units of an Annex-B byte stream, without start codes."""
    starts = []
    pos = data.find(b"\x00\x00\x01")
    while pos >= 0:
        starts.append(pos + 3)
        pos = data.find(b"\x00\x00\x01", pos + 3)
    units = []
    for begin, end in zip(starts, starts[1:] + [len(data) + 3]):
        units.append(data[begin:end - 3].rstrip(b"\x00"))
    return [unit for unit in units if unit]


def first_mb_in_slice(nal):
    """The first ue(v) of a slice header (no emulation prevention byte can
    occur in its first two bytes after a non-zero bit)."""
    bits = "".join(f"{byte:08b}" for byte in nal[1:9])
    zeros = bits.index("1")
    return int(bits[zeros:2 * zeros + 1], 2) - 1


def reorder(units):
    """The units with each picture's slices reversed."""
    out, picture = [], []

    def flush():
        out.extend(unit for piece in reversed(picture) for unit in piece)
        picture.clear()

    for unit in units:
        nal_type = unit[0] & 0x1F
        if nal_type in SLICE_STARTS:
            if first_mb_in_slice(unit) == 0:
                flush()
            picture.append([unit])
        elif nal_type in PARTITIONS_BC and picture:
            picture[-1].append(unit)
        else:
            flush()
            out.append(unit)
    flush()
    return out


def main():
    source, target = (Path(arg) for arg in sys.argv[1:3])
    units = reorder(nal_units(source.read_bytes()))
    target.write_bytes(b"".join(START_CODE + unit for unit in units))


if __name__ == "__main__":
    main()
