"""dav2d decode of an AV2 stream whose picture size is not whole 8x8 blocks.

Our AV2 streams code such a picture padded to whole 8x8 blocks and signal
the visible size in the sequence header's conformance window (AVM
av2_write_conformance_window). avmdec crops to it; the pinned dav2d
(237229e) parses the window but outputs the whole coded frame. This wrapper
streams dav2d's frames through a crop to the visible size, which it takes
from the IVF header (our AV2 streams keep one picture size), so both
decoders are compared over the same samples. Used by nceval.conformance
only for streams whose size is not a multiple of 8.

    python3 dav2d_crop.py DAV2D BITSTREAM OUTPUT
"""

from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path

CODED_MULTIPLE = 8  # AV2 frame sizes are whole 8x8 blocks
IVF_SIZE_OFFSET = 12  # little-endian u16 width, then height
CHROMA_PLANES = 2  # 4:2:0: two half-resolution planes


def visible_size(bitstream: Path) -> tuple[int, int]:
    """The picture size recorded in the IVF file header."""
    with open(bitstream, "rb") as f:
        header = f.read(IVF_SIZE_OFFSET + 4)
    return struct.unpack_from("<HH", header, IVF_SIZE_OFFSET)


def needs_crop(bitstream: Path) -> bool:
    """Whether the coded frame is larger than the visible picture."""
    return any(d % CODED_MULTIPLE for d in visible_size(bitstream))


def coded(dimension: int) -> int:
    return -(-dimension // CODED_MULTIPLE) * CODED_MULTIPLE


def crop_frames(source, sink, width: int, height: int) -> None:
    """Copy the top-left width x height of every coded I420 frame."""
    planes = [(coded(width), coded(height), width, height)]
    planes += [(coded(width) // 2, coded(height) // 2, width // 2, height // 2)] * CHROMA_PLANES
    while True:
        for coded_w, coded_h, w, h in planes:
            data = source.read(coded_w * coded_h)
            if not data:
                return
            if len(data) != coded_w * coded_h:
                raise ValueError("truncated dav2d frame")
            for y in range(h):
                sink.write(data[y * coded_w:y * coded_w + w])


def dav2d_command(dav2d: str, bitstream: str, output: str) -> list[str]:
    return [dav2d, "--threads", "1", "--filmgrain", "0", "--muxer", "yuv",
            "--input", bitstream, "--output", output]


def main(dav2d: str, bitstream: str, output: str) -> int:
    width, height = visible_size(Path(bitstream))
    with subprocess.Popen(dav2d_command(dav2d, bitstream, "/dev/stdout"),
                          stdout=subprocess.PIPE) as process, open(output, "wb") as out:
        crop_frames(process.stdout, out, width, height)
    return process.returncode


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
