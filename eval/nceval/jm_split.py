"""JM decode of an H.264 stream that changes picture size mid-stream.

The pinned JM (ldecod) misdecodes a stream whose SPS changes picture size
(also x264's), so such a stream is decoded per coded video sequence: split
before every SPS (each starts an IDR access unit, which depends on nothing
before it) and the parts decoded one after another into the same output.
Used by nceval.conformance only for streams with more than one distinct SPS.

    python3 jm_split.py LDECOD BITSTREAM OUTPUT
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

START_CODE = re.compile(b"\\x00\\x00\\x00\\x01")
NAL_TYPE_SPS = 7


def nal_starts(data: bytes) -> list[tuple[int, int]]:
    """(offset, nal_unit_type) of every 4-byte start code's NAL unit."""
    return [(m.start(), data[m.end()] & 0x1F) for m in START_CODE.finditer(data)
            if m.end() < len(data)]


def distinct_sps_count(data: bytes) -> int:
    """How many different SPS NAL units the stream carries."""
    starts = nal_starts(data)
    ends = [offset for offset, _ in starts[1:]] + [len(data)]
    return len({data[a:b] for (a, kind), b in zip(starts, ends) if kind == NAL_TYPE_SPS})


def main(ldecod: str, bitstream: str, output: str) -> int:
    data = Path(bitstream).read_bytes()
    cuts = [offset for offset, kind in nal_starts(data) if kind == NAL_TYPE_SPS] or [0]
    with tempfile.TemporaryDirectory(prefix="jm-split-") as work, open(output, "wb") as out:
        for a, b in zip(cuts, cuts[1:] + [len(data)]):
            part, decoded = Path(work) / "part.264", Path(work) / "part.yuv"
            part.write_bytes(data[a:b])
            result = subprocess.run([ldecod, "-i", str(part), "-o", str(decoded)],
                                    cwd=work, stdout=subprocess.DEVNULL, check=False)
            if result.returncode != 0:
                return result.returncode
            out.write(decoded.read_bytes())
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
