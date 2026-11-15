#!/usr/bin/env python3
"""Hold the general AV2 block layer's pictures before the loop filters
against dav2d's (`--inloopfilters none`), frame by frame (docs/handoff/av2.md
"Intra reconstruction and the residual").

    tools/dev/av2_foreign/unfiltered.py [STREAM...]

dav2d (DAV2D_UNFILTERED, default the release build compare.py uses) decodes
with every in-loop filter off; `examples/trace_blocks FILE OUT.yuv`
(TRACE_BLOCKS, a built example binary) writes our pictures before the loop
filters. Each stream reports how many leading frames agree out of what dav2d
outputs, and our first error. Frames our subset path decodes (the `min`
streams) are filtered there, so they do not compare. Without streams, the
corpus of compare.py and the extra streams of CORPUS/streams-extra (tool
streams made with avmenc). Our pictures go through a temporary file that is
removed at once; dav2d's through a pipe.
"""
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import BIN, CORPUS, corpus, y4m_hashes  # noqa: E402

HOME = Path.home()
DAV2D = Path(os.environ.get("DAV2D_UNFILTERED", BIN / "dav2d"))
OURS = Path(os.environ.get("TRACE_BLOCKS", HOME / "nanocodec-data/targets/av2-dec2/gate/examples/trace_blocks"))


def ours(path, frame_size):
    """Our per-frame MD5s before the loop filters, and the first error."""
    with tempfile.TemporaryDirectory(dir=HOME / "nanocodec-data/work") as scratch:
        out = Path(scratch) / "unfiltered.yuv"
        run = subprocess.run([str(OURS), str(path), str(out)], capture_output=True, text=True)
        error = next((l for l in run.stdout.splitlines() if l.startswith("ERROR")), "no error")
        data = out.read_bytes()
    hashes = [hashlib.md5(data[i:i + frame_size]).hexdigest() for i in range(0, len(data) - frame_size + 1, frame_size)]
    return hashes, error


def main():
    paths = [Path(a) for a in sys.argv[1:]] or corpus() + sorted((CORPUS / "streams-extra").glob("*.ivf"))
    for path in paths:
        tags, theirs, err, rc = y4m_hashes([str(DAV2D), "-q", "--threads", "1", "--inloopfilters", "none",
                                            "-i", str(path), "--muxer", "yuv4mpeg2", "-o", "-"])
        if rc or not tags:
            print(f"{path.name}: dav2d failed: rc={rc} {err}")
            continue
        w, h = int(tags["W"]), int(tags["H"])
        got, error = ours(path, w * h + 2 * ((w + 1) // 2) * ((h + 1) // 2))
        agree = next((i for i, (a, b) in enumerate(zip(theirs, got)) if a != b), min(len(theirs), len(got)))
        print(f"{path.name}: {agree} of {len(theirs)} frames identical before the loop filters; {error}")


if __name__ == "__main__":
    main()
