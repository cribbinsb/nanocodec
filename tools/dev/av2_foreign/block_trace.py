#!/usr/bin/env python3
"""Hold the general AV2 block layer's parse against dav2d's, symbol by
symbol (docs/handoff/av2.md "The general block layer").

    tools/dev/av2_foreign/block_trace.py [STREAM...]

dav2d built with DEBUG_BLOCK_INFO (DAV2D_BLOCKS, default
~/nanocodec-data/work/av2-dec/dav2d-debug/build-blk/tools/dav2d) prints a
line per partition, block and symbol with the range decoder's state;
`examples/trace_blocks` (TRACE_BLOCKS, a built example binary) prints the
same lines for what our layer parses, then the error it stopped at. dav2d's
lines are reduced to the kinds ours prints and compared with ours; each
stream reports how far they agree and where ours stopped. Without streams,
the whole corpus of compare.py.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import corpus  # noqa: E402

HOME = Path.home()
DAV2D = Path(os.environ.get("DAV2D_BLOCKS", HOME / "nanocodec-data/work/av2-dec/dav2d-debug/build-blk/tools/dav2d"))
OURS = Path(os.environ.get("TRACE_BLOCKS", HOME / "nanocodec-data/targets/av2-dec2/gate/examples/trace_blocks"))
LINE = re.compile(r"^\s*([A-Za-z_-]+)\[")


def lines(text):
    return [line.rstrip() for line in text.splitlines() if LINE.match(line)]


def main():
    paths = [Path(a) for a in sys.argv[1:]] or corpus()
    for path in paths:
        ours_run = subprocess.run([str(OURS), str(path)], capture_output=True, text=True)
        error = next((l for l in ours_run.stdout.splitlines() if l.startswith("ERROR")), "no error")
        ours = lines(ours_run.stdout)
        kinds = {LINE.match(line)[1] for line in ours}
        theirs = [line for line in lines(subprocess.run([str(DAV2D), "--threads", "1", "-i", str(path), "-o", "/dev/null"],
                                                        capture_output=True, text=True).stdout)
                  if LINE.match(line)[1] in kinds]
        first = next((i for i, (a, b) in enumerate(zip(theirs, ours)) if a != b), None)
        if first is None:
            print(f"{path.name}: {len(ours)} lines agree; {error}")
        else:
            print(f"{path.name}: line {first} of {len(ours)}: dav2d `{theirs[first].strip()}` ours `{ours[first].strip()}`")


if __name__ == "__main__":
    main()
